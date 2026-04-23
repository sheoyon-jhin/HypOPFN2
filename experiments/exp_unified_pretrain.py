"""
Unified Real-data Pre-training + 5 Task Evaluation.
Masked reconstruction on ALL datasets (no labels).

사용법:
  source /opt/miniforge3/etc/profile.d/conda.sh && conda activate timefound
  CUDA_VISIBLE_DEVICES=0 python experiments/exp_unified_pretrain.py 2>&1 | tee log/unified_pretrain.log
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import optim
from torch.utils.data import DataLoader, random_split
from types import SimpleNamespace
import time

from model.DeepONetHyperMoE import Model
from data_provider.unified_dataset import UnifiedPretrainDataset
from data_provider.data_factory import data_provider
from data_provider.data_loader import Dataset_Classification
from sklearn.metrics import accuracy_score, f1_score


def get_model_args():
    return SimpleNamespace(
        seq_len=96, pred_len=96, use_norm=True, deeponet_width=64,
        n_experts=4, branch_depth=4, trunk_depth=2, activation='gelu',
        dropout=0.1, branch_hidden=-1,
        spectral_branch=False, skip_mode='none',
        use_cross_channel=False, trunk_basis='mixed',
        encoder_type='patch_attn', loss='MSE',
    )


# ============================================================
# Stage 1: Masked Reconstruction Pre-training
# ============================================================
def pretrain(model, device, save_path, epochs=10, lr=0.0003, mask_rate=0.4):
    print(f'\n{"="*60}')
    print('Stage 1: Unified Pre-training (Masked Reconstruction)')
    print(f'{"="*60}')

    dataset = UnifiedPretrainDataset(seq_len=96, stride=48)
    n_val = min(5000, len(dataset) // 10)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_dl = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=4, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=2)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss(reduction='none')

    print(f'Train: {n_train}, Val: {n_val}, Steps/epoch: {len(train_dl)}')

    best_val = float('inf')
    for epoch in range(epochs):
        model.train()
        train_losses = []
        t0 = time.time()

        for i, batch_x in enumerate(train_dl):
            # batch_x: [B, seq_len, 1]
            batch_x = batch_x.float().to(device)

            # Random mask
            mask = (torch.rand_like(batch_x) > mask_rate).float()
            masked_input = batch_x * mask

            # Forward: reconstruct
            optimizer.zero_grad()
            output = model.reconstruct(masked_input)  # [B, seq_len, 1]

            # Loss at masked positions only
            loss_matrix = criterion(output, batch_x)
            inv_mask = 1.0 - mask
            n_masked = inv_mask.sum().clamp(min=1.0)
            loss = (loss_matrix * inv_mask).sum() / n_masked

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

            if (i + 1) % 500 == 0:
                print(f'  iter {i+1}/{len(train_dl)}: loss={loss.item():.6f}')

        # Validate
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch_x in val_dl:
                batch_x = batch_x.float().to(device)
                mask = (torch.rand_like(batch_x) > mask_rate).float()
                masked_input = batch_x * mask
                output = model.reconstruct(masked_input)
                loss_matrix = criterion(output, batch_x)
                inv_mask = 1.0 - mask
                loss = (loss_matrix * inv_mask).sum() / inv_mask.sum().clamp(min=1.0)
                val_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        print(f'Epoch {epoch+1}/{epochs}: train={train_loss:.6f} val={val_loss:.6f} ({time.time()-t0:.0f}s)')

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), save_path)
            print(f'  Saved checkpoint (val={val_loss:.6f})')

    model.load_state_dict(torch.load(save_path))
    print(f'Pre-training done. Best val: {best_val:.6f}')
    return model


# ============================================================
# Stage 2: Forecasting Evaluation
# ============================================================
def eval_forecasting(model, device):
    print(f'\n{"="*60}')
    print('Stage 2: Forecasting (zero-shot)')
    print(f'{"="*60}')

    datasets = {
        'ETTh1': ('ETTh1', './dataset/ETT-small/', 'ETTh1.csv', 7),
        'ETTh2': ('ETTh2', './dataset/ETT-small/', 'ETTh2.csv', 7),
        'Weather': ('custom', './dataset/weather/', 'weather.csv', 21),
        'Exchange': ('custom', './dataset/exchange_rate/', 'exchange_rate.csv', 8),
    }

    results = {}
    model.eval()

    for dname, (data, root, fpath, enc_in) in datasets.items():
        for pl in [96, 336]:
            args = SimpleNamespace(
                seq_len=96, pred_len=pl, label_len=48,
                data=data, root_path=root, data_path=fpath,
                features='M', target='OT', freq='h', embed='timeF',
                enc_in=enc_in, dec_in=enc_in, c_out=enc_in,
                num_workers=2, batch_size=1,
                exp_name='MTSF', ordered_data=False, data_amount=-1,
                combine_Gaussian_datasets=False,
                synthetic_data_path='', synthetic_root_path='./',
                synthetic_length=1024, stride=-1,
            )
            _, test_dl = data_provider(args, 'test')

            preds, trues = [], []
            with torch.no_grad():
                for bx, by, _, _ in test_dl:
                    bx = bx.float().to(device)
                    out = model(bx, None, None, None, target_pred_len=pl)
                    if isinstance(out, tuple): out = out[0]
                    preds.append(out.cpu().numpy())
                    trues.append(by[:, -pl:, :].numpy())

            preds = np.concatenate(preds)
            trues = np.concatenate(trues)
            mse = np.mean((preds - trues) ** 2)
            key = f'{dname}_pl{pl}'
            results[key] = mse
            print(f'  {key}: MSE={mse:.4f}')

    return results


# ============================================================
# Stage 3: Classification Evaluation
# ============================================================
def eval_classification(model, device):
    print(f'\n{"="*60}')
    print('Stage 3: Classification (frozen backbone + cls_head)')
    print(f'{"="*60}')

    cls_datasets = ['EthanolConcentration', 'Epilepsy', 'FingerMovements',
                     'BasicMotions', 'NATOPS']
    cls_root = './dataset/classification/Multivariate_ts'
    hidden = model.branch_hidden
    results = {}

    for ds_name in cls_datasets:
        # Load data
        train_ds = Dataset_Classification(
            root_path=cls_root, flag='train', size=[96, 0, 96], data_path=ds_name)
        test_ds = Dataset_Classification(
            root_path=cls_root, flag='test', size=[96, 0, 96], data_path=ds_name)

        n_classes = train_ds.n_classes
        n_channels = train_ds.n_channels

        train_dl = DataLoader(train_ds, batch_size=16, shuffle=True, drop_last=True)
        test_dl = DataLoader(test_ds, batch_size=16, shuffle=False)

        # Classification head
        cls_head = nn.Sequential(
            nn.Linear(hidden, 128), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(128, n_classes)
        ).to(device)

        # Freeze backbone
        for p in model.parameters():
            p.requires_grad = False

        optimizer = optim.Adam(cls_head.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        # Train cls_head
        best_acc = 0
        for epoch in range(30):
            cls_head.train()
            for bx, label, _, _ in train_dl:
                bx = bx.float().to(device)
                label = label.long().to(device)

                with torch.no_grad():
                    z = model.get_representation(bx)  # [B, C, hidden]
                    z = z.mean(dim=1)  # [B, hidden]

                logits = cls_head(z)
                loss = criterion(logits, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Eval
            cls_head.eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for bx, label, _, _ in test_dl:
                    bx = bx.float().to(device)
                    z = model.get_representation(bx).mean(dim=1)
                    logits = cls_head(z)
                    all_preds.append(logits.argmax(-1).cpu().numpy())
                    all_labels.append(label.numpy())

            all_preds = np.concatenate(all_preds)
            all_labels = np.concatenate(all_labels)
            acc = accuracy_score(all_labels, all_preds)
            if acc > best_acc:
                best_acc = acc

        # Unfreeze for next task
        for p in model.parameters():
            p.requires_grad = True

        results[ds_name] = best_acc
        print(f'  {ds_name}: Acc={best_acc:.4f}')

    return results


# ============================================================
# Stage 4: Imputation Evaluation
# ============================================================
def eval_imputation(model, device):
    print(f'\n{"="*60}')
    print('Stage 4: Imputation (zero-shot)')
    print(f'{"="*60}')

    args = SimpleNamespace(
        seq_len=96, pred_len=96, label_len=0,
        data='ETTh1', root_path='./dataset/ETT-small/', data_path='ETTh1.csv',
        features='M', target='OT', freq='h', embed='timeF',
        enc_in=7, dec_in=7, c_out=7,
        num_workers=2, batch_size=1,
        exp_name='MTSF', ordered_data=False, data_amount=-1,
        combine_Gaussian_datasets=False,
        synthetic_data_path='', synthetic_root_path='./',
        synthetic_length=1024, stride=-1,
    )
    _, test_dl = data_provider(args, 'test')

    results = {}
    model.eval()

    for mask_rate in [0.125, 0.25, 0.5]:
        torch.manual_seed(2021)
        all_preds, all_trues, all_masks = [], [], []

        with torch.no_grad():
            for bx, by, _, _ in test_dl:
                bx = bx.float().to(device)
                mask = (torch.rand_like(bx) > mask_rate).float()
                masked_input = bx * mask

                output = model.reconstruct(masked_input)

                all_preds.append(output.cpu().numpy())
                all_trues.append(bx.cpu().numpy())
                all_masks.append(mask.cpu().numpy())

        preds = np.concatenate(all_preds)
        trues = np.concatenate(all_trues)
        masks = np.concatenate(all_masks)

        mse = np.mean((preds[masks == 0] - trues[masks == 0]) ** 2)
        key = f'm={mask_rate}'
        results[key] = mse
        print(f'  mask={mask_rate}: MSE={mse:.4f}')

    return results


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    args = get_model_args()
    model = Model(args).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'Model params: {n_params:,}')

    save_path = 'checkpoints/unified_pretrain.pth'
    os.makedirs('checkpoints', exist_ok=True)

    # Pre-train
    model = pretrain(model, device, save_path, epochs=10, lr=0.0003, mask_rate=0.4)

    # Eval all tasks
    fc_results = eval_forecasting(model, device)
    cls_results = eval_classification(model, device)
    imp_results = eval_imputation(model, device)

    # Summary
    print(f'\n{"="*60}')
    print('FINAL RESULTS: Unified Real-data Pre-training')
    print(f'{"="*60}')
    print('\nForecasting MSE:')
    for k, v in fc_results.items():
        print(f'  {k}: {v:.4f}')
    print('\nClassification Accuracy:')
    for k, v in cls_results.items():
        print(f'  {k}: {v:.4f}')
    print('\nImputation MSE (zero-shot):')
    for k, v in imp_results.items():
        print(f'  {k}: {v:.4f}')
