"""
Scale-up Model + Time Series Pile Pre-training + 5 Task Evaluation.

Model: Hyper Trunk + PatchAttn, scaled up (~40M params)
  - width: 64 → 128 (basis functions)
  - hidden: 256 → 512 (encoder hidden dim)
  - trunk_depth: 2 (keep, hyper so can't go deep)

Data: Time Series Pile (same as MOMENT)

사용법:
  CUDA_VISIBLE_DEVICES=0 python experiments/exp_scaleup_pretrain.py 2>&1 | tee log/scaleup_pretrain.log
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
from sklearn.metrics import accuracy_score
import time

from model.DeepONetHyperMoE import Model
from data_provider.pile_dataset import PilePretrainDataset
from data_provider.data_factory import data_provider
from data_provider.data_loader import Dataset_Classification


def get_scaleup_args():
    """Scaled-up model config (~40M params)."""
    return SimpleNamespace(
        seq_len=96, pred_len=96, use_norm=True,
        deeponet_width=128,       # 64 → 128
        n_experts=4,
        branch_depth=4,
        trunk_depth=2,
        activation='gelu',
        dropout=0.1,
        branch_hidden=512,        # 256 → 512
        spectral_branch=False,
        skip_mode='none',
        use_cross_channel=False,
        trunk_basis='mixed',
        encoder_type='patch_attn',
        loss='MSE',
    )


def pretrain(model, device, save_path, epochs=20, lr=0.0003, mask_rate=0.4):
    print(f'\n{"="*60}')
    print('Stage 1: Pre-training on Time Series Pile')
    print(f'{"="*60}')

    dataset = PilePretrainDataset(seq_len=96, stride=48,
                                   pile_root='./dataset/time_series_pile')

    n_val = min(10000, len(dataset) // 10)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_dl = DataLoader(train_ds, batch_size=128, shuffle=True,
                          num_workers=4, drop_last=True, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=2)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss(reduction='none')

    print(f'Train: {n_train}, Val: {n_val}, Steps/epoch: {len(train_dl)}')

    best_val = float('inf')
    for epoch in range(epochs):
        model.train()
        train_losses = []
        t0 = time.time()

        for i, batch_x in enumerate(train_dl):
            batch_x = batch_x.float().to(device)

            mask = (torch.rand_like(batch_x) > mask_rate).float()
            masked_input = batch_x * mask

            optimizer.zero_grad()
            output = model.reconstruct(masked_input)

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

        scheduler.step()

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
        lr_now = scheduler.get_last_lr()[0]
        print(f'Epoch {epoch+1}/{epochs}: train={train_loss:.6f} val={val_loss:.6f} lr={lr_now:.6f} ({time.time()-t0:.0f}s)')

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), save_path)
            print(f'  Saved checkpoint (val={val_loss:.6f})')

    model.load_state_dict(torch.load(save_path))
    print(f'Pre-training done. Best val: {best_val:.6f}')
    return model


def eval_forecasting(model, device):
    print(f'\n{"="*60}')
    print('Stage 2: Forecasting Evaluation')
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
        for pl in [96, 336, 720]:
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
            mae = np.mean(np.abs(preds - trues))
            key = f'{dname}_pl{pl}'
            results[key] = {'mse': mse, 'mae': mae}
            print(f'  {key}: MSE={mse:.4f} MAE={mae:.4f}')

    return results


def eval_classification(model, device):
    print(f'\n{"="*60}')
    print('Stage 3: Classification Evaluation')
    print(f'{"="*60}')

    cls_datasets = ['EthanolConcentration', 'Epilepsy', 'FingerMovements',
                     'BasicMotions', 'NATOPS']
    cls_root = './dataset/classification/Multivariate_ts'
    hidden = model.branch_hidden
    results = {}

    for ds_name in cls_datasets:
        train_ds = Dataset_Classification(
            root_path=cls_root, flag='train', size=[96, 0, 96], data_path=ds_name)
        test_ds = Dataset_Classification(
            root_path=cls_root, flag='test', size=[96, 0, 96], data_path=ds_name)

        train_dl = DataLoader(train_ds, batch_size=16, shuffle=True, drop_last=True)
        test_dl = DataLoader(test_ds, batch_size=16, shuffle=False)

        cls_head = nn.Sequential(
            nn.Linear(hidden, 256), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(256, train_ds.n_classes)
        ).to(device)

        # Freeze backbone
        for p in model.parameters():
            p.requires_grad = False

        optimizer = optim.Adam(cls_head.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        best_acc = 0
        for epoch in range(30):
            cls_head.train()
            for bx, label, _, _ in train_dl:
                bx = bx.float().to(device)
                label = label.long().to(device)
                with torch.no_grad():
                    z = model.get_representation(bx).mean(dim=1)
                logits = cls_head(z)
                loss = criterion(logits, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            cls_head.eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for bx, label, _, _ in test_dl:
                    bx = bx.float().to(device)
                    z = model.get_representation(bx).mean(dim=1)
                    logits = cls_head(z)
                    all_preds.append(logits.argmax(-1).cpu().numpy())
                    all_labels.append(label.numpy())
            acc = accuracy_score(np.concatenate(all_labels), np.concatenate(all_preds))
            if acc > best_acc:
                best_acc = acc

        for p in model.parameters():
            p.requires_grad = True

        results[ds_name] = best_acc
        print(f'  {ds_name}: Acc={best_acc:.4f}')

    return results


def eval_imputation(model, device):
    print(f'\n{"="*60}')
    print('Stage 4: Imputation Evaluation')
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


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    args = get_scaleup_args()
    model = Model(args).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'Scale-up Model params: {n_params:,}')
    print(f'  width={args.deeponet_width}, hidden={args.branch_hidden}')
    print(f'  encoder={args.encoder_type}, trunk_basis={args.trunk_basis}')

    save_path = 'checkpoints/scaleup_pile_pretrain.pth'
    os.makedirs('checkpoints', exist_ok=True)

    # Pre-train on Time Series Pile
    model = pretrain(model, device, save_path, epochs=20, lr=0.0003, mask_rate=0.4)

    # Eval all tasks
    fc_results = eval_forecasting(model, device)
    cls_results = eval_classification(model, device)
    imp_results = eval_imputation(model, device)

    # Summary
    print(f'\n{"="*60}')
    print(f'FINAL RESULTS: Scale-up ({n_params/1e6:.1f}M) + Time Series Pile')
    print(f'{"="*60}')
    print('\nForecasting MSE:')
    for k, v in fc_results.items():
        print(f'  {k}: MSE={v["mse"]:.4f} MAE={v["mae"]:.4f}')
    print('\nClassification Accuracy:')
    for k, v in cls_results.items():
        print(f'  {k}: {v:.4f}')
    print('\nImputation MSE:')
    for k, v in imp_results.items():
        print(f'  {k}: {v:.4f}')
