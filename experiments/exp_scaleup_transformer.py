"""
Scale-up: Transformer + Fixed Trunk + Cross-channel + Time Series Pile pretrain
Same as exp_scaleup_pretrain.py but uses TransformerOperatorModel from exp_architecture_comparison.py

사용법:
  CUDA_VISIBLE_DEVICES=0 python experiments/exp_scaleup_transformer.py 2>&1 | tee log/scaleup_transformer.log
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

from experiments.exp_architecture_comparison import TransformerOperatorModel
from data_provider.pile_dataset import PilePretrainDataset
from data_provider.data_factory import data_provider
from data_provider.data_loader import Dataset_Classification


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
            loss = (loss_matrix * inv_mask).sum() / inv_mask.sum().clamp(min=1.0)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

            if (i + 1) % 500 == 0:
                print(f'  iter {i+1}/{len(train_dl)}: loss={loss.item():.6f}')

        scheduler.step()

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
    return model


def eval_forecasting(model, device):
    print(f'\n{"="*60}')
    print('Stage 2: Forecasting')
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
                    model.pred_len = pl
                    out = model(bx, target_pred_len=pl)
                    preds.append(out.cpu().numpy())
                    trues.append(by[:, -pl:, :].numpy())
            preds, trues = np.concatenate(preds), np.concatenate(trues)
            mse = np.mean((preds - trues) ** 2)
            mae = np.mean(np.abs(preds - trues))
            results[f'{dname}_pl{pl}'] = {'mse': mse, 'mae': mae}
            print(f'  {dname}_pl{pl}: MSE={mse:.4f} MAE={mae:.4f}')
    return results


def eval_classification(model, device):
    print(f'\n{"="*60}')
    print('Stage 3: Classification')
    print(f'{"="*60}')
    hidden = model.hidden
    results = {}
    for p in model.parameters():
        p.requires_grad = False

    for ds_name in ['Epilepsy', 'FingerMovements', 'BasicMotions', 'NATOPS', 'EthanolConcentration']:
        cls_root = './dataset/classification/Multivariate_ts'
        train_ds = Dataset_Classification(root_path=cls_root, flag='train', size=[96,0,96], data_path=ds_name)
        test_ds = Dataset_Classification(root_path=cls_root, flag='test', size=[96,0,96], data_path=ds_name)
        train_dl = DataLoader(train_ds, batch_size=16, shuffle=True, drop_last=True)
        test_dl = DataLoader(test_ds, batch_size=16, shuffle=False)

        cls_head = nn.Sequential(
            nn.Linear(hidden, 256), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(256, train_ds.n_classes)
        ).to(device)

        opt = optim.Adam(cls_head.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        best_acc = 0
        for epoch in range(30):
            cls_head.train()
            for bx, label, _, _ in train_dl:
                bx = bx.float().to(device); label = label.long().to(device)
                with torch.no_grad(): z = model.get_representation(bx).mean(dim=1)
                loss = criterion(cls_head(z), label)
                opt.zero_grad(); loss.backward(); opt.step()
            cls_head.eval()
            preds, labels = [], []
            with torch.no_grad():
                for bx, label, _, _ in test_dl:
                    bx = bx.float().to(device)
                    z = model.get_representation(bx).mean(dim=1)
                    preds.append(cls_head(z).argmax(-1).cpu().numpy())
                    labels.append(label.numpy())
            acc = accuracy_score(np.concatenate(labels), np.concatenate(preds))
            best_acc = max(best_acc, acc)
        results[ds_name] = best_acc
        print(f'  {ds_name}: Acc={best_acc:.4f}')

    for p in model.parameters():
        p.requires_grad = True
    return results


def eval_imputation(model, device):
    print(f'\n{"="*60}')
    print('Stage 4: Imputation')
    print(f'{"="*60}')
    args = SimpleNamespace(
        seq_len=96, pred_len=96, label_len=0,
        data='ETTh1', root_path='./dataset/ETT-small/', data_path='ETTh1.csv',
        features='M', target='OT', freq='h', embed='timeF',
        enc_in=7, dec_in=7, c_out=7, num_workers=2, batch_size=1,
        exp_name='MTSF', ordered_data=False, data_amount=-1,
        combine_Gaussian_datasets=False, synthetic_data_path='', synthetic_root_path='./',
        synthetic_length=1024, stride=-1,
    )
    _, test_dl = data_provider(args, 'test')
    results = {}
    model.eval()
    for mask_rate in [0.125, 0.25, 0.5]:
        torch.manual_seed(2021)
        preds, trues, masks = [], [], []
        with torch.no_grad():
            for bx, by, _, _ in test_dl:
                bx = bx.float().to(device)
                mask = (torch.rand_like(bx) > mask_rate).float()
                out = model.reconstruct(bx * mask)
                preds.append(out.cpu().numpy()); trues.append(bx.cpu().numpy()); masks.append(mask.cpu().numpy())
        p, t, m = np.concatenate(preds), np.concatenate(trues), np.concatenate(masks)
        mse = np.mean((p[m==0] - t[m==0])**2)
        results[f'm={mask_rate}'] = mse
        print(f'  mask={mask_rate}: MSE={mse:.4f}')
    return results


if __name__ == '__main__':
    device = torch.device('cuda')

    # Scale-up Transformer + Fixed Trunk + Cross-channel
    model = TransformerOperatorModel(
        seq_len=96, pred_len=96,
        width=128,       # 64 → 128
        hidden=512,      # 256 → 512
        n_heads=8,       # 4 → 8
        n_layers=4,      # 3 → 4
        trunk_depth=4,   # 2 → 4
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f'Transformer Operator Model: {n_params:,} params')

    save_path = 'checkpoints/scaleup_transformer_pile.pth'
    os.makedirs('checkpoints', exist_ok=True)

    model = pretrain(model, device, save_path, epochs=20, lr=0.0003)
    fc = eval_forecasting(model, device)
    cls = eval_classification(model, device)
    imp = eval_imputation(model, device)

    print(f'\n{"="*60}')
    print(f'FINAL: Transformer Operator ({n_params/1e6:.1f}M) + Pile')
    print(f'{"="*60}')
    for k, v in fc.items(): print(f'  {k}: MSE={v["mse"]:.4f}')
    for k, v in cls.items(): print(f'  {k}: {v:.4f}')
    for k, v in imp.items(): print(f'  {k}: {v:.4f}')
