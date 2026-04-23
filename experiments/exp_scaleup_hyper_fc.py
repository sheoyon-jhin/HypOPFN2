"""
Scale-up: Hyper Trunk (63M) + FC loss + Time Series Pile + 5 Task Eval
Masked Recon + Forecasting loss 동시 학습

사용법:
  CUDA_VISIBLE_DEVICES=0 python experiments/exp_scaleup_hyper_fc.py 2>&1 | tee log/scaleup_hyper_fc.log
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
    return SimpleNamespace(
        seq_len=96, pred_len=96, use_norm=True,
        deeponet_width=128, n_experts=4, branch_depth=4, trunk_depth=2,
        activation='gelu', dropout=0.1, branch_hidden=512,
        spectral_branch=False, skip_mode='none',
        use_cross_channel=False, trunk_basis='mixed',
        encoder_type='patch_attn', loss='MSE',
    )


def pretrain(model, device, save_path, epochs=20, lr=0.0003, mask_rate=0.4):
    print(f'\n{"="*60}')
    print('Pre-training: Masked Recon + Forecasting Loss')
    print(f'{"="*60}')

    dataset = PilePretrainDataset(seq_len=96, stride=48,
                                   pile_root='./dataset/time_series_pile')
    n_val = min(10000, len(dataset) // 10)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])
    train_dl = DataLoader(train_ds, batch_size=128, shuffle=True,
                          num_workers=4, drop_last=True, pin_memory=True)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    recon_criterion = nn.MSELoss(reduction='none')

    print(f'Train: {n_train}, Steps/epoch: {len(train_dl)}')

    best_val = float('inf')
    for epoch in range(epochs):
        model.train()
        losses, recon_losses, fc_losses = [], [], []
        t0 = time.time()

        for i, batch_x in enumerate(train_dl):
            batch_x = batch_x.float().to(device)
            B, S, C = batch_x.shape
            optimizer.zero_grad()

            # 1) Masked reconstruction
            mask = (torch.rand_like(batch_x) > mask_rate).float()
            recon_out = model.reconstruct(batch_x * mask)
            loss_mat = recon_criterion(recon_out, batch_x)
            inv_mask = 1.0 - mask
            recon_loss = (loss_mat * inv_mask).sum() / inv_mask.sum().clamp(min=1)

            # 2) Forecasting: first half → predict second half
            half = S // 2
            x_input = batch_x[:, :half, :]
            x_target = batch_x[:, half:, :]
            x_padded = F.pad(x_input, (0, 0, 0, S - half))
            fc_out = model(x_padded, None, None, None, target_pred_len=half)
            if isinstance(fc_out, tuple):
                fc_out = fc_out[0]
            fc_loss = F.mse_loss(fc_out, x_target)

            loss = recon_loss + fc_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            losses.append(loss.item())
            recon_losses.append(recon_loss.item())
            fc_losses.append(fc_loss.item())

            if (i + 1) % 500 == 0:
                print(f'  iter {i+1}/{len(train_dl)}: recon={np.mean(recon_losses[-500:]):.4f} fc={np.mean(fc_losses[-500:]):.4f}')

        scheduler.step()
        train_loss = np.mean(losses)
        lr_now = scheduler.get_last_lr()[0]
        print(f'Epoch {epoch+1}/{epochs}: loss={train_loss:.4f} (recon={np.mean(recon_losses):.4f} fc={np.mean(fc_losses):.4f}) lr={lr_now:.6f} ({time.time()-t0:.0f}s)')

        if train_loss < best_val:
            best_val = train_loss
            torch.save(model.state_dict(), save_path)
            print(f'  Saved checkpoint')

    model.load_state_dict(torch.load(save_path))
    return model


def eval_forecasting(model, device):
    print(f'\n{"="*60}')
    print('Forecasting (zero-shot)')
    print(f'{"="*60}')
    datasets = {
        'ETTh1': ('ETTh1', './dataset/ETT-small/', 'ETTh1.csv', 7),
        'ETTh2': ('ETTh2', './dataset/ETT-small/', 'ETTh2.csv', 7),
        'ETTm1': ('ETTm1', './dataset/ETT-small/', 'ETTm1.csv', 7),
        'ETTm2': ('ETTm2', './dataset/ETT-small/', 'ETTm2.csv', 7),
        'Weather': ('custom', './dataset/weather/', 'weather.csv', 21),
        'Exchange': ('custom', './dataset/exchange_rate/', 'exchange_rate.csv', 8),
    }
    results = {}
    model.eval()
    for dname, (data, root, fpath, enc_in) in datasets.items():
        for pl in [96, 192, 336, 720]:
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
            p, t = np.concatenate(preds), np.concatenate(trues)
            mse = np.mean((p - t) ** 2)
            mae = np.mean(np.abs(p - t))
            results[f'{dname}_pl{pl}'] = {'mse': mse, 'mae': mae}
            print(f'  {dname}_pl{pl}: MSE={mse:.4f} MAE={mae:.4f}')
    return results


def eval_classification(model, device):
    print(f'\n{"="*60}')
    print('Classification (frozen + head)')
    print(f'{"="*60}')
    hidden = model.branch_hidden
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
        best_acc = 0
        for epoch in range(30):
            cls_head.train()
            for bx, label, _, _ in train_dl:
                bx = bx.float().to(device); label = label.long().to(device)
                with torch.no_grad(): z = model.get_representation(bx).mean(dim=1)
                loss = nn.CrossEntropyLoss()(cls_head(z), label)
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
    print('Imputation (zero-shot)')
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
    args = get_scaleup_args()
    model = Model(args).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'Hyper Scaleup + FC loss: {n_params:,} params')

    save_path = 'checkpoints/scaleup_hyper_fc_pile.pth'
    os.makedirs('checkpoints', exist_ok=True)

    model = pretrain(model, device, save_path, epochs=20, lr=0.0003)
    fc = eval_forecasting(model, device)
    cls = eval_classification(model, device)
    imp = eval_imputation(model, device)

    print(f'\n{"="*60}')
    print(f'FINAL: Hyper+FC ({n_params/1e6:.1f}M) + Pile')
    print(f'{"="*60}')
    print('\nForecasting:')
    for k, v in fc.items(): print(f'  {k}: MSE={v["mse"]:.4f} MAE={v["mae"]:.4f}')
    print('\nClassification:')
    for k, v in cls.items(): print(f'  {k}: {v:.4f}')
    print('\nImputation:')
    for k, v in imp.items(): print(f'  {k}: {v:.4f}')
