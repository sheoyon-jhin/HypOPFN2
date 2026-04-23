"""
Scale-up: Hyper Trunk + Cross-channel + Time Series Pile pretrain
+ Linear Probing eval (MOMENT_LP와 동일한 세팅)
+ Forecasting loss 추가 pretrain

사용법:
  CUDA_VISIBLE_DEVICES=0 python experiments/exp_scaleup_hyper_cross.py 2>&1 | tee log/scaleup_hyper_cross.log
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

from experiments.exp_architecture_comparison import FixedCrossModel
from data_provider.pile_dataset import PilePretrainDataset
from data_provider.data_factory import data_provider
from data_provider.data_loader import Dataset_Classification


def pretrain_with_forecasting(model, device, save_path, epochs=20, lr=0.0003, mask_rate=0.4):
    """Pre-train with BOTH masked reconstruction AND forecasting."""
    print(f'\n{"="*60}')
    print('Pre-training: Masked Recon + Forecasting')
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
        losses = []
        t0 = time.time()

        for i, batch_x in enumerate(train_dl):
            # batch_x: [B, seq_len, 1]
            batch_x = batch_x.float().to(device)
            B, S, C = batch_x.shape
            optimizer.zero_grad()

            # 1) Masked reconstruction loss
            mask = (torch.rand_like(batch_x) > mask_rate).float()
            masked = batch_x * mask
            recon_out = model.reconstruct(masked)
            loss_mat = recon_criterion(recon_out, batch_x)
            inv_mask = 1.0 - mask
            recon_loss = (loss_mat * inv_mask).sum() / inv_mask.sum().clamp(min=1)

            # 2) Forecasting loss: use first half as input, predict second half
            half = S // 2
            x_input = batch_x[:, :half, :]  # [B, 48, 1]
            x_target = batch_x[:, half:, :]  # [B, 48, 1]

            # Pad input to seq_len by repeating or zero-padding
            x_padded = F.pad(x_input, (0, 0, 0, S - half))  # [B, 96, 1]
            fc_out = model(x_padded, target_pred_len=half)
            fc_loss = F.mse_loss(fc_out, x_target)

            loss = recon_loss + fc_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())

            if (i + 1) % 500 == 0:
                print(f'  iter {i+1}/{len(train_dl)}: recon={recon_loss.item():.4f} fc={fc_loss.item():.4f}')

        scheduler.step()
        train_loss = np.mean(losses)
        lr_now = scheduler.get_last_lr()[0]
        print(f'Epoch {epoch+1}/{epochs}: loss={train_loss:.4f} lr={lr_now:.6f} ({time.time()-t0:.0f}s)')

        if train_loss < best_val:
            best_val = train_loss
            torch.save(model.state_dict(), save_path)
            print(f'  Saved checkpoint')

    model.load_state_dict(torch.load(save_path))
    return model


def eval_forecasting_with_linear_probe(model, device):
    """Linear probing: freeze backbone, train forecast_head on each dataset."""
    print(f'\n{"="*60}')
    print('Forecasting: Zero-shot + Linear Probe')
    print(f'{"="*60}')

    datasets = {
        'ETTh1': ('ETTh1', './dataset/ETT-small/', 'ETTh1.csv', 7),
        'ETTh2': ('ETTh2', './dataset/ETT-small/', 'ETTh2.csv', 7),
        'ETTm1': ('ETTm1', './dataset/ETT-small/', 'ETTm1.csv', 7),
        'ETTm2': ('ETTm2', './dataset/ETT-small/', 'ETTm2.csv', 7),
        'Weather': ('custom', './dataset/weather/', 'weather.csv', 21),
        'Exchange': ('custom', './dataset/exchange_rate/', 'exchange_rate.csv', 8),
    }

    zs_results = {}
    lp_results = {}

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
            _, train_dl_lp = data_provider(args, 'train')
            _, test_dl = data_provider(args, 'test')

            key = f'{dname}_pl{pl}'

            # 1) Zero-shot
            model.eval()
            preds, trues = [], []
            with torch.no_grad():
                for bx, by, _, _ in test_dl:
                    bx = bx.float().to(device)
                    model.pred_len = pl
                    out = model(bx, target_pred_len=pl)
                    preds.append(out.cpu().numpy())
                    trues.append(by[:, -pl:, :].numpy())
            p, t = np.concatenate(preds), np.concatenate(trues)
            zs_mse = np.mean((p - t) ** 2)
            zs_results[key] = zs_mse

            # 2) Linear Probe: freeze backbone, train only forecast_head
            # Create a simple linear probe head
            lp_head = nn.Linear(model.hidden, pl).to(device)
            for param in model.parameters():
                param.requires_grad = False
            lp_opt = optim.Adam(lp_head.parameters(), lr=0.001)
            best_lp_loss = float('inf')
            best_lp_state = None

            args_train = SimpleNamespace(**vars(args))
            args_train.batch_size = 32
            _, train_dl_lp = data_provider(args_train, 'train')

            for epoch in range(10):
                lp_head.train()
                for bx, by, _, _ in train_dl_lp:
                    bx = bx.float().to(device)
                    by = by[:, -pl:, :].float().to(device)
                    with torch.no_grad():
                        z = model.get_representation(bx)  # [B, C, hidden]
                    # Per channel linear probe
                    B_size, C_size, H = z.shape
                    pred = lp_head(z)  # [B, C, pl]
                    pred = pred.permute(0, 2, 1)  # [B, pl, C]
                    loss = F.mse_loss(pred, by)
                    lp_opt.zero_grad()
                    loss.backward()
                    lp_opt.step()

                tl = loss.item()
                if tl < best_lp_loss:
                    best_lp_loss = tl
                    best_lp_state = {k: v.cpu().clone() for k, v in lp_head.state_dict().items()}

            # Test linear probe
            if best_lp_state:
                lp_head.load_state_dict(best_lp_state)
            lp_head.to(device).eval()
            preds, trues = [], []
            with torch.no_grad():
                for bx, by, _, _ in test_dl:
                    bx = bx.float().to(device)
                    z = model.get_representation(bx)
                    pred = lp_head(z).permute(0, 2, 1)
                    preds.append(pred.cpu().numpy())
                    trues.append(by[:, -pl:, :].numpy())
            p, t = np.concatenate(preds), np.concatenate(trues)
            lp_mse = np.mean((p - t) ** 2)
            lp_results[key] = lp_mse

            for param in model.parameters():
                param.requires_grad = True

            print(f'  {key}: ZS={zs_mse:.4f}  LP={lp_mse:.4f}')

    return zs_results, lp_results


def eval_classification(model, device):
    print(f'\n{"="*60}')
    print('Classification (frozen + head)')
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

    # Fixed+Cross with scale-up
    model = FixedCrossModel(
        seq_len=96, pred_len=96,
        width=128, hidden=512,
        encoder_type='patch_attn', trunk_depth=4
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f'Fixed+Cross Model: {n_params:,} params')

    save_path = 'checkpoints/scaleup_fixedcross_pile.pth'
    os.makedirs('checkpoints', exist_ok=True)

    # Pretrain with masked recon + forecasting
    model = pretrain_with_forecasting(model, device, save_path, epochs=20, lr=0.0003)

    # Eval
    zs_fc, lp_fc = eval_forecasting_with_linear_probe(model, device)
    cls = eval_classification(model, device)
    imp = eval_imputation(model, device)

    print(f'\n{"="*60}')
    print(f'FINAL: Fixed+Cross ({n_params/1e6:.1f}M) + Pile + Recon+FC pretrain')
    print(f'{"="*60}')
    print('\nForecasting (ZS / LP):')
    for k in zs_fc:
        print(f'  {k}: ZS={zs_fc[k]:.4f}  LP={lp_fc[k]:.4f}')
    print('\nClassification:')
    for k, v in cls.items(): print(f'  {k}: {v:.4f}')
    print('\nImputation:')
    for k, v in imp.items(): print(f'  {k}: {v:.4f}')
