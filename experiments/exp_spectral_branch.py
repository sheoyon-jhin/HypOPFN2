"""
Spectral Branch 복원 실험: spectral_branch=True로 63M + Pile pretrain
이전에 +5.4 Rank 개선이었던 설정을 다시 테스트

사용법:
  CUDA_VISIBLE_DEVICES=X python experiments/exp_spectral_branch.py 2>&1 | tee log/spectral_branch.log
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


def pretrain(model, device, save_path, epochs=20, lr=0.0003, mask_rate=0.4):
    print(f'\n{"="*60}')
    print('Pre-training: Spectral Branch ON + Masked Recon + FC loss')
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
            batch_x = batch_x.float().to(device)
            B, S, C = batch_x.shape
            optimizer.zero_grad()

            # Masked recon
            mask = (torch.rand_like(batch_x) > mask_rate).float()
            recon_out = model.reconstruct(batch_x * mask)
            loss_mat = recon_criterion(recon_out, batch_x)
            inv_mask = 1.0 - mask
            recon_loss = (loss_mat * inv_mask).sum() / inv_mask.sum().clamp(min=1)

            # Forecasting
            split = torch.randint(24, 72, (1,)).item()
            context = batch_x[:, :split, :]
            target = batch_x[:, split:, :]
            target_len = S - split
            context_padded = F.pad(context, (0, 0, 0, S - split))
            fc_out = model(context_padded, None, None, None, target_pred_len=target_len)
            if isinstance(fc_out, tuple): fc_out = fc_out[0]
            fc_loss = F.mse_loss(fc_out, target)

            loss = recon_loss + fc_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())

            if (i + 1) % 500 == 0:
                print(f'  iter {i+1}/{len(train_dl)}: recon={recon_loss.item():.4f} fc={fc_loss.item():.4f}')

        scheduler.step()
        train_loss = np.mean(losses)
        print(f'Epoch {epoch+1}/{epochs}: loss={train_loss:.4f} lr={scheduler.get_last_lr()[0]:.6f} ({time.time()-t0:.0f}s)')

        if train_loss < best_val:
            best_val = train_loss
            torch.save(model.state_dict(), save_path)
            print(f'  Saved checkpoint')

    model.load_state_dict(torch.load(save_path))
    return model


def eval_all(model, device):
    print(f'\n=== Forecasting ===')
    datasets = {
        'ETTh1': ('ETTh1', './dataset/ETT-small/', 'ETTh1.csv', 7),
        'ETTh2': ('ETTh2', './dataset/ETT-small/', 'ETTh2.csv', 7),
        'Weather': ('custom', './dataset/weather/', 'weather.csv', 21),
        'Exchange': ('custom', './dataset/exchange_rate/', 'exchange_rate.csv', 8),
    }
    model.eval()
    for dname, (data, root, fpath, enc_in) in datasets.items():
        for pl in [96, 336]:
            a = SimpleNamespace(seq_len=96, pred_len=pl, label_len=48, data=data, root_path=root,
                data_path=fpath, features='M', target='OT', freq='h', embed='timeF',
                enc_in=enc_in, dec_in=enc_in, c_out=enc_in, num_workers=2, batch_size=1,
                exp_name='MTSF', ordered_data=False, data_amount=-1,
                combine_Gaussian_datasets=False, synthetic_data_path='', synthetic_root_path='./',
                synthetic_length=1024, stride=-1)
            _, test_dl = data_provider(a, 'test')
            preds, trues = [], []
            with torch.no_grad():
                for bx, by, _, _ in test_dl:
                    bx = bx.float().to(device)
                    out = model(bx, None, None, None, target_pred_len=pl)
                    if isinstance(out, tuple): out = out[0]
                    preds.append(out.cpu().numpy()); trues.append(by[:, -pl:, :].numpy())
            p, t = np.concatenate(preds), np.concatenate(trues)
            print(f'  {dname}_pl{pl}: MSE={np.mean((p-t)**2):.4f}')

    print(f'\n=== Classification ===')
    hidden = model.branch_hidden
    for p in model.parameters(): p.requires_grad = False
    for ds_name in ['Epilepsy', 'BasicMotions', 'NATOPS', 'FingerMovements']:
        cls_root = './dataset/classification/Multivariate_ts'
        train_ds = Dataset_Classification(root_path=cls_root, flag='train', size=[96,0,96], data_path=ds_name)
        test_ds = Dataset_Classification(root_path=cls_root, flag='test', size=[96,0,96], data_path=ds_name)
        train_dl = DataLoader(train_ds, batch_size=16, shuffle=True, drop_last=True)
        test_dl = DataLoader(test_ds, batch_size=16, shuffle=False)
        cls_head = nn.Sequential(nn.Linear(hidden, 256), nn.GELU(), nn.Dropout(0.1),
                                 nn.Linear(256, train_ds.n_classes)).to(device)
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
            ps, ls = [], []
            with torch.no_grad():
                for bx, label, _, _ in test_dl:
                    bx = bx.float().to(device)
                    z = model.get_representation(bx).mean(dim=1)
                    ps.append(cls_head(z).argmax(-1).cpu().numpy()); ls.append(label.numpy())
            acc = accuracy_score(np.concatenate(ls), np.concatenate(ps))
            best_acc = max(best_acc, acc)
        print(f'  {ds_name}: Acc={best_acc:.4f}')
    for p in model.parameters(): p.requires_grad = True

    print(f'\n=== Imputation ===')
    a = SimpleNamespace(seq_len=96, pred_len=96, label_len=0, data='ETTh1',
        root_path='./dataset/ETT-small/', data_path='ETTh1.csv',
        features='M', target='OT', freq='h', embed='timeF',
        enc_in=7, dec_in=7, c_out=7, num_workers=2, batch_size=1,
        exp_name='MTSF', ordered_data=False, data_amount=-1,
        combine_Gaussian_datasets=False, synthetic_data_path='', synthetic_root_path='./',
        synthetic_length=1024, stride=-1)
    _, test_dl = data_provider(a, 'test')
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
        print(f'  mask={mask_rate}: MSE={np.mean((p[m==0] - t[m==0])**2):.4f}')

    print('\n=== DONE ===')


if __name__ == '__main__':
    device = torch.device('cuda')

    # spectral_branch=True! 나머지는 63M과 동일
    args = SimpleNamespace(
        seq_len=96, pred_len=96, use_norm=True, deeponet_width=128,
        n_experts=4, branch_depth=4, trunk_depth=2, activation='gelu',
        dropout=0.1, branch_hidden=512,
        spectral_branch=True,     # ← ON
        skip_mode='none',
        use_cross_channel=False,
        trunk_basis='mixed',
        encoder_type='patch_attn', loss='MSE'
    )
    model = Model(args).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'Spectral Branch Model: {n_params/1e6:.1f}M params')

    save_path = 'checkpoints/spectral_branch_pile.pth'
    os.makedirs('checkpoints', exist_ok=True)

    model = pretrain(model, device, save_path, epochs=20, lr=0.0003)
    eval_all(model, device)
