"""
Real + Synthetic 혼합 학습
Real (Pile): forecasting/imputation 패턴
Synthetic (TempoPFN): classification 패턴 (step, spike, sawtooth)
→ Real의 빈 구석을 Synthetic이 보완

검증된 세팅: MoE 4 + PatchAttn + Hyper Trunk + Spectral Branch
+ next-token + masked recon
+ batch=256, lr=0.0005
+ 매 epoch quick eval

사용법:
  CUDA_VISIBLE_DEVICES=1 python experiments/exp_real_synth_combined.py 2>&1 | tee log/prior/real_synth_combined.log
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import optim
from torch.utils.data import DataLoader, Dataset, ConcatDataset, random_split
from types import SimpleNamespace
from sklearn.metrics import accuracy_score
import time

from model.DeepONetHyperMoE import Model
from data_provider.pile_dataset import PilePretrainDataset
from data_provider.data_loader import Dataset_GaussianPCoregionalization, Dataset_Classification
from data_provider.data_factory import data_provider


class SyntheticWindowDataset(Dataset):
    """Synthetic arrow → 1채널 window dataset."""
    def __init__(self, arrow_path, seq_len=96, stride=48, max_samples=100000):
        self.windows = []
        ds = Dataset_GaussianPCoregionalization(
            root_path='./', data_path=arrow_path,
            n_variables=160, seq_len=seq_len, pred_len=seq_len,
            size=[seq_len, 0, seq_len], synthetic_length=1024, stride=stride
        )
        n_samples = min(len(ds), max_samples)
        for i in range(n_samples):
            x, y, _, _ = ds[i]
            if isinstance(x, torch.Tensor): x = x.numpy()
            if x.ndim == 1: x = x.reshape(-1, 1)
            ch = np.random.randint(0, x.shape[1])
            window = x[:, ch].astype(np.float32)
            std = np.std(window)
            if std > 1e-8:
                window = (window - np.mean(window)) / std
                window = np.clip(window, -10, 10)
                self.windows.append(window)
        self.windows = np.array(self.windows, dtype=np.float32)
        print(f'  Synthetic: {len(self.windows)} windows from {arrow_path}')

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return torch.tensor(self.windows[idx], dtype=torch.float32).unsqueeze(-1)


def quick_eval(model, device, epoch):
    """매 epoch 후 핵심 3개 빠르게 eval."""
    model.eval()
    results = {}

    # Forecasting: ETTh1 pl=96
    args = SimpleNamespace(
        seq_len=96, pred_len=96, label_len=48,
        data='ETTh1', root_path='./dataset/ETT-small/', data_path='ETTh1.csv',
        features='M', target='OT', freq='h', embed='timeF',
        enc_in=7, dec_in=7, c_out=7, num_workers=2, batch_size=1,
        exp_name='MTSF', ordered_data=False, data_amount=-1,
        combine_Gaussian_datasets=False, synthetic_data_path='', synthetic_root_path='./',
        synthetic_length=1024, stride=-1,
    )
    _, test_dl = data_provider(args, 'test')
    preds, trues = [], []
    with torch.no_grad():
        for bx, by, _, _ in test_dl:
            bx = bx.float().to(device)
            out = model(bx, None, None, None, target_pred_len=96)
            if isinstance(out, tuple): out = out[0]
            preds.append(out.cpu().numpy()); trues.append(by[:, -96:, :].numpy())
    p, t = np.concatenate(preds), np.concatenate(trues)
    results['FC_ETTh1'] = np.mean((p - t) ** 2)

    # Imputation: ETTh1 m=0.125
    torch.manual_seed(2021)
    preds, trues, masks = [], [], []
    with torch.no_grad():
        for bx, by, _, _ in test_dl:
            bx = bx.float().to(device)
            mask = (torch.rand_like(bx) > 0.125).float()
            out = model.reconstruct(bx * mask)
            preds.append(out.cpu().numpy()); trues.append(bx.cpu().numpy()); masks.append(mask.cpu().numpy())
    p, t, m = np.concatenate(preds), np.concatenate(trues), np.concatenate(masks)
    results['IMP_m0125'] = np.mean((p[m == 0] - t[m == 0]) ** 2)

    # Classification: Epilepsy
    try:
        cls_root = './dataset/classification/Multivariate_ts'
        train_ds = Dataset_Classification(root_path=cls_root, flag='train', size=[96, 0, 96], data_path='Epilepsy')
        test_ds = Dataset_Classification(root_path=cls_root, flag='test', size=[96, 0, 96], data_path='Epilepsy')
        train_dl = DataLoader(train_ds, batch_size=16, shuffle=True, drop_last=True)
        test_dl_cls = DataLoader(test_ds, batch_size=16, shuffle=False)
        hidden = model.branch_hidden
        cls_head = nn.Sequential(nn.Linear(hidden, 256), nn.GELU(), nn.Dropout(0.1),
                                 nn.Linear(256, train_ds.n_classes)).to(device)
        opt = optim.Adam(cls_head.parameters(), lr=0.001)
        for p_m in model.parameters(): p_m.requires_grad = False
        for ep in range(10):
            cls_head.train()
            for bx, label, _, _ in train_dl:
                bx = bx.float().to(device); label = label.long().to(device)
                with torch.no_grad(): z = model.get_representation(bx).mean(dim=1)
                loss = nn.CrossEntropyLoss()(cls_head(z), label)
                opt.zero_grad(); loss.backward(); opt.step()
        cls_head.eval()
        ps, ls = [], []
        with torch.no_grad():
            for bx, label, _, _ in test_dl_cls:
                bx = bx.float().to(device)
                z = model.get_representation(bx).mean(dim=1)
                ps.append(cls_head(z).argmax(-1).cpu().numpy()); ls.append(label.numpy())
        results['CLS_Epilepsy'] = accuracy_score(np.concatenate(ls), np.concatenate(ps))
        for p_m in model.parameters(): p_m.requires_grad = True
    except:
        results['CLS_Epilepsy'] = 0.0

    model.train()
    return results


def pretrain(model, device, save_path, epochs=10, lr=0.0005, mask_rate=0.4):
    print(f'\n{"="*60}')
    print('Real + Synthetic Combined Pre-training')
    print(f'  Real: Time Series Pile')
    print(f'  Synth: TempoPFN (step, spike, sawtooth, ...)')
    print(f'  Loss: next-token + masked_recon')
    print(f'  lr={lr}, batch=256')
    print(f'{"="*60}')

    # Real data
    real_ds = PilePretrainDataset(seq_len=96, stride=48,
                                   pile_root='./dataset/time_series_pile')
    print(f'  Real: {len(real_ds)} windows')

    # Synthetic data
    synth_ds = SyntheticWindowDataset('tempopfn_15k_1024.arrow',
                                      seq_len=96, stride=48, max_samples=100000)

    # Combine
    combined = ConcatDataset([real_ds, synth_ds])
    n_val = min(10000, len(combined) // 10)
    n_train = len(combined) - n_val
    train_ds, val_ds = random_split(combined, [n_train, n_val])

    train_dl = DataLoader(train_ds, batch_size=256, shuffle=True,
                          num_workers=4, drop_last=True, pin_memory=True)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = epochs * len(train_dl)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    recon_criterion = nn.MSELoss(reduction='none')

    print(f'  Combined: {len(combined)} total, Train: {n_train}, Steps/epoch: {len(train_dl)}')

    best_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        losses, nt_losses, recon_losses = [], [], []
        t0 = time.time()

        for i, batch_x in enumerate(train_dl):
            batch_x = batch_x.float().to(device)
            B, S, C = batch_x.shape
            optimizer.zero_grad()

            # 1) Masked recon
            mask = (torch.rand_like(batch_x) > mask_rate).float()
            recon_out = model.reconstruct(batch_x * mask)
            loss_mat = recon_criterion(recon_out, batch_x)
            inv_mask = 1.0 - mask
            recon_loss = (loss_mat * inv_mask).sum() / inv_mask.sum().clamp(min=1)

            # 2) Next-token
            split = torch.randint(24, 72, (1,)).item()
            context = batch_x[:, :split, :]
            target = batch_x[:, split:, :]
            target_len = S - split
            context_padded = F.pad(context, (0, 0, 0, S - split))
            nt_out = model(context_padded, None, None, None, target_pred_len=target_len)
            if isinstance(nt_out, tuple): nt_out = nt_out[0]
            nt_loss = F.mse_loss(nt_out, target)

            loss = nt_loss + recon_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            losses.append(loss.item())
            nt_losses.append(nt_loss.item())
            recon_losses.append(recon_loss.item())

            if (i + 1) % 200 == 0:
                print(f'  iter {i+1}/{len(train_dl)}: nt={np.mean(nt_losses[-200:]):.4f} recon={np.mean(recon_losses[-200:]):.4f}')

        train_loss = np.mean(losses)
        print(f'Epoch {epoch+1}/{epochs}: loss={train_loss:.4f} (nt={np.mean(nt_losses):.4f} recon={np.mean(recon_losses):.4f}) ({time.time()-t0:.0f}s)')

        if train_loss < best_loss:
            best_loss = train_loss
            torch.save(model.state_dict(), save_path)
            print(f'  Saved checkpoint')

        # 매 epoch quick eval
        print(f'  --- Quick Eval (epoch {epoch+1}) ---')
        qr = quick_eval(model, device, epoch + 1)
        print(f'  FC_ETTh1={qr["FC_ETTh1"]:.4f}  IMP={qr["IMP_m0125"]:.4f}  CLS_Epi={qr["CLS_Epilepsy"]:.4f}')

    model.load_state_dict(torch.load(save_path))
    return model


if __name__ == '__main__':
    from experiments.eval_all_tasks import eval_forecasting, eval_imputation, eval_classification, eval_short_term, print_summary

    device = torch.device('cuda')

    # 검증된 세팅: MoE 4 + spectral
    args = SimpleNamespace(
        seq_len=96, pred_len=96, use_norm=True,
        deeponet_width=128, n_experts=4, branch_depth=4, trunk_depth=2,
        activation='gelu', dropout=0.1, branch_hidden=512,
        spectral_branch=True,
        skip_mode='none', use_cross_channel=False, trunk_basis='mixed',
        encoder_type='patch_attn', loss='MSE',
    )
    model = Model(args).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'Model: {n_params/1e6:.1f}M (MoE 4, spectral ON)')

    save_path = 'checkpoints/real_synth_combined.pth'
    os.makedirs('checkpoints', exist_ok=True)

    model = pretrain(model, device, save_path, epochs=10, lr=0.0005)

    # Full eval
    fc = eval_forecasting(model, device)
    st = eval_short_term(model, device)
    imp = eval_imputation(model, device)
    cls = eval_classification(model, device)
    print_summary(fc, imp, cls, st, f'Real+Synth ({n_params/1e6:.1f}M)')
