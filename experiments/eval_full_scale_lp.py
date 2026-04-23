"""
Full Scale 50M Linear Probe for Forecasting

Protocol:
  - Freeze the 32.5M operator's encoder
  - For each (dataset, pred_len):
    - Extract encoder representations z for train split (channel-independent)
    - Train a linear layer z → [pred_len] on train split
    - Evaluate on test split (MSE + MAE)
  - Compare to MOMENT-LP baseline

CUDA_VISIBLE_DEVICES=1 python experiments/eval_full_scale_lp.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from torch import optim
from types import SimpleNamespace

from data_provider.data_factory import data_provider
from experiments.exp_full_scale_train import FullScaleModel, SEQ_LEN, HIDDEN

DEVICE = torch.device('cuda')
CKPT = 'checkpoints/full_scale_run.pth'


# ============================================================
# Feature extraction from the frozen model
# ============================================================
@torch.no_grad()
def extract_channel_features(model, x_enc):
    """
    x_enc: [B, S, C] → z: [B*C, HIDDEN]
    Channel-independent: each channel processed separately.
    Per-channel normalization (since model is RevIN OFF).
    """
    B, S, C = x_enc.shape
    zs = []
    means = []
    stds = []
    for ch in range(C):
        x_ch = x_enc[:, :, ch]  # [B, S]
        m = x_ch.mean(dim=1, keepdim=True)
        s = x_ch.std(dim=1, keepdim=True).clamp(min=1e-6)
        x_n = ((x_ch - m) / s).clamp(-10, 10)
        if S > SEQ_LEN:
            x_n = x_n[:, -SEQ_LEN:]
        elif S < SEQ_LEN:
            x_n = F.pad(x_n, (SEQ_LEN - S, 0))
        z = model.encoder(x_n)  # [B, HIDDEN]
        zs.append(z)
        means.append(m)
        stds.append(s)
    z_all = torch.stack(zs, dim=1)        # [B, C, HIDDEN]
    m_all = torch.cat(means, dim=1)       # [B, C]
    s_all = torch.cat(stds, dim=1)        # [B, C]
    return z_all, m_all, s_all


# ============================================================
# Linear probe train + eval
# ============================================================
def linear_probe(model, dl_train, dl_test, pred_len, n_epochs=30, lr=1e-3,
                  weight_decay=0.01):
    """Train a linear layer on top of frozen encoder."""
    head = nn.Linear(HIDDEN, pred_len).to(DEVICE)
    nn.init.xavier_normal_(head.weight, gain=0.1)
    nn.init.zeros_(head.bias)

    opt = optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)

    for ep in range(n_epochs):
        head.train()
        losses = []
        for bx, by, _, _ in dl_train:
            bx = bx.float().to(DEVICE)
            by = by.float().to(DEVICE)[:, -pred_len:, :]   # [B, pred_len, C]
            B, S, C = bx.shape
            z_all, m_all, s_all = extract_channel_features(model, bx)

            # Predict per channel in normalized space, then denormalize
            pred_norm = head(z_all)         # [B, C, pred_len]
            pred = pred_norm * s_all.unsqueeze(-1) + m_all.unsqueeze(-1)  # [B, C, pred_len]
            pred = pred.transpose(1, 2)     # [B, pred_len, C]

            loss = F.mse_loss(pred, by)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
        sched.step()

    # Eval
    head.eval()
    all_preds, all_tgts = [], []
    with torch.no_grad():
        for bx, by, _, _ in dl_test:
            bx = bx.float().to(DEVICE)
            by = by.float().to(DEVICE)[:, -pred_len:, :]
            B, S, C = bx.shape
            z_all, m_all, s_all = extract_channel_features(model, bx)
            pred_norm = head(z_all)
            pred = pred_norm * s_all.unsqueeze(-1) + m_all.unsqueeze(-1)
            pred = pred.transpose(1, 2)
            all_preds.append(pred.cpu().numpy())
            all_tgts.append(by.cpu().numpy())
    p = np.concatenate(all_preds)
    t = np.concatenate(all_tgts)
    mse = np.mean((p - t)**2)
    mae = np.mean(np.abs(p - t))
    return mse, mae


def eval_forecast_lp(model):
    print(f'\n{"="*60}\nForecasting Linear Probe\n{"="*60}')
    datasets = {
        'ETTh1':    ('ETTh1',  './dataset/ETT-small/', 'ETTh1.csv', 7),
        'ETTh2':    ('ETTh2',  './dataset/ETT-small/', 'ETTh2.csv', 7),
        'ETTm1':    ('ETTm1',  './dataset/ETT-small/', 'ETTm1.csv', 7),
        'ETTm2':    ('ETTm2',  './dataset/ETT-small/', 'ETTm2.csv', 7),
        'Weather':  ('custom', './dataset/weather/',   'weather.csv', 21),
        'Exchange': ('custom', './dataset/exchange_rate/', 'exchange_rate.csv', 8),
    }
    moment_lp = {
        'ETTh1_96':0.387,'ETTh1_192':0.410,'ETTh1_336':0.422,'ETTh1_720':0.454,
        'ETTh2_96':0.288,'ETTh2_192':0.349,'ETTh2_336':0.369,'ETTh2_720':0.403,
        'ETTm1_96':0.293,'ETTm1_192':0.326,'ETTm1_336':0.352,'ETTm1_720':0.405,
        'ETTm2_96':0.170,'ETTm2_192':0.227,'ETTm2_336':0.275,'ETTm2_720':0.363,
        'Weather_96':0.154,'Weather_192':0.197,'Weather_336':0.246,'Weather_720':0.315,
    }

    # Zero-shot results from previous eval (for comparison)
    zs = {
        'ETTh1_96':0.494,'ETTh1_192':0.559,'ETTh1_336':0.633,'ETTh1_720':0.652,
        'ETTh2_96':0.342,'ETTh2_192':0.422,'ETTh2_336':0.452,'ETTh2_720':0.450,
        'ETTm1_96':0.572,'ETTm1_192':0.641,'ETTm1_336':0.691,'ETTm1_720':0.732,
        'ETTm2_96':0.233,'ETTm2_192':0.289,'ETTm2_336':0.342,'ETTm2_720':0.436,
        'Weather_96':0.225,'Weather_192':0.278,'Weather_336':0.325,'Weather_720':0.387,
    }

    model.eval()
    all_results = {}

    for dn, (d, root, f, enc_in) in datasets.items():
        for pl in [96, 192, 336, 720]:
            try:
                a = SimpleNamespace(seq_len=96, pred_len=pl, label_len=48, data=d,
                    root_path=root, data_path=f, features='M', target='OT',
                    freq='h', embed='timeF', enc_in=enc_in, dec_in=enc_in, c_out=enc_in,
                    num_workers=2, batch_size=64, exp_name='MTSF', ordered_data=False,
                    data_amount=-1, combine_Gaussian_datasets=False,
                    synthetic_data_path='', synthetic_root_path='./',
                    synthetic_length=1024, stride=-1)
                _, dl_train = data_provider(a, 'train')
                _, dl_test = data_provider(a, 'test')

                mse, mae = linear_probe(model, dl_train, dl_test, pl)

                k = f'{dn}_{pl}'
                m_lp = moment_lp.get(k)
                z_mse = zs.get(k)
                gap_lp = f'{(mse/m_lp-1)*100:+.0f}%' if m_lp else '-'
                improvement = f'{(z_mse - mse)/z_mse*100:+.0f}%' if z_mse else '-'
                m_str = f'{m_lp:.3f}' if m_lp else 'N/A'
                print(f'  {k:<14}: MSE={mse:.4f} MAE={mae:.4f}  | ZS={z_mse if z_mse else "-"} '
                      f'→ LP={mse:.4f} ({improvement}) | MOMENT-LP={m_str} gap={gap_lp}')
                all_results[k] = mse
            except Exception as e:
                print(f'  {dn}_{pl}: ERROR ({e})')
    return all_results


if __name__ == '__main__':
    print('='*60)
    print('Full Scale 50M Linear Probe Evaluation (Forecasting)')
    print('='*60)
    print(f'Loading checkpoint: {CKPT}')
    model = FullScaleModel().to(DEVICE)
    state = torch.load(CKPT, map_location=DEVICE)
    model.load_state_dict(state)
    n = sum(p.numel() for p in model.parameters())
    print(f'Model: {n/1e6:.1f}M params')

    # Freeze model
    for p in model.parameters():
        p.requires_grad = False

    fc_res = eval_forecast_lp(model)

    print('\n' + '='*60)
    print('SUMMARY — Linear Probe vs MOMENT-LP')
    print('='*60)
    for k in ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'Weather', 'Exchange']:
        avgs = [v for k_, v in fc_res.items() if k_.startswith(k+'_')]
        if avgs:
            avg = np.mean(avgs)
            print(f'  {k:<10}: Avg LP MSE = {avg:.4f}')
    print('='*60)
