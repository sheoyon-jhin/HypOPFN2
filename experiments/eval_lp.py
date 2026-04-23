"""
Linear Probing (LP) evaluation for OperatorModel.

Protocol:
  - Freeze pretrained encoder
  - For each (dataset, pred_len):
    * Extract z from encoder for train split (channel-independent)
    * Train a LINEAR head: z → [pred_len]
    * Eval on test split (MSE + MAE)
  - Compare with ZS and FeDaL LP/FT baselines

Usage:
  CUDA_VISIBLE_DEVICES=0 python experiments/eval_lp.py \
      --ckpt checkpoints/overnight_seq720_s50.pth --seq_len 720 \
      --tag lp_seq720_overnight
"""
import sys, os, json, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from torch import optim
from types import SimpleNamespace

from experiments.exp_lotsa_scaling import OperatorModel

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATASETS = {
    'ETTh1':   ('ETTh1',  './dataset/ETT-small/', 'ETTh1.csv', 7),
    'ETTh2':   ('ETTh2',  './dataset/ETT-small/', 'ETTh2.csv', 7),
    'ETTm1':   ('ETTm1',  './dataset/ETT-small/', 'ETTm1.csv', 7),
    'ETTm2':   ('ETTm2',  './dataset/ETT-small/', 'ETTm2.csv', 7),
    'Weather': ('custom', './dataset/weather/',  'weather.csv', 21),
}

# FeDaL Full-shot reference
FEDAL_FT = {
    'ETTh1': 0.380, 'ETTh2': 0.334, 'ETTm1': 0.319,
    'ETTm2': 0.261, 'Weather': 0.213,
}
# MOMENT LP reference (from HANDOVER)
MOMENT_LP = {
    'ETTh1': 0.412, 'ETTh2': 0.340, 'ETTm1': 0.333,
    'ETTm2': 0.254, 'Weather': 0.226,
}


@torch.no_grad()
def extract_z_channel(model, x, seq_len):
    """x: (B, S, C) → z_per_channel: (B, C, d_model)"""
    B, S, C = x.shape
    if S < seq_len:
        x = F.pad(x.transpose(1, 2), (seq_len - S, 0)).transpose(1, 2)
    else:
        x = x[:, -seq_len:]
    # Process each channel independently
    zs = []
    for ch in range(C):
        x_ch = x[:, :, ch].float()  # (B, seq_len)
        m = x_ch.mean(1, keepdim=True)
        s = x_ch.std(1, keepdim=True).clamp(min=1e-6)
        x_n = ((x_ch - m) / s).clamp(-10, 10)
        z = model.encoder(x_n)  # (B, d_model)
        zs.append({'z': z, 'mean': m, 'std': s})  # keep (B, 1) for broadcasting
    return zs


def train_linear_head(model, train_dl, head, pred_len, seq_len, n_epochs=20, lr=1e-3):
    opt = optim.Adam(head.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    model.eval()
    for ep in range(n_epochs):
        total_loss = 0; n_batches = 0
        for bx, by, _, _ in train_dl:
            bx = bx.float().to(DEVICE)
            by = by.float().to(DEVICE)
            B, S, C = bx.shape
            # Extract z per channel + store normalization
            zs = extract_z_channel(model, bx, seq_len)
            losses = []
            for ch in range(C):
                z = zs[ch]['z']
                m = zs[ch]['mean']
                s = zs[ch]['std']
                pred_norm = head(z)  # (B, pred_len)
                true = by[:, -pred_len:, ch]  # (B, pred_len)
                true_norm = ((true - m) / s).clamp(-10, 10)  # prevent explosion
                losses.append(F.mse_loss(pred_norm, true_norm))
            loss = sum(losses) / C
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item(); n_batches += 1
        scheduler.step()
        avg = total_loss / max(n_batches, 1)
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f'    LP epoch {ep+1}/{n_epochs}: loss={avg:.4f}')
    return head


@torch.no_grad()
def eval_linear_head(model, test_dl, head, pred_len, seq_len):
    model.eval(); head.eval()
    all_mse, all_mae = [], []
    for bx, by, _, _ in test_dl:
        bx = bx.float().to(DEVICE)
        by = by.float().to(DEVICE)
        B, S, C = bx.shape
        zs = extract_z_channel(model, bx, seq_len)
        for ch in range(C):
            z = zs[ch]['z']
            m = zs[ch]['mean']; s = zs[ch]['std']
            pred_norm = head(z)
            pred = pred_norm * s + m
            true = by[:, -pred_len:, ch]
            all_mse.append(((pred - true) ** 2).mean().item())
            all_mae.append((pred - true).abs().mean().item())
    return np.mean(all_mse), np.mean(all_mae)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--decomp', type=int, default=0)
    p.add_argument('--seq_len', type=int, default=720)
    p.add_argument('--lp_epochs', type=int, default=20)
    p.add_argument('--tag', type=str, required=True)
    args = p.parse_args()

    from data_provider.data_factory import data_provider

    print('=' * 70)
    print(f'LINEAR PROBE: {args.ckpt}')
    print(f'  seq_len={args.seq_len}, lp_epochs={args.lp_epochs}')
    print('=' * 70)

    model = OperatorModel(seq_len=args.seq_len, use_latent_decomp=bool(args.decomp)).to(DEVICE)
    state = torch.load(args.ckpt, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    # Freeze encoder
    for p_ in model.parameters():
        p_.requires_grad = False

    d_model = model.encoder.pos_emb.shape[-1]  # V1 PatchAttnEncoder uses .transformer, pos_emb has d_model

    results = {}
    for dn, (d, root, f, enc_in) in DATASETS.items():
        for pl in [96, 192, 336, 720]:
            print(f'\n--- {dn} pred_len={pl} ---')
            try:
                a = SimpleNamespace(seq_len=args.seq_len, pred_len=pl, label_len=48, data=d,
                    root_path=root, data_path=f, features='M', target='OT', freq='h',
                    embed='timeF', enc_in=enc_in, dec_in=enc_in, c_out=enc_in,
                    num_workers=2, batch_size=32, exp_name='MTSF', ordered_data=False,
                    data_amount=-1, combine_Gaussian_datasets=False, synthetic_data_path='',
                    synthetic_root_path='./', synthetic_length=1024, stride=-1)
                _, train_dl = data_provider(a, 'train')
                _, test_dl = data_provider(a, 'test')

                # Linear head: d_model → pred_len
                head = nn.Linear(d_model, pl).to(DEVICE)
                nn.init.xavier_normal_(head.weight, gain=0.1)
                nn.init.zeros_(head.bias)

                print(f'  training linear head...')
                head = train_linear_head(model, train_dl, head, pl, args.seq_len,
                                         n_epochs=args.lp_epochs)

                mse, mae = eval_linear_head(model, test_dl, head, pl, args.seq_len)
                print(f'  LP eval: MSE={mse:.4f}  MAE={mae:.4f}')
                results[f'{dn}_{pl}'] = {'MSE': float(mse), 'MAE': float(mae)}
            except Exception as e:
                print(f'  ERROR: {e}')

    # Average per dataset + overall
    print('\n' + '=' * 70)
    print(f'{"Dataset":<10} {"Ours LP":<12} {"FeDaL FT":<10} {"MOMENT LP":<12}')
    print('-' * 70)
    ds_avgs = {}
    for dn in DATASETS:
        entries = [results[f'{dn}_{pl}'] for pl in [96, 192, 336, 720] if f'{dn}_{pl}' in results]
        if entries:
            avg_mse = np.mean([e['MSE'] for e in entries])
            avg_mae = np.mean([e['MAE'] for e in entries])
            ds_avgs[dn] = {'MSE': float(avg_mse), 'MAE': float(avg_mae)}
            results[f'{dn}_avg'] = ds_avgs[dn]
            print(f'{dn:<10} {avg_mse:.3f}/{avg_mae:.3f}  '
                  f'{FEDAL_FT.get(dn, 0):.3f}      {MOMENT_LP.get(dn, 0):.3f}')
    if ds_avgs:
        ov_mse = np.mean([v['MSE'] for v in ds_avgs.values()])
        ov_mae = np.mean([v['MAE'] for v in ds_avgs.values()])
        print('-' * 70)
        print(f'{"OVERALL":<10} {ov_mse:.3f}/{ov_mae:.3f}')
        results['overall_avg'] = {'MSE': float(ov_mse), 'MAE': float(ov_mae)}

    os.makedirs('results', exist_ok=True)
    with open(f'results/{args.tag}.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved: results/{args.tag}.json')


if __name__ == '__main__':
    main()
