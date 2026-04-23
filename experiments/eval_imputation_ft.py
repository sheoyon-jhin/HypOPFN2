"""
Fine-Tune 1-epoch Imputation on ETT/Weather (FeDaL protocol).
Updates ENCODER + TRUNKS + HEAD for 1 epoch on train split,
then evaluates on test at standard mask rates.

Usage:
  python experiments/eval_imputation_ft.py \
    --ckpt checkpoints/hyper4_10pct_full231B.pth \
    --highfreq_nf 256 --tag hyper4_10pct_impute_ft
"""
import sys, os, argparse, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import optim
from types import SimpleNamespace

from experiments.exp_v1_varlen_ext import OperatorModelVarLen, OperatorModelDecomp
from data_provider.data_factory import data_provider

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATASETS = {
    'ETTh1':   ('ETTh1',  './dataset/ETT-small/', 'ETTh1.csv', 7),
    'ETTh2':   ('ETTh2',  './dataset/ETT-small/', 'ETTh2.csv', 7),
    'ETTm1':   ('ETTm1',  './dataset/ETT-small/', 'ETTm1.csv', 7),
    'ETTm2':   ('ETTm2',  './dataset/ETT-small/', 'ETTm2.csv', 7),
    'Weather': ('custom', './dataset/weather/',  'weather.csv', 21),
}
MASK_RATES = [0.125, 0.25, 0.375, 0.5]


def build_model(args):
    if args.model_type == 'decomp':
        decomp_k = tuple(int(x) for x in args.decomp_kernels.split(','))
        m = OperatorModelDecomp(
            max_seq_len=args.seq_len, d_model=args.d_model, n_layers=args.n_layers,
            trunk_w=args.trunk_w, fourier_nf=args.fourier_nf,
            pool_type=args.pool_type, highfreq_nf=args.highfreq_nf,
            all_fixed=bool(args.all_fixed), decomp_kernels=decomp_k,
        )
    else:
        m = OperatorModelVarLen(
            max_seq_len=args.seq_len, d_model=args.d_model, n_layers=args.n_layers,
            trunk_w=args.trunk_w, hybrid_trunk=bool(args.hybrid_trunk),
            use_nll=bool(args.use_nll), fourier_nf=args.fourier_nf,
            multi_scale_fourier=bool(args.multi_scale_fourier),
            multi_scale_iq=bool(args.multi_scale_iq),
            pool_type=args.pool_type, highfreq_nf=args.highfreq_nf,
            all_fixed=bool(args.all_fixed),
        )
    return m


def make_loaders(dname, d_type, root, fname, enc_in, seq_len, batch_size):
    a = SimpleNamespace(seq_len=seq_len, pred_len=0, label_len=0, data=d_type,
                        root_path=root, data_path=fname, features='M', target='OT', freq='h',
                        embed='timeF', enc_in=enc_in, dec_in=enc_in, c_out=enc_in,
                        num_workers=2, batch_size=batch_size, exp_name='MTSF', ordered_data=False,
                        data_amount=-1, combine_Gaussian_datasets=False,
                        synthetic_data_path='', synthetic_root_path='./', synthetic_length=1024, stride=-1)
    train_ds, train_dl = data_provider(a, 'train')
    _, test_dl = data_provider(a, 'test')
    return train_dl, test_dl


def ft_one_imputation(model, train_dl, mask_rate, n_steps=500, lr=1e-5, head_lr=1e-4):
    """FT on imputation: freeze-safe differential LR.
    n_steps: max gradient updates (1-epoch worth)."""
    model.train()
    # Differential LR: encoder low, trunk/head higher
    enc_params = []
    other_params = []
    for n, p in model.named_parameters():
        if n.startswith('encoder'):
            enc_params.append(p)
        else:
            other_params.append(p)
    opt = optim.AdamW([
        {'params': enc_params, 'lr': lr},
        {'params': other_params, 'lr': head_lr},
    ], weight_decay=1e-4)

    step = 0
    for bx, by, _, _ in train_dl:
        if step >= n_steps: break
        bx = bx.float().to(DEVICE)
        B, S, C = bx.shape
        for ch in range(C):
            x_ch = bx[:, -S:, ch]
            m = x_ch.mean(-1, keepdim=True)
            s = x_ch.std(-1, keepdim=True).clamp(min=1e-6)
            x_n = ((x_ch - m) / s).clamp(-10, 10)
            # Random mask
            mask = (torch.rand(B, S, device=DEVICE) > mask_rate).float()
            x_masked = x_n * mask
            # For each sample pick a few masked positions as queries
            losses = []
            for b_idx in range(B):
                mi = torch.where(mask[b_idx] == 0)[0]
                if len(mi) == 0: continue
                # Sample up to 32 queries per sample
                if len(mi) > 32:
                    mi = mi[torch.randperm(len(mi), device=DEVICE)[:32]]
                qt = mi.float().unsqueeze(0) / S
                ctx = x_masked[b_idx].unsqueeze(0)
                pred = model.forward_train(ctx, qt)
                tgt = x_n[b_idx, mi].unsqueeze(0)
                losses.append(F.mse_loss(pred, tgt))
            if losses:
                loss = sum(losses) / len(losses)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                step += 1
                if step >= n_steps: break


@torch.no_grad()
def eval_imputation(model, test_dl, mask_rate, n_batches=30):
    model.eval()
    all_mse, all_mae = [], []
    for bi, (bx, by, _, _) in enumerate(test_dl):
        if bi >= n_batches: break
        bx = bx.float().to(DEVICE)
        B, S, C = bx.shape
        for ch in range(C):
            x_ch = bx[:, -S:, ch]
            m = x_ch.mean(-1, keepdim=True)
            s = x_ch.std(-1, keepdim=True).clamp(min=1e-6)
            x_n = ((x_ch - m) / s).clamp(-10, 10)
            mask = (torch.rand(B, S, device=DEVICE) > mask_rate).float()
            x_masked = x_n * mask
            for b_idx in range(B):
                mi = torch.where(mask[b_idx] == 0)[0]
                if len(mi) == 0: continue
                qt = mi.float().unsqueeze(0) / S
                ctx = x_masked[b_idx].unsqueeze(0)
                pred = model.forward_train(ctx, qt)
                tgt = x_n[b_idx, mi].unsqueeze(0)
                all_mse.append(F.mse_loss(pred, tgt).item())
                all_mae.append((pred - tgt).abs().mean().item())
    return float(np.mean(all_mse)) if all_mse else 0.0, float(np.mean(all_mae)) if all_mae else 0.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--tag', required=True)
    p.add_argument('--seq_len', type=int, default=720)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--ft_steps', type=int, default=500)
    p.add_argument('--lr', type=float, default=1e-5)
    p.add_argument('--model_type', type=str, default='varlen', choices=['varlen','decomp'])
    p.add_argument('--decomp_kernels', type=str, default='49,25,7')
    p.add_argument('--use_nll', type=int, default=0)
    p.add_argument('--hybrid_trunk', type=int, default=0)
    p.add_argument('--all_fixed', type=int, default=0)
    p.add_argument('--highfreq_nf', type=int, default=0)
    p.add_argument('--fourier_nf', type=int, default=32)
    p.add_argument('--multi_scale_fourier', type=int, default=0)
    p.add_argument('--multi_scale_iq', type=int, default=0)
    p.add_argument('--pool_type', type=str, default='mean')
    p.add_argument('--d_model', type=int, default=512)
    p.add_argument('--n_layers', type=int, default=6)
    p.add_argument('--trunk_w', type=int, default=192)
    args = p.parse_args()

    print('='*70)
    print(f'IMPUTATION FT (1-epoch): {args.ckpt}')
    print(f'  seq_len={args.seq_len}, ft_steps={args.ft_steps}, lr={args.lr}')
    print('='*70)

    state = torch.load(args.ckpt, map_location=DEVICE, weights_only=True)
    results = {}
    for dn, (d_type, root, fname, enc_in) in DATASETS.items():
        print(f'\n--- {dn} ---')
        ds_r = {}
        for mr in MASK_RATES:
            t0 = time.time()
            # Fresh model per (dataset, mask_rate) — standard FT protocol
            model = build_model(args).to(DEVICE)
            model.load_state_dict(state)
            train_dl, test_dl = make_loaders(dn, d_type, root, fname, enc_in, args.seq_len, args.batch_size)
            ft_one_imputation(model, train_dl, mr, n_steps=args.ft_steps, lr=args.lr)
            mse, mae = eval_imputation(model, test_dl, mr)
            ds_r[f'mask_{int(mr*1000)}'] = {'MSE': mse, 'MAE': mae}
            print(f'  mask={mr*100:.1f}%: MSE={mse:.4f}  MAE={mae:.4f}  ({time.time()-t0:.0f}s)')
        avg_mse = float(np.mean([ds_r[k]['MSE'] for k in ds_r]))
        avg_mae = float(np.mean([ds_r[k]['MAE'] for k in ds_r]))
        ds_r['avg'] = {'MSE': avg_mse, 'MAE': avg_mae}
        print(f'  avg: MSE={avg_mse:.4f}  MAE={avg_mae:.4f}')
        results[dn] = ds_r

    # Overall avg
    all_mse = [results[dn][k]['MSE'] for dn in DATASETS for k in ['mask_125','mask_250','mask_375','mask_500']]
    all_mae = [results[dn][k]['MAE'] for dn in DATASETS for k in ['mask_125','mask_250','mask_375','mask_500']]
    ov_mse = float(np.mean(all_mse))
    ov_mae = float(np.mean(all_mae))
    results['overall'] = {'MSE': ov_mse, 'MAE': ov_mae}
    print(f'\n{"="*70}')
    print(f'OVERALL: MSE={ov_mse:.4f}  MAE={ov_mae:.4f}')

    os.makedirs('results', exist_ok=True)
    with open(f'results/{args.tag}_impute_ft.json','w') as f:
        json.dump(results, f, indent=2)
    print(f'Saved: results/{args.tag}_impute_ft.json')


if __name__ == '__main__':
    main()
