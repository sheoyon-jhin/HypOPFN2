"""
Zero-shot imputation eval (FeDaL/MOMENT protocol).

Mask rates: 12.5%, 25%, 37.5%, 50% (standard TSFM benchmark)
Datasets: ETTh1, ETTh2, ETTm1, ETTm2, Weather
Metric: MSE, MAE on masked positions

Usage:
  CUDA_VISIBLE_DEVICES=0 python experiments/eval_imputation.py \
      --ckpt checkpoints/overnight_seq720_s50.pth --seq_len 720 \
      --tag overnight_seq720_imputation
"""
import sys, os, json, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn.functional as F
import numpy as np
from types import SimpleNamespace

from experiments.exp_v1_varlen_ext import OperatorModelVarLen

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# FeDaL reference (Table 5, imputation zero-shot)
REFERENCE = {
    'ETTh1':   {'MSE': 0.149, 'MAE': 0.253},
    'ETTh2':   {'MSE': 0.092, 'MAE': 0.199},
    'ETTm1':   {'MSE': 0.083, 'MAE': 0.189},
    'ETTm2':   {'MSE': 0.057, 'MAE': 0.148},
    'Weather': {'MSE': 0.030, 'MAE': 0.057},
}

MASK_RATES = [0.125, 0.25, 0.375, 0.5]
DATASETS = {
    'ETTh1':   ('ETTh1',  './dataset/ETT-small/', 'ETTh1.csv', 7),
    'ETTh2':   ('ETTh2',  './dataset/ETT-small/', 'ETTh2.csv', 7),
    'ETTm1':   ('ETTm1',  './dataset/ETT-small/', 'ETTm1.csv', 7),
    'ETTm2':   ('ETTm2',  './dataset/ETT-small/', 'ETTm2.csv', 7),
    'Weather': ('custom', './dataset/weather/',  'weather.csv', 21),
}


def eval_imputation(model, seq_len, mask_rate, dataset_name, d_type, root, fname, enc_in, n_batches=50):
    """Zero-shot imputation: random mask input, predict masked values."""
    from data_provider.data_factory import data_provider

    a = SimpleNamespace(seq_len=seq_len, pred_len=0, label_len=0, data=d_type,
                        root_path=root, data_path=fname, features='M', target='OT', freq='h',
                        embed='timeF', enc_in=enc_in, dec_in=enc_in, c_out=enc_in,
                        num_workers=2, batch_size=32, exp_name='MTSF', ordered_data=False,
                        data_amount=-1, combine_Gaussian_datasets=False, synthetic_data_path='',
                        synthetic_root_path='./', synthetic_length=1024, stride=-1)
    _, tdl = data_provider(a, 'test')
    np.random.seed(42)
    torch.manual_seed(42)

    all_mse, all_mae = [], []
    model.eval()
    with torch.no_grad():
        for batch_idx, (bx, _, _, _) in enumerate(tdl):
            if batch_idx >= n_batches:
                break
            bx = bx.float().to(DEVICE)
            B, S, C = bx.shape

            for ch in range(C):
                x_ch = bx[:, :, ch]
                if S >= seq_len:
                    x = x_ch[:, -seq_len:]
                else:
                    x = F.pad(x_ch, (seq_len - S, 0))

                # Normalize
                m = x.mean(1, keepdim=True)
                s = x.std(1, keepdim=True).clamp(min=1e-6)
                x_n = ((x - m) / s).clamp(-10, 10)

                # Random mask (same per-batch)
                mask = (torch.rand(B, seq_len, device=DEVICE) > mask_rate).float()
                x_masked = x_n * mask

                # Query points = masked positions
                # Build t ∈ [0, 1] for masked indices
                for b_idx in range(B):
                    mi = torch.where(mask[b_idx] == 0)[0]
                    if len(mi) == 0:
                        continue
                    qt = mi.float().unsqueeze(0) / seq_len  # t ∈ [0, 1] for imputation
                    ctx = x_masked[b_idx].unsqueeze(0)
                    pred = model.forward_train(ctx, qt)  # (1, n_masked)
                    true = x_n[b_idx][mi].unsqueeze(0)
                    # Denormalize
                    pred_d = pred * s[b_idx] + m[b_idx]
                    true_d = true * s[b_idx] + m[b_idx]
                    all_mse.append(((pred_d - true_d) ** 2).mean().item())
                    all_mae.append((pred_d - true_d).abs().mean().item())

    return np.mean(all_mse), np.mean(all_mae)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--decomp', type=int, default=0)
    p.add_argument('--seq_len', type=int, default=512)
    p.add_argument('--use_nll', type=int, default=0)
    p.add_argument('--tag', type=str, required=True)
    p.add_argument('--n_batches', type=int, default=30)
    p.add_argument('--d_model', type=int, default=512)
    p.add_argument('--n_layers', type=int, default=6)
    p.add_argument('--trunk_w', type=int, default=192)
    p.add_argument('--multi_scale_fourier', type=int, default=0)
    p.add_argument('--multi_scale_iq', type=int, default=0)
    p.add_argument('--hybrid_trunk', type=int, default=0)
    p.add_argument('--fourier_nf', type=int, default=32)
    p.add_argument('--highfreq_nf', type=int, default=0)
    p.add_argument('--all_fixed', type=int, default=0)
    p.add_argument('--pool_type', type=str, default='mean')
    p.add_argument('--model_type', type=str, default='varlen', choices=['varlen', 'decomp'])
    p.add_argument('--decomp_kernels', type=str, default='49,25,7')
    args = p.parse_args()

    print('=' * 70)
    print(f'IMPUTATION EVAL: {args.ckpt}')
    print(f'  seq_len={args.seq_len}, mask_rates={MASK_RATES}')
    print('=' * 70)

    if args.model_type == 'decomp':
        from experiments.exp_v1_varlen_ext import OperatorModelDecomp
        decomp_k = tuple(int(x) for x in args.decomp_kernels.split(','))
        model = OperatorModelDecomp(
            max_seq_len=args.seq_len,
            d_model=args.d_model, n_layers=args.n_layers, trunk_w=args.trunk_w,
            fourier_nf=args.fourier_nf, pool_type=args.pool_type,
            highfreq_nf=args.highfreq_nf, all_fixed=bool(args.all_fixed),
            decomp_kernels=decomp_k,
        ).to(DEVICE)
    else:
        model = OperatorModelVarLen(
            max_seq_len=args.seq_len, use_nll=bool(args.use_nll),
            d_model=args.d_model, n_layers=args.n_layers,
            trunk_w=args.trunk_w,
            multi_scale_fourier=bool(args.multi_scale_fourier),
            multi_scale_iq=bool(args.multi_scale_iq),
            hybrid_trunk=bool(args.hybrid_trunk),
            fourier_nf=args.fourier_nf,
            highfreq_nf=args.highfreq_nf,
            all_fixed=bool(args.all_fixed),
            pool_type=args.pool_type,
        ).to(DEVICE)
    state = torch.load(args.ckpt, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    results = {}
    for dn, (d_type, root, fname, enc_in) in DATASETS.items():
        print(f'\n--- {dn} ---')
        dataset_results = {}
        for mr in MASK_RATES:
            try:
                mse, mae = eval_imputation(model, args.seq_len, mr, dn,
                                           d_type, root, fname, enc_in,
                                           n_batches=args.n_batches)
                print(f'  mask={mr*100:.1f}%: MSE={mse:.4f}  MAE={mae:.4f}')
                dataset_results[f'mask_{int(mr*1000)}'] = {'MSE': float(mse), 'MAE': float(mae)}
            except Exception as e:
                print(f'  mask={mr*100:.1f}%: ERROR ({e})')
        # Average across mask rates
        if dataset_results:
            avg_mse = np.mean([v['MSE'] for v in dataset_results.values()])
            avg_mae = np.mean([v['MAE'] for v in dataset_results.values()])
            print(f'  avg: MSE={avg_mse:.4f} MAE={avg_mae:.4f}')
            dataset_results['avg'] = {'MSE': float(avg_mse), 'MAE': float(avg_mae)}
        results[dn] = dataset_results

    # Paper comparison
    print('\n' + '=' * 70)
    print(f'{"Dataset":<10} {"Ours (MSE/MAE)":<20} {"FeDaL (MSE/MAE)":<20} {"ΔMSE":>8}')
    print('-' * 70)
    for dn, ref in REFERENCE.items():
        ours = results.get(dn, {}).get('avg')
        if ours:
            gap = (ours['MSE'] - ref['MSE']) / ref['MSE'] * 100
            print(f'{dn:<10} {ours["MSE"]:.4f} / {ours["MAE"]:.4f}   '
                  f'{ref["MSE"]:.3f} / {ref["MAE"]:.3f}        {gap:>+6.1f}%')

    # Overall avg
    ov_mse = np.mean([results[d]['avg']['MSE'] for d in results if 'avg' in results[d]])
    ov_mae = np.mean([results[d]['avg']['MAE'] for d in results if 'avg' in results[d]])
    print('-' * 70)
    print(f'{"OVERALL":<10} {ov_mse:.4f} / {ov_mae:.4f}')
    results['overall_avg'] = {'MSE': float(ov_mse), 'MAE': float(ov_mae)}

    os.makedirs('results', exist_ok=True)
    with open(f'results/{args.tag}.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved: results/{args.tag}.json')


if __name__ == '__main__':
    main()
