"""
Prediction visualization — actual time series forecasts on ETT/Weather.

Shows: context + ground truth + prediction overlay.
Multiple samples per dataset + multiple pred_lens.

Usage:
  python experiments/viz_predictions.py \
      --ckpt checkpoints/v1_varlen_nll_failmode.pth \
      --use_nll 1 --max_seq_len 720 --tag predictions_best
"""
import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch, torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from types import SimpleNamespace

from experiments.exp_v1_varlen_ext import OperatorModelVarLen

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATASETS = {
    'ETTh1':   ('ETTh1',  './dataset/ETT-small/', 'ETTh1.csv', 7),
    'ETTh2':   ('ETTh2',  './dataset/ETT-small/', 'ETTh2.csv', 7),
    'ETTm1':   ('ETTm1',  './dataset/ETT-small/', 'ETTm1.csv', 7),
    'ETTm2':   ('ETTm2',  './dataset/ETT-small/', 'ETTm2.csv', 7),
    'Weather': ('custom', './dataset/weather/',  'weather.csv', 21),
}


@torch.no_grad()
def predict_one(model, x_ctx, horizon, max_seq_len):
    patch_size = model.encoder.patch_size
    eff = (x_ctx.shape[-1] // patch_size) * patch_size
    x_ctx = x_ctx[..., -eff:]
    m = x_ctx.mean(-1, keepdim=True)
    s = x_ctx.std(-1, keepdim=True).clamp(min=1e-6)
    x_n = ((x_ctx - m) / s).clamp(-10, 10)
    if x_n.dim() == 1:
        x_n = x_n.unsqueeze(0)
    pred = model.forecast(x_n, n=horizon, seq_len_ref=eff).squeeze(0)
    pred = pred * s.squeeze(-1) + m.squeeze(-1)
    return pred.cpu().numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--use_nll', type=int, default=0)
    p.add_argument('--hybrid_trunk', type=int, default=0)
    p.add_argument('--max_seq_len', type=int, default=720)
    p.add_argument('--d_model', type=int, default=512)
    p.add_argument('--n_layers', type=int, default=6)
    p.add_argument('--trunk_w', type=int, default=192)
    p.add_argument('--tag', type=str, required=True)
    p.add_argument('--horizon', type=int, default=192)
    p.add_argument('--samples_per_ds', type=int, default=2)
    p.add_argument('--all_fixed', type=int, default=0)
    p.add_argument('--highfreq_nf', type=int, default=0)
    p.add_argument('--fourier_nf', type=int, default=32)
    p.add_argument('--multi_scale_fourier', type=int, default=0)
    p.add_argument('--multi_scale_iq', type=int, default=0)
    p.add_argument('--pool_type', type=str, default='mean')
    p.add_argument('--model_type', type=str, default='varlen', choices=['varlen', 'decomp'])
    p.add_argument('--decomp_kernels', type=str, default='49,25,7')
    args = p.parse_args()

    from data_provider.data_factory import data_provider

    if args.model_type == 'decomp':
        from experiments.exp_v1_varlen_ext import OperatorModelDecomp
        decomp_k = tuple(int(x) for x in args.decomp_kernels.split(','))
        model = OperatorModelDecomp(
            max_seq_len=args.max_seq_len,
            d_model=args.d_model, n_layers=args.n_layers, trunk_w=args.trunk_w,
            fourier_nf=args.fourier_nf, pool_type=args.pool_type,
            highfreq_nf=args.highfreq_nf, all_fixed=bool(args.all_fixed),
            decomp_kernels=decomp_k,
        ).to(DEVICE)
    else:
        model = OperatorModelVarLen(
            max_seq_len=args.max_seq_len,
            d_model=args.d_model,
            n_layers=args.n_layers,
            trunk_w=args.trunk_w,
            hybrid_trunk=bool(args.hybrid_trunk),
            use_nll=bool(args.use_nll),
            fourier_nf=args.fourier_nf,
            multi_scale_fourier=bool(args.multi_scale_fourier),
            multi_scale_iq=bool(args.multi_scale_iq),
            pool_type=args.pool_type,
            highfreq_nf=args.highfreq_nf,
            all_fixed=bool(args.all_fixed),
        ).to(DEVICE)
    state = torch.load(args.ckpt, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    os.makedirs('results/figures', exist_ok=True)

    # 5 datasets × samples_per_ds samples → subplots
    n_samples_total = len(DATASETS) * args.samples_per_ds
    n_cols = 2
    n_rows = (n_samples_total + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 7, n_rows * 2.5))
    axes = axes.flatten() if n_samples_total > 1 else [axes]
    idx = 0

    for dn, (d, root, f, enc_in) in DATASETS.items():
        print(f'Loading {dn}...')
        a = SimpleNamespace(seq_len=args.max_seq_len, pred_len=args.horizon, label_len=48, data=d,
            root_path=root, data_path=f, features='M', target='OT', freq='h',
            embed='timeF', enc_in=enc_in, dec_in=enc_in, c_out=enc_in,
            num_workers=2, batch_size=32, exp_name='MTSF', ordered_data=False,
            data_amount=-1, combine_Gaussian_datasets=False, synthetic_data_path='',
            synthetic_root_path='./', synthetic_length=1024, stride=-1)
        _, tdl = data_provider(a, 'test')

        # Take samples at different batches for diversity
        stride = max(1, len(tdl) // (args.samples_per_ds + 2))
        collected = 0
        batch_i = 0
        for bx, by, _, _ in tdl:
            if collected >= args.samples_per_ds:
                break
            if batch_i % stride != 0:
                batch_i += 1
                continue
            ch = bx.shape[-1] - 1  # OT (last channel)
            b_idx = 0
            x_full = bx[b_idx, :, ch].float().numpy()
            # Truncate/pad to max_seq_len
            if len(x_full) > args.max_seq_len:
                ctx_raw = x_full[-args.max_seq_len:]
            else:
                ctx_raw = np.concatenate([np.zeros(args.max_seq_len - len(x_full)), x_full])
            truth = by[b_idx, -args.horizon:, ch].float().numpy()

            ctx_t = torch.tensor(ctx_raw, dtype=torch.float32, device=DEVICE)
            pred = predict_one(model, ctx_t, args.horizon, args.max_seq_len)
            mse = float(((pred - truth) ** 2).mean())

            ax = axes[idx]
            ctx_x = np.arange(len(ctx_raw))
            fut_x = np.arange(len(ctx_raw), len(ctx_raw) + len(truth))
            ax.plot(ctx_x, ctx_raw, color='steelblue', linewidth=0.8, label='context', alpha=0.7)
            ax.plot(fut_x, truth, color='black', linewidth=1.5, label='truth')
            ax.plot(fut_x, pred, color='red', linewidth=1.5, linestyle='--', label='prediction')
            ax.axvline(len(ctx_raw), color='gray', linestyle=':', alpha=0.5)
            ax.set_title(f'{dn} OT (sample {batch_i}, pred_len={args.horizon})  MSE={mse:.3f}',
                         fontsize=10)
            ax.legend(loc='upper left', fontsize=8)
            ax.grid(alpha=0.3)

            idx += 1; collected += 1
            batch_i += 1

    for j in range(idx, len(axes)):
        axes[j].axis('off')

    fig.suptitle(f'Forecast Predictions (ckpt: {os.path.basename(args.ckpt)}, horizon={args.horizon})',
                 fontsize=13, weight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out = f'results/figures/predictions_{args.tag}.png'
    plt.savefig(out, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'\nSaved: {out}')


if __name__ == '__main__':
    main()
