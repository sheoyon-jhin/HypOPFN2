"""
Weather-specific prediction visualization.

Shows model predictions on 6 Weather test samples from different channels.
Helps verify Weather performance claim.

Usage:
  python experiments/viz_weather.py --ckpt checkpoints/lotsa_s50_seq720_synth500k_evalbias.pth \
      --seq_len 720 --horizon 192 --tag weather_seq720
"""
import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch, torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from types import SimpleNamespace

from experiments.exp_lotsa_scaling import OperatorModel

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--decomp', type=int, default=0)
    p.add_argument('--seq_len', type=int, default=720)
    p.add_argument('--horizon', type=int, default=192)
    p.add_argument('--tag', type=str, required=True)
    p.add_argument('--n_samples', type=int, default=6)
    p.add_argument('--channels', type=str, default='OT,T,rh,wv', help='Channel name hints')
    args = p.parse_args()

    model = OperatorModel(seq_len=args.seq_len, use_latent_decomp=bool(args.decomp)).to(DEVICE)
    state = torch.load(args.ckpt, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    from data_provider.data_factory import data_provider
    a = SimpleNamespace(seq_len=args.seq_len, pred_len=args.horizon, label_len=48, data='custom',
                        root_path='./dataset/weather/', data_path='weather.csv',
                        features='M', target='OT', freq='h',
                        embed='timeF', enc_in=21, dec_in=21, c_out=21,
                        num_workers=2, batch_size=32, exp_name='MTSF', ordered_data=False,
                        data_amount=-1, combine_Gaussian_datasets=False, synthetic_data_path='',
                        synthetic_root_path='./', synthetic_length=1024, stride=-1)
    _, tdl = data_provider(a, 'test')

    # Weather columns (based on typical Weather dataset)
    # Actual column names from head of weather.csv
    with open('./dataset/weather/weather.csv') as f:
        header = f.readline().strip().split(',')
    col_names = header[1:]  # skip 'date'

    # Collect diverse samples
    torch.manual_seed(42); np.random.seed(42)
    collected = []
    # Channels to show (mix of different weather variables)
    chs_to_plot = [20, 2, 5, 11]  # OT, T_degC, rh, wv
    batch_i = 0
    stride = 20  # sample every 20th batch for diversity
    with torch.no_grad():
        for bx, by, _, _ in tdl:
            if batch_i % stride != 0:
                batch_i += 1
                continue
            b_idx = 0  # first sample in batch
            for ch in chs_to_plot:
                if len(collected) >= args.n_samples:
                    break
                x_ch = bx[b_idx, :, ch].float().to(DEVICE)
                if x_ch.shape[0] < args.seq_len:
                    x_ch = F.pad(x_ch, (args.seq_len - x_ch.shape[0], 0))
                x_ctx = x_ch[-args.seq_len:]
                m = x_ctx.mean(); s = x_ctx.std().clamp(min=1e-6)
                x_n = ((x_ctx - m) / s).clamp(-10, 10).unsqueeze(0)
                pred = model.forecast(x_n, n=args.horizon).squeeze(0)
                pred_d = pred * s + m
                truth = by[b_idx, -args.horizon:, ch].float().to(DEVICE)
                mse = ((pred_d - truth) ** 2).mean().item()
                collected.append({
                    'sample_idx': batch_i,
                    'channel_idx': ch,
                    'channel_name': col_names[ch] if ch < len(col_names) else f'ch{ch}',
                    'context': x_ctx.cpu().numpy(),
                    'truth': truth.cpu().numpy(),
                    'pred': pred_d.cpu().numpy(),
                    'mse': mse,
                })
            batch_i += 1
            if len(collected) >= args.n_samples:
                break

    # Plot first 6 (2 per sample × 3 samples)
    n_plot = min(len(collected), 8)
    collected = collected[:n_plot]
    fig, axes = plt.subplots((n_plot + 1) // 2, 2, figsize=(16, 3 * ((n_plot + 1) // 2)))
    axes = axes.flatten() if n_plot > 1 else [axes]

    avg_mse = np.mean([c['mse'] for c in collected])
    for ax, c in zip(axes, collected):
        ctx = c['context']; tgt = c['truth']; pr = c['pred']
        ctx_x = np.arange(len(ctx))
        fut_x = np.arange(len(ctx), len(ctx) + len(tgt))
        ax.plot(ctx_x, ctx, color='steelblue', linewidth=0.8, label='context', alpha=0.8)
        ax.plot(fut_x, tgt, color='black', linewidth=1.4, label='truth')
        ax.plot(fut_x, pr, color='red', linewidth=1.4, linestyle='--', label='pred')
        ax.axvline(len(ctx), color='gray', linestyle=':', alpha=0.5)
        ax.set_title(f'Weather ch="{c["channel_name"]}" idx={c["sample_idx"]}  MSE={c["mse"]:.3f}', fontsize=10)
        ax.legend(fontsize=8, loc='upper left'); ax.grid(alpha=0.3)

    for j in range(n_plot, len(axes)):
        axes[j].axis('off')

    fig.suptitle(f'Weather Forecast — {os.path.basename(args.ckpt)} — avg MSE={avg_mse:.3f}', fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs('results/figures', exist_ok=True)
    out = f'results/figures/weather_{args.tag}.png'
    plt.savefig(out, dpi=120); plt.close()
    print(f'Saved: {out}')
    print(f'Avg MSE: {avg_mse:.4f}')


if __name__ == '__main__':
    main()
