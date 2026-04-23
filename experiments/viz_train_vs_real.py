"""
Visualize model predictions on:
  (A) Training distribution: LOTSA + Synthetic samples  (B) Real-world unseen data: ETT + Weather

Shows side-by-side: context + ground-truth future + model prediction.
If model learns train data → (A) should be tight (low error).
If model generalizes → (B) should also look reasonable.

Outputs:
  results/figures/inference_train_vs_real.png

Usage:
  python experiments/viz_train_vs_real.py \
      --ckpt checkpoints/overnight_seq720_s50.pth --seq_len 720 \
      --tag seq720_train_vs_real
"""
import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch, torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from types import SimpleNamespace

from experiments.exp_lotsa_scaling import (
    OperatorModel, LOTSAScalingDataset, SyntheticGapFiller,
)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LOTSA_DIR = os.environ.get('LOTSA_DIR', './dataset/lotsa')

REAL_DATASETS = {
    'ETTh1':   ('ETTh1',  './dataset/ETT-small/', 'ETTh1.csv', 7),
    'ETTh2':   ('ETTh2',  './dataset/ETT-small/', 'ETTh2.csv', 7),
    'ETTm1':   ('ETTm1',  './dataset/ETT-small/', 'ETTm1.csv', 7),
    'ETTm2':   ('ETTm2',  './dataset/ETT-small/', 'ETTm2.csv', 7),
    'Weather': ('custom', './dataset/weather/',  'weather.csv', 21),
}


@torch.no_grad()
def predict_window(model, context_norm, horizon, seq_len):
    """Direct prediction given normalized context."""
    if len(context_norm) >= seq_len:
        ctx = context_norm[-seq_len:]
    else:
        ctx = np.concatenate([np.zeros(seq_len - len(context_norm)), context_norm])
    ctx_t = torch.tensor(ctx, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    pred = model.forecast(ctx_t, n=horizon).squeeze(0).cpu().numpy()
    return ctx, pred


def plot_panel(ax, ctx, truth, pred, title, color_ctx='steelblue'):
    ctx_x = np.arange(len(ctx))
    fut_x = np.arange(len(ctx), len(ctx) + len(truth))
    ax.plot(ctx_x, ctx, color=color_ctx, linewidth=0.8, label='context')
    ax.plot(fut_x, truth, color='black', linewidth=1.2, label='truth')
    ax.plot(fut_x, pred, color='red', linewidth=1.2, linestyle='--', label='pred')
    ax.axvline(len(ctx), color='gray', linestyle=':', alpha=0.5)
    mse = float(((pred - truth) ** 2).mean())
    ax.set_title(f'{title}  MSE={mse:.3f}', fontsize=10)
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
    return mse


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--decomp', type=int, default=0)
    p.add_argument('--seq_len', type=int, default=720)
    p.add_argument('--horizon', type=int, default=192)
    p.add_argument('--tag', type=str, required=True)
    p.add_argument('--n_train', type=int, default=4, help='Samples from train distribution (LOTSA + Synth)')
    p.add_argument('--n_real', type=int, default=5, help='Samples from real datasets (1 per ETT/Weather)')
    args = p.parse_args()

    print(f'Loading model from {args.ckpt}...')
    model = OperatorModel(seq_len=args.seq_len, use_latent_decomp=bool(args.decomp)).to(DEVICE)
    state = torch.load(args.ckpt, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    window_len = args.seq_len + args.horizon  # need seq_len + horizon for context+truth

    # ----- (A) Train distribution: Synth + LOTSA -----
    np.random.seed(42); torch.manual_seed(42)
    train_panels = []

    # LOTSA: load small subset
    print('Sampling LOTSA...')
    lotsa = LOTSAScalingDataset(LOTSA_DIR, 5, seq_len=max(args.seq_len * 2, window_len + 100))
    if len(lotsa) > 0:
        idx_l = np.random.choice(len(lotsa), min(args.n_train // 2, len(lotsa)), replace=False)
        for j, i in enumerate(idx_l):
            w = lotsa[i].numpy()
            if len(w) < window_len: continue
            ctx_raw = w[:args.seq_len]
            truth = w[args.seq_len:args.seq_len + args.horizon]
            m, s = ctx_raw.mean(), ctx_raw.std()
            if s < 1e-6: continue
            ctx_n = (ctx_raw - m) / s
            truth_n = (truth - m) / s
            _, pred_n = predict_window(model, ctx_n, args.horizon, args.seq_len)
            train_panels.append({
                'title': f'LOTSA-{j}',
                'ctx': ctx_n, 'truth': truth_n, 'pred': pred_n,
            })

    # Synthetic: generate small set
    print('Sampling Synthetic...')
    synth = SyntheticGapFiller(n_samples=max(args.n_train, 16), seq_len=window_len + 50)
    idx_s = np.random.choice(len(synth), args.n_train - len(train_panels), replace=False)
    for j, i in enumerate(idx_s):
        w = synth[i].numpy()
        ctx_raw = w[:args.seq_len]
        truth = w[args.seq_len:args.seq_len + args.horizon]
        m, s = ctx_raw.mean(), ctx_raw.std()
        if s < 1e-6: continue
        ctx_n = (ctx_raw - m) / s
        truth_n = (truth - m) / s
        _, pred_n = predict_window(model, ctx_n, args.horizon, args.seq_len)
        train_panels.append({
            'title': f'Synth-{j}',
            'ctx': ctx_n, 'truth': truth_n, 'pred': pred_n,
        })

    # ----- (B) Real-world: ETT + Weather -----
    print('Sampling real datasets...')
    real_panels = []
    from data_provider.data_factory import data_provider
    for dn, (d, root, f, enc_in) in REAL_DATASETS.items():
        try:
            a = SimpleNamespace(seq_len=args.seq_len, pred_len=args.horizon, label_len=48, data=d,
                                root_path=root, data_path=f, features='M', target='OT', freq='h',
                                embed='timeF', enc_in=enc_in, dec_in=enc_in, c_out=enc_in,
                                num_workers=2, batch_size=1, exp_name='MTSF', ordered_data=False,
                                data_amount=-1, combine_Gaussian_datasets=False, synthetic_data_path='',
                                synthetic_root_path='./', synthetic_length=1024, stride=-1)
            _, tdl = data_provider(a, 'test')
            # Take first batch, first sample, channel = target (OT = last by convention)
            for bx, by, _, _ in tdl:
                ch = bx.shape[-1] - 1  # OT channel
                x_full = bx[0, :, ch].numpy()
                if len(x_full) < args.seq_len:
                    pad = np.zeros(args.seq_len - len(x_full))
                    x_full = np.concatenate([pad, x_full])
                ctx_raw = x_full[-args.seq_len:]
                truth = by[0, -args.horizon:, ch].numpy()
                m, s = ctx_raw.mean(), ctx_raw.std()
                if s < 1e-6: break
                ctx_n = (ctx_raw - m) / s
                truth_n = (truth - m) / s
                _, pred_n = predict_window(model, ctx_n, args.horizon, args.seq_len)
                real_panels.append({
                    'title': f'{dn} (OT)',
                    'ctx': ctx_n, 'truth': truth_n, 'pred': pred_n,
                })
                break
        except Exception as e:
            print(f'  {dn}: ERROR {e}')

    # ----- Plot -----
    n_train = len(train_panels)
    n_real = len(real_panels)
    total = n_train + n_real
    n_cols = 3
    n_rows = (total + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 2.8))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes

    print(f'\nPlotting {total} panels...')
    train_mses, real_mses = [], []
    for i, p in enumerate(train_panels):
        mse = plot_panel(axes[i], p['ctx'], p['truth'], p['pred'], p['title'],
                         color_ctx='steelblue')
        train_mses.append(mse)
    for i, p in enumerate(real_panels):
        mse = plot_panel(axes[n_train + i], p['ctx'], p['truth'], p['pred'], p['title'],
                         color_ctx='darkorange')
        real_mses.append(mse)
    for j in range(total, len(axes)):
        axes[j].axis('off')

    fig.suptitle(
        f'Model: {os.path.basename(args.ckpt)} — seq_len={args.seq_len} horizon={args.horizon}\n'
        f'TRAIN avg MSE: {np.mean(train_mses):.3f} | REAL avg MSE: {np.mean(real_mses):.3f}',
        fontsize=12,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs('results/figures', exist_ok=True)
    out = f'results/figures/inference_{args.tag}.png'
    plt.savefig(out, dpi=120); plt.close()
    print(f'\nSaved: {out}')
    print(f'Train MSE: mean={np.mean(train_mses):.4f}')
    print(f'Real  MSE: mean={np.mean(real_mses):.4f}')
    print(f'Generalization gap: {np.mean(real_mses)/max(np.mean(train_mses),1e-6):.2f}x')


if __name__ == '__main__':
    main()
