"""
Failure analysis: visualize where our model fails vs ground truth.

3 tasks × 3 samples = 9 subplots:
  - Row 1: ETTm1 forecast (our worst long-term, +56% MSE)
  - Row 2: Weather imputation (our worst imputation, +369%)
  - Row 3: M4 Hourly forecast (short-term, +76%)

Usage:
  python experiments/viz_failure_analysis.py \
      --ckpt checkpoints/v1_varlen_nll_failmode.pth --use_nll 1 --tag fail_best
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
SEQ_LEN = 720


@torch.no_grad()
def forecast_normalized(model, x_ctx, horizon):
    """x_ctx: (1, L) or (L,). returns (horizon,) in original scale."""
    if x_ctx.dim() == 1:
        x_ctx = x_ctx.unsqueeze(0)
    patch_size = model.encoder.patch_size
    eff = (x_ctx.shape[-1] // patch_size) * patch_size
    x_ctx = x_ctx[..., -eff:]
    m = x_ctx.mean(-1, keepdim=True)
    s = x_ctx.std(-1, keepdim=True).clamp(min=1e-6)
    x_n = ((x_ctx - m) / s).clamp(-10, 10)
    pred = model.forecast(x_n, n=horizon, seq_len_ref=eff).squeeze(0)
    return (pred * s.squeeze(-1) + m.squeeze(-1)).cpu().numpy()


@torch.no_grad()
def impute_normalized(model, x_full, mask_rate, seed=0):
    """Impute masked positions in x_full (length seq_len). Returns (pred, true, mask)."""
    rng = np.random.RandomState(seed)
    L = len(x_full)
    m = x_full.mean()
    s = x_full.std().clip(min=1e-6)
    x_n = np.clip((x_full - m) / s, -10, 10)
    mask = rng.rand(L) > mask_rate   # True = keep
    x_masked = x_n * mask.astype(np.float32)

    ctx = torch.tensor(x_masked, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    # Query = masked positions
    mi = np.where(~mask)[0]
    qt = torch.tensor(mi / L, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    pred_n = model.forward_train(ctx, qt).squeeze(0).cpu().numpy()
    pred = pred_n * s + m
    true = x_full[mi]
    return x_masked * s + m, pred, true, mi


def load_data(ds_key, pred_len=96):
    from data_provider.data_factory import data_provider
    cfg = {
        'ETTh1':   ('ETTh1',  './dataset/ETT-small/', 'ETTh1.csv', 7),
        'ETTm1':   ('ETTm1',  './dataset/ETT-small/', 'ETTm1.csv', 7),
        'Weather': ('custom', './dataset/weather/',  'weather.csv', 21),
    }
    d, root, f, enc_in = cfg[ds_key]
    a = SimpleNamespace(seq_len=SEQ_LEN, pred_len=pred_len, label_len=48, data=d,
        root_path=root, data_path=f, features='M', target='OT', freq='h',
        embed='timeF', enc_in=enc_in, dec_in=enc_in, c_out=enc_in,
        num_workers=0, batch_size=1, exp_name='MTSF', ordered_data=False,
        data_amount=-1, combine_Gaussian_datasets=False, synthetic_data_path='',
        synthetic_root_path='./', synthetic_length=1024, stride=-1)
    _, tdl = data_provider(a, 'test')
    return tdl, enc_in


def pick_ettm1_samples(model, n_samples=3, pred_len=192):
    tdl, C = load_data('ETTm1', pred_len=pred_len)
    samples = []
    for i, (bx, by, _, _) in enumerate(tdl):
        if len(samples) >= n_samples: break
        # Use first channel; pick every 500th batch
        if i % 500 != 0: continue
        ctx = bx[0, -SEQ_LEN:, 0].numpy()
        true = by[0, -pred_len:, 0].numpy()
        pred = forecast_normalized(model,
            torch.tensor(ctx, dtype=torch.float32, device=DEVICE), pred_len)
        samples.append((ctx, pred, true))
    return samples


def pick_weather_impute_samples(model, n_samples=3, mask_rate=0.375):
    tdl, C = load_data('Weather')
    samples = []
    for i, (bx, _, _, _) in enumerate(tdl):
        if len(samples) >= n_samples: break
        if i % 500 != 0: continue
        full = bx[0, -SEQ_LEN:, 0].numpy()
        masked, pred, true, mi = impute_normalized(model, full, mask_rate, seed=i)
        samples.append((full, masked, pred, true, mi))
    return samples


def pick_m4_samples(model, freq='Hourly', horizon=48, n_samples=3):
    from gluonts.dataset.repository import get_dataset
    cfg_name = {'Hourly': 'm4_hourly', 'Daily': 'm4_daily', 'Monthly': 'm4_monthly',
                'Weekly': 'm4_weekly', 'Quarterly': 'm4_quarterly', 'Yearly': 'm4_yearly'}[freq]
    ds = get_dataset(cfg_name, regenerate=False)
    test_series = list(ds.test)
    samples = []
    step = max(1, len(test_series) // (n_samples + 2))
    for i, ts in enumerate(test_series):
        if len(samples) >= n_samples: break
        if i % step != 0: continue
        full = np.array(ts['target'], dtype=np.float32)
        if len(full) < horizon + 10: continue
        history = full[:-horizon]
        truth = full[-horizon:]
        # Left-pad history to SEQ_LEN
        if len(history) >= SEQ_LEN:
            ctx = history[-SEQ_LEN:]
        else:
            ctx = np.concatenate([np.zeros(SEQ_LEN - len(history), dtype=np.float32), history])
        pred = forecast_normalized(model,
            torch.tensor(ctx, dtype=torch.float32, device=DEVICE), horizon)
        samples.append((history, pred, truth))
    return samples


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--use_nll', type=int, default=0)
    p.add_argument('--tag', default='fail_analysis')
    args = p.parse_args()

    model = OperatorModelVarLen(max_seq_len=SEQ_LEN, use_nll=bool(args.use_nll)).to(DEVICE)
    state = torch.load(args.ckpt, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    print('Sampling ETTm1 forecast...')
    s_fc = pick_ettm1_samples(model, n_samples=3, pred_len=192)
    print('Sampling Weather imputation...')
    s_imp = pick_weather_impute_samples(model, n_samples=3, mask_rate=0.375)
    print('Sampling M4 Hourly short-term...')
    s_m4 = pick_m4_samples(model, freq='Hourly', horizon=48, n_samples=3)

    fig, axes = plt.subplots(3, 3, figsize=(18, 10))

    # Row 1: ETTm1 forecast
    for c, (ctx, pred, true) in enumerate(s_fc):
        ax = axes[0, c]
        t_ctx = np.arange(len(ctx))
        t_fut = np.arange(len(ctx), len(ctx) + len(pred))
        ax.plot(t_ctx[-200:], ctx[-200:], 'k-', alpha=0.5, label='context')
        ax.plot(t_fut, true, 'g-', label='truth', linewidth=2)
        ax.plot(t_fut, pred, 'r--', label='ours', linewidth=2)
        mse = ((pred - true) ** 2).mean()
        ax.set_title(f'ETTm1 forecast h=192 | MSE={mse:.3f}', fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    # Row 2: Weather imputation
    for c, (full, masked, pred, true, mi) in enumerate(s_imp):
        ax = axes[1, c]
        t = np.arange(len(full))
        # Show last 300 points for clarity
        win = 300
        ax.plot(t[-win:], full[-win:], 'k-', alpha=0.4, label='original')
        visible = masked.copy()
        visible[mi] = np.nan   # hide masked for visualization context
        ax.plot(t[-win:], visible[-win:], 'b.', markersize=2, alpha=0.6, label='visible (kept)')
        # Scatter masked predictions
        mi_visible = mi[mi >= len(full) - win]
        if len(mi_visible) > 0:
            pred_visible = pred[np.isin(mi, mi_visible)]
            true_visible = full[mi_visible]
            ax.scatter(mi_visible, true_visible, c='g', s=12, label='masked truth', zorder=3)
            ax.scatter(mi_visible, pred_visible, c='r', marker='x', s=18, label='ours', zorder=4)
        mse = ((pred - true) ** 2).mean()
        ax.set_title(f'Weather imp 37.5% | MSE={mse:.3f}', fontsize=10)
        ax.legend(fontsize=7, loc='upper left')
        ax.grid(alpha=0.3)

    # Row 3: M4 Hourly forecast
    for c, (history, pred, truth) in enumerate(s_m4):
        ax = axes[2, c]
        h = history[-200:] if len(history) > 200 else history
        t_ctx = np.arange(len(h))
        t_fut = np.arange(len(h), len(h) + len(pred))
        ax.plot(t_ctx, h, 'k-', alpha=0.5, label='history')
        ax.plot(t_fut, truth, 'g-', label='truth', linewidth=2)
        ax.plot(t_fut, pred, 'r--', label='ours', linewidth=2)
        smape = np.mean(np.abs(truth - pred) / ((np.abs(truth) + np.abs(pred)) / 2).clip(min=1e-8)) * 100
        ax.set_title(f'M4 Hourly h=48 | sMAPE={smape:.1f}', fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    axes[0, 0].set_ylabel('ETTm1 forecast\n(long-term, +56%)', fontsize=11, fontweight='bold')
    axes[1, 0].set_ylabel('Weather imputation\n(worst, +369%)', fontsize=11, fontweight='bold')
    axes[2, 0].set_ylabel('M4 Hourly\n(short-term, +76%)', fontsize=11, fontweight='bold')
    plt.suptitle(f'Failure Analysis: {os.path.basename(args.ckpt)}', fontsize=13)
    plt.tight_layout()

    os.makedirs('results/figures', exist_ok=True)
    out = f'results/figures/{args.tag}.png'
    plt.savefig(out, dpi=120, bbox_inches='tight')
    print(f'Saved: {out}')


if __name__ == '__main__':
    main()