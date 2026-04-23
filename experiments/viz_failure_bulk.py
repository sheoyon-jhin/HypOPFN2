"""
Bulk failure visualization: many predictions per task to see failure patterns.

Output: results/figures/failure_analysis/
  ├── forecast/
  │   ├── ETTh1_grid.png, ETTh2_grid.png, ETTm1_grid.png, ETTm2_grid.png, Weather_grid.png
  ├── imputation/
  │   ├── ETTh1_grid.png, ..., Weather_grid.png  (per mask rate × samples)
  └── m4/
      └── Yearly_grid.png, Quarterly_grid.png, ..., Hourly_grid.png

Usage:
  python experiments/viz_failure_bulk.py \
      --ckpt checkpoints/v1_varlen_nll_failmode.pth --use_nll 1
"""
import sys, os, argparse, warnings
warnings.filterwarnings('ignore')
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
PRED_LENS = [96, 192, 336, 720]
MASK_RATES = [0.125, 0.25, 0.375, 0.5]
N_SAMPLES = 4  # samples per row in grid

DATASETS = {
    'ETTh1':   ('ETTh1',  './dataset/ETT-small/', 'ETTh1.csv', 7),
    'ETTh2':   ('ETTh2',  './dataset/ETT-small/', 'ETTh2.csv', 7),
    'ETTm1':   ('ETTm1',  './dataset/ETT-small/', 'ETTm1.csv', 7),
    'ETTm2':   ('ETTm2',  './dataset/ETT-small/', 'ETTm2.csv', 7),
    'Weather': ('custom', './dataset/weather/',  'weather.csv', 21),
}

M4_CONFIGS = {
    'Yearly':    {'name': 'm4_yearly',    'horizon': 6},
    'Quarterly': {'name': 'm4_quarterly', 'horizon': 8},
    'Monthly':   {'name': 'm4_monthly',   'horizon': 18},
    'Weekly':    {'name': 'm4_weekly',    'horizon': 13},
    'Daily':     {'name': 'm4_daily',     'horizon': 14},
    'Hourly':    {'name': 'm4_hourly',    'horizon': 48},
}


@torch.no_grad()
def forecast_one(model, x_ctx, horizon):
    if x_ctx.dim() == 1: x_ctx = x_ctx.unsqueeze(0)
    ps = model.encoder.patch_size
    eff = (x_ctx.shape[-1] // ps) * ps
    x_ctx = x_ctx[..., -eff:]
    m = x_ctx.mean(-1, keepdim=True)
    s = x_ctx.std(-1, keepdim=True).clamp(min=1e-6)
    x_n = ((x_ctx - m) / s).clamp(-10, 10)
    pred = model.forecast(x_n, n=horizon, seq_len_ref=eff).squeeze(0)
    return (pred * s.squeeze(-1) + m.squeeze(-1)).cpu().numpy()


@torch.no_grad()
def impute_one(model, x_full, mask_rate, seed=0):
    rng = np.random.RandomState(seed)
    L = len(x_full)
    m = float(x_full.mean())
    s = float(x_full.std().clip(1e-6))
    x_n = np.clip((x_full - m) / s, -10, 10)
    mask = rng.rand(L) > mask_rate
    mi = np.where(~mask)[0]
    if len(mi) == 0:
        return None
    x_masked = x_n * mask.astype(np.float32)
    ctx = torch.tensor(x_masked, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    qt = torch.tensor(mi / L, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    pred_n = model.forward_train(ctx, qt).squeeze(0).cpu().numpy()
    pred = pred_n * s + m
    true = x_full[mi]
    mse = float(((pred - true) ** 2).mean())
    return x_full, mask, pred, true, mi, mse


def get_ett_batches(ds_key, pred_len, n_samples):
    from data_provider.data_factory import data_provider
    d, root, f, enc_in = DATASETS[ds_key]
    a = SimpleNamespace(seq_len=SEQ_LEN, pred_len=pred_len, label_len=48, data=d,
        root_path=root, data_path=f, features='M', target='OT', freq='h',
        embed='timeF', enc_in=enc_in, dec_in=enc_in, c_out=enc_in,
        num_workers=0, batch_size=1, exp_name='MTSF', ordered_data=False,
        data_amount=-1, combine_Gaussian_datasets=False, synthetic_data_path='',
        synthetic_root_path='./', synthetic_length=1024, stride=-1)
    _, tdl = data_provider(a, 'test')
    samples = []
    step = max(1, len(tdl) // (n_samples * 2))
    for i, (bx, by, _, _) in enumerate(tdl):
        if len(samples) >= n_samples: break
        if i % step != 0: continue
        samples.append((bx[0].numpy(), by[0].numpy()))
    return samples


def viz_forecast(model, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for ds in DATASETS:
        print(f'  forecast {ds}')
        fig, axes = plt.subplots(len(PRED_LENS), N_SAMPLES,
                                 figsize=(4 * N_SAMPLES, 2.5 * len(PRED_LENS)),
                                 squeeze=False)
        for r, pl in enumerate(PRED_LENS):
            samples = get_ett_batches(ds, pl, N_SAMPLES)
            for c, (bx, by) in enumerate(samples):
                ax = axes[r, c]
                ctx = bx[-SEQ_LEN:, 0]    # channel 0
                true = by[-pl:, 0]
                pred = forecast_one(model,
                    torch.tensor(ctx, dtype=torch.float32, device=DEVICE), pl)
                t_ctx = np.arange(len(ctx))
                t_fut = np.arange(len(ctx), len(ctx) + len(pred))
                show = min(200, len(ctx))
                ax.plot(t_ctx[-show:], ctx[-show:], 'k-', alpha=0.4, linewidth=1)
                ax.plot(t_fut, true, 'g-', linewidth=1.5, label='truth')
                ax.plot(t_fut, pred, 'r--', linewidth=1.5, label='ours')
                mse = ((pred - true) ** 2).mean()
                ax.set_title(f'h={pl} MSE={mse:.3f}', fontsize=9)
                ax.tick_params(labelsize=7)
                if r == 0 and c == 0:
                    ax.legend(fontsize=7)
                ax.grid(alpha=0.3)
        plt.suptitle(f'{ds} — Forecast (rows=pred_len, cols=samples)', fontsize=12)
        plt.tight_layout()
        fn = os.path.join(out_dir, f'{ds}_grid.png')
        plt.savefig(fn, dpi=90, bbox_inches='tight')
        plt.close()
        print(f'    saved {fn}')


def viz_imputation(model, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for ds in DATASETS:
        print(f'  imputation {ds}')
        fig, axes = plt.subplots(len(MASK_RATES), N_SAMPLES,
                                 figsize=(4.5 * N_SAMPLES, 2.5 * len(MASK_RATES)),
                                 squeeze=False)
        samples = get_ett_batches(ds, 96, N_SAMPLES * 2)[:N_SAMPLES]  # reuse ETT loader
        for r, mr in enumerate(MASK_RATES):
            for c, (bx, _) in enumerate(samples):
                ax = axes[r, c]
                full = bx[-SEQ_LEN:, 0]
                out = impute_one(model, full, mr, seed=r * 10 + c)
                if out is None: continue
                full_orig, mask, pred, true, mi, mse = out
                win = min(300, len(full_orig))
                t = np.arange(len(full_orig))
                ax.plot(t[-win:], full_orig[-win:], 'k-', alpha=0.25, linewidth=0.8)
                visible = full_orig.copy()
                visible[~mask] = np.nan
                ax.plot(t[-win:], visible[-win:], 'b.', markersize=1.5, alpha=0.6)
                mi_win = mi[mi >= len(full_orig) - win]
                if len(mi_win) > 0:
                    pred_win = pred[np.isin(mi, mi_win)]
                    true_win = full_orig[mi_win]
                    ax.scatter(mi_win, true_win, c='g', s=8, alpha=0.8, zorder=3, label='truth')
                    ax.scatter(mi_win, pred_win, c='r', marker='x', s=12, zorder=4, label='ours')
                ax.set_title(f'mr={mr:.0%} MSE={mse:.3f}', fontsize=9)
                ax.tick_params(labelsize=7)
                if r == 0 and c == 0:
                    ax.legend(fontsize=7)
                ax.grid(alpha=0.3)
        plt.suptitle(f'{ds} — Imputation (rows=mask_rate, cols=samples)', fontsize=12)
        plt.tight_layout()
        fn = os.path.join(out_dir, f'{ds}_grid.png')
        plt.savefig(fn, dpi=90, bbox_inches='tight')
        plt.close()
        print(f'    saved {fn}')


def viz_m4(model, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    try:
        from gluonts.dataset.repository import get_dataset
    except ImportError:
        print('  gluonts not available, skipping M4')
        return
    for freq, cfg in M4_CONFIGS.items():
        print(f'  m4 {freq}')
        try:
            ds = get_dataset(cfg['name'], regenerate=False)
        except Exception as e:
            print(f'    failed: {e}')
            continue
        test_series = list(ds.test)
        rows, cols = 3, 4
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 2.3 * rows), squeeze=False)
        step = max(1, len(test_series) // (rows * cols + 2))
        idx = 0
        for i, ts in enumerate(test_series):
            if idx >= rows * cols: break
            if i % step != 0: continue
            full = np.array(ts['target'], dtype=np.float32)
            if len(full) < cfg['horizon'] + 10: continue
            history = full[:-cfg['horizon']]
            truth = full[-cfg['horizon']:]
            ctx = history[-SEQ_LEN:] if len(history) >= SEQ_LEN else \
                  np.concatenate([np.zeros(SEQ_LEN - len(history), dtype=np.float32), history])
            pred = forecast_one(model,
                torch.tensor(ctx, dtype=torch.float32, device=DEVICE), cfg['horizon'])
            r, c = idx // cols, idx % cols
            ax = axes[r, c]
            h_show = min(100, len(history))
            t_ctx = np.arange(h_show)
            t_fut = np.arange(h_show, h_show + cfg['horizon'])
            ax.plot(t_ctx, history[-h_show:], 'k-', alpha=0.5, linewidth=1)
            ax.plot(t_fut, truth, 'g-', linewidth=1.5)
            ax.plot(t_fut, pred, 'r--', linewidth=1.5)
            smape = np.mean(np.abs(truth - pred) / ((np.abs(truth) + np.abs(pred)) / 2).clip(1e-8)) * 100
            ax.set_title(f'sMAPE={smape:.1f}', fontsize=9)
            ax.tick_params(labelsize=7)
            ax.grid(alpha=0.3)
            idx += 1
        # Legend on first axes
        axes[0, 0].legend(['history', 'truth', 'ours'], fontsize=7, loc='upper left')
        plt.suptitle(f'M4 {freq} (h={cfg["horizon"]}) — forecasts', fontsize=12)
        plt.tight_layout()
        fn = os.path.join(out_dir, f'{freq}_grid.png')
        plt.savefig(fn, dpi=90, bbox_inches='tight')
        plt.close()
        print(f'    saved {fn}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--use_nll', type=int, default=0)
    p.add_argument('--tag', default='failure_analysis')
    p.add_argument('--skip', nargs='+', default=[], choices=['forecast', 'imputation', 'm4'])
    args = p.parse_args()

    model = OperatorModelVarLen(max_seq_len=SEQ_LEN, use_nll=bool(args.use_nll)).to(DEVICE)
    state = torch.load(args.ckpt, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    base_dir = f'results/figures/{args.tag}'
    print(f'Saving to {base_dir}/')

    if 'forecast' not in args.skip:
        viz_forecast(model, os.path.join(base_dir, 'forecast'))
    if 'imputation' not in args.skip:
        viz_imputation(model, os.path.join(base_dir, 'imputation'))
    if 'm4' not in args.skip:
        viz_m4(model, os.path.join(base_dir, 'm4'))

    print(f'\nDone. See {base_dir}/')


if __name__ == '__main__':
    main()