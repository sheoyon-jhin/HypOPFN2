"""
Fine-Tune M4 short-term forecasting (6 frequencies × standard horizons).

Design:
  - Load series from LOTSA arrow (m4_<freq> dirs).
  - Per series: last H points = test; rest = train.
  - Build sliding-window (context, future) pairs from train portion → FT.
  - Per-sample instance normalization on context.
  - Operator forecast: qt = 1.0 + i/seq_len for i in 0..H-1.
  - Val: 10% of sliding windows → best-epoch selection.
  - Test metric: sMAPE on last H of each series (de-normalized).

Usage:
  python experiments/ft_m4.py \
    --ckpt checkpoints/hyper4_10pct_full231B.pth --highfreq_nf 256 \
    --ft_epochs 3 --lr 1e-4 --tag hyper4_10pct_m4_ep3_lr1e-4
"""
import sys, os, argparse, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import torch.nn.functional as F
import pyarrow.ipc as ipc

from experiments.ft_common import (
    DEVICE, WarmupCosineLR, build_optimizer, BestKeeper,
    build_base_model, add_std_cli,
)

LOTSA_ROOT = './dataset/lotsa'

M4_CONFIGS = {
    'Yearly':    {'dir': 'm4_yearly',    'horizon': 6,  'seq_len': 48,  'seasonality': 1},
    'Quarterly': {'dir': 'm4_quarterly', 'horizon': 8,  'seq_len': 64,  'seasonality': 4},
    'Monthly':   {'dir': 'm4_monthly',   'horizon': 18, 'seq_len': 192, 'seasonality': 12},
    'Weekly':    {'dir': 'm4_weekly',    'horizon': 13, 'seq_len': 128, 'seasonality': 1},
    'Daily':     {'dir': 'm4_daily',     'horizon': 14, 'seq_len': 192, 'seasonality': 1},
    'Hourly':    {'dir': 'm4_hourly',    'horizon': 48, 'seq_len': 384, 'seasonality': 24},
}

# FeDaL Table-7 references (sMAPE)
FEDAL_REF = {'Yearly': 13.08, 'Quarterly': 9.81, 'Monthly': 12.12,
             'Weekly': 7.86, 'Daily': 3.16, 'Hourly': 12.40}


# ------------------------------------------------------------
# Data loading
# ------------------------------------------------------------
def load_m4_series(freq_dir, max_series=None):
    ds_path = os.path.join(LOTSA_ROOT, freq_dir)
    arrow_files = [f for f in os.listdir(ds_path) if f.endswith('.arrow')]
    if not arrow_files:
        return []
    arrow_path = os.path.join(ds_path, arrow_files[0])
    try:
        table = ipc.open_file(arrow_path).read_all()
    except Exception:
        with open(arrow_path, 'rb') as _f:
            table = ipc.open_stream(_f).read_all()
    target_col = None
    for cn in ['target', 'values', 'value']:
        if cn in table.column_names:
            target_col = cn; break
    if target_col is None:
        return []
    out = []
    n = len(table) if max_series is None else min(len(table), max_series * 3)
    for row_idx in range(n):
        try:
            vals = table.column(target_col)[row_idx].as_py()
            if isinstance(vals, list):
                arr = np.array(vals[0] if (vals and isinstance(vals[0], list)) else vals, dtype=np.float32)
            else:
                continue
            if arr.ndim > 1:
                arr = arr.flatten()
            if len(arr) > 10:
                out.append(arr)
            if max_series is not None and len(out) >= max_series:
                break
        except Exception:
            continue
    return out


def build_windows(series_list, horizon, seq_len, max_per_series=10, rng=None):
    """Build (ctx, future) windows. Pads short-train series with leading zeros."""
    rng = rng or np.random.default_rng(0)
    Xs, Ys = [], []
    for s in series_list:
        if len(s) <= horizon + 2:
            continue
        train = s[:-horizon]
        if len(train) < horizon + 2:
            continue
        if len(train) < seq_len + horizon + 1:
            # Short: produce one leading-pad window covering latest train end
            end = len(train)
            start = end - seq_len
            if start < 0:
                pad = np.zeros(-start, dtype=np.float32)
                ctx = np.concatenate([pad, train[:end].astype(np.float32)])
            else:
                ctx = train[start:end].astype(np.float32)
            fut_start = end
            fut_end = fut_start + horizon
            if fut_end > len(train):
                # Shouldn't reach here — train[-horizon:] used as future? use s[len(train):len(train)+horizon]
                continue
            Xs.append(ctx); Ys.append(train[fut_start:fut_end].astype(np.float32))
            continue
        ends = np.arange(seq_len, len(train) - horizon + 1)
        if len(ends) > max_per_series:
            ends = rng.choice(ends, max_per_series, replace=False)
        for e in ends:
            Xs.append(train[e - seq_len:e].astype(np.float32))
            Ys.append(train[e:e + horizon].astype(np.float32))
    if not Xs:
        return None, None
    return np.stack(Xs), np.stack(Ys)


# ------------------------------------------------------------
# sMAPE
# ------------------------------------------------------------
def smape(y_true, y_pred):
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    denom = np.where(denom == 0, 1.0, denom)
    return np.mean(np.abs(y_true - y_pred) / denom) * 100


# ------------------------------------------------------------
# Training / eval (batched)
# ------------------------------------------------------------
def run_ft_epoch(model, X, Y, horizon, seq_len, opt, sch, batch_size=128, amp=True, train=True):
    model.train(train)
    N = len(X)
    idx = np.random.permutation(N) if train else np.arange(N)
    total, count = 0.0, 0
    amp_ctx = (lambda: torch.autocast(device_type='cuda', dtype=torch.bfloat16)) if amp else (lambda: torch.enable_grad() if train else torch.no_grad())
    ctx_mgr = torch.enable_grad() if train else torch.no_grad()
    # Pre-build t-query (same for all samples)
    qt_base = (1.0 + torch.arange(horizon, device=DEVICE, dtype=torch.float32) / seq_len)
    with ctx_mgr:
        for i in range(0, N, batch_size):
            bi = idx[i:i + batch_size]
            bx = torch.from_numpy(X[bi]).to(DEVICE, non_blocking=True)
            by = torch.from_numpy(Y[bi]).to(DEVICE, non_blocking=True)
            m = bx.mean(-1, keepdim=True)
            s = bx.std(-1, keepdim=True).clamp_min(1e-5)
            bx_n = ((bx - m) / s).clamp(-10, 10)
            by_n = ((by - m) / s).clamp(-10, 10)
            qt = qt_base.unsqueeze(0).expand(len(bi), -1)
            with amp_ctx():
                pred = model.forward_train(bx_n, qt)
                loss = F.mse_loss(pred, by_n)
            if train:
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                if sch is not None:
                    sch.step()
            total += loss.item() * len(bi)
            count += len(bi)
    return total / max(count, 1)


@torch.no_grad()
def eval_smape(model, series_list, horizon, seq_len, batch_size=256, amp=True):
    """Predict the last `horizon` of each series (univariate), compute sMAPE."""
    model.eval()
    ctxs, tests = [], []
    means, stds = [], []
    for s in series_list:
        if len(s) <= horizon + 5:
            continue
        train = s[:-horizon]
        test = s[-horizon:]
        if len(train) >= seq_len:
            ctx = train[-seq_len:]
        else:
            ctx = np.pad(train, (seq_len - len(train), 0))
        m, sd = float(ctx.mean()), float(ctx.std())
        if sd < 1e-6: sd = 1.0
        ctxs.append(np.clip((ctx - m) / sd, -10, 10).astype(np.float32))
        tests.append(test)
        means.append(m); stds.append(sd)
    if not ctxs:
        return float('nan'), 0

    ctxs = np.stack(ctxs)
    qt_base = (1.0 + torch.arange(horizon, device=DEVICE, dtype=torch.float32) / seq_len)
    smapes = []
    amp_ctx = (lambda: torch.autocast(device_type='cuda', dtype=torch.bfloat16)) if amp else (lambda: torch.no_grad())
    for i in range(0, len(ctxs), batch_size):
        bx = torch.from_numpy(ctxs[i:i + batch_size]).to(DEVICE)
        qt = qt_base.unsqueeze(0).expand(bx.shape[0], -1)
        with amp_ctx():
            pred_n = model.forward_train(bx, qt).float().cpu().numpy()
        for j in range(len(bx)):
            k = i + j
            pred = pred_n[j] * stds[k] + means[k]
            sv = smape(tests[k], pred)
            if not (np.isnan(sv) or np.isinf(sv)):
                smapes.append(sv)
    return float(np.mean(smapes)) if smapes else float('nan'), len(smapes)


# ------------------------------------------------------------
# Per-frequency FT
# ------------------------------------------------------------
def ft_freq(freq_name, cfg, args, state):
    series = load_m4_series(cfg['dir'], max_series=args.max_series)
    if not series:
        return {'error': 'no_data'}
    horizon = cfg['horizon']
    seq_len = cfg['seq_len']

    # Reserve val subset of series for best-epoch tracking (instead of train/val on windows)
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(series))
    n_val = max(50, int(len(series) * args.val_frac))
    n_val = min(n_val, len(series) // 5)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    series_tr = [series[i] for i in tr_idx]
    series_val = [series[i] for i in val_idx]

    # Build FT windows from train-series' pre-test portion
    X, Y = build_windows(series_tr, horizon, seq_len,
                          max_per_series=args.windows_per_series, rng=rng)

    model = build_base_model(args, max_seq_len=max(720, seq_len)).to(DEVICE)
    model.load_state_dict(state)

    best_ep = 0
    pre_val_smape, _ = eval_smape(model, series_val, horizon, seq_len,
                                   batch_size=args.eval_batch_size, amp=args.amp)

    if X is None or len(X) == 0 or args.ft_epochs == 0:
        # No FT possible or requested — evaluate directly
        test_smape, n_test = eval_smape(model, series, horizon, seq_len,
                                         batch_size=args.eval_batch_size, amp=args.amp)
        return {'sMAPE': test_smape, 'n_series': n_test, 'horizon': horizon,
                'ft_windows': 0, 'val_smape_pre': pre_val_smape, 'val_smape_post': pre_val_smape}

    opt, lrs = build_optimizer(model, head_module=None,
                                enc_lr=args.lr * args.enc_lr_ratio,
                                head_lr=args.lr, wd=args.wd)
    steps_per_ep = max(1, (len(X) + args.batch_size - 1) // args.batch_size)
    sch = WarmupCosineLR(opt, lrs, steps_per_ep * args.ft_epochs,
                          warmup_frac=0.1, min_lr_frac=0.05)

    keeper = BestKeeper(mode='min', patience=args.patience)
    keeper.update(model, pre_val_smape, -1)   # track pre-FT as baseline

    for ep in range(args.ft_epochs):
        run_ft_epoch(model, X, Y, horizon, seq_len, opt, sch,
                     batch_size=args.batch_size, amp=args.amp, train=True)
        val_s, _ = eval_smape(model, series_val, horizon, seq_len,
                               batch_size=args.eval_batch_size, amp=args.amp)
        improved = keeper.update(model, val_s, ep)
        if keeper.should_stop():
            break
    keeper.restore(model)

    test_smape, n_test = eval_smape(model, series, horizon, seq_len,
                                     batch_size=args.eval_batch_size, amp=args.amp)
    return {
        'sMAPE': float(test_smape), 'n_series': n_test, 'horizon': horizon,
        'ft_windows': int(len(X)),
        'val_smape_pre': float(pre_val_smape),
        'val_smape_post': float(keeper.best),
        'best_epoch': int(keeper.best_epoch),
    }


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--tag', required=True)
    p.add_argument('--freqs', type=str, default=','.join(M4_CONFIGS.keys()))
    p.add_argument('--max_series', type=int, default=3000)
    p.add_argument('--windows_per_series', type=int, default=8)
    p.add_argument('--val_frac', type=float, default=0.1)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--eval_batch_size', type=int, default=256)
    p.add_argument('--ft_epochs', type=int, default=3)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--enc_lr_ratio', type=float, default=0.1)
    p.add_argument('--wd', type=float, default=1e-4)
    p.add_argument('--patience', type=int, default=None)
    p.add_argument('--amp', type=int, default=1)
    add_std_cli(p)
    args = p.parse_args()

    print('=' * 70)
    print(f'M4 FT: {args.ckpt}')
    print(f'  ft_epochs={args.ft_epochs} lr={args.lr} (enc={args.lr*args.enc_lr_ratio:.1e})')
    print('=' * 70)

    state = torch.load(args.ckpt, map_location=DEVICE, weights_only=True)
    freqs = [f.strip() for f in args.freqs.split(',') if f.strip()]
    results = {}
    for freq in freqs:
        cfg = M4_CONFIGS.get(freq)
        if cfg is None:
            continue
        t0 = time.time()
        try:
            r = ft_freq(freq, cfg, args, state)
            if 'error' in r:
                print(f'  {freq}: {r["error"]}')
                continue
            ref = FEDAL_REF.get(freq, 0.0)
            delta = r['sMAPE'] - ref
            print(f'  {freq:<10}: sMAPE={r["sMAPE"]:6.3f}  (FeDaL={ref:5.2f}, '
                  f'Δ={delta:+.2f})  ft_win={r.get("ft_windows", 0)} n={r["n_series"]} '
                  f'({time.time()-t0:.0f}s)')
            results[freq] = r
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f'  {freq}: ERROR {e}')

    if results:
        avg = float(np.mean([r['sMAPE'] for r in results.values() if 'sMAPE' in r]))
        ref_avg = float(np.mean([FEDAL_REF[k] for k in results if k in FEDAL_REF]))
        results['Average'] = avg
        print(f'\nAverage sMAPE: {avg:.3f}  (FeDaL avg={ref_avg:.3f}, Δ={avg-ref_avg:+.3f})')

    os.makedirs('results', exist_ok=True)
    out = f'results/{args.tag}_m4_ft.json'
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Saved: {out}')


if __name__ == '__main__':
    main()
