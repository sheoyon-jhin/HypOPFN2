"""
M4 Short-term Forecasting — FT N-epoch per frequency.

Each M4 frequency (Yearly/Quarterly/Monthly/Weekly/Daily/Hourly) has a
different horizon. We:
  1. Load series from LOTSA arrow format (m4_<freq> dir).
  2. For each series, strip last `horizon` as test; train on sliding windows of rest.
  3. FT encoder+head for N epochs with MSE loss.
  4. Evaluate sMAPE on last-horizon predictions.

Usage:
  python experiments/eval_m4_ft.py \
    --ckpt checkpoints/hyper4_10pct_full231B.pth --highfreq_nf 256 \
    --ft_epochs 3 --lr 1e-4 --tag hyper4_10pct_m4ft_ep3_lr1e-4
"""
import sys, os, argparse, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn.functional as F
import numpy as np
import pyarrow.ipc as ipc
from torch import optim

from experiments.exp_v1_varlen_ext import OperatorModelVarLen, OperatorModelDecomp

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

M4_CONFIGS = {
    'Yearly':    {'dir': 'm4_yearly',    'horizon': 6,  'seasonality': 1},
    'Quarterly': {'dir': 'm4_quarterly', 'horizon': 8,  'seasonality': 4},
    'Monthly':   {'dir': 'm4_monthly',   'horizon': 18, 'seasonality': 12},
    'Weekly':    {'dir': 'm4_weekly',    'horizon': 13, 'seasonality': 1},
    'Daily':     {'dir': 'm4_daily',     'horizon': 14, 'seasonality': 1},
    'Hourly':    {'dir': 'm4_hourly',    'horizon': 48, 'seasonality': 24},
}

LOTSA_ROOT = './dataset/lotsa'


def load_m4_series(freq_dir, max_series=None):
    """Load m4 series from LOTSA arrow files (IPC stream or file format)."""
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
    series_list = []
    n = min(len(table), max_series * 3 if max_series else len(table))
    for row_idx in range(n):
        try:
            vals = table.column(target_col)[row_idx].as_py()
            if isinstance(vals, list):
                arr = np.array(vals[0] if (len(vals) > 0 and isinstance(vals[0], list)) else vals, dtype=np.float32)
            else:
                continue
            if arr.ndim > 1:
                arr = arr.flatten()
            if len(arr) > 10:
                series_list.append(arr)
            if max_series is not None and len(series_list) >= max_series:
                break
        except Exception:
            continue
    return series_list


def smape(y_true, y_pred):
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    denom = np.where(denom == 0, 1.0, denom)
    return np.mean(np.abs(y_true - y_pred) / denom) * 100


def build_model(args):
    if args.model_type == 'decomp':
        decomp_k = tuple(int(x) for x in args.decomp_kernels.split(','))
        return OperatorModelDecomp(
            max_seq_len=args.seq_len, d_model=args.d_model, n_layers=args.n_layers,
            trunk_w=args.trunk_w, fourier_nf=args.fourier_nf,
            pool_type=args.pool_type, highfreq_nf=args.highfreq_nf,
            all_fixed=bool(args.all_fixed), decomp_kernels=decomp_k,
        )
    return OperatorModelVarLen(
        max_seq_len=args.seq_len, d_model=args.d_model, n_layers=args.n_layers,
        trunk_w=args.trunk_w, hybrid_trunk=bool(args.hybrid_trunk),
        use_nll=bool(args.use_nll), fourier_nf=args.fourier_nf,
        multi_scale_fourier=bool(args.multi_scale_fourier),
        multi_scale_iq=bool(args.multi_scale_iq),
        pool_type=args.pool_type, highfreq_nf=args.highfreq_nf,
        all_fixed=bool(args.all_fixed),
    )


def build_windows(series_list, horizon, seq_len, stride=1, max_per_series=10):
    """Build (context, target_future) windows from train portion of each series.
    Returns X (N, seq_len), Y (N, horizon)."""
    Xs, Ys = [], []
    for s in series_list:
        train = s[:-horizon] if len(s) > horizon + 5 else None
        if train is None or len(train) < seq_len + horizon:
            # Small: pad-pre for context, but skip if too short
            if train is not None and len(train) > horizon + 5:
                ctx = np.pad(train, (seq_len - len(train), 0)) if len(train) < seq_len else train[-seq_len:]
                fut = s[-horizon:]  # that's the test — skip to avoid leakage
            continue
        # Collect sliding windows: end_ctx positions in [seq_len .. len(train)-horizon]
        valid_ends = np.arange(seq_len, len(train) - horizon + 1, stride)
        if len(valid_ends) > max_per_series:
            valid_ends = np.random.choice(valid_ends, max_per_series, replace=False)
        for e in valid_ends:
            ctx = train[e - seq_len:e]
            fut = train[e:e + horizon]
            Xs.append(ctx); Ys.append(fut)
    if not Xs:
        return None, None
    return np.stack(Xs).astype(np.float32), np.stack(Ys).astype(np.float32)


def ft_m4(model, X, Y, horizon, seq_len, epochs=3, lr=1e-4, head_lr_mult=10, batch_size=64):
    """FT model on M4 windows with MSE."""
    model.train()
    enc_params, other = [], []
    for n, p in model.named_parameters():
        (enc_params if n.startswith('encoder') else other).append(p)
    opt = optim.AdamW([
        {'params': enc_params, 'lr': lr},
        {'params': other, 'lr': lr * head_lr_mult},
    ], weight_decay=1e-4)
    N = len(X)
    qt = 1.0 + torch.arange(horizon, device=DEVICE, dtype=torch.float32) / seq_len
    for ep in range(epochs):
        idx = np.random.permutation(N)
        for i in range(0, N, batch_size):
            bi = idx[i:i + batch_size]
            bx = torch.from_numpy(X[bi]).to(DEVICE)
            by = torch.from_numpy(Y[bi]).to(DEVICE)
            # Per-sample normalize context (contextual)
            m = bx.mean(-1, keepdim=True)
            s = bx.std(-1, keepdim=True).clamp(min=1e-6)
            bx_n = ((bx - m) / s).clamp(-10, 10)
            by_n = ((by - m) / s).clamp(-10, 10)
            qt_b = qt.unsqueeze(0).expand(len(bi), -1)
            pred = model.forward_train(bx_n, qt_b)
            loss = F.mse_loss(pred, by_n)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()


@torch.no_grad()
def eval_m4(model, series_list, horizon, seq_len):
    model.eval()
    smapes = []
    for s in series_list:
        if len(s) <= horizon + 5:
            continue
        train = s[:-horizon]
        test = s[-horizon:]
        ctx = train[-seq_len:] if len(train) >= seq_len else np.pad(train, (seq_len - len(train), 0))
        m, sd = ctx.mean(), ctx.std()
        if sd < 1e-6: sd = 1.0
        ctx_n = np.clip((ctx - m) / sd, -10, 10)
        ctx_t = torch.tensor(ctx_n, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        qt = (1.0 + torch.arange(horizon, device=DEVICE, dtype=torch.float32) / seq_len).unsqueeze(0)
        pred_n = model.forward_train(ctx_t, qt).cpu().numpy()[0]
        pred = pred_n * sd + m
        sv = smape(test, pred)
        if not (np.isnan(sv) or np.isinf(sv)):
            smapes.append(sv)
    return float(np.mean(smapes)) if smapes else float('nan'), len(smapes)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--tag', required=True)
    p.add_argument('--seq_len', type=int, default=512)
    p.add_argument('--ft_epochs', type=int, default=3)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--max_series', type=int, default=2000)
    p.add_argument('--windows_per_series', type=int, default=5)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--model_type', type=str, default='varlen', choices=['varlen', 'decomp'])
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

    print('=' * 70)
    print(f'M4 FT: {args.ckpt}  ft_epochs={args.ft_epochs}  lr={args.lr}')
    print('=' * 70)

    state = torch.load(args.ckpt, map_location=DEVICE, weights_only=True)
    results = {}
    for freq, cfg in M4_CONFIGS.items():
        t0 = time.time()
        # Adjust seq_len if smaller than 192 (short M4 series can be tiny)
        seq_len_freq = min(args.seq_len, 192 if cfg['horizon'] < 20 else args.seq_len)
        seq_len_freq = max(192, seq_len_freq)
        seq_len_freq = (seq_len_freq // 16) * 16
        try:
            series = load_m4_series(cfg['dir'], max_series=args.max_series)
        except Exception as e:
            print(f'  {freq}: LOAD FAIL {e}')
            continue
        if not series:
            print(f'  {freq}: no series'); continue

        X, Y = build_windows(series, cfg['horizon'], seq_len_freq,
                             max_per_series=args.windows_per_series)
        model = build_model(args).to(DEVICE)
        model.load_state_dict(state)
        if X is not None and len(X) > 0:
            ft_m4(model, X, Y, cfg['horizon'], seq_len_freq,
                  epochs=args.ft_epochs, lr=args.lr,
                  batch_size=args.batch_size)
        sm, n = eval_m4(model, series, cfg['horizon'], seq_len_freq)
        results[freq] = {'sMAPE': sm, 'n_series': n, 'horizon': cfg['horizon']}
        print(f'  {freq:<10}: sMAPE={sm:.3f}  (n={n}, ftwin={0 if X is None else len(X)}, {time.time()-t0:.0f}s)')
        del model; torch.cuda.empty_cache()

    if results:
        avg = np.mean([v['sMAPE'] for v in results.values()])
        results['Average'] = float(avg)
        print(f'\nAverage sMAPE: {avg:.3f}')

    os.makedirs('results', exist_ok=True)
    with open(f'results/{args.tag}_m4_ft.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Saved: results/{args.tag}_m4_ft.json')


if __name__ == '__main__':
    main()
