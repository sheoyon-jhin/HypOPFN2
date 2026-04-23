"""
M4 Short-term Forecasting eval (FeDaL/MOMENT protocol).

Metric: sMAPE (standard M4 metric)
Horizons (from M4 official):
  - Yearly (h=6), Quarterly (h=8), Monthly (h=18)
  - Weekly (h=13), Daily (h=14), Hourly (h=48)

Dataset: loaded via GluonTS (auto-downloaded from M4 source)

Reference (FeDaL Table 7):
  - Monthly: sMAPE = 12.12
  - Our previous best (HANDOVER): 8.81 (SOTA!)

Usage:
  CUDA_VISIBLE_DEVICES=0 python experiments/eval_m4.py \
      --ckpt checkpoints/overnight_seq720_s50.pth --seq_len 720 \
      --tag overnight_seq720_m4
"""
import sys, os, json, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn.functional as F
import numpy as np

from experiments.exp_lotsa_scaling import OperatorModel

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

M4_CONFIGS = {
    'Yearly':    {'name': 'm4_yearly',    'horizon': 6,  'seasonality': 1},
    'Quarterly': {'name': 'm4_quarterly', 'horizon': 8,  'seasonality': 4},
    'Monthly':   {'name': 'm4_monthly',   'horizon': 18, 'seasonality': 12},
    'Weekly':    {'name': 'm4_weekly',    'horizon': 13, 'seasonality': 1},
    'Daily':     {'name': 'm4_daily',     'horizon': 14, 'seasonality': 1},
    'Hourly':    {'name': 'm4_hourly',    'horizon': 48, 'seasonality': 24},
}

# Reference values (Table 7 from FeDaL)
REFERENCE = {
    'Yearly':    13.08,
    'Quarterly': 9.81,
    'Monthly':   12.12,
    'Weekly':    7.86,
    'Daily':     3.16,
    'Hourly':    12.40,
}


def smape(y_true, y_pred):
    """Symmetric Mean Absolute Percentage Error (×100)."""
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    denom = np.where(denom == 0, 1.0, denom)
    return np.mean(np.abs(y_true - y_pred) / denom) * 100


@torch.no_grad()
def forecast_series(model, series, horizon, seq_len):
    """Given 1D series, predict `horizon` steps ahead. Per-series normalization."""
    if len(series) >= seq_len:
        ctx = series[-seq_len:]
    else:
        pad = np.zeros(seq_len - len(series), dtype=np.float32)
        ctx = np.concatenate([pad, series])
    ctx = torch.tensor(ctx, dtype=torch.float32, device=DEVICE).unsqueeze(0)

    m = ctx.mean(1, keepdim=True)
    s = ctx.std(1, keepdim=True).clamp(min=1e-6)
    x_n = ((ctx - m) / s).clamp(-10, 10)

    # Direct prediction — no rolling (operator learning advantage)
    pred = model.forecast(x_n, n=horizon)  # (1, horizon)
    pred = (pred * s + m).squeeze(0).cpu().numpy()
    return pred


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--decomp', type=int, default=0)
    p.add_argument('--seq_len', type=int, default=720)
    p.add_argument('--tag', type=str, required=True)
    p.add_argument('--max_series', type=int, default=0,
                   help='Limit series per freq for quick test (0=all)')
    args = p.parse_args()

    print('=' * 70)
    print(f'M4 SHORT-TERM EVAL: {args.ckpt}')
    print(f'  seq_len={args.seq_len}')
    print('=' * 70)

    model = OperatorModel(seq_len=args.seq_len, use_latent_decomp=bool(args.decomp)).to(DEVICE)
    state = torch.load(args.ckpt, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    from gluonts.dataset.repository import get_dataset

    results = {}
    for freq_name, cfg in M4_CONFIGS.items():
        print(f'\n--- {freq_name} (h={cfg["horizon"]}) ---')
        try:
            ds = get_dataset(cfg['name'], regenerate=False)
            all_series = list(ds.train)  # we'll split train/test manually
            # Actually M4 train in gluonts is the whole history; test separately
            test_series = list(ds.test)

            smapes = []
            n_eval = min(args.max_series, len(test_series)) if args.max_series else len(test_series)
            for i, ts in enumerate(test_series[:n_eval]):
                full = np.array(ts['target'], dtype=np.float32)
                # Last `horizon` values are the ground-truth test target
                history = full[:-cfg['horizon']]
                y_true = full[-cfg['horizon']:]
                if len(history) < 10:  # skip very short series
                    continue
                y_pred = forecast_series(model, history, cfg['horizon'], args.seq_len)
                smapes.append(smape(y_true, y_pred))

                if (i + 1) % 500 == 0:
                    print(f'  [{i+1}/{n_eval}] running sMAPE: {np.mean(smapes):.2f}')

            avg_smape = float(np.mean(smapes))
            ref = REFERENCE.get(freq_name, 0)
            gap = (avg_smape - ref) / ref * 100 if ref else 0
            print(f'{freq_name}: sMAPE = {avg_smape:.2f}  (FeDaL: {ref:.2f}, Δ: {gap:+.1f}%)  n={len(smapes)}')
            results[freq_name] = {'sMAPE': avg_smape, 'n_series': len(smapes)}
        except Exception as e:
            print(f'{freq_name}: ERROR ({type(e).__name__}: {str(e)[:150]})')

    # Summary
    print('\n' + '=' * 70)
    print(f'{"Freq":<12} {"Ours sMAPE":>10} {"FeDaL":>10} {"Δ%":>8}')
    print('-' * 70)
    for freq_name in M4_CONFIGS:
        r = results.get(freq_name)
        ref = REFERENCE.get(freq_name, 0)
        if r:
            gap = (r['sMAPE'] - ref) / ref * 100
            print(f'{freq_name:<12} {r["sMAPE"]:>10.2f} {ref:>10.2f} {gap:>+7.1f}%')

    os.makedirs('results', exist_ok=True)
    with open(f'results/{args.tag}.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved: results/{args.tag}.json')


if __name__ == '__main__':
    main()
