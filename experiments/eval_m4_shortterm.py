"""
M4 Short-term Forecasting Evaluation

Models to evaluate:
  1. 32.5M seq96 (original best)
  2. 32.5M seq384 (long context)

Metrics: SMAPE, MASE, OWA
Frequencies: Yearly(h=6), Quarterly(h=8), Monthly(h=18), Weekly(h=13), Daily(h=14), Hourly(h=48)

Usage:
  CUDA_VISIBLE_DEVICES=X python experiments/eval_m4_shortterm.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch, torch.nn.functional as F
import numpy as np, time

DEVICE = torch.device(os.environ.get('CUDA_DEV', 'cuda'))

M4_DIR = './dataset/time_series_pile/forecasting/monash'
M4_CONFIGS = {
    'Yearly':    {'file': 'm4_yearly_dataset.tsf',    'horizon': 6,  'freq': 1},
    'Quarterly': {'file': 'm4_quarterly_dataset.tsf',  'horizon': 8,  'freq': 4},
    'Monthly':   {'file': 'm4_monthly_dataset.tsf',    'horizon': 18, 'freq': 12},
    'Weekly':    {'file': 'm4_weekly_dataset.tsf',      'horizon': 13, 'freq': 1},
    'Daily':     {'file': 'm4_daily_dataset.tsf',       'horizon': 14, 'freq': 1},
    'Hourly':    {'file': 'm4_hourly_dataset.tsf',      'horizon': 48, 'freq': 1},
}


# ============================================================
# Metrics
# ============================================================
def smape(y_true, y_pred):
    """Symmetric Mean Absolute Percentage Error."""
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    denom = np.where(denom == 0, 1.0, denom)
    return np.mean(np.abs(y_true - y_pred) / denom) * 100


def mase(y_true, y_pred, y_train, seasonality=1):
    """Mean Absolute Scaled Error."""
    n = len(y_train)
    if n <= seasonality:
        naive_mae = np.mean(np.abs(np.diff(y_train)))
    else:
        naive_mae = np.mean(np.abs(y_train[seasonality:] - y_train[:-seasonality]))
    if naive_mae == 0:
        naive_mae = 1.0
    return np.mean(np.abs(y_true - y_pred)) / naive_mae


# ============================================================
# Load M4 data (TSF format)
# ============================================================
def load_m4_tsf(filepath, max_series=None):
    """Parse M4 .tsf → list of (train_series, test_horizon) numpy arrays."""
    series_list = []
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        in_data = False
        for line in f:
            line = line.strip()
            if line.startswith('@data'):
                in_data = True
                continue
            if not in_data or not line or line.startswith('@') or line.startswith('#'):
                continue
            parts = line.split(':')
            if len(parts) >= 2:
                values_str = parts[-1].strip()
            else:
                values_str = line
            try:
                values = [float(v) for v in values_str.split(',') if v.strip() and v.strip() != '?']
                if len(values) > 5:
                    series_list.append(np.array(values, dtype=np.float32))
            except:
                continue
            if max_series and len(series_list) >= max_series:
                break
    return series_list


# ============================================================
# Forecast with our model
# ============================================================
@torch.no_grad()
def forecast_series(model, series, horizon, seq_len):
    """
    series: 1D numpy array (full history)
    horizon: number of steps to forecast
    Returns: prediction array [horizon]
    """
    # Use last seq_len as context
    if len(series) >= seq_len:
        ctx = series[-seq_len:]
    else:
        ctx = np.pad(series, (seq_len - len(series), 0), mode='constant')

    # Normalize
    m = ctx.mean()
    s = ctx.std()
    if s < 1e-6:
        s = 1.0
    ctx_n = np.clip((ctx - m) / s, -10, 10)

    ctx_t = torch.tensor(ctx_n, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    # Query future: t = 1 + i/seq_len for i in [0, horizon)
    t = 1.0 + torch.arange(horizon, device=DEVICE, dtype=torch.float32) / seq_len
    t = t.unsqueeze(0)

    pred_n = model.forward_train(ctx_t, t).cpu().numpy()[0]
    pred = pred_n * s + m
    return pred


# ============================================================
# Evaluate one frequency
# ============================================================
def evaluate_frequency(model, freq_name, config, seq_len, max_series=2000):
    filepath = os.path.join(M4_DIR, config['file'])
    horizon = config['horizon']
    seasonality = config['freq']
    if seasonality == 0:
        seasonality = 1

    if not os.path.exists(filepath):
        print(f'  {freq_name}: FILE NOT FOUND ({filepath})')
        return None

    series_list = load_m4_tsf(filepath, max_series=max_series + horizon)

    smapes, mases = [], []
    count = 0
    for series in series_list:
        if len(series) <= horizon + 5:
            continue
        # Split: last `horizon` as test, rest as train
        train = series[:-horizon]
        test = series[-horizon:]

        # Forecast
        pred = forecast_series(model, train, horizon, seq_len)

        # Compute metrics
        s = smape(test, pred)
        m = mase(test, pred, train, seasonality=max(1, seasonality))

        if not (np.isnan(s) or np.isinf(s) or np.isnan(m) or np.isinf(m)):
            smapes.append(s)
            mases.append(m)
            count += 1

        if count >= max_series:
            break

    if not smapes:
        return None

    avg_smape = np.mean(smapes)
    avg_mase = np.mean(mases)
    avg_owa = (avg_smape / 13.564 + avg_mase / 1.912) / 2  # M4 naive2 baseline

    return {'smape': avg_smape, 'mase': avg_mase, 'owa': avg_owa, 'count': count}


# ============================================================
# Main
# ============================================================
def eval_model(model_name, model, seq_len):
    print(f'\n{"="*60}')
    print(f'M4 Short-term: {model_name} (SEQ={seq_len})')
    print(f'{"="*60}')

    results = {}
    for freq_name, config in M4_CONFIGS.items():
        t0 = time.time()
        r = evaluate_frequency(model, freq_name, config, seq_len, max_series=2000)
        elapsed = time.time() - t0
        if r:
            print(f'  {freq_name:<12}: SMAPE={r["smape"]:.3f}  MASE={r["mase"]:.3f}  '
                  f'OWA={r["owa"]:.3f}  (n={r["count"]}, {elapsed:.1f}s)')
            results[freq_name] = r
        else:
            print(f'  {freq_name:<12}: FAILED')

    # Average
    if results:
        avg_smape = np.mean([r['smape'] for r in results.values()])
        avg_mase = np.mean([r['mase'] for r in results.values()])
        avg_owa = np.mean([r['owa'] for r in results.values()])
        print(f'\n  {"Average":<12}: SMAPE={avg_smape:.3f}  MASE={avg_mase:.3f}  OWA={avg_owa:.3f}')

    return results


if __name__ == '__main__':
    print('='*60)
    print('M4 Short-term Forecasting Evaluation')
    print('='*60)

    # FeDaL reference
    fedal_ref = {
        'Yearly': {'smape': 13.102, 'mase': 2.812, 'owa': 0.748},
        'Quarterly': {'smape': 9.808, 'mase': 1.112, 'owa': 0.847},
        'Monthly': {'smape': 12.124, 'mase': 0.898, 'owa': 0.820},
        'Others': {'smape': 4.508, 'mase': 2.890, 'owa': 0.973},
        'Average': {'smape': 11.412, 'mase': 1.489, 'owa': 0.818},
    }

    all_results = {}

    # Model 1: 32.5M seq96 (original best)
    from experiments.exp_full_scale_train import FullScaleModel
    model1 = FullScaleModel().to(DEVICE)
    model1.load_state_dict(torch.load('checkpoints/full_scale_run.pth', map_location=DEVICE))
    model1.eval()
    print(f'Loaded 32.5M seq96: {sum(p.numel() for p in model1.parameters())/1e6:.1f}M')
    all_results['32.5M_seq96'] = eval_model('32.5M seq96', model1, 96)
    del model1; torch.cuda.empty_cache()

    # Model 2: 32.5M seq384
    from experiments.exp_32M_longctx import Model32MLongCtx
    model2 = Model32MLongCtx(384).to(DEVICE)
    model2.load_state_dict(torch.load('checkpoints/32M_seq384.pth', map_location=DEVICE))
    model2.eval()
    print(f'\nLoaded 32.5M seq384: {sum(p.numel() for p in model2.parameters())/1e6:.1f}M')
    all_results['32.5M_seq384'] = eval_model('32.5M seq384', model2, 384)
    del model2; torch.cuda.empty_cache()

    # Summary comparison
    print('\n' + '='*60)
    print('COMPARISON vs FeDaL')
    print('='*60)
    print(f'{"Freq":<12} {"32.5M_s96 SMAPE":<18} {"32.5M_s384 SMAPE":<18} {"FeDaL SMAPE":<15}')
    print('-'*60)
    for freq in ['Yearly', 'Quarterly', 'Monthly']:
        s96 = all_results.get('32.5M_seq96', {}).get(freq, {})
        s384 = all_results.get('32.5M_seq384', {}).get(freq, {})
        fed = fedal_ref.get(freq, {})
        print(f'{freq:<12} {s96.get("smape", "-"):<18.3f} {s384.get("smape", "-"):<18.3f} {fed.get("smape", "-"):<15.3f}')

    print('\nFeDaL Average: SMAPE=11.412, MASE=1.489, OWA=0.818')
    print('='*60)
