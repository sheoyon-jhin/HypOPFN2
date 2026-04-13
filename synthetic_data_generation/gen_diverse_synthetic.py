"""
Diverse Synthetic Data Generator — LOTSA-level diversity를 synthetic으로

LOTSA 174 datasets의 패턴을 분석해서 synthetic으로 모방:
  1. Traffic patterns (daily+weekly cycles, rush hour peaks)
  2. Energy patterns (load curves, solar/wind intermittency)
  3. Financial (random walk, volatility clustering, regime change)
  4. Weather/Climate (multi-seasonal, slow trends)
  5. Medical (periodic + anomalous events)
  6. Retail (weekly seasonality + holiday spikes)
  7. Step/regime changes (distribution shift)
  8. Chirp/non-stationary (frequency drift)
  9. Pure GP (various kernels)
  10. Compositional (multiple patterns additive)

Target: 500K-1M series, 다양한 lengths (96-2048)

Usage: python synthetic_data_generation/gen_diverse_synthetic.py --n_series 500000
"""
import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
import os, time, argparse
from multiprocessing import Pool, cpu_count


# ============================================================
# Base generators
# ============================================================
def gen_trend(n, complexity='simple'):
    t = np.linspace(0, 1, n)
    if complexity == 'simple':
        slope = np.random.uniform(-3, 3)
        return slope * t + np.random.uniform(-1, 1)
    elif complexity == 'quadratic':
        a, b = np.random.uniform(-2, 2, 2)
        return a * t**2 + b * t
    elif complexity == 'piecewise':
        n_breaks = np.random.randint(1, 4)
        breaks = sorted(np.random.uniform(0.1, 0.9, n_breaks))
        y = np.zeros(n)
        prev = 0
        for br in breaks + [1.0]:
            idx = int(br * n)
            slope = np.random.uniform(-3, 3)
            y[prev:idx] = slope * np.linspace(0, 1, idx - prev) + (y[prev-1] if prev > 0 else 0)
            prev = idx
        return y


def gen_seasonal(n, n_harmonics=None):
    t = np.linspace(0, 2 * np.pi * np.random.uniform(1, 20), n)
    if n_harmonics is None:
        n_harmonics = np.random.randint(1, 5)
    y = np.zeros(n)
    for _ in range(n_harmonics):
        amp = np.random.uniform(0.3, 2.0)
        freq = np.random.uniform(0.5, 10)
        phase = np.random.uniform(0, 2 * np.pi)
        y += amp * np.sin(freq * t + phase)
    return y


def gen_traffic(n):
    """Daily + weekly cycle with rush hour peaks."""
    t = np.linspace(0, n / 96, n)  # assume 15-min intervals
    # Daily cycle
    daily = 0.5 * np.sin(2 * np.pi * t - np.pi/2) + 0.5
    # Rush hour peaks (morning 8am, evening 6pm)
    morning = np.exp(-50 * (np.mod(t, 1) - 0.33)**2) * np.random.uniform(0.5, 1.5)
    evening = np.exp(-50 * (np.mod(t, 1) - 0.75)**2) * np.random.uniform(0.5, 1.5)
    # Weekly pattern (weekday vs weekend)
    weekly = 0.3 * np.sin(2 * np.pi * t / 7)
    noise = np.random.randn(n) * 0.1
    return daily + morning + evening + weekly + noise


def gen_energy(n):
    """Energy load curve with solar/temperature dependency."""
    t = np.linspace(0, n / 24, n)
    # Base load
    base = np.random.uniform(50, 200)
    # Temperature-driven (seasonal)
    temp = np.sin(2 * np.pi * t / 365) * np.random.uniform(10, 30)
    # Daily pattern
    daily = np.sin(2 * np.pi * t - np.pi/3) * np.random.uniform(5, 20)
    # Solar (only during day)
    solar = np.maximum(0, np.sin(2 * np.pi * t)) * np.random.uniform(0, 30)
    noise = np.random.randn(n) * np.random.uniform(1, 5)
    return base + temp + daily - solar + noise


def gen_financial(n):
    """Random walk with volatility clustering (simplified GARCH-like)."""
    returns = np.zeros(n)
    vol = np.random.uniform(0.01, 0.05)
    for i in range(1, n):
        # Volatility clustering
        vol = 0.9 * vol + 0.1 * np.abs(returns[i-1]) + 0.001
        returns[i] = np.random.randn() * vol + np.random.uniform(-0.001, 0.001)
    price = np.cumsum(returns) + np.random.uniform(10, 1000)
    return price


def gen_weather(n):
    """Multi-seasonal weather pattern."""
    t = np.linspace(0, n / (24 * 365), n)
    # Annual cycle
    annual = np.sin(2 * np.pi * t) * np.random.uniform(10, 25)
    # Daily cycle
    daily = np.sin(2 * np.pi * t * 365) * np.random.uniform(3, 8)
    # Long-term trend (climate)
    trend = t * np.random.uniform(-0.5, 2)
    # Weather noise (autocorrelated)
    noise = np.zeros(n)
    for i in range(1, n):
        noise[i] = 0.7 * noise[i-1] + np.random.randn() * np.random.uniform(1, 3)
    return annual + daily + trend + noise + np.random.uniform(-10, 30)


def gen_medical(n):
    """Periodic vital signs with occasional anomalies."""
    t = np.linspace(0, n / 60, n)
    # Heart rate-like (60-100 bpm)
    base = np.random.uniform(60, 100)
    # Respiratory modulation
    resp = np.sin(2 * np.pi * t * np.random.uniform(12, 20) / 60) * np.random.uniform(1, 5)
    # Circadian
    circadian = np.sin(2 * np.pi * t / 24) * np.random.uniform(2, 8)
    # Occasional anomalies (spikes)
    anomalies = np.zeros(n)
    n_anom = np.random.randint(0, 5)
    for _ in range(n_anom):
        pos = np.random.randint(0, n)
        width = np.random.randint(5, 30)
        amp = np.random.uniform(10, 50) * (1 if np.random.rand() > 0.5 else -1)
        start, end = max(0, pos-width), min(n, pos+width)
        anomalies[start:end] = amp * np.exp(-0.5 * np.linspace(-2, 2, end-start)**2)
    noise = np.random.randn(n) * np.random.uniform(0.5, 2)
    return base + resp + circadian + anomalies + noise


def gen_retail(n):
    """Weekly seasonality + holiday spikes + trend."""
    t = np.linspace(0, n / 7, n)
    # Weekly pattern (high on weekends)
    weekly = np.sin(2 * np.pi * t) * np.random.uniform(5, 20)
    # Upward trend
    trend = t * np.random.uniform(0, 5)
    # Holiday spikes (random)
    holidays = np.zeros(n)
    n_holidays = np.random.randint(1, 8)
    for _ in range(n_holidays):
        pos = np.random.randint(0, n)
        holidays[max(0, pos-3):min(n, pos+3)] = np.random.uniform(20, 100)
    noise = np.random.randn(n) * np.random.uniform(1, 5)
    return np.maximum(0, 50 + weekly + trend + holidays + noise)


def gen_step_regime(n):
    """Step function / regime changes."""
    n_regimes = np.random.randint(2, 6)
    y = np.zeros(n)
    boundaries = sorted(np.random.choice(range(10, n-10), n_regimes-1, replace=False))
    boundaries = [0] + list(boundaries) + [n]
    for i in range(len(boundaries) - 1):
        start, end = boundaries[i], boundaries[i+1]
        mean = np.random.uniform(-5, 5)
        std = np.random.uniform(0.1, 2)
        slope = np.random.uniform(-0.5, 0.5)
        t_seg = np.linspace(0, 1, end - start)
        y[start:end] = mean + slope * t_seg + np.random.randn(end - start) * std
    return y


def gen_chirp(n):
    """Chirp signal (time-varying frequency)."""
    t = np.linspace(0, 2, n)
    f0 = np.random.uniform(0.5, 5)
    f1 = np.random.uniform(f0 + 1, f0 + 10)
    amp = np.random.uniform(0.5, 2)
    phase = np.random.uniform(0, 2 * np.pi)
    f_t = f0 + (f1 - f0) * t / 2
    return amp * np.sin(2 * np.pi * f_t * t + phase)


def gen_gp(n):
    """Gaussian Process with random kernel."""
    t = np.linspace(0, 5, n).reshape(-1, 1)
    kernel_type = np.random.choice(['rbf', 'periodic', 'matern'])
    if kernel_type == 'rbf':
        l = np.random.uniform(0.1, 2.0)
        K = np.exp(-0.5 * (t - t.T)**2 / l**2)
    elif kernel_type == 'periodic':
        l = np.random.uniform(0.5, 2.0)
        p = np.random.uniform(0.5, 3.0)
        K = np.exp(-2 * np.sin(np.pi * np.abs(t - t.T) / p)**2 / l**2)
    elif kernel_type == 'matern':
        l = np.random.uniform(0.1, 2.0)
        d = np.abs(t - t.T)
        K = (1 + np.sqrt(3) * d / l) * np.exp(-np.sqrt(3) * d / l)
    K += 1e-6 * np.eye(n)
    try:
        L = np.linalg.cholesky(K)
        return (L @ np.random.randn(n)).astype(np.float32)
    except:
        return np.random.randn(n).astype(np.float32)


def gen_ou_process(n):
    """Ornstein-Uhlenbeck (mean-reverting)."""
    theta = np.random.uniform(0.1, 2.0)
    mu = np.random.uniform(-2, 2)
    sigma = np.random.uniform(0.1, 2.0)
    dt = 0.01
    y = np.zeros(n)
    y[0] = mu + np.random.randn() * sigma
    for i in range(1, n):
        y[i] = y[i-1] + theta * (mu - y[i-1]) * dt + sigma * np.sqrt(dt) * np.random.randn()
    return y


def gen_compositional(n):
    """Random composition of 2-4 base patterns."""
    components = [gen_trend, gen_seasonal, gen_step_regime, gen_chirp, gen_gp, gen_ou_process]
    n_comp = np.random.randint(2, 5)
    selected = np.random.choice(len(components), n_comp, replace=False)
    y = np.zeros(n)
    for idx in selected:
        weight = np.random.uniform(0.3, 1.5)
        try:
            comp = components[idx](n)
            if isinstance(comp, np.ndarray) and len(comp) == n:
                y += weight * comp
        except:
            pass
    return y


# All generators with domain labels
GENERATORS = {
    'traffic': gen_traffic,
    'energy': gen_energy,
    'financial': gen_financial,
    'weather': gen_weather,
    'medical': gen_medical,
    'retail': gen_retail,
    'step_regime': gen_step_regime,
    'chirp': gen_chirp,
    'gp_rbf': gen_gp,
    'ou_process': gen_ou_process,
    'trend_simple': lambda n: gen_trend(n, 'simple'),
    'trend_quad': lambda n: gen_trend(n, 'quadratic'),
    'trend_piecewise': lambda n: gen_trend(n, 'piecewise'),
    'seasonal_1': lambda n: gen_seasonal(n, 1),
    'seasonal_3': lambda n: gen_seasonal(n, 3),
    'seasonal_5': lambda n: gen_seasonal(n, 5),
    'compositional': gen_compositional,
}


# ============================================================
# Generate one series
# ============================================================
def generate_one(args):
    idx, length = args
    rng = np.random.RandomState(idx)
    np.random.seed(idx)

    gen_name = np.random.choice(list(GENERATORS.keys()))
    gen_fn = GENERATORS[gen_name]

    try:
        y = gen_fn(length)
        if not isinstance(y, np.ndarray):
            y = np.array(y)
        y = y.astype(np.float32)

        # Normalize
        s = np.std(y)
        if s > 1e-6:
            y = (y - np.mean(y)) / s
            y = np.clip(y, -10, 10)
        else:
            return None

        return {
            'series_id': idx,
            'values': y.tolist(),
            'length': len(y),
            'generator': gen_name,
        }
    except:
        return None


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_series', type=int, default=500000)
    parser.add_argument('--min_len', type=int, default=192)
    parser.add_argument('--max_len', type=int, default=2048)
    parser.add_argument('--output', type=str, default='synthetic_diverse_500k.arrow')
    parser.add_argument('--n_workers', type=int, default=8)
    args = parser.parse_args()

    print(f'Generating {args.n_series:,} diverse synthetic series...')
    print(f'  Length range: [{args.min_len}, {args.max_len}]')
    print(f'  Generators: {len(GENERATORS)} types')
    print(f'  Workers: {args.n_workers}')

    # Random lengths
    lengths = np.random.randint(args.min_len, args.max_len + 1, args.n_series)

    # Generate in parallel
    t0 = time.time()
    tasks = [(i, lengths[i]) for i in range(args.n_series)]

    results = []
    batch_size = 50000
    for batch_start in range(0, len(tasks), batch_size):
        batch = tasks[batch_start:batch_start + batch_size]
        with Pool(args.n_workers) as pool:
            batch_results = pool.map(generate_one, batch)
        batch_results = [r for r in batch_results if r is not None]
        results.extend(batch_results)
        elapsed = time.time() - t0
        print(f'  {len(results):,} / {args.n_series:,} generated ({elapsed:.0f}s)')

    print(f'\nGenerated {len(results):,} valid series in {time.time()-t0:.0f}s')

    # Count per generator
    gen_counts = {}
    for r in results:
        g = r['generator']
        gen_counts[g] = gen_counts.get(g, 0) + 1
    print('\nPer generator:')
    for g, c in sorted(gen_counts.items(), key=lambda x: -x[1]):
        print(f'  {g:<20}: {c:,}')

    # Save as Arrow
    print(f'\nSaving to {args.output}...')
    schema = pa.schema([
        ('series_id', pa.int64()),
        ('target', pa.list_(pa.float32())),
        ('length', pa.int32()),
        ('generator', pa.string()),
    ])

    ids = [r['series_id'] for r in results]
    targets = [r['values'] for r in results]
    lengths_arr = [r['length'] for r in results]
    gens = [r['generator'] for r in results]

    table = pa.table({
        'series_id': ids,
        'target': targets,
        'length': lengths_arr,
        'generator': gens,
    }, schema=schema)

    with ipc.RecordBatchFileWriter(args.output, schema) as writer:
        writer.write_table(table)

    file_size = os.path.getsize(args.output) / 1e9
    total_points = sum(lengths_arr)
    print(f'Saved: {args.output} ({file_size:.1f} GB)')
    print(f'Total time points: {total_points/1e9:.2f}B')
    print(f'LOTSA comparison: {total_points/231e9*100:.1f}% of LOTSA (231B)')


if __name__ == '__main__':
    main()
