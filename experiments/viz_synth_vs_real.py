"""
Compare synthetic generated data vs real ETT/Weather.
Answers: is our data distribution matching real patterns?
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from experiments.exp_lotsa_scaling import SyntheticGapFiller

SEQ_LEN = 720
N_SAMPLES = 5

DATASETS = {
    'ETTh1':   ('./dataset/ETT-small/ETTh1.csv', ['HUFL','HULL','MUFL','MULL','LUFL','LULL','OT']),
    'ETTh2':   ('./dataset/ETT-small/ETTh2.csv', ['HUFL','HULL','MUFL','MULL','LUFL','LULL','OT']),
    'ETTm1':   ('./dataset/ETT-small/ETTm1.csv', ['HUFL','HULL','MUFL','MULL','LUFL','LULL','OT']),
    'ETTm2':   ('./dataset/ETT-small/ETTm2.csv', ['HUFL','HULL','MUFL','MULL','LUFL','LULL','OT']),
}

SYNTH_DOMAINS = {
    'ett_h1': 'ETTh1',
    'ett_h2': 'ETTh2',
    'ett_m1': 'ETTm1',
    'ett_m2': 'ETTm2',
}


def load_real(ds_key, n_samples=5):
    """Extract n_samples windows of length SEQ_LEN from real data, normalized."""
    path, cols = DATASETS[ds_key]
    df = pd.read_csv(path)
    col = cols[0]  # HUFL or similar
    data = df[col].values.astype(np.float32)
    windows = []
    rng = np.random.RandomState(42)
    for _ in range(n_samples):
        start = rng.randint(0, len(data) - SEQ_LEN)
        w = data[start:start + SEQ_LEN]
        # normalize same as training
        m, s = w.mean(), w.std().clip(min=1e-6)
        w = np.clip((w - m) / s, -10, 10)
        windows.append(w)
    return windows


def gen_synth(domain, n_samples=5):
    """Generate n_samples windows of a specific domain."""
    # Monkey-patch: fix domain
    import importlib
    from experiments import exp_lotsa_scaling
    importlib.reload(exp_lotsa_scaling)
    gen_cls = exp_lotsa_scaling.SyntheticGapFiller
    # Use internal methods directly
    windows = []
    rng = np.random.RandomState(42)
    for i in range(n_samples):
        n = SEQ_LEN
        t = np.linspace(0, 2, n)
        if domain.startswith('ett_'):
            variant = domain.split('_')[1]
            y = gen_cls._gen_ett(rng, n, t, variant=variant)
        elif domain.startswith('weather_'):
            variant = domain.split('_')[1]
            y = gen_cls._gen_weather(rng, n, t, variant=variant)
        else:
            continue
        m, s = y.mean(), y.std().clip(min=1e-6)
        y = np.clip((y - m) / s, -10, 10).astype(np.float32)
        windows.append(y)
    return windows


def main():
    n_col = N_SAMPLES
    n_row = len(SYNTH_DOMAINS) * 2
    fig, axes = plt.subplots(n_row, n_col, figsize=(3.5 * n_col, 2 * n_row), squeeze=False)

    for row_idx, (synth_key, real_key) in enumerate(SYNTH_DOMAINS.items()):
        print(f'{synth_key} vs {real_key}')
        real_ws = load_real(real_key, N_SAMPLES)
        synth_ws = gen_synth(synth_key, N_SAMPLES)

        for c in range(N_SAMPLES):
            ax_r = axes[2 * row_idx, c]
            ax_s = axes[2 * row_idx + 1, c]
            ax_r.plot(real_ws[c], 'b-', linewidth=0.6)
            ax_s.plot(synth_ws[c], 'r-', linewidth=0.6)
            ax_r.set_title(f'REAL {real_key} #{c}', fontsize=9)
            ax_s.set_title(f'SYNTH {synth_key} #{c}', fontsize=9)
            ax_r.set_ylim(-4, 4); ax_s.set_ylim(-4, 4)
            ax_r.grid(alpha=0.3); ax_s.grid(alpha=0.3)
            ax_r.tick_params(labelsize=7); ax_s.tick_params(labelsize=7)

    plt.suptitle('Real vs Synth comparison (rows alternate REAL / SYNTH)', fontsize=12)
    plt.tight_layout()
    os.makedirs('results/figures/failure_analysis', exist_ok=True)
    out = 'results/figures/failure_analysis/synth_vs_real.png'
    plt.savefig(out, dpi=90, bbox_inches='tight')
    print(f'Saved {out}')


if __name__ == '__main__':
    main()