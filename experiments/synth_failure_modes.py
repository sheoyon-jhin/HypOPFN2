"""
Failure-mode synthetic generators.

Add these domains to SyntheticGapFiller.DOMAIN_WEIGHTS to target known failure modes:
  - nonstationary_burst: level shifts + bursts (Weather-style abrupt events)
  - multiscale: daily + weekly + monthly simultaneous
  - short_strong_trend: short series with strong trend (M4 Yearly-style)
  - highfreq_noise_trend: high freq on slow trend (ETTm1-style)
  - outlier_heavy: spike outliers
  - heteroskedastic: time-varying variance

Integration: after running analyze_failures.py, pick the clusters that match
these patterns and bump their weights in the DOMAIN_WEIGHTS dict.

This is a MODULE — import from other scripts or copy generators into
exp_lotsa_scaling.py's SyntheticGapFiller class.
"""
import numpy as np


def _ar1(rng, n, phi=0.5, sigma=1.0):
    out = np.zeros(n)
    out[0] = rng.randn() * sigma
    for i in range(1, n):
        out[i] = phi * out[i - 1] + rng.randn() * sigma
    return out


def gen_nonstationary_burst(rng, n, t):
    """Level shifts + occasional bursts. Weather-like abrupt events."""
    y = _ar1(rng, n, phi=rng.uniform(0.5, 0.85), sigma=rng.uniform(0.3, 1.0))
    # 1-3 level shifts
    n_shifts = rng.randint(1, 4)
    for _ in range(n_shifts):
        pos = rng.randint(n // 8, 7 * n // 8)
        shift = rng.uniform(-3, 3)
        decay = rng.uniform(0.9, 0.99)
        for i in range(pos, n):
            y[i] += shift * (decay ** (i - pos))
    # 1-3 bursts (wide-spread impulses)
    n_bursts = rng.randint(1, 4)
    for _ in range(n_bursts):
        pos = rng.randint(0, n)
        width = rng.randint(3, max(5, n // 30))
        height = rng.choice([-1, 1]) * rng.exponential(3.0)
        burst = height * np.exp(-0.5 * ((np.arange(n) - pos) / width) ** 2)
        y += burst
    return y


def gen_multiscale(rng, n, t):
    """Daily + weekly + monthly patterns simultaneously. Dense multi-periodic."""
    # Daily
    y = rng.uniform(3.0, 8.0) * np.sin(2 * np.pi * rng.uniform(0.9, 1.1) * t + rng.uniform(0, 2*np.pi))
    # Weekly (7x slower)
    y += rng.uniform(1.0, 4.0) * np.sin(2 * np.pi * rng.uniform(0.9, 1.1) / 7 * t + rng.uniform(0, 2*np.pi))
    # Monthly (30x slower)
    y += rng.uniform(0.5, 3.0) * np.sin(2 * np.pi * rng.uniform(0.9, 1.1) / 30 * t + rng.uniform(0, 2*np.pi))
    # Optional yearly (365x)
    if rng.rand() < 0.5:
        y += rng.uniform(0.2, 2.0) * np.sin(2 * np.pi * rng.uniform(0.9, 1.1) / 365 * t)
    # Noise
    y += _ar1(rng, n, phi=rng.uniform(0.3, 0.7), sigma=rng.uniform(0.2, 0.8))
    return y


def gen_short_strong_trend(rng, n, t):
    """Simulates short M4 Yearly-like: strong trend dominates, minimal seasonal."""
    # Emphasize the TAIL of the sequence being the "real" part,
    # earlier part may be flat/small (simulating padding-like behavior learned naturally)
    trend_type = rng.choice(['linear', 'poly2', 'exp', 'piecewise'])
    if trend_type == 'linear':
        y = rng.uniform(-3, 3) * t
    elif trend_type == 'poly2':
        y = rng.uniform(-1, 1) * t + rng.uniform(-0.5, 0.5) * t**2
    elif trend_type == 'exp':
        y = rng.uniform(0.5, 2.0) * np.exp(rng.uniform(-1, 1) * t)
    else:
        # Piecewise linear (2-3 segments)
        n_seg = rng.randint(2, 4)
        breaks = sorted([rng.uniform(0.2, 1.8) for _ in range(n_seg - 1)])
        slopes = rng.uniform(-2, 2, n_seg)
        y = np.zeros(n); prev_bp = 0.0; prev_val = 0.0
        for i, bp in enumerate(list(breaks) + [2.0]):
            mask = (t >= prev_bp) & (t < bp)
            y[mask] = prev_val + slopes[i] * (t[mask] - prev_bp)
            prev_val += slopes[i] * (bp - prev_bp); prev_bp = bp
    y += rng.randn(n) * rng.uniform(0.1, 0.5)
    return y


def gen_highfreq_noise_trend(rng, n, t):
    """High-freq noise on top of slow trend. ETTm1-style (15-min resolution)."""
    # Fast oscillations
    hf_freq = rng.uniform(15, 40)
    y = rng.uniform(1, 3) * np.sin(2 * np.pi * hf_freq * t + rng.uniform(0, 2*np.pi))
    # Daily slow envelope
    y += rng.uniform(2, 5) * np.sin(2 * np.pi * rng.uniform(0.9, 1.1) * t)
    # Long trend
    y += rng.uniform(-0.5, 0.5) * t
    # Heavy noise
    y += _ar1(rng, n, phi=rng.uniform(0.1, 0.4), sigma=rng.uniform(0.5, 1.5))
    return y


def gen_outlier_heavy(rng, n, t):
    """Base signal + many spike outliers."""
    y = rng.uniform(0.3, 1.5) * np.sin(2 * np.pi * rng.uniform(0.5, 3.0) * t)
    y += rng.randn(n) * rng.uniform(0.2, 0.5)
    # Many spikes (5-15)
    n_spikes = rng.randint(5, 15)
    for _ in range(n_spikes):
        pos = rng.randint(0, n)
        height = rng.choice([-1, 1]) * rng.uniform(3, 8)
        y[pos] += height
    return y


def gen_heteroskedastic(rng, n, t):
    """Time-varying variance. Vol(t) modulates AR(1) noise amplitude."""
    # Base sinusoid
    base = rng.uniform(0.5, 2.0) * np.sin(2 * np.pi * rng.uniform(0.5, 3.0) * t)
    # Time-varying sigma (slow oscillation)
    sigma_t = 0.2 + 0.8 * (0.5 + 0.5 * np.sin(2 * np.pi * rng.uniform(0.3, 1.0) * t))
    noise = _ar1(rng, n, phi=0.5, sigma=1.0) * sigma_t
    return base + noise


# Export as a dict for easy import
FAILURE_MODE_GENERATORS = {
    'nonstationary_burst': gen_nonstationary_burst,
    'multiscale': gen_multiscale,
    'short_strong_trend': gen_short_strong_trend,
    'highfreq_noise_trend': gen_highfreq_noise_trend,
    'outlier_heavy': gen_outlier_heavy,
    'heteroskedastic': gen_heteroskedastic,
}


if __name__ == '__main__':
    # Smoke test: visualize each generator
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    rng = np.random.RandomState(42)
    n = 500
    t = np.linspace(0, 2, n)

    fig, axes = plt.subplots(3, 2, figsize=(14, 8))
    axes = axes.flatten()
    for ax, (name, gen) in zip(axes, FAILURE_MODE_GENERATORS.items()):
        y = gen(rng, n, t)
        ax.plot(y, linewidth=0.7)
        ax.set_title(name)
        ax.grid(alpha=0.3)
    plt.tight_layout()
    import os; os.makedirs('results/figures', exist_ok=True)
    plt.savefig('results/figures/failure_mode_generators.png', dpi=110)
    print('Saved: results/figures/failure_mode_generators.png')
