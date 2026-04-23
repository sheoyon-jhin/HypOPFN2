"""
Visualize Fourier basis fitting on ETTh2 with different nf values.
Shows: as nf increases, fit quality improves dramatically.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

SEQ_LEN = 720

def fourier_basis(t, nf):
    t = t.reshape(-1, 1)
    f = np.arange(1, nf+1).reshape(1, -1)
    return np.hstack([t, np.sin(2*np.pi*f*t), np.cos(2*np.pi*f*t)])

def ls_fit(basis, y):
    coef, _, _, _ = np.linalg.lstsq(basis, y, rcond=None)
    pred = basis @ coef
    mse = ((pred - y)**2).mean()
    return pred, mse

def load_norm(path, start, length=SEQ_LEN):
    df = pd.read_csv(path)
    y = df.iloc[start:start+length, 1].values.astype(np.float32)
    m, s = y.mean(), y.std().clip(min=1e-6)
    return np.clip((y - m) / s, -10, 10)

def main():
    datasets = {
        'ETTm1': ('./dataset/ETT-small/ETTm1.csv', 5000, 'periodic (good)'),
        'ETTh2': ('./dataset/ETT-small/ETTh2.csv', 5000, 'noisy (bad)'),
        'ETTm2': ('./dataset/ETT-small/ETTm2.csv', 5000, 'low-var (bad)'),
        'Weather': ('./dataset/weather/weather.csv', 5000, 'trend (mixed)'),
    }
    nf_values = [8, 32, 64, 128, 256]

    fig, axes = plt.subplots(len(datasets), len(nf_values) + 1,
                             figsize=(4 * (len(nf_values) + 1), 3 * len(datasets)),
                             squeeze=False)

    for row, (ds, (path, start, desc)) in enumerate(datasets.items()):
        y = load_norm(path, start)
        t = np.linspace(0, 1, len(y))

        # Col 0: original
        ax = axes[row, 0]
        ax.plot(y, 'b-', linewidth=0.8)
        ax.set_title(f'{ds} (original)', fontsize=10, fontweight='bold')
        ax.set_ylabel(f'{ds}\n{desc}', fontsize=10, fontweight='bold')
        ax.set_ylim(-4, 4)
        ax.grid(alpha=0.3)

        # Col 1+: Fourier fit with increasing nf
        for col, nf in enumerate(nf_values):
            ax = axes[row, col + 1]
            basis = fourier_basis(t, nf)
            if basis.shape[1] >= len(y):
                ax.text(0.5, 0.5, f'nf={nf}\noverparam', ha='center', va='center',
                        transform=ax.transAxes, fontsize=12)
                ax.set_ylim(-4, 4)
                continue
            pred, mse = ls_fit(basis, y)
            ax.plot(y, 'b-', alpha=0.4, linewidth=0.6)
            ax.plot(pred, 'r-', linewidth=1.2)
            residual = y - pred
            ax.fill_between(range(len(y)), pred, y, alpha=0.15, color='orange')
            color = '#2ecc71' if mse < 0.05 else ('#f39c12' if mse < 0.2 else '#e74c3c')
            ax.set_title(f'nf={nf} (dim={basis.shape[1]})\nMSE={mse:.4f}',
                        fontsize=9, color=color, fontweight='bold')
            ax.set_ylim(-4, 4)
            ax.grid(alpha=0.3)

    fig.suptitle(
        'Fourier Basis Fit Test: blue=truth, red=best LS fit, orange=residual\n'
        'Shows: how many Fourier modes (nf) needed to represent each dataset',
        fontsize=13, fontweight='bold')
    plt.tight_layout()
    out = 'results/figures/capabilities/basis_nf_comparison.png'
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, dpi=100, bbox_inches='tight')
    print(f'Saved {out}')

if __name__ == '__main__':
    main()