"""
Can our basis functions (Fourier/Poly/RBF) actually FIT the target data
if given OPTIMAL coefficients (no HyperNet, just least squares)?

This answers: is the problem BASIS EXPRESSIVENESS or HYPERNET ACTIVATION?
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import torch, numpy as np, math
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from types import SimpleNamespace
import pandas as pd

SEQ_LEN = 720
NF = 32; DEG = 6; NC = 20


def make_fourier_basis(t, nf=NF):
    """(L, 1+2*nf)"""
    t = t.reshape(-1, 1)
    f = np.arange(1, nf+1).reshape(1, -1)
    return np.hstack([t, np.sin(2*np.pi*f*t), np.cos(2*np.pi*f*t)])

def make_poly_basis(t, deg=DEG):
    """(L, deg+1)"""
    return np.stack([t**i for i in range(deg+1)], axis=-1)

def make_rbf_basis(t, nc=NC, width=20):
    """(L, 1+nc)"""
    centers = np.linspace(0, 2, nc)
    t = t.reshape(-1, 1)
    return np.hstack([t, np.exp(-width * (t - centers.reshape(1, -1))**2)])

def make_combined_basis(t):
    return np.hstack([make_fourier_basis(t), make_poly_basis(t), make_rbf_basis(t)])

def least_squares_fit(basis, y):
    """Best possible fit with given basis (Oracle)"""
    coef, res, _, _ = np.linalg.lstsq(basis, y, rcond=None)
    pred = basis @ coef
    mse = ((pred - y)**2).mean()
    return pred, mse, coef


def load_real(ds_key, start=5000):
    cfg = {
        'ETTm1': './dataset/ETT-small/ETTm1.csv',
        'ETTh2': './dataset/ETT-small/ETTh2.csv',
        'ETTm2': './dataset/ETT-small/ETTm2.csv',
        'Weather': './dataset/weather/weather.csv',
    }
    df = pd.read_csv(cfg[ds_key])
    col = df.columns[1]  # first feature column
    data = df[col].values[start:start+SEQ_LEN].astype(np.float32)
    m, s = data.mean(), data.std().clip(min=1e-6)
    return np.clip((data - m) / s, -10, 10)


def main():
    datasets = ['ETTm1', 'ETTh2', 'ETTm2', 'Weather']
    labels = [
        'ETTm1 (periodic, good)',
        'ETTh2 (noisy, bad)',
        'ETTm2 (low var, bad)',
        'Weather (trend, mixed)',
    ]
    basis_names = ['Fourier (nf=32)', 'Polynomial (deg=6)', 'RBF (nc=20)', 'All combined']

    fig, axes = plt.subplots(len(datasets), 5, figsize=(22, 3 * len(datasets)))

    for row, (ds, label) in enumerate(zip(datasets, labels)):
        y = load_real(ds)
        L = len(y)
        # Use t ∈ [0, 1] for context, then future would be [1, 2]
        # But here we fit the CONTEXT itself (can basis represent the data at all?)
        t = np.linspace(0, 1, L)

        bases = {
            'Fourier (nf=32)': make_fourier_basis(t),
            'Polynomial (deg=6)': make_poly_basis(t),
            'RBF (nc=20)': make_rbf_basis(t),
            'All combined': make_combined_basis(t),
        }

        # Col 0: raw data
        ax = axes[row, 0]
        ax.plot(y, 'b-', linewidth=0.8)
        ax.set_title(f'{ds} (original)', fontsize=9)
        ax.set_ylabel(label, fontsize=9, fontweight='bold')
        ax.grid(alpha=0.3)
        ax.set_ylim(-4, 4)

        # Col 1-4: basis fits
        for col, (bname, basis) in enumerate(bases.items()):
            ax = axes[row, col+1]
            pred, mse, coef = least_squares_fit(basis, y)
            ax.plot(y, 'b-', alpha=0.3, linewidth=0.6)
            ax.plot(pred, 'r-', linewidth=1)
            ax.set_title(f'{bname}\nMSE={mse:.4f} (dim={basis.shape[1]})', fontsize=8)
            ax.grid(alpha=0.3)
            ax.set_ylim(-4, 4)

    fig.suptitle(
        'Basis Expressiveness Test: Can Fourier/Poly/RBF FIT the data with optimal (LS) coefficients?\n'
        'Blue=truth, Red=best-fit. Low MSE = basis CAN represent. High MSE = basis CANNOT.',
        fontsize=12, fontweight='bold')
    plt.tight_layout()
    out = 'results/figures/capabilities/basis_expressiveness.png'
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, dpi=100, bbox_inches='tight')
    print(f'Saved {out}')


if __name__ == '__main__':
    main()