"""
Paper Figure: Function-to-Function Mapping Verification (32.5M)

Diagnostics that prove our model learned function-to-function mapping, not memorization.

Tests:
  (a) Query density invariance — predict same function at different densities
  (b) Smoothness — output continuity with respect to query t
  (c) Input perturbation robustness — small input noise → small output delta
  (d) Continuous decoding — dense visualization of learned function

CUDA_VISIBLE_DEVICES=X python for_figure/fig_function_learning.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch, torch.nn.functional as F
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from types import SimpleNamespace

from data_provider.data_factory import data_provider
from experiments.exp_full_scale_train import FullScaleModel, SEQ_LEN, extract_freq, build_iq, hyper_forward_with_iq, HIDDEN

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CKPT = 'checkpoints/full_scale_run.pth'
SAVE_DIR = 'for_figure/figures'
os.makedirs(SAVE_DIR, exist_ok=True)


@torch.no_grad()
def query_at_times(model, ctx_n, t_values):
    """ctx_n: [B, 96] normalized input, t_values: [B, nq] → [B, nq]"""
    z = model.encoder(ctx_n)
    # Build informed query from normalized input
    freqs, phases, lv, ls = extract_freq(ctx_n)
    # Per-sample build
    B, nq = t_values.shape
    t_col = t_values[0:1].t()    # [nq, 1] — use first sample's t for all (they're same)
    iq = build_iq(t_col, freqs, phases, lv, ls)    # [B, nq, informed_dim]

    out = torch.zeros(B, nq, device=ctx_n.device, dtype=ctx_n.dtype)
    t_flat = t_values.reshape(-1)
    for head, trunk, bias in zip(model.heads, model.trunks, model.biases):
        ho = head(z)
        out = out + hyper_forward_with_iq(trunk, t_flat, ho, iq) + bias
    return out


def get_sample(dataset_name='ETTh1'):
    """Get a representative sample from ETTh1 test set."""
    datasets = {
        'ETTh1':   ('ETTh1', './dataset/ETT-small/', 'ETTh1.csv', 7),
        'Weather': ('custom','./dataset/weather/',   'weather.csv', 21),
    }
    d, root, f, enc_in = datasets[dataset_name]
    a = SimpleNamespace(seq_len=96, pred_len=96, label_len=48, data=d,
        root_path=root, data_path=f, features='M', target='OT',
        freq='h', embed='timeF', enc_in=enc_in, dec_in=enc_in, c_out=enc_in,
        num_workers=2, batch_size=1, exp_name='MTSF', ordered_data=False,
        data_amount=-1, combine_Gaussian_datasets=False,
        synthetic_data_path='', synthetic_root_path='./',
        synthetic_length=1024, stride=-1)
    _, tdl = data_provider(a, 'test')
    for bx, by, _, _ in tdl:
        # Use channel 0 (OT)
        x = bx[0, :, 0].numpy()     # context [96]
        y = by[0, :, 0].numpy()     # target (label_len + pred_len) = 48 + 96
        return x, y[-96:]            # return 96 context + 96 future


def normalize(x):
    m = x.mean()
    s = x.std() + 1e-6
    return (x - m) / s, m, s


def denormalize(y_n, m, s):
    return y_n * s + m


# ============================================================
# Test (a): Query density invariance
# ============================================================
def test_query_density(model, ctx, future, save_path):
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))

    ctx_n, m, s = normalize(ctx)
    ctx_t = torch.tensor(ctx_n, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    # Ground truth
    full_gt = np.concatenate([ctx, future])

    # Row 1: Forecast at different densities
    densities = [10, 24, 48, 96, 192, 500]
    colors = plt.cm.viridis(np.linspace(0, 1, len(densities)))

    ax = axes[0, 0]
    ax.plot(range(96), ctx, 'k-', linewidth=2, label='context')
    ax.plot(range(96, 192), future, 'k--', linewidth=1, alpha=0.5, label='gt future')
    for i, (n, c) in enumerate(zip(densities, colors)):
        t = torch.linspace(1, 2, n, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            pred_n = query_at_times(model, ctx_t, t).cpu().numpy()[0]
        pred = denormalize(pred_n, m, s)
        t_abs = np.linspace(96, 192, n)
        ax.plot(t_abs, pred, '-', color=c, linewidth=1.2, alpha=0.8,
                label=f'n={n}' if n in [10, 96, 500] else None)
    ax.set_title('(a) Query density invariance (Forecast)', fontsize=11)
    ax.set_xlabel('time'); ax.set_ylabel('value')
    ax.axvline(96, color='gray', linestyle=':', alpha=0.5)
    ax.legend(fontsize=8, loc='best'); ax.grid(alpha=0.3)

    # Row 1: Imputation at different densities
    ax = axes[0, 1]
    ax.plot(range(96), ctx, 'k-', linewidth=1, alpha=0.6, label='full context')
    for i, (n, c) in enumerate(zip(densities, colors)):
        t = torch.linspace(0, 1, n, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            pred_n = query_at_times(model, ctx_t, t).cpu().numpy()[0]
        pred = denormalize(pred_n, m, s)
        t_abs = np.linspace(0, 96, n)
        ax.plot(t_abs, pred, '-', color=c, linewidth=1.2, alpha=0.8)
    ax.set_title('(a) Query density invariance (Impute)', fontsize=11)
    ax.set_xlabel('time'); ax.grid(alpha=0.3)

    # Extended — query beyond training (extrapolation)
    ax = axes[0, 2]
    ax.plot(range(96), ctx, 'k-', linewidth=2, label='context')
    ax.plot(range(96, 192), future, 'k--', linewidth=1, alpha=0.5, label='gt future')
    # Very dense query
    n = 1000
    t_ext = torch.linspace(-0.5, 2.5, n, device=DEVICE).unsqueeze(0)
    with torch.no_grad():
        pred_n = query_at_times(model, ctx_t, t_ext).cpu().numpy()[0]
    pred = denormalize(pred_n, m, s)
    t_abs = np.linspace(-48, 240, n)
    ax.plot(t_abs, pred, 'r-', linewidth=1.2, alpha=0.7, label='model (t ∈ [-0.5, 2.5])')
    ax.axvspan(0, 96, alpha=0.1, color='blue', label='train context')
    ax.axvspan(96, 192, alpha=0.1, color='green', label='train forecast')
    ax.set_title('(d) Continuous decoding (in/out of training range)', fontsize=11)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # Row 2: Smoothness test
    ax = axes[1, 0]
    t_base = torch.linspace(1, 2, 96, device=DEVICE).unsqueeze(0)
    with torch.no_grad():
        y_base = query_at_times(model, ctx_t, t_base).cpu().numpy()[0]
    eps_list = [0.001, 0.005, 0.01]
    ax.plot(range(96), y_base, 'k-', linewidth=2, label='y(t)')
    for eps in eps_list:
        t_pert = t_base + eps
        with torch.no_grad():
            y_pert = query_at_times(model, ctx_t, t_pert).cpu().numpy()[0]
        ax.plot(range(96), y_pert, '--', alpha=0.7, label=f'y(t+{eps})')
    ax.set_title('(b) Smoothness: y(t) vs y(t+ε)', fontsize=11)
    ax.set_xlabel('query idx'); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # Input perturbation test
    ax = axes[1, 1]
    rng = np.random.RandomState(42)
    t = torch.linspace(1, 2, 96, device=DEVICE).unsqueeze(0)
    with torch.no_grad():
        y_orig = query_at_times(model, ctx_t, t).cpu().numpy()[0]
    ax.plot(range(96), y_orig, 'k-', linewidth=2, label='y(x)')
    for noise_scale in [0.01, 0.05, 0.1]:
        noise = rng.randn(96).astype(np.float32) * noise_scale
        ctx_noisy = torch.tensor(ctx_n + noise, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            y_noisy = query_at_times(model, ctx_noisy, t).cpu().numpy()[0]
        diff = np.mean(np.abs(y_noisy - y_orig))
        ax.plot(range(96), y_noisy, '--', alpha=0.7,
                label=f'noise σ={noise_scale} (Δ={diff:.3f})')
    ax.set_title('(c) Input perturbation robustness', fontsize=11)
    ax.set_xlabel('query idx'); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # Per-trunk decomposition (show which basis is active)
    ax = axes[1, 2]
    t = torch.linspace(1, 2, 96, device=DEVICE).unsqueeze(0)
    with torch.no_grad():
        z = model.encoder(ctx_t)
        freqs, phases, lv, ls = extract_freq(ctx_t)
        t_col = t[0:1].t()
        iq = build_iq(t_col, freqs, phases, lv, ls)
        t_flat = t.reshape(-1)
        per_trunk = []
        for head, trunk, bias in zip(model.heads, model.trunks, model.biases):
            ho = head(z)
            pt = hyper_forward_with_iq(trunk, t_flat, ho, iq) + bias
            per_trunk.append(pt.cpu().numpy()[0])
    trunk_names = ['Fourier', 'Poly', 'RBF']
    trunk_colors = ['#3b82f6', '#10b981', '#f59e0b']
    total = sum(per_trunk)
    ax.plot(range(96), future, 'k--', linewidth=1, alpha=0.4, label='gt')
    ax.plot(range(96), denormalize(total, m, s), 'k-', linewidth=1.5, label='sum')
    for pt, name, col in zip(per_trunk, trunk_names, trunk_colors):
        ax.plot(range(96), denormalize(pt, m, s), '-', color=col, alpha=0.7, label=name, linewidth=1)
    ax.set_title('Per-trunk decomposition (forecast)', fontsize=11)
    ax.set_xlabel('future step'); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    plt.suptitle(f'32.5M Function-to-Function Mapping Diagnostics', fontsize=13, y=1.00)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {save_path}')


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    print('='*60)
    print('Function Learning Diagnostics')
    print('='*60)

    model = FullScaleModel().to(DEVICE)
    state = torch.load(CKPT, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    n = sum(p.numel() for p in model.parameters())
    print(f'Model: {n/1e6:.1f}M params')

    for ds in ['ETTh1', 'Weather']:
        print(f'\n--- {ds} ---')
        ctx, future = get_sample(ds)
        print(f'  Context shape: {ctx.shape}, future shape: {future.shape}')
        save_path = f'{SAVE_DIR}/fig_function_learning_{ds}.png'
        test_query_density(model, ctx, future, save_path)

    print('\nDONE')
