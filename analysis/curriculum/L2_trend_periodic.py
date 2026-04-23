"""
Curriculum L2: Trend + Periodic
y(t) = (a*t + b) + sin(2π f t + φ)

검증 질문:
  1. Poly trunk가 trend, Fourier trunk가 periodic을 따로 학습하나?
     (Additive decomposition 진짜 작동?)
  2. Compositional OOD: 본 적 없는 (slope, freq) 조합에 일반화되나?
  3. Trend slope 외삽: 본 적 없는 slope range에서 어떻게 되나? (extrapolative)

CUDA_VISIBLE_DEVICES=3 python analysis/curriculum/L2_trend_periodic.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch, torch.nn.functional as F
import numpy as np, time
from torch import optim
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Reuse model from L1
from analysis.curriculum.L1_sum_sinusoids import (
    SmallOperator, train_model, eval_model,
    SEQ_LEN, PRED_LEN, DEVICE, SAVE_DIR
)


# ============================================================
# Data: trend + periodic
# ============================================================
def gen_sample(slope, freq, n=192, t_max=2.0):
    t = np.linspace(0, t_max, n)
    a = slope
    b = np.random.uniform(-1, 1)
    amp = np.random.uniform(0.5, 1.5)
    phi = np.random.uniform(0, 2*np.pi)
    trend = a * t + b
    periodic = amp * np.sin(2*np.pi*freq*t + phi)
    y = trend + periodic
    s = y.std()
    if s > 1e-6:
        y = (y - y.mean()) / s
    # Return both y and components (for attribution analysis)
    trend_n = (trend - y.mean()*0) / s if s > 1e-6 else trend  # crude
    return y.astype(np.float32)


def gen_sample_with_components(slope, freq, n=192, t_max=2.0):
    """Returns y, trend, periodic — all in normalized space."""
    t = np.linspace(0, t_max, n)
    a = slope
    b = np.random.uniform(-1, 1)
    amp = np.random.uniform(0.5, 1.5)
    phi = np.random.uniform(0, 2*np.pi)
    trend = a * t + b
    periodic = amp * np.sin(2*np.pi*freq*t + phi)
    y = trend + periodic
    m, s = y.mean(), y.std()
    if s > 1e-6:
        y_n = (y - m) / s
        trend_n = (trend - m) / s   # trend in same normalized frame
        periodic_n = periodic / s    # periodic centered around 0 already
    else:
        y_n, trend_n, periodic_n = y, trend, periodic
    return y_n.astype(np.float32), trend_n.astype(np.float32), periodic_n.astype(np.float32)


def make_dataset(slope_freq_pairs, n_per_pair=300):
    data = []
    for sl, fr in slope_freq_pairs:
        for _ in range(n_per_pair):
            data.append(gen_sample(sl, fr))
    return np.stack(data)


# Splits
TRAIN_PAIRS = [(0.5, 1), (0.5, 3), (1.0, 2), (1.0, 4), (1.5, 1), (1.5, 4),
               (2.0, 2), (2.0, 3), (-0.5, 1), (-0.5, 3), (-1.0, 2), (-1.0, 4)]

# Compositional OOD: same slopes, same freqs, NEW combinations
TEST_OOD_COMP = [(0.5, 2), (0.5, 4), (1.0, 1), (1.0, 3), (1.5, 2), (1.5, 3),
                 (2.0, 1), (2.0, 4), (-0.5, 2), (-0.5, 4), (-1.0, 1), (-1.0, 3)]

# Extrapolative OOD: new slopes (3.0, -2.0) AND new freqs (5, 6)
TEST_OOD_EXTR = [(3.0, 5), (3.0, 6), (-2.0, 5), (-2.0, 6),
                 (3.0, 1), (-2.0, 2), (1.0, 5), (1.5, 6)]


# ============================================================
# Visualization
# ============================================================
def visualize(model, train_data, id_data, ood_comp, ood_extr,
              losses, results, train_pairs, ood_comp_pairs, ood_extr_pairs):
    fig = plt.figure(figsize=(22, 16))

    # Row 1: Loss curve, ID/OOD bars, info text
    ax1 = plt.subplot(5, 5, 1)
    ax1.plot(losses)
    ax1.set_title('Train Loss'); ax1.set_xlabel('Epoch'); ax1.set_ylabel('MSE')
    ax1.set_yscale('log'); ax1.grid(alpha=0.3)

    ax2 = plt.subplot(5, 5, 2)
    metrics = ['ID', 'OOD-C', 'OOD-E']
    fc_vals = [results['fc_id'], results['fc_ood_c'], results['fc_ood_e']]
    imp_vals = [results['imp_id'], results['imp_ood_c'], results['imp_ood_e']]
    x = np.arange(len(metrics)); width = 0.35
    ax2.bar(x - width/2, fc_vals, width, label='FC', color='#3b82f6')
    ax2.bar(x + width/2, imp_vals, width, label='IMP', color='#10b981')
    ax2.set_xticks(x); ax2.set_xticklabels(metrics)
    ax2.set_title('ID vs Compositional vs Extrapolative OOD')
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3, axis='y')

    ax3 = plt.subplot(5, 5, 3)
    ax3.axis('off')
    txt = (f'Train: 12 (slope, freq) pairs\n'
           f'  slopes ∈ {{-1, -0.5, 0.5, 1, 1.5, 2}}\n'
           f'  freqs ∈ {{1, 2, 3, 4}}\n\n'
           f'OOD-C (new combinations):\n'
           f'  same slopes & freqs\n\n'
           f'OOD-E (extrapolation):\n'
           f'  new slopes {{3, -2}}\n'
           f'  new freqs {{5, 6}}\n\n'
           f'FC gap C: ×{results["fc_ood_c"]/results["fc_id"]:.2f}\n'
           f'FC gap E: ×{results["fc_ood_e"]/results["fc_id"]:.2f}\n'
           f'IMP gap C: ×{results["imp_ood_c"]/results["imp_id"]:.2f}\n'
           f'IMP gap E: ×{results["imp_ood_e"]/results["imp_id"]:.2f}')
    ax3.text(0.0, 0.5, txt, fontsize=8, family='monospace', va='center')

    # Forecast examples row
    model.eval()
    examples = [(id_data, 'ID', 'blue'), (ood_comp, 'OOD-C', 'orange'), (ood_extr, 'OOD-E', 'red')]
    for col, (ds, name, color) in enumerate(examples):
        idx = col * 30
        w = ds[idx]
        ctx = torch.tensor(w[:SEQ_LEN]).unsqueeze(0).float().to(DEVICE)
        with torch.no_grad():
            fp = model.forecast(ctx).cpu().numpy()[0]
        ax = plt.subplot(5, 5, 5 + col + 1)
        ax.plot(range(192), w, 'k-', alpha=0.3, label='GT')
        ax.plot(range(SEQ_LEN), w[:SEQ_LEN], 'k-', linewidth=1.5)
        ax.plot(range(SEQ_LEN, 192), fp, color=color, linewidth=1.5, label='pred')
        ax.axvline(SEQ_LEN, color='gray', linestyle='--', alpha=0.3)
        mse = np.mean((fp - w[SEQ_LEN:])**2)
        ax.set_title(f'{name}  FC MSE={mse:.3f}', fontsize=9)
        ax.legend(fontsize=7); ax.grid(alpha=0.2)

    # ============================================================
    # CRITICAL ROW 3: Per-trunk decomposition
    # 만약 Poly가 trend, Fourier가 periodic을 분해해서 학습했다면 — 이게 핵심
    # ============================================================
    trunk_names = ['Fourier', 'Poly', 'RBF']
    trunk_colors = ['#3b82f6', '#10b981', '#f59e0b']
    sample_sets = [(id_data, 'ID'), (ood_comp, 'OOD-C'), (ood_extr, 'OOD-E')]
    for col, (ds, name) in enumerate(sample_sets):
        idx = col * 50
        w = ds[idx]
        ctx = torch.tensor(w[:SEQ_LEN]).unsqueeze(0).float().to(DEVICE)
        with torch.no_grad():
            per_trunk = model.per_trunk_forecast(ctx)
            full = model.forecast(ctx).cpu().numpy()[0]
        per_trunk_np = [pt.cpu().numpy()[0] for pt in per_trunk]

        ax = plt.subplot(5, 5, 10 + col + 1)
        ax.plot(range(SEQ_LEN, 192), w[SEQ_LEN:], 'k--', alpha=0.5, label='GT', linewidth=2)
        ax.plot(range(SEQ_LEN, 192), full, 'k-', alpha=0.6, label='sum', linewidth=1.5)
        for pt, tn, tc in zip(per_trunk_np, trunk_names, trunk_colors):
            ax.plot(range(SEQ_LEN, 192), pt, color=tc, alpha=0.85, label=tn, linewidth=1.2)
        ax.axhline(0, color='k', alpha=0.2, linewidth=0.5)
        ax.set_title(f'Per-Trunk Decomp ({name})', fontsize=9)
        ax.legend(fontsize=7); ax.grid(alpha=0.2)

    # Row 4: ID context — show learned trend (Poly contribution) vs GT trend
    # We need to check: does Poly trunk's output approximate the trend component?
    for col in range(3):
        idx = col * 70
        w = id_data[idx]
        ctx = torch.tensor(w[:SEQ_LEN]).unsqueeze(0).float().to(DEVICE)
        with torch.no_grad():
            per_trunk = model.per_trunk_forecast(ctx)
        poly_out = per_trunk[1].cpu().numpy()[0]  # Poly trunk output
        # Linear fit on context to extract "true" trend
        t_ctx = np.arange(SEQ_LEN)
        coef = np.polyfit(t_ctx, w[:SEQ_LEN], 1)
        # Extend trend to forecast region
        t_fc = np.arange(SEQ_LEN, 192)
        true_trend = coef[0] * t_fc + coef[1]

        ax = plt.subplot(5, 5, 15 + col + 1)
        ax.plot(range(SEQ_LEN, 192), poly_out, color='#10b981', linewidth=1.5, label='Poly out')
        ax.plot(range(SEQ_LEN, 192), true_trend, 'k--', linewidth=1.2, label='Linear fit (GT trend)')
        ax.set_title(f'Poly trunk vs trend (ID#{col})', fontsize=9)
        ax.legend(fontsize=7); ax.grid(alpha=0.2)

    # Row 5: FFT compare
    for col, (ds, name) in enumerate(sample_sets):
        idx = col * 60
        w = ds[idx]
        ctx = torch.tensor(w[:SEQ_LEN]).unsqueeze(0).float().to(DEVICE)
        with torch.no_grad():
            fp = model.forecast(ctx).cpu().numpy()[0]
        full_pred = np.concatenate([w[:SEQ_LEN], fp])
        fft_gt = np.abs(np.fft.rfft(w))
        fft_pred = np.abs(np.fft.rfft(full_pred))
        ax = plt.subplot(5, 5, 20 + col + 1)
        ax.plot(fft_gt[:25], 'k-', label='GT', linewidth=1.5)
        ax.plot(fft_pred[:25], 'r-', alpha=0.7, label='pred', linewidth=1.5)
        ax.set_title(f'{name} FFT', fontsize=9)
        ax.set_xlabel('freq idx'); ax.legend(fontsize=7); ax.grid(alpha=0.2)

    plt.suptitle(
        f'L2: Trend + Periodic — Additive Decomposition Test\n'
        f'FC: ID={results["fc_id"]:.4f}, OOD-C={results["fc_ood_c"]:.4f} (×{results["fc_ood_c"]/results["fc_id"]:.2f}), '
        f'OOD-E={results["fc_ood_e"]:.4f} (×{results["fc_ood_e"]/results["fc_id"]:.2f}) | '
        f'IMP: ID={results["imp_id"]:.4f}, OOD-C={results["imp_ood_c"]:.4f}, OOD-E={results["imp_ood_e"]:.4f}',
        fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{SAVE_DIR}/L2_trend_periodic.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {SAVE_DIR}/L2_trend_periodic.png')


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    np.random.seed(42); torch.manual_seed(42)
    print('='*60)
    print('Curriculum L2: Trend + Periodic')
    print(f'  Train pairs: {len(TRAIN_PAIRS)} (slope, freq)')
    print(f'  OOD-C (compositional): {len(TEST_OOD_COMP)}')
    print(f'  OOD-E (extrapolative): {len(TEST_OOD_EXTR)}')
    print('='*60)

    print('\nGenerating data...')
    train_data = make_dataset(TRAIN_PAIRS, n_per_pair=200)
    id_data = make_dataset(TRAIN_PAIRS, n_per_pair=50)
    ood_comp = make_dataset(TEST_OOD_COMP, n_per_pair=50)
    ood_extr = make_dataset(TEST_OOD_EXTR, n_per_pair=50)
    print(f'  Train: {len(train_data)}, ID: {len(id_data)}, OOD-C: {len(ood_comp)}, OOD-E: {len(ood_extr)}')

    model = SmallOperator().to(DEVICE)
    n = sum(p.numel() for p in model.parameters())
    print(f'\nModel: {n/1e6:.2f}M params')

    print('\nTraining...')
    t0 = time.time()
    losses = train_model(model, train_data, epochs=120, lr=5e-4)
    print(f'Training time: {time.time()-t0:.1f}s')

    print('\nEval:')
    fc_id, imp_id = eval_model(model, id_data, 'ID')
    fc_ood_c, imp_ood_c = eval_model(model, ood_comp, 'OOD-C')
    fc_ood_e, imp_ood_e = eval_model(model, ood_extr, 'OOD-E')

    results = {
        'fc_id': fc_id, 'imp_id': imp_id,
        'fc_ood_c': fc_ood_c, 'imp_ood_c': imp_ood_c,
        'fc_ood_e': fc_ood_e, 'imp_ood_e': imp_ood_e,
    }

    print(f'\nGap analysis:')
    print(f'  FC  Comp: {fc_ood_c/fc_id:.2f}x  | Extr: {fc_ood_e/fc_id:.2f}x')
    print(f'  IMP Comp: {imp_ood_c/imp_id:.2f}x  | Extr: {imp_ood_e/imp_id:.2f}x')

    visualize(model, train_data, id_data, ood_comp, ood_extr,
              losses, results, TRAIN_PAIRS, TEST_OOD_COMP, TEST_OOD_EXTR)

    print('\n' + '='*60)
    print('L2 DONE')
    print('='*60)
