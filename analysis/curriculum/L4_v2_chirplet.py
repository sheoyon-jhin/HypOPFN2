"""
Curriculum L4 v2: Chirp WITH Chirplet Trunk

Hypothesis: chirp이 안 풀린 이유는 inductive bias 부족 (Fourier+Poly+RBF로는 표현 불가).
검증: chirplet trunk 추가하면 갑자기 풀리는가?

비교:
  Baseline: 3 trunks (Fourier, Poly, RBF) — L4_chirp.py
  v2:       4 trunks (Fourier, Poly, RBF, Chirplet)

CUDA_VISIBLE_DEVICES=3 python analysis/curriculum/L4_v2_chirplet.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch, torch.nn.functional as F
import numpy as np, time
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

from analysis.curriculum.shared import (
    SmallOperator, train_model, eval_model,
    SEQ_LEN, PRED_LEN, DEVICE, SAVE_DIR
)
from analysis.curriculum.L4_chirp import (
    make_dataset, TRAIN_PAIRS, TEST_OOD_COMP, TEST_OOD_EXTR,
)


def visualize(model, id_data, ood_comp, ood_extr, losses, results, baseline_results=None):
    fig = plt.figure(figsize=(22, 16))
    n_trunk = len(model.trunk_names)
    trunk_palette = ['#3b82f6', '#10b981', '#f59e0b', '#a855f7', '#ef4444']
    trunk_colors = trunk_palette[:n_trunk]

    # Row 1
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
    ax2.set_title('ID vs OOD'); ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3, axis='y')

    ax3 = plt.subplot(5, 5, 3)
    ax3.axis('off')
    bl_txt = ''
    if baseline_results:
        bl_txt = (
            f'\n\nBASELINE (3 trunks):\n'
            f'  ID FC: {baseline_results["fc_id"]:.4f}\n'
            f'  OOD-C FC: {baseline_results["fc_ood_c"]:.4f}\n'
            f'  Comp gap: ×{baseline_results["fc_ood_c"]/baseline_results["fc_id"]:.2f}\n\n'
            f'v2 (4 trunks +chirplet):\n'
            f'  ID FC: {results["fc_id"]:.4f}\n'
            f'  OOD-C FC: {results["fc_ood_c"]:.4f}\n'
            f'  Comp gap: ×{results["fc_ood_c"]/results["fc_id"]:.2f}'
        )
    txt = (
        f'4 trunks: {model.trunk_names}\n\n'
        f'Linear chirp:\n'
        f'  y = sin(2π·f(t)·t + φ)\n'
        f'  f(t) = f0 + (f1-f0)·t/2\n\n'
        f'  → phase: 2π f0 t + π α t²\n'
        f'  α = (f1-f0)\n\n'
        f'Chirplet trunk encodes\n'
        f'this phase exactly!'
        f'{bl_txt}'
    )
    ax3.text(0.0, 0.5, txt, fontsize=8, family='monospace', va='center')

    # Row 2: forecast examples
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

    # Row 3: per-trunk decomp (4 trunks now)
    for col, (ds, name) in enumerate([(id_data, 'ID'), (ood_comp, 'OOD-C'), (ood_extr, 'OOD-E')]):
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
        for pt, tn, tc in zip(per_trunk_np, model.trunk_names, trunk_colors):
            ax.plot(range(SEQ_LEN, 192), pt, color=tc, alpha=0.85, label=tn, linewidth=1.2)
        ax.axhline(0, color='k', alpha=0.2, linewidth=0.5)
        ax.set_title(f'Per-Trunk Decomp ({name})', fontsize=9)
        ax.legend(fontsize=6); ax.grid(alpha=0.2)

    # Row 4: full traj decomp
    for col in range(3):
        idx = col * 70
        w = id_data[idx]
        ctx = torch.tensor(w[:SEQ_LEN]).unsqueeze(0).float().to(DEVICE)
        with torch.no_grad():
            t_imp = torch.linspace(0, 1, SEQ_LEN, device=DEVICE).unsqueeze(0)
            t_fc = torch.linspace(1, 2, PRED_LEN, device=DEVICE).unsqueeze(0)
            z = model.enc(ctx)
            pti = model._query_per_trunk(z, t_imp)
            ptf = model._query_per_trunk(z, t_fc)
        per_trunk_full = [np.concatenate([a.cpu().numpy()[0], b.cpu().numpy()[0]])
                          for a, b in zip(pti, ptf)]

        ax = plt.subplot(5, 5, 15 + col + 1)
        ax.plot(range(192), w, 'k--', alpha=0.6, label='GT', linewidth=1.5)
        for pt, tn, tc in zip(per_trunk_full, model.trunk_names, trunk_colors):
            ax.plot(range(192), pt, color=tc, alpha=0.85, label=tn, linewidth=1.0)
        ax.axvline(SEQ_LEN, color='gray', linestyle='--', alpha=0.3)
        ax.axhline(0, color='k', alpha=0.2, linewidth=0.5)
        ax.set_title(f'Full traj decomp (ID#{col})', fontsize=9)
        ax.legend(fontsize=6); ax.grid(alpha=0.2)

    # Row 5: trunk magnitudes + FFT
    sample_sets = [(id_data, 'ID'), (ood_comp, 'OOD-C'), (ood_extr, 'OOD-E')]
    trunk_mags = {n: [] for n in model.trunk_names}
    for ds, name in sample_sets:
        mags = [[] for _ in range(n_trunk)]
        for w in ds[:100]:
            ctx = torch.tensor(w[:SEQ_LEN]).unsqueeze(0).float().to(DEVICE)
            with torch.no_grad():
                pts = model.per_trunk_forecast(ctx)
            for i, pt in enumerate(pts):
                mags[i].append(np.abs(pt.cpu().numpy()[0]).mean())
        for i, k in enumerate(model.trunk_names):
            trunk_mags[k].append(np.mean(mags[i]))

    ax = plt.subplot(5, 5, 21)
    x = np.arange(3); width = 0.18
    for i, (k, c) in enumerate(zip(model.trunk_names, trunk_colors)):
        ax.bar(x + (i - (n_trunk-1)/2)*width, trunk_mags[k], width, label=k, color=c)
    ax.set_xticks(x); ax.set_xticklabels(['ID', 'OOD-C', 'OOD-E'])
    ax.set_title('Trunk activations |output|', fontsize=9)
    ax.legend(fontsize=6); ax.grid(alpha=0.3, axis='y')

    for col, (ds, name) in enumerate(sample_sets):
        idx = col * 60
        w = ds[idx]
        ctx = torch.tensor(w[:SEQ_LEN]).unsqueeze(0).float().to(DEVICE)
        with torch.no_grad():
            fp = model.forecast(ctx).cpu().numpy()[0]
        full_pred = np.concatenate([w[:SEQ_LEN], fp])
        fft_gt = np.abs(np.fft.rfft(w))
        fft_pred = np.abs(np.fft.rfft(full_pred))
        ax = plt.subplot(5, 5, 22 + col)
        ax.plot(fft_gt[:30], 'k-', label='GT', linewidth=1.5)
        ax.plot(fft_pred[:30], 'r-', alpha=0.7, label='pred', linewidth=1.5)
        ax.set_title(f'{name} FFT', fontsize=9)
        ax.set_xlabel('freq idx'); ax.legend(fontsize=7); ax.grid(alpha=0.2)

    plt.suptitle(
        f'L4 v2: Chirp WITH Chirplet Trunk (4 trunks)\n'
        f'FC: ID={results["fc_id"]:.4f}, OOD-C={results["fc_ood_c"]:.4f} (×{results["fc_ood_c"]/results["fc_id"]:.2f}), '
        f'OOD-E={results["fc_ood_e"]:.4f} (×{results["fc_ood_e"]/results["fc_id"]:.2f}) | '
        f'IMP: ID={results["imp_id"]:.4f}, OOD-C={results["imp_ood_c"]:.4f}, OOD-E={results["imp_ood_e"]:.4f}',
        fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{SAVE_DIR}/L4_v2_chirplet.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {SAVE_DIR}/L4_v2_chirplet.png')


if __name__ == '__main__':
    np.random.seed(42); torch.manual_seed(42)
    print('='*60)
    print('Curriculum L4 v2: Chirp + Chirplet Trunk')
    print('='*60)

    print('\nGenerating chirp data...')
    train_data = make_dataset(TRAIN_PAIRS, n_per_pair=150)
    id_data = make_dataset(TRAIN_PAIRS, n_per_pair=40)
    ood_comp = make_dataset(TEST_OOD_COMP, n_per_pair=60)
    ood_extr = make_dataset(TEST_OOD_EXTR, n_per_pair=60)
    print(f'  Train: {len(train_data)}, ID: {len(id_data)}, OOD-C: {len(ood_comp)}, OOD-E: {len(ood_extr)}')

    # 4 trunks: add chirplet
    model = SmallOperator(trunk_types=('fourier', 'poly', 'rbf', 'chirplet')).to(DEVICE)
    n = sum(p.numel() for p in model.parameters())
    print(f'\nModel: {n/1e6:.2f}M params (4 trunks: {model.trunk_names})')

    print('\nTraining...')
    t0 = time.time()
    losses = train_model(model, train_data, epochs=150, lr=5e-4)
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

    # Baseline (3 trunks) — hardcoded from L4_chirp.py results
    baseline = {
        'fc_id': 0.0620, 'imp_id': 0.0633,
        'fc_ood_c': 1.3069, 'imp_ood_c': 0.2380,
        'fc_ood_e': 1.4363, 'imp_ood_e': 0.5595,
    }

    print(f'\nGap analysis:')
    print(f'  FC  Comp: {fc_ood_c/fc_id:.2f}x  | Extr: {fc_ood_e/fc_id:.2f}x')
    print(f'  IMP Comp: {imp_ood_c/imp_id:.2f}x  | Extr: {imp_ood_e/imp_id:.2f}x')

    print(f'\nComparison vs baseline (3 trunks):')
    print(f'  Baseline ID FC: {baseline["fc_id"]:.4f} → v2: {fc_id:.4f}  ({fc_id/baseline["fc_id"]-1:+.0%})')
    print(f'  Baseline OOD-C FC: {baseline["fc_ood_c"]:.4f} → v2: {fc_ood_c:.4f}  ({fc_ood_c/baseline["fc_ood_c"]-1:+.0%})')

    visualize(model, id_data, ood_comp, ood_extr, losses, results, baseline_results=baseline)

    print('\n' + '='*60)
    print('L4 v2 DONE')
    print('='*60)
