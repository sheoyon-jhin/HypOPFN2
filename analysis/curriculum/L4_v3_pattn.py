"""
Curriculum L4 v3: Chirp + PatchAttn Encoder

Hypothesis: MLP encoder가 chirp의 (f0, f1) 추출을 못함.
PatchAttn은 local sequential 정보 보존 → frequency 변화 잡아야 함.

3개 비교:
  Baseline (L4):    MLP enc + 3 trunks (no chirplet)
  v2 (L4_v2):       MLP enc + 4 trunks (chirplet)
  v3a (this):       PatchAttn enc + 3 trunks
  v3b (this):       PatchAttn enc + 4 trunks (chirplet)

이걸로 어떤 component가 진짜 bottleneck인지 disentangle.

CUDA_VISIBLE_DEVICES=3 python analysis/curriculum/L4_v3_pattn.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
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


def run(label, trunk_types, encoder, train_data, id_data, ood_comp, ood_extr):
    print(f'\n{"="*60}')
    print(f'[{label}] enc={encoder}, trunks={trunk_types}')
    print('='*60)
    torch.manual_seed(42); np.random.seed(42)
    model = SmallOperator(trunk_types=trunk_types, encoder=encoder).to(DEVICE)
    n = sum(p.numel() for p in model.parameters())
    print(f'Params: {n/1e6:.2f}M')

    t0 = time.time()
    losses = train_model(model, train_data, epochs=150, lr=5e-4)
    print(f'Time: {time.time()-t0:.1f}s')

    fc_id, imp_id = eval_model(model, id_data, 'ID')
    fc_oc, imp_oc = eval_model(model, ood_comp, 'OOD-C')
    fc_oe, imp_oe = eval_model(model, ood_extr, 'OOD-E')

    return {
        'label': label, 'model': model, 'losses': losses, 'params': n,
        'fc_id': fc_id, 'imp_id': imp_id,
        'fc_ood_c': fc_oc, 'imp_ood_c': imp_oc,
        'fc_ood_e': fc_oe, 'imp_ood_e': imp_oe,
    }


def visualize_compare(results, id_data, ood_comp, ood_extr):
    fig = plt.figure(figsize=(22, 16))
    n_runs = len(results)

    # Row 1: Bar comparison FC ID/OOD-C
    ax = plt.subplot(4, 4, 1)
    labels = [r['label'] for r in results]
    fc_id = [r['fc_id'] for r in results]
    fc_oc = [r['fc_ood_c'] for r in results]
    x = np.arange(len(labels)); width = 0.35
    ax.bar(x - width/2, fc_id, width, label='ID', color='#3b82f6')
    ax.bar(x + width/2, fc_oc, width, label='OOD-C', color='#ef4444')
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, fontsize=7)
    ax.set_title('FC MSE comparison', fontsize=9)
    ax.set_ylabel('MSE'); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis='y')

    # Row 1: Comp gap
    ax = plt.subplot(4, 4, 2)
    gaps = [r['fc_ood_c']/r['fc_id'] for r in results]
    bars = ax.bar(x, gaps, color=['#94a3b8','#94a3b8','#3b82f6','#10b981'][:n_runs])
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, fontsize=7)
    ax.set_title('Compositional gap (OOD-C / ID)', fontsize=9)
    ax.grid(alpha=0.3, axis='y')
    for b, v in zip(bars, gaps):
        ax.text(b.get_x()+b.get_width()/2, v, f'{v:.1f}x', ha='center', va='bottom', fontsize=8)

    # Row 1: Loss curves
    ax = plt.subplot(4, 4, 3)
    for r in results:
        ax.plot(r['losses'], label=r['label'])
    ax.set_yscale('log')
    ax.set_title('Train loss', fontsize=9)
    ax.set_xlabel('Epoch'); ax.legend(fontsize=7); ax.grid(alpha=0.3)

    # Row 1: text
    ax = plt.subplot(4, 4, 4)
    ax.axis('off')
    txt = 'L4 chirp ablation:\n\n'
    for r in results:
        txt += (f'{r["label"]}\n'
                f'  ID FC: {r["fc_id"]:.4f}\n'
                f'  OOD-C: {r["fc_ood_c"]:.4f} (×{r["fc_ood_c"]/r["fc_id"]:.1f})\n'
                f'  Params: {r["params"]/1e6:.2f}M\n\n')
    ax.text(0.0, 1.0, txt, fontsize=7, family='monospace', va='top')

    # Row 2-4: Forecast examples per model (3 examples each: ID, OOD-C, OOD-E)
    examples = [(id_data, 'ID', 'blue'), (ood_comp, 'OOD-C', 'orange'), (ood_extr, 'OOD-E', 'red')]
    for row_idx, (ds, name, color) in enumerate(examples):
        for col_idx, r in enumerate(results):
            model = r['model']
            model.eval()
            idx = 30
            w = ds[idx]
            ctx = torch.tensor(w[:SEQ_LEN]).unsqueeze(0).float().to(DEVICE)
            with torch.no_grad():
                fp = model.forecast(ctx).cpu().numpy()[0]
            ax = plt.subplot(4, 4, 4 + row_idx*4 + col_idx + 1)
            ax.plot(range(192), w, 'k-', alpha=0.3, label='GT')
            ax.plot(range(SEQ_LEN), w[:SEQ_LEN], 'k-', linewidth=1.2)
            ax.plot(range(SEQ_LEN, 192), fp, color=color, linewidth=1.5, label='pred')
            ax.axvline(SEQ_LEN, color='gray', linestyle='--', alpha=0.3)
            mse = np.mean((fp - w[SEQ_LEN:])**2)
            ax.set_title(f'{r["label"]} {name}\nMSE={mse:.3f}', fontsize=8)
            if col_idx == 0: ax.set_ylabel(name, fontsize=9)

    plt.suptitle('L4 Chirp: Encoder × Trunk Ablation', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{SAVE_DIR}/L4_v3_pattn_ablation.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {SAVE_DIR}/L4_v3_pattn_ablation.png')


if __name__ == '__main__':
    np.random.seed(42); torch.manual_seed(42)
    print('='*60)
    print('Curriculum L4 v3: Encoder Ablation')
    print('='*60)

    print('\nGenerating chirp data...')
    train_data = make_dataset(TRAIN_PAIRS, n_per_pair=150)
    id_data = make_dataset(TRAIN_PAIRS, n_per_pair=40)
    ood_comp = make_dataset(TEST_OOD_COMP, n_per_pair=60)
    ood_extr = make_dataset(TEST_OOD_EXTR, n_per_pair=60)
    print(f'Train: {len(train_data)}')

    runs = [
        ('MLP+3T',     ('fourier','poly','rbf'), 'mlp'),
        ('MLP+4T',     ('fourier','poly','rbf','chirplet'), 'mlp'),
        ('PAttn+3T',   ('fourier','poly','rbf'), 'pattn'),
        ('PAttn+4T',   ('fourier','poly','rbf','chirplet'), 'pattn'),
    ]

    results = []
    for label, trunks, enc in runs:
        r = run(label, trunks, enc, train_data, id_data, ood_comp, ood_extr)
        results.append(r)

    print('\n' + '='*60)
    print('SUMMARY (FC)')
    print('='*60)
    print(f'{"label":<12} {"params":<10} {"ID":<10} {"OOD-C":<10} {"gap":<8}')
    print('-'*60)
    for r in results:
        gap = r['fc_ood_c'] / r['fc_id']
        print(f'{r["label"]:<12} {r["params"]/1e6:<10.2f} {r["fc_id"]:<10.4f} {r["fc_ood_c"]:<10.4f} {gap:<8.2f}')
    print('='*60)

    visualize_compare(results, id_data, ood_comp, ood_extr)
    print('\nL4 v3 DONE')
