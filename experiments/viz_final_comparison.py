"""
Paper-quality final comparison: VarLen+NLL (Best ZS) vs FeDaL ZS (Table 19).
Also includes LP results where available.

Output:
  results/figures/final_comparison.png   : Per-dataset per-pred_len bars
  results/figures/zs_ranking.png         : Overall ranking across all experiments
  results/figures/per_dataset_summary.png: Avg per dataset
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RES_DIR = '/workspace/HypOPFN2/results'
FIG_DIR = '/workspace/HypOPFN2/results/figures'
os.makedirs(FIG_DIR, exist_ok=True)

# FeDaL ZS per pred_len (Table 19)
FEDAL_ZS = {
    'ETTh1':   {96: 0.347, 192: 0.398, 336: 0.425, 720: 0.457, 'avg': 0.407},
    'ETTh2':   {96: 0.307, 192: 0.349, 336: 0.387, 720: 0.401, 'avg': 0.361},
    'ETTm1':   {96: 0.289, 192: 0.317, 336: 0.370, 720: 0.464, 'avg': 0.360},
    'ETTm2':   {96: 0.207, 192: 0.248, 336: 0.316, 720: 0.397, 'avg': 0.292},
    'Weather': {96: 0.159, 192: 0.217, 336: 0.285, 720: 0.359, 'avg': 0.255},
}

# Our VarLen + NLL (Best ZS) from log
OURS_BEST = {
    'ETTh1':   {96: 0.4289, 192: 0.4577, 336: 0.4731, 720: 0.4974, 'avg': 0.4643},
    'ETTh2':   {96: 0.3130, 192: 0.3709, 336: 0.3890, 720: 0.4222, 'avg': 0.3738},
    'ETTm1':   {96: 0.5056, 192: 0.5429, 336: 0.5803, 720: 0.6195, 'avg': 0.5621},
    'ETTm2':   {96: 0.2111, 192: 0.2707, 336: 0.3198, 720: 0.4055, 'avg': 0.3018},
    'Weather': {96: 0.1883, 192: 0.2495, 336: 0.3086, 720: 0.3714, 'avg': 0.2794},
}

DATASETS = ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'Weather']
PRED_LENS = [96, 192, 336, 720]


# ============================================================
# 1) Per-dataset per-pred_len bar chart (paper table-style viz)
# ============================================================
def plot_per_pred_len():
    fig, axes = plt.subplots(1, 5, figsize=(20, 5), sharey=False)

    for ax, ds in zip(axes, DATASETS):
        x = np.arange(len(PRED_LENS))
        width = 0.35
        ours = [OURS_BEST[ds][pl] for pl in PRED_LENS]
        feda = [FEDAL_ZS[ds][pl] for pl in PRED_LENS]

        b1 = ax.bar(x - width/2, ours, width, label='HypOPFN (Ours)', color='#2E86AB')
        b2 = ax.bar(x + width/2, feda, width, label='FeDaL ZS', color='#A23B72', hatch='//')

        for i, (o, f) in enumerate(zip(ours, feda)):
            gap = (o - f) / f * 100
            color = 'green' if gap < 5 else ('orange' if gap < 20 else 'red')
            ax.text(x[i], max(o, f) + 0.01, f'+{gap:.0f}%', ha='center', fontsize=8, color=color, weight='bold')

        ax.set_xticks(x)
        ax.set_xticklabels(PRED_LENS)
        ax.set_xlabel('pred_len')
        ax.set_title(f'{ds}', fontsize=12, weight='bold')
        ax.grid(alpha=0.3, axis='y')
        if ax is axes[0]:
            ax.set_ylabel('MSE (↓)', fontsize=11)
        if ax is axes[-1]:
            ax.legend(loc='upper right', fontsize=9)

    plt.suptitle('Zero-Shot Forecast MSE: HypOPFN (VarLen+NLL, 32.5M) vs FeDaL ZS (28M)',
                 fontsize=14, weight='bold')
    plt.tight_layout()
    out = f'{FIG_DIR}/final_comparison.png'
    plt.savefig(out, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out}')


# ============================================================
# 2) Per-dataset average comparison (bar)
# ============================================================
def plot_per_dataset_summary():
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(DATASETS))
    width = 0.35
    ours = [OURS_BEST[ds]['avg'] for ds in DATASETS]
    feda = [FEDAL_ZS[ds]['avg'] for ds in DATASETS]

    b1 = ax.bar(x - width/2, ours, width, label='HypOPFN (Ours, 32.5M)', color='#2E86AB')
    b2 = ax.bar(x + width/2, feda, width, label='FeDaL ZS (28M)', color='#A23B72', hatch='//')

    for i, (o, f) in enumerate(zip(ours, feda)):
        gap = (o - f) / f * 100
        color = 'green' if gap < 5 else ('orange' if gap < 20 else 'red')
        ax.text(x[i], max(o, f) + 0.01, f'+{gap:.1f}%', ha='center', fontsize=11,
                color=color, weight='bold')
        ax.text(x[i] - width/2, o + 0.005, f'{o:.3f}', ha='center', fontsize=9)
        ax.text(x[i] + width/2, f + 0.005, f'{f:.3f}', ha='center', fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(DATASETS, fontsize=11)
    ax.set_ylabel('Avg MSE across pred_lens {96, 192, 336, 720}', fontsize=11)
    ax.set_title('HypOPFN vs FeDaL — Zero-Shot Per-Dataset Average',
                 fontsize=13, weight='bold')
    ax.legend(fontsize=11); ax.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    out = f'{FIG_DIR}/per_dataset_summary.png'
    plt.savefig(out, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out}')


# ============================================================
# 3) All experiments ranking
# ============================================================
def plot_zs_ranking():
    experiments = [
        ('v1_varlen_nll_failmode', 'VarLen + NLL (Best)'),
        ('v1_varlen_hybrid_failmode', 'VarLen + Hybrid'),
        ('v1_varlen_spectral_failmode', 'VarLen + Spectral'),
        ('v1_varlen_failmode', 'VarLen (base)'),
        ('overnight_seq720_s50', 'seq720 (baseline)'),
        ('v2_xattn_s50_seq512', 'V2 CrossAttn 76M'),
        ('v1_hybrid_seq720', 'Hybrid + seq720'),
        ('v1_seq720_spectral_failmode', 'seq720 + Spectral'),
    ]

    data = []
    for tag, label in experiments:
        path = f'{RES_DIR}/{tag}.json'
        if not os.path.exists(path): continue
        with open(path) as f:
            d = json.load(f)
        ov = d.get('overall_avg')
        if isinstance(ov, dict):
            data.append((label, ov['MSE']))
    data.sort(key=lambda x: x[1])

    fig, ax = plt.subplots(figsize=(11, 6))
    labels = [d[0] for d in data]
    values = [d[1] for d in data]
    y = np.arange(len(labels))
    colors = ['#2E86AB' if i == 0 else '#A6CEE3' for i in range(len(values))]

    bars = ax.barh(y, values, color=colors)
    for i, (bar, v) in enumerate(zip(bars, values)):
        ax.text(v + 0.003, i, f'{v:.4f}', va='center', fontsize=10)

    ax.axvline(0.335, color='red', linestyle='--', alpha=0.7, label='FeDaL ZS (0.335)')
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel('Overall ZS MSE (↓)', fontsize=11)
    ax.set_title('HypOPFN Experiments Ranked by Zero-Shot Overall MSE',
                 fontsize=13, weight='bold')
    ax.legend(loc='lower right'); ax.grid(alpha=0.3, axis='x')
    plt.tight_layout()
    out = f'{FIG_DIR}/zs_ranking.png'
    plt.savefig(out, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out}')


if __name__ == '__main__':
    plot_per_pred_len()
    plot_per_dataset_summary()
    plot_zs_ranking()
    print(f'\nAll figures in: {FIG_DIR}')
