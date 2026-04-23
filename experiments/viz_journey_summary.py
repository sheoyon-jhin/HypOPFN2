"""
Journey dashboard: what did we try, what worked, where are we.

Creates ONE comprehensive figure answering:
  1. All experiments tried — ranked by Overall MSE
  2. Current best (NLL) vs FeDaL per-dataset
  3. M4 short-term comparison
  4. Near-parity highlights (pred_lens where we're within 5%)
  5. Key learnings box
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


# ============================================================
# Data
# ============================================================

EXPERIMENTS = [
    ('v1_varlen_nll_failmode',    'NLL baseline ⭐',   'clean', 'forecast'),
    ('v1_varlen_nll_huber',       'Huber',             'tie',   'forecast'),
    ('v1_varlen_failmode',        'Baseline (MSE)',    'tie',   'forecast'),
    ('v1_varlen_hybrid_failmode', 'Hybrid trunk',      'tie',   'forecast'),
    ('v1_varlen_spectral_failmode','Spectral (weak)',  'tie',   'forecast'),
    ('v1_varlen_nll_big',         'Big 60M',           'tie',   'forecast'),
    ('v1_varlen_mse_msf',         'MSE + MSF',         'tie',   'forecast'),
    ('v1_varlen_nll_m4synth',     'M4 synth',          'worse', 'forecast'),
    ('v1_varlen_nll_shortpred',   'short_pred',        'worse', 'forecast'),
    ('v1_varlen_nll_unified',     'Unified (buggy)',   'worse', 'forecast'),
    ('v2_xattn_s50_seq512',       'V2 XAttn',          'worse', 'forecast'),
    ('v2_lrntrunk_s50_seq512',    'V2 Learned trunk',  'worse', 'forecast'),
]

DIVERGED = [
    'msf (NLL)', 'msiq (NLL)', 'attnpool (NLL)',
    'varMask', 'optionA', 'padmask',
]

RUNNING = [
    'full_short (GPU 0)',
    'strongspec (GPU 1)',
    'tgtboost (GPU 2)',
    'unified_fixed (GPU 3)',
]

# FeDaL ZS reference (per-dataset avg across 4 pred_lens)
FEDAL_LT = {
    'ETTh1': 0.407, 'ETTh2': 0.349, 'ETTm1': 0.360, 'ETTm2': 0.256, 'Weather': 0.255,
}
FEDAL_M4 = {'Yearly': 13.08, 'Quarterly': 9.81, 'Monthly': 12.12,
            'Weekly': 7.86, 'Daily': 3.16, 'Hourly': 12.40}
FEDAL_IMP = {'ETTh1': 0.149, 'ETTh2': 0.092, 'ETTm1': 0.083, 'ETTm2': 0.057, 'Weather': 0.030}


# ============================================================
# Load results
# ============================================================

def load(tag):
    path = f'/workspace/HypOPFN2/results/{tag}.json'
    if not os.path.exists(path):
        return None
    return json.load(open(path))


# ============================================================
# Figure
# ============================================================

def main():
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.3)

    # ---------- Panel 1: Experiments ranked ----------
    ax1 = fig.add_subplot(gs[0, :])
    data = []
    for tag, name, cat, _ in EXPERIMENTS:
        r = load(tag)
        if r and 'overall_avg' in r:
            data.append((name, r['overall_avg']['MSE'], cat))
    data.sort(key=lambda x: x[1])
    names = [d[0] for d in data]
    mses = [d[1] for d in data]
    cats = [d[2] for d in data]
    colors = {'clean': '#2ecc71', 'tie': '#f39c12', 'worse': '#e74c3c'}
    bar_colors = [colors[c] for c in cats]
    ax1.barh(names, mses, color=bar_colors, edgecolor='black', linewidth=0.5)
    ax1.axvline(0.325, color='blue', linestyle='--', linewidth=1.5, label='FeDaL ZS (0.325)')
    ax1.axvline(data[0][1], color='green', linestyle=':', linewidth=1.5, label=f'Our best ({data[0][1]:.3f})')
    ax1.set_xlabel('Overall MSE (ETT+Weather ZS, lower=better)', fontsize=11)
    ax1.set_title('Experiment Journey — All Completed Runs vs FeDaL', fontsize=13, fontweight='bold')
    ax1.legend(loc='lower right')
    ax1.set_xlim(0.30, 0.65)
    ax1.grid(axis='x', alpha=0.3)
    for i, (_, mse, _) in enumerate(data):
        ax1.text(mse + 0.005, i, f'{mse:.3f}', va='center', fontsize=8)

    # ---------- Panel 2: Per-dataset (Forecast) ----------
    ax2 = fig.add_subplot(gs[1, 0])
    nll = load('v1_varlen_nll_failmode')
    datasets = list(FEDAL_LT.keys())
    ours = [nll[f'{d}_avg']['MSE'] for d in datasets]
    fed = [FEDAL_LT[d] for d in datasets]
    x = np.arange(len(datasets))
    w = 0.35
    ax2.bar(x - w/2, ours, w, label='Ours', color='#2ecc71', edgecolor='black')
    ax2.bar(x + w/2, fed, w, label='FeDaL', color='#3498db', edgecolor='black')
    ax2.set_xticks(x); ax2.set_xticklabels(datasets, rotation=30)
    ax2.set_ylabel('Avg MSE (ZS)')
    ax2.set_title('Forecast — Per Dataset', fontsize=11, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(axis='y', alpha=0.3)
    for i, (o, f) in enumerate(zip(ours, fed)):
        gap = (o - f) / f * 100
        color = 'green' if abs(gap) < 10 else ('orange' if gap < 20 else 'red')
        ax2.text(i, max(o, f) + 0.02, f'{gap:+.0f}%', ha='center', fontsize=9, color=color, fontweight='bold')

    # ---------- Panel 3: M4 ----------
    ax3 = fig.add_subplot(gs[1, 1])
    m4 = load('m4_varlen_nll')
    if m4:
        freqs = list(FEDAL_M4.keys())
        ours_m = [m4.get(f, {}).get('sMAPE', 0) for f in freqs]
        fed_m = [FEDAL_M4[f] for f in freqs]
        x = np.arange(len(freqs))
        ax3.bar(x - w/2, ours_m, w, label='Ours', color='#2ecc71', edgecolor='black')
        ax3.bar(x + w/2, fed_m, w, label='FeDaL', color='#3498db', edgecolor='black')
        ax3.set_xticks(x); ax3.set_xticklabels(freqs, rotation=30, fontsize=9)
        ax3.set_ylabel('sMAPE')
        ax3.set_title('M4 Short-term', fontsize=11, fontweight='bold')
        ax3.legend(fontsize=9)
        ax3.grid(axis='y', alpha=0.3)

    # ---------- Panel 4: Imputation ----------
    ax4 = fig.add_subplot(gs[1, 2])
    imp = load('imp_varlen_nll')
    if imp:
        datasets = list(FEDAL_IMP.keys())
        ours_i = [imp[d]['avg']['MSE'] for d in datasets]
        fed_i = [FEDAL_IMP[d] for d in datasets]
        x = np.arange(len(datasets))
        ax4.bar(x - w/2, ours_i, w, label='Ours', color='#2ecc71', edgecolor='black')
        ax4.bar(x + w/2, fed_i, w, label='FeDaL', color='#3498db', edgecolor='black')
        ax4.set_xticks(x); ax4.set_xticklabels(datasets, rotation=30, fontsize=9)
        ax4.set_ylabel('MSE (avg mask rates)')
        ax4.set_title('Imputation', fontsize=11, fontweight='bold')
        ax4.legend(fontsize=9)
        ax4.grid(axis='y', alpha=0.3)

    # ---------- Panel 5: Near-parity highlights ----------
    ax5 = fig.add_subplot(gs[2, 0])
    # Per-pred_len detail
    FEDAL_PRED = {
        'ETTh2_336': 0.371, 'ETTh2_720': 0.397,
        'ETTm2_336': 0.279, 'ETTm2_720': 0.354,
        'Weather_336': 0.285, 'Weather_720': 0.359,
    }
    labels, ours_np, fed_np = [], [], []
    for key, f in FEDAL_PRED.items():
        if key in nll:
            o = nll[key]['MSE']
            labels.append(key)
            ours_np.append(o)
            fed_np.append(f)
    x = np.arange(len(labels))
    ax5.bar(x - w/2, ours_np, w, label='Ours', color='#2ecc71', edgecolor='black')
    ax5.bar(x + w/2, fed_np, w, label='FeDaL', color='#3498db', edgecolor='black')
    ax5.set_xticks(x); ax5.set_xticklabels(labels, rotation=45, fontsize=8, ha='right')
    ax5.set_ylabel('MSE')
    ax5.set_title('Near-Parity Pred_len (our strongest)', fontsize=11, fontweight='bold')
    ax5.legend(fontsize=9)
    ax5.grid(axis='y', alpha=0.3)
    for i, (o, f) in enumerate(zip(ours_np, fed_np)):
        gap = (o - f) / f * 100
        color = 'green' if gap < 5 else 'orange'
        ax5.text(i, max(o, f) + 0.01, f'{gap:+.0f}%', ha='center', fontsize=8, color=color, fontweight='bold')

    # ---------- Panel 6: Diverged experiments ----------
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.axis('off')
    ax6.text(0.05, 0.95, 'What FAILED to train:', fontsize=12, fontweight='bold',
             transform=ax6.transAxes, verticalalignment='top')
    for i, d in enumerate(DIVERGED):
        ax6.text(0.1, 0.85 - i*0.12, f'❌ {d}',
                 fontsize=10, color='#e74c3c',
                 transform=ax6.transAxes, verticalalignment='top')
    ax6.text(0.05, 0.08,
             'Pattern: NLL + architecture change\n→ σ²→0 → gradient explosion\n'
             '→ 6 experiments collapsed at epoch 5-7',
             fontsize=9, style='italic',
             transform=ax6.transAxes, verticalalignment='bottom',
             bbox=dict(boxstyle='round', facecolor='#ffcccc', alpha=0.5))

    # ---------- Panel 7: Currently running + takeaways ----------
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.axis('off')
    ax7.text(0.05, 0.95, '🔄 Currently Running:', fontsize=12, fontweight='bold',
             transform=ax7.transAxes, verticalalignment='top')
    for i, r in enumerate(RUNNING):
        ax7.text(0.1, 0.85 - i*0.1, f'• {r}',
                 fontsize=10, transform=ax7.transAxes, verticalalignment='top')

    ax7.text(0.05, 0.42, '🎯 Key Findings:', fontsize=12, fontweight='bold',
             transform=ax7.transAxes, verticalalignment='top')
    findings = [
        'Best: NLL + failmode synth (0.396)',
        '+7% ETTh2 is our strongest near-parity',
        'Weather_720 +3% (best pred_len)',
        'ETTm2_336 +1% (near-tie)',
        'M4 Hourly full eval +76% (not win)',
        'Capacity↑ doesn\'t help (Big 60M same)',
        'Data 1/185 vs FeDaL is real bottleneck',
    ]
    for i, f in enumerate(findings):
        ax7.text(0.1, 0.34 - i*0.05, f'• {f}',
                 fontsize=9, transform=ax7.transAxes, verticalalignment='top')

    # ---------- Super title ----------
    fig.suptitle(
        'HypOPFN — Experimental Journey Dashboard\n'
        '32.5M params | LOTSA 0.6% + 500K synth | Zero-Shot on ETT/Weather/M4',
        fontsize=14, fontweight='bold', y=0.995,
    )

    os.makedirs('/workspace/HypOPFN2/results/figures/failure_analysis', exist_ok=True)
    out = '/workspace/HypOPFN2/results/figures/failure_analysis/journey_dashboard.png'
    plt.savefig(out, dpi=110, bbox_inches='tight')
    print(f'Saved {out}')


if __name__ == '__main__':
    main()