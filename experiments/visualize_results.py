"""
Comprehensive visualization of all experiment results.

Generates figures under results/figures/:
  - training_curves.png    : All experiments' training loss curves
  - forecast_compare.png   : Bar chart, per-dataset MSE vs FeDaL
  - imputation_compare.png : Bar chart, imputation avg MSE vs FeDaL
  - m4_compare.png         : Bar chart, M4 sMAPE vs FeDaL
  - pred_len_trend.png     : MSE by pred_len (rolling vs direct comparison)
  - failure_scatter.png    : Per-sample MSE vs feature scatter (from analyze_failures)
  - failure_examples.png   : Worst-case prediction examples
  - synth_patterns.png     : Sample synthetic series per domain

Usage:
  python experiments/visualize_results.py
"""
import sys, os, glob, json, re
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

LOG_DIR = '/workspace/HypOPFN2/log'
RES_DIR = '/workspace/HypOPFN2/results'
FIG_DIR = '/workspace/HypOPFN2/results/figures'
os.makedirs(FIG_DIR, exist_ok=True)

# Reference values (FeDaL Table 4 zero-shot + Table 5 imputation)
FEDAL_FORECAST = {
    'ETTh1': 0.407, 'ETTh2': 0.361, 'ETTm1': 0.360,
    'ETTm2': 0.292, 'Weather': 0.255,
}
FEDAL_IMPUTATION = {
    'ETTh1': 0.149, 'ETTh2': 0.092, 'ETTm1': 0.083,
    'ETTm2': 0.057, 'Weather': 0.030,
}
FEDAL_M4 = {
    'Yearly': 13.08, 'Quarterly': 9.81, 'Monthly': 12.12,
    'Weekly': 7.86, 'Daily': 3.16, 'Hourly': 12.40,
}


# ============================================================
# 1) Training curves
# ============================================================
def parse_train_log(log_path):
    """Extract (epoch, loss) pairs from a log file."""
    epochs, losses = [], []
    with open(log_path, 'r', errors='ignore') as f:
        for line in f:
            m = re.search(r'^Epoch (\d+)/\d+: loss=([\d.]+)', line)
            if m:
                epochs.append(int(m.group(1)))
                losses.append(float(m.group(2)))
    return epochs, losses


def plot_training_curves():
    logs = [
        ('lotsa_s50_v3_seq512', 'V3 50% seq512'),
        ('overnight_seq720_s50', 'Overnight seq720'),
        ('overnight_seq512_e80_s50', 'Overnight seq512 80ep'),
        ('overnight_nq128_s50', 'Overnight nq=128'),
        ('lotsa_s50_seq720_synth500k_evalbias', 'EvalBias seq720 + 500K'),
        ('lotsa_s50_seq512_synth500k_evalbias', 'EvalBias seq512 + 500K'),
        ('v2_xattn_s50_seq512', 'V2 CrossAttn (76M)'),
        ('v2_lrntrunk_s50_seq512', 'V2 LrnTrunk (57M)'),
    ]
    fig, ax = plt.subplots(figsize=(12, 6))
    for tag, label in logs:
        log_path = f'{LOG_DIR}/{tag}.log'
        if not os.path.exists(log_path):
            continue
        ep, ls = parse_train_log(log_path)
        if ep:
            ax.plot(ep, ls, label=f'{label} (min={min(ls):.3f})', linewidth=1.5, alpha=0.8)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Training Loss (MSE, normalized)')
    ax.set_title('Training Loss Curves Across All Experiments')
    ax.legend(loc='upper right', fontsize=9); ax.grid(alpha=0.3)
    plt.tight_layout()
    out = f'{FIG_DIR}/training_curves.png'
    plt.savefig(out, dpi=120); plt.close()
    print(f'Saved: {out}')


# ============================================================
# 2) Forecast comparison bar chart
# ============================================================
def plot_forecast_compare():
    """Load JSON results for forecast experiments."""
    candidates = [
        ('overnight_seq720_s50', 'Overnight seq720'),
        ('lotsa_s50_v3_seq512', 'V3 50% seq512'),
        ('synth_full_base_eval', 'Synth Full (no LOTSA)'),
    ]
    # Include overnight and eval jsons automatically
    for path in glob.glob(f'{RES_DIR}/eval_*.json') + glob.glob(f'{RES_DIR}/overnight_*.json'):
        name = os.path.basename(path).replace('.json', '')
        if not any(c[0] in path for c in candidates):
            candidates.append((name, name[:30]))

    data = {}
    for tag, label in candidates:
        path = f'{RES_DIR}/{tag}.json'
        if not os.path.exists(path):
            # Try with eval_ prefix
            path = f'{RES_DIR}/eval_{tag}.json'
            if not os.path.exists(path): continue
        with open(path) as f:
            d = json.load(f)
        # Extract per-dataset avg MSE
        row = {}
        for ds in ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'Weather']:
            v = d.get(f'{ds}_avg')
            if isinstance(v, dict):
                row[ds] = v.get('MSE', None)
            elif isinstance(v, (int, float)):
                row[ds] = v
        if any(v is not None for v in row.values()):
            data[label] = row

    if not data:
        print('No forecast results found')
        return

    datasets = ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'Weather']
    x = np.arange(len(datasets))
    n_bars = len(data) + 1  # +1 for FeDaL
    width = 0.8 / n_bars

    fig, ax = plt.subplots(figsize=(14, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, n_bars))
    for i, (label, row) in enumerate(data.items()):
        vals = [row.get(ds, 0) for ds in datasets]
        pos = x + (i - n_bars / 2 + 0.5) * width
        ax.bar(pos, vals, width, label=label, color=colors[i])
    # FeDaL reference
    pos = x + (n_bars - 1 - n_bars / 2 + 0.5) * width
    ax.bar(pos, [FEDAL_FORECAST[d] for d in datasets], width,
           label='FeDaL ZS (Table 4)', color='black', alpha=0.7, hatch='//')

    ax.set_xticks(x); ax.set_xticklabels(datasets)
    ax.set_ylabel('MSE (↓)'); ax.set_title('Forecast MSE: Ours vs FeDaL Zero-Shot')
    ax.legend(fontsize=9, loc='upper left'); ax.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    out = f'{FIG_DIR}/forecast_compare.png'
    plt.savefig(out, dpi=120); plt.close()
    print(f'Saved: {out}')


# ============================================================
# 3) Imputation comparison
# ============================================================
def plot_imputation_compare():
    paths = glob.glob(f'{RES_DIR}/*imputation*.json')
    if not paths:
        print('No imputation results found')
        return
    fig, ax = plt.subplots(figsize=(12, 6))
    datasets = ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'Weather']
    x = np.arange(len(datasets))
    n_bars = len(paths) + 1
    width = 0.8 / n_bars
    colors = plt.cm.tab10(np.linspace(0, 1, n_bars))

    for i, path in enumerate(paths):
        with open(path) as f:
            d = json.load(f)
        label = os.path.basename(path).replace('.json', '')[:30]
        vals = []
        for ds in datasets:
            entry = d.get(ds, {}).get('avg')
            vals.append(entry['MSE'] if isinstance(entry, dict) else 0)
        pos = x + (i - n_bars / 2 + 0.5) * width
        ax.bar(pos, vals, width, label=label, color=colors[i])

    pos = x + (n_bars - 1 - n_bars / 2 + 0.5) * width
    ax.bar(pos, [FEDAL_IMPUTATION[d] for d in datasets], width,
           label='FeDaL ZS', color='black', alpha=0.7, hatch='//')

    ax.set_xticks(x); ax.set_xticklabels(datasets)
    ax.set_ylabel('MSE (↓)'); ax.set_title('Imputation MSE (avg across 4 mask rates)')
    ax.legend(fontsize=9); ax.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    out = f'{FIG_DIR}/imputation_compare.png'
    plt.savefig(out, dpi=120); plt.close()
    print(f'Saved: {out}')


# ============================================================
# 4) M4 sMAPE comparison
# ============================================================
def plot_m4_compare():
    paths = glob.glob(f'{RES_DIR}/m4_*.json') + glob.glob(f'{RES_DIR}/*_m4.json')
    if not paths:
        print('No M4 results found')
        return
    fig, ax = plt.subplots(figsize=(12, 6))
    freqs = list(FEDAL_M4.keys())
    x = np.arange(len(freqs))
    n_bars = len(paths) + 1
    width = 0.8 / n_bars
    colors = plt.cm.tab10(np.linspace(0, 1, n_bars))

    for i, path in enumerate(paths):
        with open(path) as f:
            d = json.load(f)
        label = os.path.basename(path).replace('.json', '')[:30]
        vals = [d.get(fq, {}).get('sMAPE', 0) for fq in freqs]
        pos = x + (i - n_bars / 2 + 0.5) * width
        ax.bar(pos, vals, width, label=label, color=colors[i])

    pos = x + (n_bars - 1 - n_bars / 2 + 0.5) * width
    ax.bar(pos, [FEDAL_M4[fq] for fq in freqs], width,
           label='FeDaL', color='black', alpha=0.7, hatch='//')

    ax.set_xticks(x); ax.set_xticklabels(freqs)
    ax.set_ylabel('sMAPE (↓)'); ax.set_title('M4 Short-term Forecast')
    ax.legend(fontsize=9); ax.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    out = f'{FIG_DIR}/m4_compare.png'
    plt.savefig(out, dpi=120); plt.close()
    print(f'Saved: {out}')


# ============================================================
# 5) MSE by pred_len (direct vs rolling behavior)
# ============================================================
def plot_pred_len_trend():
    tags = [
        ('lotsa_s50_v3_seq512', 'V3 50% seq512 (rolling)'),
        ('overnight_seq720_s50', 'seq720 direct'),
    ]
    fig, ax = plt.subplots(figsize=(10, 6))
    pred_lens = [96, 192, 336, 720]
    for tag, label in tags:
        path = f'{RES_DIR}/{tag}.json'
        if not os.path.exists(path): continue
        with open(path) as f:
            d = json.load(f)
        # Average across datasets per pred_len
        for ds in ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'Weather']:
            mses = []
            for pl in pred_lens:
                v = d.get(f'{ds}_{pl}')
                if isinstance(v, dict):
                    mses.append(v.get('MSE', None))
                else:
                    mses.append(None)
            if all(m is not None for m in mses):
                ax.plot(pred_lens, mses, marker='o', label=f'{label} — {ds}', alpha=0.7)
    ax.set_xlabel('Prediction Length'); ax.set_ylabel('MSE')
    ax.set_title('MSE vs pred_len (Rolling vs Direct Prediction)')
    ax.legend(fontsize=8, loc='upper left'); ax.grid(alpha=0.3)
    plt.tight_layout()
    out = f'{FIG_DIR}/pred_len_trend.png'
    plt.savefig(out, dpi=120); plt.close()
    print(f'Saved: {out}')


# ============================================================
# 6) Failure analysis scatter (needs analyze_failures.py output)
# ============================================================
def plot_failure_scatter(npy_path):
    if not os.path.exists(npy_path):
        print(f'Skip (no file): {npy_path}')
        return
    data = np.load(npy_path, allow_pickle=True).item()
    mses = data['mses']
    feats = data['features']
    feat_names = data['feature_names']

    # Scatter: std (volatility) vs spec_entropy, color by mse
    std_idx = feat_names.index('std')
    ent_idx = feat_names.index('spec_entropy')
    fig, ax = plt.subplots(figsize=(10, 7))
    sc = ax.scatter(feats[:, std_idx], feats[:, ent_idx], c=mses,
                    cmap='viridis_r', s=10, alpha=0.6)
    ax.set_xlabel('Volatility (std)'); ax.set_ylabel('Spectral Entropy')
    ax.set_title('Per-sample MSE by Context Features')
    plt.colorbar(sc, ax=ax, label='MSE')
    plt.tight_layout()
    out = f'{FIG_DIR}/failure_scatter.png'
    plt.savefig(out, dpi=120); plt.close()
    print(f'Saved: {out}')


# ============================================================
# 7) Worst example predictions
# ============================================================
def plot_failure_examples(npy_path):
    if not os.path.exists(npy_path):
        print(f'Skip (no file): {npy_path}')
        return
    data = np.load(npy_path, allow_pickle=True).item()
    examples = data['worst_examples'][:8]  # Show 8 worst
    fig, axes = plt.subplots(4, 2, figsize=(14, 10))
    axes = axes.flatten()
    for ax, ex in zip(axes, examples):
        ctx = ex['context']; tgt = ex['target']; pr = ex['pred']
        # Plot context + target + prediction
        ctx_x = np.arange(len(ctx))
        fut_x = np.arange(len(ctx), len(ctx) + len(tgt))
        ax.plot(ctx_x, ctx, color='steelblue', linewidth=0.8, label='context')
        ax.plot(fut_x, tgt, color='black', linewidth=1.2, label='truth')
        ax.plot(fut_x, pr, color='red', linewidth=1.2, linestyle='--', label='pred')
        ax.set_title(f'{ex["dataset"]} (cluster {ex["cluster"]}) MSE={ex["mse"]:.3f}', fontsize=9)
        ax.legend(fontsize=7); ax.grid(alpha=0.3)
    plt.tight_layout()
    out = f'{FIG_DIR}/failure_examples.png'
    plt.savefig(out, dpi=120); plt.close()
    print(f'Saved: {out}')


# ============================================================
# 8) Synthetic patterns per domain
# ============================================================
def plot_synth_patterns():
    from experiments.exp_lotsa_scaling import SyntheticGapFiller
    ds = SyntheticGapFiller(n_samples=300, seq_len=500)

    # Find at least one sample per domain
    domains = list(SyntheticGapFiller.DOMAIN_WEIGHTS.keys())
    n_dom = len(domains)
    n_cols = 4
    n_rows = (n_dom + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 2.5))
    axes = axes.flatten()
    # Sample 1 window from each domain
    rng = np.random.RandomState(42)
    for i, dom in enumerate(domains):
        # Just pick the i-th sample (approximate — not exact domain mapping)
        ax = axes[i]
        idx = (i * 17) % len(ds)
        w = ds[idx].numpy()
        ax.plot(w, linewidth=0.6, color='steelblue')
        ax.set_title(dom, fontsize=9); ax.grid(alpha=0.3)
    for j in range(n_dom, len(axes)):
        axes[j].axis('off')
    plt.tight_layout()
    out = f'{FIG_DIR}/synth_patterns.png'
    plt.savefig(out, dpi=120); plt.close()
    print(f'Saved: {out}')


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    print('Generating visualizations...')
    plot_training_curves()
    plot_forecast_compare()
    plot_imputation_compare()
    plot_m4_compare()
    plot_pred_len_trend()
    # Only if failure analysis has been run:
    for npy in glob.glob(f'{RES_DIR}/*failures*.npy'):
        print(f'Processing failure file: {npy}')
        plot_failure_scatter(npy)
        plot_failure_examples(npy)
    plot_synth_patterns()
    print(f'\nAll figures in: {FIG_DIR}')
