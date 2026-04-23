"""
Per-sample failure analysis on ETT/Weather eval.

Runs eval, stores per-sample MSE + context stats.
Extracts worst 10% samples.
Clusters by statistical features (volatility, skew, acf, spec entropy, ...).

Outputs:
  results/{tag}_failures.json — cluster stats + exemplars
  results/{tag}_failures.npy — full per-sample info for viz

Usage:
  python experiments/analyze_failures.py \
      --ckpt checkpoints/overnight_seq720_s50.pth --seq_len 720 \
      --pred_len 192 --tag failures_seq720_pred192
"""
import sys, os, json, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn.functional as F
import numpy as np
from types import SimpleNamespace

from experiments.exp_lotsa_scaling import OperatorModel

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATASETS = {
    'ETTh1':   ('ETTh1',  './dataset/ETT-small/', 'ETTh1.csv', 7),
    'ETTh2':   ('ETTh2',  './dataset/ETT-small/', 'ETTh2.csv', 7),
    'ETTm1':   ('ETTm1',  './dataset/ETT-small/', 'ETTm1.csv', 7),
    'ETTm2':   ('ETTm2',  './dataset/ETT-small/', 'ETTm2.csv', 7),
    'Weather': ('custom', './dataset/weather/',  'weather.csv', 21),
}


def context_features(x):
    """Statistical features of a context window (normalized, 1D numpy)."""
    x = np.asarray(x, dtype=np.float64)
    feats = {}
    feats['mean'] = float(x.mean())
    feats['std'] = float(x.std())
    feats['skew'] = float(((x - x.mean()) ** 3).mean() / (x.std() ** 3 + 1e-8))
    feats['kurt'] = float(((x - x.mean()) ** 4).mean() / (x.std() ** 4 + 1e-8) - 3)
    feats['range'] = float(x.max() - x.min())
    feats['slope'] = float(np.polyfit(np.arange(len(x)), x, 1)[0])
    # FFT power concentration (max bin / total)
    fft_mag = np.abs(np.fft.rfft(x))
    fft_mag[0] = 0
    total = (fft_mag ** 2).sum() + 1e-8
    feats['max_freq_ratio'] = float((fft_mag.max() ** 2) / total)
    # ACF at lag 1, 24
    if len(x) > 24:
        feats['acf_1'] = float(np.corrcoef(x[:-1], x[1:])[0, 1])
        feats['acf_24'] = float(np.corrcoef(x[:-24], x[24:])[0, 1])
    # Spectral entropy
    power = fft_mag ** 2
    prob = power / (power.sum() + 1e-10)
    feats['spec_entropy'] = float(-(prob * np.log(prob + 1e-10)).sum())
    # Number of zero crossings (complexity proxy)
    feats['zc_rate'] = float(np.sum(np.diff(np.sign(x - x.mean())) != 0) / len(x))
    return feats


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--decomp', type=int, default=0)
    p.add_argument('--seq_len', type=int, default=720)
    p.add_argument('--pred_len', type=int, default=192)
    p.add_argument('--tag', type=str, required=True)
    p.add_argument('--max_samples', type=int, default=2000, help='Max samples per dataset')
    args = p.parse_args()

    from data_provider.data_factory import data_provider

    print('=' * 60)
    print(f'FAILURE ANALYSIS: {args.ckpt}')
    print(f'  seq_len={args.seq_len}, pred_len={args.pred_len}')
    print('=' * 60)

    model = OperatorModel(seq_len=args.seq_len, use_latent_decomp=bool(args.decomp)).to(DEVICE)
    state = torch.load(args.ckpt, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    all_samples = []  # list of {dataset, sample_idx, mse, features, context (raw), target, pred}

    for dn, (d, root, f, enc_in) in DATASETS.items():
        print(f'\n{dn}...')
        try:
            a = SimpleNamespace(seq_len=args.seq_len, pred_len=args.pred_len, label_len=48, data=d,
                                root_path=root, data_path=f, features='M', target='OT', freq='h',
                                embed='timeF', enc_in=enc_in, dec_in=enc_in, c_out=enc_in,
                                num_workers=2, batch_size=32, exp_name='MTSF', ordered_data=False,
                                data_amount=-1, combine_Gaussian_datasets=False, synthetic_data_path='',
                                synthetic_root_path='./', synthetic_length=1024, stride=-1)
            _, tdl = data_provider(a, 'test')

            count = 0
            with torch.no_grad():
                for bx, by, _, _ in tdl:
                    if count >= args.max_samples:
                        break
                    bx = bx.float().to(DEVICE)
                    B, S, C = bx.shape
                    for b_idx in range(B):
                        if count >= args.max_samples:
                            break
                        for ch in range(C):
                            x_ch = bx[b_idx, :, ch]
                            if S >= args.seq_len:
                                x_ctx = x_ch[-args.seq_len:]
                            else:
                                x_ctx = F.pad(x_ch, (args.seq_len - S, 0))
                            m = x_ctx.mean()
                            s = x_ctx.std().clamp(min=1e-6)
                            x_n = ((x_ctx - m) / s).clamp(-10, 10).unsqueeze(0)
                            pred = model.forecast(x_n, n=args.pred_len).squeeze(0)
                            pred = pred * s + m
                            true = by[b_idx, -args.pred_len:, ch].to(DEVICE)
                            mse = ((pred - true) ** 2).mean().item()

                            feats = context_features(x_n.squeeze(0).cpu().numpy())
                            all_samples.append({
                                'dataset': dn,
                                'channel': int(ch),
                                'sample_idx': int(b_idx + count),
                                'mse': float(mse),
                                'features': feats,
                                'context': x_n.squeeze(0).cpu().numpy().astype(np.float32),
                                'target': true.cpu().numpy().astype(np.float32),
                                'pred': pred.cpu().numpy().astype(np.float32),
                            })
                        count += 1
            print(f'  collected {count} samples')
        except Exception as e:
            print(f'  ERROR: {e}')

    print(f'\nTotal samples: {len(all_samples)}')
    mses = np.array([s['mse'] for s in all_samples])
    print(f'MSE: mean={mses.mean():.4f}, median={np.median(mses):.4f}, max={mses.max():.4f}')

    # Identify worst 10%
    threshold = np.percentile(mses, 90)
    worst_mask = mses >= threshold
    worst_samples = [all_samples[i] for i in np.where(worst_mask)[0]]
    print(f'\nWorst 10% ({len(worst_samples)} samples), MSE >= {threshold:.4f}:')

    # Per-dataset breakdown of worst
    worst_per_ds = {}
    for s in worst_samples:
        worst_per_ds.setdefault(s['dataset'], 0)
        worst_per_ds[s['dataset']] += 1
    for ds, n in worst_per_ds.items():
        total_ds = sum(1 for s in all_samples if s['dataset'] == ds)
        print(f'  {ds}: {n}/{total_ds} ({100*n/total_ds:.1f}%)')

    # Cluster worst by features (k-means)
    from sklearn.cluster import KMeans
    feat_names = list(worst_samples[0]['features'].keys())
    feat_matrix = np.array([[s['features'][fn] for fn in feat_names] for s in worst_samples])
    # Normalize features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X = scaler.fit_transform(feat_matrix)
    n_clusters = min(4, len(worst_samples) // 10 + 1)
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(X)
    labels = km.labels_

    # Cluster characterization: mean feature values per cluster
    clusters = {}
    for k in range(n_clusters):
        mask = labels == k
        if mask.sum() == 0:
            continue
        cluster_samples = [worst_samples[i] for i in np.where(mask)[0]]
        cluster_mses = [s['mse'] for s in cluster_samples]
        cluster_feats = {fn: float(np.mean([s['features'][fn] for s in cluster_samples]))
                         for fn in feat_names}
        # Datasets in this cluster
        ds_counts = {}
        for s in cluster_samples:
            ds_counts.setdefault(s['dataset'], 0)
            ds_counts[s['dataset']] += 1
        clusters[f'cluster_{k}'] = {
            'n_samples': int(mask.sum()),
            'avg_mse': float(np.mean(cluster_mses)),
            'mean_features': cluster_feats,
            'dataset_distribution': ds_counts,
        }

    print(f'\n{n_clusters} failure clusters:')
    for cname, cdata in clusters.items():
        print(f'\n{cname}: n={cdata["n_samples"]}, avg_mse={cdata["avg_mse"]:.4f}')
        print(f'  datasets: {cdata["dataset_distribution"]}')
        top_feats = sorted(cdata['mean_features'].items(), key=lambda x: abs(x[1]), reverse=True)[:3]
        print(f'  top features: {[(f, round(v, 3)) for f, v in top_feats]}')

    # Save results
    os.makedirs('results', exist_ok=True)
    summary = {
        'total_samples': len(all_samples),
        'mean_mse': float(mses.mean()),
        'median_mse': float(np.median(mses)),
        'worst10_threshold': float(threshold),
        'worst_by_dataset': worst_per_ds,
        'clusters': clusters,
    }
    with open(f'results/{args.tag}.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # Save full data (without features_matrix raw) for viz
    np.save(f'results/{args.tag}.npy', {
        'mses': mses,
        'features': feat_matrix,
        'feature_names': feat_names,
        'cluster_labels': labels,
        'worst_mask': worst_mask,
        'datasets': [s['dataset'] for s in all_samples],
        'worst_examples': [{
            'dataset': s['dataset'],
            'mse': s['mse'],
            'context': s['context'],
            'target': s['target'],
            'pred': s['pred'],
            'cluster': int(labels[i]),
        } for i, s in enumerate(worst_samples)],
    }, allow_pickle=True)

    print(f'\nSaved: results/{args.tag}.json, results/{args.tag}.npy')


if __name__ == '__main__':
    main()
