"""
Train Split Diagnostic: How well does the model fit training distribution?

Loads checkpoint and evaluates on LOTSA + Synthetic samples (train distribution).
Reports per-domain MSE to diagnose:
  - Is model underfit (high train MSE)?
  - Is model learning each domain equally?
  - Generalization gap (train vs test MSE)

Usage:
  python experiments/eval_train_split.py \
      --ckpt checkpoints/overnight_seq720_s50.pth --seq_len 720 \
      --scale 5 --tag train_diag_seq720
"""
import sys, os, json, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch, torch.nn.functional as F
import numpy as np

from experiments.exp_lotsa_scaling import (
    OperatorModel, LOTSAScalingDataset, SyntheticGapFiller, collate_batch,
)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LOTSA_DIR = os.environ.get('LOTSA_DIR', './dataset/lotsa')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--decomp', type=int, default=0)
    p.add_argument('--seq_len', type=int, default=720)
    p.add_argument('--scale', type=int, default=5)
    p.add_argument('--n_eval', type=int, default=1000,
                   help='Samples per source (LOTSA/synth)')
    p.add_argument('--tag', type=str, required=True)
    args = p.parse_args()

    print('=' * 60)
    print(f'TRAIN SPLIT DIAGNOSTIC: {args.ckpt}')
    print(f'  seq_len={args.seq_len}, scale={args.scale}%')
    print('=' * 60)

    model = OperatorModel(seq_len=args.seq_len, use_latent_decomp=bool(args.decomp)).to(DEVICE)
    state = torch.load(args.ckpt, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    window_len = args.seq_len * 3

    # Load a small LOTSA subset
    print(f'\nLoading LOTSA {args.scale}%...')
    lotsa = LOTSAScalingDataset(LOTSA_DIR, args.scale, seq_len=window_len)
    print(f'Loading Synth...')
    synth = SyntheticGapFiller(n_samples=args.n_eval, seq_len=window_len)

    results = {}
    torch.manual_seed(42); np.random.seed(42)

    # Evaluate LOTSA (sample args.n_eval windows)
    for src_name, ds in [('LOTSA', lotsa), ('Synth', synth)]:
        if len(ds) == 0:
            continue
        idx = np.random.choice(len(ds), min(args.n_eval, len(ds)), replace=False)
        windows = [ds[i] for i in idx]
        mses_fc, mses_imp = [], []

        with torch.no_grad():
            # Forecast task
            for i in range(0, len(windows), 32):
                batch = collate_batch(windows[i:i+32], seq_len=args.seq_len, n_query=64)
                if batch is None: continue
                ctx, qt, qv = [x.to(DEVICE) for x in batch]
                # Split forecast vs imputation by t value
                pred = model.forward_train(ctx, qt)
                per_point_sq = (pred - qv) ** 2  # (B, nq)
                # Forecast queries: t >= 1
                fc_mask = (qt >= 1.0).float()
                imp_mask = (qt < 1.0).float()
                if fc_mask.sum() > 0:
                    mses_fc.append(((per_point_sq * fc_mask).sum() / fc_mask.sum()).item())
                if imp_mask.sum() > 0:
                    mses_imp.append(((per_point_sq * imp_mask).sum() / imp_mask.sum()).item())

        avg_fc = float(np.mean(mses_fc)) if mses_fc else None
        avg_imp = float(np.mean(mses_imp)) if mses_imp else None
        results[src_name] = {
            'forecast_MSE': avg_fc,
            'imputation_MSE': avg_imp,
            'n_samples': len(windows),
        }
        print(f'\n{src_name}: n={len(windows)}')
        print(f'  Forecast MSE: {avg_fc:.4f}' if avg_fc else '  Forecast: N/A')
        print(f'  Imputation MSE: {avg_imp:.4f}' if avg_imp else '  Imputation: N/A')

    # Per-synthetic domain breakdown
    print('\nPer-synth-domain breakdown...')
    domain_mses = {}
    # Group synth samples by domain using a re-run of small sample
    np.random.seed(42)
    rng = np.random.RandomState(42)
    domains = list(SyntheticGapFiller.DOMAIN_WEIGHTS.keys())
    probs = np.array([SyntheticGapFiller.DOMAIN_WEIGHTS[d] for d in domains])
    probs = probs / probs.sum()
    # Use the same seed as init, so order of domain choices matches
    n_check = min(len(synth), 2000)
    domain_labels = []
    for _ in range(n_check):
        domain_labels.append(rng.choice(domains, p=probs))

    with torch.no_grad():
        for i in range(0, n_check, 32):
            batch_idx = list(range(i, min(i + 32, n_check)))
            windows = [synth[j] for j in batch_idx]
            batch = collate_batch(windows, seq_len=args.seq_len, n_query=64)
            if batch is None: continue
            ctx, qt, qv = [x.to(DEVICE) for x in batch]
            pred = model.forward_train(ctx, qt)
            per_sample_mse = ((pred - qv) ** 2).mean(dim=-1).cpu().numpy()
            for k, j in enumerate(batch_idx):
                if k < len(per_sample_mse):
                    dom = domain_labels[j]
                    domain_mses.setdefault(dom, []).append(float(per_sample_mse[k]))

    print(f'\n{"Domain":<20} {"MSE":>10} {"n":>8}')
    print('-' * 40)
    for dom in domains:
        mses = domain_mses.get(dom, [])
        if mses:
            print(f'{dom:<20} {np.mean(mses):>10.4f} {len(mses):>8}')
            results[f'synth_{dom}'] = {'MSE': float(np.mean(mses)), 'n': len(mses)}

    os.makedirs('results', exist_ok=True)
    with open(f'results/{args.tag}.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved: results/{args.tag}.json')


if __name__ == '__main__':
    main()
