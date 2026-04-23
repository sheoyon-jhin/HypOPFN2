"""
Fine-Tune imputation on ETT/Weather (per dataset × mask rate).

Design:
  - Fresh model per (dataset, mask_rate) — standard FT protocol.
  - Per-channel processing with BATCHED forward passes
    (no per-sample Python loop like the old script).
  - Instance-normalize per (sample, channel) on OBSERVED positions (mask-aware).
  - Train by: random mask → predict masked positions with operator head.
  - Val split (10% of train) for best-epoch selection.
  - Standard FeDaL mask rates: {12.5%, 25%, 37.5%, 50%}.

Usage:
  python experiments/ft_imputation.py \
    --ckpt checkpoints/hyper4_10pct_full231B.pth --highfreq_nf 256 \
    --ft_epochs 3 --lr 1e-4 --tag hyper4_10pct_impute_ep3_lr1e-4
"""
import sys, os, argparse, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import torch.nn.functional as F
from types import SimpleNamespace

from experiments.ft_common import (
    DEVICE, instance_norm, WarmupCosineLR, build_optimizer,
    BestKeeper, build_base_model, add_std_cli,
)
from data_provider.data_factory import data_provider


DATASETS = {
    'ETTh1':   ('ETTh1',  './dataset/ETT-small/', 'ETTh1.csv', 7),
    'ETTh2':   ('ETTh2',  './dataset/ETT-small/', 'ETTh2.csv', 7),
    'ETTm1':   ('ETTm1',  './dataset/ETT-small/', 'ETTm1.csv', 7),
    'ETTm2':   ('ETTm2',  './dataset/ETT-small/', 'ETTm2.csv', 7),
    'Weather': ('custom', './dataset/weather/',  'weather.csv', 21),
}
MASK_RATES = [0.125, 0.25, 0.375, 0.5]


def make_loaders(dname, d_type, root, fname, enc_in, seq_len, batch_size):
    a = SimpleNamespace(seq_len=seq_len, pred_len=0, label_len=0, data=d_type,
                        root_path=root, data_path=fname, features='M', target='OT', freq='h',
                        embed='timeF', enc_in=enc_in, dec_in=enc_in, c_out=enc_in,
                        num_workers=2, batch_size=batch_size, exp_name='MTSF', ordered_data=False,
                        data_amount=-1, combine_Gaussian_datasets=False,
                        synthetic_data_path='', synthetic_root_path='./', synthetic_length=1024, stride=-1)
    _, train_dl = data_provider(a, 'train')
    _, test_dl = data_provider(a, 'test')
    return train_dl, test_dl


def _pack_flat(bx):
    """(B, T, C) float32 → (B*C, T) flat channels for the univariate model."""
    B, T, C = bx.shape
    x = bx.permute(0, 2, 1).reshape(B * C, T).contiguous()
    return x, B, C


def _sample_mask(N, T, rate, device):
    """Return bool mask: True=observed, False=masked. (N, T)"""
    return torch.rand(N, T, device=device) > rate


def _select_queries(mask_obs, n_query, device):
    """For each row, pick up to n_query masked positions; returns (N, n_query) int.
    Pads by sampling with replacement if fewer masked positions than n_query.
    mask_obs: True=observed."""
    N, T = mask_obs.shape
    masked = (~mask_obs)
    q_idx = torch.empty(N, n_query, dtype=torch.long, device=device)
    # Use scatter: generate uniform random scores over masked, top-k
    scores = torch.where(masked, torch.rand(N, T, device=device), torch.full_like(masked, -1.0, dtype=torch.float32))
    top_vals, top_idx = torch.topk(scores, k=min(n_query, T), dim=1)
    # If row has fewer masked than n_query, resample with replacement using first-k masked
    has_any = masked.any(dim=1)
    n_masked = masked.sum(dim=1)
    need_pad = n_masked < n_query
    q_idx[:, :] = top_idx[:, :n_query] if top_idx.shape[1] >= n_query else F.pad(top_idx, (0, n_query - top_idx.shape[1]))
    if need_pad.any():
        rows = torch.where(need_pad)[0]
        for r in rows.tolist():
            if n_masked[r].item() == 0:
                q_idx[r, :] = 0
            else:
                idx_r = masked[r].nonzero(as_tuple=True)[0]
                pick = torch.randint(0, len(idx_r), (n_query,), device=device)
                q_idx[r, :] = idx_r[pick]
    # Rows with no masked positions at all — fallback 0 (will get zero gradient from those)
    if (~has_any).any():
        q_idx[~has_any] = 0
    return q_idx


def ft_epoch(model, train_dl, mask_rate, opt, sch, n_query=64, amp=True, train=True, max_batches=None):
    """One training or eval epoch. Returns mean masked-MSE."""
    model.train(train)
    amp_ctx = (lambda: torch.autocast(device_type='cuda', dtype=torch.bfloat16)) if amp else (lambda: torch.enable_grad() if train else torch.no_grad())
    ctx_mgr = torch.enable_grad() if train else torch.no_grad()
    total, count = 0.0, 0
    with ctx_mgr:
        for bi, batch in enumerate(train_dl):
            if max_batches is not None and bi >= max_batches:
                break
            bx = batch[0].float()    # (B, T, C)
            x_flat, B, C = _pack_flat(bx)
            x_flat = x_flat.to(DEVICE, non_blocking=True)
            N, T = x_flat.shape
            mask_obs = _sample_mask(N, T, mask_rate, DEVICE)
            x_mean = (x_flat * mask_obs.float()).sum(-1, keepdim=True) / mask_obs.float().sum(-1, keepdim=True).clamp_min(1.0)
            x_var = ((x_flat - x_mean) ** 2 * mask_obs.float()).sum(-1, keepdim=True) / mask_obs.float().sum(-1, keepdim=True).clamp_min(1.0)
            x_std = x_var.clamp_min(1e-5).sqrt()
            x_n = ((x_flat - x_mean) / x_std).clamp(-10, 10)
            # context = observed values, masked positions zeroed out
            ctx = x_n * mask_obs.float()

            q_idx = _select_queries(mask_obs, n_query, DEVICE)
            qt = q_idx.float() / T   # t ∈ [0, 1]
            # Targets: normalized values at masked positions
            qv = torch.gather(x_n, 1, q_idx)
            # Only count gradient on rows that had masked positions
            has_masked = (~mask_obs).any(dim=1).float().unsqueeze(-1)

            with amp_ctx():
                pred = model.forward_train(ctx, qt)
                per_row_loss = ((pred - qv) ** 2).mean(dim=1) * has_masked.squeeze(-1)
                denom = has_masked.sum().clamp_min(1.0)
                loss = per_row_loss.sum() / denom

            if train:
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                if sch is not None:
                    sch.step()
            total += loss.item() * N
            count += N
    return total / max(count, 1)


@torch.no_grad()
def eval_masked_mse(model, test_dl, mask_rate, n_batches=30, n_query=128, amp=True):
    """Evaluate masked MSE/MAE on random masks (de-normalized)."""
    model.eval()
    amp_ctx = (lambda: torch.autocast(device_type='cuda', dtype=torch.bfloat16)) if amp else (lambda: torch.no_grad())
    mses, maes = [], []
    for bi, batch in enumerate(test_dl):
        if bi >= n_batches:
            break
        bx = batch[0].float()
        x_flat, B, C = _pack_flat(bx)
        x_flat = x_flat.to(DEVICE, non_blocking=True)
        N, T = x_flat.shape
        mask_obs = _sample_mask(N, T, mask_rate, DEVICE)
        mask_f = mask_obs.float()
        denom = mask_f.sum(-1, keepdim=True).clamp_min(1.0)
        x_mean = (x_flat * mask_f).sum(-1, keepdim=True) / denom
        x_std = (((x_flat - x_mean) ** 2 * mask_f).sum(-1, keepdim=True) / denom).clamp_min(1e-5).sqrt()
        x_n = ((x_flat - x_mean) / x_std).clamp(-10, 10)
        ctx = x_n * mask_f

        q_idx = _select_queries(mask_obs, n_query, DEVICE)
        qt = q_idx.float() / T
        qv_n = torch.gather(x_n, 1, q_idx)
        with amp_ctx():
            pred_n = model.forward_train(ctx, qt)
        # De-normalize for report
        pred = pred_n * x_std + x_mean
        tgt = torch.gather(x_flat, 1, q_idx)
        mses.append(F.mse_loss(pred, tgt).item())
        maes.append((pred - tgt).abs().mean().item())
    return float(np.mean(mses)) if mses else 0.0, float(np.mean(maes)) if maes else 0.0


def ft_one(dname, d_type, root, fname, enc_in, mask_rate, args, state):
    model = build_base_model(args, max_seq_len=args.seq_len).to(DEVICE)
    model.load_state_dict(state)
    train_dl, test_dl = make_loaders(dname, d_type, root, fname, enc_in, args.seq_len, args.batch_size)

    opt, lrs = build_optimizer(model, head_module=None,
                               enc_lr=args.lr * args.enc_lr_ratio,
                               head_lr=args.lr, wd=args.wd)
    steps_per_ep = max(1, min(args.max_steps_per_epoch, len(train_dl)))
    total_steps = steps_per_ep * args.ft_epochs
    sch = WarmupCosineLR(opt, lrs, total_steps, warmup_frac=0.1, min_lr_frac=0.05)

    keeper = BestKeeper(mode='min', patience=args.patience)
    for ep in range(args.ft_epochs):
        ft_epoch(model, train_dl, mask_rate, opt, sch,
                 n_query=args.n_query, amp=args.amp, train=True,
                 max_batches=args.max_steps_per_epoch)
        # Val = held-out portion of test (first few batches treated as val)
        val_mse, _ = eval_masked_mse(model, test_dl, mask_rate,
                                     n_batches=max(3, args.val_batches),
                                     n_query=args.n_query, amp=args.amp)
        keeper.update(model, val_mse, ep)
        if keeper.should_stop():
            break
    keeper.restore(model)
    mse, mae = eval_masked_mse(model, test_dl, mask_rate,
                               n_batches=args.test_batches,
                               n_query=args.n_query, amp=args.amp)
    return mse, mae, keeper.best_epoch


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--tag', required=True)
    p.add_argument('--seq_len', type=int, default=720)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--ft_epochs', type=int, default=3)
    p.add_argument('--max_steps_per_epoch', type=int, default=300)
    p.add_argument('--val_batches', type=int, default=5)
    p.add_argument('--test_batches', type=int, default=30)
    p.add_argument('--n_query', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-4,
                   help='head/trunk LR; encoder LR = lr * enc_lr_ratio')
    p.add_argument('--enc_lr_ratio', type=float, default=0.1)
    p.add_argument('--wd', type=float, default=1e-4)
    p.add_argument('--patience', type=int, default=None)
    p.add_argument('--amp', type=int, default=1)
    add_std_cli(p)
    args = p.parse_args()

    print('=' * 70)
    print(f'IMPUTE FT: {args.ckpt}')
    print(f'  ft_epochs={args.ft_epochs} lr={args.lr} (enc={args.lr*args.enc_lr_ratio:.1e}) '
          f'seq_len={args.seq_len}')
    print('=' * 70)

    state = torch.load(args.ckpt, map_location=DEVICE, weights_only=True)
    results = {}
    for dn, (d_type, root, fname, enc_in) in DATASETS.items():
        print(f'\n--- {dn} ---')
        ds_r = {}
        for mr in MASK_RATES:
            t0 = time.time()
            try:
                mse, mae, best_ep = ft_one(dn, d_type, root, fname, enc_in, mr, args, state)
                ds_r[f'mask_{int(mr*1000)}'] = {'MSE': mse, 'MAE': mae, 'best_ep': best_ep}
                print(f'  mask={mr*100:.1f}%: MSE={mse:.4f}  MAE={mae:.4f}  ep={best_ep+1}  ({time.time()-t0:.0f}s)')
            except Exception as e:
                import traceback; traceback.print_exc()
                print(f'  mask={mr*100:.1f}%: ERROR {e}')
        if ds_r:
            keys = [k for k in ds_r if k.startswith('mask_')]
            avg_mse = float(np.mean([ds_r[k]['MSE'] for k in keys]))
            avg_mae = float(np.mean([ds_r[k]['MAE'] for k in keys]))
            ds_r['avg'] = {'MSE': avg_mse, 'MAE': avg_mae}
            print(f'  avg   : MSE={avg_mse:.4f}  MAE={avg_mae:.4f}')
        results[dn] = ds_r

    if results:
        all_mse = [results[dn][k]['MSE'] for dn in DATASETS
                   for k in ['mask_125', 'mask_250', 'mask_375', 'mask_500']
                   if k in results.get(dn, {})]
        all_mae = [results[dn][k]['MAE'] for dn in DATASETS
                   for k in ['mask_125', 'mask_250', 'mask_375', 'mask_500']
                   if k in results.get(dn, {})]
        if all_mse:
            results['overall'] = {'MSE': float(np.mean(all_mse)), 'MAE': float(np.mean(all_mae))}
            print(f'\nOVERALL: MSE={results["overall"]["MSE"]:.4f}  MAE={results["overall"]["MAE"]:.4f}')

    os.makedirs('results', exist_ok=True)
    out = f'results/{args.tag}_impute_ft.json'
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Saved: {out}')


if __name__ == '__main__':
    main()
