"""
Train a PFN time-series model on a synthetic prior.

Sampling strategy per training step:
  1. Sample a synth window of length L (full).
  2. Pick context fraction r ∈ [0.4, 0.85] → N_ctx = int(L * r).
  3. Remaining L - N_ctx are query positions (forecast / interior fill).
  4. Optionally drop random context points to simulate irregular sampling.

Each (t, y) pair becomes a token. Timestamps are normalized to [0, 1] within
the window so the time-embedding stays in a stable range.

This is a pure-synth experiment: no LOTSA. (Set --use_lotsa 0 to be explicit;
the script ignores LOTSA path entirely.)

Usage:
  CUDA_VISIBLE_DEVICES=0 python experiments/exp_pfn_train.py \
    --tag pfn_synth500K_v1 --synth_n 500000 --epochs 5 --batch_size 256 \
    --d_model 512 --n_layers 8
"""
import sys, os, math, time, argparse, contextlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader

from experiments.exp_lotsa_scaling import SyntheticGapFiller
from experiments.pfn_model import PFNTimeSeriesModel, gaussian_nll

DEVICE = torch.device(os.environ.get('CUDA_DEV', 'cuda'))


# ============================================================
# Per-sample query sampler
# ============================================================
def make_pfn_collate(seq_len, n_ctx_min=80, n_ctx_max=600,
                     n_qry=64, irreg_drop_p=0.0):
    """
    Returns a collate_fn that, for each window in the batch,
      - normalizes y to per-sample (mean, std)
      - samples N_ctx in [n_ctx_min, n_ctx_max]
      - splits indices into ctx (random subset of [0, seq_len)) and qry
        (random subset of remaining indices)
      - returns batched tensors of t_ctx, y_ctx, t_qry, y_qry

    All per-sample N_ctx are clipped to a single batch-wide N_ctx for tensor
    stacking (we use min across the sample). N_qry is fixed.

    Timestamps are scaled to [0, 1] (i / seq_len).
    """
    def _collate(batch_windows):
        # batch_windows is a list of np.array (seq_len,) windows from synth
        B = len(batch_windows)
        # Sample one N_ctx for the whole batch (simpler, batch-uniform)
        N_ctx = np.random.randint(n_ctx_min, n_ctx_max + 1)
        N_ctx = min(N_ctx, seq_len - n_qry - 4)
        N_ctx = max(N_ctx, 8)

        t_ctx_list, y_ctx_list, t_qry_list, y_qry_list = [], [], [], []
        for w in batch_windows:
            w = np.asarray(w, dtype=np.float32)
            # Permute then split: ctx = first N_ctx, qry = next n_qry
            perm = np.random.permutation(seq_len)
            ctx_idx = perm[:N_ctx]
            qry_idx = perm[N_ctx:N_ctx + n_qry]

            # Optional irregular dropping (some ctx points removed)
            if irreg_drop_p > 0:
                keep = np.random.rand(N_ctx) > irreg_drop_p
                if keep.sum() < 8:
                    keep[:8] = True
                ctx_idx = ctx_idx[keep]

            # Per-sample normalization based on context only (mask-aware)
            y_ctx_raw = w[ctx_idx]
            mu = y_ctx_raw.mean()
            sigma = y_ctx_raw.std()
            sigma = max(sigma, 1e-6)
            y_ctx = ((y_ctx_raw - mu) / sigma).clip(-10, 10)
            y_qry = ((w[qry_idx] - mu) / sigma).clip(-10, 10)

            t_ctx_list.append(ctx_idx.astype(np.float32) / seq_len)
            y_ctx_list.append(y_ctx.astype(np.float32))
            t_qry_list.append(qry_idx.astype(np.float32) / seq_len)
            y_qry_list.append(y_qry.astype(np.float32))

        # Pad ctx to max-len in batch (since irreg_drop may produce variable lengths)
        max_ctx = max(len(a) for a in t_ctx_list)
        t_ctx_pad = np.zeros((B, max_ctx), dtype=np.float32)
        y_ctx_pad = np.zeros((B, max_ctx), dtype=np.float32)
        ctx_kpm = np.ones((B, max_ctx), dtype=bool)  # True = pad position
        for i, (tc, yc) in enumerate(zip(t_ctx_list, y_ctx_list)):
            n = len(tc)
            t_ctx_pad[i, :n] = tc
            y_ctx_pad[i, :n] = yc
            ctx_kpm[i, :n] = False

        t_qry = np.stack(t_qry_list)
        y_qry = np.stack(y_qry_list)

        # key_padding_mask must cover the full sequence (ctx + qry).
        # Query positions never padded.
        full_kpm = np.concatenate(
            [ctx_kpm, np.zeros((B, n_qry), dtype=bool)], axis=1
        )

        return (
            torch.from_numpy(t_ctx_pad),
            torch.from_numpy(y_ctx_pad),
            torch.from_numpy(t_qry),
            torch.from_numpy(y_qry),
            torch.from_numpy(full_kpm),
        )

    return _collate


# ============================================================
# Training
# ============================================================
def train_pfn(model, dataset, save_path, epochs=5, lr=1e-4,
              batch_size=256, seq_len=2160, n_qry=64,
              n_ctx_min=80, n_ctx_max=1024, irreg_drop_p=0.0,
              use_nll=False, amp=True, log_every=200):
    print(f'\n{"="*60}')
    print(f'PFN Training')
    print(f'  Model: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params')
    print(f'  Data : {len(dataset):,} windows, seq_len={seq_len}')
    print(f'  Ctx  : {n_ctx_min} ≤ N_ctx ≤ {n_ctx_max}, irreg_drop_p={irreg_drop_p}')
    print(f'  Qry  : {n_qry}, loss={"nll" if use_nll else "mse"}, AMP={"bf16" if amp else "off"}')
    print(f'{"="*60}')

    collate = make_pfn_collate(seq_len, n_ctx_min, n_ctx_max, n_qry, irreg_drop_p)
    dl = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=8, drop_last=True, pin_memory=True,
        persistent_workers=True, collate_fn=collate, prefetch_factor=4,
    )

    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    amp_ctx = (lambda: torch.autocast(device_type='cuda', dtype=torch.bfloat16)) \
              if amp else (lambda: contextlib.nullcontext())

    best = float('inf')
    for ep in range(epochs):
        model.train()
        t0 = time.time()
        losses = []
        for i, (t_ctx, y_ctx, t_qry, y_qry, kpm) in enumerate(dl):
            t_ctx = t_ctx.to(DEVICE, non_blocking=True)
            y_ctx = y_ctx.to(DEVICE, non_blocking=True)
            t_qry = t_qry.to(DEVICE, non_blocking=True)
            y_qry = y_qry.to(DEVICE, non_blocking=True)
            kpm   = kpm.to(DEVICE, non_blocking=True)

            opt.zero_grad()
            with amp_ctx():
                if use_nll:
                    mean, log_sigma = model(t_ctx, y_ctx, t_qry, key_padding_mask=kpm)
                    loss = gaussian_nll(y_qry, mean, log_sigma)
                else:
                    pred = model(t_ctx, y_ctx, t_qry, key_padding_mask=kpm)
                    loss = F.mse_loss(pred, y_qry)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
            if (i + 1) % log_every == 0:
                avg = sum(losses[-log_every:]) / log_every
                print(f'  iter {i+1}/{len(dl)}: loss={avg:.4f}', flush=True)

        ep_loss = sum(losses) / max(len(losses), 1)
        elapsed = time.time() - t0
        print(f'  Epoch {ep+1}/{epochs}: loss={ep_loss:.4f}  ({elapsed:.0f}s)')
        sched.step()
        if ep_loss < best:
            best = ep_loss
            torch.save(model.state_dict(), save_path)
            print(f'  Saved → {save_path}')


# ============================================================
# Main
# ============================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tag', required=True)
    p.add_argument('--synth_n', type=int, default=500000)
    p.add_argument('--seq_len', type=int, default=2160)
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--n_qry', type=int, default=64)
    p.add_argument('--n_ctx_min', type=int, default=80)
    p.add_argument('--n_ctx_max', type=int, default=1024)
    p.add_argument('--irreg_drop_p', type=float, default=0.0)
    p.add_argument('--d_model', type=int, default=512)
    p.add_argument('--n_layers', type=int, default=8)
    p.add_argument('--n_heads', type=int, default=8)
    p.add_argument('--use_nll', type=int, default=0)
    p.add_argument('--amp', type=int, default=1)
    args = p.parse_args()

    print('=' * 70)
    print(f'PFN: {args.tag}')
    print(f'  synth_n={args.synth_n}, epochs={args.epochs}, lr={args.lr}')
    print('=' * 70)

    print('Loading synth dataset...')
    ds = SyntheticGapFiller(n_samples=args.synth_n, seq_len=args.seq_len)
    print(f'  loaded {len(ds):,} windows')

    model = PFNTimeSeriesModel(
        d_model=args.d_model, n_layers=args.n_layers, n_heads=args.n_heads,
        dist_output=bool(args.use_nll),
    ).to(DEVICE)

    os.makedirs('checkpoints', exist_ok=True)
    save_path = f'checkpoints/{args.tag}.pth'
    train_pfn(
        model, ds, save_path, epochs=args.epochs, lr=args.lr,
        batch_size=args.batch_size, seq_len=args.seq_len,
        n_qry=args.n_qry, n_ctx_min=args.n_ctx_min, n_ctx_max=args.n_ctx_max,
        irreg_drop_p=args.irreg_drop_p, use_nll=bool(args.use_nll),
        amp=bool(args.amp),
    )


if __name__ == '__main__':
    main()
