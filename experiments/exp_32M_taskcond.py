"""
Method 2: 32.5M with Task-Conditional Normalization

Forecast: center on last value (LVR style)
Imputation: center on mean (standard z-score)

Same architecture as 32.5M, just different normalization per task during training.

Usage:
  CUDA_VISIBLE_DEVICES=X python experiments/exp_32M_taskcond.py --seq 192 --tag tc_seq192
"""
import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, math, time, pyarrow as pa
from torch import optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset

DEVICE = torch.device(os.environ.get('CUDA_DEV', 'cuda'))

SEQ_LEN = None  # set by arg
PATCH_SIZE = 16
HIDDEN = 512
N_LAYERS = 6
WIDTH = 192
N_FREQ = 32
N_RBF = 20
TOP_K_IQ = 5
INFORMED_DIM = 1 + 2*TOP_K_IQ + 2

# Reuse model from exp_32M_longctx
from experiments.exp_32M_longctx import (
    Model32MLongCtx, make_datasets, collate_batch
)


def collate_batch_taskaware(windows, seq_len, mode='forecast', n_query=16, mr=0.375):
    """Same as collate_batch but returns (ctx, qt, qv, task_flag)."""
    batch = collate_batch(windows, seq_len, mode, n_query, mr)
    if batch is None: return None
    ctx, qt, qv = batch
    return ctx, qt, qv, mode


def train(model, datasets, save_path, seq_len, epochs=20, lr=3e-4, batch_size=64):
    n = sum(p.numel() for p in model.parameters())
    print(f'\n{"="*60}')
    print(f'32.5M Task-Conditional (SEQ={seq_len})')
    print(f'  Forecast: last value center')
    print(f'  Imputation: mean center')
    print(f'  Params: {n/1e6:.1f}M')
    print(f'{"="*60}')

    combined = ConcatDataset(datasets)
    dl = DataLoader(combined, batch_size=batch_size, shuffle=True,
                    num_workers=4, drop_last=True, pin_memory=True)
    steps = len(dl)
    print(f'Data: {len(combined):,}, Steps/ep: {steps:,}')

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        losses, fc_l, imp_l = [], [], []
        t0 = time.time()
        for i, batch_windows in enumerate(dl):
            is_forecast = np.random.rand() < 0.5
            mode = 'forecast' if is_forecast else 'impute'
            batch = collate_batch(batch_windows, seq_len, mode)
            if batch is None: continue
            ctx, qt, qv = [x.to(DEVICE) for x in batch]

            # Task-conditional normalization
            if is_forecast:
                # LVR style: center on last value, but use GLOBAL std for stability
                last = ctx[:, -1:]
                ctx_c = ctx - last
                s = ctx.std(dim=1, keepdim=True).clamp(min=0.1)  # use ctx std (not centered), clamp higher
                ctx_n = (ctx_c / s).clamp(-10, 10)
                # Target also needs adjustment
                qv_n = ((qv - last) / s).clamp(-10, 10)  # clamp target too
            else:
                # Standard z-score
                m = ctx.mean(dim=1, keepdim=True)
                s = ctx.std(dim=1, keepdim=True).clamp(min=1e-6)
                ctx_n = ((ctx - m) / s).clamp(-10, 10)
                qv_n = (qv - m) / s

            optimizer.zero_grad()
            pred_n = model.forward_train(ctx_n, qt)
            loss = F.mse_loss(pred_n, qv_n)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
            if is_forecast: fc_l.append(loss.item())
            else: imp_l.append(loss.item())

            if (i+1) % 500 == 0:
                print(f'  iter {i+1}/{steps}: loss={np.mean(losses[-500:]):.4f} '
                      f'fc={np.mean(fc_l[-500:]) if fc_l else 0:.4f} '
                      f'imp={np.mean(imp_l[-500:]) if imp_l else 0:.4f}')

        scheduler.step()
        el = time.time() - t0
        avg = np.mean(losses)
        print(f'Epoch {epoch+1}/{epochs}: loss={avg:.4f} '
              f'(fc={np.mean(fc_l) if fc_l else 0:.4f} imp={np.mean(imp_l) if imp_l else 0:.4f}) '
              f'lr={scheduler.get_last_lr()[0]:.6f} ({el:.0f}s)')
        if avg < best_loss:
            best_loss = avg
            torch.save(model.state_dict(), save_path)
            print(f'  Saved (best={best_loss:.4f})')

    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq', type=int, default=192)
    parser.add_argument('--tag', type=str, default='tc_seq192')
    args = parser.parse_args()

    SEQ_LEN = args.seq
    np.random.seed(42); torch.manual_seed(42)

    print(f'32.5M Task-Conditional [SEQ={SEQ_LEN}]')
    datasets = make_datasets(SEQ_LEN, pile_max=1000000, tempopfn_max=200000)
    model = Model32MLongCtx(SEQ_LEN).to(DEVICE)
    n = sum(p.numel() for p in model.parameters())
    print(f'Model: {n/1e6:.1f}M')

    save_path = f'checkpoints/32M_{args.tag}.pth'
    os.makedirs('checkpoints', exist_ok=True)
    train(model, datasets, save_path, SEQ_LEN, epochs=20, lr=3e-4)
    print('\nDONE')
