"""
개선된 Pre-training: lr 높임 + batch 크게 + nt weight 높임 + warmup
141M Hyper Trunk + Next-token + Masked Recon

개선사항:
  1. lr: 0.0003 → 0.001 (3배)
  2. batch_size: 64 → 256 (4배)
  3. loss = 2.0 * nt + 0.5 * recon (nt에 집중)
  4. warmup: 첫 2000 step lr 서서히 올림
  5. spectral_branch: True

사용법:
  CUDA_VISIBLE_DEVICES=X python experiments/exp_improved_pretrain.py 2>&1 | tee log/scaleup/improved_141m.log
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import optim
from torch.utils.data import DataLoader, random_split
from types import SimpleNamespace
import time

from model.DeepONetHyperMoE import Model
from data_provider.pile_dataset import PilePretrainDataset
from experiments.eval_all_tasks import eval_forecasting, eval_imputation, eval_classification, eval_short_term, print_summary


def get_model_args():
    return SimpleNamespace(
        seq_len=96, pred_len=96, use_norm=True,
        deeponet_width=192,       # 141M
        n_experts=4,
        branch_depth=4,
        trunk_depth=2,
        activation='gelu',
        dropout=0.1,
        branch_hidden=768,
        spectral_branch=True,     # ← ON
        skip_mode='none',
        use_cross_channel=False,
        trunk_basis='mixed',
        encoder_type='patch_attn',
        loss='MSE',
    )


class WarmupCosineScheduler:
    """Linear warmup + cosine decay."""
    def __init__(self, optimizer, warmup_steps, total_steps, peak_lr):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.peak_lr = peak_lr
        self.step_count = 0

    def step(self):
        self.step_count += 1
        if self.step_count <= self.warmup_steps:
            lr = self.peak_lr * (self.step_count / self.warmup_steps)
        else:
            progress = (self.step_count - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = self.peak_lr * 0.5 * (1 + np.cos(np.pi * progress))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr

    def get_last_lr(self):
        return [pg['lr'] for pg in self.optimizer.param_groups]


def pretrain(model, device, save_path, epochs=10, peak_lr=0.001, mask_rate=0.4):
    print(f'\n{"="*60}')
    print('Improved Pre-training')
    print(f'  lr={peak_lr}, batch=256, nt_weight=2.0, warmup=2000')
    print(f'{"="*60}')

    dataset = PilePretrainDataset(seq_len=96, stride=48,
                                   pile_root='./dataset/time_series_pile')
    n_val = min(10000, len(dataset) // 10)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_dl = DataLoader(train_ds, batch_size=256, shuffle=True,
                          num_workers=4, drop_last=True, pin_memory=True)

    optimizer = optim.AdamW(model.parameters(), lr=peak_lr, weight_decay=0.01)
    total_steps = epochs * len(train_dl)
    scheduler = WarmupCosineScheduler(optimizer, warmup_steps=2000,
                                       total_steps=total_steps, peak_lr=peak_lr)
    recon_criterion = nn.MSELoss(reduction='none')

    print(f'Train: {n_train}, Steps/epoch: {len(train_dl)}, Total steps: {total_steps}')

    best_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        losses, nt_losses, recon_losses = [], [], []
        t0 = time.time()

        for i, batch_x in enumerate(train_dl):
            batch_x = batch_x.float().to(device)
            B, S, C = batch_x.shape
            optimizer.zero_grad()

            # 1) Masked recon
            mask = (torch.rand_like(batch_x) > mask_rate).float()
            recon_out = model.reconstruct(batch_x * mask)
            loss_mat = recon_criterion(recon_out, batch_x)
            inv_mask = 1.0 - mask
            recon_loss = (loss_mat * inv_mask).sum() / inv_mask.sum().clamp(min=1)

            # 2) Next-token prediction (random split)
            split = torch.randint(24, 72, (1,)).item()
            context = batch_x[:, :split, :]
            target = batch_x[:, split:, :]
            target_len = S - split
            context_padded = F.pad(context, (0, 0, 0, S - split))
            nt_out = model(context_padded, None, None, None, target_pred_len=target_len)
            if isinstance(nt_out, tuple): nt_out = nt_out[0]
            nt_loss = F.mse_loss(nt_out, target)

            # Weighted loss: nt에 집중
            loss = 2.0 * nt_loss + 0.5 * recon_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr = scheduler.step()

            losses.append(loss.item())
            nt_losses.append(nt_loss.item())
            recon_losses.append(recon_loss.item())

            if (i + 1) % 200 == 0:
                print(f'  iter {i+1}/{len(train_dl)}: nt={np.mean(nt_losses[-200:]):.4f} recon={np.mean(recon_losses[-200:]):.4f} lr={lr:.6f}')

        train_loss = np.mean(losses)
        mean_nt = np.mean(nt_losses)
        mean_recon = np.mean(recon_losses)
        print(f'Epoch {epoch+1}/{epochs}: loss={train_loss:.4f} (nt={mean_nt:.4f} recon={mean_recon:.4f}) lr={scheduler.get_last_lr()[0]:.6f} ({time.time()-t0:.0f}s)')

        if train_loss < best_loss:
            best_loss = train_loss
            torch.save(model.state_dict(), save_path)
            print(f'  Saved checkpoint')

    model.load_state_dict(torch.load(save_path))
    return model


if __name__ == '__main__':
    device = torch.device('cuda')

    args = get_model_args()
    model = Model(args).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'Model: {n_params/1e6:.1f}M params')
    print(f'  spectral_branch={args.spectral_branch}')

    save_path = 'checkpoints/improved_141m.pth'
    os.makedirs('checkpoints', exist_ok=True)

    model = pretrain(model, device, save_path, epochs=10, peak_lr=0.001)

    # Full eval
    fc = eval_forecasting(model, device)
    st = eval_short_term(model, device)
    imp = eval_imputation(model, device)
    cls = eval_classification(model, device)
    print_summary(fc, imp, cls, st, f'Improved 141M ({n_params/1e6:.1f}M)')
