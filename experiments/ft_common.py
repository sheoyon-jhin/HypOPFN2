"""
Shared utilities for Fine-Tuning evaluation across tasks.

Principles:
  - Reversible instance normalization (RevIN style, optional mask-aware)
  - Warmup + cosine LR schedule
  - Differential LR (encoder vs head)
  - Best-checkpoint selection via validation metric
  - Batched operations (no per-sample Python loops in hot path)
"""
import os, math, copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============================================================
# Normalization
# ============================================================
def instance_norm(x, mask=None, eps=1e-5):
    """RevIN-style per-sample normalization on last dim.
    x: (..., T). mask: (..., T) 1=observed. If given, mean/std computed on observed only.
    Returns: x_n, mean, std (broadcasted to x shape on last dim).
    """
    if mask is None:
        m = x.mean(-1, keepdim=True)
        s = x.std(-1, keepdim=True).clamp_min(eps)
    else:
        denom = mask.sum(-1, keepdim=True).clamp_min(1.0)
        m = (x * mask).sum(-1, keepdim=True) / denom
        v = ((x - m) ** 2 * mask).sum(-1, keepdim=True) / denom
        s = v.clamp_min(eps).sqrt()
    return ((x - m) / s).clamp(-10, 10), m, s


# ============================================================
# LR schedule
# ============================================================
class WarmupCosineLR:
    """Linear warmup for `warmup_steps`, then cosine decay to `min_lr_frac` * base_lr."""
    def __init__(self, optimizer, base_lrs, total_steps, warmup_frac=0.1, min_lr_frac=0.05):
        self.opt = optimizer
        self.base_lrs = list(base_lrs)
        self.total = max(1, total_steps)
        self.warmup = max(1, int(self.total * warmup_frac))
        self.min_frac = min_lr_frac
        self.step_idx = 0

    def step(self):
        self.step_idx += 1
        if self.step_idx <= self.warmup:
            scale = self.step_idx / self.warmup
        else:
            progress = (self.step_idx - self.warmup) / max(1, self.total - self.warmup)
            progress = min(1.0, progress)
            cos = 0.5 * (1.0 + math.cos(math.pi * progress))
            scale = self.min_frac + (1.0 - self.min_frac) * cos
        for i, pg in enumerate(self.opt.param_groups):
            pg['lr'] = self.base_lrs[i] * scale


def build_optimizer(model, head_module=None, enc_lr=1e-5, head_lr=1e-4, wd=1e-4):
    """Build AdamW with differential LR (encoder vs head/non-encoder)."""
    head_ids = set()
    if head_module is not None:
        head_ids = {id(p) for p in head_module.parameters()}
    enc_params, head_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if id(p) in head_ids:
            head_params.append(p)
        elif n.startswith('encoder'):
            enc_params.append(p)
        else:
            head_params.append(p)  # trunks, etc. treated as head-tier
    groups = []
    lrs = []
    if enc_params:
        groups.append({'params': enc_params, 'lr': enc_lr, 'weight_decay': wd})
        lrs.append(enc_lr)
    if head_params:
        groups.append({'params': head_params, 'lr': head_lr, 'weight_decay': wd})
        lrs.append(head_lr)
    return optim.AdamW(groups), lrs


# ============================================================
# Best-checkpoint early-stopping helper
# ============================================================
class BestKeeper:
    """Keep state_dict of best model by val metric. mode='min' or 'max'."""
    def __init__(self, mode='min', patience=None):
        assert mode in ('min', 'max')
        self.mode = mode
        self.best = float('inf') if mode == 'min' else float('-inf')
        self.best_state = None
        self.best_epoch = -1
        self.patience = patience
        self.stale = 0

    def update(self, model, metric, epoch):
        better = (metric < self.best) if self.mode == 'min' else (metric > self.best)
        if better:
            self.best = metric
            self.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            self.best_epoch = epoch
            self.stale = 0
            return True
        self.stale += 1
        return False

    def should_stop(self):
        return self.patience is not None and self.stale >= self.patience

    def restore(self, model):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


# ============================================================
# Model builder (shared)
# ============================================================
def build_base_model(args, max_seq_len=720):
    from experiments.exp_v1_varlen_ext import OperatorModelVarLen, OperatorModelDecomp
    if args.model_type == 'decomp':
        decomp_k = tuple(int(x) for x in args.decomp_kernels.split(','))
        m = OperatorModelDecomp(
            max_seq_len=max_seq_len, d_model=args.d_model, n_layers=args.n_layers,
            trunk_w=args.trunk_w, fourier_nf=args.fourier_nf,
            pool_type=args.pool_type, highfreq_nf=args.highfreq_nf,
            all_fixed=bool(args.all_fixed), decomp_kernels=decomp_k,
        )
    else:
        m = OperatorModelVarLen(
            max_seq_len=max_seq_len, d_model=args.d_model, n_layers=args.n_layers,
            trunk_w=args.trunk_w, hybrid_trunk=bool(args.hybrid_trunk),
            use_nll=bool(args.use_nll), fourier_nf=args.fourier_nf,
            multi_scale_fourier=bool(args.multi_scale_fourier),
            multi_scale_iq=bool(args.multi_scale_iq),
            pool_type=args.pool_type, highfreq_nf=args.highfreq_nf,
            all_fixed=bool(args.all_fixed),
        )
    return m


def add_std_cli(p):
    p.add_argument('--model_type', type=str, default='varlen', choices=['varlen', 'decomp'])
    p.add_argument('--decomp_kernels', type=str, default='49,25,7')
    p.add_argument('--use_nll', type=int, default=0)
    p.add_argument('--hybrid_trunk', type=int, default=0)
    p.add_argument('--all_fixed', type=int, default=0)
    p.add_argument('--highfreq_nf', type=int, default=0)
    p.add_argument('--fourier_nf', type=int, default=32)
    p.add_argument('--multi_scale_fourier', type=int, default=0)
    p.add_argument('--multi_scale_iq', type=int, default=0)
    p.add_argument('--pool_type', type=str, default='mean')
    p.add_argument('--d_model', type=int, default=512)
    p.add_argument('--n_layers', type=int, default=6)
    p.add_argument('--trunk_w', type=int, default=192)
