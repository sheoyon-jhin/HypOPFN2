"""
Fine-Tune classification (UEA, FeDaL 10-dataset protocol).

Design:
  - Per-channel encoder features → mean pool across channels → linear (or MLP) head
  - Instance-normalize each channel per sample
  - Train/val split from TRAIN (stratified-ish 80/20), early stopping on val acc
  - Differential LR: encoder < head
  - Warmup + cosine LR
  - Test accuracy reported at best-val epoch

Usage:
  python experiments/ft_classification.py \
    --ckpt checkpoints/hyper4_10pct_full231B.pth --highfreq_nf 256 \
    --ft_epochs 5 --lr 1e-3 --tag hyper4_10pct_cls_ep5_lr1e-3
"""
import sys, os, argparse, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from experiments.ft_common import (
    DEVICE, instance_norm, WarmupCosineLR, build_optimizer,
    BestKeeper, build_base_model, add_std_cli,
)
from data_provider.data_loader import _parse_ts_file

FEDAL_UEA = [
    'EthanolConcentration', 'FaceDetection', 'Handwriting', 'Heartbeat',
    'JapaneseVowels', 'PEMS-SF', 'SelfRegulationSCP1', 'SelfRegulationSCP2',
    'SpokenArabicDigits', 'UWaveGestureLibrary',
]


# ------------------------------------------------------------
# Classification model wrapping pretrained encoder
# ------------------------------------------------------------
class ClsHead(nn.Module):
    """Head on top of pooled feature. head_type: 'linear' or 'mlp'."""
    def __init__(self, feat_dim, n_classes, head_type='linear', dropout=0.1):
        super().__init__()
        if head_type == 'mlp':
            self.net = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(feat_dim, feat_dim), nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(feat_dim, n_classes),
            )
        else:
            self.net = nn.Sequential(nn.Dropout(dropout), nn.Linear(feat_dim, n_classes))

    def forward(self, z):
        return self.net(z)


class EncoderCls(nn.Module):
    def __init__(self, base, n_classes, is_decomp, head_type='linear', dropout=0.1):
        super().__init__()
        self.base = base
        self.is_decomp = is_decomp
        d = base.encoder.d_model
        feat_dim = d * 4 if is_decomp else d
        self.head = ClsHead(feat_dim, n_classes, head_type=head_type, dropout=dropout)

    def _extract(self, x_ch):
        """x_ch: (B, T). Returns (B, feat_dim)."""
        if self.is_decomp:
            comps = self.base.decomposer(x_ch)
            z_list = self.base.encoder(comps)
            return torch.cat(z_list, dim=-1)
        return self.base.encoder(x_ch)

    def forward(self, x):
        """x: (B, C, T). Per-channel encode → mean pool → head."""
        B, C, T = x.shape
        # Flatten channels into batch dim for one big forward
        x_flat = x.reshape(B * C, T)
        x_n, _, _ = instance_norm(x_flat)
        z_flat = self._extract(x_n)                # (B*C, feat_dim)
        z = z_flat.view(B, C, -1).mean(dim=1)       # mean-pool over channels
        return self.head(z)


# ------------------------------------------------------------
# Data prep
# ------------------------------------------------------------
def prep_data(arr, patch_size, max_seq_len):
    """Truncate/pad to a valid seq_len (multiple of patch_size, <= max_seq_len)."""
    N, C, T = arr.shape
    target = min(T, max_seq_len)
    target = max(patch_size, (target // patch_size) * patch_size)
    if T > target:
        arr = arr[:, :, -target:]
    elif T < target:
        pad = np.zeros((N, C, target - T), dtype=arr.dtype)
        arr = np.concatenate([pad, arr], axis=-1)
    return arr.astype(np.float32)


def stratified_split(y, val_frac=0.2, seed=0):
    """Per-class split for balanced val. Returns (train_idx, val_idx)."""
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    train_idx, val_idx = [], []
    for c in classes:
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        k = max(1, int(len(idx) * val_frac)) if len(idx) >= 5 else 0
        val_idx.extend(idx[:k].tolist())
        train_idx.extend(idx[k:].tolist())
    return np.array(train_idx), np.array(val_idx)


# ------------------------------------------------------------
# Train / eval loops (batched)
# ------------------------------------------------------------
def run_epoch(model, X, y, batch_size, train=True, optimizer=None, scheduler=None, amp=True):
    model.train(train)
    N = len(X)
    idx = np.random.permutation(N) if train else np.arange(N)
    losses, correct, count = 0.0, 0, 0
    amp_ctx = (lambda: torch.autocast(device_type='cuda', dtype=torch.bfloat16)) if amp else (lambda: torch.enable_grad() if train else torch.no_grad())
    ctx_mgr = torch.enable_grad() if train else torch.no_grad()
    with ctx_mgr:
        for i in range(0, N, batch_size):
            bi = idx[i:i + batch_size]
            bx = torch.from_numpy(X[bi]).to(DEVICE, non_blocking=True)
            by = torch.from_numpy(y[bi]).long().to(DEVICE, non_blocking=True)
            with amp_ctx():
                logits = model(bx)
                loss = F.cross_entropy(logits, by)
            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
            losses += loss.item() * len(bi)
            correct += (logits.argmax(-1) == by).sum().item()
            count += len(bi)
    return losses / count, correct / count


# ------------------------------------------------------------
# Per-dataset FT
# ------------------------------------------------------------
def ft_one_dataset(ds, args, state):
    base_dir = os.path.join(args.uea_root, ds)
    train_f = os.path.join(base_dir, f'{ds}_TRAIN.ts')
    test_f = os.path.join(base_dir, f'{ds}_TEST.ts')
    if not os.path.exists(train_f):
        return {'error': 'not_found'}

    train_arr, train_y, n_cls = _parse_ts_file(train_f)
    test_arr, test_y, _ = _parse_ts_file(test_f)

    base = build_base_model(args, max_seq_len=args.max_seq_len).to(DEVICE)
    base.load_state_dict(state)
    is_decomp = hasattr(base, 'decomposer')
    model = EncoderCls(base, n_cls, is_decomp,
                       head_type=args.head_type, dropout=args.dropout).to(DEVICE)
    ps = base.encoder.patch_size

    train_arr = prep_data(train_arr, ps, args.max_seq_len)
    test_arr = prep_data(test_arr, ps, args.max_seq_len)

    # Split train → train/val
    tr_idx, val_idx = stratified_split(train_y, val_frac=args.val_frac, seed=0)
    X_tr, y_tr = train_arr[tr_idx], train_y[tr_idx]
    X_val, y_val = train_arr[val_idx], train_y[val_idx]

    # LP warmup (head only) for small datasets
    if len(X_tr) < 500 and args.lp_warmup_epochs > 0:
        for p in base.parameters(): p.requires_grad_(False)
        opt_h, lrs_h = build_optimizer(model, head_module=model.head,
                                        enc_lr=0.0, head_lr=args.lr * 5, wd=args.wd)
        steps_h = max(1, (len(X_tr) // args.batch_size) * args.lp_warmup_epochs)
        sch_h = WarmupCosineLR(opt_h, lrs_h, steps_h, warmup_frac=0.1, min_lr_frac=0.1)
        for _ in range(args.lp_warmup_epochs):
            run_epoch(model, X_tr, y_tr, args.batch_size, train=True,
                      optimizer=opt_h, scheduler=sch_h, amp=args.amp)
        for p in base.parameters(): p.requires_grad_(True)

    # Full FT with differential LR
    opt, lrs = build_optimizer(model, head_module=model.head,
                               enc_lr=args.lr * args.enc_lr_ratio,
                               head_lr=args.lr, wd=args.wd)
    steps = max(1, (len(X_tr) // args.batch_size) * args.ft_epochs)
    sch = WarmupCosineLR(opt, lrs, steps, warmup_frac=0.1, min_lr_frac=0.05)

    keeper = BestKeeper(mode='max', patience=args.patience)
    val_accs = []
    for ep in range(args.ft_epochs):
        tr_loss, tr_acc = run_epoch(model, X_tr, y_tr, args.batch_size,
                                    train=True, optimizer=opt, scheduler=sch, amp=args.amp)
        if len(X_val) > 0:
            _, val_acc = run_epoch(model, X_val, y_val, args.batch_size, train=False, amp=args.amp)
        else:
            val_acc = tr_acc
        val_accs.append(val_acc)
        keeper.update(model, val_acc, ep)
        if keeper.should_stop():
            break

    keeper.restore(model)
    _, test_acc = run_epoch(model, test_arr, test_y, args.batch_size, train=False, amp=args.amp)
    return {
        'test_acc': float(test_acc),
        'val_acc': float(keeper.best),
        'best_epoch': int(keeper.best_epoch),
        'n_classes': int(n_cls),
        'n_train': int(len(X_tr)), 'n_val': int(len(X_val)), 'n_test': int(len(test_arr)),
        'T': int(train_arr.shape[-1]), 'C': int(train_arr.shape[1]),
    }


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--tag', required=True)
    p.add_argument('--uea_root', default='./dataset/classification/uea_mv/Multivariate_ts')
    p.add_argument('--datasets', type=str, default=','.join(FEDAL_UEA))
    p.add_argument('--max_seq_len', type=int, default=720)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--ft_epochs', type=int, default=5)
    p.add_argument('--lp_warmup_epochs', type=int, default=3)
    p.add_argument('--lr', type=float, default=1e-3,
                   help='head LR; encoder LR = lr * enc_lr_ratio')
    p.add_argument('--enc_lr_ratio', type=float, default=0.05)
    p.add_argument('--wd', type=float, default=1e-4)
    p.add_argument('--val_frac', type=float, default=0.2)
    p.add_argument('--patience', type=int, default=None)
    p.add_argument('--head_type', type=str, default='linear', choices=['linear', 'mlp'])
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--amp', type=int, default=1)
    add_std_cli(p)
    args = p.parse_args()

    print('=' * 70)
    print(f'CLS FT: {args.ckpt}')
    print(f'  ft_epochs={args.ft_epochs} lr={args.lr} (enc={args.lr*args.enc_lr_ratio:.1e}) '
          f'head={args.head_type} dropout={args.dropout}')
    print('=' * 70)

    state = torch.load(args.ckpt, map_location=DEVICE, weights_only=True)
    results = {}
    datasets = [d.strip() for d in args.datasets.split(',') if d.strip()]
    for ds in datasets:
        t0 = time.time()
        try:
            r = ft_one_dataset(ds, args, state)
            if 'error' in r:
                print(f'  {ds}: {r["error"]}')
                continue
            elapsed = time.time() - t0
            print(f'  {ds:<28} | test={r["test_acc"]*100:5.2f} val={r["val_acc"]*100:5.2f} '
                  f'ep={r["best_epoch"]+1}/{args.ft_epochs} '
                  f'({r["n_train"]}tr/{r["n_val"]}val/{r["n_test"]}te {r["C"]}ch T={r["T"]}) '
                  f'({elapsed:.0f}s)')
            results[ds] = r
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f'  {ds}: ERROR {e}')
            results[ds] = {'error': str(e)}

    # Aggregate
    accs = [r['test_acc'] for r in results.values() if isinstance(r, dict) and 'test_acc' in r]
    if accs:
        avg = float(np.mean(accs))
        print(f'\n{"="*70}\nAVG test_acc: {avg*100:.2f}%  ({len(accs)} datasets)')
        results['AVG'] = avg

    os.makedirs('results', exist_ok=True)
    out_path = f'results/{args.tag}_cls_ft.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Saved: {out_path}')


if __name__ == '__main__':
    main()
