"""
Fine-Tune 1-epoch classification on UEA datasets (matches FeDaL protocol).
Updates ENCODER + HEAD for 1 epoch, then evaluates.

Usage:
  python experiments/eval_classification_ft.py \
    --ckpt checkpoints/hyper4_10pct_full231B.pth \
    --highfreq_nf 256 --tag hyper4_10pct_ft
"""
import sys, os, argparse, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import optim
from torch.utils.data import DataLoader, Dataset

from experiments.exp_v1_varlen_ext import OperatorModelVarLen, OperatorModelDecomp
from data_provider.data_loader import _parse_ts_file

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

FEDAL_UEA = [
    'EthanolConcentration', 'FaceDetection', 'Handwriting', 'Heartbeat',
    'JapaneseVowels', 'PEMS-SF', 'SelfRegulationSCP1', 'SelfRegulationSCP2',
    'SpokenArabicDigits', 'UWaveGestureLibrary',
]


class EncoderCls(nn.Module):
    """Wraps pretrained encoder + linear head for per-channel mean pool."""
    def __init__(self, base_model, n_classes, is_decomp=False):
        super().__init__()
        self.base = base_model
        self.is_decomp = is_decomp
        d = base_model.encoder.d_model
        feat_dim = d * 4 if is_decomp else d
        self.head = nn.Linear(feat_dim, n_classes)

    def extract_one(self, x):
        """x: (B, T) one channel. Returns (B, feat_dim)"""
        if self.is_decomp:
            components = self.base.decomposer(x)
            z_list = self.base.encoder(components)
            return torch.cat(z_list, dim=-1)
        else:
            return self.base.encoder(x)

    def forward(self, batch_xT):
        """batch_xT: (B, C, T). Per-channel forward + mean pool + head"""
        B, C, T = batch_xT.shape
        z_sum = None
        for c in range(C):
            x = batch_xT[:, c, :]
            m = x.mean(-1, keepdim=True)
            s = x.std(-1, keepdim=True).clamp(min=1e-6)
            x_n = ((x - m) / s).clamp(-10, 10)
            z = self.extract_one(x_n)
            z_sum = z if z_sum is None else z_sum + z
        z_mean = z_sum / C
        return self.head(z_mean)


def prep_data(arr, max_seq_len=720, patch_size=16):
    """Pad/truncate to valid seq_len."""
    N, C, T = arr.shape
    target = min(T, max_seq_len)
    target = (target // patch_size) * patch_size
    if target < patch_size: target = patch_size
    if T > target:
        arr = arr[:, :, -target:]
    elif T < target:
        pad = np.zeros((N, C, target - T), dtype=arr.dtype)
        arr = np.concatenate([pad, arr], axis=-1)
    return arr


def ft_train(model, train_arr, train_y, patch_size, epochs=1, lr=1e-4, batch_size=16):
    """Differential LR: head gets `lr`, encoder gets `lr/100` (standard FT practice)."""
    model.train()
    head_params = list(model.head.parameters())
    head_ids = {id(p) for p in head_params}
    enc_params = [p for p in model.parameters() if id(p) not in head_ids]
    opt = optim.AdamW([
        {'params': enc_params, 'lr': lr / 100.0, 'weight_decay': 1e-4},   # encoder: 1e-6 if lr=1e-4
        {'params': head_params, 'lr': lr,        'weight_decay': 1e-4},   # head: 1e-4
    ])
    N = len(train_arr)
    # For tiny datasets, do LP-warmup first (head only) then full FT
    if N < 500:
        # warmup: head-only for a few mini-epochs
        for p in enc_params: p.requires_grad_(False)
        opt_head = optim.AdamW(head_params, lr=lr*10, weight_decay=1e-4)
        for _ in range(5):
            idx = np.random.permutation(N)
            for i in range(0, N, batch_size):
                bi = idx[i:i+batch_size]
                x = torch.from_numpy(train_arr[bi]).float().to(DEVICE)
                y = torch.from_numpy(train_y[bi]).long().to(DEVICE)
                out = model(x)
                loss = F.cross_entropy(out, y)
                opt_head.zero_grad(); loss.backward(); opt_head.step()
        for p in enc_params: p.requires_grad_(True)

    for ep in range(epochs):
        idx = np.random.permutation(N)
        for i in range(0, N, batch_size):
            bi = idx[i:i+batch_size]
            x = torch.from_numpy(train_arr[bi]).float().to(DEVICE)
            y = torch.from_numpy(train_y[bi]).long().to(DEVICE)
            out = model(x)
            loss = F.cross_entropy(out, y)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()


@torch.no_grad()
def ft_eval(model, test_arr, test_y, batch_size=32):
    model.eval()
    preds = []
    for i in range(0, len(test_arr), batch_size):
        x = torch.from_numpy(test_arr[i:i+batch_size]).float().to(DEVICE)
        out = model(x)
        preds.append(out.argmax(-1).cpu().numpy())
    preds = np.concatenate(preds)
    return (preds == test_y).mean()


def build_base(args):
    if args.model_type == 'decomp':
        decomp_k = tuple(int(x) for x in args.decomp_kernels.split(','))
        m = OperatorModelDecomp(
            max_seq_len=720, d_model=args.d_model, n_layers=args.n_layers,
            trunk_w=args.trunk_w, fourier_nf=args.fourier_nf,
            pool_type=args.pool_type, highfreq_nf=args.highfreq_nf,
            all_fixed=bool(args.all_fixed), decomp_kernels=decomp_k,
        )
    else:
        m = OperatorModelVarLen(
            max_seq_len=720, d_model=args.d_model, n_layers=args.n_layers,
            trunk_w=args.trunk_w, hybrid_trunk=bool(args.hybrid_trunk),
            use_nll=bool(args.use_nll), fourier_nf=args.fourier_nf,
            multi_scale_fourier=bool(args.multi_scale_fourier),
            multi_scale_iq=bool(args.multi_scale_iq),
            pool_type=args.pool_type, highfreq_nf=args.highfreq_nf,
            all_fixed=bool(args.all_fixed),
        )
    return m


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--tag', required=True)
    p.add_argument('--uea_root', default='./dataset/classification/uea_mv/Multivariate_ts')
    p.add_argument('--datasets', type=str, default=','.join(FEDAL_UEA))
    p.add_argument('--max_seq_len', type=int, default=720)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--ft_epochs', type=int, default=1)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--model_type', type=str, default='varlen', choices=['varlen','decomp'])
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
    args = p.parse_args()

    print('='*70)
    print(f'CLASSIFICATION FT (1-epoch, FeDaL protocol): {args.ckpt}')
    print('='*70)

    state = torch.load(args.ckpt, map_location=DEVICE, weights_only=True)
    datasets = args.datasets.split(',')
    results = {}
    for ds in datasets:
        ds = ds.strip()
        base = os.path.join(args.uea_root, ds)
        train_f = os.path.join(base, f'{ds}_TRAIN.ts')
        test_f = os.path.join(base, f'{ds}_TEST.ts')
        if not os.path.exists(train_f):
            print(f'  {ds}: not found, skip'); continue
        try:
            t0 = time.time()
            train_arr, train_y, n_cls = _parse_ts_file(train_f)
            test_arr, test_y, _ = _parse_ts_file(test_f)

            # Build fresh model + head per dataset
            base_m = build_base(args)
            base_m.load_state_dict(state)
            is_decomp = hasattr(base_m, 'decomposer')
            model = EncoderCls(base_m, n_cls, is_decomp=is_decomp).to(DEVICE)

            ps = base_m.encoder.patch_size
            train_arr = prep_data(train_arr, args.max_seq_len, ps)
            test_arr = prep_data(test_arr, args.max_seq_len, ps)

            print(f'\n--- {ds} ({n_cls}cls, {train_arr.shape[0]}tr/{test_arr.shape[0]}te, {train_arr.shape[1]}ch, T={train_arr.shape[2]}) ---')
            ft_train(model, train_arr, train_y, ps, epochs=args.ft_epochs,
                     lr=args.lr, batch_size=args.batch_size)
            te_acc = ft_eval(model, test_arr, test_y, batch_size=args.batch_size)
            print(f'  FT {args.ft_epochs}ep → test_acc={te_acc*100:.2f}% ({time.time()-t0:.0f}s)')
            results[ds] = {'test_acc': float(te_acc), 'n_classes': n_cls}
        except Exception as e:
            print(f'  {ds}: ERROR {e}')

    if results:
        avg = np.mean([r['test_acc'] for r in results.values() if isinstance(r, dict)])
        print(f'\n{"="*70}')
        print(f'AVG test_acc: {avg*100:.2f}% ({len(results)} datasets)')
        results['AVG'] = float(avg)

    os.makedirs('results', exist_ok=True)
    with open(f'results/{args.tag}_cls_ft.json','w') as f:
        json.dump(results, f, indent=2)
    print(f'Saved: results/{args.tag}_cls_ft.json')


if __name__ == '__main__':
    main()
