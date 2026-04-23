"""
Linear Probe classification on UEA multivariate TS datasets.
Freezes pretrained encoder; trains only small classifier head per dataset.

Usage:
  python experiments/eval_classification_lp.py \
    --ckpt checkpoints/hyper4_10pct_full231B.pth \
    --highfreq_nf 256 --tag hyper4_10pct_cls
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

# FeDaL selected 10 UEA datasets (TimesBERT/FeDaL protocol)
FEDAL_UEA = [
    'EthanolConcentration',
    'FaceDetection',
    'Handwriting',
    'Heartbeat',
    'JapaneseVowels',
    'PEMS-SF',
    'SelfRegulationSCP1',
    'SelfRegulationSCP2',
    'SpokenArabicDigits',
    'UWaveGestureLibrary',
]


@torch.no_grad()
def extract_features(model, arr, max_seq_len=720, batch_size=32):
    """Extract encoder features by per-channel encoding + channel mean pool.
    Supports both OperatorModelVarLen (single encoder) and OperatorModelDecomp
    (4 encoder outputs via input decomposition).

    arr: np.ndarray [N, C, T]
    returns: torch.Tensor [N, d_model] (or [N, 4*d_model] for decomp)
    """
    model.eval()
    is_decomp = hasattr(model, 'decomposer')
    ps = model.encoder.patch_size
    N, C, T = arr.shape
    target_len = min(T, max_seq_len)
    target_len = (target_len // ps) * ps
    if target_len < ps:
        target_len = ps

    feats = []
    for i in range(0, N, batch_size):
        batch_np = arr[i:i+batch_size]
        bsz = batch_np.shape[0]
        # accumulator (size determined by first iter)
        z_sum = None
        for c in range(C):
            x = torch.from_numpy(batch_np[:, c, :]).float().to(DEVICE)
            if T > target_len:
                x = x[:, -target_len:]
            elif T < target_len:
                pad = target_len - T
                x = torch.cat([torch.zeros(bsz, pad, device=DEVICE), x], dim=1)
            m = x.mean(-1, keepdim=True)
            s = x.std(-1, keepdim=True).clamp(min=1e-6)
            x_n = ((x - m) / s).clamp(-10, 10)

            if is_decomp:
                # Decompose input into components, then encode
                components = model.decomposer(x_n)
                z_list = model.encoder(components)  # list of (B, d_model)
                z = torch.cat(z_list, dim=-1)       # (B, 4*d_model)
            else:
                z = model.encoder(x_n)              # (B, d_model)

            if z_sum is None:
                z_sum = z
            else:
                z_sum = z_sum + z
        z_mean = z_sum / C
        feats.append(z_mean.cpu())
    return torch.cat(feats, dim=0)


class LinearHead(nn.Module):
    def __init__(self, d, n_cls):
        super().__init__()
        self.fc = nn.Linear(d, n_cls)
    def forward(self, x):
        return self.fc(x)


def train_lp(train_z, train_y, n_cls, epochs=100, lr=1e-3, weight_decay=1e-4, batch_size=64):
    head = LinearHead(train_z.shape[1], n_cls).to(DEVICE)
    opt = optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    N = len(train_z)
    best_acc = 0.0
    train_z = train_z.to(DEVICE)
    train_y = torch.from_numpy(train_y).long().to(DEVICE)
    for ep in range(epochs):
        head.train()
        idx = torch.randperm(N, device=DEVICE)
        losses = []
        for i in range(0, N, batch_size):
            b = idx[i:i+batch_size]
            out = head(train_z[b])
            loss = F.cross_entropy(out, train_y[b])
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
    head.eval()
    with torch.no_grad():
        pred = head(train_z).argmax(-1)
        train_acc = (pred == train_y).float().mean().item()
    return head, train_acc


@torch.no_grad()
def eval_lp(head, test_z, test_y):
    head.eval()
    test_z = test_z.to(DEVICE)
    out = head(test_z)
    pred = out.argmax(-1).cpu().numpy()
    acc = (pred == test_y).mean()
    return acc


def build_model(args):
    if args.model_type == 'decomp':
        decomp_k = tuple(int(x) for x in args.decomp_kernels.split(','))
        model = OperatorModelDecomp(
            max_seq_len=720, d_model=args.d_model, n_layers=args.n_layers,
            trunk_w=args.trunk_w, fourier_nf=args.fourier_nf,
            pool_type=args.pool_type, highfreq_nf=args.highfreq_nf,
            all_fixed=bool(args.all_fixed), decomp_kernels=decomp_k,
        )
    else:
        model = OperatorModelVarLen(
            max_seq_len=720, d_model=args.d_model, n_layers=args.n_layers,
            trunk_w=args.trunk_w, hybrid_trunk=bool(args.hybrid_trunk),
            use_nll=bool(args.use_nll), fourier_nf=args.fourier_nf,
            multi_scale_fourier=bool(args.multi_scale_fourier),
            multi_scale_iq=bool(args.multi_scale_iq),
            pool_type=args.pool_type, highfreq_nf=args.highfreq_nf,
            all_fixed=bool(args.all_fixed),
        )
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--tag', required=True)
    p.add_argument('--uea_root', default='./dataset/classification/uea_mv/Multivariate_ts')
    p.add_argument('--datasets', type=str, default=','.join(FEDAL_UEA),
                   help='comma-separated UEA dataset names')
    p.add_argument('--max_seq_len', type=int, default=720)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--lr', type=float, default=1e-3)
    # model args (same as eval_m4)
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
    print(f'CLASSIFICATION LINEAR PROBE: {args.ckpt}')
    print('='*70)

    # Build + load
    model = build_model(args).to(DEVICE)
    state = torch.load(args.ckpt, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    # Freeze
    for p_ in model.parameters():
        p_.requires_grad_(False)

    datasets = args.datasets.split(',')
    results = {}
    for ds in datasets:
        ds = ds.strip()
        base = os.path.join(args.uea_root, ds)
        train_f = os.path.join(base, f'{ds}_TRAIN.ts')
        test_f = os.path.join(base, f'{ds}_TEST.ts')
        if not os.path.exists(train_f):
            print(f'  {ds}: not found at {train_f}, skip')
            continue
        try:
            t0 = time.time()
            train_arr, train_y, n_cls = _parse_ts_file(train_f)
            test_arr, test_y, _ = _parse_ts_file(test_f)
            print(f'\n--- {ds} ({n_cls}cls, {train_arr.shape[0]}tr/{test_arr.shape[0]}te, {train_arr.shape[1]}ch, T={train_arr.shape[2]}) ---')

            # Extract features
            train_z = extract_features(model, train_arr, args.max_seq_len, args.batch_size)
            test_z = extract_features(model, test_arr, args.max_seq_len, args.batch_size)

            # Train LP head
            head, tr_acc = train_lp(train_z, train_y, n_cls,
                                    epochs=args.epochs, lr=args.lr,
                                    batch_size=args.batch_size)
            # Eval
            te_acc = eval_lp(head, test_z, test_y)
            print(f'  train_acc={tr_acc*100:.2f}%  test_acc={te_acc*100:.2f}%  ({time.time()-t0:.0f}s)')
            results[ds] = {'train_acc': float(tr_acc), 'test_acc': float(te_acc), 'n_classes': n_cls}
        except Exception as e:
            print(f'  {ds}: ERROR {e}')

    # Average
    if results:
        avg = np.mean([r['test_acc'] for r in results.values()])
        print(f'\n{"="*70}')
        print(f'AVG test_acc: {avg*100:.2f}% ({len(results)} datasets)')
        results['AVG'] = float(avg)

    os.makedirs('results', exist_ok=True)
    with open(f'results/{args.tag}_cls.json','w') as f:
        json.dump(results, f, indent=2)
    print(f'Saved: results/{args.tag}_cls.json')


if __name__ == '__main__':
    main()
