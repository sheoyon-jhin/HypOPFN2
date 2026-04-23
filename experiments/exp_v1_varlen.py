"""
V1 Anchor + Variable-Length Support (Attention Masking + Random seq_len)

Changes vs V1 (exp_lotsa_scaling.py):
  - Sinusoidal positional encoding (replaces learned pos_emb)
      → Works for any seq_len up to max_seq_len
  - Padding mask in encoder (attention + mean pool)
      → Short series can use their actual length (no zero context harm)
  - Random seq_len sampled per batch during training
      → Model sees {192, 384, 512, 720} during one training
  - V1 trunks unchanged (3 HyperTrunks, Fourier/Poly/RBF)
  - V1 informed query unchanged (FFT top-5 + last_val + slope)

Eval:
  - ETT/Weather use max_seq_len (720 default)
  - M4 short series use actual length with padding mask

Usage:
  CUDA_VISIBLE_DEVICES=0 python experiments/exp_v1_varlen.py \
      --scale 50 --max_seq_len 720 --epochs 40 --synth_n 500000 \
      --tag v1_varlen_s50
"""
import sys, os, math, time, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from torch import optim
from torch.utils.data import DataLoader, ConcatDataset

from experiments.exp_lotsa_scaling import (
    LOTSAScalingDataset, SyntheticGapFiller, HyperTrunk, hyper_fwd,
    extract_freq, LatentDecomp, TOP_K_IQ, INFORMED_DIM, PATCH_SIZE,
)

DEVICE = torch.device(os.environ.get('CUDA_DEV', 'cuda'))
LOTSA_DIR = os.environ.get('LOTSA_DIR', './dataset/lotsa')


# ============================================================
# Sinusoidal positional embedding (length-independent)
# ============================================================
def sinusoidal_pe(n, d_model, device=None):
    pe = torch.zeros(n, d_model, device=device)
    pos = torch.arange(0, n, dtype=torch.float32, device=device).unsqueeze(1)
    div = torch.exp(torch.arange(0, d_model, 2, device=device).float()
                    * -(math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


# ============================================================
# Variable-Length Encoder
# ============================================================
class VarLenPatchAttnEncoder(nn.Module):
    """
    Patch transformer encoder with:
      - Sinusoidal PE (supports any seq_len ≤ max_seq_len)
      - Padding mask (ignores padded patches in attention + pool)
    """
    def __init__(self, max_seq_len, d_model=512, n_layers=6, nhead=8, patch_size=PATCH_SIZE):
        super().__init__()
        self.patch_size = patch_size
        self.max_patches = max_seq_len // patch_size
        self.d_model = d_model

        self.patch_embed = nn.Linear(patch_size, d_model)
        # Pre-compute full-length PE
        self.register_buffer('pe', sinusoidal_pe(self.max_patches, d_model))

        layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=d_model * 4, batch_first=True,
            norm_first=True, activation='gelu',
        )
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, padding_mask=None):
        """
        x: (B, L) — L may vary per batch, must be multiple of patch_size
        padding_mask: (B, L) bool, True = padded position to ignore.
                      If None, no padding (all valid).
        Returns: (B, d_model) pooled latent
        """
        B, L = x.shape
        assert L % self.patch_size == 0, f'L={L} not divisible by patch_size={self.patch_size}'
        n_patches = L // self.patch_size

        # Patch embed
        x_p = x.view(B, n_patches, self.patch_size)
        h = self.patch_embed(x_p) + self.pe[:n_patches].unsqueeze(0)

        # Patch-level padding mask (patch is padded if ALL its points are padded)
        patch_mask = None
        if padding_mask is not None:
            patch_mask = padding_mask.view(B, n_patches, self.patch_size).all(dim=-1)

        # Transformer (key_padding_mask: True = ignore)
        h = self.encoder(h, src_key_padding_mask=patch_mask)
        h = self.norm(h)

        # Mean pool over non-padded patches
        if patch_mask is not None:
            valid = (~patch_mask).float().unsqueeze(-1)  # (B, n_patches, 1)
            denom = valid.sum(dim=1).clamp(min=1.0)
            z = (h * valid).sum(dim=1) / denom
        else:
            z = h.mean(dim=1)
        return z


# ============================================================
# Operator Model V1 + Variable Length
# ============================================================
class OperatorModelVarLen(nn.Module):
    """V1 structure (3 HyperTrunks + HyperNet) with variable-length encoder."""

    def __init__(self, max_seq_len=720, d_model=512, n_layers=6, trunk_w=192,
                 use_latent_decomp=False):
        super().__init__()
        self.max_seq_len = max_seq_len  # used to normalize t (anchor scale)
        self.d_model = d_model
        self.use_latent_decomp = use_latent_decomp

        self.encoder = VarLenPatchAttnEncoder(max_seq_len, d_model, n_layers)

        if use_latent_decomp:
            self.latent_decomp = LatentDecomp(d_model)
            self.trunks = nn.ModuleList([
                HyperTrunk(trunk_w, 'poly'),
                HyperTrunk(trunk_w, 'fourier'),
                HyperTrunk(trunk_w, 'rbf'),
            ])
        else:
            self.trunks = nn.ModuleList([
                HyperTrunk(trunk_w, 'fourier'),
                HyperTrunk(trunk_w, 'poly'),
                HyperTrunk(trunk_w, 'rbf'),
            ])
        self.heads = nn.ModuleList([nn.Linear(d_model, t.odim) for t in self.trunks])
        for h in self.heads:
            nn.init.xavier_normal_(h.weight, gain=0.1)
            nn.init.constant_(h.bias, 0)
        self.biases = nn.ParameterList([nn.Parameter(torch.zeros(1)) for _ in self.trunks])

    def _build_iq(self, ctx, qt):
        """Informed query (V1 style). ctx may have padded zeros — FFT on full is fine."""
        freqs, phases, lv, ls = extract_freq(ctx)
        B, nq = qt.shape
        t_exp = qt.unsqueeze(-1)
        f, p = freqs.unsqueeze(1), phases.unsqueeze(1)
        ang = 2 * math.pi * f * t_exp + p
        return torch.cat([t_exp, torch.sin(ang), torch.cos(ang),
                          lv.unsqueeze(1).expand(-1, nq, -1),
                          ls.unsqueeze(1).expand(-1, nq, -1)], dim=-1)

    def forward_train(self, ctx, qt, padding_mask=None, return_per_trunk=False):
        z = self.encoder(ctx, padding_mask=padding_mask)
        iq = self._build_iq(ctx, qt)
        t_flat = qt.reshape(-1)

        if self.use_latent_decomp:
            z_t, z_s = self.latent_decomp(z)
            z_list = [z_t, z_s, z]
        else:
            z_list = [z, z, z]

        trunk_outs = [hyper_fwd(trunk, t_flat, head(zi), iq) + bias
                      for zi, trunk, head, bias in zip(z_list, self.trunks, self.heads, self.biases)]
        out = sum(trunk_outs)
        if return_per_trunk:
            return out, torch.stack(trunk_outs, dim=0)
        return out

    def forecast(self, ctx, n=None, padding_mask=None, seq_len_ref=None):
        """
        Forecast `n` steps ahead from ctx.
        seq_len_ref: reference seq_len for t normalization (defaults to ctx.shape[1])
        """
        if n is None: n = self.max_seq_len
        ref = seq_len_ref if seq_len_ref is not None else ctx.shape[1]
        t_end = 1.0 + n / ref
        t = torch.linspace(1.0, t_end, n, device=ctx.device).unsqueeze(0).expand(ctx.shape[0], -1)
        return self.forward_train(ctx, t, padding_mask=padding_mask)


# ============================================================
# Variable-length collate
# ============================================================
def collate_batch_varlen(windows, seq_len, n_query=64, mr=0.375, max_horizon_mult=2):
    """
    Same logic as V1 collate_batch but with fixed seq_len (caller samples per batch).
    Window samples should be >= seq_len*(1+max_horizon_mult).
    """
    ctxs, qts, qvs = [], [], []
    max_future = seq_len * max_horizon_mult
    for w in windows:
        w = w.numpy() if isinstance(w, torch.Tensor) else w
        avail_future = len(w) - seq_len
        do_forecast = (np.random.rand() < 0.5 and avail_future > 0)
        if do_forecast:
            ctx = w[:seq_len]
            future_len = min(avail_future, max_future)
            future = w[seq_len:seq_len + future_len]
            horizon_cap = np.random.randint(max(8, seq_len // 8), future_len + 1)
            qi = np.random.choice(horizon_cap, min(n_query, horizon_cap), replace=False)
            if len(qi) < n_query:
                qi = np.concatenate([qi, np.random.choice(horizon_cap, n_query - len(qi), replace=True)])
            qt = 1.0 + qi.astype(np.float32) / seq_len
            qv = future[qi]
        else:
            full = w[:seq_len] if len(w) >= seq_len else np.pad(w, (0, seq_len - len(w)))
            mask = np.random.rand(seq_len) > mr
            qi_idx = np.where(~mask)[0]
            if len(qi_idx) == 0:
                continue
            if len(qi_idx) >= n_query:
                qi_idx = np.random.choice(qi_idx, n_query, replace=False)
            else:
                qi_idx = np.concatenate([qi_idx, np.random.choice(qi_idx, n_query - len(qi_idx), replace=True)])
            ctx = full * mask.astype(np.float32)
            qt = qi_idx.astype(np.float32) / seq_len
            qv = full[qi_idx]
        ctxs.append(ctx); qts.append(qt); qvs.append(qv)
    if not ctxs: return None
    return (torch.tensor(np.stack(ctxs), dtype=torch.float32),
            torch.tensor(np.stack(qts), dtype=torch.float32),
            torch.tensor(np.stack(qvs), dtype=torch.float32))


# ============================================================
# Training with random seq_len per batch
# ============================================================
def train_varlen(model, datasets, save_path, max_seq_len, seq_len_choices,
                 epochs=40, lr=3e-4, batch_size=64, n_query=64):
    n_params = sum(p.numel() for p in model.parameters())
    combined = ConcatDataset(datasets)
    dl = DataLoader(combined, batch_size=batch_size, shuffle=True,
                    num_workers=4, drop_last=True, pin_memory=True)

    print(f'\n{"="*60}')
    print(f'VarLen V1 Training')
    print(f'  Model: {n_params/1e6:.1f}M, max_seq_len={max_seq_len}')
    print(f'  seq_len_choices: {seq_len_choices}')
    print(f'  Data: {len(combined):,} windows, Steps/epoch: {len(dl):,}')
    print(f'{"="*60}')

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        losses = []
        t0 = time.time()
        for i, batch_windows in enumerate(dl):
            # Sample seq_len for THIS batch
            seq_len_b = int(np.random.choice(seq_len_choices))
            batch = collate_batch_varlen(batch_windows, seq_len=seq_len_b, n_query=n_query)
            if batch is None: continue
            ctx, qt, qv = [x.to(DEVICE) for x in batch]
            # No padding during training (we resample to fixed length per batch)
            optimizer.zero_grad()
            pred = model.forward_train(ctx, qt)
            loss = F.mse_loss(pred, qv)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
            if (i + 1) % 500 == 0:
                print(f'  iter {i+1}/{len(dl)}: loss={np.mean(losses[-500:]):.4f}')
        scheduler.step()
        avg = float(np.mean(losses))
        elapsed = time.time() - t0
        print(f'Epoch {epoch+1}/{epochs}: loss={avg:.4f} ({elapsed:.0f}s)')
        if avg < best_loss:
            best_loss = avg
            torch.save(model.state_dict(), save_path)
            print(f'  Saved (best={avg:.4f})')
    return best_loss


# ============================================================
# Eval (variable length — short series use actual length)
# ============================================================
def eval_varlen_forecast(model, max_seq_len):
    from types import SimpleNamespace
    from data_provider.data_factory import data_provider

    datasets = {
        'ETTh1':   ('ETTh1',  './dataset/ETT-small/', 'ETTh1.csv', 7),
        'ETTh2':   ('ETTh2',  './dataset/ETT-small/', 'ETTh2.csv', 7),
        'ETTm1':   ('ETTm1',  './dataset/ETT-small/', 'ETTm1.csv', 7),
        'ETTm2':   ('ETTm2',  './dataset/ETT-small/', 'ETTm2.csv', 7),
        'Weather': ('custom', './dataset/weather/',  'weather.csv', 21),
    }
    model.eval()
    results = {}
    patch_size = model.encoder.patch_size

    for dn, (d, root, f, enc_in) in datasets.items():
        for pl in [96, 192, 336, 720]:
            try:
                a = SimpleNamespace(seq_len=max_seq_len, pred_len=pl, label_len=48, data=d,
                    root_path=root, data_path=f, features='M', target='OT', freq='h',
                    embed='timeF', enc_in=enc_in, dec_in=enc_in, c_out=enc_in,
                    num_workers=2, batch_size=32, exp_name='MTSF', ordered_data=False,
                    data_amount=-1, combine_Gaussian_datasets=False, synthetic_data_path='',
                    synthetic_root_path='./', synthetic_length=1024, stride=-1)
                _, tdl = data_provider(a, 'test')
                preds, tgts = [], []
                with torch.no_grad():
                    for bx, by, _, _ in tdl:
                        bx = bx.float().to(DEVICE)
                        B, S, C = bx.shape
                        # Use actual S (up to max_seq_len), round down to patch_size multiple
                        effective_len = min(S, max_seq_len)
                        effective_len = (effective_len // patch_size) * patch_size
                        outs = []
                        for ch in range(C):
                            x_ch = bx[:, -effective_len:, ch]
                            m = x_ch.mean(1, keepdim=True)
                            s = x_ch.std(1, keepdim=True).clamp(min=1e-6)
                            x_n = ((x_ch - m) / s).clamp(-10, 10)
                            pred_n = model.forecast(x_n, n=pl, seq_len_ref=effective_len)
                            outs.append(pred_n * s + m)
                        preds.append(torch.stack(outs, dim=-1).cpu().numpy())
                        tgts.append(by[:, -pl:, :].numpy())
                p, t = np.concatenate(preds), np.concatenate(tgts)
                mse = float(np.mean((p - t) ** 2))
                mae = float(np.mean(np.abs(p - t)))
                k = f'{dn}_{pl}'
                print(f'  {k}: MSE={mse:.4f}  MAE={mae:.4f}')
                results[k] = {'MSE': mse, 'MAE': mae}
            except Exception as e:
                print(f'  {dn}_{pl}: ERROR ({e})')

    print('\n' + '-' * 60)
    print(f'{"Dataset":<10} {"MSE":>8} {"MAE":>8}')
    print('-' * 60)
    for dn in ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'Weather']:
        entries = [v for k, v in results.items() if k.startswith(dn + '_') and 'avg' not in k]
        if entries:
            avg_mse = np.mean([e['MSE'] for e in entries])
            avg_mae = np.mean([e['MAE'] for e in entries])
            print(f'{dn:<10} {avg_mse:>8.4f} {avg_mae:>8.4f}')
            results[f'{dn}_avg'] = {'MSE': float(avg_mse), 'MAE': float(avg_mae)}
    # Overall
    ds_entries = [results[f'{dn}_avg'] for dn in ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'Weather']
                  if f'{dn}_avg' in results]
    if ds_entries:
        ov_mse = np.mean([e['MSE'] for e in ds_entries])
        ov_mae = np.mean([e['MAE'] for e in ds_entries])
        print('-' * 60)
        print(f'{"OVERALL":<10} {ov_mse:>8.4f} {ov_mae:>8.4f}')
        results['overall_avg'] = {'MSE': float(ov_mse), 'MAE': float(ov_mae)}
    return results


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--scale', type=int, required=True)
    p.add_argument('--max_seq_len', type=int, default=720)
    p.add_argument('--seq_choices', type=str, default='192,384,512,720',
                   help='Comma-sep seq_len choices sampled per batch')
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--synth_ratio', type=float, default=0.3)
    p.add_argument('--synth_n', type=int, default=500000)
    p.add_argument('--tag', type=str, required=True)
    p.add_argument('--eval_after', type=int, default=1)
    args = p.parse_args()

    np.random.seed(42); torch.manual_seed(42)
    seq_choices = [int(x) for x in args.seq_choices.split(',')]

    print('=' * 60)
    print(f'V1 VarLen: {args.tag}')
    print(f'  max_seq_len={args.max_seq_len}, choices={seq_choices}')
    print(f'  scale={args.scale}%, synth_n={args.synth_n}, epochs={args.epochs}')
    print('=' * 60)

    # Window size must accommodate max_seq_len with 2x future
    window_len = args.max_seq_len * 3
    lotsa_ds = LOTSAScalingDataset(LOTSA_DIR, args.scale, seq_len=window_len)
    n_synth = args.synth_n if args.synth_n > 0 else max(10000, int(len(lotsa_ds) * args.synth_ratio))
    synth_ds = SyntheticGapFiller(n_samples=n_synth, seq_len=window_len)
    datasets = [lotsa_ds, synth_ds]
    total = sum(len(d) for d in datasets)
    print(f'Total: {total:,} (LOTSA: {len(lotsa_ds):,}, Synth: {len(synth_ds):,})')

    model = OperatorModelVarLen(max_seq_len=args.max_seq_len).to(DEVICE)
    n = sum(p.numel() for p in model.parameters())
    print(f'Model V1 VarLen: {n/1e6:.1f}M params')

    save_path = f'checkpoints/{args.tag}.pth'
    os.makedirs('checkpoints', exist_ok=True)
    best = train_varlen(model, datasets, save_path,
                        max_seq_len=args.max_seq_len,
                        seq_len_choices=seq_choices,
                        epochs=args.epochs)

    if args.eval_after:
        print('\n' + '=' * 60)
        print('EVAL (ETT/Weather @ max_seq_len)')
        print('=' * 60)
        model.load_state_dict(torch.load(save_path, map_location=DEVICE, weights_only=True))
        results = eval_varlen_forecast(model, args.max_seq_len)

        import json
        os.makedirs('results', exist_ok=True)
        with open(f'results/{args.tag}.json', 'w') as f:
            json.dump(results, f, indent=2)
        print(f'Saved: results/{args.tag}.json')
