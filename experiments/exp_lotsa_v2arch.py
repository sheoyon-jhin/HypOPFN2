"""
OperatorModel V2: Learned Trunk + Cross-Attention Decoder + Rich Informed Query.

Key changes vs v1:
- Learned trunk basis (MLP) instead of fixed Fourier/Poly/RBF
- Optional Cross-Attention decoder (each query attends to encoder patches)
- Richer Informed Query: FFT + ACF + moments + spectral entropy (33 features)
- Smaller patch size (8 vs 16) — finer encoder
- Larger model (d_model=768, layers=8, trunk_w=256)

All while KEEPING Operator Learning (query at arbitrary t).

Usage:
  CUDA_VISIBLE_DEVICES=0 python experiments/exp_lotsa_v2arch.py \
      --scale 50 --seq_len 512 --epochs 40 --synth_n 500000 \
      --arch cross_attn --tag v2arch_s50_seq512_xattn
"""
import sys, os, math, time, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from torch import optim
from torch.utils.data import DataLoader, ConcatDataset

from experiments.exp_lotsa_scaling import (
    LOTSAScalingDataset, SyntheticGapFiller, collate_batch, eval_forecast
)

DEVICE = torch.device(os.environ.get('CUDA_DEV', 'cuda'))
LOTSA_DIR = os.environ.get('LOTSA_DIR', './dataset/lotsa')

PATCH_SIZE = 8  # Smaller patch for finer encoding (was 16)
TOP_K_IQ = 10   # Expanded FFT (was 5)
ACF_LAGS = [1, 2, 4, 8, 16, 32]  # Autocorrelation at key lags
GLOBAL_FEATS = 2 + 3 + len(ACF_LAGS)  # last_val + slope + 3 moments (skew,kurt,entropy) + 6 ACFs
INFORMED_DIM = 1 + 2 * TOP_K_IQ + GLOBAL_FEATS  # 1 + 20 + 11 = 32


# ============================================================
# Informed Query V2 — rich signal processing features
# ============================================================
def compute_informed_features(x):
    """
    x: (B, seq_len) — normalized (mean=0, std=1) per window
    Returns dict of tensors.
    """
    B, L = x.shape

    # FFT top-K (frequencies + phases + amplitudes)
    fft = torch.fft.rfft(x, dim=-1)
    mag = fft.abs()
    mag_nodc = mag.clone(); mag_nodc[:, 0] = 0  # exclude DC
    idx = torch.topk(mag_nodc, TOP_K_IQ, dim=-1).indices
    phase = torch.angle(torch.gather(fft, 1, idx))

    # Last value + slope (temporal momentum)
    last_val = x[:, -1:]
    slope = (x[:, -1:] - x[:, -2:-1]).detach()

    # Higher-order statistical moments
    mean_val = x.mean(dim=-1, keepdim=True)
    std_val = x.std(dim=-1, keepdim=True).clamp(min=1e-6)
    centered = x - mean_val
    skew = (centered ** 3).mean(dim=-1, keepdim=True) / (std_val ** 3 + 1e-8)
    kurt = (centered ** 4).mean(dim=-1, keepdim=True) / (std_val ** 4 + 1e-8) - 3.0

    # Autocorrelation at key lags (memory structure)
    acfs = []
    for lag in ACF_LAGS:
        if lag < L:
            a = x[:, :-lag]
            b = x[:, lag:]
            # Since x is normalized (mean=0, std=1), correlation ≈ mean product
            acf = (a * b).mean(dim=-1, keepdim=True)
        else:
            acf = torch.zeros(B, 1, device=x.device, dtype=x.dtype)
        acfs.append(acf)
    acf_feats = torch.cat(acfs, dim=-1)

    # Spectral entropy (complexity)
    power = mag ** 2
    power = power / power.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    spec_entropy = -(power * torch.log(power + 1e-10)).sum(dim=-1, keepdim=True)

    return {
        'freqs': idx.float(),
        'phases': phase,
        'last_val': last_val,
        'slope': slope,
        'skew': skew,
        'kurt': kurt,
        'spec_entropy': spec_entropy,
        'acfs': acf_feats,
    }


# ============================================================
# Learned Trunk — MLP-based basis instead of fixed Fourier/Poly/RBF
# ============================================================
class LearnedTrunk(nn.Module):
    """
    Operator trunk with LEARNED basis functions.
    Input: (t, iq) — per-query features
    Output: width-dim basis vector, combined with context-aware coefficients.
    """
    def __init__(self, width, d_model, informed_dim=INFORMED_DIM,
                 hidden=256, n_layers=3):
        super().__init__()
        self.width = width

        layers = []
        prev = informed_dim
        for _ in range(n_layers - 1):
            layers += [nn.Linear(prev, hidden), nn.GELU()]
            prev = hidden
        layers.append(nn.Linear(prev, width))
        self.basis_mlp = nn.Sequential(*layers)

        # Coefficient head from z → width
        self.coef_head = nn.Linear(d_model, width)
        nn.init.xavier_normal_(self.coef_head.weight, gain=0.1)
        nn.init.zeros_(self.coef_head.bias)

        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, iq, z):
        """iq: (B, nq, informed_dim), z: (B, d_model)"""
        basis = self.basis_mlp(iq)  # (B, nq, width)
        coef = self.coef_head(z)    # (B, width)
        out = torch.einsum('bnw,bw->bn', basis, coef)
        return out + self.bias


# ============================================================
# Cross-Attention Trunk — queries attend to encoder patches
# ============================================================
class CrossAttentionTrunk(nn.Module):
    """
    Operator trunk with CROSS-ATTENTION to encoder patches.
    Queries (derived from t + informed features) attend to patch embeddings.
    """
    def __init__(self, d_model, informed_dim=INFORMED_DIM, n_heads=8, n_layers=2, dropout=0.1):
        super().__init__()
        self.query_proj = nn.Sequential(
            nn.Linear(informed_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True, activation='gelu'
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, n_layers)
        self.out = nn.Linear(d_model, 1)

    def forward(self, iq, ctx_patches):
        """iq: (B, nq, informed_dim), ctx_patches: (B, n_patches, d_model)"""
        q = self.query_proj(iq)  # (B, nq, d)
        attn_out = self.decoder(q, ctx_patches)  # (B, nq, d)
        return self.out(attn_out).squeeze(-1)  # (B, nq)


# ============================================================
# Patch Encoder
# ============================================================
class PatchAttnEncoder(nn.Module):
    def __init__(self, seq_len, d_model=768, n_layers=8, nhead=8, patch_size=PATCH_SIZE):
        super().__init__()
        assert seq_len % patch_size == 0
        self.patch_size = patch_size
        n_patches = seq_len // patch_size
        self.patch_embed = nn.Linear(patch_size, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, n_patches, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=d_model * 4, batch_first=True,
            norm_first=True, activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        B, L = x.shape
        x = x.view(B, L // self.patch_size, self.patch_size)
        x = self.patch_embed(x) + self.pos_emb
        x = self.encoder(x)
        return self.norm(x)  # (B, n_patches, d)


# ============================================================
# Operator Model V2
# ============================================================
class OperatorModelV2(nn.Module):
    def __init__(self, seq_len=512, d_model=768, n_layers=8, nhead=8,
                 trunk_w=256, patch_size=PATCH_SIZE, arch='cross_attn'):
        """
        arch: 'cross_attn' | 'learned_trunk'
        """
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.arch = arch

        self.encoder = PatchAttnEncoder(seq_len, d_model, n_layers, nhead, patch_size)

        if arch == 'cross_attn':
            self.trunk = CrossAttentionTrunk(d_model, INFORMED_DIM, n_heads=nhead)
        elif arch == 'learned_trunk':
            self.trunk = LearnedTrunk(trunk_w, d_model, INFORMED_DIM)
        else:
            raise ValueError(f'unknown arch: {arch}')

    def _build_iq(self, ctx, qt):
        feats = compute_informed_features(ctx)
        B, nq = qt.shape
        t_exp = qt.unsqueeze(-1)  # (B, nq, 1)
        # Point-wise FFT features (depend on t)
        f = feats['freqs'].unsqueeze(1)  # (B, 1, K)
        p = feats['phases'].unsqueeze(1)
        ang = 2 * math.pi * f * t_exp + p
        fft_sin = torch.sin(ang)
        fft_cos = torch.cos(ang)
        # Global features (broadcast over queries)
        globals_feat = torch.cat([
            feats['last_val'], feats['slope'],
            feats['skew'], feats['kurt'], feats['spec_entropy'],
            feats['acfs'],
        ], dim=-1)
        globals_feat = globals_feat.unsqueeze(1).expand(-1, nq, -1)  # (B, nq, GLOBAL_FEATS+2)
        return torch.cat([t_exp, fft_sin, fft_cos, globals_feat], dim=-1)

    def forward_train(self, ctx, qt, return_per_trunk=False):
        z_patches = self.encoder(ctx)  # (B, n_patches, d)
        iq = self._build_iq(ctx, qt)   # (B, nq, INFORMED_DIM)

        if self.arch == 'cross_attn':
            out = self.trunk(iq, z_patches)
        else:
            z = z_patches.mean(dim=1)  # pool to single context vector
            out = self.trunk(iq, z)

        if return_per_trunk:
            # Single trunk, no diversity — return out × 3 as dummy
            per_trunk = out.unsqueeze(0).expand(3, -1, -1)
            return out, per_trunk
        return out

    def forecast(self, ctx, n=None):
        if n is None: n = self.seq_len
        t_end = 1.0 + n / self.seq_len
        t = torch.linspace(1.0, t_end, n, device=ctx.device).unsqueeze(0).expand(ctx.shape[0], -1)
        return self.forward_train(ctx, t)


# ============================================================
# Training (same structure, diversity off by default for v2)
# ============================================================
def train(model, datasets, save_path, seq_len, epochs=40, lr=3e-4, batch_size=64,
          n_query=64, diversity_weight=0.0):
    n_params = sum(p.numel() for p in model.parameters())
    combined = ConcatDataset(datasets)
    dl = DataLoader(combined, batch_size=batch_size, shuffle=True,
                    num_workers=4, drop_last=True, pin_memory=True)

    print(f'\n{"="*60}')
    print(f'V2 Arch Training — {model.arch}')
    print(f'  Model: {n_params/1e6:.1f}M, seq_len={seq_len}, d_model={model.d_model}')
    print(f'  Data: {len(combined):,} windows, Steps/epoch: {len(dl):,}')
    print(f'  n_query: {n_query}, diversity: {diversity_weight}')
    print(f'{"="*60}')

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        losses = []
        t0 = time.time()
        for i, batch_windows in enumerate(dl):
            batch = collate_batch(batch_windows, seq_len=seq_len, n_query=n_query)
            if batch is None: continue
            ctx, qt, qv = [x.to(DEVICE) for x in batch]
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
# Main
# ============================================================
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--scale', type=int, required=True)
    p.add_argument('--seq_len', type=int, default=512)
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--synth_ratio', type=float, default=0.3)
    p.add_argument('--synth_n', type=int, default=500000)
    p.add_argument('--d_model', type=int, default=768)
    p.add_argument('--n_layers', type=int, default=8)
    p.add_argument('--trunk_w', type=int, default=256)
    p.add_argument('--patch_size', type=int, default=PATCH_SIZE)
    p.add_argument('--arch', choices=['cross_attn', 'learned_trunk'], default='cross_attn')
    p.add_argument('--tag', type=str, required=True)
    p.add_argument('--eval_after', type=int, default=1)
    args = p.parse_args()

    np.random.seed(42); torch.manual_seed(42)

    print('=' * 60)
    print(f'V2 Arch Experiment: {args.tag}')
    print(f'  arch={args.arch}, seq={args.seq_len}, d={args.d_model}, L={args.n_layers}')
    print(f'  scale={args.scale}%, synth={args.synth_n}, epochs={args.epochs}')
    print(f'  patch_size={args.patch_size}, INFORMED_DIM={INFORMED_DIM}')
    print('=' * 60)

    window_len = args.seq_len * 3
    lotsa_ds = LOTSAScalingDataset(LOTSA_DIR, args.scale, seq_len=window_len)
    n_synth = args.synth_n if args.synth_n > 0 else max(10000, int(len(lotsa_ds) * args.synth_ratio))
    synth_ds = SyntheticGapFiller(n_samples=n_synth, seq_len=window_len)
    datasets = [lotsa_ds, synth_ds]
    total = sum(len(d) for d in datasets)
    print(f'Total: {total:,} (LOTSA: {len(lotsa_ds):,}, Synth: {len(synth_ds):,})')

    model = OperatorModelV2(
        seq_len=args.seq_len,
        d_model=args.d_model,
        n_layers=args.n_layers,
        trunk_w=args.trunk_w,
        patch_size=args.patch_size,
        arch=args.arch,
    ).to(DEVICE)
    n = sum(p.numel() for p in model.parameters())
    print(f'Model V2 ({args.arch}): {n/1e6:.1f}M params')

    save_path = f'checkpoints/{args.tag}.pth'
    os.makedirs('checkpoints', exist_ok=True)
    best = train(model, datasets, save_path, seq_len=args.seq_len, epochs=args.epochs)

    if args.eval_after:
        print('\n' + '=' * 60)
        print('EVAL')
        print('=' * 60)
        model.load_state_dict(torch.load(save_path, map_location=DEVICE, weights_only=True))
        results = eval_forecast(model, args.seq_len)

        import json
        os.makedirs('results', exist_ok=True)
        with open(f'results/{args.tag}.json', 'w') as f:
            # Convert nested dict for JSON
            out = {k: (v if isinstance(v, dict) else {'MSE': float(v)}) for k, v in results.items()}
            json.dump(out, f, indent=2)
        print(f'Saved: results/{args.tag}.json')
