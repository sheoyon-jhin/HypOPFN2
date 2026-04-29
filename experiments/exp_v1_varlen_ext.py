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
import sys, os, math, time, argparse, contextlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from torch import optim
from torch.utils.data import DataLoader, ConcatDataset

from experiments.exp_lotsa_scaling import (
    LOTSAScalingDataset, LOTSASubsetDataset, TARGET_SIMILAR_DATASETS,
    SyntheticGapFiller, CauKerSynthDataset, HyperTrunk, FixedTrunk, hyper_fwd,
    extract_freq, extract_freq_multiscale, LatentDecomp, TOP_K_IQ, INFORMED_DIM, PATCH_SIZE,
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
      - Pool mode: 'mean' (default) or 'attn' (learnable CLS token cross-attention)
    """
    def __init__(self, max_seq_len, d_model=512, n_layers=6, nhead=8, patch_size=PATCH_SIZE,
                 pool_type='mean'):
        super().__init__()
        self.patch_size = patch_size
        self.max_patches = max_seq_len // patch_size
        self.d_model = d_model
        self.pool_type = pool_type

        self.patch_embed = nn.Linear(patch_size, d_model)
        # Pre-compute full-length PE
        self.register_buffer('pe', sinusoidal_pe(self.max_patches, d_model))

        layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=d_model * 4, batch_first=True,
            norm_first=True, activation='gelu',
        )
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.norm = nn.LayerNorm(d_model)

        if pool_type == 'attn':
            # Learnable CLS token queries patches via cross-attention
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
            self.pool_attn = nn.MultiheadAttention(
                d_model, num_heads=nhead, batch_first=True, dropout=0.0,
            )
            self.pool_norm = nn.LayerNorm(d_model)

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

        if self.pool_type == 'attn':
            # Cross-attention: CLS token queries patches
            cls = self.cls_token.expand(B, -1, -1)   # (B, 1, d)
            # MultiheadAttention.key_padding_mask: True = ignore
            z, _ = self.pool_attn(cls, h, h, key_padding_mask=patch_mask)
            z = self.pool_norm(z.squeeze(1))
        else:
            # Mean pool over non-padded patches
            if patch_mask is not None:
                valid = (~patch_mask).float().unsqueeze(-1)
                denom = valid.sum(dim=1).clamp(min=1.0)
                z = (h * valid).sum(dim=1) / denom
            else:
                z = h.mean(dim=1)
        return z


# ============================================================
# Input-level Signal Decomposition
# ============================================================
class SeriesDecomposer(nn.Module):
    """
    Decompose time series into components aligned with trunk basis functions.

    3-component mode (kernel_sizes length 2):
      - Trend (→ Poly trunk): MA(k0), smooth
      - Seasonal (→ Fourier trunk): MA(k1) on detrended, periodic
      - Residual (→ RBF trunk): remainder, local spikes

    4-component mode (kernel_sizes length 3):
      - Trend (→ Poly trunk): MA(k0), very smooth
      - Seasonal (→ Fourier nf=32): MA(k1) on detrended, medium periodic
      - High-freq (→ Fourier nf=256): MA(k2) on de-seasoned, fast oscillation
      - Residual (→ RBF trunk): remainder, spikes/noise

    All components sum to original x (perfect reconstruction).
    Differentiable, works with variable-length + padding.
    """
    def __init__(self, kernel_sizes=(49, 25)):
        super().__init__()
        self.kernel_sizes = kernel_sizes
        self.n_components = len(kernel_sizes) + 1

    def _moving_avg(self, x, k):
        """Reflect-padded moving average. x: (B, L)"""
        pad = k // 2
        x_pad = F.pad(x.unsqueeze(1), (pad, pad), mode='reflect')
        avg = F.avg_pool1d(x_pad, k, stride=1)
        return avg.squeeze(1)

    def forward(self, x):
        """
        x: (B, L)
        Returns: list of (B, L) tensors — one per component.
        """
        components = []
        remainder = x
        for k in self.kernel_sizes:
            smooth = self._moving_avg(remainder, k)
            components.append(smooth)
            remainder = remainder - smooth
        components.append(remainder)  # final residual
        return components


class DecompEncoder(nn.Module):
    """
    Shared Transformer backbone + per-component projection heads.
    Each component gets its own z vector for its matched trunk.
    """
    def __init__(self, n_components, max_seq_len, d_model=512, n_layers=6,
                 nhead=8, pool_type='mean'):
        super().__init__()
        self.n_components = n_components
        self.backbone = VarLenPatchAttnEncoder(
            max_seq_len, d_model, n_layers, nhead=nhead, pool_type=pool_type)
        self.proj_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            ) for _ in range(n_components)
        ])

    @property
    def patch_size(self):
        return self.backbone.patch_size

    @property
    def d_model(self):
        return self.backbone.d_model

    def forward(self, components, padding_mask=None):
        """
        components: list of (B, L) tensors
        Returns: list of (B, d_model) z vectors, one per component
        """
        zs = []
        for comp, proj in zip(components, self.proj_heads):
            z_shared = self.backbone(comp, padding_mask=padding_mask)
            zs.append(proj(z_shared))
        return zs


# ============================================================
# Operator Model with Input Decomposition
# ============================================================
class OperatorModelDecomp(nn.Module):
    """
    Input-level decomposition → separate z per trunk.

    4-component mode (decomp_kernels=(49, 25, 7)):
      x → Decompose → [trend, seasonal, highfreq, residual]
                          ↓          ↓          ↓          ↓
                       Encoder    Encoder     Encoder    Encoder
                       +proj₀    +proj₁      +proj₂     +proj₃
                          ↓          ↓          ↓          ↓
                       z_trend  z_seasonal  z_highfreq z_residual
                          ↓          ↓          ↓          ↓
                      Poly(6)  Fourier(32) Fourier(256)  RBF(20)

    Each trunk receives z matched to its signal component.
    HyperTrunk is safe here because each z is specialized (no "give up" problem).
    """
    def __init__(self, max_seq_len=720, d_model=512, n_layers=6, trunk_w=192,
                 nhead=8, fourier_nf=32, pool_type='mean', highfreq_nf=0,
                 all_fixed=False, decomp_kernels=(49, 25, 7),
                 learnable_alpha=False):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.d_model = d_model

        self.decomposer = SeriesDecomposer(kernel_sizes=decomp_kernels)
        n_components = self.decomposer.n_components  # len(kernels) + 1

        self.encoder = DecompEncoder(
            n_components=n_components, max_seq_len=max_seq_len,
            d_model=d_model, n_layers=n_layers, nhead=nhead, pool_type=pool_type)

        informed_dim = INFORMED_DIM
        idim_kw = {'informed_dim': informed_dim, 'learnable_alpha': learnable_alpha}

        # Build trunks matched to decomposition components
        # Order: [trend→Poly, seasonal→Fourier(nf), highfreq→Fourier(highfreq_nf), residual→RBF]
        # For 3-component: [trend→Poly, seasonal→Fourier, residual→RBF]
        TrunkCls = FixedTrunk if all_fixed else HyperTrunk

        if n_components == 4 and highfreq_nf > 0:
            # 4-way decomposition: trend, seasonal, highfreq, residual
            if all_fixed:
                self.trunks = nn.ModuleList([
                    FixedTrunk(trunk_w, 'poly', d_model, **idim_kw),
                    FixedTrunk(trunk_w, 'fourier', d_model, nf=fourier_nf, **idim_kw),
                    FixedTrunk(trunk_w, 'fourier', d_model, nf=highfreq_nf, **idim_kw),
                    FixedTrunk(trunk_w, 'rbf', d_model, **idim_kw),
                ])
            else:
                self.trunks = nn.ModuleList([
                    HyperTrunk(trunk_w, 'poly', **idim_kw),
                    HyperTrunk(trunk_w, 'fourier', nf=fourier_nf, **idim_kw),
                    HyperTrunk(trunk_w, 'fourier', nf=highfreq_nf, **idim_kw),
                    HyperTrunk(trunk_w, 'rbf', **idim_kw),
                ])
        else:
            # 3-way decomposition: trend, seasonal, residual
            if all_fixed:
                self.trunks = nn.ModuleList([
                    FixedTrunk(trunk_w, 'poly', d_model, **idim_kw),
                    FixedTrunk(trunk_w, 'fourier', d_model, nf=fourier_nf, **idim_kw),
                    FixedTrunk(trunk_w, 'rbf', d_model, **idim_kw),
                ])
            else:
                self.trunks = nn.ModuleList([
                    HyperTrunk(trunk_w, 'poly', **idim_kw),
                    HyperTrunk(trunk_w, 'fourier', nf=fourier_nf, **idim_kw),
                    HyperTrunk(trunk_w, 'rbf', **idim_kw),
                ])
            if highfreq_nf > 0:
                # Add extra trunk but reuse last z (residual)
                if all_fixed:
                    self.trunks.append(FixedTrunk(trunk_w, 'fourier', d_model, nf=highfreq_nf, **idim_kw))
                else:
                    self.trunks.append(HyperTrunk(trunk_w, 'fourier', nf=highfreq_nf, **idim_kw))

        self.heads = nn.ModuleList([
            nn.Linear(d_model, t.odim) if not getattr(t, 'is_fixed', False)
            else nn.Identity()
            for t in self.trunks
        ])
        for h in self.heads:
            if isinstance(h, nn.Linear):
                nn.init.xavier_normal_(h.weight, gain=0.1)
                nn.init.constant_(h.bias, 0)
        self.biases = nn.ParameterList([nn.Parameter(torch.zeros(1)) for _ in self.trunks])

    def _build_iq(self, ctx, qt):
        freqs, phases, lv, ls = extract_freq(ctx)
        B, nq = qt.shape
        t_exp = qt.unsqueeze(-1)
        f, p = freqs.unsqueeze(1), phases.unsqueeze(1)
        ang = 2 * math.pi * f * t_exp + p
        return torch.cat([t_exp, torch.sin(ang), torch.cos(ang),
                          lv.unsqueeze(1).expand(-1, nq, -1),
                          ls.unsqueeze(1).expand(-1, nq, -1)], dim=-1)

    def forward_train(self, ctx, qt, padding_mask=None, return_per_trunk=False):
        # Decompose input
        components = self.decomposer(ctx)
        # Encode each component → separate z
        z_list = self.encoder(components, padding_mask=padding_mask)
        # If 4th trunk (highfreq), use full-signal z (residual's z)
        if len(self.trunks) > len(z_list):
            z_list = z_list + [z_list[-1]] * (len(self.trunks) - len(z_list))

        iq = self._build_iq(ctx, qt)  # IQ from original (full) signal
        t_flat = qt.reshape(-1)

        trunk_outs = []
        for zi, trunk, head, bias in zip(z_list, self.trunks, self.heads, self.biases):
            if getattr(trunk, 'is_fixed', False):
                trunk_outs.append(trunk(t_flat, iq, zi) + bias)
            else:
                trunk_outs.append(hyper_fwd(trunk, t_flat, head(zi), iq) + bias)
        out = sum(trunk_outs)
        if return_per_trunk:
            return out, torch.stack(trunk_outs, dim=0)
        return out

    def forecast(self, ctx, n=None, padding_mask=None, seq_len_ref=None):
        if n is None: n = self.max_seq_len
        ref = seq_len_ref if seq_len_ref is not None else ctx.shape[1]
        t_end = 1.0 + n / ref
        t = torch.linspace(1.0, t_end, n, device=ctx.device).unsqueeze(0).expand(ctx.shape[0], -1)
        return self.forward_train(ctx, t, padding_mask=padding_mask)


# ============================================================
# Operator Model V1 + Variable Length (original)
# ============================================================
class OperatorModelVarLen(nn.Module):
    """V1 structure (3 HyperTrunks + HyperNet) with variable-length encoder."""

    def __init__(self, max_seq_len=720, d_model=512, n_layers=6, trunk_w=192,
                 use_latent_decomp=False, hybrid_trunk=False, use_nll=False, nhead=8,
                 fourier_nf=32, multi_scale_fourier=False,
                 multi_scale_iq=False, ms_iq_k=3, ms_iq_scales=(1.0, 0.5, 0.25),
                 pool_type='mean', highfreq_nf=0, highfreq2_nf=0, all_fixed=False,
                 use_cheby=False, learnable_alpha=False):
        super().__init__()
        self.max_seq_len = max_seq_len  # used to normalize t (anchor scale)
        self.d_model = d_model
        self.use_latent_decomp = use_latent_decomp
        self.hybrid_trunk = hybrid_trunk
        all_fixed = all_fixed
        self.use_nll = use_nll
        self.multi_scale_iq = multi_scale_iq
        self.ms_iq_k = ms_iq_k
        self.ms_iq_scales = ms_iq_scales
        # Compute informed_dim: 1 (t) + 2 * K_total (sin/cos) + 2 (last_val, slope)
        k_total = ms_iq_k * len(ms_iq_scales) if multi_scale_iq else TOP_K_IQ
        self.informed_dim = 1 + 2 * k_total + 2

        self.encoder = VarLenPatchAttnEncoder(max_seq_len, d_model, n_layers, nhead=nhead,
                                               pool_type=pool_type)

        if use_nll:
            # Separate sigma head: z → log_sigma² (per sample)
            self.sigma_head = nn.Sequential(
                nn.Linear(d_model, 64),
                nn.GELU(),
                nn.Linear(64, 1),
            )

        idim_kw = {'informed_dim': self.informed_dim, 'learnable_alpha': learnable_alpha}
        if multi_scale_fourier:
            # 3 Fourier trunks covering low/mid/high frequencies
            self.trunks = nn.ModuleList([
                HyperTrunk(trunk_w, 'fourier', nf=8, **idim_kw),
                HyperTrunk(trunk_w, 'fourier', nf=32, **idim_kw),
                HyperTrunk(trunk_w, 'fourier', nf=128, **idim_kw),
            ])
        elif hybrid_trunk:
            self.trunks = nn.ModuleList([
                HyperTrunk(trunk_w, 'fourier', nf=fourier_nf, **idim_kw),
                FixedTrunk(trunk_w, 'poly', d_model, **idim_kw),
                HyperTrunk(trunk_w, 'rbf', **idim_kw),
            ])
        elif use_latent_decomp:
            self.latent_decomp = LatentDecomp(d_model)
            self.trunks = nn.ModuleList([
                HyperTrunk(trunk_w, 'poly', **idim_kw),
                HyperTrunk(trunk_w, 'fourier', nf=fourier_nf, **idim_kw),
                HyperTrunk(trunk_w, 'rbf', **idim_kw),
            ])
        elif all_fixed:
            self.trunks = nn.ModuleList([
                FixedTrunk(trunk_w, 'fourier', d_model, nf=fourier_nf, **idim_kw),
                FixedTrunk(trunk_w, 'poly', d_model, **idim_kw),
                FixedTrunk(trunk_w, 'rbf', d_model, **idim_kw),
            ])
        else:
            poly_btype = 'cheby' if use_cheby else 'poly'
            self.trunks = nn.ModuleList([
                HyperTrunk(trunk_w, 'fourier', nf=fourier_nf, **idim_kw),
                HyperTrunk(trunk_w, poly_btype, **idim_kw),
                HyperTrunk(trunk_w, 'rbf', **idim_kw),
            ])
        # Optional: add high-frequency FixedTrunk for ETTh2/ETTm2 noise capture
        if highfreq_nf > 0:
            self.trunks.append(FixedTrunk(trunk_w, 'fourier', d_model, nf=highfreq_nf, **idim_kw))
        # Optional: 2nd higher-frequency trunk (e.g. nf=512) for sudden drift / sharp transitions
        if highfreq2_nf > 0:
            self.trunks.append(FixedTrunk(trunk_w, 'fourier', d_model, nf=highfreq2_nf, **idim_kw))
        # Heads only for HyperTrunk (FixedTrunk has its own coef_head)
        self.heads = nn.ModuleList([
            nn.Linear(d_model, t.odim) if not getattr(t, 'is_fixed', False)
            else nn.Identity()
            for t in self.trunks
        ])
        for h in self.heads:
            if isinstance(h, nn.Linear):
                nn.init.xavier_normal_(h.weight, gain=0.1)
                nn.init.constant_(h.bias, 0)
        self.biases = nn.ParameterList([nn.Parameter(torch.zeros(1)) for _ in self.trunks])

    def _build_iq(self, ctx, qt):
        """Informed query. ctx may have padded zeros."""
        if self.multi_scale_iq:
            freqs, phases, lv, ls = extract_freq_multiscale(
                ctx, top_k_per_scale=self.ms_iq_k, scales=self.ms_iq_scales)
        else:
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

        trunk_outs = []
        for zi, trunk, head, bias in zip(z_list, self.trunks, self.heads, self.biases):
            if getattr(trunk, 'is_fixed', False):
                trunk_outs.append(trunk(t_flat, iq, zi) + bias)
            else:
                trunk_outs.append(hyper_fwd(trunk, t_flat, head(zi), iq) + bias)
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
def collate_batch_varlen(windows, seq_len, n_query=64, mr=0.375, max_horizon_mult=2,
                         return_dense=False, unified_task=False, var_mask=False,
                         short_pred=False, mask_recon_only=False):
    """
    Same logic as V1 collate_batch but with fixed seq_len (caller samples per batch).
    Window samples should be >= seq_len*(1+max_horizon_mult).
    If return_dense=True, also returns dense future trajectory (for spectral loss).

    unified_task: if True, each sample does BOTH forecast and imputation
                  (half n_query for each, concatenated).
    mask_recon_only: if True, EVERY sample is masked reconstruction (FeDaL/MAE-style
                     pretrain). No forecast in training. Mutually exclusive with unified_task.
    var_mask: if True, mask rate sampled from U[0.1, 0.7] per sample.
    short_pred: if True, allow much smaller min horizon (max(4, seq_len//16)
                instead of max(8, seq_len//8)) — covers M4 short-term distribution.
    """
    ctxs, qts, qvs, pad_masks = [], [], [], []
    dense_futures, is_fc = [], []
    max_future = seq_len * max_horizon_mult
    min_horizon = max(4, seq_len // 16) if short_pred else max(8, seq_len // 8)

    def _pad_mask(real_len):
        m = np.zeros(seq_len, dtype=bool)
        if real_len < seq_len:
            m[real_len:] = True   # True = padded position (ignore)
        return m

    def _mask_rate():
        return float(np.random.uniform(0.1, 0.7)) if var_mask else mr

    for w in windows:
        w = w.numpy() if isinstance(w, torch.Tensor) else w
        avail_future = len(w) - seq_len

        if mask_recon_only:
            # Pure masked reconstruction: mask 35% (or var) of context, predict masked.
            real_len = min(len(w), seq_len)
            full_clean = w[:seq_len] if len(w) >= seq_len else np.pad(w, (0, seq_len - len(w)))
            this_pad = _pad_mask(real_len)
            this_mr = _mask_rate()
            mask = np.random.rand(seq_len) > this_mr
            mask_idx = np.where(~mask)[0]
            if len(mask_idx) == 0:
                mask[0] = False; mask_idx = np.array([0])
            if len(mask_idx) >= n_query:
                qi = np.random.choice(mask_idx, n_query, replace=False)
            else:
                qi = np.concatenate([mask_idx, np.random.choice(mask_idx, n_query - len(mask_idx), replace=True)])
            ctx_masked = full_clean * mask.astype(np.float32)
            qt = qi.astype(np.float32) / seq_len   # t ∈ [0, 1] within context
            qv = full_clean[qi]
            ctxs.append(ctx_masked.astype(np.float32))
            qts.append(qt); qvs.append(qv.astype(np.float32))
            pad_masks.append(this_pad.copy())
            dense_futures.append(np.zeros(max_future, dtype=np.float32))
            is_fc.append(False)
            continue

        if unified_task and avail_future > 0:
            # ===== Unified FIXED: produce TWO samples per window — =====
            # Sample A: clean ctx + forecast queries (t ∈ [1, 1+H/n])
            # Sample B: masked ctx + imputation queries (t ∈ [0, 1])
            # This keeps forecast context clean (matches eval distribution)
            # while giving imputation its own masked context.

            real_len = min(len(w), seq_len)
            full_clean = w[:seq_len] if len(w) >= seq_len else np.pad(w, (0, seq_len - len(w)))
            this_pad = _pad_mask(real_len)

            # ---- Sample A: Forecast on clean context ----
            future_len = min(avail_future, max_future)
            future = w[seq_len:seq_len + future_len]
            horizon_cap = np.random.randint(min(min_horizon, future_len), future_len + 1)
            fc_qi = np.random.choice(horizon_cap, min(n_query, horizon_cap), replace=False)
            if len(fc_qi) < n_query:
                fc_qi = np.concatenate([fc_qi, np.random.choice(horizon_cap, n_query - len(fc_qi), replace=True)])
            fc_qt = 1.0 + fc_qi.astype(np.float32) / seq_len
            fc_qv = future[fc_qi]
            if len(future) < max_future:
                dense_tgt = np.concatenate([future, np.zeros(max_future - len(future), dtype=np.float32)])
            else:
                dense_tgt = future[:max_future]
            ctxs.append(full_clean.astype(np.float32))
            qts.append(fc_qt); qvs.append(fc_qv.astype(np.float32))
            pad_masks.append(this_pad.copy())
            dense_futures.append(dense_tgt.astype(np.float32))
            is_fc.append(True)

            # ---- Sample B: Imputation on masked context ----
            this_mr = _mask_rate()
            mask = np.random.rand(seq_len) > this_mr
            imp_idx = np.where(~mask)[0]
            if len(imp_idx) == 0:
                mask[0] = False; imp_idx = np.array([0])
            if len(imp_idx) >= n_query:
                imp_qi = np.random.choice(imp_idx, n_query, replace=False)
            else:
                imp_qi = np.concatenate([imp_idx, np.random.choice(imp_idx, n_query - len(imp_idx), replace=True)])
            ctx_masked = full_clean * mask.astype(np.float32)
            imp_qt = imp_qi.astype(np.float32) / seq_len
            imp_qv = full_clean[imp_qi]
            ctxs.append(ctx_masked.astype(np.float32))
            qts.append(imp_qt); qvs.append(imp_qv.astype(np.float32))
            pad_masks.append(this_pad.copy())
            dense_futures.append(np.zeros(max_future, dtype=np.float32))
            is_fc.append(False)
            continue  # skip trailing append below (already appended both)
        else:
            real_len = min(len(w), seq_len)
            this_pad = _pad_mask(real_len)
            do_forecast = (np.random.rand() < 0.5 and avail_future > 0)
            if do_forecast:
                ctx = w[:seq_len]
                future_len = min(avail_future, max_future)
                future = w[seq_len:seq_len + future_len]
                horizon_cap = np.random.randint(min(min_horizon, future_len), future_len + 1)
                qi = np.random.choice(horizon_cap, min(n_query, horizon_cap), replace=False)
                if len(qi) < n_query:
                    qi = np.concatenate([qi, np.random.choice(horizon_cap, n_query - len(qi), replace=True)])
                qt = 1.0 + qi.astype(np.float32) / seq_len
                qv = future[qi]
                if len(future) < max_future:
                    dense_tgt = np.concatenate([future, np.zeros(max_future - len(future), dtype=np.float32)])
                else:
                    dense_tgt = future[:max_future]
                dense_futures.append(dense_tgt.astype(np.float32))
                is_fc.append(True)
            else:
                full = w[:seq_len] if len(w) >= seq_len else np.pad(w, (0, seq_len - len(w)))
                this_mr = _mask_rate()
                mask = np.random.rand(seq_len) > this_mr
                qi_idx = np.where(~mask)[0]
                # CRITICAL: restrict query positions to real (non-padded) region
                qi_idx = qi_idx[qi_idx < real_len]
                if len(qi_idx) == 0:
                    continue
                if len(qi_idx) >= n_query:
                    qi_idx = np.random.choice(qi_idx, n_query, replace=False)
                else:
                    qi_idx = np.concatenate([qi_idx, np.random.choice(qi_idx, n_query - len(qi_idx), replace=True)])
                ctx = full * mask.astype(np.float32)
                qt = qi_idx.astype(np.float32) / seq_len
                qv = full[qi_idx]
                dense_futures.append(np.zeros(max_future, dtype=np.float32))
                is_fc.append(False)
        ctxs.append(ctx); qts.append(qt); qvs.append(qv)
        pad_masks.append(this_pad)
    if not ctxs: return None
    out = (torch.tensor(np.stack(ctxs), dtype=torch.float32),
           torch.tensor(np.stack(qts), dtype=torch.float32),
           torch.tensor(np.stack(qvs), dtype=torch.float32),
           torch.tensor(np.stack(pad_masks), dtype=torch.bool))
    if return_dense:
        out = out + (
            torch.tensor(np.stack(dense_futures), dtype=torch.float32),
            torch.tensor(is_fc, dtype=torch.bool),
        )
    return out


# ============================================================
# Training with random seq_len per batch
# ============================================================
def train_varlen(model, datasets, save_path, max_seq_len, seq_len_choices,
                 epochs=40, lr=3e-4, batch_size=64, n_query=64,
                 spectral_weight=0.0, max_horizon_mult=2, use_nll=False,
                 use_huber=False, huber_delta=1.0,
                 unified_task=False, var_mask=False, short_pred=False,
                 resume_from=None, start_epoch=0, amp=False,
                 mask_recon_only=False):
    n_params = sum(p.numel() for p in model.parameters())
    combined = ConcatDataset(datasets)

    use_spec = spectral_weight > 0

    def _collate(batch_windows):
        seq_len_b = int(np.random.choice(seq_len_choices))
        return collate_batch_varlen(
            batch_windows, seq_len=seq_len_b, n_query=n_query,
            return_dense=use_spec, max_horizon_mult=max_horizon_mult,
            unified_task=unified_task, var_mask=var_mask,
            short_pred=short_pred, mask_recon_only=mask_recon_only)

    def _worker_init(wid):
        np.random.seed((torch.initial_seed() + wid) % (2**32))

    dl = DataLoader(combined, batch_size=batch_size, shuffle=True,
                    num_workers=16, drop_last=True, pin_memory=True,
                    persistent_workers=True,
                    collate_fn=_collate, worker_init_fn=_worker_init,
                    prefetch_factor=4)

    print(f'\n{"="*60}')
    print(f'VarLen V1 Training')
    print(f'  Model: {n_params/1e6:.1f}M, max_seq_len={max_seq_len}')
    print(f'  seq_len_choices: {seq_len_choices}')
    print(f'  Data: {len(combined):,} windows, Steps/epoch: {len(dl):,}')
    if resume_from:
        print(f'  Resume from: {resume_from} (starting at epoch {start_epoch+1})')
    print(f'{"="*60}')

    if resume_from:
        # strict=False so newly added params (e.g. learnable log_alpha) are
        # initialized to their default while everything else loads cleanly.
        sd = torch.load(resume_from, map_location=next(model.parameters()).device, weights_only=True)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing:
            print(f'  [resume] missing keys (defaulted): {missing}')
        if unexpected:
            print(f'  [resume] unexpected keys (skipped): {unexpected}')

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    for _ in range(start_epoch):  # advance LR scheduler to match epoch count
        scheduler.step()

    best_loss = float('inf')
    amp_ctx = (lambda: torch.autocast(device_type='cuda', dtype=torch.bfloat16)) if amp else (lambda: contextlib.nullcontext())
    if amp:
        print('  AMP: bfloat16 autocast enabled')
    for epoch in range(start_epoch, epochs):
        model.train()
        losses = []
        t0 = time.time()
        for i, batch in enumerate(dl):
            if batch is None: continue
            if use_spec:
                ctx, qt, qv, pad, dense_fut, is_fc = [x.to(DEVICE, non_blocking=True) if isinstance(x, torch.Tensor) else x
                                                       for x in batch]
            else:
                ctx, qt, qv, pad = [x.to(DEVICE, non_blocking=True) for x in batch]
            # Only pass mask if any padding actually exists (saves work for clean batches)
            pmask = pad if pad.any() else None
            optimizer.zero_grad()
            with amp_ctx():
                pred = model.forward_train(ctx, qt, padding_mask=pmask)
                if use_huber:
                    mse = F.huber_loss(pred, qv, delta=huber_delta)
                else:
                    mse = F.mse_loss(pred, qv)
                total = mse
                if use_nll and hasattr(model, 'sigma_head'):
                    z_for_sigma = model.encoder(ctx, padding_mask=pmask)
                    log_sigma2 = model.sigma_head(z_for_sigma).squeeze(-1)
                    var = log_sigma2.exp().clamp(min=1e-4)
                    sq_err = (pred - qv) ** 2
                    nll = 0.5 * log_sigma2.unsqueeze(-1) + 0.5 * sq_err / var.unsqueeze(-1)
                    total = total + nll.mean()
                if use_spec and is_fc.any():
                    max_future = seq_len_b * max_horizon_mult
                    fc_mask = is_fc
                    ctx_fc = ctx[fc_mask]
                    tgt_fc = dense_fut[fc_mask]
                    dense_t = torch.linspace(1.0, 1.0 + max_future / seq_len_b, max_future,
                                             device=DEVICE).unsqueeze(0).expand(ctx_fc.shape[0], -1)
                    pmask_fc = pad[fc_mask] if pmask is not None else None
                    if pmask_fc is not None and not pmask_fc.any():
                        pmask_fc = None
                    dense_pred = model.forward_train(ctx_fc, dense_t, padding_mask=pmask_fc)
                    pred_fft = torch.fft.rfft(dense_pred, dim=-1).abs()
                    tgt_fft = torch.fft.rfft(tgt_fc, dim=-1).abs()
                    spec_loss = F.mse_loss(pred_fft, tgt_fft) / max_future
                    total = total + spectral_weight * spec_loss
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(mse.item())
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
    p.add_argument('--eval_only', type=int, default=0,
                   help='Skip training; load checkpoint matching --tag and run eval only')
    p.add_argument('--resume', type=int, default=0,
                   help='If 1, resume training from checkpoints/{tag}.pth (combined with --start_epoch)')
    p.add_argument('--start_epoch', type=int, default=0,
                   help='Epoch to resume from (0-indexed; e.g. if last completed epoch was 23, pass 23)')
    p.add_argument('--windows_per_series', type=int, default=5,
                   help='LOTSA stride control: max windows extracted per series. 5=current (~0.12% coverage), 50=10x, 500=100x.')
    p.add_argument('--lotsa_subsample', type=int, default=0,
                   help='If >0, randomly subsample N windows from LOTSA cache (e.g. 10000000 for 10pct of MV cache). 0=use all.')
    p.add_argument('--use_cauker', type=int, default=0,
                   help='If 1, replace SyntheticGapFiller with CauKerSynthDataset (loads pre-built CauKer cache).')
    p.add_argument('--cauker_name', type=str, default='cauker_synth',
                   help='CauKer cache base name (lookup at dataset/cauker_cache/{name}.dat).')
    p.add_argument('--mask_recon_only', type=int, default=0,
                   help='If 1, use masked reconstruction as sole pretrain objective (FeDaL/MAE-style). Disables forecast pretrain.')
    p.add_argument('--lr', type=float, default=3e-4,
                   help='Learning rate (default 3e-4). Scale linearly with batch size.')
    p.add_argument('--batch_size', type=int, default=64,
                   help='Per-GPU batch size (default 64). Increase with AMP for speedup.')
    p.add_argument('--amp', type=int, default=0,
                   help='Enable bf16 mixed precision training (2x speedup on A100).')
    p.add_argument('--hybrid_trunk', type=int, default=0,
                   help='1: 1 Fixed (poly) + 2 Hyper (fourier/rbf)')
    p.add_argument('--spectral_weight', type=float, default=0.0,
                   help='Spectral loss weight (FFT magnitude MSE)')
    p.add_argument('--use_nll', type=int, default=0,
                   help='1: use Gaussian NLL loss (adds sigma head)')
    p.add_argument('--use_huber', type=int, default=0,
                   help='1: use Huber loss instead of MSE (outlier robust)')
    p.add_argument('--huber_delta', type=float, default=1.0)
    p.add_argument('--d_model', type=int, default=512)
    p.add_argument('--n_layers', type=int, default=6)
    p.add_argument('--trunk_w', type=int, default=192)
    p.add_argument('--fourier_nf', type=int, default=32,
                   help='Fourier basis frequencies (default 32; try 128 for high-freq capture)')
    p.add_argument('--multi_scale_fourier', type=int, default=0,
                   help='1: use 3 Fourier trunks with nf=[8, 32, 128] covering low/mid/high freq')
    p.add_argument('--unified_task', type=int, default=0,
                   help='1: each sample does BOTH forecast + imputation (half n_query each)')
    p.add_argument('--var_mask', type=int, default=0,
                   help='1: sample mask rate from U[0.1, 0.7] per sample (for imputation variety)')
    p.add_argument('--short_pred', type=int, default=0,
                   help='1: allow min horizon max(4, seq_len//16) for M4 short-term coverage')
    p.add_argument('--lotsa_short_aux', type=int, default=0,
                   help='Additional LOTSA loader at shorter window_len to capture short series (0=off, else window_len value e.g. 512)')
    p.add_argument('--multi_scale_iq', type=int, default=0,
                   help='1: use multi-scale FFT for IQ (3 scales × 3 peaks = 9 freqs, IQ dim 21)')
    p.add_argument('--pool_type', type=str, default='mean', choices=['mean', 'attn'],
                   help='Encoder pool: mean (default) or attn (learnable CLS cross-attention)')
    p.add_argument('--highfreq2_nf', type=int, default=0,
                   help='Optional 2nd Fourier trunk nf (e.g. 512). For sudden-drift / sharp transitions.')
    p.add_argument('--highfreq_nf', type=int, default=0,
                   help='Add 4th FixedTrunk Fourier with this nf (e.g. 256) for high-freq noise capture')
    p.add_argument('--all_fixed', type=int, default=0,
                   help='1: use all FixedTrunks (no HyperNet, no 0.01 scaling issue)')
    p.add_argument('--use_cheby', type=int, default=0,
                   help='1: replace HyperTrunk poly with Chebyshev T_n basis (shifted to t-1)')
    p.add_argument('--learnable_alpha', type=int, default=0,
                   help='1: make hypernet output scale α learnable per HyperTrunk '
                        '(initialized to 0.01 = legacy behavior). Adds one scalar param per trunk.')
    p.add_argument('--target_boost', type=int, default=0,
                   help='Windows per target-similar LOTSA dataset (0=off, e.g. 30000)')
    p.add_argument('--model_type', type=str, default='varlen', choices=['varlen', 'decomp'],
                   help='Model type: varlen (original) or decomp (input decomposition)')
    p.add_argument('--decomp_kernels', type=str, default='49,25,7',
                   help='Comma-sep MA kernel sizes for decomposition (e.g. 49,25 for 3-way or 49,25,7 for 4-way)')
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
    if args.eval_only:
        print('[eval_only] skipping dataset loading')
        lotsa_ds = synth_ds = None
        datasets = []
    else:
        lotsa_ds = LOTSAScalingDataset(LOTSA_DIR, args.scale, seq_len=window_len,
                                        windows_per_series=args.windows_per_series)
        if args.lotsa_subsample > 0 and args.lotsa_subsample < len(lotsa_ds):
            import random
            random.seed(42)
            n = args.lotsa_subsample
            print(f'[LOTSA subsample] {n:,} / {len(lotsa_ds):,} windows (seed=42)')
            indices = random.sample(range(len(lotsa_ds)), n)
            lotsa_ds = torch.utils.data.Subset(lotsa_ds, indices)
        if args.use_cauker:
            print(f'[Synth source] CauKer cache: {args.cauker_name}')
            synth_ds = CauKerSynthDataset(cache_name=args.cauker_name, seq_len=window_len)
        else:
            n_synth = args.synth_n if args.synth_n > 0 else max(10000, int(len(lotsa_ds) * args.synth_ratio))
            synth_ds = SyntheticGapFiller(n_samples=n_synth, seq_len=window_len)
        datasets = [lotsa_ds, synth_ds]
    # Option A: additional LOTSA loader at shorter window_len to capture short M4-like series
    lotsa_short = None
    if args.lotsa_short_aux > 0:
        print(f'\n[Option A] Loading short LOTSA at window_len={args.lotsa_short_aux}...')
        lotsa_short = LOTSAScalingDataset(LOTSA_DIR, args.scale, seq_len=args.lotsa_short_aux,
                                           windows_per_series=args.windows_per_series)
        datasets.append(lotsa_short)
    # Target boost: upweight LOTSA datasets similar to ETTh2/ETTm2/Weather
    lotsa_boost = None
    if args.target_boost > 0:
        print(f'\n[Target Boost] Loading target-similar LOTSA at wpd={args.target_boost}...')
        lotsa_boost = LOTSASubsetDataset(LOTSA_DIR, TARGET_SIMILAR_DATASETS,
                                          seq_len=window_len, wpd=args.target_boost)
        datasets.append(lotsa_boost)
    if not args.eval_only:
        total = sum(len(d) for d in datasets)
        short_n = len(lotsa_short) if lotsa_short is not None else 0
        boost_n = len(lotsa_boost) if lotsa_boost is not None else 0
        print(f'Total: {total:,} (LOTSA: {len(lotsa_ds):,}, Synth: {len(synth_ds):,}, '
              f'LOTSA_short: {short_n:,}, LOTSA_boost: {boost_n:,})')

    if args.model_type == 'decomp':
        decomp_k = tuple(int(x) for x in args.decomp_kernels.split(','))
        model = OperatorModelDecomp(
            max_seq_len=args.max_seq_len,
            d_model=args.d_model,
            n_layers=args.n_layers,
            trunk_w=args.trunk_w,
            nhead=8,
            fourier_nf=args.fourier_nf,
            pool_type=args.pool_type,
            highfreq_nf=args.highfreq_nf,
            all_fixed=bool(args.all_fixed),
            decomp_kernels=decomp_k,
            learnable_alpha=bool(args.learnable_alpha),
        ).to(DEVICE)
        n = sum(p.numel() for p in model.parameters())
        print(f'Model Decomp: {n/1e6:.1f}M params (all_fixed={bool(args.all_fixed)})')
    else:
        model = OperatorModelVarLen(
            max_seq_len=args.max_seq_len,
            d_model=args.d_model,
            n_layers=args.n_layers,
            trunk_w=args.trunk_w,
            hybrid_trunk=bool(args.hybrid_trunk),
            use_nll=bool(args.use_nll),
            fourier_nf=args.fourier_nf,
            multi_scale_fourier=bool(args.multi_scale_fourier),
            multi_scale_iq=bool(args.multi_scale_iq),
            pool_type=args.pool_type,
            highfreq_nf=args.highfreq_nf,
            highfreq2_nf=args.highfreq2_nf,
            all_fixed=bool(args.all_fixed),
            use_cheby=bool(args.use_cheby),
            learnable_alpha=bool(args.learnable_alpha),
        ).to(DEVICE)
        n = sum(p.numel() for p in model.parameters())
        print(f'Model V1 VarLen: {n/1e6:.1f}M params (hybrid={bool(args.hybrid_trunk)})')

    save_path = f'checkpoints/{args.tag}.pth'
    os.makedirs('checkpoints', exist_ok=True)
    if args.eval_only:
        if not os.path.exists(save_path):
            raise FileNotFoundError(f'checkpoint not found: {save_path}')
        print(f'[eval_only] skipping training, will load {save_path}')
    else:
        resume_from = save_path if args.resume else None
        best = train_varlen(model, datasets, save_path,
                            max_seq_len=args.max_seq_len,
                            seq_len_choices=seq_choices,
                            epochs=args.epochs,
                            batch_size=args.batch_size,
                            lr=args.lr,
                            spectral_weight=args.spectral_weight,
                            use_nll=bool(args.use_nll),
                            use_huber=bool(args.use_huber),
                            huber_delta=args.huber_delta,
                            unified_task=bool(args.unified_task),
                            var_mask=bool(args.var_mask),
                            short_pred=bool(args.short_pred),
                            resume_from=resume_from,
                            start_epoch=args.start_epoch,
                            amp=bool(args.amp),
                            mask_recon_only=bool(args.mask_recon_only))

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
