# HypOPFN — Model Architecture Reference

## Core Concept: Operator Learning for Time Series

Standard TSFM: `f: R^n → R^H` (fixed horizon)
Ours: `G: R^n → C(R)` (continuous function, arbitrary t)

```
y(t) = Σ_k β_k(x) · φ_k(t; x)     (DeepONet)
       ^^^^^^^^     ^^^^^^^^^^^^^
       Branch        Trunk (context-adaptive via HyperNet)
```

---

## Architecture Variants

### 1. OperatorModelVarLen (Original, 32.5M)
```
Context x₁:ₙ
  └→ VarLenPatchAttnEncoder (6L Transformer, d=512, patch=16, sinusoidal PE)
      └→ z ∈ R^512 (mean pool or attn pool)
          ├→ HyperNet head₁(z) × 0.01 → W₁,b₁,β₁ → Fourier trunk (nf=32)
          ├→ HyperNet head₂(z) × 0.01 → W₂,b₂,β₂ → Poly trunk (deg=6)
          └→ HyperNet head₃(z) × 0.01 → W₃,b₃,β₃ → RBF trunk (nc=20)
  
Context x₁:ₙ
  └→ extract_freq(x) → FFT top-5 freq/phase + last_val + slope
      └→ IQ(t) = [t, sin/cos(2πf·t+φ)×5, last_val, slope]  (13-dim)

Query t + IQ(t) → each trunk → sum → μ(t)
z → sigma_head → σ (per-sample, NLL only)
```

### 2. OperatorModelVarLen + all_fixed (Best eval, 19.3M)
Same but all 3 trunks are FixedTrunk:
```
FixedTrunk: basis(t,IQ) → fixed MLP → φ(t)
            z → coef_head → β
            out = φ · β
```
- No HyperNet, no 0.01 scaling → no "give up" problem
- 41% fewer params, same performance

### 3. OperatorModelVarLen + highfreq_nf=256 (4-trunk, 32.7M or 19.5M)
Adds 4th FixedTrunk Fourier(nf=256) for high-frequency noise capture:
```
Trunk 0: Fourier(nf=32)  — medium periodicity
Trunk 1: Poly(deg=6)     — trend
Trunk 2: RBF(nc=20)      — local events
Trunk 3: Fourier(nf=256) — high-freq (FixedTrunk, ETTh2/ETTm2 target)
```

### 4. OperatorModelDecomp (New, 86.6M with HyperTrunk)
Input-level decomposition → separate z per trunk:
```
Input x
  ├→ MA(49) → trend      → Encoder+proj₀ → z_trend    → HyperTrunk Poly(deg=6)
  ├→ MA(25) → seasonal   → Encoder+proj₁ → z_seasonal → HyperTrunk Fourier(nf=32)
  ├→ MA(7)  → highfreq   → Encoder+proj₂ → z_highfreq → HyperTrunk Fourier(nf=256)
  └→ rest   → residual   → Encoder+proj₃ → z_residual → HyperTrunk RBF(nc=20)

trend + seasonal + highfreq + residual = x  (perfect reconstruction)
```
- Each trunk receives specialized z → HyperTrunk won't "give up"
- Shared Transformer backbone + 4 projection heads (MLP)

---

## Key Components

### VarLenPatchAttnEncoder
- Patch size: 16
- Sinusoidal PE (length-independent, up to max_seq_len)
- Padding mask support (key_padding_mask in attention + masked mean pool)
- Pool types: 'mean' (default), 'attn' (learnable CLS cross-attention)
- Supports seq_len ∈ {192, 384, 512, 720} per batch

### HyperTrunk
- Basis: explicit formula (sin/cos, polynomial, Gaussian RBF)
- Weights: generated per-sample by HyperNet head(z)
- Scaling: × 0.01 (causes "give up" on noisy data)
- Forward: Φ = GELU(basis·W + b), out = β·Φ

### FixedTrunk
- Basis: same explicit formula
- Weights: fixed learnable MLP (not per-sample)
- Context adaptation: coef = coef_head(z), out = MLP(basis)·coef
- More stable, no 0.01 scaling issue

### SeriesDecomposer
- Moving average cascade: MA(k₁) → MA(k₂) → MA(k₃) → residual
- Default kernels: (49, 25, 7) for 4-way decomposition
- Differentiable, perfect reconstruction (components sum to input)

### Informed Query (IQ)
- FFT top-5 frequencies + phases from context
- last_val + slope (local trend)
- 13-dim (standard) or 21-dim (multi-scale IQ with 3 scales × 3 peaks)

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW, lr=3e-4, weight_decay=0.01 |
| Scheduler | CosineAnnealingLR, T_max=epochs |
| Batch size | 64 |
| n_query | 64 queries per sample |
| Grad clip | L2 norm 1.0 |
| Epochs | 40 |
| seq_len_choices | {192, 384, 512, 720} per batch |
| max_horizon_mult | 2 (future up to 2×seq_len) |
| Loss | MSE (NLL unstable with any change) |

## Data

| Component | Amount |
|-----------|--------|
| LOTSA (current server) | 74/174 datasets, ~27B obs |
| LOTSA (full, new server) | 174 datasets, ~231B obs |
| Synthetic | 500K windows, 20 domain generators |
| Eval excluded | ETT, Weather, M4, M1, M3 |

## Eval Results Summary (Best: mse_allfixed, 19.3M)

| Task | Dataset | Ours | FeDaL | Gap |
|------|---------|------|-------|-----|
| Forecast | ETTh2 avg | 0.374 | 0.349 | +7% |
| Forecast | Weather avg | 0.279 | 0.255 | +10% |
| Forecast | OVERALL | 0.398 | 0.325 | +22% |
| Imputation | ETTh2 | 0.097 | 0.092 | +6% |
| Imputation | OVERALL | 0.149 | 0.082 | +81% |
| M4 | AVG sMAPE | 25.46 | 9.74 | +161% |