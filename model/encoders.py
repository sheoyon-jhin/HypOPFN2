"""
Encoder variants for DeepONet Branch network.

All encoders have the same interface:
  Input:  x [B, input_dim]  — branch input (x_ch + x_cross + optional FFT)
  Output: z [B, output_dim] — task-agnostic representation

  For our setup: input_dim=290 (96+96+98), output_dim=256 (branch_hidden)

Variants:
  1. FNNEncoder     — Current baseline (3-layer FNN)
  2. PatchAttnEncoder — Patch the input, self-attention, project
  3. ConvEncoder    — 1D Conv for local patterns, then project
  4. LinearAttnEncoder — Linear attention (O(n) instead of O(n²))
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class FNNEncoder(nn.Module):
    """Original FNN encoder (baseline). Flat input → MLP → z."""
    def __init__(self, input_dim, output_dim, depth=3, activation='gelu', dropout=0.0):
        super().__init__()
        layers = []
        layers.append(nn.Linear(input_dim, output_dim))
        layers.append(self._get_activation(activation))
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        for _ in range(depth - 2):
            layers.append(nn.Linear(output_dim, output_dim))
            layers.append(self._get_activation(activation))
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)
        self._initialize()

    def _get_activation(self, activation):
        if activation == 'gelu': return nn.GELU()
        elif activation == 'tanh': return nn.Tanh()
        return nn.ReLU()

    def _initialize(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.net(x)


class PatchAttnEncoder(nn.Module):
    """Patch-based self-attention encoder.

    Splits the raw time series part of branch_input into patches,
    applies self-attention, then projects to output_dim.

    input_dim = seq_len + cross_dim + fft_dim
    We extract seq_len from the first part, patch it, attend, pool.
    """
    def __init__(self, input_dim, output_dim, seq_len=96, patch_size=16,
                 n_heads=4, n_layers=2, dropout=0.1, activation='gelu'):
        super().__init__()
        self.seq_len = seq_len
        self.patch_size = patch_size
        self.n_patches = seq_len // patch_size
        self.extra_dim = input_dim - seq_len  # cross_dim + fft_dim

        # Patch embedding: each patch → d_model
        d_model = output_dim
        self.patch_embed = nn.Linear(patch_size, d_model)

        # Positional encoding (learnable)
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches, d_model) * 0.02)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 2,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True
        )
        self.attn_layers = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Project extra features (cross-channel + FFT) to same dim
        if self.extra_dim > 0:
            self.extra_proj = nn.Linear(self.extra_dim, d_model)
        else:
            self.extra_proj = None

        # Final projection: combine attended patches + extra → output
        self.out_proj = nn.Sequential(
            nn.Linear(d_model, output_dim),
            nn.GELU(),
        )
        self._initialize()

    def _initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x: [B, input_dim] = [B, seq_len + extra_dim]
        B = x.shape[0]

        # Split into time series and extra features
        x_ts = x[:, :self.seq_len]       # [B, seq_len]
        x_extra = x[:, self.seq_len:]    # [B, extra_dim]

        # Patch: [B, seq_len] → [B, n_patches, patch_size]
        patches = x_ts.reshape(B, self.n_patches, self.patch_size)

        # Embed patches
        tokens = self.patch_embed(patches) + self.pos_embed  # [B, n_patches, d_model]

        # Self-attention
        tokens = self.attn_layers(tokens)  # [B, n_patches, d_model]

        # Pool: mean over patches
        z = tokens.mean(dim=1)  # [B, d_model]

        # Add extra features
        if self.extra_proj is not None and self.extra_dim > 0:
            z = z + self.extra_proj(x_extra)

        return self.out_proj(z)


class ConvEncoder(nn.Module):
    """1D Convolution encoder.

    Applies 1D convolutions to extract local temporal patterns,
    then pools and projects to output_dim.
    """
    def __init__(self, input_dim, output_dim, seq_len=96, dropout=0.1, activation='gelu'):
        super().__init__()
        self.seq_len = seq_len
        self.extra_dim = input_dim - seq_len

        # Conv layers: treat time series as 1-channel 1D signal
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, padding=3),
            nn.GELU(),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(128, output_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),  # [B, output_dim, 1]
        )

        # Project extra features
        if self.extra_dim > 0:
            self.extra_proj = nn.Linear(self.extra_dim, output_dim)
        else:
            self.extra_proj = None

        self.out_proj = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.GELU(),
        )
        self._initialize()

    def _initialize(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        B = x.shape[0]
        x_ts = x[:, :self.seq_len]     # [B, seq_len]
        x_extra = x[:, self.seq_len:]  # [B, extra_dim]

        # Conv: [B, seq_len] → [B, 1, seq_len] → [B, output_dim, 1]
        z = self.conv_layers(x_ts.unsqueeze(1)).squeeze(-1)  # [B, output_dim]

        if self.extra_proj is not None and self.extra_dim > 0:
            z = z + self.extra_proj(x_extra)

        return self.out_proj(z)


class LinearAttnEncoder(nn.Module):
    """Linear Attention encoder (O(n) complexity).

    Uses ELU-based kernel approximation for linear attention.
    Patches input like PatchAttnEncoder but with O(n) attention.
    """
    def __init__(self, input_dim, output_dim, seq_len=96, patch_size=16,
                 n_heads=4, n_layers=2, dropout=0.1, activation='gelu'):
        super().__init__()
        self.seq_len = seq_len
        self.patch_size = patch_size
        self.n_patches = seq_len // patch_size
        self.extra_dim = input_dim - seq_len
        self.n_heads = n_heads

        d_model = output_dim
        self.d_head = d_model // n_heads

        self.patch_embed = nn.Linear(patch_size, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches, d_model) * 0.02)

        # Linear attention layers
        self.layers = nn.ModuleList([
            LinearAttnLayer(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])

        if self.extra_dim > 0:
            self.extra_proj = nn.Linear(self.extra_dim, d_model)
        else:
            self.extra_proj = None

        self.out_proj = nn.Sequential(
            nn.Linear(d_model, output_dim),
            nn.GELU(),
        )
        self._initialize()

    def _initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        B = x.shape[0]
        x_ts = x[:, :self.seq_len]
        x_extra = x[:, self.seq_len:]

        patches = x_ts.reshape(B, self.n_patches, self.patch_size)
        tokens = self.patch_embed(patches) + self.pos_embed

        for layer in self.layers:
            tokens = layer(tokens)

        z = tokens.mean(dim=1)

        if self.extra_proj is not None and self.extra_dim > 0:
            z = z + self.extra_proj(x_extra)

        return self.out_proj(z)


class LinearAttnLayer(nn.Module):
    """Single linear attention layer with pre-norm."""
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.norm1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # Pre-norm linear attention
        h = self.norm1(x)
        B, N, D = h.shape
        qkv = self.qkv(h).reshape(B, N, 3, self.n_heads, self.d_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, heads, N, d_head]

        # ELU kernel: φ(x) = ELU(x) + 1
        q = F.elu(q) + 1
        k = F.elu(k) + 1

        # Linear attention: O(n) via KV first
        kv = torch.einsum('bhnd,bhne->bhde', k, v)  # [B, heads, d_head, d_head]
        z = torch.einsum('bhnd,bhde->bhne', q, kv)   # [B, heads, N, d_head]
        # Normalize
        k_sum = k.sum(dim=2, keepdim=True)  # [B, heads, 1, d_head]
        denom = torch.einsum('bhnd,bhkd->bhn', q, k_sum).unsqueeze(-1).clamp(min=1e-6)
        z = z / denom

        z = z.transpose(1, 2).reshape(B, N, D)
        x = x + self.drop(self.out(z))

        # FFN
        x = x + self.ff(self.norm2(x))
        return x


def build_encoder(encoder_type, input_dim, output_dim, seq_len=96, depth=3,
                  activation='gelu', dropout=0.1):
    """Factory function to create encoder by type string."""
    if encoder_type == 'fnn':
        return FNNEncoder(input_dim, output_dim, depth=depth,
                          activation=activation, dropout=dropout)
    elif encoder_type == 'patch_attn':
        return PatchAttnEncoder(input_dim, output_dim, seq_len=seq_len,
                                patch_size=16, n_heads=4, n_layers=2, dropout=dropout)
    elif encoder_type == 'conv':
        return ConvEncoder(input_dim, output_dim, seq_len=seq_len, dropout=dropout)
    elif encoder_type == 'linear_attn':
        return LinearAttnEncoder(input_dim, output_dim, seq_len=seq_len,
                                  patch_size=16, n_heads=4, n_layers=2, dropout=dropout)
    else:
        raise ValueError(f'Unknown encoder_type: {encoder_type}')
