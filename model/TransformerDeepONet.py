"""
Transformer-Enhanced DeepONet
- Transformer encoder replaces FNN Branch (attention-based input processing)
- DeepONet Trunk remains the same (hypernetwork basis functions)
- Still operator learning! Just with better input representation.

Advantages over FNN Branch:
  1. Attention can do "pattern copying" (key for beating Seasonal Naive)
  2. Naturally handles variable-length context
  3. Scales better with parameters
  4. Position-aware (RoPE or learned positional encoding)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerBranch(nn.Module):
    """Transformer encoder that replaces FNN Branch.
    Input: [batch, seq_len] (single channel time series)
    Output: [batch, output_dim] (trunk_params + B, same as FNN Branch)
    """
    def __init__(self, seq_len, output_dim, d_model=128, n_heads=4, n_layers=3, dropout=0.1,
                 use_spectral=False):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.use_spectral = use_spectral

        # Input embedding: value + position
        # Each timestep: value (1) + cross_channel_mean (1) = 2
        input_dim = 2
        if use_spectral:
            # Add FFT features per token (not flattened, but per-position)
            input_dim += 2  # will add sin/cos of dominant frequency

        self.input_proj = nn.Linear(input_dim, d_model)

        # Learnable positional encoding
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Pool + project to Branch output
        self.pool_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, output_dim),
        )

    def forward(self, x_ch, x_cross):
        """
        x_ch: [batch, seq_len] - single channel values
        x_cross: [batch, seq_len] - cross-channel mean
        Returns: [batch, output_dim] - same as FNN Branch output
        """
        batch_size = x_ch.shape[0]

        # Build per-token features: [batch, seq_len, input_dim]
        tokens = torch.stack([x_ch, x_cross], dim=-1)  # [batch, seq_len, 2]

        if self.use_spectral:
            # Add local spectral info per token
            # Simple: normalized position within dominant frequency
            x_fft = torch.fft.rfft(x_ch, dim=-1)
            magnitudes = x_fft.abs()
            # Find dominant frequency (skip DC)
            dom_freq = magnitudes[:, 1:].argmax(dim=-1) + 1  # [batch]
            # Per-position phase features
            t = torch.arange(self.seq_len, device=x_ch.device).float().unsqueeze(0)  # [1, seq_len]
            phase = 2 * math.pi * dom_freq.unsqueeze(1).float() * t / self.seq_len  # [batch, seq_len]
            spec_features = torch.stack([torch.sin(phase), torch.cos(phase)], dim=-1)  # [batch, seq_len, 2]
            tokens = torch.cat([tokens, spec_features], dim=-1)  # [batch, seq_len, 4]

        # Project to d_model
        x = self.input_proj(tokens)  # [batch, seq_len, d_model]

        # Add positional encoding
        x = x + self.pos_embedding[:, :self.seq_len, :]

        # Transformer encoder
        x = self.transformer(x)  # [batch, seq_len, d_model]

        # Pool: mean over sequence
        pooled = x.mean(dim=1)  # [batch, d_model]

        # Project to Branch output dimension
        output = self.pool_proj(pooled)  # [batch, output_dim]

        return output


class Model(nn.Module):
    """
    Transformer-Enhanced DeepONet

    Architecture per channel:
      Transformer(x_channel, x_cross) → [Trunk weights θ, Coefficients B]
      Trunk_θ(fourier_t) → Basis functions Φ
      Output = Linear(x) + B ⊗ Φ

    Same operator learning structure as DeepONetHyper,
    but with Transformer replacing FNN Branch.
    """
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.use_norm = getattr(configs, 'use_norm', True)
        self.width = getattr(configs, 'deeponet_width', 64)
        self.trunk_depth = getattr(configs, 'trunk_depth', 2)
        self.activation = getattr(configs, 'activation', 'gelu')
        self.dropout = getattr(configs, 'dropout', 0.1)
        self.use_spectral = getattr(configs, 'spectral_branch', False)

        # Transformer config
        self.d_model = getattr(configs, 'd_model', 128)
        self.n_heads = getattr(configs, 'n_heads', 4)
        self.n_layers = getattr(configs, 'e_layers', 3)

        # Linear skip (same as DeepONetHyper)
        self.linear_skip = nn.Linear(self.seq_len, self.pred_len)

        # Fourier features for Trunk
        self.n_freq = 32
        trunk_input_dim = 1 + 2 * self.n_freq

        # Trunk parameter shapes
        trunk_param_count = 0
        trunk_param_shapes = []
        trunk_param_count += trunk_input_dim * self.width + self.width
        trunk_param_shapes.append((trunk_input_dim, self.width, self.width))
        for _ in range(2, self.trunk_depth):
            trunk_param_count += self.width * self.width + self.width
            trunk_param_shapes.append((self.width, self.width, self.width))

        self.trunk_param_shapes = trunk_param_shapes
        self.trunk_param_count = trunk_param_count

        # Branch output dim
        branch_output_dim = trunk_param_count + self.width

        # Transformer Branch (replaces FNN)
        self.branch = TransformerBranch(
            seq_len=self.seq_len,
            output_dim=branch_output_dim,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            dropout=self.dropout,
            use_spectral=self.use_spectral,
        )

        self.bias = nn.Parameter(torch.zeros([1]))

    def _get_fourier_features(self, t):
        freqs = torch.arange(1, self.n_freq + 1, dtype=t.dtype, device=t.device)
        return torch.cat([t, torch.sin(2*math.pi*freqs.unsqueeze(0)*t),
                         torch.cos(2*math.pi*freqs.unsqueeze(0)*t)], dim=-1)

    def _forward_single_channel(self, x_ch, x_cross, target_pred_len=None):
        if target_pred_len is None:
            target_pred_len = self.pred_len

        batch_size = x_ch.shape[0]

        # Linear skip
        base = self.linear_skip(x_ch)
        if target_pred_len != self.pred_len:
            base = F.interpolate(base.unsqueeze(1), size=target_pred_len,
                               mode='linear', align_corners=True).squeeze(1)

        # Transformer Branch
        branch_output = self.branch(x_ch, x_cross)  # [batch, trunk_params + width]

        trunk_params = branch_output[:, :self.trunk_param_count] * 0.01
        B = branch_output[:, self.trunk_param_count:]

        # Build trunk weights
        trunk_weights = []
        idx = 0
        for in_dim, out_dim, bias_size in self.trunk_param_shapes:
            w_size = in_dim * out_dim
            w = trunk_params[:, idx:idx+w_size].view(batch_size, in_dim, out_dim)
            idx += w_size
            b = trunk_params[:, idx:idx+bias_size].view(batch_size, out_dim)
            idx += bias_size
            trunk_weights.append((w, b))

        # Trunk forward
        t_output = torch.linspace(0, 1, target_pred_len,
                                  dtype=x_ch.dtype, device=x_ch.device).unsqueeze(-1)
        Phi = self._get_fourier_features(t_output).unsqueeze(0).expand(batch_size, -1, -1)

        act_fn = F.gelu if self.activation == 'gelu' else F.relu
        for i, (w, b) in enumerate(trunk_weights):
            Phi = torch.bmm(Phi, w) + b.unsqueeze(1)
            if i < len(trunk_weights) - 1:
                Phi = act_fn(Phi)

        deeponet_out = torch.einsum('bp,bqp->bq', B, Phi)
        return base + deeponet_out + self.bias

    def forecast(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None,
                 target_pred_len=None, query_points=None):
        if target_pred_len is None:
            target_pred_len = self.pred_len

        batch_size, seq_len, n_channels = x_enc.shape

        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc = x_enc / stdev

        x_cross = x_enc.mean(dim=-1)

        outputs = []
        for ch in range(n_channels):
            x_ch = x_enc[:, :, ch]
            out_ch = self._forward_single_channel(x_ch, x_cross, target_pred_len)
            outputs.append(out_ch)

        output = torch.stack(outputs, dim=-1)

        if self.use_norm:
            output = output * stdev + means

        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                target_pred_len=None, query_points=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec,
                                target_pred_len, query_points)
        return dec_out[:, -dec_out.shape[1]:, :]
