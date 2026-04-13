"""
DeepONet with Hypernetwork-based Trunk for Time Series Forecasting
+ Residual Linear skip connection + Fourier features for stable optimization
- Channel-independent processing: works with any number of channels

New capabilities (all optional, off by default):
  --spectral_branch      : FFT features added to Branch input (aligns with LMC frequency prior)
  --latent_cross_channel : Replace raw channel mean with learnable latent projection
  --n_latent K           : Number of latent dims for cross-channel (default 16)
  --spectral_trunk       : Add parallel spectral synthesis output path (frequency domain)
  --n_spectral_modes K   : Number of Fourier modes for spectral trunk (default 32)
  --functional_input     : Add Fourier-encoded time coordinates to Branch input
  --n_freq_input K       : Number of freq for functional input encoding (default 16)
  --mc_dropout           : Enable MC Dropout at inference for probabilistic output
  --mc_samples N         : Number of MC Dropout forward passes (default 100)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class FNN(nn.Module):
    """Feedforward Neural Network"""
    def __init__(self, input_dim, output_dim, depth, width, activation='relu', dropout=0.0):
        super(FNN, self).__init__()

        layers = []
        layers.append(nn.Linear(input_dim, width))
        layers.append(self._get_activation(activation))
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))

        for _ in range(depth - 2):
            layers.append(nn.Linear(width, width))
            layers.append(self._get_activation(activation))
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(width, output_dim))

        self.net = nn.Sequential(*layers)
        self._initialize()

    def _get_activation(self, activation):
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'gelu':
            return nn.GELU()
        elif activation == 'tanh':
            return nn.Tanh()
        else:
            return nn.ReLU()

    def _initialize(self):
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        return self.net(x)


class Model(nn.Module):
    """
    Channel-Independent DeepONet with Hypernetwork-based Trunk Network
    + Residual Linear skip connection for stable optimization

    Architecture per channel:
      Linear(x_channel) → base prediction [batch, pred_len]   (residual)
      Branch(x_channel) → [Trunk weights θ, Coefficients B]
      Trunk_θ(fourier_t) → Basis functions Φ (adaptive per input!)
      Output = Linear(x) + B ⊗ Φ

    Optional extensions:
      spectral_branch:      Branch also receives FFT(x_channel) as input
      latent_cross_channel: Cross-channel summary via shared latent projection instead of raw mean
      spectral_trunk:       Branch additionally outputs Fourier coefficients for a parallel
                            spectral synthesis path: output += Σ_k c_k cos(2πkt) + d_k sin(2πkt)
    """

    def __init__(self, configs):
        super(Model, self).__init__()

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.use_norm = getattr(configs, 'use_norm', True)
        self.output_variance = getattr(configs, 'loss', 'MSE') == 'gaussian_nll'

        self.branch_depth = getattr(configs, 'branch_depth', 3)
        self.trunk_depth = getattr(configs, 'trunk_depth', 3)
        self.width = getattr(configs, 'deeponet_width', 64)
        self.activation = getattr(configs, 'activation', 'gelu')
        self.dropout = getattr(configs, 'dropout', 0.0)

        # --- New capability flags ---
        self.spectral_branch = getattr(configs, 'spectral_branch', False)
        self.latent_cross = getattr(configs, 'latent_cross_channel', False)
        self.n_latent = getattr(configs, 'n_latent', 16)
        self.spectral_trunk_flag = getattr(configs, 'spectral_trunk', False)
        self.n_spectral_modes = getattr(configs, 'n_spectral_modes', 32)
        self.functional_input = getattr(configs, 'functional_input', False)
        self.n_freq_input = getattr(configs, 'n_freq_input', 16)
        self.mc_dropout = getattr(configs, 'mc_dropout', False)
        self.mc_samples = getattr(configs, 'mc_samples', 100)
        # skip_mode: 'default' = linear_skip as-is, 'none' = no skip, 'adaptive' = Trunk-based skip
        self.skip_mode = getattr(configs, 'skip_mode', 'default')

        # Residual linear: direct mapping from input to output
        self.linear_skip = nn.Linear(self.seq_len, self.pred_len)

        # Fourier features for Trunk input
        self.n_freq = 32
        trunk_input_dim = 1 + 2 * self.n_freq

        # Calculate Trunk parameter count
        trunk_param_count = 0
        trunk_param_shapes = []

        # Layer 1: [trunk_input_dim, width]
        trunk_param_count += trunk_input_dim * self.width + self.width
        trunk_param_shapes.append((trunk_input_dim, self.width, self.width))

        # Layers 2 to depth-1: [width, width]
        for _ in range(2, self.trunk_depth):
            trunk_param_count += self.width * self.width + self.width
            trunk_param_shapes.append((self.width, self.width, self.width))

        self.trunk_param_shapes = trunk_param_shapes
        self.trunk_param_count = trunk_param_count

        # --- Latent cross-channel projector ---
        # Replaces raw channel mean with a learned projection to n_latent dims
        if self.latent_cross:
            self.latent_proj = nn.Linear(self.seq_len, self.n_latent)
            cross_dim = self.n_latent
        else:
            cross_dim = self.seq_len  # raw mean (original behavior)

        # --- Functional input encoding ---
        # Adds Fourier-encoded time coordinates to Branch input,
        # so the Branch knows WHERE each value sits in time (function → function mapping)
        if self.functional_input:
            # t_i = linspace(0,1,seq_len) → [sin(2πkt), cos(2πkt)] for k=1..n_freq_input
            # Flattened: seq_len * (1 + 2*n_freq_input) features
            func_dim = self.seq_len * (1 + 2 * self.n_freq_input)
            # Pre-compute and register as buffer (fixed, not learned)
            t_input = torch.linspace(0, 1, self.seq_len).unsqueeze(-1)  # [seq_len, 1]
            freq_k = torch.arange(1, self.n_freq_input + 1).float()
            t_encoded = torch.cat([
                t_input,
                torch.sin(2 * math.pi * freq_k.unsqueeze(0) * t_input),
                torch.cos(2 * math.pi * freq_k.unsqueeze(0) * t_input),
            ], dim=-1)  # [seq_len, 1+2*n_freq_input]
            self.register_buffer('func_input_encoding', t_encoded.flatten())  # [func_dim]
        else:
            func_dim = 0

        # --- Branch input dimension ---
        # Base: [x_ch (seq_len) + cross_channel_info (cross_dim)]
        branch_dim = self.seq_len + cross_dim + func_dim
        if self.spectral_branch:
            # FFT of x_ch: seq_len → seq_len//2+1 complex → real+imag
            n_fft_feats = (self.seq_len // 2 + 1) * 2
            branch_dim += n_fft_feats

        # --- Branch output dimension ---
        # Base: trunk_params + B (width)
        # Spectral trunk adds: 2 * n_spectral_modes (c_k, d_k for each mode)
        branch_output_dim = trunk_param_count + self.width
        if self.spectral_trunk_flag:
            branch_output_dim += 2 * self.n_spectral_modes
            # Register fixed frequencies (1, 2, ..., K)
            freqs = torch.arange(1, self.n_spectral_modes + 1).float()
            self.register_buffer('spectral_freqs', freqs)

        _bh = getattr(configs, 'branch_hidden', -1)
        self.branch_hidden = _bh if _bh > 0 else self.width * 4
        self.branch_net = FNN(branch_dim, branch_output_dim, self.branch_depth,
                             self.branch_hidden, self.activation, self.dropout)

        # Bias
        self.bias = nn.Parameter(torch.zeros([1]))

        # Variance head for Gaussian NLL loss
        # Predicts log_var per time step from the context (channel-independent)
        if self.output_variance:
            self.log_var_head = nn.Sequential(
                nn.Linear(self.seq_len, self.width * 2),
                nn.GELU(),
                nn.Linear(self.width * 2, self.pred_len),
            )

    def _get_fourier_features(self, t):
        """
        t: [pred_len, 1]
        Returns: [pred_len, 1 + 2*n_freq]
        """
        freqs = torch.arange(1, self.n_freq + 1, dtype=t.dtype, device=t.device)
        sin_features = torch.sin(2 * math.pi * freqs.unsqueeze(0) * t)
        cos_features = torch.cos(2 * math.pi * freqs.unsqueeze(0) * t)
        return torch.cat([t, sin_features, cos_features], dim=-1)

    def _build_trunk_weights(self, trunk_params, batch_size):
        """Build per-sample Trunk weights from Branch output."""
        trunk_weights = []
        param_idx = 0

        for in_dim, out_dim, bias_size in self.trunk_param_shapes:
            weight_size = in_dim * out_dim
            weight = trunk_params[:, param_idx:param_idx + weight_size].view(batch_size, in_dim, out_dim)
            param_idx += weight_size

            bias = trunk_params[:, param_idx:param_idx + bias_size].view(batch_size, out_dim)
            param_idx += bias_size

            trunk_weights.append((weight, bias))

        return trunk_weights

    def _forward_single_channel(self, x_ch, x_cross, target_pred_len=None, query_points=None):
        """
        x_ch:    [batch, seq_len]       - this channel
        x_cross: [batch, cross_dim]     - cross-channel info (raw mean OR latent projection)
        target_pred_len: if None, uses self.pred_len
        query_points: [n_points] tensor in [0, 1] for irregular query; None = linspace
        Returns: [batch, n_query_points]
        """
        if target_pred_len is None:
            target_pred_len = self.pred_len

        batch_size = x_ch.shape[0]

        # Residual: skip connection
        if self.skip_mode == 'none':
            # No skip: pure DeepONet output only
            base = torch.zeros(batch_size, target_pred_len, dtype=x_ch.dtype, device=x_ch.device)
        else:
            # 'default' or 'adaptive': use linear_skip with interpolation
            base = self.linear_skip(x_ch)  # [batch, self.pred_len]
            if query_points is not None:
                n_points = query_points.shape[0]
                idx_float = query_points * (self.pred_len - 1)
                idx_low = idx_float.long().clamp(0, self.pred_len - 2)
                idx_high = (idx_low + 1).clamp(max=self.pred_len - 1)
                w = (idx_float - idx_low.float()).unsqueeze(0)
                base = base[:, idx_low] * (1 - w) + base[:, idx_high] * w
                target_pred_len = n_points
            elif target_pred_len != self.pred_len:
                base = F.interpolate(
                    base.unsqueeze(1), size=target_pred_len,
                    mode='linear', align_corners=True
                ).squeeze(1)

        # --- Build Branch input ---
        branch_input = torch.cat([x_ch, x_cross], dim=-1)  # [batch, seq_len + cross_dim]

        if self.functional_input:
            # Append fixed time coordinate encoding (same for every sample)
            func_enc = self.func_input_encoding.unsqueeze(0).expand(batch_size, -1)
            branch_input = torch.cat([branch_input, func_enc], dim=-1)

        if self.spectral_branch:
            # FFT of input: captures frequency content aligned with LMC prior structure
            x_fft = torch.fft.rfft(x_ch, dim=-1)             # [batch, seq_len//2+1] complex
            x_spectral = torch.cat([x_fft.real, x_fft.imag], dim=-1)  # [batch, 2*(seq_len//2+1)]
            branch_input = torch.cat([branch_input, x_spectral], dim=-1)

        branch_output = self.branch_net(branch_input)

        # --- Parse Branch output ---
        trunk_params = branch_output[:, :self.trunk_param_count]
        B = branch_output[:, self.trunk_param_count:self.trunk_param_count + self.width]

        if self.spectral_trunk_flag:
            spectral_coeffs = branch_output[:, self.trunk_param_count + self.width:]  # [batch, 2*K]
            K = self.n_spectral_modes
            c_k = spectral_coeffs[:, :K]   # [batch, K] cosine coefficients
            d_k = spectral_coeffs[:, K:]   # [batch, K] sine coefficients

        # --- Build dynamic Trunk weights (hypernetwork) ---
        trunk_params = trunk_params * 0.01
        trunk_weights = self._build_trunk_weights(trunk_params, batch_size)

        # --- Trunk: query at specified time points ---
        if query_points is not None:
            t_output = query_points.unsqueeze(-1)  # [n_points, 1]
        else:
            t_output = torch.linspace(0, 1, target_pred_len, dtype=x_ch.dtype, device=x_ch.device)
            t_output = t_output.unsqueeze(-1)  # [target_pred_len, 1]
        t_fourier = self._get_fourier_features(t_output)  # [n_points, 1+2*n_freq]

        Phi = t_fourier.unsqueeze(0).expand(batch_size, -1, -1)

        act_fn = F.gelu if self.activation == 'gelu' else F.relu
        for i, (weight, bias) in enumerate(trunk_weights):
            Phi = torch.bmm(Phi, weight) + bias.unsqueeze(1)
            if i < len(trunk_weights) - 1:
                Phi = act_fn(Phi)

        # DeepONet output: sum_k B[b,k] * Phi[b,q,k]
        deeponet_out = torch.einsum('bp,bqp->bq', B, Phi)  # [batch, n_points]

        # --- Spectral synthesis path (parallel to DeepONet) ---
        if self.spectral_trunk_flag:
            # output += Σ_k c_k * cos(2π*k*t) + d_k * sin(2π*k*t)
            freqs = self.spectral_freqs  # [K]
            t_flat = t_output.squeeze(-1)  # [n_points]
            cos_bank = torch.cos(2 * math.pi * freqs.unsqueeze(0) * t_flat.unsqueeze(-1))  # [n_points, K]
            sin_bank = torch.sin(2 * math.pi * freqs.unsqueeze(0) * t_flat.unsqueeze(-1))  # [n_points, K]
            spectral_out = (torch.einsum('bk,qk->bq', c_k, cos_bank) +
                            torch.einsum('bk,qk->bq', d_k, sin_bank))  # [batch, n_points]
            output = base + deeponet_out + spectral_out + self.bias
        else:
            output = base + deeponet_out + self.bias

        return output

    def forecast(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None,
                 target_pred_len=None, query_points=None):
        """
        x_enc: [batch, seq_len, channels]
        target_pred_len: if None, uses self.pred_len
        query_points: [n_points] tensor for irregular time query
        Returns: [batch, target_pred_len or n_points, channels]
        """
        if target_pred_len is None and query_points is None:
            target_pred_len = self.pred_len

        batch_size, seq_len, n_channels = x_enc.shape

        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc = x_enc / stdev

        # --- Cross-channel summary ---
        if self.latent_cross:
            # Each channel projected to n_latent dims, then averaged across channels
            # x_enc: [batch, seq_len, n_ch] → [batch, n_ch, seq_len]
            x_all = x_enc.permute(0, 2, 1)                      # [batch, n_ch, seq_len]
            latent_per_ch = self.latent_proj(x_all)             # [batch, n_ch, n_latent]
            x_cross = latent_per_ch.mean(dim=1)                 # [batch, n_latent]
        else:
            x_cross = x_enc.mean(dim=-1)                        # [batch, seq_len] — original

        outputs = []
        for ch in range(n_channels):
            x_ch = x_enc[:, :, ch]  # [batch, seq_len]
            out_ch = self._forward_single_channel(x_ch, x_cross, target_pred_len, query_points)
            outputs.append(out_ch)

        output = torch.stack(outputs, dim=-1)  # [batch, n_points, channels]

        # Variance prediction for Gaussian NLL
        log_var = None
        if self.output_variance:
            log_var_list = []
            for ch in range(n_channels):
                x_ch = x_enc[:, :, ch]  # [batch, seq_len]
                lv = self.log_var_head(x_ch)  # [batch, pred_len]
                # Clamp to prevent variance explosion/collapse
                # exp(-6)≈0.0025, exp(4)≈55 — reasonable variance range
                lv = torch.clamp(lv, min=-6.0, max=4.0)
                log_var_list.append(lv)
            log_var = torch.stack(log_var_list, dim=-1)  # [batch, pred_len, channels]

        if self.use_norm:
            output = output * stdev
            output = output + means
            if log_var is not None:
                # Scale variance by stdev^2: log(var * stdev^2) = log_var + 2*log(stdev)
                log_var = log_var + 2 * torch.log(stdev + 1e-5)

        if log_var is not None:
            return output, log_var
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                target_pred_len=None, query_points=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec,
                                target_pred_len, query_points)
        if isinstance(dec_out, tuple):
            mean, log_var = dec_out
            return mean[:, -mean.shape[1]:, :], log_var[:, -log_var.shape[1]:, :]
        return dec_out[:, -dec_out.shape[1]:, :]

    def mc_forecast(self, x_enc, target_pred_len=None, n_samples=None):
        """
        MC Dropout forecast: run forward N times with dropout enabled.
        Returns: [n_samples, batch, pred_len, channels]
        """
        if n_samples is None:
            n_samples = self.mc_samples

        # Enable dropout during inference
        self.branch_net.train()  # turns on Dropout layers

        samples = []
        with torch.no_grad():
            for _ in range(n_samples):
                out = self.forecast(x_enc, target_pred_len=target_pred_len)
                samples.append(out)

        self.branch_net.eval()  # restore eval mode
        return torch.stack(samples, dim=0)  # [n_samples, batch, pred_len, channels]
