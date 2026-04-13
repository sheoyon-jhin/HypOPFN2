"""
MoE-DeepONet: Mixture of Expert DeepONets
- Multiple small DeepONet experts (each with own Branch + Trunk)
- Router network selects/weights experts based on input
- Increases total capacity without increasing per-expert width (avoids NaN)
- Experts naturally specialize through training

Encoder-Head split:
  encoder (layers 1..depth-1) produces a task-agnostic representation z
  forecast_head (last layer) maps z → trunk_params + B for forecasting
  Other task heads can reuse z directly.

  Checkpoint compatibility: old key 'branch_net.net.X' maps to
  encoder.net.X (X < last) and forecast_head.weight/bias (X == last).

Usage:
  --model DeepONetHyperMoE --n_experts 4
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.encoders import build_encoder


class FNN(nn.Module):
    """Feedforward Neural Network"""
    def __init__(self, input_dim, output_dim, depth, width, activation='gelu', dropout=0.0):
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


class Encoder(nn.Module):
    """Shared encoder: maps branch input → representation z.

    This is layers 1..(depth-1) of the old FNN branch_net.
    Output dim = width (256 by default).
    """
    def __init__(self, input_dim, width, depth, activation='gelu', dropout=0.0):
        super().__init__()
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


class SingleExpert(nn.Module):
    """One DeepONet expert: Encoder → z → task_head → Trunk weights + B

    Has separate heads for different tasks:
      - forecast_head: z → trunk_params + B (for future prediction)
      - recon_head: z → trunk_params + B (for reconstruction/imputation)
    Both share the same Trunk architecture but with independent weights.
    """
    def __init__(self, branch_dim, width, branch_depth, trunk_depth,
                 branch_hidden, n_freq, activation, dropout,
                 encoder_type='fnn', seq_len=96, trunk_basis='fourier',
                 n_rbf=16):
        super().__init__()

        self.width = width
        self.n_freq = n_freq
        self.n_rbf = n_rbf
        self.activation = activation
        self.trunk_basis = trunk_basis

        # Trunk input dim depends on basis type
        if trunk_basis == 'fourier':
            trunk_input_dim = 1 + 2 * n_freq                    # 65
        elif trunk_basis == 'mixed':
            trunk_input_dim = 1 + 2 * n_freq + n_rbf + 3        # 65 + 16 + 3 = 84
            # Register RBF centers as buffer (fixed, not learned)
            self.register_buffer('rbf_centers',
                                 torch.linspace(0, 1, n_rbf))   # [n_rbf]
        else:
            trunk_input_dim = 1 + 2 * n_freq

        # Trunk param shapes
        trunk_param_count = 0
        trunk_param_shapes = []
        trunk_param_count += trunk_input_dim * width + width
        trunk_param_shapes.append((trunk_input_dim, width, width))
        for _ in range(2, trunk_depth):
            trunk_param_count += width * width + width
            trunk_param_shapes.append((width, width, width))

        self.trunk_param_shapes = trunk_param_shapes
        self.trunk_param_count = trunk_param_count

        forecast_output_dim = trunk_param_count + width

        # Encoder: pluggable via encoder_type
        self.encoder = build_encoder(encoder_type, branch_dim, branch_hidden,
                                     seq_len=seq_len, depth=branch_depth,
                                     activation=activation, dropout=dropout)

        # Forecast head: z → trunk_params + B (for future prediction)
        self.forecast_head = nn.Linear(branch_hidden, forecast_output_dim)
        nn.init.xavier_normal_(self.forecast_head.weight)
        nn.init.constant_(self.forecast_head.bias, 0)

        # Reconstruction head: z → trunk_params + B (for imputation/anomaly)
        self.recon_head = nn.Linear(branch_hidden, forecast_output_dim)
        nn.init.xavier_normal_(self.recon_head.weight)
        nn.init.constant_(self.recon_head.bias, 0)

        self.bias = nn.Parameter(torch.zeros([1]))
        self.recon_bias = nn.Parameter(torch.zeros([1]))

    def _get_trunk_features(self, t):
        """Build trunk input features from query points t.

        fourier: [t, sin(2πkt), cos(2πkt)] — global periodic
        mixed:   fourier + RBF (local bumps) + polynomial (trend)
        """
        freqs = torch.arange(1, self.n_freq + 1, dtype=t.dtype, device=t.device)
        sin_f = torch.sin(2 * math.pi * freqs.unsqueeze(0) * t)
        cos_f = torch.cos(2 * math.pi * freqs.unsqueeze(0) * t)
        fourier = torch.cat([t, sin_f, cos_f], dim=-1)  # [n_points, 1+2*n_freq]

        if self.trunk_basis == 'mixed':
            # RBF: exp(-γ * (t - c)²) for local patterns
            centers = self.rbf_centers.unsqueeze(0)  # [1, n_rbf]
            rbf = torch.exp(-50.0 * (t - centers) ** 2)  # [n_points, n_rbf]

            # Polynomial: t, t², t³ for trends
            poly = torch.cat([t, t ** 2, t ** 3], dim=-1)  # [n_points, 3]

            return torch.cat([fourier, rbf, poly], dim=-1)

        return fourier

    def get_representation(self, branch_input):
        
        """Return task-agnostic representation z."""
        return self.encoder(branch_input)

    def forward(self, branch_input, base, target_pred_len):
        
        batch_size = branch_input.shape[0]

        z = self.encoder(branch_input)
        branch_output = self.forecast_head(z)
        trunk_params = branch_output[:, :self.trunk_param_count] * 0.01
        B = branch_output[:, self.trunk_param_count:] # 여기서는  trunk param이 왜 있는거야? trunk를 이렇게 구해?

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
                                  dtype=branch_input.dtype,
                                  device=branch_input.device).unsqueeze(-1)
        t_fourier = self._get_trunk_features(t_output)
        Phi = t_fourier.unsqueeze(0).expand(batch_size, -1, -1)

        act_fn = F.gelu if self.activation == 'gelu' else F.relu
        for i, (w, b) in enumerate(trunk_weights):
            Phi = torch.bmm(Phi, w) + b.unsqueeze(1)
            if i < len(trunk_weights) - 1:
                Phi = act_fn(Phi)

        deeponet_out = torch.einsum('bp,bqp->bq', B, Phi)
        return base + deeponet_out + self.bias

    def reconstruct(self, branch_input, base, target_len):
        # import pdb ; pdb.set_trace()
        """Reconstruction via recon_head. Same Trunk mechanism, separate weights."""
        batch_size = branch_input.shape[0]

        z = self.encoder(branch_input)
        branch_output = self.recon_head(z)
        trunk_params = branch_output[:, :self.trunk_param_count] * 0.01
        B = branch_output[:, self.trunk_param_count:]

        trunk_weights = []
        idx = 0
        for in_dim, out_dim, bias_size in self.trunk_param_shapes:
            w_size = in_dim * out_dim
            w = trunk_params[:, idx:idx+w_size].view(batch_size, in_dim, out_dim)
            idx += w_size
            b = trunk_params[:, idx:idx+bias_size].view(batch_size, out_dim)
            idx += bias_size
            trunk_weights.append((w, b))

        # Query points cover the input range [0, 1] for reconstruction
        t_output = torch.linspace(0, 1, target_len,
                                  dtype=branch_input.dtype,
                                  device=branch_input.device).unsqueeze(-1)
        t_fourier = self._get_trunk_features(t_output)
        Phi = t_fourier.unsqueeze(0).expand(batch_size, -1, -1)

        act_fn = F.gelu if self.activation == 'gelu' else F.relu
        for i, (w, b) in enumerate(trunk_weights):
            Phi = torch.bmm(Phi, w) + b.unsqueeze(1)
            if i < len(trunk_weights) - 1:
                Phi = act_fn(Phi)

        deeponet_out = torch.einsum('bp,bqp->bq', B, Phi)
        return base + deeponet_out + self.recon_bias

    def _load_legacy_branch_net(self, branch_net_state):

        """Load old-style branch_net weights into encoder + forecast_head.

        Old key pattern: branch_net.net.{0,3,6}.weight → encoder.net.{0,3,6}
                         branch_net.net.9.weight → forecast_head.weight
        """
        encoder_state = {}
        head_state = {}

        # Find the last layer index
        layer_indices = sorted(set(
            int(k.split('.')[0]) for k in branch_net_state.keys()
            if k.split('.')[0].isdigit()
        ))
        last_idx = max(layer_indices)

        for k, v in branch_net_state.items():
            parts = k.split('.')
            layer_idx = int(parts[0])
            param_name = parts[1]  # 'weight' or 'bias'

            if layer_idx == last_idx:
                head_state[param_name] = v
            else:
                encoder_state[f'net.{layer_idx}.{param_name}'] = v

        self.encoder.load_state_dict(encoder_state, strict=False)
        self.forecast_head.load_state_dict(head_state, strict=False)


class Model(nn.Module):
    """
    MoE-DeepONet: Multiple expert DeepONets with learned routing.

    Each expert has: Encoder (shared representation) → forecast_head → Trunk → output.
    Router examines the input and produces soft weights over experts.
    Final output = weighted sum of expert outputs.

    For downstream tasks, use get_representation() to extract encoder features
    without going through the forecast_head / trunk.
    """
    def __init__(self, configs):
        super().__init__()
        
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.use_norm = getattr(configs, 'use_norm', True)
        self.width = getattr(configs, 'deeponet_width', 64)
        self.n_experts = getattr(configs, 'n_experts', 4)
        self.branch_depth = getattr(configs, 'branch_depth', 4)
        self.trunk_depth = getattr(configs, 'trunk_depth', 2)
        self.activation = getattr(configs, 'activation', 'gelu')
        self.dropout = getattr(configs, 'dropout', 0.0)
        self.spectral_branch = getattr(configs, 'spectral_branch', False)
        self.encoder_type = getattr(configs, 'encoder_type', 'fnn')
        self.use_skip = getattr(configs, 'skip_mode', 'default') != 'none'
        self.use_cross = getattr(configs, 'use_cross_channel', True)
        self.trunk_basis = getattr(configs, 'trunk_basis', 'fourier')
        self.n_freq = 32
        self.n_rbf = 16

        _bh = getattr(configs, 'branch_hidden', -1)
        self.branch_hidden = _bh if _bh > 0 else self.width * 4

        # Linear skip (optional)
        if self.use_skip:
            self.linear_skip = nn.Linear(self.seq_len, self.pred_len)

        # Branch input dimension
        branch_dim = self.seq_len
        if self.use_cross:
            branch_dim += self.seq_len  # x_cross
        if self.spectral_branch:
            n_fft_feats = (self.seq_len // 2 + 1) * 2
            branch_dim += n_fft_feats
        self.branch_dim = branch_dim

        # Create N experts
        self.experts = nn.ModuleList([
            SingleExpert(
                branch_dim=branch_dim,
                width=self.width,
                branch_depth=self.branch_depth,
                trunk_depth=self.trunk_depth,
                branch_hidden=self.branch_hidden,
                n_freq=self.n_freq,
                activation=self.activation,
                dropout=self.dropout,
                encoder_type=self.encoder_type,
                seq_len=self.seq_len,
                trunk_basis=self.trunk_basis,
                n_rbf=self.n_rbf,
            )
            for _ in range(self.n_experts)
        ])

        # Router: input → soft weights over experts
        self.router = FNN(
            input_dim=branch_dim,
            output_dim=self.n_experts,
            depth=3,
            width=256,
            activation='gelu',
            dropout=0.0,
        )

        # Variance head for Gaussian NLL loss
        self.output_variance = getattr(configs, 'loss', 'MSE') == 'gaussian_nll'
        if self.output_variance:
            self.log_var_head = nn.Sequential(
                nn.Linear(self.seq_len, self.width * 2),
                nn.GELU(),
                nn.Linear(self.width * 2, self.pred_len),
            )

    def _build_branch_input(self, x_ch, x_cross):
        """Build branch input from channel and cross-channel info."""
        if self.use_cross:
            branch_input = torch.cat([x_ch, x_cross], dim=-1)
        else:
            branch_input = x_ch
        if self.spectral_branch:
            x_fft = torch.fft.rfft(x_ch, dim=-1)
            x_spectral = torch.cat([x_fft.real, x_fft.imag], dim=-1)
            branch_input = torch.cat([branch_input, x_spectral], dim=-1)
        return branch_input

    def _get_expert_representations(self, x_ch, x_cross):

        """Get weighted representation from all experts for one channel.

        Returns: [batch, branch_hidden] — router-weighted average of expert encodings.
        """
        branch_input = self._build_branch_input(x_ch, x_cross)

        # Router weights
        router_logits = self.router(branch_input)
        expert_weights = F.softmax(router_logits, dim=-1)  # [batch, n_experts]

        # Weighted sum of expert representations
        z = torch.zeros(x_ch.shape[0], self.branch_hidden,
                        dtype=x_ch.dtype, device=x_ch.device)
        for i, expert in enumerate(self.experts):
            z_expert = expert.get_representation(branch_input)  # [batch, branch_hidden]
            weight = expert_weights[:, i].unsqueeze(-1)
            z = z + weight * z_expert

        return z  # [batch, branch_hidden]

    def get_representation(self, x_enc):
        """Extract task-agnostic representation from the encoder.

        Args:
            x_enc: [batch, seq_len, n_channels]
        Returns:
            z: [batch, n_channels, branch_hidden] — per-channel representation
        """

        batch_size, seq_len, n_channels = x_enc.shape

        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc = x_enc / stdev

        x_cross = x_enc.mean(dim=-1)  # [batch, seq_len]

        representations = []
        for ch in range(n_channels):
            x_ch = x_enc[:, :, ch]
            z_ch = self._get_expert_representations(x_ch, x_cross)  # [batch, branch_hidden]
            representations.append(z_ch)

        z = torch.stack(representations, dim=1)  # [batch, n_channels, branch_hidden]
        return z

    def _forward_single_channel(self, x_ch, x_cross, target_pred_len=None):
        if target_pred_len is None:
            target_pred_len = self.pred_len

        batch_size = x_ch.shape[0]

        # Linear skip (optional)
        if self.use_skip:
            base = self.linear_skip(x_ch)
            if target_pred_len != self.pred_len:
                base = F.interpolate(
                    base.unsqueeze(1), size=target_pred_len,
                    mode='linear', align_corners=True
                ).squeeze(1)
        else:
            base = torch.zeros(batch_size, target_pred_len,
                               dtype=x_ch.dtype, device=x_ch.device)

        # Build branch input
        branch_input = self._build_branch_input(x_ch, x_cross)
        
        # Router: compute expert weights
        router_logits = self.router(branch_input)  # [batch, n_experts]
        expert_weights = F.softmax(router_logits, dim=-1)  # [batch, n_experts]

        # Run each expert and weighted sum
        output = torch.zeros(batch_size, target_pred_len,
                           dtype=x_ch.dtype, device=x_ch.device)

        for i, expert in enumerate(self.experts):
            expert_out = expert(branch_input, base, target_pred_len)  # [batch, pred_len]
            weight = expert_weights[:, i].unsqueeze(-1)  # [batch, 1]
            output = output + weight * expert_out

        return output

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

        # Variance prediction for Gaussian NLL
        log_var = None
        if self.output_variance:
            log_var_list = []
            for ch in range(n_channels):
                x_ch = x_enc[:, :, ch]
                lv = self.log_var_head(x_ch)  # [batch, pred_len]
                lv = torch.clamp(lv, min=-6.0, max=4.0)
                log_var_list.append(lv)
            log_var = torch.stack(log_var_list, dim=-1)  # [batch, pred_len, channels]

        if self.use_norm:
            output = output * stdev + means
            if log_var is not None:
                log_var = log_var + 2 * torch.log(stdev + 1e-5)

        if log_var is not None:
            return output, log_var
        return output

    def reconstruct(self, x_enc):
        """Reconstruct the input sequence using recon_head.

        Used for imputation and anomaly detection.
        Args:
            x_enc: [batch, seq_len, n_channels] (possibly with masked values)
        Returns:
            output: [batch, seq_len, n_channels] — reconstructed full sequence
        """
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

            if self.use_skip:
                base = self.linear_skip(x_ch)
                if seq_len != self.pred_len:
                    base = F.interpolate(
                        base.unsqueeze(1), size=seq_len,
                        mode='linear', align_corners=True
                    ).squeeze(1)
            else:
                base = torch.zeros(batch_size, seq_len,
                                   dtype=x_ch.dtype, device=x_ch.device)

            branch_input = self._build_branch_input(x_ch, x_cross)

            # Router
            router_logits = self.router(branch_input)
            expert_weights = F.softmax(router_logits, dim=-1)

            out_ch = torch.zeros(batch_size, seq_len,
                                 dtype=x_ch.dtype, device=x_ch.device)
            for i, expert in enumerate(self.experts):
                expert_out = expert.reconstruct(branch_input, base, seq_len)
                weight = expert_weights[:, i].unsqueeze(-1)
                out_ch = out_ch + weight * expert_out

            outputs.append(out_ch)

        output = torch.stack(outputs, dim=-1)

        if self.use_norm:
            output = output * stdev + means

        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                target_pred_len=None, query_points=None):
        
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec,
                                target_pred_len, query_points)
        if isinstance(dec_out, tuple):
            mean, log_var = dec_out
            return mean[:, -mean.shape[1]:, :], log_var[:, -log_var.shape[1]:, :]
        return dec_out[:, -dec_out.shape[1]:, :]

    def load_state_dict(self, state_dict, strict=True):

        """Override to handle legacy branch_net → encoder + forecast_head mapping."""
        new_state = {}
        for k, v in state_dict.items():
            if '.branch_net.net.' in k:
                # e.g., experts.0.branch_net.net.9.weight
                parts = k.split('.')
                expert_prefix = '.'.join(parts[:2])  # experts.0
                layer_idx = int(parts[4])  # 0, 3, 6, 9
                param_type = parts[5]  # weight or bias

                # Find the last layer index for this expert
                last_idx = max(
                    int(k2.split('.')[4])
                    for k2 in state_dict.keys()
                    if k2.startswith(f'{expert_prefix}.branch_net.net.')
                    and k2.split('.')[4].isdigit()
                )

                if layer_idx == last_idx:
                    # Last layer → forecast_head
                    new_key = f'{expert_prefix}.forecast_head.{param_type}'
                else:
                    # Other layers → encoder
                    new_key = f'{expert_prefix}.encoder.net.{layer_idx}.{param_type}'
                new_state[new_key] = v
            else:
                new_state[k] = v

        # Filter out shape mismatches when strict=False
        if not strict:
            model_state = super().state_dict()
            filtered = {}
            for k, v in new_state.items():
                if k in model_state and v.shape == model_state[k].shape:
                    filtered[k] = v
            return super().load_state_dict(filtered, strict=False)

        return super().load_state_dict(new_state, strict=True)
