"""
MoE-DeepONet v2: Hard Routing + Load Balancing + Diversity Loss
- Top-1 hard routing: each input goes to ONE expert (forces specialization)
- Load balancing loss: all experts used equally (~25% each)
- Diversity loss: experts penalized for producing similar outputs
- Different architecture per expert for natural specialization
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class FNN(nn.Module):
    def __init__(self, input_dim, output_dim, depth, width, activation='gelu', dropout=0.0):
        super().__init__()
        layers = []
        layers.append(nn.Linear(input_dim, width))
        layers.append(nn.GELU() if activation == 'gelu' else nn.ReLU())
        if dropout > 0: layers.append(nn.Dropout(dropout))
        for _ in range(depth - 2):
            layers.append(nn.Linear(width, width))
            layers.append(nn.GELU() if activation == 'gelu' else nn.ReLU())
            if dropout > 0: layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(width, output_dim))
        self.net = nn.Sequential(*layers)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.net(x)


class SingleExpert(nn.Module):
    def __init__(self, branch_dim, width, branch_depth, trunk_depth,
                 branch_hidden, n_freq, activation, dropout,
                 use_spectral_branch=False, seq_len=96):
        super().__init__()
        self.width = width
        self.n_freq = n_freq
        self.activation = activation
        self.use_spectral_branch = use_spectral_branch
        self.seq_len = seq_len

        # Adjust branch_dim if this expert uses spectral_branch
        expert_branch_dim = branch_dim
        if use_spectral_branch:
            n_fft = (seq_len // 2 + 1) * 2
            expert_branch_dim += n_fft

        trunk_input_dim = 1 + 2 * n_freq
        trunk_param_count = 0
        trunk_param_shapes = []
        trunk_param_count += trunk_input_dim * width + width
        trunk_param_shapes.append((trunk_input_dim, width, width))
        for _ in range(2, trunk_depth):
            trunk_param_count += width * width + width
            trunk_param_shapes.append((width, width, width))

        self.trunk_param_shapes = trunk_param_shapes
        self.trunk_param_count = trunk_param_count
        branch_output_dim = trunk_param_count + width

        self.branch_net = FNN(expert_branch_dim, branch_output_dim, branch_depth,
                             branch_hidden, activation, dropout)
        self.bias = nn.Parameter(torch.zeros([1]))

    def _get_fourier_features(self, t):
        freqs = torch.arange(1, self.n_freq + 1, dtype=t.dtype, device=t.device)
        return torch.cat([t, torch.sin(2*math.pi*freqs.unsqueeze(0)*t),
                         torch.cos(2*math.pi*freqs.unsqueeze(0)*t)], dim=-1)

    def forward(self, x_ch, x_cross, base, target_pred_len):
        batch_size = x_ch.shape[0]

        # Build expert-specific branch input
        branch_input = torch.cat([x_ch, x_cross], dim=-1)
        if self.use_spectral_branch:
            x_fft = torch.fft.rfft(x_ch, dim=-1)
            x_spectral = torch.cat([x_fft.real, x_fft.imag], dim=-1)
            branch_input = torch.cat([branch_input, x_spectral], dim=-1)

        branch_output = self.branch_net(branch_input)
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


class Model(nn.Module):
    """
    MoE-DeepONet v2: Top-1 Hard Routing + Load Balancing + Diversity

    Key differences from v1:
    1. Hard routing: each input → 1 expert only (not soft weighted sum)
    2. Load balancing loss: forces equal expert usage
    3. Each expert has different architecture (spectral/plain/deep)
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
        self.n_freq = 32
        self.training_mode = True  # for load balancing

        _bh = getattr(configs, 'branch_hidden', -1)
        branch_hidden = _bh if _bh > 0 else self.width * 4

        self.linear_skip = nn.Linear(self.seq_len, self.pred_len)

        # Base branch dim (without spectral)
        cross_dim = self.seq_len
        base_branch_dim = self.seq_len + cross_dim

        # Create experts with DIFFERENT architectures
        expert_configs = [
            {'use_spectral_branch': True, 'trunk_depth': 2},   # Expert 0: spectral
            {'use_spectral_branch': False, 'trunk_depth': 2},   # Expert 1: plain
            {'use_spectral_branch': True, 'trunk_depth': 3},    # Expert 2: spectral + deeper trunk
            {'use_spectral_branch': False, 'trunk_depth': 3},   # Expert 3: plain + deeper trunk
        ]

        self.experts = nn.ModuleList([
            SingleExpert(
                branch_dim=base_branch_dim,
                width=self.width,
                branch_depth=self.branch_depth,
                trunk_depth=ec['trunk_depth'],
                branch_hidden=branch_hidden,
                n_freq=self.n_freq,
                activation=self.activation,
                dropout=self.dropout,
                use_spectral_branch=ec['use_spectral_branch'],
                seq_len=self.seq_len,
            )
            for ec in expert_configs[:self.n_experts]
        ])

        # Router: input → logits over experts
        # Uses base branch_dim (spectral added inside each expert)
        self.router = FNN(base_branch_dim, self.n_experts, 3, 256, 'gelu', 0.0)

        # Load balancing tracking
        self.register_buffer('expert_counts', torch.zeros(self.n_experts))
        self.load_balance_weight = 0.01
        self.diversity_weight = 0.01

    def _get_auxiliary_loss(self, router_logits):
        """Load balancing loss: penalize uneven expert usage."""
        if not self.training:
            return torch.tensor(0.0, device=router_logits.device)

        # Fraction of tokens routed to each expert
        probs = F.softmax(router_logits, dim=-1)  # [batch, n_experts]
        avg_probs = probs.mean(dim=0)  # [n_experts]

        # Top-1 assignment fractions
        assignments = torch.argmax(router_logits, dim=-1)  # [batch]
        freq = torch.zeros(self.n_experts, device=router_logits.device)
        for i in range(self.n_experts):
            freq[i] = (assignments == i).float().mean()

        # Load balancing: want freq ≈ 1/n_experts for all experts
        target = 1.0 / self.n_experts
        load_balance_loss = self.n_experts * (freq * avg_probs).sum()

        return self.load_balance_weight * load_balance_loss

    def _forward_single_channel(self, x_ch, x_cross, target_pred_len=None):
        if target_pred_len is None:
            target_pred_len = self.pred_len

        batch_size = x_ch.shape[0]

        # Shared linear skip
        base = self.linear_skip(x_ch)
        if target_pred_len != self.pred_len:
            base = F.interpolate(base.unsqueeze(1), size=target_pred_len,
                               mode='linear', align_corners=True).squeeze(1)

        # Router input (base, without spectral — each expert adds its own)
        router_input = torch.cat([x_ch, x_cross], dim=-1)
        router_logits = self.router(router_input)  # [batch, n_experts]

        if self.training:
            # TOP-1 hard routing with straight-through gradient
            # Hard forward, soft backward (STE)
            soft_weights = F.softmax(router_logits, dim=-1)
            top1_idx = torch.argmax(router_logits, dim=-1)  # [batch]
            hard_weights = F.one_hot(top1_idx, self.n_experts).float()  # [batch, n_experts]
            # Straight-through estimator
            weights = hard_weights - soft_weights.detach() + soft_weights
        else:
            # At inference: top-1 hard routing
            top1_idx = torch.argmax(router_logits, dim=-1)
            weights = F.one_hot(top1_idx, self.n_experts).float()

        # Run selected experts
        output = torch.zeros(batch_size, target_pred_len,
                           dtype=x_ch.dtype, device=x_ch.device)

        # Collect expert outputs for diversity loss
        expert_outputs = []
        for i, expert in enumerate(self.experts):
            mask = weights[:, i]  # [batch] — 0 or 1
            if mask.sum() == 0 and not self.training:
                continue
            expert_out = expert(x_ch, x_cross, base, target_pred_len)
            expert_outputs.append(expert_out)
            output = output + mask.unsqueeze(-1) * expert_out

        # Store for auxiliary losses
        self._last_router_logits = router_logits
        self._last_expert_outputs = expert_outputs

        return output

    def get_auxiliary_loss(self):
        """Call after forward to get load balancing + diversity loss."""
        loss = torch.tensor(0.0, device=next(self.parameters()).device)

        if hasattr(self, '_last_router_logits'):
            loss = loss + self._get_auxiliary_loss(self._last_router_logits)

        # Diversity loss: penalize experts for similar outputs
        if hasattr(self, '_last_expert_outputs') and len(self._last_expert_outputs) > 1:
            outputs = torch.stack(self._last_expert_outputs, dim=0)  # [n_active, batch, pred]
            # Variance across experts (want high variance = diverse)
            var = outputs.var(dim=0).mean()
            diversity_loss = -self.diversity_weight * var  # negative because we maximize variance
            loss = loss + diversity_loss

        return loss

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
