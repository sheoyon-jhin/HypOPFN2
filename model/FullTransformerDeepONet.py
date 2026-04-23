"""
Full Transformer DeepONet (Branch + Trunk both Transformer)

Branch: Transformer encoder processes input time series
        → produces latent representation Z

Trunk:  Transformer decoder takes time query points t
        + cross-attends to Branch's Z
        → produces basis values at each query point

Output: Linear combination of Trunk outputs, weighted by Branch coefficients

This is still Operator Learning:
  - Input function u(x) → Branch encoder → Z
  - Output coordinates t → Trunk decoder + cross-attention to Z → G(u)(t)
  - "Function in, function out" = Neural Operator
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.use_norm = getattr(configs, 'use_norm', True)
        self.d_model = getattr(configs, 'd_model', 128)
        self.n_heads = getattr(configs, 'n_heads', 4)
        self.n_enc_layers = getattr(configs, 'e_layers', 3)
        self.n_dec_layers = getattr(configs, 'd_layers', 2)
        self.width = getattr(configs, 'deeponet_width', 64)
        self.dropout = getattr(configs, 'dropout', 0.1)
        self.use_spectral = getattr(configs, 'spectral_branch', False)

        # ============================================================
        # Branch: Transformer Encoder (processes input time series)
        # ============================================================
        # Input per token: value(1) + cross_mean(1) [+ spectral(2)]
        branch_token_dim = 2
        if self.use_spectral:
            branch_token_dim += 2

        self.branch_input_proj = nn.Linear(branch_token_dim, self.d_model)
        self.branch_pos = nn.Parameter(torch.randn(1, self.seq_len, self.d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, nhead=self.n_heads,
            dim_feedforward=self.d_model * 4, dropout=self.dropout,
            activation='gelu', batch_first=True,
        )
        self.branch_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.n_enc_layers)

        # Branch output: coefficients B (width) from pooled encoder output
        self.branch_to_B = nn.Linear(self.d_model, self.width)

        # ============================================================
        # Trunk: Transformer Decoder (processes output time points)
        # Cross-attends to Branch encoder output!
        # ============================================================
        # Trunk input: Fourier features of query time t
        self.n_freq = 32
        trunk_token_dim = 1 + 2 * self.n_freq  # t + sin/cos features

        self.trunk_input_proj = nn.Linear(trunk_token_dim, self.d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.d_model, nhead=self.n_heads,
            dim_feedforward=self.d_model * 4, dropout=self.dropout,
            activation='gelu', batch_first=True,
        )
        self.trunk_decoder = nn.TransformerDecoder(decoder_layer, num_layers=self.n_dec_layers)

        # Trunk output: basis value (width) at each query point
        self.trunk_to_basis = nn.Linear(self.d_model, self.width)

        # Linear skip (shared)
        self.linear_skip = nn.Linear(self.seq_len, self.pred_len)

        # Bias
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

        # ============================================================
        # Branch Encoder: process input time series
        # ============================================================
        tokens = torch.stack([x_ch, x_cross], dim=-1)  # [batch, seq_len, 2]

        if self.use_spectral:
            x_fft = torch.fft.rfft(x_ch, dim=-1)
            dom_freq = x_fft.abs()[:, 1:].argmax(dim=-1) + 1
            t_pos = torch.arange(self.seq_len, device=x_ch.device).float().unsqueeze(0)
            phase = 2 * math.pi * dom_freq.unsqueeze(1).float() * t_pos / self.seq_len
            spec = torch.stack([torch.sin(phase), torch.cos(phase)], dim=-1)
            tokens = torch.cat([tokens, spec], dim=-1)

        branch_in = self.branch_input_proj(tokens) + self.branch_pos[:, :self.seq_len, :]
        encoder_out = self.branch_encoder(branch_in)  # [batch, seq_len, d_model]

        # Branch coefficients B from pooled encoder output
        B = self.branch_to_B(encoder_out.mean(dim=1))  # [batch, width]

        # ============================================================
        # Trunk Decoder: process output time points with cross-attention
        # ============================================================
        t_output = torch.linspace(0, 1, target_pred_len,
                                  dtype=x_ch.dtype, device=x_ch.device).unsqueeze(-1)
        t_fourier = self._get_fourier_features(t_output)  # [pred_len, trunk_token_dim]
        trunk_in = self.trunk_input_proj(t_fourier).unsqueeze(0).expand(batch_size, -1, -1)
        # [batch, pred_len, d_model]

        # Cross-attention: Trunk attends to Branch encoder output!
        # This is the key: Trunk "asks" the input representation
        # what basis functions to produce at each time point
        decoder_out = self.trunk_decoder(
            tgt=trunk_in,           # [batch, pred_len, d_model] — query points
            memory=encoder_out,     # [batch, seq_len, d_model]  — input representation
        )  # [batch, pred_len, d_model]

        # Basis values at each query point
        Phi = self.trunk_to_basis(decoder_out)  # [batch, pred_len, width]

        # DeepONet combination: output(t) = sum_k B_k * Phi_k(t)
        deeponet_out = torch.einsum('bp,bqp->bq', B, Phi)  # [batch, pred_len]

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
