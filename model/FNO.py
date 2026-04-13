"""
FNO (Fourier Neural Operator) for Time Series Forecasting
- Channel-independent processing: works with any number of channels
- Residual Linear skip connection for stable optimization
- Spectral convolution in frequency domain per FNO layer
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv1d(nn.Module):
    """1D Spectral Convolution: FFT → learned complex weights → IFFT"""

    def __init__(self, in_channels, out_channels, modes):
        super().__init__()
        self.modes = modes
        scale = 1 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes, dtype=torch.cfloat)
        )

    def forward(self, x):
        # x: [batch, in_channels, length]
        length = x.shape[-1]
        x_ft = torch.fft.rfft(x, dim=-1)  # [batch, in_channels, length//2+1]

        out_ft = torch.zeros(x.shape[0], self.weights.shape[1], length // 2 + 1,
                             dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes] = torch.einsum(
            'bim,iom->bom', x_ft[:, :, :self.modes], self.weights
        )
        return torch.fft.irfft(out_ft, n=length, dim=-1)  # [batch, out_channels, length]


class FNOLayer(nn.Module):
    """Single FNO layer: SpectralConv + pointwise Conv residual + LayerNorm + GELU"""

    def __init__(self, width, modes):
        super().__init__()
        self.spectral = SpectralConv1d(width, width, modes)
        self.w = nn.Conv1d(width, width, 1)   # pointwise residual (local path)
        self.norm = nn.LayerNorm(width)

    def forward(self, x):
        # x: [batch, width, length]
        x1 = self.spectral(x)          # global frequency path
        x2 = self.w(x)                 # local pointwise path
        x = x1 + x2
        x = x.transpose(-1, -2)        # [batch, length, width]
        x = self.norm(x)
        x = F.gelu(x)
        return x.transpose(-1, -2)     # [batch, width, length]


class Model(nn.Module):
    """
    Channel-Independent FNO for Time Series Forecasting
    + Residual Linear skip connection for stable optimization

    Architecture per channel:
      Linear(x_channel) → base prediction [batch, pred_len]   (residual)
      Lift(x_channel)   → [batch, width, seq_len]
      N × FNOLayer      → [batch, width, seq_len]
      Project           → [batch, pred_len]
      Output = base + FNO_out
    """

    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.use_norm = getattr(configs, 'use_norm', True)

        # reuse --deeponet_width for FNO hidden width
        # reuse --e_layers for number of FNO layers
        # reuse --fno_modes (added to run.py) for number of Fourier modes
        self.width = getattr(configs, 'deeponet_width', 64)
        n_layers = getattr(configs, 'e_layers', 4)
        default_modes = max(4, self.seq_len // 8)  # 12 for seq_len=96
        fno_modes_cfg = getattr(configs, 'fno_modes', -1)
        self.modes = min(fno_modes_cfg if fno_modes_cfg > 0 else default_modes, self.seq_len // 2)
        dropout = getattr(configs, 'dropout', 0.0)

        # Residual linear skip
        self.linear_skip = nn.Linear(self.seq_len, self.pred_len)

        # Lifting: scalar → width
        self.lift = nn.Linear(1, self.width)

        # FNO layers
        self.fno_layers = nn.ModuleList(
            [FNOLayer(self.width, self.modes) for _ in range(n_layers)]
        )

        # Projection: [batch, width, seq_len] → [batch, pred_len]
        self.proj_channel = nn.Conv1d(self.width, 1, 1)
        self.proj_time = nn.Linear(self.seq_len, self.pred_len)

        self.dropout = nn.Dropout(dropout)

    def _forward_single_channel(self, x_ch):
        """
        x_ch: [batch, seq_len]
        Returns: [batch, pred_len]
        """
        # Residual skip
        base = self.linear_skip(x_ch)          # [batch, pred_len]

        # Lift to hidden dim
        x = x_ch.unsqueeze(-1)                 # [batch, seq_len, 1]
        x = self.lift(x)                       # [batch, seq_len, width]
        x = x.transpose(-1, -2)                # [batch, width, seq_len]

        # FNO layers
        for layer in self.fno_layers:
            x = self.dropout(layer(x))

        # Project back to pred_len
        x = self.proj_channel(x)               # [batch, 1, seq_len]
        x = x.squeeze(1)                       # [batch, seq_len]
        fno_out = self.proj_time(x)            # [batch, pred_len]

        return base + fno_out

    def forecast(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        """
        x_enc: [batch, seq_len, channels]
        Returns: [batch, pred_len, channels]
        """
        batch_size, seq_len, n_channels = x_enc.shape

        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc = x_enc / stdev

        outputs = []
        for ch in range(n_channels):
            x_ch = x_enc[:, :, ch]
            out_ch = self._forward_single_channel(x_ch)
            outputs.append(out_ch)

        output = torch.stack(outputs, dim=-1)  # [batch, pred_len, channels]

        if self.use_norm:
            output = output * stdev
            output = output + means

        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]
