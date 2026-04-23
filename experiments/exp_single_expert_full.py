"""
Single Expert (No MoE) + PatchAttn + Cross-channel Attention + Hyper Trunk + Spectral Branch
Time Series Pile pretrain + masked_recon + FC loss + 전체 eval

사용법:
  CUDA_VISIBLE_DEVICES=1 python experiments/exp_single_expert_full.py 2>&1 | tee log/single_expert_full.log
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import optim
from torch.utils.data import DataLoader, random_split
from types import SimpleNamespace
import time

from model.encoders import build_encoder
from data_provider.pile_dataset import PilePretrainDataset
from experiments.eval_all_tasks import eval_forecasting, eval_imputation, eval_classification, eval_short_term, print_summary


class SingleExpertModel(nn.Module):
    """
    Single Expert (No MoE, No Router)
    + PatchAttn Encoder
    + Cross-channel Attention
    + Hyper Trunk
    + Spectral Branch

    파라미터를 MoE 4개에 분산하지 않고 하나의 큰 expert에 집중.
    """
    def __init__(self, seq_len=96, pred_len=96, width=128, hidden=512,
                 trunk_depth=2, n_freq=32, n_rbf=16):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.width = width
        self.hidden = hidden
        self.branch_hidden = hidden  # for eval compatibility
        self.n_freq = n_freq
        self.n_rbf = n_rbf

        # Branch input: x_ch + spectral (FFT)
        n_fft = (seq_len // 2 + 1) * 2
        branch_dim = seq_len + n_fft  # 96 + 98 = 194
        self.branch_dim = branch_dim

        # Encoder (bigger since no MoE split)
        self.encoder = build_encoder('patch_attn', branch_dim, hidden,
                                     seq_len=seq_len, depth=4,
                                     activation='gelu', dropout=0.1)

        # Cross-channel Attention
        self.cross_attn = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=8, dim_feedforward=hidden * 4,
            dropout=0.1, activation='gelu', batch_first=True, norm_first=True
        )

        # Trunk input dim (mixed basis)
        trunk_input_dim = 1 + 2 * n_freq + n_rbf + 3  # 84
        self.register_buffer('rbf_centers', torch.linspace(0, 1, n_rbf))

        # Hyper Trunk: head generates trunk MLP weights
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

        # Task heads
        self.forecast_head = nn.Linear(hidden, forecast_output_dim)
        self.recon_head = nn.Linear(hidden, forecast_output_dim)
        self.bias = nn.Parameter(torch.zeros(1))
        self.recon_bias = nn.Parameter(torch.zeros(1))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _get_trunk_features(self, t):
        freqs = torch.arange(1, self.n_freq + 1, dtype=t.dtype, device=t.device)
        sin_f = torch.sin(2 * math.pi * freqs.unsqueeze(0) * t)
        cos_f = torch.cos(2 * math.pi * freqs.unsqueeze(0) * t)
        fourier = torch.cat([t, sin_f, cos_f], dim=-1)
        rbf = torch.exp(-50.0 * (t - self.rbf_centers.unsqueeze(0)) ** 2)
        poly = torch.cat([t, t ** 2, t ** 3], dim=-1)
        return torch.cat([fourier, rbf, poly], dim=-1)

    def _build_branch_input(self, x_ch):
        """x_ch [B, seq_len] → branch_input [B, branch_dim] with spectral"""
        x_fft = torch.fft.rfft(x_ch, dim=-1)
        x_spectral = torch.cat([x_fft.real, x_fft.imag], dim=-1)
        return torch.cat([x_ch, x_spectral], dim=-1)

    def _trunk_forward(self, trunk_params, B_coeff, target_len, bias):
        """Hyper trunk: generate Phi from trunk_params, compute B·Phi"""
        batch_size = trunk_params.shape[0]
        params = trunk_params * 0.01

        # Build trunk weights
        trunk_weights = []
        idx = 0
        for in_dim, out_dim, bias_size in self.trunk_param_shapes:
            w_size = in_dim * out_dim
            w = params[:, idx:idx+w_size].view(batch_size, in_dim, out_dim)
            idx += w_size
            b = params[:, idx:idx+bias_size].view(batch_size, out_dim)
            idx += bias_size
            trunk_weights.append((w, b))

        # Query points
        t = torch.linspace(0, 1, target_len, dtype=trunk_params.dtype,
                           device=trunk_params.device).unsqueeze(-1)
        t_features = self._get_trunk_features(t)
        Phi = t_features.unsqueeze(0).expand(batch_size, -1, -1)

        for i, (w, b) in enumerate(trunk_weights):
            Phi = torch.bmm(Phi, w) + b.unsqueeze(1)
            if i < len(trunk_weights) - 1:
                Phi = F.gelu(Phi)

        output = torch.einsum('bp,bqp->bq', B_coeff, Phi)
        return output + bias

    def _encode_all_channels(self, x_enc):
        """Per-channel encode + cross-channel attention."""
        B, S, C = x_enc.shape

        # RevIN
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev

        # Per-channel encoder
        z_list = []
        for ch in range(C):
            x_ch = x_enc[:, :, ch]
            branch_input = self._build_branch_input(x_ch)
            z_ch = self.encoder(branch_input)  # [B, hidden]
            z_list.append(z_ch)
        z_all = torch.stack(z_list, dim=1)  # [B, C, hidden]

        # Cross-channel attention
        if C > 1:
            z_all = self.cross_attn(z_all)

        return z_all, means, stdev

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None,
                target_pred_len=None, **kwargs):
        if target_pred_len is None:
            target_pred_len = self.pred_len
        B, S, C = x_enc.shape

        z_all, means, stdev = self._encode_all_channels(x_enc)

        outputs = []
        for ch in range(C):
            head_out = self.forecast_head(z_all[:, ch, :])
            trunk_params = head_out[:, :self.trunk_param_count]
            B_coeff = head_out[:, self.trunk_param_count:]
            out_ch = self._trunk_forward(trunk_params, B_coeff, target_pred_len, self.bias)
            outputs.append(out_ch)

        output = torch.stack(outputs, dim=-1)
        output = output * stdev + means
        return output

    def reconstruct(self, x_enc):
        B, S, C = x_enc.shape
        z_all, means, stdev = self._encode_all_channels(x_enc)

        outputs = []
        for ch in range(C):
            head_out = self.recon_head(z_all[:, ch, :])
            trunk_params = head_out[:, :self.trunk_param_count]
            B_coeff = head_out[:, self.trunk_param_count:]
            out_ch = self._trunk_forward(trunk_params, B_coeff, S, self.recon_bias)
            outputs.append(out_ch)

        output = torch.stack(outputs, dim=-1)
        output = output * stdev + means
        return output

    def get_representation(self, x_enc):
        z_all, _, _ = self._encode_all_channels(x_enc)
        return z_all  # [B, C, hidden]


def pretrain(model, device, save_path, epochs=20, lr=0.0003, mask_rate=0.4):
    print(f'\n{"="*60}')
    print('Pre-training: Single Expert + Cross-channel + Spectral')
    print(f'{"="*60}')

    dataset = PilePretrainDataset(seq_len=96, stride=48,
                                   pile_root='./dataset/time_series_pile')
    n_val = min(10000, len(dataset) // 10)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])
    train_dl = DataLoader(train_ds, batch_size=128, shuffle=True,
                          num_workers=4, drop_last=True, pin_memory=True)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    recon_criterion = nn.MSELoss(reduction='none')

    print(f'Train: {n_train}, Steps/epoch: {len(train_dl)}')

    best_val = float('inf')
    for epoch in range(epochs):
        model.train()
        losses = []
        t0 = time.time()

        for i, batch_x in enumerate(train_dl):
            batch_x = batch_x.float().to(device)
            B, S, C = batch_x.shape
            optimizer.zero_grad()

            # 1) Masked recon
            mask = (torch.rand_like(batch_x) > mask_rate).float()
            recon_out = model.reconstruct(batch_x * mask)
            loss_mat = recon_criterion(recon_out, batch_x)
            inv_mask = 1.0 - mask
            recon_loss = (loss_mat * inv_mask).sum() / inv_mask.sum().clamp(min=1)

            # 2) Forecasting loss
            half = S // 2
            x_input = batch_x[:, :half, :]
            x_target = batch_x[:, half:, :]
            x_padded = F.pad(x_input, (0, 0, 0, S - half))
            fc_out = model(x_padded, target_pred_len=half)
            fc_loss = F.mse_loss(fc_out, x_target)

            loss = recon_loss + fc_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())

            if (i + 1) % 500 == 0:
                print(f'  iter {i+1}/{len(train_dl)}: recon={recon_loss.item():.4f} fc={fc_loss.item():.4f}')

        scheduler.step()
        train_loss = np.mean(losses)
        print(f'Epoch {epoch+1}/{epochs}: loss={train_loss:.4f} lr={scheduler.get_last_lr()[0]:.6f} ({time.time()-t0:.0f}s)')

        if train_loss < best_val:
            best_val = train_loss
            torch.save(model.state_dict(), save_path)
            print(f'  Saved checkpoint')

    model.load_state_dict(torch.load(save_path))
    return model


if __name__ == '__main__':
    device = torch.device('cuda')

    model = SingleExpertModel(
        seq_len=96, pred_len=96,
        width=128, hidden=512,
        trunk_depth=2,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f'Single Expert Model: {n_params:,} params')
    print(f'  PatchAttn Encoder + Cross-channel Attn + Hyper Trunk + Spectral Branch')

    save_path = 'checkpoints/single_expert_full.pth'
    os.makedirs('checkpoints', exist_ok=True)

    model = pretrain(model, device, save_path, epochs=20, lr=0.0003)

    fc = eval_forecasting(model, device)
    st = eval_short_term(model, device)
    imp = eval_imputation(model, device)
    cls = eval_classification(model, device)
    print_summary(fc, imp, cls, st, 'Single Expert Full')
