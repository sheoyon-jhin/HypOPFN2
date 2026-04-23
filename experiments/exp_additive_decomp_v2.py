"""
Additive Decomposition v2: Spectral Loss + Frequency Conditioning

v1 대비 변경:
  1. Spectral Loss: MSE(FFT(pred), FFT(target)) 추가 → 주파수 보존 강제
  2. Frequency Conditioning: 입력의 dominant freq를 Fourier trunk query에 주입
     → trunk이 "이 시계열의 주기가 뭔지" 알게 됨

Config: medium, Pile+Synth, 20 epoch

사용법:
  CUDA_VISIBLE_DEVICES=X python experiments/exp_additive_decomp_v2.py 2>&1 | tee log/additive_decomp_v2.log
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch import optim
from torch.utils.data import DataLoader, Dataset, ConcatDataset, random_split
from types import SimpleNamespace
from sklearn.metrics import accuracy_score
import time

from data_provider.pile_dataset import PilePretrainDataset
from data_provider.data_factory import data_provider
from data_provider.data_loader import Dataset_Classification, Dataset_GaussianPCoregionalization

DEVICE = torch.device('cuda')


# ============================================================
# Model Components
# ============================================================
class PatchAttentionEncoder(nn.Module):
    """Shared encoder: branch_input → z (task-agnostic representation)."""
    def __init__(self, input_dim, hidden_dim, seq_len=96, depth=4,
                 activation='gelu', dropout=0.1):
        super().__init__()
        # Patch embedding
        self.patch_size = 16
        n_patches = seq_len // self.patch_size  # 6
        patch_dim = input_dim // seq_len * self.patch_size if input_dim > seq_len else self.patch_size

        # For branch_input which is flattened: just reshape into patches
        self.input_proj = nn.Linear(input_dim // n_patches if input_dim % n_patches == 0
                                    else input_dim, hidden_dim)
        self.use_flat_proj = (input_dim % n_patches != 0)
        if self.use_flat_proj:
            self.flat_proj = nn.Linear(input_dim, hidden_dim)

        self.pos_embed = nn.Parameter(torch.randn(1, n_patches, hidden_dim) * 0.02)
        self.n_patches = n_patches

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=4,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=min(depth, 3))
        self.norm = nn.LayerNorm(hidden_dim)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        B = x.shape[0]
        if self.use_flat_proj:
            z = self.flat_proj(x).unsqueeze(1).expand(-1, self.n_patches, -1)
        else:
            x_patches = x.view(B, self.n_patches, -1)
            z = self.input_proj(x_patches)

        z = z + self.pos_embed
        z = self.transformer(z)
        z = self.norm(z)
        z = z.mean(dim=1)  # [B, hidden]
        return z


def extract_dominant_freqs(x_ch, top_k=5):
    """입력 시계열에서 dominant frequency를 추출.
    Args: x_ch [batch, seq_len]
    Returns: freqs [batch, top_k] — 0~seq_len//2 범위의 주파수 인덱스 (normalized)
    """
    fft_mag = torch.abs(torch.fft.rfft(x_ch, dim=-1))  # [B, seq_len//2+1]
    fft_mag[:, 0] = 0  # DC 제거
    _, top_indices = torch.topk(fft_mag, top_k, dim=-1)  # [B, top_k]
    # Normalize to [0, 0.5] (Nyquist)
    seq_len = x_ch.shape[-1]
    return top_indices.float() / seq_len


class TrunkOperator(nn.Module):
    """Single trunk operator with specific basis type.
    v2: Fourier trunk에 frequency conditioning 추가.
    """
    def __init__(self, width, trunk_depth, trunk_type, n_freq=32, n_rbf=16,
                 n_cond_freq=5):
        super().__init__()
        self.width = width
        self.trunk_type = trunk_type
        self.n_freq = n_freq
        self.n_rbf = n_rbf
        self.n_cond_freq = n_cond_freq  # input-adaptive frequency 개수

        if trunk_type == 'fourier':
            # 고정 Fourier + input-adaptive Fourier
            self.trunk_input_dim = 1 + 2 * n_freq + 2 * n_cond_freq  # +10
        elif trunk_type == 'rbf':
            self.trunk_input_dim = 1 + n_rbf
            self.register_buffer('rbf_centers', torch.linspace(0, 1, n_rbf))
        elif trunk_type == 'polynomial':
            self.trunk_input_dim = 7
        elif trunk_type == 'wavelet':
            self.n_scales = 8
            self.trunk_input_dim = 1 + self.n_scales

        trunk_param_count = 0
        trunk_param_shapes = []
        trunk_param_count += self.trunk_input_dim * width + width
        trunk_param_shapes.append((self.trunk_input_dim, width, width))
        for _ in range(2, trunk_depth):
            trunk_param_count += width * width + width
            trunk_param_shapes.append((width, width, width))

        self.trunk_param_count = trunk_param_count
        self.trunk_param_shapes = trunk_param_shapes
        self.head_output_dim = trunk_param_count + width
        self.bias = nn.Parameter(torch.zeros([1]))

    def get_trunk_features(self, t, dom_freqs=None):
        """Build trunk features. For fourier type, dom_freqs adds input-adaptive basis.
        Args:
            t: [target_len, 1]
            dom_freqs: [batch, n_cond_freq] or None
        Returns:
            features: [target_len, trunk_input_dim] (no batch) or
                      [batch, target_len, trunk_input_dim] (with freq conditioning)
        """
        if self.trunk_type == 'fourier':
            # Fixed Fourier basis (same as v1)
            freqs = torch.arange(1, self.n_freq + 1, dtype=t.dtype, device=t.device)
            sin_f = torch.sin(2 * math.pi * freqs.unsqueeze(0) * t)
            cos_f = torch.cos(2 * math.pi * freqs.unsqueeze(0) * t)
            fixed = torch.cat([t, sin_f, cos_f], dim=-1)  # [T, 1+2*n_freq]

            if dom_freqs is not None:
                # Input-adaptive Fourier: sin/cos at dominant frequencies
                # dom_freqs: [B, n_cond_freq], t: [T, 1]
                B = dom_freqs.shape[0]
                T = t.shape[0]
                # [B, n_cond_freq] → [B, 1, n_cond_freq] * [1, T, 1] → [B, T, n_cond_freq]
                f = dom_freqs.unsqueeze(1)  # [B, 1, K]
                t_exp = t.squeeze(-1).unsqueeze(0).unsqueeze(-1)  # [1, T, 1]
                phase = 2 * math.pi * f * t_exp * 96  # scale to match seq_len
                cond_sin = torch.sin(phase)  # [B, T, K]
                cond_cos = torch.cos(phase)  # [B, T, K]
                cond = torch.cat([cond_sin, cond_cos], dim=-1)  # [B, T, 2K]
                fixed_exp = fixed.unsqueeze(0).expand(B, -1, -1)  # [B, T, fixed_dim]
                return torch.cat([fixed_exp, cond], dim=-1)  # [B, T, full_dim]
            else:
                # No conditioning — pad with zeros
                pad = torch.zeros(t.shape[0], 2 * self.n_cond_freq,
                                  dtype=t.dtype, device=t.device)
                return torch.cat([fixed, pad], dim=-1)

        elif self.trunk_type == 'rbf':
            rbf = torch.exp(-50.0 * (t - self.rbf_centers.unsqueeze(0)) ** 2)
            return torch.cat([t, rbf], dim=-1)
        elif self.trunk_type == 'polynomial':
            return torch.cat([torch.ones_like(t), t, t**2, t**3, t**4, t**5, t**6], dim=-1)
        elif self.trunk_type == 'wavelet':
            scales = torch.logspace(-1, 1, self.n_scales, device=t.device, dtype=t.dtype)
            wavelets = [(1 - ((t-0.5)/s)**2) * torch.exp(-((t-0.5)/s)**2 / 2) for s in scales]
            return torch.cat([t] + wavelets, dim=-1)

    def forward(self, head_output, target_len, dom_freqs=None):
        """dom_freqs: [B, n_cond_freq] — only used by fourier trunk."""
        B = head_output.shape[0]
        trunk_params = head_output[:, :self.trunk_param_count] * 0.01
        B_coeff = head_output[:, self.trunk_param_count:]

        trunk_weights = []
        idx = 0
        for in_dim, out_dim, bias_size in self.trunk_param_shapes:
            w_size = in_dim * out_dim
            w = trunk_params[:, idx:idx+w_size].view(B, in_dim, out_dim)
            idx += w_size
            b = trunk_params[:, idx:idx+bias_size].view(B, out_dim)
            idx += bias_size
            trunk_weights.append((w, b))

        t = torch.linspace(0, 1, target_len, dtype=head_output.dtype,
                           device=head_output.device).unsqueeze(-1)

        # Get features — fourier gets frequency conditioning
        if self.trunk_type == 'fourier' and dom_freqs is not None:
            Phi = self.get_trunk_features(t, dom_freqs)  # [B, T, dim]
        else:
            feat = self.get_trunk_features(t)  # [T, dim]
            Phi = feat.unsqueeze(0).expand(B, -1, -1)

        for i, (w, b) in enumerate(trunk_weights):
            Phi = torch.bmm(Phi, w) + b.unsqueeze(1)
            if i < len(trunk_weights) - 1:
                Phi = F.gelu(Phi)

        return torch.einsum('bp,bqp->bq', B_coeff, Phi) + self.bias


def spectral_loss(pred, target):
    """FFT domain MSE loss — 주파수 보존을 강제."""
    pred_fft = torch.fft.rfft(pred, dim=1)
    target_fft = torch.fft.rfft(target, dim=1)
    # Magnitude loss (amplitude spectrum)
    mag_loss = F.mse_loss(pred_fft.abs(), target_fft.abs())
    # Phase loss (weighted by magnitude — 큰 주파수의 위상이 더 중요)
    weights = target_fft.abs().detach() + 1e-8
    phase_diff = torch.angle(pred_fft) - torch.angle(target_fft)
    phase_loss = (weights * (1 - torch.cos(phase_diff))).mean()
    return mag_loss + 0.1 * phase_loss


class AdditiveDecompModel(nn.Module):
    """
    Additive Decomposition via Diverse Operators.

    Output = Fourier(x) + RBF(x) + Polynomial(x) + Wavelet(x)
    Each component: Shared Encoder → z → Head_i → Trunk_i(t) → component_i
    """
    def __init__(self, seq_len=96, pred_len=96, width=96, branch_hidden=384,
                 trunk_depth=2, spectral_branch=True, n_freq=32, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.use_norm = True
        self.spectral_branch = spectral_branch
        self.branch_hidden = branch_hidden

        # Branch input dim
        branch_dim = seq_len * 2  # x_ch + x_cross
        if spectral_branch:
            branch_dim += (seq_len // 2 + 1) * 2
        self.branch_dim = branch_dim

        # Shared encoder
        self.encoder = PatchAttentionEncoder(
            branch_dim, branch_hidden, seq_len=seq_len, depth=4, dropout=dropout)

        # Diverse trunk operators
        trunk_types = ['fourier', 'rbf', 'polynomial', 'wavelet']
        self.trunk_ops = nn.ModuleList([
            TrunkOperator(width, trunk_depth, tt, n_freq=n_freq)
            for tt in trunk_types
        ])
        self.trunk_types = trunk_types

        # Forecast heads: z → trunk_params + B (per operator)
        self.forecast_heads = nn.ModuleList([
            nn.Linear(branch_hidden, op.head_output_dim)
            for op in self.trunk_ops
        ])
        # Reconstruction heads: separate from forecast (for imputation)
        self.recon_heads = nn.ModuleList([
            nn.Linear(branch_hidden, op.head_output_dim)
            for op in self.trunk_ops
        ])
        for heads in [self.forecast_heads, self.recon_heads]:
            for h in heads:
                nn.init.xavier_normal_(h.weight, gain=0.1)
                nn.init.constant_(h.bias, 0)

    def _build_branch_input(self, x_ch, x_cross):
        branch_input = torch.cat([x_ch, x_cross], dim=-1)
        if self.spectral_branch:
            x_fft = torch.fft.rfft(x_ch, dim=-1)
            branch_input = torch.cat([branch_input, x_fft.real, x_fft.imag], dim=-1)
        return branch_input

    def _forward_channel(self, x_ch, x_cross, target_len, mode='forecast'):
        branch_input = self._build_branch_input(x_ch, x_cross)
        z = self.encoder(branch_input)

        # Extract dominant frequencies for conditioning
        dom_freqs = extract_dominant_freqs(x_ch, top_k=5)  # [B, 5]

        heads = self.forecast_heads if mode == 'forecast' else self.recon_heads
        output = torch.zeros(x_ch.shape[0], target_len,
                             dtype=x_ch.dtype, device=x_ch.device)
        for head, trunk_op in zip(heads, self.trunk_ops):
            component = trunk_op(head(z), target_len, dom_freqs=dom_freqs)
            output = output + component
        return output

    def get_representation(self, x_enc):
        """Extract representation for classification etc."""
        B, S, C = x_enc.shape
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = (x_enc - means) / torch.sqrt(
                torch.var(x_enc - means, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_cross = x_enc.mean(dim=-1)
        reps = []
        for ch in range(C):
            branch_input = self._build_branch_input(x_enc[:, :, ch], x_cross)
            reps.append(self.encoder(branch_input))
        return torch.stack(reps, dim=1)  # [B, C, hidden]

    def forecast(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None,
                 target_pred_len=None, **kwargs):
        if target_pred_len is None:
            target_pred_len = self.pred_len
        B, S, C = x_enc.shape

        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc = x_enc / stdev

        x_cross = x_enc.mean(dim=-1)
        outputs = []
        for ch in range(C):
            out_ch = self._forward_channel(x_enc[:, :, ch], x_cross, target_pred_len, 'forecast')
            outputs.append(out_ch)
        output = torch.stack(outputs, dim=-1)

        if self.use_norm:
            output = output * stdev + means
        return output

    def reconstruct(self, x_enc):
        """Reconstruction for imputation."""
        B, S, C = x_enc.shape
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc = x_enc / stdev

        x_cross = x_enc.mean(dim=-1)
        outputs = []
        for ch in range(C):
            out_ch = self._forward_channel(x_enc[:, :, ch], x_cross, S, 'recon')
            outputs.append(out_ch)
        output = torch.stack(outputs, dim=-1)

        if self.use_norm:
            output = output * stdev + means
        return output

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None,
                target_pred_len=None, **kwargs):
        return self.forecast(x_enc, target_pred_len=target_pred_len)


# ============================================================
# Synthetic Dataset
# ============================================================
class SyntheticWindowDataset(Dataset):
    def __init__(self, arrow_path, seq_len=96, stride=48, max_samples=100000):
        self.seq_len = seq_len
        self.windows = []
        ds = Dataset_GaussianPCoregionalization(
            root_path='./', data_path=arrow_path,
            n_variables=160, seq_len=seq_len, pred_len=seq_len,
            size=[seq_len, 0, seq_len], synthetic_length=1024, stride=stride)
        n_samples = min(len(ds), max_samples)
        for i in range(n_samples):
            x, y, _, _ = ds[i]
            if isinstance(x, torch.Tensor): x = x.numpy()
            if x.ndim == 1: x = x.reshape(-1, 1)
            ch = np.random.randint(0, x.shape[1])
            window = x[:, ch].astype(np.float32)
            std = np.std(window)
            if std > 1e-8:
                window = np.clip((window - np.mean(window)) / std, -10, 10)
                self.windows.append(window)
        self.windows = np.array(self.windows, dtype=np.float32)
        print(f'SyntheticWindowDataset: {len(self.windows)} windows')

    def __len__(self): return len(self.windows)
    def __getitem__(self, idx):
        return torch.tensor(self.windows[idx], dtype=torch.float32).unsqueeze(-1)


# ============================================================
# Pre-training
# ============================================================
def pretrain(model, device, save_path, epochs=20, lr=0.0003, mask_rate=0.4,
             spectral_weight=0.5):
    print(f'\n{"="*60}')
    print(f'Pre-training: Additive Decomp v2 (Spectral Loss + Freq Conditioning)')
    print(f'  spectral_weight={spectral_weight}')
    print(f'{"="*60}')

    real_ds = PilePretrainDataset(seq_len=96, stride=48,
                                   pile_root='./dataset/time_series_pile')
    print(f'Real: {len(real_ds)} windows')

    synth_ds = SyntheticWindowDataset('tempopfn_15k_1024.arrow',
                                      seq_len=96, stride=48, max_samples=100000)
    print(f'Synthetic: {len(synth_ds)} windows')

    combined = ConcatDataset([real_ds, synth_ds])
    n_val = min(10000, len(combined) // 10)
    n_train = len(combined) - n_val
    train_ds, val_ds = random_split(combined, [n_train, n_val])
    train_dl = DataLoader(train_ds, batch_size=128, shuffle=True,
                          num_workers=4, drop_last=True, pin_memory=True)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print(f'Combined: {len(combined)}, Train: {n_train}, Steps/epoch: {len(train_dl)}')

    best_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        losses, recon_losses, nt_losses = [], [], []
        t0 = time.time()

        for i, batch_x in enumerate(train_dl):
            batch_x = batch_x.float().to(device)
            B, S, C = batch_x.shape
            optimizer.zero_grad()

            # 1) Masked recon
            mask = (torch.rand_like(batch_x) > mask_rate).float()
            recon_out = model.reconstruct(batch_x * mask)
            loss_mat = F.mse_loss(recon_out, batch_x, reduction='none')
            inv_mask = 1.0 - mask
            recon_loss = (loss_mat * inv_mask).sum() / inv_mask.sum().clamp(min=1)

            # 2) Next-token prediction
            split = torch.randint(24, 72, (1,)).item()
            context = batch_x[:, :split, :]
            target = batch_x[:, split:, :]
            target_len = S - split
            context_padded = F.pad(context, (0, 0, 0, S - split))
            nt_out = model.forecast(context_padded, target_pred_len=target_len)
            nt_loss = F.mse_loss(nt_out, target)

            # 3) Spectral loss — 주파수 보존 강제
            # recon의 spectral loss
            spec_recon = spectral_loss(
                recon_out.squeeze(-1), batch_x.squeeze(-1))
            # next-token의 spectral loss
            spec_nt = spectral_loss(
                nt_out.squeeze(-1), target.squeeze(-1))
            spec_loss = spec_recon + spec_nt

            loss = recon_loss + nt_loss + spectral_weight * spec_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            losses.append(loss.item())
            recon_losses.append(recon_loss.item())
            nt_losses.append(nt_loss.item())

            if (i + 1) % 500 == 0:
                print(f'  iter {i+1}/{len(train_dl)}: recon={np.mean(recon_losses[-500:]):.4f} '
                      f'nt={np.mean(nt_losses[-500:]):.4f} spec={spec_loss.item():.4f}')

        scheduler.step()
        avg_loss = np.mean(losses)
        elapsed = time.time() - t0
        print(f'Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f} '
              f'(recon={np.mean(recon_losses):.4f} nt={np.mean(nt_losses):.4f}) '
              f'lr={scheduler.get_last_lr()[0]:.6f} ({elapsed:.0f}s) [v2+spectral+freqcond]')

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), save_path)
            print(f'  Saved checkpoint (best={best_loss:.4f})')

    model.load_state_dict(torch.load(save_path))
    return model


# ============================================================
# Evaluation
# ============================================================
def eval_forecasting(model, device):
    print(f'\n{"="*60}')
    print('Forecasting Eval (zero-shot)')
    print(f'{"="*60}')

    datasets = {
        'ETTh1': ('ETTh1', './dataset/ETT-small/', 'ETTh1.csv', 7),
        'ETTh2': ('ETTh2', './dataset/ETT-small/', 'ETTh2.csv', 7),
        'ETTm1': ('ETTm1', './dataset/ETT-small/', 'ETTm1.csv', 7),
        'ETTm2': ('ETTm2', './dataset/ETT-small/', 'ETTm2.csv', 7),
        'Weather': ('custom', './dataset/weather/', 'weather.csv', 21),
        'Exchange': ('custom', './dataset/exchange_rate/', 'exchange_rate.csv', 8),
    }

    moment_lp = {
        'ETTh1_96': 0.387, 'ETTh1_192': 0.410, 'ETTh1_336': 0.422, 'ETTh1_720': 0.454,
        'ETTh2_96': 0.288, 'ETTh2_192': 0.349, 'ETTh2_336': 0.369, 'ETTh2_720': 0.403,
        'ETTm1_96': 0.293, 'ETTm1_192': 0.326, 'ETTm1_336': 0.352, 'ETTm1_720': 0.405,
        'ETTm2_96': 0.170, 'ETTm2_192': 0.227, 'ETTm2_336': 0.275, 'ETTm2_720': 0.363,
        'Weather_96': 0.154, 'Weather_192': 0.197, 'Weather_336': 0.246, 'Weather_720': 0.315,
    }

    model.eval()
    results = {}
    for dname, (data, root, fpath, enc_in) in datasets.items():
        for pl in [96, 192, 336, 720]:
            a = SimpleNamespace(
                seq_len=96, pred_len=pl, label_len=48, data=data, root_path=root,
                data_path=fpath, features='M', target='OT', freq='h', embed='timeF',
                enc_in=enc_in, dec_in=enc_in, c_out=enc_in, num_workers=2, batch_size=32,
                exp_name='MTSF', ordered_data=False, data_amount=-1,
                combine_Gaussian_datasets=False, synthetic_data_path='',
                synthetic_root_path='./', synthetic_length=1024, stride=-1)
            _, test_dl = data_provider(a, 'test')
            preds, trues = [], []
            with torch.no_grad():
                for bx, by, _, _ in test_dl:
                    bx = bx.float().to(device)
                    out = model.forecast(bx, target_pred_len=pl)
                    preds.append(out.cpu().numpy())
                    trues.append(by[:, -pl:, :].numpy())
            p, t = np.concatenate(preds), np.concatenate(trues)
            mse = np.mean((p - t) ** 2)
            mae = np.mean(np.abs(p - t))
            key = f'{dname}_{pl}'
            results[key] = mse
            mlp = moment_lp.get(key, None)
            gap = f'{(mse/mlp-1)*100:+.1f}%' if mlp else '-'
            print(f'  {key}: MSE={mse:.4f} MAE={mae:.4f}  MOMENT_LP={mlp or "-"}  gap={gap}')

    return results


def eval_imputation(model, device):
    print(f'\n{"="*60}')
    print('Imputation Eval (zero-shot)')
    print(f'{"="*60}')

    datasets = {
        'ETTh1': ('ETTh1', './dataset/ETT-small/', 'ETTh1.csv', 7),
        'ETTh2': ('ETTh2', './dataset/ETT-small/', 'ETTh2.csv', 7),
        'ETTm1': ('ETTm1', './dataset/ETT-small/', 'ETTm1.csv', 7),
        'ETTm2': ('ETTm2', './dataset/ETT-small/', 'ETTm2.csv', 7),
        'Weather': ('custom', './dataset/weather/', 'weather.csv', 21),
    }

    moment = {
        'ETTh1': (0.402, 0.139), 'ETTh2': (0.125, 0.061),
        'ETTm1': (0.202, 0.074), 'ETTm2': (0.078, 0.031),
        'Weather': (0.082, 0.035),
    }

    model.eval()
    for dname, (data, root, fpath, enc_in) in datasets.items():
        a = SimpleNamespace(
            seq_len=96, pred_len=96, label_len=0, data=data, root_path=root,
            data_path=fpath, features='M', target='OT', freq='h', embed='timeF',
            enc_in=enc_in, dec_in=enc_in, c_out=enc_in, num_workers=2, batch_size=32,
            exp_name='MTSF', ordered_data=False, data_amount=-1,
            combine_Gaussian_datasets=False, synthetic_data_path='',
            synthetic_root_path='./', synthetic_length=1024, stride=-1)
        _, test_dl = data_provider(a, 'test')

        all_mse = []
        for mask_rate in [0.125, 0.25, 0.375, 0.5]:
            torch.manual_seed(2021)
            preds, trues, masks = [], [], []
            with torch.no_grad():
                for bx, by, _, _ in test_dl:
                    bx = bx.float().to(device)
                    mask = (torch.rand_like(bx) > mask_rate).float()
                    out = model.reconstruct(bx * mask)
                    preds.append(out.cpu().numpy())
                    trues.append(bx.cpu().numpy())
                    masks.append(mask.cpu().numpy())
            p, t, m = np.concatenate(preds), np.concatenate(trues), np.concatenate(masks)
            mse = np.mean((p[m == 0] - t[m == 0]) ** 2)
            all_mse.append(mse)
            print(f'  {dname} mask={mask_rate}: MSE={mse:.4f}')

        avg = np.mean(all_mse)
        m0, mlp = moment.get(dname, (None, None))
        print(f'  {dname} Mean: MSE={avg:.4f}  (MOMENT_0={m0}, MOMENT_LP={mlp})')


def eval_classification(model, device):
    print(f'\n{"="*60}')
    print('Classification Eval (linear probe)')
    print(f'{"="*60}')

    hidden = model.branch_hidden
    for p in model.parameters():
        p.requires_grad = False

    cls_datasets = ['Epilepsy', 'FingerMovements', 'BasicMotions',
                    'NATOPS', 'EthanolConcentration']

    for ds_name in cls_datasets:
        try:
            cls_root = './dataset/classification/Multivariate_ts'
            train_ds = Dataset_Classification(root_path=cls_root, flag='train',
                                              size=[96, 0, 96], data_path=ds_name)
            test_ds = Dataset_Classification(root_path=cls_root, flag='test',
                                             size=[96, 0, 96], data_path=ds_name)
            train_dl = DataLoader(train_ds, batch_size=16, shuffle=True, drop_last=True)
            test_dl = DataLoader(test_ds, batch_size=16, shuffle=False)

            cls_head = nn.Sequential(
                nn.Linear(hidden, 256), nn.GELU(), nn.Dropout(0.1),
                nn.Linear(256, train_ds.n_classes)
            ).to(device)
            opt = optim.Adam(cls_head.parameters(), lr=0.001)

            best_acc = 0
            for epoch in range(30):
                cls_head.train()
                for bx, label, _, _ in train_dl:
                    bx = bx.float().to(device)
                    label = label.long().to(device)
                    with torch.no_grad():
                        z = model.get_representation(bx).mean(dim=1)
                    loss = nn.CrossEntropyLoss()(cls_head(z), label)
                    opt.zero_grad(); loss.backward(); opt.step()

                cls_head.eval()
                ps, ls = [], []
                with torch.no_grad():
                    for bx, label, _, _ in test_dl:
                        bx = bx.float().to(device)
                        z = model.get_representation(bx).mean(dim=1)
                        ps.append(cls_head(z).argmax(-1).cpu().numpy())
                        ls.append(label.numpy())
                acc = accuracy_score(np.concatenate(ls), np.concatenate(ps))
                best_acc = max(best_acc, acc)

            print(f'  {ds_name}: Acc={best_acc:.4f}')
        except Exception as e:
            print(f'  {ds_name}: SKIP ({e})')

    for p in model.parameters():
        p.requires_grad = True


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    print('=' * 60)
    print('Additive Decomposition v2: Spectral Loss + Freq Conditioning')
    print('  Architecture: Shared Encoder + 4 Diverse Trunk (freq-conditioned)')
    print('  Config: medium (width=96, hidden=384)')
    print('  Data: Pile (Real) + TempoPFN (Synthetic)')
    print('  Loss: Masked Recon + Next-token + Spectral Loss')
    print('  New: Fourier trunk gets input-adaptive frequency basis')
    print('=' * 60)

    model = AdditiveDecompModel(
        seq_len=96, pred_len=96, width=96, branch_hidden=384,
        trunk_depth=2, spectral_branch=True, n_freq=32, dropout=0.1
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    n_enc = sum(p.numel() for p in model.encoder.parameters())
    n_fh = sum(sum(p.numel() for p in h.parameters()) for h in model.forecast_heads)
    n_rh = sum(sum(p.numel() for p in h.parameters()) for h in model.recon_heads)
    print(f'\nModel: {n_params/1e6:.1f}M total')
    print(f'  Shared Encoder: {n_enc/1e6:.1f}M')
    print(f'  Forecast Heads: {n_fh/1e6:.1f}M')
    print(f'  Recon Heads:    {n_rh/1e6:.1f}M')
    print(f'  Trunk types: {model.trunk_types}')

    save_path = 'checkpoints/additive_decomp_v2.pth'
    os.makedirs('checkpoints', exist_ok=True)

    # Pre-train with spectral loss
    model = pretrain(model, DEVICE, save_path, epochs=20, lr=0.0003,
                     spectral_weight=0.5)

    # Eval all tasks
    eval_forecasting(model, DEVICE)
    eval_imputation(model, DEVICE)
    eval_classification(model, DEVICE)

    print(f'\n{"="*60}')
    print('ALL DONE')
    print(f'{"="*60}')
