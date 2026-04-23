"""
Forecasting 성능 극대화 실험
Step 1: Pile pretrain (masked recon only) — 이미 있는 checkpoint 활용
Step 2: Frozen backbone + deeper forecast head 학습 (per dataset)

기존 LP: Linear(512→4288) — 1층
이번:    Linear(512→1024) → GELU → Linear(1024→1024) → GELU → Linear(1024→4288) — 3층

사용법:
  CUDA_VISIBLE_DEVICES=X python experiments/exp_forecast_best.py 2>&1 | tee log/eval/forecast_best.log
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import optim
from types import SimpleNamespace
from data_provider.data_factory import data_provider
import time

from model.DeepONetHyperMoE import Model
import math


class DeeperForecastHead(nn.Module):
    """3층 forecast head: z → trunk_params + B
    기존 Linear 1층 대비 훨씬 좋은 trunk_params 생성 가능.
    """
    def __init__(self, hidden, output_dim, dropout=0.1):
        super().__init__()
        mid = max(hidden, output_dim // 2)
        self.net = nn.Sequential(
            nn.Linear(hidden, mid),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mid, mid),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mid, output_dim),
        )
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, z):
        return self.net(z)


def forecast_with_deeper_head(model, deeper_head, x_enc, device, target_pred_len, expert_idx=0):
    """Frozen model의 encoder로 z 뽑고, deeper_head로 forecast."""
    B_size, S, C = x_enc.shape

    # RevIN
    means = x_enc.mean(1, keepdim=True).detach()
    x_enc = x_enc - means
    stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
    x_enc = x_enc / stdev

    expert = model.experts[expert_idx]

    outputs = []
    for ch in range(C):
        x_ch = x_enc[:, :, ch]

        # Branch input
        if model.spectral_branch:
            x_fft = torch.fft.rfft(x_ch, dim=-1)
            x_spectral = torch.cat([x_fft.real, x_fft.imag], dim=-1)
            branch_input = torch.cat([x_ch, x_spectral], dim=-1)
        else:
            branch_input = x_ch

        # Encoder (frozen)
        with torch.no_grad():
            z = expert.encoder(branch_input)  # [B, hidden]

        # Deeper head (trainable)
        head_out = deeper_head(z)  # [B, trunk_param_count + width]

        trunk_param_count = expert.trunk_param_count
        width = expert.width
        trunk_params = head_out[:, :trunk_param_count] * 0.01
        B_coeff = head_out[:, trunk_param_count:]

        # Trunk forward (frozen weights used as structure, but trunk_params from head)
        trunk_weights = []
        idx = 0
        for in_dim, out_dim, bias_size in expert.trunk_param_shapes:
            w_size = in_dim * out_dim
            w = trunk_params[:, idx:idx+w_size].view(B_size, in_dim, out_dim)
            idx += w_size
            b = trunk_params[:, idx:idx+bias_size].view(B_size, out_dim)
            idx += bias_size
            trunk_weights.append((w, b))

        t = torch.linspace(0, 1, target_pred_len, dtype=x_ch.dtype, device=device).unsqueeze(-1)
        freqs = torch.arange(1, expert.n_freq + 1, dtype=t.dtype, device=device)
        sin_f = torch.sin(2 * math.pi * freqs.unsqueeze(0) * t)
        cos_f = torch.cos(2 * math.pi * freqs.unsqueeze(0) * t)
        t_features = torch.cat([t, sin_f, cos_f], dim=-1)

        if hasattr(expert, 'rbf_centers'):
            rbf = torch.exp(-50.0 * (t - expert.rbf_centers.unsqueeze(0)) ** 2)
            poly = torch.cat([t, t**2, t**3], dim=-1)
            t_features = torch.cat([t_features, rbf, poly], dim=-1)

        Phi = t_features.unsqueeze(0).expand(B_size, -1, -1)
        for i, (w, b) in enumerate(trunk_weights):
            Phi = torch.bmm(Phi, w) + b.unsqueeze(1)
            if i < len(trunk_weights) - 1:
                Phi = F.gelu(Phi)

        out_ch = torch.einsum('bp,bqp->bq', B_coeff, Phi) + expert.bias
        outputs.append(out_ch)

    output = torch.stack(outputs, dim=-1)
    output = output * stdev + means
    return output


def train_and_eval(model, device, data, root, fpath, enc_in, pred_lens=[96, 192, 336, 720]):
    """한 데이터셋에 대해 deeper head 학습 + eval."""
    results = {}

    for pl in pred_lens:
        # Data
        args = SimpleNamespace(
            seq_len=96, pred_len=pl, label_len=48,
            data=data, root_path=root, data_path=fpath,
            features='M', target='OT', freq='h', embed='timeF',
            enc_in=enc_in, dec_in=enc_in, c_out=enc_in,
            num_workers=2, batch_size=32,
            exp_name='MTSF', ordered_data=False, data_amount=-1,
            combine_Gaussian_datasets=False,
            synthetic_data_path='', synthetic_root_path='./',
            synthetic_length=1024, stride=-1,
        )
        _, train_dl = data_provider(args, 'train')
        _, test_dl = data_provider(args, 'test')

        # Deeper head
        expert = model.experts[0]
        output_dim = expert.trunk_param_count + expert.width
        hidden = model.branch_hidden
        deeper_head = DeeperForecastHead(hidden, output_dim).to(device)

        # Freeze backbone
        for p in model.parameters():
            p.requires_grad = False

        optimizer = optim.Adam(deeper_head.parameters(), lr=0.001)
        best_loss, patience, best_state = float('inf'), 0, None

        for epoch in range(30):
            deeper_head.train()
            losses = []
            for bx, by, _, _ in train_dl:
                bx, by = bx.float().to(device), by.float().to(device)
                optimizer.zero_grad()
                out = forecast_with_deeper_head(model, deeper_head, bx, device, pl)
                loss = F.mse_loss(out, by[:, -pl:, :])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(deeper_head.parameters(), 1.0)
                optimizer.step()
                losses.append(loss.item())

            tl = np.mean(losses)
            print(f'    epoch {epoch+1}: train_loss={tl:.4f} (best={best_loss:.4f}, patience={patience})')
            if tl < best_loss:
                best_loss = tl
                best_state = {k: v.cpu().clone() for k, v in deeper_head.state_dict().items()}
                patience = 0
            else:
                patience += 1
                if patience >= 5:
                    print(f'    early stopping at epoch {epoch+1}')
                    break

        # Test
        if best_state:
            deeper_head.load_state_dict(best_state)
        deeper_head.to(device).eval()
        preds, trues = [], []
        with torch.no_grad():
            for bx, by, _, _ in test_dl:
                bx = bx.float().to(device)
                out = forecast_with_deeper_head(model, deeper_head, bx, device, pl)
                preds.append(out.cpu().numpy())
                trues.append(by[:, -pl:, :].numpy())
        p, t = np.concatenate(preds), np.concatenate(trues)
        mse = np.mean((p - t) ** 2)
        mae = np.mean(np.abs(p - t))

        for p_m in model.parameters():
            p_m.requires_grad = True

        dname = fpath.replace('.csv', '')
        key = f'{dname}_pl{pl}'
        results[key] = {'mse': mse, 'mae': mae}
        print(f'  {key}: MSE={mse:.4f} MAE={mae:.4f}')

    return results


if __name__ == '__main__':
    device = torch.device('cuda')

    # 기존 checkpoint 로드 (masked recon only pretrain)
    ckpt_path = 'checkpoints/scaleup_pile_pretrain.pth'
    if not os.path.exists(ckpt_path):
        print(f'ERROR: {ckpt_path} not found!')
        print('Run exp_scaleup_pretrain.py first.')
        sys.exit(1)

    args = SimpleNamespace(
        seq_len=96, pred_len=96, use_norm=True,
        deeponet_width=128, n_experts=4, branch_depth=4, trunk_depth=2,
        activation='gelu', dropout=0.1, branch_hidden=512,
        spectral_branch=False, skip_mode='none',
        use_cross_channel=False, trunk_basis='mixed',
        encoder_type='patch_attn', loss='MSE',
    )
    model = Model(args).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device), strict=False)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'Loaded model: {n_params/1e6:.1f}M params')
    print(f'Using DEEPER forecast head (3-layer MLP)')

    # All forecasting datasets
    datasets = {
        'ETTh1': ('ETTh1', './dataset/ETT-small/', 'ETTh1.csv', 7),
        'ETTh2': ('ETTh2', './dataset/ETT-small/', 'ETTh2.csv', 7),
        'ETTm1': ('ETTm1', './dataset/ETT-small/', 'ETTm1.csv', 7),
        'ETTm2': ('ETTm2', './dataset/ETT-small/', 'ETTm2.csv', 7),
        'Weather': ('custom', './dataset/weather/', 'weather.csv', 21),
        'Exchange': ('custom', './dataset/exchange_rate/', 'exchange_rate.csv', 8),
    }

    all_results = {}
    print(f'\n{"="*60}')
    print('Forecasting with Deeper Head (frozen backbone)')
    print(f'{"="*60}')

    # MOMENT LP values for comparison
    moment_lp = {
        'ETTh1_pl96': 0.387, 'ETTh1_pl192': 0.410, 'ETTh1_pl336': 0.422, 'ETTh1_pl720': 0.454,
        'ETTh2_pl96': 0.288, 'ETTh2_pl192': 0.349, 'ETTh2_pl336': 0.369, 'ETTh2_pl720': 0.403,
        'ETTm1_pl96': 0.293, 'ETTm1_pl192': 0.326, 'ETTm1_pl336': 0.352, 'ETTm1_pl720': 0.405,
        'ETTm2_pl96': 0.170, 'ETTm2_pl192': 0.227, 'ETTm2_pl336': 0.275, 'ETTm2_pl720': 0.363,
        'Weather_pl96': 0.154, 'Weather_pl192': 0.197, 'Weather_pl336': 0.246, 'Weather_pl720': 0.315,
    }

    for dname, (data, root, fpath, enc_in) in datasets.items():
        print(f'\n--- {dname} ---')
        results = train_and_eval(model, device, data, root, fpath, enc_in)
        all_results.update(results)

    # Summary
    print(f'\n{"="*60}')
    print('SUMMARY: Deeper Head vs LP vs MOMENT_LP')
    print(f'{"="*60}')
    print(f'{"Dataset":<20} {"Deeper":>8} {"LP(1층)":>8} {"MOMENT":>8}')
    print('-' * 48)

    lp_results = {
        'ETTh1_pl96': 0.409, 'ETTh1_pl192': 0.477, 'ETTh1_pl336': 0.527, 'ETTh1_pl720': 0.614,
        'ETTh2_pl96': 0.355, 'ETTm1_pl96': 0.338, 'ETTm2_pl96': 0.213, 'Weather_pl96': 0.185,
    }

    for k, v in all_results.items():
        lp = lp_results.get(k, '-')
        mlp = moment_lp.get(k, '-')
        lp_str = f'{lp:.3f}' if isinstance(lp, float) else lp
        mlp_str = f'{mlp:.3f}' if isinstance(mlp, float) else mlp
        print(f'  {k:<20} {v["mse"]:>8.4f} {lp_str:>8} {mlp_str:>8}')
