"""
Forecasting v2: get_representation으로 MoE 전체 활용 + deeper head
v1 문제: expert[0]만 사용 → MoE의 3/4를 버림
v2: model.get_representation() → 4 expert 가중합 representation 사용

사용법:
  CUDA_VISIBLE_DEVICES=X python experiments/exp_forecast_best_v2.py 2>&1 | tee log/eval/forecast_best_v2.log
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
import math

from model.DeepONetHyperMoE import Model


class ForecastHead(nn.Module):
    """Deeper forecast head: z → prediction via operator learning.

    Takes representation z, produces trunk_params + B,
    then uses Trunk to generate output at arbitrary time points.
    """
    def __init__(self, hidden, width, trunk_param_count, trunk_param_shapes,
                 n_freq=32, n_rbf=16, dropout=0.1):
        super().__init__()
        self.width = width
        self.trunk_param_count = trunk_param_count
        self.trunk_param_shapes = trunk_param_shapes
        self.n_freq = n_freq
        self.n_rbf = n_rbf

        output_dim = trunk_param_count + width

        # 3층 MLP
        mid = max(hidden, 1024)
        self.head = nn.Sequential(
            nn.Linear(hidden, mid),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mid, mid),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mid, output_dim),
        )
        self.bias = nn.Parameter(torch.zeros(1))

        # RBF centers
        self.register_buffer('rbf_centers', torch.linspace(0, 1, n_rbf))
        self._init()

    def _init(self):
        for m in self.head:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, z, target_pred_len, device):
        """z: [B, hidden] → output: [B, pred_len]"""
        batch_size = z.shape[0]

        head_out = self.head(z)
        trunk_params = head_out[:, :self.trunk_param_count] * 0.01
        B_coeff = head_out[:, self.trunk_param_count:]

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

        # Trunk features
        t = torch.linspace(0, 1, target_pred_len, dtype=z.dtype, device=device).unsqueeze(-1)
        freqs = torch.arange(1, self.n_freq + 1, dtype=t.dtype, device=device)
        sin_f = torch.sin(2 * math.pi * freqs.unsqueeze(0) * t)
        cos_f = torch.cos(2 * math.pi * freqs.unsqueeze(0) * t)
        features = torch.cat([t, sin_f, cos_f], dim=-1)

        # RBF + Poly
        rbf = torch.exp(-50.0 * (t - self.rbf_centers.unsqueeze(0)) ** 2)
        poly = torch.cat([t, t**2, t**3], dim=-1)
        features = torch.cat([features, rbf, poly], dim=-1)

        Phi = features.unsqueeze(0).expand(batch_size, -1, -1)
        for i, (w, b) in enumerate(trunk_weights):
            Phi = torch.bmm(Phi, w) + b.unsqueeze(1)
            if i < len(trunk_weights) - 1:
                Phi = F.gelu(Phi)

        output = torch.einsum('bp,bqp->bq', B_coeff, Phi)
        return output + self.bias


def train_and_eval(model, device, data, root, fpath, enc_in, pred_lens=[96, 192, 336, 720]):
    results = {}

    # Get trunk info from model's first expert
    expert = model.experts[0]
    trunk_param_count = expert.trunk_param_count
    trunk_param_shapes = expert.trunk_param_shapes
    width = expert.width
    hidden = model.branch_hidden

    for pl in pred_lens:
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

        # New deeper head
        forecast_head = ForecastHead(
            hidden, width, trunk_param_count, trunk_param_shapes
        ).to(device)

        # Freeze backbone
        for p in model.parameters():
            p.requires_grad = False

        optimizer = optim.Adam(forecast_head.parameters(), lr=0.001)
        best_loss, patience, best_state = float('inf'), 0, None

        for epoch in range(30):
            forecast_head.train()
            losses = []
            for bx, by, _, _ in train_dl:
                bx, by = bx.float().to(device), by.float().to(device)
                B_size, S, C = bx.shape

                # RevIN
                means = bx.mean(1, keepdim=True).detach()
                bx_norm = bx - means
                stdev = torch.sqrt(torch.var(bx_norm, dim=1, keepdim=True, unbiased=False) + 1e-5)
                bx_norm = bx_norm / stdev

                # Get representation from ALL experts (MoE weighted)
                with torch.no_grad():
                    z_all = model.get_representation(bx)  # [B, C, hidden]

                # Per channel forecast
                optimizer.zero_grad()
                outputs = []
                for ch in range(C):
                    z_ch = z_all[:, ch, :]  # [B, hidden]
                    out_ch = forecast_head(z_ch, pl, device)  # [B, pl]
                    outputs.append(out_ch)
                pred = torch.stack(outputs, dim=-1)  # [B, pl, C]

                # De-RevIN
                pred = pred * stdev + means

                loss = F.mse_loss(pred, by[:, -pl:, :])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(forecast_head.parameters(), 1.0)
                optimizer.step()
                losses.append(loss.item())

            tl = np.mean(losses)
            print(f'    epoch {epoch+1}: train_loss={tl:.4f} (best={best_loss:.4f}, patience={patience})')
            if tl < best_loss:
                best_loss = tl
                best_state = {k: v.cpu().clone() for k, v in forecast_head.state_dict().items()}
                patience = 0
            else:
                patience += 1
                if patience >= 5:
                    print(f'    early stopping at epoch {epoch+1}')
                    break

        # Test
        if best_state:
            forecast_head.load_state_dict(best_state)
        forecast_head.to(device).eval()

        preds, trues = [], []
        with torch.no_grad():
            for bx, by, _, _ in test_dl:
                bx = bx.float().to(device)
                B_size, S, C = bx.shape

                means = bx.mean(1, keepdim=True).detach()
                bx_norm = bx - means
                stdev = torch.sqrt(torch.var(bx_norm, dim=1, keepdim=True, unbiased=False) + 1e-5)

                z_all = model.get_representation(bx)

                outputs = []
                for ch in range(C):
                    z_ch = z_all[:, ch, :]
                    out_ch = forecast_head(z_ch, pl, device)
                    outputs.append(out_ch)
                pred = torch.stack(outputs, dim=-1)
                pred = pred * stdev + means

                preds.append(pred.cpu().numpy())
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

    ckpt_path = 'checkpoints/scaleup_pile_pretrain.pth'
    if not os.path.exists(ckpt_path):
        print(f'ERROR: {ckpt_path} not found!')
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
    model.eval()
    print(f'Loaded model, using get_representation() for ALL experts')

    datasets = {
        'ETTh1': ('ETTh1', './dataset/ETT-small/', 'ETTh1.csv', 7),
        'ETTh2': ('ETTh2', './dataset/ETT-small/', 'ETTh2.csv', 7),
        'ETTm1': ('ETTm1', './dataset/ETT-small/', 'ETTm1.csv', 7),
        'ETTm2': ('ETTm2', './dataset/ETT-small/', 'ETTm2.csv', 7),
        'Weather': ('custom', './dataset/weather/', 'weather.csv', 21),
        'Exchange': ('custom', './dataset/exchange_rate/', 'exchange_rate.csv', 8),
    }

    moment_lp = {
        'ETTh1_pl96': 0.387, 'ETTh1_pl192': 0.410, 'ETTh1_pl336': 0.422, 'ETTh1_pl720': 0.454,
        'ETTh2_pl96': 0.288, 'ETTh2_pl192': 0.349, 'ETTh2_pl336': 0.369, 'ETTh2_pl720': 0.403,
        'ETTm1_pl96': 0.293, 'ETTm1_pl192': 0.326, 'ETTm1_pl336': 0.352, 'ETTm1_pl720': 0.405,
        'ETTm2_pl96': 0.170, 'ETTm2_pl192': 0.227, 'ETTm2_pl336': 0.275, 'ETTm2_pl720': 0.363,
        'Weather_pl96': 0.154, 'Weather_pl192': 0.197, 'Weather_pl336': 0.246, 'Weather_pl720': 0.315,
    }

    all_results = {}
    print(f'\n{"="*60}')
    print('Forecasting v2: get_representation + Deeper Head')
    print(f'{"="*60}')

    for dname, (data, root, fpath, enc_in) in datasets.items():
        print(f'\n--- {dname} ---')
        results = train_and_eval(model, device, data, root, fpath, enc_in)
        all_results.update(results)

    print(f'\n{"="*60}')
    print('SUMMARY: Deeper v2 vs LP vs MOMENT_LP')
    print(f'{"="*60}')
    print(f'{"Dataset":<20} {"Deeper_v2":>10} {"LP(1층)":>10} {"MOMENT":>10}')
    print('-' * 52)

    lp_results = {
        'ETTh1_pl96': 0.409, 'ETTh1_pl192': 0.477, 'ETTh1_pl336': 0.527, 'ETTh1_pl720': 0.614,
        'ETTh2_pl96': 0.355, 'ETTh2_pl192': 0.438, 'ETTh2_pl336': 0.542, 'ETTh2_pl720': 0.766,
        'ETTm1_pl96': 0.338, 'ETTm1_pl192': 0.383, 'ETTm1_pl336': 0.415, 'ETTm1_pl720': 0.478,
        'ETTm2_pl96': 0.213, 'ETTm2_pl192': 0.300, 'ETTm2_pl336': 0.376, 'ETTm2_pl720': 0.491,
        'Weather_pl96': 0.185, 'Weather_pl192': 0.229, 'Weather_pl336': 0.281, 'Weather_pl720': 0.357,
        'Exchange_pl96': 0.093, 'Exchange_pl192': 0.187, 'Exchange_pl336': 0.357, 'Exchange_pl720': 0.905,
    }

    for k, v in all_results.items():
        lp = lp_results.get(k, '-')
        mlp = moment_lp.get(k, '-')
        lp_str = f'{lp:.3f}' if isinstance(lp, float) else lp
        mlp_str = f'{mlp:.3f}' if isinstance(mlp, float) else mlp
        print(f'  {k:<20} {v["mse"]:>10.4f} {lp_str:>10} {mlp_str:>10}')

    # ============================================================
    # GIFT-eval
    # ============================================================
    print(f'\n{"="*60}')
    print('GIFT-eval (zero-shot probabilistic forecasting)')
    print(f'{"="*60}')

    import subprocess
    gift_eval_script = '/workspace/gift-eval/eval_deeponet.py'
    if os.path.exists(gift_eval_script):
        tag = 'deeper_head_v2'
        spectral_flag = ['--spectral_branch'] if getattr(args, 'spectral_branch', False) else []
        cmd = [
            sys.executable, gift_eval_script,
            '--checkpoint', ckpt_path,
            '--device', 'cuda:0',
            '--output_tag', tag,
            '--model_type', 'DeepONetHyperMoE',
            '--branch_depth', '4', '--trunk_depth', '2',
            '--deeponet_width', '128', '--branch_hidden', '512',
            '--n_experts', '4',
            '--loss', 'MSE',
            '--seq_len', '96', '--pred_len', '96',
        ] + spectral_flag
        print(f'  Running GIFT-eval with tag={tag} ...')
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
        print(result.stdout[-2000:] if result.stdout else '')
        if result.returncode != 0:
            print(f'  GIFT-eval failed: {result.stderr[-500:]}')
        else:
            print(f'  GIFT-eval done! Results: /workspace/gift-eval/results/DeepONetHyper/all_results_{tag}.csv')
    else:
        print(f'  SKIP: {gift_eval_script} not found')
