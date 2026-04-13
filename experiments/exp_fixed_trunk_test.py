"""
Fixed Trunk vs Hypernetwork Trunk 비교 (from-scratch, real data)
독립 실험 파일 — 기존 코드 안 건드림

사용법:
  source /opt/miniforge3/etc/profile.d/conda.sh && conda activate timefound
  CUDA_VISIBLE_DEVICES=1 python experiments/exp_fixed_trunk_test.py 2>&1 | tee log/fixed_trunk_test.log
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import optim
from types import SimpleNamespace
from data_provider.data_factory import data_provider
from data_provider.data_loader import Dataset_Classification
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from model.encoders import build_encoder


class FixedTrunkExpert(nn.Module):
    """DeepONet expert with FIXED trunk (learned but not input-dependent)."""
    def __init__(self, branch_dim, width, branch_depth, trunk_depth,
                 branch_hidden, n_freq, activation, dropout,
                 encoder_type='patch_attn', seq_len=96, n_rbf=16):
        super().__init__()
        self.width = width
        self.n_freq = n_freq
        self.n_rbf = n_rbf
        self.activation = activation

        # Trunk input: mixed basis
        trunk_input_dim = 1 + 2 * n_freq + n_rbf + 3  # 84

        # Fixed Trunk MLP (standard nn.Linear, NOT hypernetwork)
        trunk_layers = []
        trunk_layers.append(nn.Linear(trunk_input_dim, width))
        trunk_layers.append(nn.GELU())
        for _ in range(trunk_depth - 2):
            trunk_layers.append(nn.Linear(width, width))
            trunk_layers.append(nn.GELU())
        # No activation on last layer
        if trunk_depth > 1:
            trunk_layers.append(nn.Linear(width, width))
        self.trunk_net = nn.Sequential(*trunk_layers)

        # RBF centers
        self.register_buffer('rbf_centers', torch.linspace(0, 1, n_rbf))

        # Encoder
        self.encoder = build_encoder(encoder_type, branch_dim, branch_hidden,
                                     seq_len=seq_len, depth=branch_depth,
                                     activation=activation, dropout=dropout)

        # Head: only outputs B coefficients (NOT trunk weights!)
        self.forecast_head = nn.Linear(branch_hidden, width)  # 256 → 64 (was 4288!)
        self.recon_head = nn.Linear(branch_hidden, width)

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
        centers = self.rbf_centers.unsqueeze(0)
        rbf = torch.exp(-50.0 * (t - centers) ** 2)
        poly = torch.cat([t, t ** 2, t ** 3], dim=-1)
        return torch.cat([fourier, rbf, poly], dim=-1)

    def get_representation(self, branch_input):
        return self.encoder(branch_input)

    def forward(self, branch_input, target_pred_len, head='forecast'):
        batch_size = branch_input.shape[0]

        z = self.encoder(branch_input)

        if head == 'forecast':
            B = self.forecast_head(z)  # [B, 64] — just coefficients!
            bias = self.bias
        else:
            B = self.recon_head(z)
            bias = self.recon_bias

        # Fixed Trunk: same basis for all inputs
        t = torch.linspace(0, 1, target_pred_len,
                           dtype=branch_input.dtype,
                           device=branch_input.device).unsqueeze(-1)
        t_features = self._get_trunk_features(t)  # [pred_len, 84]
        Phi = self.trunk_net(t_features)  # [pred_len, 64] — FIXED, not input-dependent

        # DeepONet: B · Φ
        output = torch.einsum('bp,qp->bq', B, Phi)  # [B, pred_len]
        return output + bias


class FixedTrunkModel(nn.Module):
    """Full model with Fixed Trunk experts + MoE routing."""
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.use_norm = getattr(configs, 'use_norm', True)
        self.width = getattr(configs, 'deeponet_width', 64)
        self.n_experts = getattr(configs, 'n_experts', 4)
        self.branch_hidden = self.width * 4  # 256

        branch_dim = self.seq_len  # no cross, no spectral

        self.experts = nn.ModuleList([
            FixedTrunkExpert(
                branch_dim=branch_dim, width=self.width,
                branch_depth=getattr(configs, 'branch_depth', 4),
                trunk_depth=getattr(configs, 'trunk_depth', 2),
                branch_hidden=self.branch_hidden,
                n_freq=32, activation='gelu', dropout=0.1,
                encoder_type=getattr(configs, 'encoder_type', 'patch_attn'),
                seq_len=self.seq_len,
            )
            for _ in range(self.n_experts)
        ])

        from model.DeepONetHyperMoE import FNN
        self.router = FNN(branch_dim, self.n_experts, depth=3, width=256,
                          activation='gelu', dropout=0.0)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None,
                target_pred_len=None, **kwargs):
        if target_pred_len is None:
            target_pred_len = self.pred_len

        B, S, C = x_enc.shape

        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc = x_enc / stdev

        outputs = []
        for ch in range(C):
            x_ch = x_enc[:, :, ch]  # [B, 96]

            router_logits = self.router(x_ch)
            expert_weights = F.softmax(router_logits, dim=-1)

            out_ch = torch.zeros(B, target_pred_len, dtype=x_ch.dtype, device=x_ch.device)
            for i, expert in enumerate(self.experts):
                expert_out = expert(x_ch, target_pred_len, head='forecast')
                weight = expert_weights[:, i].unsqueeze(-1)
                out_ch = out_ch + weight * expert_out
            outputs.append(out_ch)

        output = torch.stack(outputs, dim=-1)

        if self.use_norm:
            output = output * stdev + means

        return output

    def reconstruct(self, x_enc):
        B, S, C = x_enc.shape

        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc = x_enc / stdev

        outputs = []
        for ch in range(C):
            x_ch = x_enc[:, :, ch]
            router_logits = self.router(x_ch)
            expert_weights = F.softmax(router_logits, dim=-1)

            out_ch = torch.zeros(B, S, dtype=x_ch.dtype, device=x_ch.device)
            for i, expert in enumerate(self.experts):
                expert_out = expert(x_ch, S, head='recon')
                weight = expert_weights[:, i].unsqueeze(-1)
                out_ch = out_ch + weight * expert_out
            outputs.append(out_ch)

        output = torch.stack(outputs, dim=-1)

        if self.use_norm:
            output = output * stdev + means
        return output

    def get_representation(self, x_enc):
        B, S, C = x_enc.shape

        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc = x_enc / stdev

        z_list = []
        for ch in range(C):
            x_ch = x_enc[:, :, ch]
            router_logits = self.router(x_ch)
            expert_weights = F.softmax(router_logits, dim=-1)

            z_ch = torch.zeros(B, self.branch_hidden, dtype=x_ch.dtype, device=x_ch.device)
            for i, expert in enumerate(self.experts):
                z_e = expert.get_representation(x_ch)
                weight = expert_weights[:, i].unsqueeze(-1)
                z_ch = z_ch + weight * z_e
            z_list.append(z_ch)

        return torch.stack(z_list, dim=1)  # [B, C, hidden]


def run_experiment():
    device = torch.device('cuda')

    datasets = {
        'ETTh1': ('ETTh1', './dataset/ETT-small/', 'ETTh1.csv', 7),
        'ETTh2': ('ETTh2', './dataset/ETT-small/', 'ETTh2.csv', 7),
        'Weather': ('custom', './dataset/weather/', 'weather.csv', 21),
        'Exchange': ('custom', './dataset/exchange_rate/', 'exchange_rate.csv', 8),
    }

    all_results = {}

    # ============ Forecasting ============
    print('=' * 60)
    print('FORECASTING (Fixed Trunk, from-scratch)')
    print('=' * 60)

    for dname, (data, root, fpath, enc_in) in datasets.items():
        for pl in [96, 336]:
            args = SimpleNamespace(
                seq_len=96, pred_len=pl, label_len=48, use_norm=True,
                deeponet_width=64, n_experts=4, branch_depth=4, trunk_depth=2,
                encoder_type='patch_attn',
                data=data, root_path=root, data_path=fpath,
                features='M', target='OT', freq='h', embed='timeF',
                enc_in=enc_in, dec_in=enc_in, c_out=enc_in,
                num_workers=2, batch_size=32,
                exp_name='MTSF', ordered_data=False, data_amount=-1,
                combine_Gaussian_datasets=False,
                synthetic_data_path='', synthetic_root_path='./',
                synthetic_length=1024, stride=-1,
            )

            model = FixedTrunkModel(args).to(device)
            if dname == 'ETTh1' and pl == 96:
                n_params = sum(p.numel() for p in model.parameters())
                print(f'Model params: {n_params:,}')

            _, train_dl = data_provider(args, 'train')
            _, test_dl = data_provider(args, 'test')

            optimizer = optim.Adam(model.parameters(), lr=0.001)
            best_loss = float('inf')

            for epoch in range(20):
                model.train()
                losses = []
                for bx, by, _, _ in train_dl:
                    bx, by = bx.float().to(device), by.float().to(device)
                    optimizer.zero_grad()
                    out = model(bx, target_pred_len=pl)
                    loss = F.mse_loss(out, by[:, -pl:, :])
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    losses.append(loss.item())

                tl = np.mean(losses)
                if tl < best_loss:
                    best_loss = tl
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    patience = 0
                else:
                    patience += 1
                    if patience >= 5:
                        break

            model.load_state_dict(best_state)
            model.to(device).eval()
            preds, trues = [], []
            with torch.no_grad():
                for bx, by, _, _ in test_dl:
                    bx = bx.float().to(device)
                    out = model(bx, target_pred_len=pl)
                    preds.append(out.cpu().numpy())
                    trues.append(by[:, -pl:, :].numpy())

            preds, trues = np.concatenate(preds), np.concatenate(trues)
            mse = np.mean((preds - trues) ** 2)
            key = f'{dname}_pl{pl}'
            all_results[key] = mse
            print(f'  {key}: MSE={mse:.4f}')

    # ============ Classification ============
    print('\n' + '=' * 60)
    print('CLASSIFICATION (Fixed Trunk, from-scratch)')
    print('=' * 60)

    cls_root = './dataset/classification/Multivariate_ts'
    hidden = 256

    for ds_name in ['Epilepsy', 'FingerMovements', 'BasicMotions', 'NATOPS', 'EthanolConcentration']:
        args = SimpleNamespace(
            seq_len=96, pred_len=96, use_norm=True, deeponet_width=64,
            n_experts=4, branch_depth=4, trunk_depth=2, encoder_type='patch_attn',
        )
        model = FixedTrunkModel(args).to(device)

        train_ds = Dataset_Classification(root_path=cls_root, flag='train', size=[96, 0, 96], data_path=ds_name)
        test_ds = Dataset_Classification(root_path=cls_root, flag='test', size=[96, 0, 96], data_path=ds_name)
        train_dl = DataLoader(train_ds, batch_size=16, shuffle=True, drop_last=True)
        test_dl = DataLoader(test_ds, batch_size=16, shuffle=False)

        cls_head = nn.Sequential(
            nn.Linear(hidden, 128), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(128, train_ds.n_classes)
        ).to(device)

        optimizer = optim.Adam(list(model.parameters()) + list(cls_head.parameters()), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        best_acc = 0
        for epoch in range(50):
            model.train(); cls_head.train()
            for bx, label, _, _ in train_dl:
                bx = bx.float().to(device)
                label = label.long().to(device)
                z = model.get_representation(bx).mean(dim=1)
                loss = criterion(cls_head(z), label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            model.eval(); cls_head.eval()
            preds, labels = [], []
            with torch.no_grad():
                for bx, label, _, _ in test_dl:
                    bx = bx.float().to(device)
                    z = model.get_representation(bx).mean(dim=1)
                    preds.append(cls_head(z).argmax(-1).cpu().numpy())
                    labels.append(label.numpy())
            acc = accuracy_score(np.concatenate(labels), np.concatenate(preds))
            best_acc = max(best_acc, acc)

        all_results[f'cls_{ds_name}'] = best_acc
        print(f'  {ds_name}: Acc={best_acc:.4f}')

    # ============ Summary ============
    print('\n' + '=' * 60)
    print('SUMMARY: Fixed Trunk (from-scratch, real data)')
    print('=' * 60)
    for k, v in all_results.items():
        print(f'  {k}: {v:.4f}')


if __name__ == '__main__':
    run_experiment()
