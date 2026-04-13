"""
Basis 함수 탐색: Trunk input을 다양한 조합으로 비교
Fixed Trunk (small model)로 빠르게 from-scratch 비교

조합:
  A: Fourier only (sin/cos)
  B: Fourier + RBF + Poly (현재 mixed)
  C: Fourier + Wavelet + Sigmoid (step) + Decay
  D: 전부 (Fourier + RBF + Poly + Wavelet + Sigmoid + Decay)
  E: Learnable (nn.Parameter)

사용법:
  CUDA_VISIBLE_DEVICES=3 python experiments/exp_basis_exploration.py 2>&1 | tee log/basis_exploration.log
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


class FlexibleTrunk(nn.Module):
    """Trunk with configurable basis functions."""
    def __init__(self, width=64, basis_type='mixed', n_freq=32, n_rbf=16, depth=4):
        super().__init__()
        self.basis_type = basis_type
        self.n_freq = n_freq
        self.n_rbf = n_rbf
        self.width = width

        # Calculate input dim based on basis type
        dim = 1  # t itself
        if 'fourier' in basis_type:
            dim += 2 * n_freq  # sin + cos
        if 'rbf' in basis_type:
            dim += n_rbf
            self.register_buffer('rbf_centers', torch.linspace(0, 1, n_rbf))
        if 'poly' in basis_type:
            dim += 3  # t, t², t³
        if 'wavelet' in basis_type:
            n_scales = 4
            n_translations = 8
            dim += n_scales * n_translations
            self.n_scales = n_scales
            self.n_translations = n_translations
        if 'sigmoid' in basis_type:
            n_steps = 16
            dim += n_steps
            self.register_buffer('step_centers', torch.linspace(0, 1, n_steps))
        if 'decay' in basis_type:
            n_decays = 8
            dim += n_decays * 2  # exp(-αt) + exp(-α(1-t))
        if basis_type == 'learnable':
            dim = width  # directly learnable

        self.input_dim = dim

        if basis_type == 'learnable':
            # Learnable positional encoding
            self.pos_encoding = nn.Parameter(torch.randn(1024, width) * 0.02)  # max 1024 length
        else:
            # MLP trunk
            layers = [nn.Linear(dim, width * 2), nn.GELU()]
            for _ in range(depth - 2):
                layers.extend([nn.Linear(width * 2, width * 2), nn.GELU()])
            layers.append(nn.Linear(width * 2, width))
            self.net = nn.Sequential(*layers)

        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, pred_len):
        t = torch.linspace(0, 1, pred_len, device=self._get_device()).unsqueeze(-1)

        if self.basis_type == 'learnable':
            return self.pos_encoding[:pred_len]  # [pred_len, width]

        features = [t]

        if 'fourier' in self.basis_type:
            freqs = torch.arange(1, self.n_freq + 1, dtype=t.dtype, device=t.device)
            features.append(torch.sin(2 * math.pi * freqs.unsqueeze(0) * t))
            features.append(torch.cos(2 * math.pi * freqs.unsqueeze(0) * t))

        if 'rbf' in self.basis_type:
            features.append(torch.exp(-50.0 * (t - self.rbf_centers.unsqueeze(0)) ** 2))

        if 'poly' in self.basis_type:
            features.extend([t, t**2, t**3])

        if 'wavelet' in self.basis_type:
            # Mexican hat wavelet at different scales/translations
            for s in range(self.n_scales):
                scale = 2.0 ** s
                for k in range(self.n_translations):
                    center = k / (self.n_translations - 1)
                    u = scale * (t - center)
                    # Mexican hat: (1 - u²) * exp(-u²/2)
                    wav = (1 - u**2) * torch.exp(-u**2 / 2)
                    features.append(wav)

        if 'sigmoid' in self.basis_type:
            # Sigmoid step functions at different positions
            features.append(torch.sigmoid(20.0 * (t - self.step_centers.unsqueeze(0))))

        if 'decay' in self.basis_type:
            for alpha in [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]:
                features.append(torch.exp(-alpha * t))
                features.append(torch.exp(-alpha * (1 - t)))

        x = torch.cat(features, dim=-1)
        return self.net(x)

    def _get_device(self):
        if hasattr(self, 'rbf_centers'):
            return self.rbf_centers.device
        if hasattr(self, 'step_centers'):
            return self.step_centers.device
        if hasattr(self, 'pos_encoding'):
            return self.pos_encoding.device
        return next(self.parameters()).device


class BasisTestModel(nn.Module):
    def __init__(self, seq_len=96, pred_len=96, width=64, hidden=256, basis_type='mixed'):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.width = width
        self.hidden = hidden

        self.encoder = build_encoder('patch_attn', seq_len, hidden,
                                     seq_len=seq_len, depth=3,
                                     activation='gelu', dropout=0.1)
        self.trunk = FlexibleTrunk(width=width, basis_type=basis_type, depth=4)
        self.forecast_head = nn.Linear(hidden, width)
        self.recon_head = nn.Linear(hidden, width)
        self.bias = nn.Parameter(torch.zeros(1))
        self.recon_bias = nn.Parameter(torch.zeros(1))

    def _encode(self, x_enc):
        B, S, C = x_enc.shape
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev
        z_list = [self.encoder(x_enc[:, :, ch]) for ch in range(C)]
        z_all = torch.stack(z_list, dim=1)
        return z_all, means, stdev

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None,
                target_pred_len=None, **kwargs):
        if target_pred_len is None:
            target_pred_len = self.pred_len
        B, S, C = x_enc.shape
        z_all, means, stdev = self._encode(x_enc)
        Phi = self.trunk(target_pred_len)
        outputs = []
        for ch in range(C):
            B_coeff = self.forecast_head(z_all[:, ch, :])
            out_ch = torch.einsum('bp,qp->bq', B_coeff, Phi) + self.bias
            outputs.append(out_ch)
        output = torch.stack(outputs, dim=-1)
        return output * stdev + means

    def reconstruct(self, x_enc):
        B, S, C = x_enc.shape
        z_all, means, stdev = self._encode(x_enc)
        Phi = self.trunk(S)
        outputs = []
        for ch in range(C):
            B_coeff = self.recon_head(z_all[:, ch, :])
            out_ch = torch.einsum('bp,qp->bq', B_coeff, Phi) + self.recon_bias
            outputs.append(out_ch)
        output = torch.stack(outputs, dim=-1)
        return output * stdev + means

    def get_representation(self, x_enc):
        z_all, _, _ = self._encode(x_enc)
        return z_all


def test_basis(basis_type, device):
    print(f'\n{"="*50}')
    print(f'Basis: {basis_type}')
    print(f'{"="*50}')

    results = {}

    # Forecasting
    datasets = {
        'ETTh1': ('ETTh1', './dataset/ETT-small/', 'ETTh1.csv', 7),
        'ETTh2': ('ETTh2', './dataset/ETT-small/', 'ETTh2.csv', 7),
    }
    for dname, (data, root, fpath, enc_in) in datasets.items():
        for pl in [96]:
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
            model = BasisTestModel(96, pl, 64, 256, basis_type).to(device)
            if dname == 'ETTh1':
                n_params = sum(p.numel() for p in model.parameters())
                print(f'  Params: {n_params:,}')

            _, train_dl = data_provider(args, 'train')
            _, test_dl = data_provider(args, 'test')
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            best_loss, patience = float('inf'), 0
            best_state = None

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
                    if patience >= 5: break

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
            results[f'{dname}_pl{pl}'] = mse
            print(f'  {dname}_pl{pl}: MSE={mse:.4f}')

    # Classification
    for ds_name in ['Epilepsy', 'BasicMotions']:
        cls_root = './dataset/classification/Multivariate_ts'
        model = BasisTestModel(96, 96, 64, 256, basis_type).to(device)
        train_ds = Dataset_Classification(root_path=cls_root, flag='train', size=[96,0,96], data_path=ds_name)
        test_ds = Dataset_Classification(root_path=cls_root, flag='test', size=[96,0,96], data_path=ds_name)
        train_dl = DataLoader(train_ds, batch_size=16, shuffle=True, drop_last=True)
        test_dl = DataLoader(test_ds, batch_size=16, shuffle=False)

        cls_head = nn.Sequential(nn.Linear(256, 128), nn.GELU(), nn.Dropout(0.1),
                                 nn.Linear(128, train_ds.n_classes)).to(device)
        optimizer = optim.Adam(list(model.parameters()) + list(cls_head.parameters()), lr=0.001)
        best_acc = 0
        for epoch in range(50):
            model.train(); cls_head.train()
            for bx, label, _, _ in train_dl:
                bx = bx.float().to(device); label = label.long().to(device)
                z = model.get_representation(bx).mean(dim=1)
                loss = nn.CrossEntropyLoss()(cls_head(z), label)
                optimizer.zero_grad(); loss.backward(); optimizer.step()
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
        results[f'cls_{ds_name}'] = best_acc
        print(f'  {ds_name}: Acc={best_acc:.4f}')

    return results


if __name__ == '__main__':
    device = torch.device('cuda')

    basis_types = {
        'A_fourier': 'fourier',
        'B_mixed': 'fourier_rbf_poly',
        'C_wavelet': 'fourier_wavelet_sigmoid_decay',
        'D_all': 'fourier_rbf_poly_wavelet_sigmoid_decay',
        'E_learnable': 'learnable',
    }

    all_results = {}
    for name, btype in basis_types.items():
        results = test_basis(btype, device)
        all_results[name] = results

    print(f'\n{"="*60}')
    print('SUMMARY: Basis Function Comparison')
    print(f'{"="*60}')
    header = f'{"Basis":<20} {"ETTh1":>8} {"ETTh2":>8} {"Epilepsy":>10} {"BasicMot":>10}'
    print(header)
    print('-' * len(header))
    for name, res in all_results.items():
        e1 = res.get('ETTh1_pl96', 0)
        e2 = res.get('ETTh2_pl96', 0)
        ep = res.get('cls_Epilepsy', 0)
        bm = res.get('cls_BasicMotions', 0)
        print(f'{name:<20} {e1:>8.4f} {e2:>8.4f} {ep:>10.4f} {bm:>10.4f}')
