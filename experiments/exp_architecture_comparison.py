"""
Architecture Comparison: 2 models × 2 settings = 4 experiments
  Model A: Fixed Trunk + Cross-channel attention
  Model B: Transformer encoder + Fixed Trunk (operator learning 유지)

Each model tested on:
  1. From-scratch (real data, per dataset)
  2. Unified pretrain (all real data, masked recon) → zero-shot eval

사용법:
  CUDA_VISIBLE_DEVICES=0 python experiments/exp_architecture_comparison.py --model fixed_cross --mode scratch
  CUDA_VISIBLE_DEVICES=1 python experiments/exp_architecture_comparison.py --model fixed_cross --mode pretrain
  CUDA_VISIBLE_DEVICES=2 python experiments/exp_architecture_comparison.py --model transformer --mode scratch
  CUDA_VISIBLE_DEVICES=3 python experiments/exp_architecture_comparison.py --model transformer --mode pretrain
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import math
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import optim
from torch.utils.data import DataLoader, random_split
from types import SimpleNamespace
from sklearn.metrics import accuracy_score
import time

from model.encoders import build_encoder
from data_provider.data_factory import data_provider
from data_provider.data_loader import Dataset_Classification


# ============================================================
# Fixed Trunk (shared, deep)
# ============================================================
class FixedTrunk(nn.Module):
    """Deep fixed trunk: learns diverse basis functions."""
    def __init__(self, width=64, n_freq=32, n_rbf=16, depth=4):
        super().__init__()
        self.n_freq = n_freq
        self.n_rbf = n_rbf
        trunk_input_dim = 1 + 2 * n_freq + n_rbf + 3  # 84

        layers = []
        layers.append(nn.Linear(trunk_input_dim, width * 2))
        layers.append(nn.GELU())
        for _ in range(depth - 2):
            layers.append(nn.Linear(width * 2, width * 2))
            layers.append(nn.GELU())
        layers.append(nn.Linear(width * 2, width))
        self.net = nn.Sequential(*layers)

        self.register_buffer('rbf_centers', torch.linspace(0, 1, n_rbf))
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, t):
        """t: [N, 1] → Φ: [N, width]"""
        freqs = torch.arange(1, self.n_freq + 1, dtype=t.dtype, device=t.device)
        sin_f = torch.sin(2 * math.pi * freqs.unsqueeze(0) * t)
        cos_f = torch.cos(2 * math.pi * freqs.unsqueeze(0) * t)
        fourier = torch.cat([t, sin_f, cos_f], dim=-1)
        rbf = torch.exp(-50.0 * (t - self.rbf_centers.unsqueeze(0)) ** 2)
        poly = torch.cat([t, t**2, t**3], dim=-1)
        features = torch.cat([fourier, rbf, poly], dim=-1)
        return self.net(features)


# ============================================================
# Model A: Fixed Trunk + Cross-channel
# ============================================================
class FixedCrossModel(nn.Module):
    def __init__(self, seq_len=96, pred_len=96, width=64, hidden=256,
                 encoder_type='patch_attn', trunk_depth=4):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.width = width
        self.hidden = hidden

        # Encoder
        self.encoder = build_encoder(encoder_type, seq_len, hidden,
                                     seq_len=seq_len, depth=3,
                                     activation='gelu', dropout=0.1)

        # Cross-channel attention
        self.cross_attn = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=4, dim_feedforward=hidden * 2,
            dropout=0.1, activation='gelu', batch_first=True, norm_first=True
        )

        # Fixed Trunk (shared, deep)
        self.trunk = FixedTrunk(width=width, depth=trunk_depth)

        # Task heads: output B coefficients only
        self.forecast_head = nn.Linear(hidden, width)
        self.recon_head = nn.Linear(hidden, width)
        self.bias = nn.Parameter(torch.zeros(1))
        self.recon_bias = nn.Parameter(torch.zeros(1))

    def _encode(self, x_enc):
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
            z_ch = self.encoder(x_enc[:, :, ch])  # [B, hidden]
            z_list.append(z_ch)
        z_all = torch.stack(z_list, dim=1)  # [B, C, hidden]

        # Cross-channel attention
        z_all = self.cross_attn(z_all)  # [B, C, hidden]

        return z_all, means, stdev

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None,
                target_pred_len=None, **kwargs):
        if target_pred_len is None:
            target_pred_len = self.pred_len
        B, S, C = x_enc.shape

        z_all, means, stdev = self._encode(x_enc)

        # Trunk: fixed basis
        t = torch.linspace(0, 1, target_pred_len, dtype=x_enc.dtype,
                           device=x_enc.device).unsqueeze(-1)
        Phi = self.trunk(t)  # [pred_len, width]

        # Per-channel: B · Φ
        outputs = []
        for ch in range(C):
            B_coeff = self.forecast_head(z_all[:, ch, :])  # [B, width]
            out_ch = torch.einsum('bp,qp->bq', B_coeff, Phi) + self.bias
            outputs.append(out_ch)
        output = torch.stack(outputs, dim=-1)  # [B, pred_len, C]

        output = output * stdev + means
        return output

    def reconstruct(self, x_enc):
        B, S, C = x_enc.shape
        z_all, means, stdev = self._encode(x_enc)

        t = torch.linspace(0, 1, S, dtype=x_enc.dtype, device=x_enc.device).unsqueeze(-1)
        Phi = self.trunk(t)

        outputs = []
        for ch in range(C):
            B_coeff = self.recon_head(z_all[:, ch, :])
            out_ch = torch.einsum('bp,qp->bq', B_coeff, Phi) + self.recon_bias
            outputs.append(out_ch)
        output = torch.stack(outputs, dim=-1)

        output = output * stdev + means
        return output

    def get_representation(self, x_enc):
        z_all, _, _ = self._encode(x_enc)
        return z_all  # [B, C, hidden]


# ============================================================
# Model B: Transformer Encoder + Fixed Trunk
# ============================================================
class TransformerOperatorModel(nn.Module):
    """Full Transformer encoder but output through operator learning (Fixed Trunk)."""
    def __init__(self, seq_len=96, pred_len=96, width=64, hidden=256,
                 n_heads=4, n_layers=3, trunk_depth=4):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.width = width
        self.hidden = hidden

        # Patch embedding (like PatchTST)
        self.patch_size = 16
        self.n_patches = seq_len // self.patch_size
        self.patch_embed = nn.Linear(self.patch_size, hidden)
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches, hidden) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=n_heads, dim_feedforward=hidden * 4,
            dropout=0.1, activation='gelu', batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(hidden)

        # Cross-channel attention
        self.cross_attn = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=n_heads, dim_feedforward=hidden * 2,
            dropout=0.1, activation='gelu', batch_first=True, norm_first=True
        )

        # Fixed Trunk (deep)
        self.trunk = FixedTrunk(width=width, depth=trunk_depth)

        # Heads
        self.forecast_head = nn.Linear(hidden, width)
        self.recon_head = nn.Linear(hidden, width)
        self.bias = nn.Parameter(torch.zeros(1))
        self.recon_bias = nn.Parameter(torch.zeros(1))

    def _encode_channel(self, x_ch):
        """x_ch: [B, seq_len] → z: [B, hidden]"""
        B = x_ch.shape[0]
        patches = x_ch.reshape(B, self.n_patches, self.patch_size)
        tokens = self.patch_embed(patches) + self.pos_embed
        tokens = self.transformer(tokens)
        tokens = self.norm(tokens)
        z = tokens.mean(dim=1)  # [B, hidden]
        return z

    def _encode(self, x_enc):
        B, S, C = x_enc.shape

        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev

        z_list = []
        for ch in range(C):
            z_ch = self._encode_channel(x_enc[:, :, ch])
            z_list.append(z_ch)
        z_all = torch.stack(z_list, dim=1)  # [B, C, hidden]

        z_all = self.cross_attn(z_all)
        return z_all, means, stdev

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None,
                target_pred_len=None, **kwargs):
        if target_pred_len is None:
            target_pred_len = self.pred_len
        B, S, C = x_enc.shape

        z_all, means, stdev = self._encode(x_enc)

        t = torch.linspace(0, 1, target_pred_len, dtype=x_enc.dtype,
                           device=x_enc.device).unsqueeze(-1)
        Phi = self.trunk(t)

        outputs = []
        for ch in range(C):
            B_coeff = self.forecast_head(z_all[:, ch, :])
            out_ch = torch.einsum('bp,qp->bq', B_coeff, Phi) + self.bias
            outputs.append(out_ch)
        output = torch.stack(outputs, dim=-1)

        output = output * stdev + means
        return output

    def reconstruct(self, x_enc):
        B, S, C = x_enc.shape
        z_all, means, stdev = self._encode(x_enc)

        t = torch.linspace(0, 1, S, dtype=x_enc.dtype, device=x_enc.device).unsqueeze(-1)
        Phi = self.trunk(t)

        outputs = []
        for ch in range(C):
            B_coeff = self.recon_head(z_all[:, ch, :])
            out_ch = torch.einsum('bp,qp->bq', B_coeff, Phi) + self.recon_bias
            outputs.append(out_ch)
        output = torch.stack(outputs, dim=-1)
        output = output * stdev + means
        return output

    def get_representation(self, x_enc):
        z_all, _, _ = self._encode(x_enc)
        return z_all


# ============================================================
# Training & Evaluation Functions
# ============================================================
def train_forecasting(model, device, data, root, fpath, enc_in, pl):
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

    model.pred_len = pl
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
    return np.mean((preds - trues) ** 2)


def train_classification(model, device, ds_name):
    cls_root = './dataset/classification/Multivariate_ts'
    hidden = model.hidden

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
            bx = bx.float().to(device); label = label.long().to(device)
            z = model.get_representation(bx).mean(dim=1)
            loss = criterion(cls_head(z), label)
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
    return best_acc


def unified_pretrain(model, device, epochs=10, mask_rate=0.4):
    from data_provider.unified_dataset import UnifiedPretrainDataset

    dataset = UnifiedPretrainDataset(seq_len=96, stride=48)
    n_val = min(5000, len(dataset) // 10)
    train_ds, val_ds = random_split(dataset, [len(dataset) - n_val, n_val])
    train_dl = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=4, drop_last=True)

    optimizer = optim.Adam(model.parameters(), lr=0.0003)
    criterion = nn.MSELoss(reduction='none')

    best_val = float('inf')
    save_path = f'checkpoints/{model.__class__.__name__}_unified.pth'

    for epoch in range(epochs):
        model.train()
        losses = []
        t0 = time.time()
        for i, batch_x in enumerate(train_dl):
            batch_x = batch_x.float().to(device)
            mask = (torch.rand_like(batch_x) > mask_rate).float()
            masked = batch_x * mask
            optimizer.zero_grad()
            output = model.reconstruct(masked)
            loss_mat = criterion(output, batch_x)
            inv_mask = 1.0 - mask
            loss = (loss_mat * inv_mask).sum() / inv_mask.sum().clamp(min=1)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
            if (i+1) % 1000 == 0:
                print(f'  iter {i+1}/{len(train_dl)}: loss={loss.item():.4f}')

        tl = np.mean(losses)
        print(f'Epoch {epoch+1}: train={tl:.4f} ({time.time()-t0:.0f}s)')
        if tl < best_val:
            best_val = tl
            torch.save(model.state_dict(), save_path)

    model.load_state_dict(torch.load(save_path))
    return model


def eval_zeroshot_forecasting(model, device):
    datasets = {
        'ETTh1': ('ETTh1', './dataset/ETT-small/', 'ETTh1.csv', 7),
        'ETTh2': ('ETTh2', './dataset/ETT-small/', 'ETTh2.csv', 7),
        'Weather': ('custom', './dataset/weather/', 'weather.csv', 21),
        'Exchange': ('custom', './dataset/exchange_rate/', 'exchange_rate.csv', 8),
    }
    results = {}
    model.eval()
    for dname, (data, root, fpath, enc_in) in datasets.items():
        for pl in [96, 336]:
            args = SimpleNamespace(
                seq_len=96, pred_len=pl, label_len=48,
                data=data, root_path=root, data_path=fpath,
                features='M', target='OT', freq='h', embed='timeF',
                enc_in=enc_in, dec_in=enc_in, c_out=enc_in,
                num_workers=2, batch_size=1,
                exp_name='MTSF', ordered_data=False, data_amount=-1,
                combine_Gaussian_datasets=False,
                synthetic_data_path='', synthetic_root_path='./',
                synthetic_length=1024, stride=-1,
            )
            _, test_dl = data_provider(args, 'test')
            preds, trues = [], []
            with torch.no_grad():
                for bx, by, _, _ in test_dl:
                    bx = bx.float().to(device)
                    model.pred_len = pl
                    out = model(bx, target_pred_len=pl)
                    preds.append(out.cpu().numpy())
                    trues.append(by[:, -pl:, :].numpy())
            preds, trues = np.concatenate(preds), np.concatenate(trues)
            mse = np.mean((preds - trues) ** 2)
            results[f'{dname}_pl{pl}'] = mse
            print(f'  {dname}_pl{pl}: MSE={mse:.4f}')
    return results


def eval_zeroshot_classification(model, device):
    hidden = model.hidden
    results = {}

    for p in model.parameters():
        p.requires_grad = False

    for ds_name in ['Epilepsy', 'FingerMovements', 'BasicMotions', 'NATOPS', 'EthanolConcentration']:
        cls_root = './dataset/classification/Multivariate_ts'
        train_ds = Dataset_Classification(root_path=cls_root, flag='train', size=[96, 0, 96], data_path=ds_name)
        test_ds = Dataset_Classification(root_path=cls_root, flag='test', size=[96, 0, 96], data_path=ds_name)
        train_dl = DataLoader(train_ds, batch_size=16, shuffle=True, drop_last=True)
        test_dl = DataLoader(test_ds, batch_size=16, shuffle=False)

        cls_head = nn.Sequential(
            nn.Linear(hidden, 128), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(128, train_ds.n_classes)
        ).to(device)

        opt = optim.Adam(cls_head.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        best_acc = 0
        for epoch in range(30):
            cls_head.train()
            for bx, label, _, _ in train_dl:
                bx = bx.float().to(device); label = label.long().to(device)
                with torch.no_grad(): z = model.get_representation(bx).mean(dim=1)
                loss = criterion(cls_head(z), label)
                opt.zero_grad(); loss.backward(); opt.step()

            cls_head.eval()
            preds, labels = [], []
            with torch.no_grad():
                for bx, label, _, _ in test_dl:
                    bx = bx.float().to(device)
                    z = model.get_representation(bx).mean(dim=1)
                    preds.append(cls_head(z).argmax(-1).cpu().numpy())
                    labels.append(label.numpy())
            acc = accuracy_score(np.concatenate(labels), np.concatenate(preds))
            best_acc = max(best_acc, acc)

        results[ds_name] = best_acc
        print(f'  {ds_name}: Acc={best_acc:.4f}')

    for p in model.parameters():
        p.requires_grad = True
    return results


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['fixed_cross', 'transformer'], required=True)
    parser.add_argument('--mode', choices=['scratch', 'pretrain'], required=True)
    args = parser.parse_args()

    device = torch.device('cuda')

    # Build model
    if args.model == 'fixed_cross':
        model = FixedCrossModel(seq_len=96, pred_len=96, width=64, hidden=256,
                                encoder_type='patch_attn', trunk_depth=4).to(device)
    else:
        model = TransformerOperatorModel(seq_len=96, pred_len=96, width=64, hidden=256,
                                          n_heads=4, n_layers=3, trunk_depth=4).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f'Model: {args.model}, Mode: {args.mode}, Params: {n_params:,}')

    if args.mode == 'scratch':
        # From-scratch per dataset
        print('\n=== FORECASTING (from-scratch) ===')
        datasets = {
            'ETTh1': ('ETTh1', './dataset/ETT-small/', 'ETTh1.csv', 7),
            'ETTh2': ('ETTh2', './dataset/ETT-small/', 'ETTh2.csv', 7),
            'Weather': ('custom', './dataset/weather/', 'weather.csv', 21),
            'Exchange': ('custom', './dataset/exchange_rate/', 'exchange_rate.csv', 8),
        }
        fc_results = {}
        for dname, (data, root, fpath, enc_in) in datasets.items():
            for pl in [96, 336]:
                # Fresh model each time
                if args.model == 'fixed_cross':
                    m = FixedCrossModel(seq_len=96, pred_len=pl, width=64, hidden=256,
                                        encoder_type='patch_attn', trunk_depth=4).to(device)
                else:
                    m = TransformerOperatorModel(seq_len=96, pred_len=pl, width=64, hidden=256,
                                                  n_heads=4, n_layers=3, trunk_depth=4).to(device)
                mse = train_forecasting(m, device, data, root, fpath, enc_in, pl)
                fc_results[f'{dname}_pl{pl}'] = mse
                print(f'  {dname}_pl{pl}: MSE={mse:.4f}')

        print('\n=== CLASSIFICATION (from-scratch) ===')
        cls_results = {}
        for ds_name in ['Epilepsy', 'FingerMovements', 'BasicMotions', 'NATOPS', 'EthanolConcentration']:
            if args.model == 'fixed_cross':
                m = FixedCrossModel(seq_len=96, pred_len=96, width=64, hidden=256,
                                    encoder_type='patch_attn', trunk_depth=4).to(device)
            else:
                m = TransformerOperatorModel(seq_len=96, pred_len=96, width=64, hidden=256,
                                              n_heads=4, n_layers=3, trunk_depth=4).to(device)
            acc = train_classification(m, device, ds_name)
            cls_results[ds_name] = acc
            print(f'  {ds_name}: Acc={acc:.4f}')

        print(f'\n=== SUMMARY: {args.model} from-scratch ===')
        for k, v in {**fc_results, **cls_results}.items():
            print(f'  {k}: {v:.4f}')

    else:  # pretrain
        print('\n=== UNIFIED PRE-TRAINING ===')
        model = unified_pretrain(model, device, epochs=10)

        print('\n=== FORECASTING (zero-shot) ===')
        fc_results = eval_zeroshot_forecasting(model, device)

        print('\n=== CLASSIFICATION (frozen + cls_head) ===')
        cls_results = eval_zeroshot_classification(model, device)

        print(f'\n=== SUMMARY: {args.model} pretrain ===')
        for k, v in {**fc_results, **cls_results}.items():
            print(f'  {k}: {v:.4f}')
