"""
Single Expert v2: 크기 키움 (75M) + next-token + spectral + cross-channel
+ 매 epoch마다 실제 downstream zero-shot eval (forecasting, imputation, classification)
+ 개선된 학습: lr=0.001, batch=256, warmup

사용법:
  CUDA_VISIBLE_DEVICES=1 python experiments/exp_single_expert_v2.py 2>&1 | tee log/scaleup/single_expert_v2.log
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
from sklearn.metrics import accuracy_score
import time

from model.encoders import build_encoder
from data_provider.pile_dataset import PilePretrainDataset
from data_provider.data_factory import data_provider
from data_provider.data_loader import Dataset_Classification


class SingleExpertModelV2(nn.Module):
    """Single Expert (No MoE) + PatchAttn + Cross-channel + Hyper Trunk + Spectral
    크기: width=256, hidden=1024 → ~75M
    """
    def __init__(self, seq_len=96, pred_len=96, width=256, hidden=1024,
                 trunk_depth=2, n_freq=32, n_rbf=16):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.width = width
        self.hidden = hidden
        self.branch_hidden = hidden
        self.n_freq = n_freq
        self.n_rbf = n_rbf

        # Branch input: x_ch + FFT
        n_fft = (seq_len // 2 + 1) * 2
        branch_dim = seq_len + n_fft
        self.branch_dim = branch_dim

        # Encoder (deeper for larger model)
        self.encoder = build_encoder('patch_attn', branch_dim, hidden,
                                     seq_len=seq_len, depth=4,
                                     activation='gelu', dropout=0.1)

        # Cross-channel Attention
        self.cross_attn = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=8, dim_feedforward=hidden * 4,
            dropout=0.1, activation='gelu', batch_first=True, norm_first=True
        )

        # Trunk
        trunk_input_dim = 1 + 2 * n_freq + n_rbf + 3
        self.register_buffer('rbf_centers', torch.linspace(0, 1, n_rbf))

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
        x_fft = torch.fft.rfft(x_ch, dim=-1)
        x_spectral = torch.cat([x_fft.real, x_fft.imag], dim=-1)
        return torch.cat([x_ch, x_spectral], dim=-1)

    def _trunk_forward(self, trunk_params, B_coeff, target_len, bias):
        batch_size = trunk_params.shape[0]
        params = trunk_params * 0.01
        trunk_weights = []
        idx = 0
        for in_dim, out_dim, bias_size in self.trunk_param_shapes:
            w_size = in_dim * out_dim
            w = params[:, idx:idx+w_size].view(batch_size, in_dim, out_dim)
            idx += w_size
            b = params[:, idx:idx+bias_size].view(batch_size, out_dim)
            idx += bias_size
            trunk_weights.append((w, b))

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
        B, S, C = x_enc.shape
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev

        z_list = []
        for ch in range(C):
            branch_input = self._build_branch_input(x_enc[:, :, ch])
            z_ch = self.encoder(branch_input)
            z_list.append(z_ch)
        z_all = torch.stack(z_list, dim=1)

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
            tp = head_out[:, :self.trunk_param_count]
            bc = head_out[:, self.trunk_param_count:]
            outputs.append(self._trunk_forward(tp, bc, target_pred_len, self.bias))
        output = torch.stack(outputs, dim=-1)
        return output * stdev + means

    def reconstruct(self, x_enc):
        B, S, C = x_enc.shape
        z_all, means, stdev = self._encode_all_channels(x_enc)
        outputs = []
        for ch in range(C):
            head_out = self.recon_head(z_all[:, ch, :])
            tp = head_out[:, :self.trunk_param_count]
            bc = head_out[:, self.trunk_param_count:]
            outputs.append(self._trunk_forward(tp, bc, S, self.recon_bias))
        output = torch.stack(outputs, dim=-1)
        return output * stdev + means

    def get_representation(self, x_enc):
        z_all, _, _ = self._encode_all_channels(x_enc)
        return z_all


# ============================================================
# Quick eval (매 epoch 후 빠르게 확인)
# ============================================================
def quick_eval(model, device, epoch):
    """매 epoch 후 핵심 데이터셋 3개만 빠르게 eval."""
    model.eval()
    results = {}

    # 1. Forecasting: ETTh1 pl=96 (빠름)
    args = SimpleNamespace(
        seq_len=96, pred_len=96, label_len=48,
        data='ETTh1', root_path='./dataset/ETT-small/', data_path='ETTh1.csv',
        features='M', target='OT', freq='h', embed='timeF',
        enc_in=7, dec_in=7, c_out=7, num_workers=2, batch_size=1,
        exp_name='MTSF', ordered_data=False, data_amount=-1,
        combine_Gaussian_datasets=False, synthetic_data_path='', synthetic_root_path='./',
        synthetic_length=1024, stride=-1,
    )
    _, test_dl = data_provider(args, 'test')
    preds, trues = [], []
    with torch.no_grad():
        for bx, by, _, _ in test_dl:
            bx = bx.float().to(device)
            out = model(bx, None, None, None, target_pred_len=96)
            if isinstance(out, tuple): out = out[0]
            preds.append(out.cpu().numpy())
            trues.append(by[:, -96:, :].numpy())
    p, t = np.concatenate(preds), np.concatenate(trues)
    results['FC_ETTh1'] = np.mean((p - t) ** 2)

    # 2. Imputation: ETTh1 m=0.125
    torch.manual_seed(2021)
    preds, trues, masks = [], [], []
    with torch.no_grad():
        for bx, by, _, _ in test_dl:
            bx = bx.float().to(device)
            mask = (torch.rand_like(bx) > 0.125).float()
            out = model.reconstruct(bx * mask)
            preds.append(out.cpu().numpy()); trues.append(bx.cpu().numpy()); masks.append(mask.cpu().numpy())
    p, t, m = np.concatenate(preds), np.concatenate(trues), np.concatenate(masks)
    results['IMP_m0125'] = np.mean((p[m == 0] - t[m == 0]) ** 2)

    # 3. Classification: Epilepsy (빠름)
    cls_root = './dataset/classification/Multivariate_ts'
    try:
        train_ds = Dataset_Classification(root_path=cls_root, flag='train', size=[96, 0, 96], data_path='Epilepsy')
        test_ds = Dataset_Classification(root_path=cls_root, flag='test', size=[96, 0, 96], data_path='Epilepsy')
        train_dl = DataLoader(train_ds, batch_size=16, shuffle=True, drop_last=True)
        test_dl_cls = DataLoader(test_ds, batch_size=16, shuffle=False)

        hidden = model.hidden
        cls_head = nn.Sequential(nn.Linear(hidden, 256), nn.GELU(), nn.Dropout(0.1),
                                 nn.Linear(256, train_ds.n_classes)).to(device)
        opt = optim.Adam(cls_head.parameters(), lr=0.001)

        for p_model in model.parameters(): p_model.requires_grad = False
        for ep in range(10):  # quick 10 epochs
            cls_head.train()
            for bx, label, _, _ in train_dl:
                bx = bx.float().to(device); label = label.long().to(device)
                with torch.no_grad(): z = model.get_representation(bx).mean(dim=1)
                loss = nn.CrossEntropyLoss()(cls_head(z), label)
                opt.zero_grad(); loss.backward(); opt.step()

        cls_head.eval()
        ps, ls = [], []
        with torch.no_grad():
            for bx, label, _, _ in test_dl_cls:
                bx = bx.float().to(device)
                z = model.get_representation(bx).mean(dim=1)
                ps.append(cls_head(z).argmax(-1).cpu().numpy()); ls.append(label.numpy())
        results['CLS_Epilepsy'] = accuracy_score(np.concatenate(ls), np.concatenate(ps))
        for p_model in model.parameters(): p_model.requires_grad = True
    except:
        results['CLS_Epilepsy'] = 0.0

    model.train()
    return results


# ============================================================
# Pre-training
# ============================================================
def pretrain(model, device, save_path, epochs=10, peak_lr=0.001, mask_rate=0.4):
    print(f'\n{"="*60}')
    print('Single Expert V2: 75M + Cross + Spectral + Next-token')
    print(f'  lr={peak_lr}, batch=256, warmup=2000')
    print(f'{"="*60}')

    dataset = PilePretrainDataset(seq_len=96, stride=48,
                                   pile_root='./dataset/time_series_pile')
    n_val = min(10000, len(dataset) // 10)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])
    train_dl = DataLoader(train_ds, batch_size=256, shuffle=True,
                          num_workers=4, drop_last=True, pin_memory=True)

    optimizer = optim.AdamW(model.parameters(), lr=peak_lr, weight_decay=0.01)

    # Warmup + cosine decay
    total_steps = epochs * len(train_dl)
    warmup_steps = 2000

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    recon_criterion = nn.MSELoss(reduction='none')

    print(f'Train: {n_train}, Steps/epoch: {len(train_dl)}, Total: {total_steps}')

    best_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        losses, nt_losses, recon_losses = [], [], []
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

            # 2) Next-token
            split = torch.randint(24, 72, (1,)).item()
            context = batch_x[:, :split, :]
            target = batch_x[:, split:, :]
            target_len = S - split
            context_padded = F.pad(context, (0, 0, 0, S - split))
            nt_out = model(context_padded, target_pred_len=target_len)
            if isinstance(nt_out, tuple): nt_out = nt_out[0]
            nt_loss = F.mse_loss(nt_out, target)

            # Weighted: nt에 집중
            loss = 2.0 * nt_loss + 0.5 * recon_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            losses.append(loss.item())
            nt_losses.append(nt_loss.item())
            recon_losses.append(recon_loss.item())

            if (i + 1) % 200 == 0:
                lr = scheduler.get_last_lr()[0]
                print(f'  iter {i+1}/{len(train_dl)}: nt={np.mean(nt_losses[-200:]):.4f} recon={np.mean(recon_losses[-200:]):.4f} lr={lr:.6f}')

        train_loss = np.mean(losses)
        print(f'Epoch {epoch+1}/{epochs}: loss={train_loss:.4f} (nt={np.mean(nt_losses):.4f} recon={np.mean(recon_losses):.4f}) ({time.time()-t0:.0f}s)')

        if train_loss < best_loss:
            best_loss = train_loss
            torch.save(model.state_dict(), save_path)
            print(f'  Saved checkpoint')

        # 매 epoch downstream eval
        print(f'  --- Quick Eval (epoch {epoch+1}) ---')
        qr = quick_eval(model, device, epoch + 1)
        print(f'  FC_ETTh1={qr["FC_ETTh1"]:.4f}  IMP={qr["IMP_m0125"]:.4f}  CLS_Epi={qr["CLS_Epilepsy"]:.4f}')

    model.load_state_dict(torch.load(save_path))
    return model


if __name__ == '__main__':
    from experiments.eval_all_tasks import eval_forecasting, eval_imputation, eval_classification, eval_short_term, print_summary

    device = torch.device('cuda')

    model = SingleExpertModelV2(
        seq_len=96, pred_len=96,
        width=256, hidden=1024,
        trunk_depth=2,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f'Single Expert V2: {n_params/1e6:.1f}M params')
    print(f'  width=256, hidden=1024, PatchAttn, Cross-channel, Spectral, Hyper Trunk')

    save_path = 'checkpoints/single_expert_v2.pth'
    os.makedirs('checkpoints', exist_ok=True)

    model = pretrain(model, device, save_path, epochs=10, peak_lr=0.001)

    # Full eval
    print('\n' + '='*60)
    print('Full Evaluation')
    print('='*60)
    fc = eval_forecasting(model, device)
    st = eval_short_term(model, device)
    imp = eval_imputation(model, device)
    cls = eval_classification(model, device)
    print_summary(fc, imp, cls, st, f'Single Expert V2 ({n_params/1e6:.1f}M)')
