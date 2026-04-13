"""
v3: Informed Query DeepONet — Full Pre-training + 5 Task Eval

핵심 혁신:
  Query point에 FFT 주파수/위상을 직접 넣어서 trunk이 시간/주파수를 앎
  → 주파수 보존이 구조적으로 보장 (학습 불필요)
  → 고주파 corr: 63M MoE=-0.06 → v3=0.46 (Period=6)
  → 위상 구분: 63M MoE≈0 → v3=0.84

구조:
  Input → FFT(비학습) → freq, phase, last_val, last_slope
  Input → Encoder(학습) → z → head(z) → [trunk_params, B]
  Query = [t_real, sin(2π·freq·t+phase), cos(...), last_val, last_slope]
  Trunk(query; trunk_params) → Φ
  Output = B · Φ

Loss: MSE + Spectral Loss
Data: Pile(Real) + TempoPFN(Synth), 20 epochs

사용법:
  CUDA_VISIBLE_DEVICES=X python experiments/exp_v3_informed_query_pretrain.py
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
from model.DeepONetHyperMoE import build_encoder

DEVICE = torch.device('cuda')


# ============================================================
# Non-learned signal processing
# ============================================================
def extract_freq_info(x_ch, top_k=5):
    """FFT로 주파수/위상/경계 추출 (비학습)."""
    fft = torch.fft.rfft(x_ch, dim=-1)
    magnitude = fft.abs()
    magnitude[:, 0] = 0
    top_k_idx = torch.topk(magnitude, top_k, dim=-1).indices
    top_k_phase = torch.angle(torch.gather(fft, 1, top_k_idx))
    freqs = top_k_idx.float()
    last_val = x_ch[:, -1:].detach()
    last_slope = (x_ch[:, -1:] - x_ch[:, -2:-1]).detach()
    return freqs, top_k_phase, last_val, last_slope


def build_informed_query(t, freqs, phases, last_val, last_slope):
    """Query features: [t, sin(2π·f·t+φ), cos(...), last_val, last_slope]."""
    B = freqs.shape[0]
    T = t.shape[0]
    t_exp = t.squeeze(-1).unsqueeze(0).unsqueeze(-1)
    f = freqs.unsqueeze(1)
    p = phases.unsqueeze(1)
    angle = 2 * math.pi * f * t_exp + p
    sin_feat = torch.sin(angle)
    cos_feat = torch.cos(angle)
    t_broadcast = t.squeeze(-1).unsqueeze(0).expand(B, -1)
    lv = last_val.unsqueeze(1).expand(-1, T, -1)
    ls = last_slope.unsqueeze(1).expand(-1, T, -1)
    return torch.cat([t_broadcast.unsqueeze(-1), sin_feat, cos_feat, lv, ls], dim=-1)


def spectral_loss(pred, target):
    """FFT domain loss."""
    pred_fft = torch.fft.rfft(pred, dim=1)
    target_fft = torch.fft.rfft(target, dim=1)
    mag_loss = F.mse_loss(pred_fft.abs(), target_fft.abs())
    weights = target_fft.abs().detach() + 1e-8
    phase_diff = torch.angle(pred_fft) - torch.angle(target_fft)
    phase_loss = (weights * (1 - torch.cos(phase_diff))).mean()
    return mag_loss + 0.1 * phase_loss


# ============================================================
# Model
# ============================================================
class InformedQueryDeepONet(nn.Module):
    def __init__(self, seq_len=96, pred_len=96, width=96, branch_hidden=384,
                 trunk_depth=2, top_k_freq=5, spectral_branch=True, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.width = width
        self.top_k_freq = top_k_freq
        self.spectral_branch = spectral_branch
        self.branch_hidden = branch_hidden
        self.use_norm = True

        self.query_dim = 1 + 2 * top_k_freq + 2

        # Trunk param shapes
        trunk_param_count = 0
        trunk_param_shapes = []
        trunk_param_count += self.query_dim * width + width
        trunk_param_shapes.append((self.query_dim, width, width))
        for _ in range(2, trunk_depth):
            trunk_param_count += width * width + width
            trunk_param_shapes.append((width, width, width))
        self.trunk_param_count = trunk_param_count
        self.trunk_param_shapes = trunk_param_shapes

        branch_dim = seq_len * 2
        if spectral_branch:
            branch_dim += (seq_len // 2 + 1) * 2

        self.encoder = build_encoder('patch_attn', branch_dim, branch_hidden,
                                     seq_len=seq_len, depth=4,
                                     activation='gelu', dropout=dropout)

        forecast_output_dim = trunk_param_count + width
        self.forecast_head = nn.Linear(branch_hidden, forecast_output_dim)
        self.recon_head = nn.Linear(branch_hidden, forecast_output_dim)
        for h in [self.forecast_head, self.recon_head]:
            nn.init.xavier_normal_(h.weight, gain=0.1)
            nn.init.constant_(h.bias, 0)

        self.bias = nn.Parameter(torch.zeros([1]))
        self.recon_bias = nn.Parameter(torch.zeros([1]))

    def _build_branch_input(self, x_ch, x_cross):
        branch_input = torch.cat([x_ch, x_cross], dim=-1)
        if self.spectral_branch:
            x_fft = torch.fft.rfft(x_ch, dim=-1)
            branch_input = torch.cat([branch_input, x_fft.real, x_fft.imag], dim=-1)
        return branch_input

    def _trunk_forward(self, head_output, query, bias):
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
        Phi = query
        for i, (w, b) in enumerate(trunk_weights):
            Phi = torch.bmm(Phi, w) + b.unsqueeze(1)
            if i < len(trunk_weights) - 1:
                Phi = F.gelu(Phi)
        return torch.einsum('bp,bqp->bq', B_coeff, Phi) + bias

    def _forward_channel(self, x_ch, x_cross, target_len, mode='forecast'):
        branch_input = self._build_branch_input(x_ch, x_cross)
        z = self.encoder(branch_input)
        freqs, phases, last_val, last_slope = extract_freq_info(x_ch, self.top_k_freq)

        if mode == 'forecast':
            t = torch.linspace(1.0, 1.0 + target_len / self.seq_len, target_len,
                               device=x_ch.device, dtype=x_ch.dtype).unsqueeze(-1)
        else:
            t = torch.linspace(0.0, 1.0, target_len,
                               device=x_ch.device, dtype=x_ch.dtype).unsqueeze(-1)

        query = build_informed_query(t, freqs, phases, last_val, last_slope)
        head = self.forecast_head if mode == 'forecast' else self.recon_head
        bias = self.bias if mode == 'forecast' else self.recon_bias
        return self._trunk_forward(head(z), query, bias)

    def get_representation(self, x_enc):
        B, S, C = x_enc.shape
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = (x_enc - means) / torch.sqrt(
                torch.var(x_enc - means, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_cross = x_enc.mean(dim=-1)
        reps = []
        for ch in range(C):
            bi = self._build_branch_input(x_enc[:, :, ch], x_cross)
            reps.append(self.encoder(bi))
        return torch.stack(reps, dim=1)

    def forecast(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None,
                 target_pred_len=None, **kwargs):
        if target_pred_len is None:
            target_pred_len = self.pred_len
        B, S, C = x_enc.shape
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev
        x_cross = x_enc.mean(dim=-1)
        outputs = []
        for ch in range(C):
            outputs.append(self._forward_channel(x_enc[:, :, ch], x_cross,
                                                 target_pred_len, 'forecast'))
        output = torch.stack(outputs, dim=-1)
        return output * stdev + means

    def reconstruct(self, x_enc):
        B, S, C = x_enc.shape
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev
        x_cross = x_enc.mean(dim=-1)
        outputs = []
        for ch in range(C):
            outputs.append(self._forward_channel(x_enc[:, :, ch], x_cross, S, 'recon'))
        output = torch.stack(outputs, dim=-1)
        return output * stdev + means

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None,
                target_pred_len=None, **kwargs):
        return self.forecast(x_enc, target_pred_len=target_pred_len)


# ============================================================
# Synthetic Dataset
# ============================================================
class SyntheticWindowDataset(Dataset):
    def __init__(self, arrow_path, seq_len=96, stride=48, max_samples=100000):
        self.windows = []
        ds = Dataset_GaussianPCoregionalization(
            root_path='./', data_path=arrow_path,
            n_variables=160, seq_len=seq_len, pred_len=seq_len,
            size=[seq_len, 0, seq_len], synthetic_length=1024, stride=stride)
        for i in range(min(len(ds), max_samples)):
            x, y, _, _ = ds[i]
            if isinstance(x, torch.Tensor): x = x.numpy()
            if x.ndim == 1: x = x.reshape(-1, 1)
            ch = np.random.randint(0, x.shape[1])
            window = x[:, ch].astype(np.float32)
            std = np.std(window)
            if std > 1e-8:
                self.windows.append(np.clip((window - np.mean(window)) / std, -10, 10))
        self.windows = np.array(self.windows, dtype=np.float32)
        print(f'SyntheticWindowDataset: {len(self.windows)} windows')
    def __len__(self): return len(self.windows)
    def __getitem__(self, idx):
        return torch.tensor(self.windows[idx], dtype=torch.float32).unsqueeze(-1)


# ============================================================
# Pre-training
# ============================================================
def pretrain(model, device, save_path, epochs=20, lr=0.0003, mask_rate=0.4,
             spectral_weight=0.01):
    print(f'\n{"="*60}')
    print(f'Pre-training: v3 Informed Query DeepONet')
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
    train_ds, _ = random_split(combined, [n_train, n_val])
    train_dl = DataLoader(train_ds, batch_size=128, shuffle=True,
                          num_workers=4, drop_last=True, pin_memory=True)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    print(f'Train: {n_train}, Steps/epoch: {len(train_dl)}')

    best_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        losses, recon_ls, nt_ls, spec_ls = [], [], [], []
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

            # 3) Spectral loss
            spec_r = spectral_loss(recon_out.squeeze(-1), batch_x.squeeze(-1))
            spec_n = spectral_loss(nt_out.squeeze(-1), target.squeeze(-1))
            spec = spec_r + spec_n

            loss = recon_loss + nt_loss + spectral_weight * spec
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            losses.append(loss.item())
            recon_ls.append(recon_loss.item())
            nt_ls.append(nt_loss.item())
            spec_ls.append(spec.item())

            if (i + 1) % 500 == 0:
                print(f'  iter {i+1}/{len(train_dl)}: '
                      f'recon={np.mean(recon_ls[-500:]):.4f} '
                      f'nt={np.mean(nt_ls[-500:]):.4f} '
                      f'spec={np.mean(spec_ls[-500:]):.4f}')

        scheduler.step()
        elapsed = time.time() - t0
        print(f'Epoch {epoch+1}/{epochs}: loss={np.mean(losses):.4f} '
              f'(recon={np.mean(recon_ls):.4f} nt={np.mean(nt_ls):.4f} '
              f'spec={np.mean(spec_ls):.4f}) '
              f'lr={scheduler.get_last_lr()[0]:.6f} ({elapsed:.0f}s)')

        if np.mean(losses) < best_loss:
            best_loss = np.mean(losses)
            torch.save(model.state_dict(), save_path)
            print(f'  Saved (best={best_loss:.4f})')

    model.load_state_dict(torch.load(save_path))
    return model


# ============================================================
# Eval functions (same as v1)
# ============================================================
def eval_forecasting(model, device):
    print(f'\n{"="*60}')
    print('Forecasting Eval')
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
            key = f'{dname}_{pl}'
            mlp = moment_lp.get(key, None)
            gap = f'{(mse/mlp-1)*100:+.1f}%' if mlp else '-'
            print(f'  {key}: MSE={mse:.4f}  MOMENT_LP={mlp or "-"}  gap={gap}')


def eval_imputation(model, device):
    print(f'\n{"="*60}')
    print('Imputation Eval')
    print(f'{"="*60}')
    datasets = {
        'ETTh1': ('ETTh1', './dataset/ETT-small/', 'ETTh1.csv', 7),
        'ETTh2': ('ETTh2', './dataset/ETT-small/', 'ETTh2.csv', 7),
        'ETTm1': ('ETTm1', './dataset/ETT-small/', 'ETTm1.csv', 7),
        'ETTm2': ('ETTm2', './dataset/ETT-small/', 'ETTm2.csv', 7),
        'Weather': ('custom', './dataset/weather/', 'weather.csv', 21),
    }
    moment = {'ETTh1': (0.402, 0.139), 'ETTh2': (0.125, 0.061),
              'ETTm1': (0.202, 0.074), 'ETTm2': (0.078, 0.031),
              'Weather': (0.082, 0.035)}
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
        avg = np.mean(all_mse)
        m0, mlp = moment.get(dname, (None, None))
        print(f'  {dname}: Mean MSE={avg:.4f}  (MOMENT_0={m0}, LP={mlp})')


def eval_classification(model, device):
    print(f'\n{"="*60}')
    print('Classification Eval')
    print(f'{"="*60}')
    hidden = model.branch_hidden
    for p in model.parameters(): p.requires_grad = False
    for ds_name in ['Epilepsy', 'FingerMovements', 'BasicMotions',
                    'NATOPS', 'EthanolConcentration']:
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
                nn.Linear(256, train_ds.n_classes)).to(device)
            opt = optim.Adam(cls_head.parameters(), lr=0.001)
            best_acc = 0
            for epoch in range(30):
                cls_head.train()
                for bx, label, _, _ in train_dl:
                    bx = bx.float().to(device); label = label.long().to(device)
                    with torch.no_grad(): z = model.get_representation(bx).mean(dim=1)
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
    for p in model.parameters(): p.requires_grad = True


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    print('=' * 60)
    print('v3: Informed Query DeepONet — Full Pre-training')
    print('  Query = [t_real, sin/cos(FFT_freq+phase), last_val, last_slope]')
    print('  Loss = MSE + Spectral Loss')
    print('  Data = Pile + TempoPFN Synth, 20 epochs')
    print('=' * 60)

    model = InformedQueryDeepONet(
        seq_len=96, pred_len=96, width=96, branch_hidden=384,
        trunk_depth=2, top_k_freq=5, spectral_branch=True, dropout=0.1
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    n_enc = sum(p.numel() for p in model.encoder.parameters())
    print(f'\nModel: {n_params/1e6:.1f}M total (encoder={n_enc/1e6:.1f}M)')
    print(f'Query dim: {model.query_dim}')

    save_path = 'checkpoints/v3_informed_query.pth'
    os.makedirs('checkpoints', exist_ok=True)

    model = pretrain(model, DEVICE, save_path, epochs=20, lr=0.0003, spectral_weight=0.01)
    eval_forecasting(model, DEVICE)
    eval_imputation(model, DEVICE)
    eval_classification(model, DEVICE)

    print(f'\n{"="*60}')
    print('ALL DONE')
    print(f'{"="*60}')
