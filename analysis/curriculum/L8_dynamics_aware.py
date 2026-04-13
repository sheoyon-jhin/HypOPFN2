"""
Curriculum L8: Dynamics-aware Encoder

핵심 아이디어:
  Encoder가 input의 dynamics를 명시적으로 추출
  - Windowed FFT (frequency over time)
  - Windowed slope (trend over time)
  - Windowed amplitude (envelope)
  - Local recent context

이렇게 하면 모델이 "주기가 어떻게 변하는지", "스케일이 어떻게 변하는지" 직접 봄.

비교:
  Baseline (L7 v3): Input decomp + MLP encoder
  L8:               Input decomp + Dynamics-aware encoder

데이터: 같은 chirp-dominant (L7 v2/v3와 동일)

CUDA_VISIBLE_DEVICES=3 python analysis/curriculum/L8_dynamics_aware.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, math, time
from torch import optim
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAVE_DIR = 'analysis/figures/curriculum'; os.makedirs(SAVE_DIR, exist_ok=True)

SEQ_LEN = 192
PRED_LEN = 192
TOTAL_LEN = 384
W = 64
H = 192
N_QUERY = 32
N_WINDOWS = 4   # for windowed feature extraction
TOP_K_FFT = 3


# Reuse trunks and decomposition from L7 v2/v3
from analysis.curriculum.L7_v2_v3 import (
    HTrunk, hfwd, decompose_3way, decompose_4way,
    moving_average, fft_topk_filter, fft_bandpass,
    gen_sample, make_dataset, make_random_configs,
    TRAIN_CONFIGS, OOD_CONFIGS, train_model, eval_model,
    Baseline, L7V3,
)


# ============================================================
# Dynamics feature extraction
# ============================================================
def extract_dynamics_features(x_component, n_windows=N_WINDOWS, top_k=TOP_K_FFT):
    """
    x_component: [B, L]
    Returns rich features showing how things change over time.
    """
    B, L = x_component.shape
    feats = []

    # 1. Windowed dominant frequencies (chirp dynamics)
    win_size = L // n_windows
    for i in range(n_windows):
        chunk = x_component[:, i*win_size:(i+1)*win_size]
        fft = torch.fft.rfft(chunk, dim=-1)
        mag = fft.abs()
        if mag.shape[-1] > 1:
            mag[:, 0] = 0
        # Top-k indices
        k = min(top_k, mag.shape[-1])
        _, top_idx = torch.topk(mag, k, dim=-1)
        feats.append(top_idx.float())  # [B, k]

    # 2. Windowed slope (trend dynamics)
    for i in range(n_windows):
        chunk = x_component[:, i*win_size:(i+1)*win_size]
        chunk_len = chunk.shape[-1]
        t = torch.arange(chunk_len, device=x_component.device, dtype=x_component.dtype) / chunk_len
        t_centered = t - t.mean()
        c_centered = chunk - chunk.mean(dim=-1, keepdim=True)
        slope = (c_centered * t_centered.unsqueeze(0)).sum(dim=-1, keepdim=True) / (t_centered**2).sum().clamp(min=1e-6)
        feats.append(slope)  # [B, 1]

    # 3. Windowed std (amplitude/scale dynamics)
    for i in range(n_windows):
        chunk = x_component[:, i*win_size:(i+1)*win_size]
        feats.append(chunk.std(dim=-1, keepdim=True))  # [B, 1]

    # 4. Global features
    feats.append(x_component.mean(dim=-1, keepdim=True))
    feats.append(x_component.std(dim=-1, keepdim=True))

    # 5. Last values + diffs (recent context)
    feats.append(x_component[:, -5:])
    feats.append(x_component[:, -1:] - x_component[:, -2:-1])
    feats.append(x_component[:, -2:-1] - x_component[:, -3:-2])

    return torch.cat(feats, dim=-1)  # [B, total_feat_dim]


def get_feature_dim(seq_len=SEQ_LEN, n_windows=N_WINDOWS, top_k=TOP_K_FFT):
    """Compute total feature dim."""
    win_freq = n_windows * top_k       # windowed FFT indices
    win_slope = n_windows              # windowed slopes
    win_amp = n_windows                # windowed std
    global_stats = 2                   # mean, std
    last_vals = 5 + 1 + 1              # last 5 + 2 diffs
    return win_freq + win_slope + win_amp + global_stats + last_vals


FEAT_DIM = get_feature_dim()


# ============================================================
# Dynamics-aware Encoder
# ============================================================
class DynamicsAwareEnc(nn.Module):
    def __init__(self):
        super().__init__()
        # Main MLP path
        self.mlp = nn.Sequential(
            nn.Linear(SEQ_LEN, H), nn.GELU(),
            nn.Linear(H, H), nn.GELU())
        # Feature path
        self.feat_proj = nn.Sequential(
            nn.Linear(FEAT_DIM, H // 2), nn.GELU(),
            nn.Linear(H // 2, H // 2), nn.GELU())
        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(H + H // 2, H), nn.GELU(),
            nn.Linear(H, H))

    def forward(self, x):
        z_raw = self.mlp(x)
        z_feat = self.feat_proj(extract_dynamics_features(x))
        z = self.fusion(torch.cat([z_raw, z_feat], dim=-1))
        return z


class TrunkBlockDyn(nn.Module):
    """Encoder (dynamics-aware) + one or more trunks."""
    def __init__(self, trunk_types):
        super().__init__()
        self.enc = DynamicsAwareEnc()
        self.trunks = nn.ModuleList([HTrunk(W, t) for t in trunk_types])
        self.heads = nn.ModuleList([nn.Linear(H, t.odim) for t in self.trunks])
        for h in self.heads: nn.init.xavier_normal_(h.weight, gain=0.1)
        self.biases = nn.ParameterList([nn.Parameter(torch.zeros(1)) for _ in trunk_types])

    def forward(self, x_part, qt):
        z = self.enc(x_part)
        t_flat = qt.reshape(-1)
        return sum(hfwd(tr, t_flat, h(z)) + b
                   for tr, h, b in zip(self.trunks, self.heads, self.biases))


class L8DynamicsAware(nn.Module):
    """L7 v3 (4-way decomp) + dynamics-aware encoders."""
    def __init__(self):
        super().__init__()
        self.block_T = TrunkBlockDyn(['poly'])
        self.block_S = TrunkBlockDyn(['fourier'])
        self.block_C = TrunkBlockDyn(['chirplet'])
        self.block_R = TrunkBlockDyn(['rbf'])

    def _q(self, ctx, qt):
        trend, season, cycle, resid = decompose_4way(ctx)
        return (self.block_T(trend, qt) + self.block_S(season, qt) +
                self.block_C(cycle, qt) + self.block_R(resid, qt))

    def forward_train(self, ctx, qt): return self._q(ctx, qt)
    def forecast(self, ctx, n=PRED_LEN):
        t = torch.linspace(1, 2, n, device=ctx.device).unsqueeze(0).expand(ctx.shape[0], -1)
        return self._q(ctx, t)


# ============================================================
# Run helper
# ============================================================
def run(label, model_cls, train_data, id_data, ood_data):
    print(f'\n{"="*60}\n[{label}]\n{"="*60}')
    torch.manual_seed(42); np.random.seed(42)
    model = model_cls().to(DEVICE)
    n = sum(p.numel() for p in model.parameters())
    print(f'Params: {n/1e6:.2f}M')
    t0 = time.time()
    losses = train_model(model, train_data, epochs=120, lr=5e-4)
    print(f'Time: {time.time()-t0:.1f}s')
    fc_id = eval_model(model, id_data, 'ID')
    fc_oc = eval_model(model, ood_data, 'OOD-C')
    return {'label':label, 'model':model, 'losses':losses, 'params':n,
            'fc_id':fc_id, 'fc_oc':fc_oc}


# ============================================================
# Visualization
# ============================================================
def visualize(results, id_data, ood_data):
    fig = plt.figure(figsize=(20, 14))
    palette = ['#94a3b8', '#3b82f6', '#10b981']
    n_runs = len(results)

    # Bars
    ax = plt.subplot(4, 4, 1)
    labels = [r['label'] for r in results]
    fc_id = [r['fc_id'] for r in results]
    fc_oc = [r['fc_oc'] for r in results]
    x = np.arange(len(labels)); w = 0.35
    ax.bar(x - w/2, fc_id, w, label='ID', color='#3b82f6')
    ax.bar(x + w/2, fc_oc, w, label='OOD-C', color='#ef4444')
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8, rotation=10)
    ax.set_title('FC MSE'); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis='y')

    ax = plt.subplot(4, 4, 2)
    gaps = [r['fc_oc']/r['fc_id'] for r in results]
    bars = ax.bar(x, gaps, color=palette[:n_runs])
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8, rotation=10)
    ax.set_title('Comp gap (OOD-C/ID)'); ax.grid(alpha=0.3, axis='y')
    for b, v in zip(bars, gaps):
        ax.text(b.get_x()+b.get_width()/2, v, f'{v:.1f}x', ha='center', va='bottom', fontsize=9)

    ax = plt.subplot(4, 4, 3)
    for r, c in zip(results, palette):
        ax.plot(r['losses'], label=r['label'], color=c)
    ax.set_yscale('log'); ax.set_title('Train loss')
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    # Feature dim info
    ax = plt.subplot(4, 4, 4); ax.axis('off')
    txt = (f'Dynamics features:\n'
           f'  windows: {N_WINDOWS}\n'
           f'  win FFT top-k: {TOP_K_FFT}\n'
           f'  feat dim: {FEAT_DIM}\n\n'
           f'Per window:\n'
           f'  - dominant freqs ({TOP_K_FFT})\n'
           f'  - slope (1)\n'
           f'  - std (1)\n\n'
           f'Global:\n'
           f'  - mean, std\n'
           f'  - last 5 vals\n'
           f'  - 2 diffs\n')
    ax.text(0, 1, txt, fontsize=8, family='monospace', va='top')

    # Forecast examples per model: 3 ID + 3 OOD-C
    examples = [(id_data, 'ID'), (ood_data, 'OOD-C')]
    for r_idx, r in enumerate(results):
        if r_idx >= 3: break
        model = r['model']; model.eval()
        for ds_idx, (ds, name) in enumerate(examples):
            for j in range(2):
                idx = j * 30 + 10
                w_full = ds[idx]
                ctx = torch.tensor(w_full[:SEQ_LEN]).unsqueeze(0).float().to(DEVICE)
                with torch.no_grad():
                    fp = model.forecast(ctx).cpu().numpy()[0]
                ax_idx = 4 + r_idx*4 + ds_idx*2 + j + 1
                if ax_idx > 16: continue
                ax = plt.subplot(4, 4, ax_idx)
                ax.plot(range(TOTAL_LEN), w_full, 'k-', alpha=0.3)
                ax.plot(range(SEQ_LEN), w_full[:SEQ_LEN], 'k-', linewidth=1)
                ax.plot(range(SEQ_LEN, TOTAL_LEN), fp, color=palette[r_idx], linewidth=1.5)
                ax.axvline(SEQ_LEN, color='gray', linestyle='--', alpha=0.3)
                mse = np.mean((fp - w_full[SEQ_LEN:])**2)
                ax.set_title(f'{r["label"]} {name}#{j} MSE={mse:.3f}', fontsize=8)
                ax.grid(alpha=0.2)

    plt.suptitle('L8: Dynamics-aware Encoder (windowed freq/slope/amp)', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{SAVE_DIR}/L8_dynamics_aware.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {SAVE_DIR}/L8_dynamics_aware.png')


if __name__ == '__main__':
    np.random.seed(42); torch.manual_seed(42)
    print('='*60)
    print('Curriculum L8: Dynamics-aware Encoder')
    print(f'  Feature dim per component: {FEAT_DIM}')
    print('='*60)

    print('\nGenerating data (chirp-dominant, same as L7 v2/v3)...')
    train_data = make_dataset(TRAIN_CONFIGS, n_per_config=120)
    id_data = make_dataset(TRAIN_CONFIGS, n_per_config=30)
    ood_data = make_dataset(OOD_CONFIGS, n_per_config=40)
    print(f'Train: {len(train_data)}, ID: {len(id_data)}, OOD-C: {len(ood_data)}')

    runs = [
        ('Baseline',         Baseline),
        ('L7 v3',            L7V3),
        ('L8 (dyn-aware)',   L8DynamicsAware),
    ]
    results = []
    for label, mc in runs:
        results.append(run(label, mc, train_data, id_data, ood_data))

    print('\n' + '='*60)
    print('SUMMARY')
    print('='*60)
    print(f'{"label":<18} {"params":<10} {"ID":<10} {"OOD-C":<10} {"gap":<10}')
    print('-'*60)
    for r in results:
        gap = r['fc_oc'] / r['fc_id']
        print(f'{r["label"]:<18} {r["params"]/1e6:<10.2f} {r["fc_id"]:<10.4f} {r["fc_oc"]:<10.4f} {gap:<10.2f}')
    print('='*60)

    visualize(results, id_data, ood_data)
    print('\nL8 DONE')
