"""
Curriculum L7 v2 / v3: Improved decomposition strategies

V2: 3-way decomposition, Residual has TWO trunks (RBF + Chirplet)
V3: 4-way decomposition with bandpass cycle extraction

데이터: L4-style chirp + trend + season + bumps (chirp dominant)
비교: Baseline (no decomp) vs L7 v1 (RBF only) vs V2 vs V3

CUDA_VISIBLE_DEVICES=3 python analysis/curriculum/L7_v2_v3.py
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
MA_WINDOW = 25
FFT_TOP_K_SEASON = 3


# ============================================================
# Differentiable decompositions
# ============================================================
def moving_average(x, window=MA_WINDOW):
    pad = window // 2
    x_padded = F.pad(x.unsqueeze(1), (pad, pad), mode='reflect').squeeze(1)
    kernel = torch.ones(1, 1, window, device=x.device) / window
    return F.conv1d(x_padded.unsqueeze(1), kernel).squeeze(1)


def fft_topk_filter(x, k):
    fft = torch.fft.rfft(x, dim=-1)
    mag = fft.abs(); mag[:, 0] = 0
    _, top_idx = torch.topk(mag, k, dim=-1)
    mask = torch.zeros_like(fft)
    mask.scatter_(1, top_idx, 1.0)
    return torch.fft.irfft(fft * mask, n=x.shape[-1], dim=-1)


def fft_bandpass(x, low_idx, high_idx):
    """Keep only frequencies in [low_idx, high_idx)."""
    fft = torch.fft.rfft(x, dim=-1)
    mask = torch.zeros_like(fft)
    mask[:, low_idx:high_idx] = 1.0
    return torch.fft.irfft(fft * mask, n=x.shape[-1], dim=-1)


def decompose_3way(x):
    """Trend (MA) + Season (top-k FFT) + Residual."""
    trend = moving_average(x)
    detrended = x - trend
    season = fft_topk_filter(detrended, k=FFT_TOP_K_SEASON)
    resid = detrended - season
    return trend, season, resid


def decompose_4way(x):
    """Trend + Season (sharp peaks) + Cycle (mid-band) + Residual (high-freq)."""
    trend = moving_average(x)
    detrended = x - trend
    season = fft_topk_filter(detrended, k=FFT_TOP_K_SEASON)
    after_season = detrended - season
    # Cycle: mid frequencies (bands 4-30) where chirp typically lives
    cycle = fft_bandpass(after_season, low_idx=4, high_idx=30)
    resid = after_season - cycle
    return trend, season, cycle, resid


# ============================================================
# Trunks (with chirplet)
# ============================================================
class HTrunk(nn.Module):
    def __init__(self, w, btype, nf=24, deg=5, nc=30, bw=200):
        super().__init__(); self.w=w; self.btype=btype
        if btype=='fourier': self.nf=nf; self.idim=1+2*nf
        elif btype=='poly': self.deg=deg; self.idim=deg+1
        elif btype=='rbf':
            self.bw=bw
            self.register_buffer('centers',torch.linspace(0,2,nc))
            self.idim=1+nc
        elif btype=='chirplet':
            f0_grid = torch.tensor([1.,2.,3.,4.,5.,6.,7.,8.])
            a_grid = torch.tensor([-6.,-3.,0.,3.,6.])
            f0_m, a_m = torch.meshgrid(f0_grid, a_grid, indexing='ij')
            self.register_buffer('f0', f0_m.flatten())
            self.register_buffer('alpha', a_m.flatten())
            self.idim = 1 + 2*self.f0.shape[0]
        self.pc = self.idim*w + w
        self.odim = self.pc + w

    def feat(self, t):
        t = t.unsqueeze(-1) if t.dim()==1 else t
        if self.btype=='fourier':
            f=torch.arange(1,self.nf+1,device=t.device,dtype=t.dtype)
            return torch.cat([t,torch.sin(2*math.pi*f*t),torch.cos(2*math.pi*f*t)],dim=-1)
        elif self.btype=='poly':
            return torch.cat([t**i for i in range(self.deg+1)],dim=-1)
        elif self.btype=='rbf':
            return torch.cat([t,torch.exp(-self.bw*(t-self.centers.unsqueeze(0))**2)],dim=-1)
        elif self.btype=='chirplet':
            f0=self.f0.unsqueeze(0); a=self.alpha.unsqueeze(0)
            phase = 2*math.pi*f0*t + math.pi*a*(t**2)
            return torch.cat([t, torch.sin(phase), torch.cos(phase)], dim=-1)


def hfwd(trunk, t_flat, head_out):
    B = head_out.shape[0]
    ft = trunk.feat(t_flat); nq = ft.shape[0] // B
    ft = ft.view(B, nq, trunk.idim)
    tp = head_out[:, :trunk.pc] * 0.01
    Wm = tp[:, :trunk.idim*trunk.w].view(B, trunk.idim, trunk.w)
    bm = tp[:, trunk.idim*trunk.w:].view(B, trunk.w)
    Phi = F.gelu(torch.bmm(ft, Wm) + bm.unsqueeze(1))
    Bc = head_out[:, trunk.pc:]
    return torch.einsum('bw,bqw->bq', Bc, Phi)


# ============================================================
# Encoder
# ============================================================
class MLPEnc(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(SEQ_LEN, H), nn.GELU(),
            nn.Linear(H, H), nn.GELU(),
            nn.Linear(H, H), nn.GELU())
    def forward(self, x): return self.net(x)


class TrunkBlock(nn.Module):
    """One encoder + one or more trunks (additive)."""
    def __init__(self, trunk_types):
        super().__init__()
        self.enc = MLPEnc()
        self.trunks = nn.ModuleList([HTrunk(W, t) for t in trunk_types])
        self.heads = nn.ModuleList([nn.Linear(H, t.odim) for t in self.trunks])
        for h in self.heads: nn.init.xavier_normal_(h.weight, gain=0.1)
        self.biases = nn.ParameterList([nn.Parameter(torch.zeros(1)) for _ in trunk_types])

    def forward(self, x_part, qt):
        z = self.enc(x_part)
        t_flat = qt.reshape(-1)
        return sum(hfwd(tr, t_flat, h(z)) + b
                   for tr, h, b in zip(self.trunks, self.heads, self.biases))


# ============================================================
# Models
# ============================================================
class Baseline(nn.Module):
    """No decomposition, single shared encoder + 4 trunks (chirplet included)."""
    def __init__(self):
        super().__init__()
        self.block = TrunkBlock(['poly', 'fourier', 'rbf', 'chirplet'])

    def forward_train(self, ctx, qt): return self.block(ctx, qt)

    def forecast(self, ctx, n=PRED_LEN):
        t = torch.linspace(1, 2, n, device=ctx.device).unsqueeze(0).expand(ctx.shape[0], -1)
        return self.block(ctx, t)


class L7V1(nn.Module):
    """3-way decomp, R = RBF only."""
    def __init__(self):
        super().__init__()
        self.block_T = TrunkBlock(['poly'])
        self.block_S = TrunkBlock(['fourier'])
        self.block_R = TrunkBlock(['rbf'])

    def _q(self, ctx, qt):
        trend, season, resid = decompose_3way(ctx)
        return self.block_T(trend, qt) + self.block_S(season, qt) + self.block_R(resid, qt)

    def forward_train(self, ctx, qt): return self._q(ctx, qt)
    def forecast(self, ctx, n=PRED_LEN):
        t = torch.linspace(1, 2, n, device=ctx.device).unsqueeze(0).expand(ctx.shape[0], -1)
        return self._q(ctx, t)


class L7V2(nn.Module):
    """3-way decomp, R = RBF + Chirplet (boosted residual)."""
    def __init__(self):
        super().__init__()
        self.block_T = TrunkBlock(['poly'])
        self.block_S = TrunkBlock(['fourier'])
        self.block_R = TrunkBlock(['rbf', 'chirplet'])  # ← boost

    def _q(self, ctx, qt):
        trend, season, resid = decompose_3way(ctx)
        return self.block_T(trend, qt) + self.block_S(season, qt) + self.block_R(resid, qt)

    def forward_train(self, ctx, qt): return self._q(ctx, qt)
    def forecast(self, ctx, n=PRED_LEN):
        t = torch.linspace(1, 2, n, device=ctx.device).unsqueeze(0).expand(ctx.shape[0], -1)
        return self._q(ctx, t)


class L7V3(nn.Module):
    """4-way decomp with bandpass cycle. T/S/C/R → Poly/Fourier/Chirplet/RBF."""
    def __init__(self):
        super().__init__()
        self.block_T = TrunkBlock(['poly'])
        self.block_S = TrunkBlock(['fourier'])
        self.block_C = TrunkBlock(['chirplet'])
        self.block_R = TrunkBlock(['rbf'])

    def _q(self, ctx, qt):
        trend, season, cycle, resid = decompose_4way(ctx)
        return (self.block_T(trend, qt) + self.block_S(season, qt) +
                self.block_C(cycle, qt) + self.block_R(resid, qt))

    def forward_train(self, ctx, qt): return self._q(ctx, qt)
    def forecast(self, ctx, n=PRED_LEN):
        t = torch.linspace(1, 2, n, device=ctx.device).unsqueeze(0).expand(ctx.shape[0], -1)
        return self._q(ctx, t)


# ============================================================
# Data: L4-style chirp dominant + trend + season + bumps
# ============================================================
def gen_sample(slope, seas_freq, chirp_f0, chirp_f1, bump_centers, t_max=2.0, n=TOTAL_LEN):
    t = np.linspace(0, t_max, n)
    b = np.random.uniform(-0.5, 0.5)
    u_T = slope * t + b
    amp_s = np.random.uniform(0.3, 0.7)
    phi_s = np.random.uniform(0, 2*np.pi)
    u_S = amp_s * np.sin(2*np.pi * seas_freq * t + phi_s)
    # Chirp DOMINANT
    amp_c = np.random.uniform(1.0, 1.5)
    phi_c = np.random.uniform(0, 2*np.pi)
    f_t = chirp_f0 + (chirp_f1 - chirp_f0) * t / t_max
    u_C = amp_c * np.sin(2*np.pi * f_t * t + phi_c)
    u_R = np.zeros(n)
    for c in bump_centers:
        A = np.random.uniform(0.8, 1.5) * (1 if np.random.rand() > 0.5 else -1)
        w = np.random.uniform(0.05, 0.10)
        u_R += A * np.exp(-((t - c) / w)**2)
    y = u_T + u_S + u_C + u_R
    s = y.std()
    if s > 1e-6: y = (y - y.mean()) / s
    return y.astype(np.float32)


def make_dataset(configs, n_per_config=200):
    return np.stack([gen_sample(**cfg) for cfg in configs for _ in range(n_per_config)])


def make_random_configs(n, slope_range, seas_freqs, chirp_pairs, bump_options, seed=0):
    rng = np.random.RandomState(seed)
    return [{
        'slope': rng.uniform(*slope_range),
        'seas_freq': rng.choice(seas_freqs),
        'chirp_f0': chirp_pairs[rng.randint(len(chirp_pairs))][0],
        'chirp_f1': chirp_pairs[rng.randint(len(chirp_pairs))][1],
        'bump_centers': bump_options[rng.randint(len(bump_options))],
    } for _ in range(n)]


# Wider chirp range (like L4)
TRAIN_CONFIGS = make_random_configs(
    20, slope_range=(-1, 2), seas_freqs=[2,3,4,5],
    chirp_pairs=[(1,4),(2,5),(3,6),(4,1),(5,2),(6,3),(1,5),(2,6),(3,1),(4,2)],
    bump_options=[[0.3,1.4],[0.5,1.6],[0.6,1.3],[0.4,1.5]], seed=0)

OOD_CONFIGS = make_random_configs(
    12, slope_range=(-1, 2), seas_freqs=[2,3,4,5],
    chirp_pairs=[(1,3),(2,4),(3,5),(4,6),(5,1),(6,2)],  # new chirp pairs
    bump_options=[[0.4,1.6],[0.3,1.5],[0.5,1.4]], seed=100)


# ============================================================
# Train / Eval
# ============================================================
def train_model(model, data, epochs=120, lr=5e-4, n_query=N_QUERY, bs=64):
    opt = optim.Adam(model.parameters(), lr=lr)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    losses = []
    bpe = max(50, len(data) // bs)
    for ep in range(epochs):
        model.train(); ls = []
        for _ in range(bpe):
            idxs = np.random.choice(len(data), bs)
            batch = data[idxs]
            ctxs = batch[:, :SEQ_LEN]; futures = batch[:, SEQ_LEN:]
            if np.random.rand() < 0.5:
                qi = np.random.choice(PRED_LEN, n_query, replace=False)
                qt = 1.0 + qi.astype(np.float32) / PRED_LEN
                qv = futures[:, qi]
                qt_b = np.tile(qt, (bs, 1))
                ctx_t = ctxs
            else:
                masks = np.random.rand(bs, SEQ_LEN) > 0.375
                qt_b = np.zeros((bs, n_query), dtype=np.float32)
                qv = np.zeros((bs, n_query), dtype=np.float32)
                ctx_t = ctxs * masks.astype(np.float32)
                for b in range(bs):
                    qi = np.where(~masks[b])[0]
                    if len(qi) == 0: qi = np.array([0])
                    if len(qi) >= n_query:
                        qi = np.random.choice(qi, n_query, replace=False)
                    else:
                        qi = np.tile(qi, (n_query // len(qi) + 1))[:n_query]
                    qt_b[b] = qi.astype(np.float32) / SEQ_LEN
                    qv[b] = ctxs[b][qi]
            c = torch.tensor(ctx_t).float().to(DEVICE)
            q = torch.tensor(qt_b).float().to(DEVICE)
            v = torch.tensor(qv).float().to(DEVICE)
            opt.zero_grad()
            loss = F.mse_loss(model.forward_train(c, q), v)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ls.append(loss.item())
        sched.step()
        avg = np.mean(ls)
        losses.append(avg)
        if ep % 20 == 0 or ep == epochs-1:
            print(f'  Ep {ep+1}/{epochs}: loss={avg:.5f}')
    return losses


def eval_model(model, data, label=''):
    model.eval()
    fc = []
    with torch.no_grad():
        for w in data[:300]:
            ctx = torch.tensor(w[:SEQ_LEN]).unsqueeze(0).float().to(DEVICE)
            pred = model.forecast(ctx).cpu().numpy()[0]
            tgt = w[SEQ_LEN:]
            fc.append(np.mean((pred - tgt)**2))
    fc_m = np.mean(fc)
    print(f'  [{label}] FC={fc_m:.4f}')
    return fc_m


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
# Visualization (max 16 panels, 4x4)
# ============================================================
def visualize(results, id_data, ood_data):
    fig = plt.figure(figsize=(20, 14))
    n_runs = len(results)
    palette = ['#94a3b8', '#3b82f6', '#10b981', '#a855f7']

    # Row 1: bars
    ax = plt.subplot(4, 4, 1)
    labels = [r['label'] for r in results]
    fc_id = [r['fc_id'] for r in results]
    fc_oc = [r['fc_oc'] for r in results]
    x = np.arange(len(labels)); w = 0.35
    ax.bar(x - w/2, fc_id, w, label='ID', color='#3b82f6')
    ax.bar(x + w/2, fc_oc, w, label='OOD-C', color='#ef4444')
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7, rotation=15)
    ax.set_title('FC MSE'); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis='y')

    ax = plt.subplot(4, 4, 2)
    gaps = [r['fc_oc']/r['fc_id'] for r in results]
    bars = ax.bar(x, gaps, color=palette[:n_runs])
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7, rotation=15)
    ax.set_title('Comp gap (OOD-C / ID)'); ax.grid(alpha=0.3, axis='y')
    for b, v in zip(bars, gaps):
        ax.text(b.get_x()+b.get_width()/2, v, f'{v:.1f}x', ha='center', va='bottom', fontsize=8)

    ax = plt.subplot(4, 4, 3)
    for r, c in zip(results, palette):
        ax.plot(r['losses'], label=r['label'], color=c)
    ax.set_yscale('log'); ax.set_title('Train loss')
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    # Decomp demos
    ax = plt.subplot(4, 4, 4)
    sample = id_data[10]
    ctx_t = torch.tensor(sample[:SEQ_LEN]).unsqueeze(0).float().to(DEVICE)
    with torch.no_grad():
        t3, s3, r3 = decompose_3way(ctx_t)
        t4, s4, c4, r4 = decompose_4way(ctx_t)
    ax.plot(sample[:SEQ_LEN], 'k-', alpha=0.4, label='input', linewidth=1)
    ax.plot(t4.cpu().numpy()[0], '#10b981', label='trend', linewidth=1.2)
    ax.plot(s4.cpu().numpy()[0], '#3b82f6', label='season', linewidth=1)
    ax.plot(c4.cpu().numpy()[0], '#a855f7', label='cycle', linewidth=1)
    ax.plot(r4.cpu().numpy()[0], '#f59e0b', label='resid', linewidth=0.8, alpha=0.7)
    ax.set_title('4-way decomp example')
    ax.legend(fontsize=6); ax.grid(alpha=0.2)

    # Row 2-4: forecast examples per model (3 examples each = 12 panels)
    examples = [(id_data, 'ID'), (ood_data, 'OOD-C')]
    for r_idx, r in enumerate(results):
        if r_idx >= 4: break
        model = r['model']; model.eval()
        for ds_idx, (ds, name) in enumerate(examples):
            for j in range(1):
                idx = j * 30 + 10
                w = ds[idx]
                ctx = torch.tensor(w[:SEQ_LEN]).unsqueeze(0).float().to(DEVICE)
                with torch.no_grad():
                    fp = model.forecast(ctx).cpu().numpy()[0]
                ax_idx = 4 + r_idx*2 + ds_idx + 1  # rows 2-3
                if ax_idx > 16: continue
                ax = plt.subplot(4, 4, ax_idx)
                ax.plot(range(TOTAL_LEN), w, 'k-', alpha=0.3)
                ax.plot(range(SEQ_LEN), w[:SEQ_LEN], 'k-', linewidth=1)
                ax.plot(range(SEQ_LEN, TOTAL_LEN), fp, color=palette[r_idx], linewidth=1.5)
                ax.axvline(SEQ_LEN, color='gray', linestyle='--', alpha=0.3)
                mse = np.mean((fp - w[SEQ_LEN:])**2)
                ax.set_title(f'{r["label"]} {name} MSE={mse:.3f}', fontsize=8)
                ax.grid(alpha=0.2)

    plt.suptitle('L7 v2/v3: Improved Decomposition Strategies (chirp-dominant data)', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{SAVE_DIR}/L7_v2_v3.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {SAVE_DIR}/L7_v2_v3.png')


if __name__ == '__main__':
    np.random.seed(42); torch.manual_seed(42)
    print('='*60)
    print('Curriculum L7 v2/v3: Decomp variants on chirp-dominant data')
    print('='*60)

    print('\nGenerating data (chirp-dominant)...')
    train_data = make_dataset(TRAIN_CONFIGS, n_per_config=120)
    id_data = make_dataset(TRAIN_CONFIGS, n_per_config=30)
    ood_data = make_dataset(OOD_CONFIGS, n_per_config=40)
    print(f'Train: {len(train_data)}, ID: {len(id_data)}, OOD-C: {len(ood_data)}')

    runs = [
        ('Baseline',  Baseline),
        ('L7 v1',     L7V1),
        ('L7 v2',     L7V2),
        ('L7 v3',     L7V3),
    ]
    results = []
    for label, mc in runs:
        results.append(run(label, mc, train_data, id_data, ood_data))

    print('\n' + '='*60)
    print('SUMMARY')
    print('='*60)
    print(f'{"label":<15} {"params":<10} {"ID":<10} {"OOD-C":<10} {"gap":<10}')
    print('-'*60)
    for r in results:
        gap = r['fc_oc'] / r['fc_id']
        print(f'{r["label"]:<15} {r["params"]/1e6:<10.2f} {r["fc_id"]:<10.4f} {r["fc_oc"]:<10.4f} {gap:<10.2f}')
    print('='*60)

    visualize(results, id_data, ood_data)
    print('\nL7 v2/v3 DONE')
