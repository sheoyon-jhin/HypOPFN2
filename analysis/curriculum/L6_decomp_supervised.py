"""
Curriculum L6: Supervised Decomposition with Matched Trunks

핵심 아이디어:
  데이터 = trend + seasonality + cycle + residual (4 components)
  각 encoder가 자기 component만 추출 (supervised)
  각 trunk가 매칭된 basis로 출력 함수 생성

  Trend       → Poly trunk        (smooth, low degree)
  Seasonality → Fourier trunk     (stationary frequency)
  Cycle       → Chirplet trunk    (time-varying frequency)
  Residual    → RBF trunk         (local bumps)

Losses:
  L_total = ||v - y||² + λ Σᵢ ||vᵢ - uᵢ_GT||²

비교 ablation:
  A: λ=0  (no supervision, like L5)
  B: λ=1  (full supervision)

CUDA_VISIBLE_DEVICES=3 python analysis/curriculum/L6_decomp_supervised.py
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


# ============================================================
# Data: 4-component synthetic (with GT components)
# ============================================================
def gen_sample(slope, seas_freq, chirp_f0, chirp_f1, bump_centers, t_max=2.0, n=TOTAL_LEN):
    """Returns y, [u_T, u_S, u_C, u_R] all in normalized space."""
    t = np.linspace(0, t_max, n)

    # Trend
    b = np.random.uniform(-0.5, 0.5)
    u_T = slope * t + b

    # Seasonality (stationary sin)
    amp_s = np.random.uniform(0.5, 1.0)
    phi_s = np.random.uniform(0, 2*np.pi)
    u_S = amp_s * np.sin(2*np.pi * seas_freq * t + phi_s)

    # Cycle (chirp)
    amp_c = np.random.uniform(0.5, 1.0)
    phi_c = np.random.uniform(0, 2*np.pi)
    f_t = chirp_f0 + (chirp_f1 - chirp_f0) * t / t_max
    u_C = amp_c * np.sin(2*np.pi * f_t * t + phi_c)

    # Residual (Gaussian bumps)
    u_R = np.zeros(n)
    for c in bump_centers:
        A = np.random.uniform(1.0, 2.0) * (1 if np.random.rand() > 0.5 else -1)
        w = np.random.uniform(0.05, 0.10)
        u_R += A * np.exp(-((t - c) / w)**2)

    y = u_T + u_S + u_C + u_R
    s = y.std()
    if s > 1e-6:
        m = y.mean()
        y_n = (y - m) / s
        u_T_n = (u_T - u_T.mean()) / s
        u_S_n = u_S / s
        u_C_n = u_C / s
        u_R_n = u_R / s
        # Adjust trend center so total reconstructs
        bias_shift = (u_T.mean() - m) / s
        u_T_n = u_T_n + bias_shift
    else:
        y_n = y; u_T_n = u_T; u_S_n = u_S; u_C_n = u_C; u_R_n = u_R

    return (y_n.astype(np.float32),
            np.stack([u_T_n, u_S_n, u_C_n, u_R_n]).astype(np.float32))


def make_dataset(configs, n_per_config=200):
    ys, comps = [], []
    for cfg in configs:
        for _ in range(n_per_config):
            y, c = gen_sample(**cfg)
            ys.append(y); comps.append(c)
    return np.stack(ys), np.stack(comps)


def make_random_configs(n, slope_range, seas_freqs, chirp_pairs, bump_options, seed=0):
    rng = np.random.RandomState(seed)
    configs = []
    for _ in range(n):
        configs.append({
            'slope': rng.uniform(*slope_range),
            'seas_freq': rng.choice(seas_freqs),
            'chirp_f0': chirp_pairs[rng.randint(len(chirp_pairs))][0],
            'chirp_f1': chirp_pairs[rng.randint(len(chirp_pairs))][1],
            'bump_centers': bump_options[rng.randint(len(bump_options))],
        })
    return configs


# Splits
TRAIN_CONFIGS = make_random_configs(
    n=20,
    slope_range=(-1, 2),
    seas_freqs=[2, 3, 4, 5],
    chirp_pairs=[(1,3),(2,4),(3,5),(4,6),(5,3),(6,2)],
    bump_options=[[0.3, 1.4],[0.5, 1.6],[0.6, 1.3],[0.4, 1.5]],
    seed=0,
)
OOD_C_CONFIGS = make_random_configs(
    n=12,
    slope_range=(-1, 2),
    seas_freqs=[2, 3, 4, 5],
    chirp_pairs=[(1,4),(2,5),(3,6),(4,5),(5,2),(6,1)],  # new chirp pairs
    bump_options=[[0.4, 1.6],[0.3, 1.5],[0.5, 1.4]],
    seed=100,
)


# ============================================================
# Trunks
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
# Model: 4 encoders + 4 matched trunks
# ============================================================
class MLPEnc(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(SEQ_LEN, H), nn.GELU(),
            nn.Linear(H, H), nn.GELU(),
            nn.Linear(H, H), nn.GELU())
    def forward(self, x): return self.net(x)


COMPONENT_NAMES = ['Trend', 'Season', 'Cycle', 'Residual']
TRUNK_TYPES = ['poly', 'fourier', 'chirplet', 'rbf']  # matched order


class DecompOperator(nn.Module):
    def __init__(self):
        super().__init__()
        self.encs = nn.ModuleList([MLPEnc() for _ in TRUNK_TYPES])
        self.trunks = nn.ModuleList([HTrunk(W, t) for t in TRUNK_TYPES])
        self.heads = nn.ModuleList([nn.Linear(H, t.odim) for t in self.trunks])
        for h in self.heads: nn.init.xavier_normal_(h.weight, gain=0.1)
        self.biases = nn.ParameterList([nn.Parameter(torch.zeros(1)) for _ in TRUNK_TYPES])

    def per_component(self, ctx, qt):
        """Returns list of [B, nq] per component."""
        t_flat = qt.reshape(-1)
        outs = []
        for enc, trunk, head, bias in zip(self.encs, self.trunks, self.heads, self.biases):
            z = enc(ctx)
            outs.append(hfwd(trunk, t_flat, head(z)) + bias)
        return outs

    def forward_train(self, ctx, qt):
        outs = self.per_component(ctx, qt)
        return sum(outs), outs

    def forecast(self, ctx, n=PRED_LEN, return_components=False):
        t = torch.linspace(1, 2, n, device=ctx.device).unsqueeze(0).expand(ctx.shape[0], -1)
        outs = self.per_component(ctx, t)
        full = sum(outs)
        if return_components:
            return full, outs
        return full

    def query(self, ctx, t_norm, return_components=False):
        outs = self.per_component(ctx, t_norm)
        full = sum(outs)
        if return_components:
            return full, outs
        return full


# ============================================================
# Train
# ============================================================
def train_model(model, train_data, train_comps, lambda_sup=1.0, epochs=120, lr=5e-4, n_query=N_QUERY, bs=64):
    """train_data: [N, 384], train_comps: [N, 4, 384]"""
    opt = optim.Adam(model.parameters(), lr=lr)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    losses = []
    sup_losses = []
    bpe = max(50, len(train_data) // bs)

    for ep in range(epochs):
        model.train(); ls = []; sl = []
        for _ in range(bpe):
            idxs = np.random.choice(len(train_data), bs)
            batch = train_data[idxs]
            comps = train_comps[idxs]  # [bs, 4, 384]
            ctxs = batch[:, :SEQ_LEN]
            futures = batch[:, SEQ_LEN:]

            if np.random.rand() < 0.5:
                # Forecast
                qi = np.random.choice(PRED_LEN, n_query, replace=False)
                qt = 1.0 + qi.astype(np.float32) / PRED_LEN
                qv = futures[:, qi]                   # [bs, n_query]
                qv_comps = comps[:, :, SEQ_LEN:][:, :, qi]  # [bs, 4, n_query]
                qt_b = np.tile(qt, (bs, 1))
                ctx_t = ctxs
            else:
                # Imputation
                masks = np.random.rand(bs, SEQ_LEN) > 0.375
                qt_b = np.zeros((bs, n_query), dtype=np.float32)
                qv = np.zeros((bs, n_query), dtype=np.float32)
                qv_comps = np.zeros((bs, 4, n_query), dtype=np.float32)
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
                    qv_comps[b] = comps[b, :, :SEQ_LEN][:, qi]

            c = torch.tensor(ctx_t).float().to(DEVICE)
            q = torch.tensor(qt_b).float().to(DEVICE)
            v = torch.tensor(qv).float().to(DEVICE)
            v_comps = torch.tensor(qv_comps).float().to(DEVICE)

            opt.zero_grad()
            pred_full, pred_comps = model.forward_train(c, q)
            loss_total = F.mse_loss(pred_full, v)
            if lambda_sup > 0:
                loss_sup = sum(F.mse_loss(pc, v_comps[:, i]) for i, pc in enumerate(pred_comps))
                loss = loss_total + lambda_sup * loss_sup
                sl.append(loss_sup.item())
            else:
                loss = loss_total
                sl.append(0.0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ls.append(loss_total.item())

        sched.step()
        avg = np.mean(ls); avg_sup = np.mean(sl)
        losses.append(avg); sup_losses.append(avg_sup)
        if ep % 20 == 0 or ep == epochs-1:
            print(f'  Ep {ep+1}/{epochs}: recon={avg:.5f} sup={avg_sup:.5f}')
    return losses, sup_losses


def eval_model(model, data, comps, label=''):
    model.eval()
    fc_total = []
    fc_per_comp = [[] for _ in range(4)]
    with torch.no_grad():
        for w, c in zip(data[:300], comps[:300]):
            ctx = torch.tensor(w[:SEQ_LEN]).unsqueeze(0).float().to(DEVICE)
            pred_full, pred_comps = model.forecast(ctx, return_components=True)
            pred_full = pred_full.cpu().numpy()[0]
            tgt = w[SEQ_LEN:]
            fc_total.append(np.mean((pred_full - tgt)**2))
            for i in range(4):
                pc = pred_comps[i].cpu().numpy()[0]
                fc_per_comp[i].append(np.mean((pc - c[i, SEQ_LEN:])**2))
    fc = np.mean(fc_total)
    fc_c = [np.mean(x) for x in fc_per_comp]
    print(f'  [{label}] Total FC={fc:.4f}  per-comp: T={fc_c[0]:.4f} S={fc_c[1]:.4f} C={fc_c[2]:.4f} R={fc_c[3]:.4f}')
    return fc, fc_c


def run_experiment(label, lambda_sup, train_data, train_comps, id_data, id_comps, ood_data, ood_comps):
    print(f'\n{"="*60}')
    print(f'[{label}] λ_sup={lambda_sup}')
    print('='*60)
    torch.manual_seed(42); np.random.seed(42)
    model = DecompOperator().to(DEVICE)
    n = sum(p.numel() for p in model.parameters())
    print(f'Params: {n/1e6:.2f}M')
    t0 = time.time()
    losses, sup_losses = train_model(model, train_data, train_comps,
                                      lambda_sup=lambda_sup, epochs=120, lr=5e-4)
    print(f'Time: {time.time()-t0:.1f}s')
    fc_id, fc_id_c = eval_model(model, id_data, id_comps, 'ID')
    fc_oc, fc_oc_c = eval_model(model, ood_data, ood_comps, 'OOD-C')
    return {
        'label': label, 'lambda': lambda_sup, 'model': model, 'params': n,
        'losses': losses, 'sup_losses': sup_losses,
        'fc_id': fc_id, 'fc_id_c': fc_id_c,
        'fc_oc': fc_oc, 'fc_oc_c': fc_oc_c,
    }


# ============================================================
# Visualization (max 12 subplots, 3x4)
# ============================================================
def visualize(results, id_data, id_comps, ood_data, ood_comps):
    fig = plt.figure(figsize=(20, 14))
    n_runs = len(results)
    comp_colors = ['#10b981', '#3b82f6', '#a855f7', '#f59e0b']

    # Row 1: bars + loss curves
    ax = plt.subplot(4, 4, 1)
    labels = [r['label'] for r in results]
    fc_id = [r['fc_id'] for r in results]
    fc_oc = [r['fc_oc'] for r in results]
    x = np.arange(len(labels)); w = 0.35
    ax.bar(x - w/2, fc_id, w, label='ID', color='#3b82f6')
    ax.bar(x + w/2, fc_oc, w, label='OOD-C', color='#ef4444')
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_title('Total FC MSE', fontsize=10); ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis='y')

    ax = plt.subplot(4, 4, 2)
    gaps = [r['fc_oc']/r['fc_id'] for r in results]
    bars = ax.bar(x, gaps, color=['#94a3b8','#10b981'])
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_title('Comp gap', fontsize=10); ax.grid(alpha=0.3, axis='y')
    for b, v in zip(bars, gaps):
        ax.text(b.get_x()+b.get_width()/2, v, f'{v:.1f}x', ha='center', va='bottom', fontsize=9)

    ax = plt.subplot(4, 4, 3)
    for r in results:
        ax.plot(r['losses'], label=f'{r["label"]} recon')
    ax.set_yscale('log'); ax.set_title('Recon loss', fontsize=10)
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    ax = plt.subplot(4, 4, 4)
    # Per-component comp gap (only for supervised)
    sup_r = [r for r in results if r['lambda'] > 0]
    if sup_r:
        r = sup_r[0]
        comp_gaps = [r['fc_oc_c'][i] / max(r['fc_id_c'][i], 1e-6) for i in range(4)]
        bars = ax.bar(COMPONENT_NAMES, comp_gaps, color=comp_colors)
        ax.set_title(f'{r["label"]} per-component gap', fontsize=9)
        for b, v in zip(bars, comp_gaps):
            ax.text(b.get_x()+b.get_width()/2, v, f'{v:.1f}x', ha='center', va='bottom', fontsize=8)
        ax.grid(alpha=0.3, axis='y')

    # Row 2-4: per-model component decomposition for 1 ID + 1 OOD-C example
    for r_idx, r in enumerate(results):
        model = r['model']; model.eval()
        for ds_idx, (ds_name, ds, ds_c) in enumerate([('ID', id_data, id_comps), ('OOD-C', ood_data, ood_comps)]):
            idx = 30
            w_full = ds[idx]
            true_comps = ds_c[idx]  # [4, 384]
            ctx = torch.tensor(w_full[:SEQ_LEN]).unsqueeze(0).float().to(DEVICE)
            with torch.no_grad():
                pred_full, pred_comps = model.forecast(ctx, return_components=True)
            pred_full = pred_full.cpu().numpy()[0]
            pred_comps_np = [pc.cpu().numpy()[0] for pc in pred_comps]

            row = 1 + ds_idx + r_idx*2
            if row >= 4: continue
            for ci in range(4):
                ax_idx = row*4 + ci + 1
                if ax_idx > 16: continue
                ax = plt.subplot(4, 4, ax_idx)
                # Plot GT component (full range)
                ax.plot(range(TOTAL_LEN), true_comps[ci], 'k--', alpha=0.6, label='GT', linewidth=1.2)
                # Plot predicted component (forecast region only)
                ax.plot(range(SEQ_LEN, TOTAL_LEN), pred_comps_np[ci],
                        color=comp_colors[ci], linewidth=1.5, label='pred')
                ax.axvline(SEQ_LEN, color='gray', linestyle='--', alpha=0.3)
                ax.set_title(f'{r["label"]} {ds_name} {COMPONENT_NAMES[ci]}', fontsize=8)
                ax.legend(fontsize=6); ax.grid(alpha=0.2)

    plt.suptitle('L6: Supervised Decomposition (Trend+Season+Cycle+Residual)', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{SAVE_DIR}/L6_decomp_supervised.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {SAVE_DIR}/L6_decomp_supervised.png')


if __name__ == '__main__':
    np.random.seed(42); torch.manual_seed(42)
    print('='*60)
    print('Curriculum L6: Supervised Decomposition')
    print(f'  Train configs: {len(TRAIN_CONFIGS)}')
    print(f'  OOD-C configs: {len(OOD_C_CONFIGS)}')
    print('='*60)

    print('\nGenerating data with GT components...')
    train_data, train_comps = make_dataset(TRAIN_CONFIGS, n_per_config=120)
    id_data, id_comps = make_dataset(TRAIN_CONFIGS, n_per_config=30)
    ood_data, ood_comps = make_dataset(OOD_C_CONFIGS, n_per_config=40)
    print(f'Train: {len(train_data)} (comps shape {train_comps.shape})')
    print(f'ID: {len(id_data)}, OOD-C: {len(ood_data)}')

    results = []
    # Run A: no supervision (just architecture)
    r = run_experiment('No-sup', 0.0, train_data, train_comps, id_data, id_comps, ood_data, ood_comps)
    results.append(r)

    # Run B: with supervision
    r = run_experiment('Sup', 1.0, train_data, train_comps, id_data, id_comps, ood_data, ood_comps)
    results.append(r)

    print('\n' + '='*60)
    print('SUMMARY')
    print('='*60)
    print(f'{"label":<10} {"params":<10} {"ID":<10} {"OOD-C":<10} {"gap":<10}')
    print('-'*60)
    for r in results:
        gap = r['fc_oc'] / r['fc_id']
        print(f'{r["label"]:<10} {r["params"]/1e6:<10.2f} {r["fc_id"]:<10.4f} {r["fc_oc"]:<10.4f} {gap:<10.2f}')
    print('='*60)

    print('\nPer-component test FC (for Sup model):')
    sup_r = [r for r in results if r['lambda'] > 0][0]
    for i, name in enumerate(COMPONENT_NAMES):
        gap_i = sup_r['fc_oc_c'][i] / max(sup_r['fc_id_c'][i], 1e-6)
        print(f'  {name:<10}: ID={sup_r["fc_id_c"][i]:.4f}  OOD-C={sup_r["fc_oc_c"][i]:.4f}  gap=×{gap_i:.2f}')

    visualize(results, id_data, id_comps, ood_data, ood_comps)
    print('\nL6 DONE')
