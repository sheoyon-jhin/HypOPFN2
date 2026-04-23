"""
Curriculum L4 v4: Chirp with Identifiable Pairs + Long Context

두 가지 변경:
  1. Identifiable train pairs (각 f0마다 f1 하나)
  2. Context 96 → 192 (chirp rate α 추정 가능)

데이터 길이: 384 (context 192 + future 192)

CUDA_VISIBLE_DEVICES=3 python analysis/curriculum/L4_v4_long_ctx.py
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

# === Long context ===
SEQ_LEN = 192   # ← 96 → 192
PRED_LEN = 192
TOTAL_LEN = 384
W = 64
H = 192
N_QUERY = 32


# ============================================================
# Trunk (with chirplet)
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


class PatchAttnEnc(nn.Module):
    def __init__(self, ps=16, nl=4):
        super().__init__()
        self.ps = ps
        np_ = SEQ_LEN // ps
        self.proj = nn.Linear(ps, H)
        self.pos = nn.Parameter(torch.randn(1, np_, H)*0.02)
        layer = nn.TransformerEncoderLayer(d_model=H, nhead=4, dim_feedforward=H*4,
                                            dropout=0.1, activation='gelu',
                                            batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(layer, num_layers=nl)
        self.norm = nn.LayerNorm(H)

    def forward(self, x):
        B = x.shape[0]
        z = self.proj(x.view(B, -1, self.ps)) + self.pos
        return self.norm(self.tf(z).mean(dim=1))


class Operator(nn.Module):
    def __init__(self, trunk_types=('fourier','poly','rbf','chirplet')):
        super().__init__()
        self.enc = PatchAttnEnc()
        self.trunk_names = list(trunk_types)
        self.trunks = nn.ModuleList([HTrunk(W, t) for t in trunk_types])
        self.heads = nn.ModuleList([nn.Linear(H, t.odim) for t in self.trunks])
        for h in self.heads: nn.init.xavier_normal_(h.weight, gain=0.1)
        self.biases = nn.ParameterList([nn.Parameter(torch.zeros(1)) for _ in trunk_types])

    def _query(self, z, qt):
        B, nq = qt.shape
        t_flat = qt.reshape(-1)
        return sum(hfwd(t, t_flat, h(z)) + b for t, h, b in zip(self.trunks, self.heads, self.biases))

    def _query_per_trunk(self, z, qt):
        t_flat = qt.reshape(-1)
        return [hfwd(t, t_flat, h(z)) + b for t, h, b in zip(self.trunks, self.heads, self.biases)]

    def forward_train(self, ctx, qt): return self._query(self.enc(ctx), qt)

    def forecast(self, ctx, n=PRED_LEN):
        # Context normalized to t ∈ [0, 1], future t ∈ [1, 2]
        t = torch.linspace(1, 2, n, device=ctx.device).unsqueeze(0).expand(ctx.shape[0], -1)
        return self._query(self.enc(ctx), t)

    def per_trunk_forecast(self, ctx, n=PRED_LEN):
        t = torch.linspace(1, 2, n, device=ctx.device).unsqueeze(0).expand(ctx.shape[0], -1)
        return self._query_per_trunk(self.enc(ctx), t)


# ============================================================
# Data: chirp 384-length, identifiable pairs
# ============================================================
def gen_chirp(f0, f1, t_max=2.0, n=TOTAL_LEN):
    t = np.linspace(0, t_max, n)
    f_t = f0 + (f1 - f0) * t / t_max
    phi = np.random.uniform(0, 2*np.pi)
    amp = np.random.uniform(0.7, 1.3)
    y = amp * np.sin(2*np.pi * f_t * t + phi)
    s = y.std()
    if s > 1e-6: y = (y - y.mean()) / s
    return y.astype(np.float32)


def make_dataset(pairs, n_per_pair=200):
    data = []
    for f0, f1 in pairs:
        for _ in range(n_per_pair):
            data.append(gen_chirp(f0, f1))
    return np.stack(data)


# Identifiable train: each f0 → unique f1 (mapping rule: f1 = f0 + 3 mod 6)
# Map: 1→4, 2→5, 3→6, 4→1, 5→2, 6→3 (also 1→5, 2→6, 3→1, ...)
TRAIN_PAIRS = [
    (1, 4), (2, 5), (3, 6), (4, 1), (5, 2), (6, 3),  # +3 mapping
    (1, 5), (2, 6), (3, 1), (4, 2), (5, 3), (6, 4),  # +4 mapping
]
# Each f0 has 2 f1 values, but distinct enough that long context can disambiguate

# OOD-C: new (f0, f1) combos within same range
TEST_OOD_COMP = [
    (1, 3), (2, 4), (3, 5), (4, 6), (5, 1), (6, 2),  # +2 mapping
    (1, 6), (2, 1), (3, 4), (4, 5), (5, 6), (6, 1),  # +5/wrap
]

# OOD-E: range outside [1, 6]
TEST_OOD_EXTR = [
    (1, 8), (8, 1), (0.5, 7), (7, 0.5), (8, 9), (9, 8),
]


# ============================================================
# Train (long context)
# ============================================================
def train_model(model, train_data, epochs=120, lr=5e-4, n_query=N_QUERY, batch_size=64):
    opt = optim.Adam(model.parameters(), lr=lr)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    losses = []
    N = len(train_data)
    bpe = max(50, N // batch_size)

    for ep in range(epochs):
        model.train(); ls = []
        for _ in range(bpe):
            idxs = np.random.choice(N, batch_size)
            batch = train_data[idxs]            # [B, 384]
            ctxs = batch[:, :SEQ_LEN]           # [B, 192]
            futures = batch[:, SEQ_LEN:]        # [B, 192]

            if np.random.rand() < 0.5:
                # Forecast
                qi = np.random.choice(PRED_LEN, n_query, replace=False)
                qt = 1.0 + qi.astype(np.float32) / PRED_LEN
                qv = futures[:, qi]
                qt_b = np.tile(qt, (batch_size, 1))
                ctx_t = ctxs
            else:
                # Imputation
                masks = np.random.rand(batch_size, SEQ_LEN) > 0.375
                qt_b = np.zeros((batch_size, n_query), dtype=np.float32)
                qv = np.zeros((batch_size, n_query), dtype=np.float32)
                ctx_t = ctxs * masks.astype(np.float32)
                for b in range(batch_size):
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
    fc, imp = [], []
    with torch.no_grad():
        for w in data[:500]:
            ctx = torch.tensor(w[:SEQ_LEN]).unsqueeze(0).float().to(DEVICE)
            pred = model.forecast(ctx).cpu().numpy()[0]
            tgt = w[SEQ_LEN:SEQ_LEN+PRED_LEN]
            fc.append(np.mean((pred[:len(tgt)] - tgt)**2))

            mask = (np.random.rand(SEQ_LEN) > 0.375).astype(np.float32)
            full_t = torch.tensor(w[:SEQ_LEN]).unsqueeze(0).float().to(DEVICE)
            t_imp = torch.linspace(0, 1, SEQ_LEN, device=DEVICE).unsqueeze(0)
            recon = model._query(model.enc(full_t * torch.tensor(mask).to(DEVICE)), t_imp).cpu().numpy()[0]
            imp.append(np.mean((recon[mask==0] - w[:SEQ_LEN][mask==0])**2))
    fc_m, imp_m = np.mean(fc), np.mean(imp)
    print(f'  [{label}] FC={fc_m:.4f} IMP={imp_m:.4f}')
    return fc_m, imp_m


# ============================================================
# Visualization
# ============================================================
def visualize(model, id_data, ood_comp, ood_extr, losses, results):
    fig = plt.figure(figsize=(20, 14))
    n_trunk = len(model.trunk_names)
    trunk_colors = ['#3b82f6', '#10b981', '#f59e0b', '#a855f7'][:n_trunk]

    ax = plt.subplot(4, 4, 1)
    ax.plot(losses); ax.set_title('Train Loss'); ax.set_yscale('log')
    ax.set_xlabel('Epoch'); ax.grid(alpha=0.3)

    ax = plt.subplot(4, 4, 2)
    metrics = ['ID', 'OOD-C', 'OOD-E']
    fc_v = [results['fc_id'], results['fc_ood_c'], results['fc_ood_e']]
    imp_v = [results['imp_id'], results['imp_ood_c'], results['imp_ood_e']]
    x = np.arange(3); w = 0.35
    ax.bar(x - w/2, fc_v, w, label='FC', color='#3b82f6')
    ax.bar(x + w/2, imp_v, w, label='IMP', color='#10b981')
    ax.set_xticks(x); ax.set_xticklabels(metrics)
    ax.set_title('ID vs OOD'); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis='y')

    ax = plt.subplot(4, 4, 3); ax.axis('off')
    txt = (
        f'Long context (192) +\n'
        f'Identifiable pairs:\n\n'
        f'Train: {len(TRAIN_PAIRS)} (f0,f1)\n'
        f'  +3 mapping: 1→4..\n'
        f'  +4 mapping: 1→5..\n\n'
        f'OOD-C: +2/+5 mapping\n'
        f'OOD-E: f∈{{0.5,7,8,9}}\n\n'
        f'FC gap C: ×{results["fc_ood_c"]/results["fc_id"]:.2f}\n'
        f'FC gap E: ×{results["fc_ood_e"]/results["fc_id"]:.2f}\n'
        f'IMP gap C: ×{results["imp_ood_c"]/results["imp_id"]:.2f}\n'
        f'IMP gap E: ×{results["imp_ood_e"]/results["imp_id"]:.2f}\n\n'
        f'L4 v3 was 21x.\n'
        f'L4 v4 target: <5x'
    )
    ax.text(0.0, 0.5, txt, fontsize=8, family='monospace', va='center')

    ax = plt.subplot(4, 4, 4); ax.axis('off')

    # Forecast examples
    model.eval()
    examples = [(id_data, 'ID', 'blue'), (ood_comp, 'OOD-C', 'orange'), (ood_extr, 'OOD-E', 'red')]
    for col, (ds, name, color) in enumerate(examples):
        for j in range(2):
            idx = j * 30
            w = ds[idx]
            ctx = torch.tensor(w[:SEQ_LEN]).unsqueeze(0).float().to(DEVICE)
            with torch.no_grad():
                fp = model.forecast(ctx).cpu().numpy()[0]
            ax = plt.subplot(4, 4, 4 + col*2 + j + 1)
            ax.plot(range(TOTAL_LEN), w, 'k-', alpha=0.3)
            ax.plot(range(SEQ_LEN), w[:SEQ_LEN], 'k-', linewidth=1.2)
            ax.plot(range(SEQ_LEN, TOTAL_LEN), fp, color=color, linewidth=1.5)
            ax.axvline(SEQ_LEN, color='gray', linestyle='--', alpha=0.3)
            mse = np.mean((fp - w[SEQ_LEN:])**2)
            ax.set_title(f'{name}#{j} MSE={mse:.3f}', fontsize=8)
            ax.grid(alpha=0.2)

    # Per-trunk decomp
    for col, (ds, name) in enumerate([(id_data, 'ID'), (ood_comp, 'OOD-C'), (ood_extr, 'OOD-E')]):
        idx = col * 50
        w = ds[idx]
        ctx = torch.tensor(w[:SEQ_LEN]).unsqueeze(0).float().to(DEVICE)
        with torch.no_grad():
            per_trunk = model.per_trunk_forecast(ctx)
            full = model.forecast(ctx).cpu().numpy()[0]
        per_trunk_np = [pt.cpu().numpy()[0] for pt in per_trunk]

        ax = plt.subplot(4, 4, 10 + col + 1)
        ax.plot(range(SEQ_LEN, TOTAL_LEN), w[SEQ_LEN:], 'k--', alpha=0.5, label='GT', linewidth=2)
        ax.plot(range(SEQ_LEN, TOTAL_LEN), full, 'k-', alpha=0.6, label='sum', linewidth=1.2)
        for pt, tn, tc in zip(per_trunk_np, model.trunk_names, trunk_colors):
            ax.plot(range(SEQ_LEN, TOTAL_LEN), pt, color=tc, alpha=0.85, label=tn, linewidth=1)
        ax.axhline(0, color='k', alpha=0.2, linewidth=0.5)
        ax.set_title(f'Per-trunk ({name})', fontsize=8)
        ax.legend(fontsize=6); ax.grid(alpha=0.2)

    # FFT
    for col, (ds, name) in enumerate([(id_data, 'ID'), (ood_comp, 'OOD-C'), (ood_extr, 'OOD-E')]):
        idx = col * 60
        w = ds[idx]
        ctx = torch.tensor(w[:SEQ_LEN]).unsqueeze(0).float().to(DEVICE)
        with torch.no_grad():
            fp = model.forecast(ctx).cpu().numpy()[0]
        full_pred = np.concatenate([w[:SEQ_LEN], fp])
        fft_gt = np.abs(np.fft.rfft(w))
        fft_pred = np.abs(np.fft.rfft(full_pred))
        ax = plt.subplot(4, 4, 14 + col)
        ax.plot(fft_gt[:40], 'k-', label='GT', linewidth=1.2)
        ax.plot(fft_pred[:40], 'r-', alpha=0.7, label='pred', linewidth=1.2)
        ax.set_title(f'{name} FFT', fontsize=8)
        ax.legend(fontsize=7); ax.grid(alpha=0.2)

    plt.suptitle(
        f'L4 v4: Long Context (192) + Identifiable Chirp Pairs\n'
        f'FC: ID={results["fc_id"]:.4f}, OOD-C={results["fc_ood_c"]:.4f} (×{results["fc_ood_c"]/results["fc_id"]:.2f}), '
        f'OOD-E={results["fc_ood_e"]:.4f} (×{results["fc_ood_e"]/results["fc_id"]:.2f})',
        fontsize=11)
    plt.tight_layout()
    plt.savefig(f'{SAVE_DIR}/L4_v4_long_ctx.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {SAVE_DIR}/L4_v4_long_ctx.png')


if __name__ == '__main__':
    np.random.seed(42); torch.manual_seed(42)
    print('='*60)
    print('Curriculum L4 v4: Long context (192) + Identifiable pairs')
    print(f'  Train: {len(TRAIN_PAIRS)} pairs, OOD-C: {len(TEST_OOD_COMP)}, OOD-E: {len(TEST_OOD_EXTR)}')
    print('='*60)

    print('\nGenerating data (length 384)...')
    train_data = make_dataset(TRAIN_PAIRS, n_per_pair=250)
    id_data = make_dataset(TRAIN_PAIRS, n_per_pair=50)
    ood_comp = make_dataset(TEST_OOD_COMP, n_per_pair=50)
    ood_extr = make_dataset(TEST_OOD_EXTR, n_per_pair=50)
    print(f'  Train: {len(train_data)}, ID: {len(id_data)}, OOD-C: {len(ood_comp)}, OOD-E: {len(ood_extr)}')

    model = Operator().to(DEVICE)
    n = sum(p.numel() for p in model.parameters())
    print(f'\nModel: PatchAttn + 4 trunks (chirplet) — {n/1e6:.2f}M params')

    print('\nTraining...')
    t0 = time.time()
    losses = train_model(model, train_data, epochs=150, lr=5e-4)
    print(f'Training time: {time.time()-t0:.1f}s')

    print('\nEval:')
    fc_id, imp_id = eval_model(model, id_data, 'ID')
    fc_oc, imp_oc = eval_model(model, ood_comp, 'OOD-C')
    fc_oe, imp_oe = eval_model(model, ood_extr, 'OOD-E')

    results = {
        'fc_id': fc_id, 'imp_id': imp_id,
        'fc_ood_c': fc_oc, 'imp_ood_c': imp_oc,
        'fc_ood_e': fc_oe, 'imp_ood_e': imp_oe,
    }

    print(f'\nGap analysis:')
    print(f'  FC  Comp: {fc_oc/fc_id:.2f}x  | Extr: {fc_oe/fc_id:.2f}x')
    print(f'  IMP Comp: {imp_oc/imp_id:.2f}x  | Extr: {imp_oe/imp_id:.2f}x')
    print(f'\nvs L4 v3 baseline: FC Comp 21x → ?')

    visualize(model, id_data, ood_comp, ood_extr, losses, results)

    print('\n' + '='*60)
    print('L4 v4 DONE')
    print('='*60)
