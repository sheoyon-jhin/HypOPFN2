"""
Curriculum L5: Split Encoder per Trunk

Hypothesis: Single shared encoder가 bottleneck. Trunk마다 다른 encoder를 두면
각자 specialize 가능 → chirp 같은 어려운 신호도 풀릴 수 있음.

비교 (L4 chirp 데이터, long context 192):
  v3 (이전): Shared MLP encoder + 4 trunks      → gap 21x
  v4 (이전): Shared PAttn encoder + 4 trunks    → gap 141x (worse)
  v5a (this): Split encoder, MLP per trunk      → ?
  v5b (this): Split encoder, PAttn per trunk    → ?

CUDA_VISIBLE_DEVICES=3 python analysis/curriculum/L5_split_encoder.py
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

# Long context (from L4 v4)
SEQ_LEN = 192
PRED_LEN = 192
TOTAL_LEN = 384
W = 64
H = 192
N_QUERY = 32


# ============================================================
# Trunks (same as L4 v4)
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
# Encoders
# ============================================================
class MLPEnc(nn.Module):
    def __init__(self, h=H):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(SEQ_LEN, h), nn.GELU(),
            nn.Linear(h, h), nn.GELU(),
            nn.Linear(h, h), nn.GELU())
    def forward(self, x): return self.net(x)


class PatchAttnEnc(nn.Module):
    def __init__(self, ps=16, nl=4, h=H):
        super().__init__()
        self.ps = ps
        np_ = SEQ_LEN // ps
        self.proj = nn.Linear(ps, h)
        self.pos = nn.Parameter(torch.randn(1, np_, h)*0.02)
        layer = nn.TransformerEncoderLayer(d_model=h, nhead=4, dim_feedforward=h*4,
                                            dropout=0.1, activation='gelu',
                                            batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(layer, num_layers=nl)
        self.norm = nn.LayerNorm(h)
    def forward(self, x):
        B = x.shape[0]
        z = self.proj(x.view(B, -1, self.ps)) + self.pos
        return self.norm(self.tf(z).mean(dim=1))


# ============================================================
# Operators
# ============================================================
class SharedEncoderOp(nn.Module):
    """Baseline: shared encoder + multiple heads"""
    def __init__(self, trunk_types, encoder_cls):
        super().__init__()
        self.enc = encoder_cls()
        self.trunk_names = list(trunk_types)
        self.trunks = nn.ModuleList([HTrunk(W, t) for t in trunk_types])
        self.heads = nn.ModuleList([nn.Linear(H, t.odim) for t in self.trunks])
        for h in self.heads: nn.init.xavier_normal_(h.weight, gain=0.1)
        self.biases = nn.ParameterList([nn.Parameter(torch.zeros(1)) for _ in trunk_types])

    def _query(self, ctx, qt):
        z = self.enc(ctx)
        t_flat = qt.reshape(-1)
        return sum(hfwd(t, t_flat, h(z)) + b for t, h, b in zip(self.trunks, self.heads, self.biases))

    def _query_per_trunk(self, ctx, qt):
        z = self.enc(ctx)
        t_flat = qt.reshape(-1)
        return [hfwd(t, t_flat, h(z)) + b for t, h, b in zip(self.trunks, self.heads, self.biases)]

    def forward_train(self, ctx, qt): return self._query(ctx, qt)

    def forecast(self, ctx, n=PRED_LEN):
        t = torch.linspace(1, 2, n, device=ctx.device).unsqueeze(0).expand(ctx.shape[0], -1)
        return self._query(ctx, t)


class SplitEncoderOp(nn.Module):
    """L5: Each trunk has its own encoder"""
    def __init__(self, trunk_types, encoder_cls):
        super().__init__()
        self.encs = nn.ModuleList([encoder_cls() for _ in trunk_types])
        self.trunk_names = list(trunk_types)
        self.trunks = nn.ModuleList([HTrunk(W, t) for t in trunk_types])
        self.heads = nn.ModuleList([nn.Linear(H, t.odim) for t in self.trunks])
        for h in self.heads: nn.init.xavier_normal_(h.weight, gain=0.1)
        self.biases = nn.ParameterList([nn.Parameter(torch.zeros(1)) for _ in trunk_types])

    def _query(self, ctx, qt):
        t_flat = qt.reshape(-1)
        out = 0
        for enc, trunk, head, bias in zip(self.encs, self.trunks, self.heads, self.biases):
            z = enc(ctx)
            out = out + hfwd(trunk, t_flat, head(z)) + bias
        return out

    def _query_per_trunk(self, ctx, qt):
        t_flat = qt.reshape(-1)
        outs = []
        for enc, trunk, head, bias in zip(self.encs, self.trunks, self.heads, self.biases):
            z = enc(ctx)
            outs.append(hfwd(trunk, t_flat, head(z)) + bias)
        return outs

    def forward_train(self, ctx, qt): return self._query(ctx, qt)

    def forecast(self, ctx, n=PRED_LEN):
        t = torch.linspace(1, 2, n, device=ctx.device).unsqueeze(0).expand(ctx.shape[0], -1)
        return self._query(ctx, t)


# ============================================================
# Data (same as L4 v4)
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
    return np.stack([gen_chirp(f0, f1) for f0, f1 in pairs for _ in range(n_per_pair)])


TRAIN_PAIRS = [
    (1, 4), (2, 5), (3, 6), (4, 1), (5, 2), (6, 3),
    (1, 5), (2, 6), (3, 1), (4, 2), (5, 3), (6, 4),
]
TEST_OOD_COMP = [
    (1, 3), (2, 4), (3, 5), (4, 6), (5, 1), (6, 2),
    (1, 6), (2, 1), (3, 4), (4, 5), (5, 6), (6, 1),
]
TEST_OOD_EXTR = [(1, 8), (8, 1), (0.5, 7), (7, 0.5), (8, 9), (9, 8)]


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
        for w in data[:500]:
            ctx = torch.tensor(w[:SEQ_LEN]).unsqueeze(0).float().to(DEVICE)
            pred = model.forecast(ctx).cpu().numpy()[0]
            tgt = w[SEQ_LEN:SEQ_LEN+PRED_LEN]
            fc.append(np.mean((pred[:len(tgt)] - tgt)**2))
    fc_m = np.mean(fc)
    print(f'  [{label}] FC={fc_m:.4f}')
    return fc_m


def run(label, model_cls, encoder_cls, train_data, id_data, ood_c, ood_e):
    print(f'\n{"="*60}')
    print(f'[{label}]')
    print('='*60)
    torch.manual_seed(42); np.random.seed(42)
    model = model_cls(('fourier','poly','rbf','chirplet'), encoder_cls).to(DEVICE)
    n = sum(p.numel() for p in model.parameters())
    print(f'Params: {n/1e6:.2f}M')
    t0 = time.time()
    losses = train_model(model, train_data, epochs=150, lr=5e-4)
    print(f'Time: {time.time()-t0:.1f}s')
    fc_id = eval_model(model, id_data, 'ID')
    fc_oc = eval_model(model, ood_c, 'OOD-C')
    fc_oe = eval_model(model, ood_e, 'OOD-E')
    return {'label': label, 'losses': losses, 'params': n,
            'fc_id': fc_id, 'fc_ood_c': fc_oc, 'fc_ood_e': fc_oe,
            'model': model}


def viz_compare(results, id_data, ood_c, ood_e):
    fig = plt.figure(figsize=(20, 12))
    n_runs = len(results)

    # Bar charts
    ax = plt.subplot(3, 4, 1)
    labels = [r['label'] for r in results]
    fc_id = [r['fc_id'] for r in results]
    fc_oc = [r['fc_ood_c'] for r in results]
    x = np.arange(len(labels)); w = 0.35
    ax.bar(x - w/2, fc_id, w, label='ID', color='#3b82f6')
    ax.bar(x + w/2, fc_oc, w, label='OOD-C', color='#ef4444')
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15, fontsize=8)
    ax.set_title('FC MSE', fontsize=10); ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis='y')

    ax = plt.subplot(3, 4, 2)
    gaps = [r['fc_ood_c']/r['fc_id'] for r in results]
    bars = ax.bar(x, gaps, color=['#94a3b8','#94a3b8','#3b82f6','#10b981'][:n_runs])
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15, fontsize=8)
    ax.set_title('Compositional gap', fontsize=10)
    ax.grid(alpha=0.3, axis='y')
    for b, v in zip(bars, gaps):
        ax.text(b.get_x()+b.get_width()/2, v, f'{v:.1f}x', ha='center', va='bottom', fontsize=8)

    ax = plt.subplot(3, 4, 3)
    for r in results:
        ax.plot(r['losses'], label=r['label'])
    ax.set_yscale('log'); ax.set_title('Train loss', fontsize=10)
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    ax = plt.subplot(3, 4, 4); ax.axis('off')
    txt = 'L5 split encoder ablation\n\n'
    for r in results:
        txt += (f'{r["label"]}\n'
                f'  ID: {r["fc_id"]:.4f}\n'
                f'  OOD-C: {r["fc_ood_c"]:.4f}\n'
                f'  gap: ×{r["fc_ood_c"]/r["fc_id"]:.1f}\n'
                f'  Params: {r["params"]/1e6:.2f}M\n\n')
    ax.text(0, 1, txt, fontsize=7, family='monospace', va='top')

    # Forecast examples per model
    examples = [(id_data, 'ID', 'blue'), (ood_c, 'OOD-C', 'orange'), (ood_e, 'OOD-E', 'red')]
    for row_idx, (ds, name, color) in enumerate(examples):
        for col_idx, r in enumerate(results):
            model = r['model']; model.eval()
            idx = 30
            w = ds[idx]
            ctx = torch.tensor(w[:SEQ_LEN]).unsqueeze(0).float().to(DEVICE)
            with torch.no_grad():
                fp = model.forecast(ctx).cpu().numpy()[0]
            ax = plt.subplot(3, 4, 4 + row_idx*4 + col_idx + 1)
            ax.plot(range(TOTAL_LEN), w, 'k-', alpha=0.3)
            ax.plot(range(SEQ_LEN), w[:SEQ_LEN], 'k-', linewidth=1.0)
            ax.plot(range(SEQ_LEN, TOTAL_LEN), fp, color=color, linewidth=1.5)
            ax.axvline(SEQ_LEN, color='gray', linestyle='--', alpha=0.3)
            mse = np.mean((fp - w[SEQ_LEN:])**2)
            ax.set_title(f'{r["label"]} {name}\nMSE={mse:.3f}', fontsize=7)

    plt.suptitle('L5: Split Encoder per Trunk (chirp data)', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{SAVE_DIR}/L5_split_encoder.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {SAVE_DIR}/L5_split_encoder.png')


if __name__ == '__main__':
    np.random.seed(42); torch.manual_seed(42)
    print('='*60)
    print('Curriculum L5: Split Encoder per Trunk')
    print('='*60)

    print('\nGenerating chirp data...')
    train_data = make_dataset(TRAIN_PAIRS, n_per_pair=250)
    id_data = make_dataset(TRAIN_PAIRS, n_per_pair=50)
    ood_c = make_dataset(TEST_OOD_COMP, n_per_pair=50)
    ood_e = make_dataset(TEST_OOD_EXTR, n_per_pair=50)
    print(f'Train: {len(train_data)}')

    runs = [
        ('Shared MLP',   SharedEncoderOp, MLPEnc),
        ('Shared PAttn', SharedEncoderOp, PatchAttnEnc),
        ('Split MLP',    SplitEncoderOp,  MLPEnc),
        ('Split PAttn',  SplitEncoderOp,  PatchAttnEnc),
    ]

    results = []
    for label, mc, ec in runs:
        r = run(label, mc, ec, train_data, id_data, ood_c, ood_e)
        results.append(r)

    print('\n' + '='*60)
    print('SUMMARY')
    print('='*60)
    print(f'{"label":<14} {"params":<10} {"ID":<10} {"OOD-C":<10} {"gap":<10}')
    print('-'*60)
    for r in results:
        gap = r['fc_ood_c'] / r['fc_id']
        print(f'{r["label"]:<14} {r["params"]/1e6:<10.2f} {r["fc_id"]:<10.4f} {r["fc_ood_c"]:<10.4f} {gap:<10.2f}')
    print('='*60)

    viz_compare(results, id_data, ood_c, ood_e)
    print('\nL5 DONE')
