"""
Curriculum L1: Sum of Sinusoids
y(t) = a1*sin(2π f1 t + φ1) + a2*sin(2π f2 t + φ2)

검증 질문:
  1. Operator가 두 frequency를 분리해서 학습하나?
  2. Composition test: 본 적 없는 frequency pair에 일반화되나?
  3. Per-trunk attribution: Fourier trunk가 dominant 한가?

CUDA_VISIBLE_DEVICES=3 python analysis/curriculum/L1_sum_sinusoids.py
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

SEQ_LEN = 96      # context length
PRED_LEN = 96     # forecast length (so total 192)
W = 64            # trunk width (small)
H = 128           # encoder hidden (small)


# ============================================================
# Data: sum of two sinusoids
# ============================================================
def gen_sample(freq_pair, n=192, t_max=2.0, noise=0.0):
    f1, f2 = freq_pair
    t = np.linspace(0, t_max, n)
    a1 = np.random.uniform(0.5, 1.5)
    a2 = np.random.uniform(0.5, 1.5)
    p1 = np.random.uniform(0, 2*np.pi)
    p2 = np.random.uniform(0, 2*np.pi)
    y = a1*np.sin(2*np.pi*f1*t + p1) + a2*np.sin(2*np.pi*f2*t + p2)
    if noise > 0:
        y = y + np.random.randn(n) * noise
    # Normalize per sample
    s = y.std()
    if s > 1e-6:
        y = (y - y.mean()) / s
    return y.astype(np.float32)


def make_dataset(freq_pairs, n_per_pair=200, noise=0.0):
    """For each freq pair, make n_per_pair samples."""
    data = []
    for fp in freq_pairs:
        for _ in range(n_per_pair):
            data.append(gen_sample(fp, noise=noise))
    return np.stack(data)


# Frequency splits
TRAIN_PAIRS = [(1,3), (2,5), (3,7), (1,5), (2,4), (4,7), (1,6), (3,6)]
TEST_ID_PAIRS = TRAIN_PAIRS  # in-distribution
TEST_OOD_PAIRS = [(1,4), (2,3), (3,5), (4,6), (5,7), (1,7), (2,6), (4,5)]


# ============================================================
# Small Operator Model
# ============================================================
class HTrunk(nn.Module):
    def __init__(self, w, btype, nf=16, deg=5, nc=10):
        super().__init__(); self.w=w; self.btype=btype
        if btype=='fourier': self.nf=nf; self.idim=1+2*nf
        elif btype=='poly': self.deg=deg; self.idim=deg+1
        elif btype=='rbf':
            self.register_buffer('centers', torch.linspace(0, 2, nc))
            self.idim=1+nc
        self.pc = self.idim*w + w
        self.odim = self.pc + w

    def feat(self, t):
        t = t.unsqueeze(-1) if t.dim()==1 else t
        if self.btype=='fourier':
            f = torch.arange(1, self.nf+1, device=t.device, dtype=t.dtype)
            return torch.cat([t, torch.sin(2*math.pi*f*t), torch.cos(2*math.pi*f*t)], dim=-1)
        elif self.btype=='poly':
            return torch.cat([t**i for i in range(self.deg+1)], dim=-1)
        elif self.btype=='rbf':
            return torch.cat([t, torch.exp(-20*(t-self.centers.unsqueeze(0))**2)], dim=-1)


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


class SmallOperator(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(SEQ_LEN, H), nn.GELU(),
            nn.Linear(H, H), nn.GELU(),
            nn.Linear(H, H), nn.GELU())
        self.trunks = nn.ModuleList([
            HTrunk(W, 'fourier'),
            HTrunk(W, 'poly'),
            HTrunk(W, 'rbf'),
        ])
        self.heads = nn.ModuleList([nn.Linear(H, t.odim) for t in self.trunks])
        for h in self.heads: nn.init.xavier_normal_(h.weight, gain=0.1)
        self.biases = nn.ParameterList([nn.Parameter(torch.zeros(1)) for _ in range(3)])

    def _query(self, z, qt):
        B, nq = qt.shape
        t_flat = qt.reshape(-1)
        return sum(hfwd(t, t_flat, h(z)) + b for t, h, b in zip(self.trunks, self.heads, self.biases))

    def _query_per_trunk(self, z, qt):
        """Return list of per-trunk outputs (for attribution)."""
        B, nq = qt.shape
        t_flat = qt.reshape(-1)
        return [hfwd(t, t_flat, h(z)) + b for t, h, b in zip(self.trunks, self.heads, self.biases)]

    def forward_train(self, ctx, qt):
        return self._query(self.enc(ctx), qt)

    def forecast(self, ctx, n=PRED_LEN):
        t = torch.linspace(1, 2, n, device=ctx.device).unsqueeze(0).expand(ctx.shape[0], -1)
        return self._query(self.enc(ctx), t)

    def impute(self, ctx, n=SEQ_LEN):
        t = torch.linspace(0, 1, n, device=ctx.device).unsqueeze(0).expand(ctx.shape[0], -1)
        return self._query(self.enc(ctx), t)

    def per_trunk_forecast(self, ctx, n=PRED_LEN):
        t = torch.linspace(1, 2, n, device=ctx.device).unsqueeze(0).expand(ctx.shape[0], -1)
        return self._query_per_trunk(self.enc(ctx), t)


# ============================================================
# Train (True OP point-wise)
# ============================================================
def train_model(model, train_data, epochs=100, lr=5e-4, n_query=16, batch_size=64):
    opt = optim.Adam(model.parameters(), lr=lr)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    losses_per_ep = []
    N = len(train_data)
    batches_per_ep = max(50, N // batch_size)

    for ep in range(epochs):
        model.train(); ls = []
        for _ in range(batches_per_ep):
            idxs = np.random.choice(N, batch_size)
            batch = train_data[idxs]  # [B, 192]
            ctxs = batch[:, :SEQ_LEN]
            futures = batch[:, SEQ_LEN:]

            # Random task
            if np.random.rand() < 0.5:
                # Forecast: query in [1, 2)
                qi = np.random.choice(PRED_LEN, n_query, replace=False)
                qt = 1.0 + qi.astype(np.float32) / PRED_LEN  # [n_query]
                qv = futures[:, qi]  # [B, n_query]
                qt_b = np.tile(qt, (batch_size, 1))
                ctx_t = ctxs
            else:
                # Imputation: random mask, query at masked points
                masks = np.random.rand(batch_size, SEQ_LEN) > 0.375
                qt_b = np.zeros((batch_size, n_query), dtype=np.float32)
                qv = np.zeros((batch_size, n_query), dtype=np.float32)
                ctx_t = (ctxs * masks.astype(np.float32))
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
            pred = model.forward_train(c, q)
            loss = F.mse_loss(pred, v)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ls.append(loss.item())

        sched.step()
        avg = np.mean(ls)
        losses_per_ep.append(avg)
        if ep % 20 == 0 or ep == epochs-1:
            print(f'  Ep {ep+1}/{epochs}: loss={avg:.5f} lr={sched.get_last_lr()[0]:.5f}')

    return losses_per_ep


# ============================================================
# Eval
# ============================================================
def eval_model(model, data, label=''):
    model.eval()
    fc_mses, imp_mses = [], []
    with torch.no_grad():
        for w in data[:500]:
            ctx = torch.tensor(w[:SEQ_LEN]).unsqueeze(0).float().to(DEVICE)
            pred = model.forecast(ctx).cpu().numpy()[0]
            tgt = w[SEQ_LEN:SEQ_LEN+PRED_LEN]
            fc_mses.append(np.mean((pred[:len(tgt)] - tgt)**2))

            mask = (np.random.rand(SEQ_LEN) > 0.375).astype(np.float32)
            full_t = torch.tensor(w[:SEQ_LEN]).unsqueeze(0).float().to(DEVICE)
            recon = model.impute(full_t * torch.tensor(mask).to(DEVICE)).cpu().numpy()[0]
            imp_mses.append(np.mean((recon[mask==0] - w[:SEQ_LEN][mask==0])**2))
    fc, imp = np.mean(fc_mses), np.mean(imp_mses)
    print(f'  [{label}] FC={fc:.4f} IMP={imp:.4f}')
    return fc, imp


# ============================================================
# Visualization
# ============================================================
def visualize(model, train_data, id_data, ood_data, losses, fc_id, imp_id, fc_ood, imp_ood, train_pairs, ood_pairs):
    fig = plt.figure(figsize=(20, 14))

    # Row 1: Loss curve + In-vs-OOD bars
    ax1 = plt.subplot(4, 5, 1)
    ax1.plot(losses)
    ax1.set_title('Train Loss')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('MSE')
    ax1.grid(alpha=0.3); ax1.set_yscale('log')

    ax2 = plt.subplot(4, 5, 2)
    metrics = ['FC ID', 'FC OOD', 'IMP ID', 'IMP OOD']
    vals = [fc_id, fc_ood, imp_id, imp_ood]
    colors = ['#3b82f6', '#ef4444', '#3b82f6', '#ef4444']
    bars = ax2.bar(metrics, vals, color=colors)
    ax2.set_title('ID vs OOD')
    ax2.grid(alpha=0.3, axis='y')
    for b, v in zip(bars, vals):
        ax2.text(b.get_x() + b.get_width()/2, v, f'{v:.3f}',
                 ha='center', va='bottom', fontsize=8)

    # Row 1 cont: composition gap text
    ax3 = plt.subplot(4, 5, 3)
    ax3.axis('off')
    txt = f'Train pairs: {train_pairs}\n\nOOD pairs: {ood_pairs}\n\n'
    txt += f'FC OOD/ID = {fc_ood/fc_id:.2f}x\n'
    txt += f'IMP OOD/ID = {imp_ood/imp_id:.2f}x\n\n'
    txt += '<2x = generalizes\n>5x = memorizes'
    ax3.text(0.05, 0.5, txt, fontsize=9, family='monospace', va='center')

    # Row 1 cont: ID examples (forecast)
    model.eval()
    for col_idx, (sample_set, name, color) in enumerate([(id_data, 'ID', 'blue'), (ood_data, 'OOD', 'red')]):
        for j in range(2):
            idx = j * 50
            w = sample_set[idx]
            ctx = torch.tensor(w[:SEQ_LEN]).unsqueeze(0).float().to(DEVICE)
            with torch.no_grad():
                fp = model.forecast(ctx).cpu().numpy()[0]

            ax = plt.subplot(4, 5, 5*1 + col_idx*2 + j + 1)
            ax.plot(range(192), w, 'k-', alpha=0.3, label='GT')
            ax.plot(range(SEQ_LEN), w[:SEQ_LEN], 'k-', linewidth=1.5, label='ctx')
            ax.plot(range(SEQ_LEN, SEQ_LEN+PRED_LEN), fp, color=color, linewidth=1.5, label='pred')
            ax.axvline(SEQ_LEN, color='gray', linestyle='--', alpha=0.3)
            mse = np.mean((fp - w[SEQ_LEN:SEQ_LEN+PRED_LEN])**2)
            ax.set_title(f'{name} #{j}  FC MSE={mse:.3f}', fontsize=9)
            ax.legend(fontsize=7)

    # Row 3: Per-trunk attribution (3 ID samples)
    trunk_names = ['Fourier', 'Poly', 'RBF']
    trunk_colors = ['#3b82f6', '#10b981', '#f59e0b']
    for j in range(3):
        idx = j * 100
        w = id_data[idx]
        ctx = torch.tensor(w[:SEQ_LEN]).unsqueeze(0).float().to(DEVICE)
        with torch.no_grad():
            per_trunk = model.per_trunk_forecast(ctx)
            full = model.forecast(ctx).cpu().numpy()[0]
        per_trunk_np = [pt.cpu().numpy()[0] for pt in per_trunk]

        ax = plt.subplot(4, 5, 5*2 + j + 1)
        ax.plot(range(SEQ_LEN, SEQ_LEN+PRED_LEN), w[SEQ_LEN:], 'k--', alpha=0.5, label='GT', linewidth=2)
        ax.plot(range(SEQ_LEN, SEQ_LEN+PRED_LEN), full, 'k-', alpha=0.7, label='sum', linewidth=1.5)
        for k, (pt, tn, tc) in enumerate(zip(per_trunk_np, trunk_names, trunk_colors)):
            ax.plot(range(SEQ_LEN, SEQ_LEN+PRED_LEN), pt, color=tc, alpha=0.8, label=tn, linewidth=1)
        ax.set_title(f'Per-Trunk Decomp ID#{j}', fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.2)

    # Row 4: FFT comparison (ID vs OOD)
    for j, (sample_set, name) in enumerate([(id_data, 'ID'), (ood_data, 'OOD')]):
        for col in range(2):
            idx = col * 80
            w = sample_set[idx]
            ctx = torch.tensor(w[:SEQ_LEN]).unsqueeze(0).float().to(DEVICE)
            with torch.no_grad():
                fp = model.forecast(ctx).cpu().numpy()[0]
            full_pred = np.concatenate([w[:SEQ_LEN], fp])
            fft_gt = np.abs(np.fft.rfft(w))
            fft_pred = np.abs(np.fft.rfft(full_pred))

            ax = plt.subplot(4, 5, 5*3 + j*2 + col + 1)
            ax.plot(fft_gt[:25], 'k-', label='GT', linewidth=1.5)
            ax.plot(fft_pred[:25], 'r-', alpha=0.7, label='pred', linewidth=1.5)
            ax.set_title(f'{name}#{col} FFT', fontsize=9)
            ax.set_xlabel('freq idx'); ax.legend(fontsize=7)
            ax.grid(alpha=0.2)

    plt.suptitle(
        f'L1: Sum of Sinusoids — Operator Composition Test\n'
        f'FC: ID={fc_id:.4f}, OOD={fc_ood:.4f} (×{fc_ood/fc_id:.2f}) | '
        f'IMP: ID={imp_id:.4f}, OOD={imp_ood:.4f} (×{imp_ood/imp_id:.2f})',
        fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{SAVE_DIR}/L1_sum_sinusoids.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {SAVE_DIR}/L1_sum_sinusoids.png')


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    np.random.seed(42); torch.manual_seed(42)
    print('='*60)
    print('Curriculum L1: Sum of Sinusoids')
    print(f'  Train pairs: {TRAIN_PAIRS}')
    print(f'  OOD pairs:   {TEST_OOD_PAIRS}')
    print('='*60)

    print('\nGenerating data...')
    train_data = make_dataset(TRAIN_PAIRS, n_per_pair=300)
    id_data = make_dataset(TEST_ID_PAIRS, n_per_pair=80)
    ood_data = make_dataset(TEST_OOD_PAIRS, n_per_pair=80)
    print(f'  Train: {len(train_data)}, ID test: {len(id_data)}, OOD test: {len(ood_data)}')

    model = SmallOperator().to(DEVICE)
    n = sum(p.numel() for p in model.parameters())
    print(f'\nModel: {n/1e6:.2f}M params')

    print('\nTraining...')
    t0 = time.time()
    losses = train_model(model, train_data, epochs=100, lr=5e-4)
    elapsed = time.time() - t0
    print(f'Training time: {elapsed:.1f}s')

    print('\nEval:')
    fc_id, imp_id = eval_model(model, id_data, 'ID')
    fc_ood, imp_ood = eval_model(model, ood_data, 'OOD')

    print(f'\nComposition gap:')
    print(f'  FC OOD/ID = {fc_ood/fc_id:.2f}x')
    print(f'  IMP OOD/ID = {imp_ood/imp_id:.2f}x')

    visualize(model, train_data, id_data, ood_data, losses,
              fc_id, imp_id, fc_ood, imp_ood,
              TRAIN_PAIRS, TEST_OOD_PAIRS)

    print('\n' + '='*60)
    print('L1 DONE')
    print('='*60)
