"""
Curriculum Shared: Improved Operator (more RBF capacity, denser queries)

Changes from L1:
  - RBF: 10 centers → 30 centers, bandwidth 20 → 200 (σ≈0.05 vs 0.16)
  - Query points: 16 → 32 (better gradient signal for narrow features)
  - Trunk width: 64 (same)

목적: L3+ local features에서 RBF가 제대로 활성화되도록.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, math
from torch import optim

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAVE_DIR = 'analysis/figures/curriculum'; os.makedirs(SAVE_DIR, exist_ok=True)

SEQ_LEN = 96
PRED_LEN = 96
W = 64
H = 128
N_RBF = 30          # ← 10 → 30
RBF_BANDWIDTH = 200 # ← 20 → 200 (narrower kernels)
N_QUERY = 32        # ← 16 → 32


class HTrunk(nn.Module):
    def __init__(self, w, btype, nf=16, deg=5, nc=N_RBF, bw=RBF_BANDWIDTH):
        super().__init__(); self.w=w; self.btype=btype
        if btype=='fourier': self.nf=nf; self.idim=1+2*nf
        elif btype=='poly': self.deg=deg; self.idim=deg+1
        elif btype=='rbf':
            self.bw = bw
            self.register_buffer('centers', torch.linspace(0, 2, nc))
            self.idim = 1 + nc
        elif btype=='chirplet':
            # chirplet(t; f0, α) = sin/cos(2π f0 t + π α t²)
            # f0 ∈ {1,2,3,4,5,6,7,8}, α ∈ {-6,-3,0,3,6} → 8*5 = 40 pairs
            f0_grid = torch.tensor([1.,2.,3.,4.,5.,6.,7.,8.])
            a_grid = torch.tensor([-6.,-3.,0.,3.,6.])
            f0_mesh, a_mesh = torch.meshgrid(f0_grid, a_grid, indexing='ij')
            self.register_buffer('chirp_f0', f0_mesh.flatten())  # [40]
            self.register_buffer('chirp_alpha', a_mesh.flatten())  # [40]
            self.n_chirp = self.chirp_f0.shape[0]
            self.idim = 1 + 2*self.n_chirp  # 1 + 80 = 81
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
            return torch.cat([t, torch.exp(-self.bw*(t-self.centers.unsqueeze(0))**2)], dim=-1)
        elif self.btype=='chirplet':
            # phase = 2π f0 t + π α t²
            f0 = self.chirp_f0.unsqueeze(0)  # [1, 40]
            a = self.chirp_alpha.unsqueeze(0)  # [1, 40]
            phase = 2*math.pi*f0*t + math.pi*a*(t**2)  # [N, 40]
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


class PatchAttnEncoder(nn.Module):
    def __init__(self, patch_size=8, n_layers=3, nhead=4):
        super().__init__()
        self.patch_size = patch_size
        n_patches = SEQ_LEN // patch_size
        self.proj = nn.Linear(patch_size, H)
        self.pos = nn.Parameter(torch.randn(1, n_patches, H) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=H, nhead=nhead, dim_feedforward=H*4,
            dropout=0.1, activation='gelu', batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(H)

    def forward(self, x):
        B = x.shape[0]
        z = self.proj(x.view(B, -1, self.patch_size)) + self.pos
        z = self.tf(z)
        return self.norm(z.mean(dim=1))


class SmallOperator(nn.Module):
    def __init__(self, trunk_types=('fourier', 'poly', 'rbf'), encoder='mlp'):
        super().__init__()
        if encoder == 'mlp':
            self.enc = nn.Sequential(
                nn.Linear(SEQ_LEN, H), nn.GELU(),
                nn.Linear(H, H), nn.GELU(),
                nn.Linear(H, H), nn.GELU())
        elif encoder == 'pattn':
            self.enc = PatchAttnEncoder()
        self.encoder_name = encoder
        self.trunk_names = list(trunk_types)
        self.trunks = nn.ModuleList([HTrunk(W, t) for t in trunk_types])
        self.heads = nn.ModuleList([nn.Linear(H, t.odim) for t in self.trunks])
        for h in self.heads: nn.init.xavier_normal_(h.weight, gain=0.1)
        self.biases = nn.ParameterList([nn.Parameter(torch.zeros(1)) for _ in range(len(trunk_types))])

    def _query(self, z, qt):
        B, nq = qt.shape
        t_flat = qt.reshape(-1)
        return sum(hfwd(t, t_flat, h(z)) + b for t, h, b in zip(self.trunks, self.heads, self.biases))

    def _query_per_trunk(self, z, qt):
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
# Train
# ============================================================
def train_model(model, train_data, epochs=120, lr=5e-4, n_query=N_QUERY, batch_size=64):
    opt = optim.Adam(model.parameters(), lr=lr)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    losses_per_ep = []
    N = len(train_data)
    batches_per_ep = max(50, N // batch_size)

    for ep in range(epochs):
        model.train(); ls = []
        for _ in range(batches_per_ep):
            idxs = np.random.choice(N, batch_size)
            batch = train_data[idxs]
            ctxs = batch[:, :SEQ_LEN]
            futures = batch[:, SEQ_LEN:]

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
