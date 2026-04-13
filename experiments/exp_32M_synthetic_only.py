"""
32.5M Synthetic-Only Training — TRUE Zero-shot을 위한 공정한 모델

학습: TempoPFN ONLY (순수 synthetic — ETT/M4/Weather/UCR 전혀 없음)
모델: 32.5M 원본과 동일 architecture
평가: ETT, Weather, M4, Solar, PEMS → 전부 true zero-shot

+ Latent Decomposition (FeDaL inspired): encoder output을 trend/seasonal로 분해

Usage:
  CUDA_VISIBLE_DEVICES=2 python experiments/exp_32M_synthetic_only.py --tag synth_only
  CUDA_VISIBLE_DEVICES=2 python experiments/exp_32M_synthetic_only.py --tag synth_latent_decomp --latent_decomp 1
"""
import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, math, time, pyarrow as pa
from torch import optim
from torch.utils.data import Dataset, DataLoader

DEVICE = torch.device(os.environ.get('CUDA_DEV', 'cuda'))

SEQ_LEN = 96
PATCH_SIZE = 16
HIDDEN = 512
N_LAYERS = 6
WIDTH = 192
N_FREQ = 32
N_RBF = 20
TOP_K_IQ = 5
INFORMED_DIM = 1 + 2 * TOP_K_IQ + 2


# ============================================================
# Reuse core components from 32.5M
# ============================================================
def extract_freq(x, top_k=TOP_K_IQ):
    fft = torch.fft.rfft(x, dim=-1)
    mag = fft.abs(); mag[:, 0] = 0
    idx = torch.topk(mag, top_k, dim=-1).indices
    phase = torch.angle(torch.gather(fft, 1, idx))
    return idx.float(), phase, x[:, -1:].detach(), (x[:, -1:] - x[:, -2:-1]).detach()


class HyperTrunk(nn.Module):
    def __init__(self, w, btype, nf=N_FREQ, deg=6, nc=N_RBF):
        super().__init__(); self.w = w; self.btype = btype
        if btype == 'fourier': self.nf = nf; self.idim = 1 + 2 * nf
        elif btype == 'poly': self.deg = deg; self.idim = deg + 1
        elif btype == 'rbf':
            self.register_buffer('centers', torch.linspace(0, 2, nc))
            self.idim = 1 + nc
        self.informed_dim = INFORMED_DIM
        self.full_idim = self.idim + self.informed_dim
        self.pc = self.full_idim * w + w
        self.odim = self.pc + w

    def base_feat(self, t):
        t = t.unsqueeze(-1) if t.dim() == 1 else t
        if self.btype == 'fourier':
            f = torch.arange(1, self.nf + 1, device=t.device, dtype=t.dtype)
            return torch.cat([t, torch.sin(2*math.pi*f*t), torch.cos(2*math.pi*f*t)], dim=-1)
        elif self.btype == 'poly':
            return torch.cat([t**i for i in range(self.deg + 1)], dim=-1)
        elif self.btype == 'rbf':
            return torch.cat([t, torch.exp(-20*(t-self.centers.unsqueeze(0))**2)], dim=-1)


def hyper_forward_iq(trunk, t_flat, head_output, iq_features):
    B = head_output.shape[0]
    base = trunk.base_feat(t_flat)
    nq = base.shape[0] // B
    base = base.view(B, nq, trunk.idim)
    full = torch.cat([base, iq_features], dim=-1)
    tp = head_output[:, :trunk.pc] * 0.01
    W = tp[:, :trunk.full_idim*trunk.w].view(B, trunk.full_idim, trunk.w)
    b = tp[:, trunk.full_idim*trunk.w:].view(B, trunk.w)
    Phi = F.gelu(torch.bmm(full, W) + b.unsqueeze(1))
    Bc = head_output[:, trunk.pc:]
    return torch.einsum('bw,bqw->bq', Bc, Phi)


class PatchAttnEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        n_patches = SEQ_LEN // PATCH_SIZE
        self.proj = nn.Linear(PATCH_SIZE, HIDDEN)
        self.pos_emb = nn.Parameter(torch.randn(1, n_patches, HIDDEN) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=HIDDEN, nhead=8, dim_feedforward=HIDDEN*4,
            dropout=0.1, activation='gelu', batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=N_LAYERS)
        self.norm = nn.LayerNorm(HIDDEN)

    def forward(self, x):
        B = x.shape[0]
        patches = x.view(B, -1, PATCH_SIZE)
        z = self.proj(patches) + self.pos_emb
        z = self.transformer(z)
        return self.norm(z.mean(dim=1))


# ============================================================
# Latent Decomposition (FeDaL-inspired)
# ============================================================
class LatentDecomp(nn.Module):
    """Encoder output z를 trend/seasonal로 분해 (FeDaL DBE 아이디어)."""
    def __init__(self, hidden=HIDDEN):
        super().__init__()
        self.trend_proj = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.seasonal_proj = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, hidden))

    def forward(self, z):
        z_trend = self.trend_proj(z)
        z_seasonal = self.seasonal_proj(z)
        return z_trend, z_seasonal


# ============================================================
# Model
# ============================================================
class Model32MSynthetic(nn.Module):
    def __init__(self, use_latent_decomp=False):
        super().__init__()
        self.encoder = PatchAttnEncoder()
        self.use_latent_decomp = use_latent_decomp

        if use_latent_decomp:
            self.latent_decomp = LatentDecomp()
            # Trend → Poly trunk, Seasonal → Fourier trunk, Shared → RBF trunk
            self.trunks = nn.ModuleList([
                HyperTrunk(WIDTH, 'poly'),     # trend
                HyperTrunk(WIDTH, 'fourier'),  # seasonal
                HyperTrunk(WIDTH, 'rbf'),      # residual (from full z)
            ])
            self.heads = nn.ModuleList([
                nn.Linear(HIDDEN, self.trunks[0].odim),  # trend head ← z_trend
                nn.Linear(HIDDEN, self.trunks[1].odim),  # seasonal head ← z_seasonal
                nn.Linear(HIDDEN, self.trunks[2].odim),  # residual head ← z (full)
            ])
        else:
            self.trunks = nn.ModuleList([
                HyperTrunk(WIDTH, 'fourier'),
                HyperTrunk(WIDTH, 'poly'),
                HyperTrunk(WIDTH, 'rbf'),
            ])
            self.heads = nn.ModuleList([nn.Linear(HIDDEN, t.odim) for t in self.trunks])

        for h in self.heads:
            nn.init.xavier_normal_(h.weight, gain=0.1)
            nn.init.constant_(h.bias, 0)
        self.biases = nn.ParameterList([nn.Parameter(torch.zeros(1)) for _ in self.trunks])

    def _build_iq(self, ctx_n, qt):
        freqs, phases, lv, ls = extract_freq(ctx_n)
        B, nq = qt.shape
        t_exp = qt.unsqueeze(-1)
        f = freqs.unsqueeze(1)
        p = phases.unsqueeze(1)
        ang = 2 * math.pi * f * t_exp + p
        return torch.cat([
            t_exp, torch.sin(ang), torch.cos(ang),
            lv.unsqueeze(1).expand(-1, nq, -1),
            ls.unsqueeze(1).expand(-1, nq, -1),
        ], dim=-1)

    def forward_train(self, ctx, qt):
        z = self.encoder(ctx)
        iq = self._build_iq(ctx, qt)
        t_flat = qt.reshape(-1)

        if self.use_latent_decomp:
            z_trend, z_seasonal = self.latent_decomp(z)
            # Trend trunk ← z_trend, Seasonal trunk ← z_seasonal, RBF ← z (full)
            z_list = [z_trend, z_seasonal, z]
            out = torch.zeros(qt.shape[0], qt.shape[1], device=qt.device)
            for z_i, head, trunk, bias in zip(z_list, self.heads, self.trunks, self.biases):
                out = out + hyper_forward_iq(trunk, t_flat, head(z_i), iq) + bias
            return out
        else:
            out = torch.zeros(qt.shape[0], qt.shape[1], device=qt.device)
            for head, trunk, bias in zip(self.heads, self.trunks, self.biases):
                out = out + hyper_forward_iq(trunk, t_flat, head(z), iq) + bias
            return out

    def forecast(self, ctx, n=96):
        t = torch.linspace(1, 2, n, device=ctx.device).unsqueeze(0).expand(ctx.shape[0], -1)
        return self.forward_train(ctx, t)


# ============================================================
# Dataset: TempoPFN only
# ============================================================
class TempoPFNOnlyDataset(Dataset):
    def __init__(self, max_samples=500000):
        path = 'tempopfn_15k_1024.arrow'
        print(f'Loading {path} (SYNTHETIC ONLY)...')
        table = pa.ipc.open_file(path).read_all()
        self.windows = []
        for i in range(len(table)):
            ts = np.array(table.column('target')[i].as_py(), dtype=np.float32)
            if len(ts) > 1024:
                n_w = min(50, len(ts) // 192)
                for _ in range(n_w):
                    start = np.random.randint(0, len(ts) - 192)
                    w = ts[start:start + 192]
                    s = np.std(w)
                    if s > 1e-6:
                        self.windows.append(np.clip((w - np.mean(w)) / s, -10, 10).astype(np.float32))
            elif len(ts) >= 192:
                s = np.std(ts[:192])
                if s > 1e-6:
                    self.windows.append(np.clip((ts[:192] - np.mean(ts[:192])) / s, -10, 10).astype(np.float32))
            if len(self.windows) >= max_samples: break
            if i % 5000 == 0: print(f'  row {i}, {len(self.windows)} windows')
        self.windows = np.array(self.windows, dtype=np.float32)
        print(f'TempoPFN-only: {len(self.windows)} windows (NO real data!)')

    def __len__(self): return len(self.windows)
    def __getitem__(self, idx):
        return torch.tensor(self.windows[idx], dtype=torch.float32)


# ============================================================
# Training
# ============================================================
def collate_batch(windows, n_query=16, mr=0.375):
    ctxs, qts, qvs = [], [], []
    for w in windows:
        w = w.numpy() if isinstance(w, torch.Tensor) else w
        if np.random.rand() < 0.5 and len(w) >= 192:
            ctx = w[:96]; future = w[96:]
            qi = np.random.choice(96, n_query, replace=False)
            qt = 1.0 + qi.astype(np.float32) / 96
            qv = future[qi]
        else:
            full = w[:96]
            mask = np.random.rand(96) > mr
            qi = np.where(~mask)[0]
            if len(qi) == 0: continue
            if len(qi) > n_query: qi = np.random.choice(qi, n_query, replace=False)
            elif len(qi) < n_query: qi = np.tile(qi, n_query // len(qi) + 1)[:n_query]
            ctx = full * mask.astype(np.float32)
            qt = qi.astype(np.float32) / 96
            qv = full[qi]
        ctxs.append(ctx); qts.append(qt); qvs.append(qv)
    if not ctxs: return None
    return (torch.tensor(np.stack(ctxs), dtype=torch.float32),
            torch.tensor(np.stack(qts), dtype=torch.float32),
            torch.tensor(np.stack(qvs), dtype=torch.float32))


def train(model, dataset, save_path, epochs=20, lr=3e-4, batch_size=64):
    n = sum(p.numel() for p in model.parameters())
    print(f'\n{"="*60}')
    print(f'Synthetic-Only Training')
    print(f'  Params: {n/1e6:.1f}M')
    print(f'  Latent decomp: {model.use_latent_decomp}')
    print(f'  Data: {len(dataset):,} windows (SYNTHETIC ONLY)')
    print(f'  NO real data — true zero-shot eval possible!')
    print(f'{"="*60}')

    dl = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                    num_workers=4, drop_last=True, pin_memory=True)
    steps = len(dl)
    print(f'Steps/epoch: {steps}')

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        losses, fc_l, imp_l = [], [], []
        t0 = time.time()
        for i, batch_windows in enumerate(dl):
            batch = collate_batch(batch_windows)
            if batch is None: continue
            ctx, qt, qv = [x.to(DEVICE) for x in batch]
            optimizer.zero_grad()
            pred = model.forward_train(ctx, qt)
            loss = F.mse_loss(pred, qv)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
            if (i+1) % 500 == 0:
                print(f'  iter {i+1}/{steps}: loss={np.mean(losses[-500:]):.4f}')
        scheduler.step()
        avg = np.mean(losses)
        print(f'Epoch {epoch+1}/{epochs}: loss={avg:.4f} lr={scheduler.get_last_lr()[0]:.6f} ({time.time()-t0:.0f}s)')
        if avg < best_loss:
            best_loss = avg
            torch.save(model.state_dict(), save_path)
            print(f'  Saved (best={best_loss:.4f})')
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', type=str, default='synth_only')
    parser.add_argument('--latent_decomp', type=int, default=0)
    parser.add_argument('--max_samples', type=int, default=500000)
    args = parser.parse_args()

    np.random.seed(42); torch.manual_seed(42)

    print('='*60)
    print(f'32.5M Synthetic-Only [{args.tag}]')
    print(f'  Latent decomp: {bool(args.latent_decomp)}')
    print('='*60)

    dataset = TempoPFNOnlyDataset(max_samples=args.max_samples)
    model = Model32MSynthetic(use_latent_decomp=bool(args.latent_decomp)).to(DEVICE)
    n = sum(p.numel() for p in model.parameters())
    print(f'Model: {n/1e6:.1f}M')

    save_path = f'checkpoints/32M_{args.tag}.pth'
    os.makedirs('checkpoints', exist_ok=True)
    train(model, dataset, save_path, epochs=20, lr=3e-4)
    print('\nDONE')
