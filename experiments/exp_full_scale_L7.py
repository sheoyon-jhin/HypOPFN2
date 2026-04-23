"""
Full Scale L7 Pre-training: Input Decomposition + Per-component Encoders + Matched Trunks

L7 architecture (curriculum 검증):
  - 4-way input decomposition (MA + FFT topk + bandpass + residual)
  - 4 per-component encoders (PatchAttn each)
  - 4 matched basis trunks (Poly, Fourier, Chirplet, RBF)
  - Hypernet trunk weights
  - True OP point-wise loss
  - RevIN OFF (per-window normalize)
  - Multi-task: forecast + imputation

Variants:
  Pile + TempoPFN (mixed): --pile_max 2000000 --tempopfn_max 400000
  Pile only:               --pile_max 5000000 --tempopfn_max 0

CUDA_VISIBLE_DEVICES=3 python experiments/exp_full_scale_L7.py [--pile_max N --tempopfn_max N --tag NAME]
"""
import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, math, time, pyarrow as pa
from torch import optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset

DEVICE = torch.device('cuda')
SEQ_LEN = 96
WIDTH = 192
HIDDEN = 384
N_LAYERS = 4
PATCH_SIZE = 16
N_FREQ = 24
N_RBF = 30
RBF_BW = 200
MA_WINDOW = 13         # for SEQ_LEN=96
FFT_TOP_K_SEASON = 3
CYCLE_LOW = 4
CYCLE_HIGH = 20


# ============================================================
# Differentiable input decomposition
# ============================================================
def moving_average(x, window=MA_WINDOW):
    pad = window // 2
    x_padded = F.pad(x.unsqueeze(1), (pad, pad), mode='reflect').squeeze(1)
    kernel = torch.ones(1, 1, window, device=x.device) / window
    return F.conv1d(x_padded.unsqueeze(1), kernel).squeeze(1)


def fft_topk(x, k):
    fft = torch.fft.rfft(x, dim=-1)
    mag = fft.abs(); mag[:, 0] = 0
    _, top_idx = torch.topk(mag, k, dim=-1)
    mask = torch.zeros_like(fft)
    mask.scatter_(1, top_idx, 1.0)
    return torch.fft.irfft(fft * mask, n=x.shape[-1], dim=-1)


def fft_bandpass(x, low_idx, high_idx):
    fft = torch.fft.rfft(x, dim=-1)
    mask = torch.zeros_like(fft)
    end = min(high_idx, fft.shape[-1])
    mask[:, low_idx:end] = 1.0
    return torch.fft.irfft(fft * mask, n=x.shape[-1], dim=-1)


def decompose_4way(x):
    """x: [B, L] → trend, season, cycle, resid"""
    trend = moving_average(x)
    detrended = x - trend
    season = fft_topk(detrended, k=FFT_TOP_K_SEASON)
    after_season = detrended - season
    cycle = fft_bandpass(after_season, CYCLE_LOW, CYCLE_HIGH)
    resid = after_season - cycle
    return trend, season, cycle, resid


# ============================================================
# Trunks
# ============================================================
class HTrunk(nn.Module):
    def __init__(self, w, btype, nf=N_FREQ, deg=6, nc=N_RBF, bw=RBF_BW):
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
# Encoder: PatchAttention
# ============================================================
class PatchAttnEncoder(nn.Module):
    def __init__(self, d_model=HIDDEN, patch_size=PATCH_SIZE, n_layers=N_LAYERS, nhead=8):
        super().__init__()
        self.patch_size = patch_size
        n_patches = SEQ_LEN // patch_size
        self.proj = nn.Linear(patch_size, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, n_patches, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4,
            dropout=0.1, activation='gelu', batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        B = x.shape[0]
        patches = x.view(B, -1, self.patch_size)
        z = self.proj(patches) + self.pos_emb
        z = self.transformer(z)
        return self.norm(z.mean(dim=1))


class TrunkBlock(nn.Module):
    """One encoder + one trunk (per-component)."""
    def __init__(self, trunk_type):
        super().__init__()
        self.enc = PatchAttnEncoder()
        self.trunk = HTrunk(WIDTH, trunk_type)
        self.head = nn.Linear(HIDDEN, self.trunk.odim)
        nn.init.xavier_normal_(self.head.weight, gain=0.1)
        nn.init.constant_(self.head.bias, 0)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x_part, qt):
        z = self.enc(x_part)
        t_flat = qt.reshape(-1)
        return hfwd(self.trunk, t_flat, self.head(z)) + self.bias


# ============================================================
# Full L7 Model
# ============================================================
class FullScaleL7Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.block_T = TrunkBlock('poly')
        self.block_S = TrunkBlock('fourier')
        self.block_C = TrunkBlock('chirplet')
        self.block_R = TrunkBlock('rbf')

    def _q(self, ctx, qt):
        trend, season, cycle, resid = decompose_4way(ctx)
        return (self.block_T(trend, qt) +
                self.block_S(season, qt) +
                self.block_C(cycle, qt) +
                self.block_R(resid, qt))

    def forward_train(self, ctx, qt): return self._q(ctx, qt)

    def forecast(self, ctx, n=96):
        t = torch.linspace(1, 2, n, device=ctx.device).unsqueeze(0).expand(ctx.shape[0], -1)
        return self._q(ctx, t)

    def impute(self, ctx, n=96):
        t = torch.linspace(0, 1, n, device=ctx.device).unsqueeze(0).expand(ctx.shape[0], -1)
        return self._q(ctx, t)


# ============================================================
# Datasets
# ============================================================
class StreamingPileDataset(Dataset):
    """Pile data (no anomaly), lazy loading."""
    def __init__(self, max_samples=2000000):
        from data_provider.pile_dataset import PilePretrainDataset
        print('Initializing PilePretrainDataset (no anomaly)...')
        self.ds = PilePretrainDataset(seq_len=192, stride=96,
                                       pile_root='./dataset/time_series_pile',
                                       skip_anomaly=True)
        self.max_samples = min(max_samples, len(self.ds))
        self.indices = np.random.choice(len(self.ds), self.max_samples, replace=False)
        print(f'Pile (no anomaly): {len(self.ds):,} → using {self.max_samples:,}')

    def __len__(self): return self.max_samples

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        w = self.ds[real_idx]
        if isinstance(w, torch.Tensor): w = w.numpy().flatten()
        else: w = np.array(w).flatten()
        if len(w) >= 192: w = w[:192]
        else: w = np.pad(w, (0, 192-len(w)))
        s = w.std()
        if s > 1e-6: w = np.clip((w - w.mean()) / s, -10, 10)
        return torch.tensor(w, dtype=torch.float32)


class TempoPFNDataset(Dataset):
    """TempoPFN — extract many windows per series."""
    def __init__(self, max_samples=400000):
        path = 'tempopfn_15k_1024.arrow'
        print(f'Loading {path}...')
        table = pa.ipc.open_file(path).read_all()
        self.windows = []
        for i in range(len(table)):
            ts = np.array(table.column('target')[i].as_py(), dtype=np.float32)
            if len(ts) > 1024:
                # Multiple windows per long series
                n_w = min(40, len(ts) // 192)
                for _ in range(n_w):
                    start = np.random.randint(0, len(ts)-192)
                    w = ts[start:start+192]
                    s = np.std(w)
                    if s > 1e-6:
                        self.windows.append(np.clip((w-np.mean(w))/s, -10, 10).astype(np.float32))
            elif len(ts) >= 192:
                s = np.std(ts[:192])
                if s > 1e-6:
                    self.windows.append(np.clip((ts[:192]-np.mean(ts[:192]))/s, -10, 10).astype(np.float32))
            if len(self.windows) >= max_samples: break
            if i % 5000 == 0: print(f'  TempoPFN row {i}, {len(self.windows)} windows')
        self.windows = np.array(self.windows, dtype=np.float32)
        print(f'TempoPFN: {len(self.windows)} windows')

    def __len__(self): return len(self.windows)

    def __getitem__(self, idx):
        return torch.tensor(self.windows[idx], dtype=torch.float32)


# ============================================================
# Sampling
# ============================================================
def collate_batch(windows, mode='forecast', n_query=16, mr=0.375):
    ctxs, qts, qvs = [], [], []
    for w in windows:
        w = w.numpy() if isinstance(w, torch.Tensor) else w
        if mode == 'forecast' and len(w) >= 192:
            ctx = w[:96]
            future = w[96:]
            qi = np.random.choice(96, n_query, replace=False)
            qt = (1.0 + qi.astype(np.float32) / 96)
            qv = future[qi]
        else:
            full = w[:96] if len(w) >= 96 else np.pad(w, (0, 96-len(w)))
            mask = np.random.rand(96) > mr
            qi = np.where(~mask)[0]
            if len(qi) == 0: continue
            if len(qi) > n_query:
                qi = np.random.choice(qi, n_query, replace=False)
            elif len(qi) < n_query:
                qi = np.tile(qi, n_query // len(qi) + 1)[:n_query]
            ctx = full * mask.astype(np.float32)
            qt = qi.astype(np.float32) / 96
            qv = full[qi]
        ctxs.append(ctx); qts.append(qt); qvs.append(qv)
    if not ctxs: return None
    return (torch.tensor(np.stack(ctxs), dtype=torch.float32),
            torch.tensor(np.stack(qts), dtype=torch.float32),
            torch.tensor(np.stack(qvs), dtype=torch.float32))


# ============================================================
# Training
# ============================================================
def train(model, datasets, save_path, epochs=20, lr=3e-4, batch_size=64):
    print(f'\n{"="*60}')
    print('L7 Full Scale Pre-training')
    print(f'  Architecture: 4-way decomp + per-component PatchAttn + matched trunks')
    print(f'  Components: Trend (Poly) / Season (Fourier) / Cycle (Chirplet) / Residual (RBF)')
    print(f'  RevIN OFF, True OP, multi-task')
    print(f'{"="*60}')

    n = sum(p.numel() for p in model.parameters())
    print(f'Model: {n/1e6:.1f}M params')
    for name, mod in [('block_T', model.block_T), ('block_S', model.block_S),
                      ('block_C', model.block_C), ('block_R', model.block_R)]:
        nm = sum(p.numel() for p in mod.parameters())
        print(f'  {name}: {nm/1e6:.1f}M')

    combined = ConcatDataset(datasets)
    print(f'Total data: {len(combined):,} windows')

    dl = DataLoader(combined, batch_size=batch_size, shuffle=True,
                    num_workers=4, drop_last=True, pin_memory=True)
    steps_per_epoch = len(dl)
    print(f'Steps/epoch: {steps_per_epoch:,}')
    print(f'Total steps: {steps_per_epoch * epochs:,}')

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        losses, fc_losses, imp_losses = [], [], []
        t0 = time.time()

        for i, batch_windows in enumerate(dl):
            if np.random.rand() < 0.5:
                batch = collate_batch(batch_windows, mode='forecast')
                task = 'fc'
            else:
                batch = collate_batch(batch_windows, mode='impute')
                task = 'imp'

            if batch is None: continue
            ctx, qt, qv = [x.to(DEVICE) for x in batch]

            optimizer.zero_grad()
            pred = model.forward_train(ctx, qt)
            loss = F.mse_loss(pred, qv)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            losses.append(loss.item())
            if task == 'fc': fc_losses.append(loss.item())
            else: imp_losses.append(loss.item())

            if (i+1) % 500 == 0:
                avg_fc = np.mean(fc_losses[-500:]) if fc_losses else 0
                avg_imp = np.mean(imp_losses[-500:]) if imp_losses else 0
                print(f'  iter {i+1}/{steps_per_epoch}: loss={np.mean(losses[-500:]):.4f} '
                      f'fc={avg_fc:.4f} imp={avg_imp:.4f}')

        scheduler.step()
        elapsed = time.time() - t0
        avg_loss = np.mean(losses)
        print(f'Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f} '
              f'(fc={np.mean(fc_losses) if fc_losses else 0:.4f} '
              f'imp={np.mean(imp_losses) if imp_losses else 0:.4f}) '
              f'lr={scheduler.get_last_lr()[0]:.6f} ({elapsed:.0f}s)')

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), save_path)
            print(f'  Saved (best={best_loss:.4f})')

    return model


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pile_max', type=int, default=2000000)
    parser.add_argument('--tempopfn_max', type=int, default=400000)
    parser.add_argument('--tag', type=str, default='mixed')
    args = parser.parse_args()

    np.random.seed(42)
    torch.manual_seed(42)

    print('='*60)
    print(f'Full Scale L7 [{args.tag}]')
    print(f'  Pile: {args.pile_max:,}')
    print(f'  TempoPFN: {args.tempopfn_max:,}')
    print('='*60)

    print('\nLoading datasets...')
    datasets = []
    if args.pile_max > 0:
        pile_ds = StreamingPileDataset(max_samples=args.pile_max)
        datasets.append(pile_ds)
    if args.tempopfn_max > 0:
        tempopfn_ds = TempoPFNDataset(max_samples=args.tempopfn_max)
        datasets.append(tempopfn_ds)

    total = sum(len(d) for d in datasets)
    print(f'\nTotal data: {total:,} windows')

    model = FullScaleL7Model().to(DEVICE)
    n = sum(p.numel() for p in model.parameters())
    print(f'\nModel: {n/1e6:.1f}M params (4 PatchAttn encoders + 4 matched trunks)')

    save_path = f'checkpoints/full_scale_L7_{args.tag}.pth'
    os.makedirs('checkpoints', exist_ok=True)

    model = train(model, datasets, save_path, epochs=20, lr=3e-4)
    print('\nDONE')
