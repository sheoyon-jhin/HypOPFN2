"""
V3 Training: SEQ_LEN=384 (jump to long context)

Key improvements over V2:
  ✓ SEQ_LEN: 192 → 384 (2x further)
  ✓ Informed query + last value residual (from V2)
  ✓ Scale: similar to V2 (~45-55M)

Target: Approach/beat FeDaL (obs 512-3072) performance

Usage:
  CUDA_VISIBLE_DEVICES=3 python experiments/exp_v3.py --decomp 0 --tag v3_nodecomp
"""
import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, math, time, pyarrow as pa
from torch import optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset

_dev = os.environ.get('CUDA_DEV', 'cuda')
DEVICE = torch.device(_dev)

# ============================================================
# V3 Config: longer context, otherwise same as V2
# ============================================================
SEQ_LEN = 384            # ← 192 → 384 (V2보다 2x)
PATCH_SIZE = 16           # 384/16 = 24 patches
HIDDEN = 512
N_LAYERS = 8
WIDTH = 256
HIDDEN_DECOMP = 384
N_LAYERS_DECOMP = 4
N_FREQ = 32               # 32 freq (longer seq → more useful)
N_RBF = 30
RBF_BW = 300
MA_WINDOW = 49            # 25 → 49 for longer sequence
FFT_TOP_K_SEASON = 4
CYCLE_LOW = 5
CYCLE_HIGH = 40

TOP_K_IQ = 5
INFORMED_DIM = 1 + 2*TOP_K_IQ + 2


# ============================================================
# Reuse from V2 (decomp functions, trunks, encoder, etc.)
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
    trend = moving_average(x)
    detrended = x - trend
    season = fft_topk(detrended, k=FFT_TOP_K_SEASON)
    after_season = detrended - season
    cycle = fft_bandpass(after_season, CYCLE_LOW, CYCLE_HIGH)
    resid = after_season - cycle
    return trend, season, cycle, resid


def extract_freq(x, top_k=TOP_K_IQ):
    fft = torch.fft.rfft(x, dim=-1)
    mag = fft.abs(); mag[:, 0] = 0
    idx = torch.topk(mag, top_k, dim=-1).indices
    phase = torch.angle(torch.gather(fft, 1, idx))
    last_val = x[:, -1:].detach()
    slope = (x[:, -1:] - x[:, -2:-1]).detach()
    return idx.float(), phase, last_val, slope


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
        self.full_idim = self.idim + INFORMED_DIM
        self.pc = self.full_idim*w + w
        self.odim = self.pc + w

    def base_feat(self, t):
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


def hfwd_iq(trunk, t_flat, head_out, iq_features):
    B = head_out.shape[0]
    base = trunk.base_feat(t_flat)
    nq = base.shape[0] // B
    base = base.view(B, nq, trunk.idim)
    full = torch.cat([base, iq_features], dim=-1)
    tp = head_out[:, :trunk.pc] * 0.01
    Wm = tp[:, :trunk.full_idim*trunk.w].view(B, trunk.full_idim, trunk.w)
    bm = tp[:, trunk.full_idim*trunk.w:].view(B, trunk.w)
    Phi = F.gelu(torch.bmm(full, Wm) + bm.unsqueeze(1))
    Bc = head_out[:, trunk.pc:]
    return torch.einsum('bw,bqw->bq', Bc, Phi)


class PatchAttnEncoder(nn.Module):
    def __init__(self, d_model, n_layers, nhead=8, seq_len=SEQ_LEN, patch_size=PATCH_SIZE):
        super().__init__()
        self.patch_size = patch_size
        n_patches = seq_len // patch_size
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
    def __init__(self, encoder, trunk_types, hidden_dim):
        super().__init__()
        self.encoder = encoder
        self.trunks = nn.ModuleList([HTrunk(WIDTH, t) for t in trunk_types])
        self.heads = nn.ModuleList([nn.Linear(hidden_dim, t.odim) for t in self.trunks])
        for h in self.heads: nn.init.xavier_normal_(h.weight, gain=0.1)
        self.biases = nn.ParameterList([nn.Parameter(torch.zeros(1)) for _ in trunk_types])

    def forward(self, x_part, qt, iq_features):
        z = self.encoder(x_part)
        t_flat = qt.reshape(-1)
        return sum(hfwd_iq(tr, t_flat, h(z), iq_features) + b
                   for tr, h, b in zip(self.trunks, self.heads, self.biases))


def build_iq_for_model(ctx_n, qt):
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


class ModelV3NoDecomp(nn.Module):
    def __init__(self):
        super().__init__()
        enc = PatchAttnEncoder(HIDDEN, N_LAYERS)
        self.block = TrunkBlock(enc, ['fourier', 'poly', 'rbf'], HIDDEN)

    def _forward_normalized(self, ctx_n, qt):
        iq = build_iq_for_model(ctx_n, qt)
        return self.block(ctx_n, qt, iq)

    def forward_train(self, ctx, qt):
        last = ctx[:, -1:]
        ctx_c = ctx - last
        s = ctx_c.std(dim=1, keepdim=True).clamp(min=1e-6)
        ctx_n = (ctx_c / s).clamp(-10, 10)
        pred_delta_n = self._forward_normalized(ctx_n, qt)
        return last + pred_delta_n * s

    def forecast(self, ctx, n=SEQ_LEN):
        t = torch.linspace(1, 2, n, device=ctx.device).unsqueeze(0).expand(ctx.shape[0], -1)
        return self.forward_train(ctx, t)


# ============================================================
# Datasets (WINDOW_LEN = 2*SEQ_LEN)
# ============================================================
WINDOW_LEN = SEQ_LEN * 2   # 768


class StreamingPileDataset(Dataset):
    def __init__(self, max_samples=1500000, skip_anomaly=True):
        from data_provider.pile_dataset import PilePretrainDataset
        print(f'Initializing PilePretrainDataset (skip_anomaly={skip_anomaly})...')
        self.ds = PilePretrainDataset(
            seq_len=WINDOW_LEN, stride=WINDOW_LEN//2,
            pile_root='./dataset/time_series_pile',
            skip_anomaly=skip_anomaly)
        self.max_samples = min(max_samples, len(self.ds))
        self.indices = np.random.choice(len(self.ds), self.max_samples, replace=False)
        print(f'Pile: {len(self.ds):,} → using {self.max_samples:,}')

    def __len__(self): return self.max_samples

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        w = self.ds[real_idx]
        if isinstance(w, torch.Tensor): w = w.numpy().flatten()
        else: w = np.array(w).flatten()
        if len(w) >= WINDOW_LEN: w = w[:WINDOW_LEN]
        else: w = np.pad(w, (0, WINDOW_LEN-len(w)))
        s = w.std()
        if s > 1e-6: w = np.clip((w - w.mean()) / s, -10, 10)
        return torch.tensor(w, dtype=torch.float32)


class TempoPFNDataset(Dataset):
    def __init__(self, max_samples=300000):
        path = 'tempopfn_15k_1024.arrow'
        print(f'Loading {path}...')
        table = pa.ipc.open_file(path).read_all()
        self.windows = []
        for i in range(len(table)):
            ts = np.array(table.column('target')[i].as_py(), dtype=np.float32)
            if len(ts) > 1024 and len(ts) >= WINDOW_LEN:
                n_w = min(5, len(ts) // WINDOW_LEN)
                for _ in range(n_w):
                    start = np.random.randint(0, len(ts)-WINDOW_LEN)
                    w = ts[start:start+WINDOW_LEN]
                    s = np.std(w)
                    if s > 1e-6:
                        self.windows.append(np.clip((w-np.mean(w))/s, -10, 10).astype(np.float32))
            if len(self.windows) >= max_samples: break
            if i % 5000 == 0: print(f'  TempoPFN row {i}, {len(self.windows)} windows')
        self.windows = np.array(self.windows, dtype=np.float32)
        print(f'TempoPFN: {len(self.windows)} windows')

    def __len__(self): return len(self.windows)
    def __getitem__(self, idx):
        return torch.tensor(self.windows[idx], dtype=torch.float32)


def collate_batch(windows, mode='forecast', n_query=32, mr=0.375):
    ctxs, qts, qvs = [], [], []
    for w in windows:
        w = w.numpy() if isinstance(w, torch.Tensor) else w
        if mode == 'forecast' and len(w) >= WINDOW_LEN:
            ctx = w[:SEQ_LEN]
            future = w[SEQ_LEN:]
            qi = np.random.choice(SEQ_LEN, n_query, replace=False)
            qt = (1.0 + qi.astype(np.float32) / SEQ_LEN)
            qv = future[qi]
        else:
            full = w[:SEQ_LEN] if len(w) >= SEQ_LEN else np.pad(w, (0, SEQ_LEN-len(w)))
            mask = np.random.rand(SEQ_LEN) > mr
            qi = np.where(~mask)[0]
            if len(qi) == 0: continue
            if len(qi) > n_query:
                qi = np.random.choice(qi, n_query, replace=False)
            elif len(qi) < n_query:
                qi = np.tile(qi, n_query // len(qi) + 1)[:n_query]
            ctx = full * mask.astype(np.float32)
            qt = qi.astype(np.float32) / SEQ_LEN
            qv = full[qi]
        ctxs.append(ctx); qts.append(qt); qvs.append(qv)
    if not ctxs: return None
    return (torch.tensor(np.stack(ctxs), dtype=torch.float32),
            torch.tensor(np.stack(qts), dtype=torch.float32),
            torch.tensor(np.stack(qvs), dtype=torch.float32))


def train(model, datasets, save_path, epochs=20, lr=3e-4, batch_size=48):
    n = sum(p.numel() for p in model.parameters())
    print(f'\n{"="*60}')
    print(f'V3 Pre-training (no-decomp, SEQ_LEN={SEQ_LEN})')
    print(f'  Params: {n/1e6:.1f}M')
    print(f'{"="*60}')

    combined = ConcatDataset(datasets)
    print(f'Total data: {len(combined):,} windows')
    dl = DataLoader(combined, batch_size=batch_size, shuffle=True,
                    num_workers=4, drop_last=True, pin_memory=True)
    steps_per_epoch = len(dl)
    print(f'Steps/epoch: {steps_per_epoch:,}')

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', type=str, default='v3_nodecomp')
    parser.add_argument('--pile_max', type=int, default=1000000)
    parser.add_argument('--tempopfn_max', type=int, default=200000)
    args = parser.parse_args()

    np.random.seed(42); torch.manual_seed(42)

    print('='*60)
    print(f'V3 Pre-training [SEQ_LEN={SEQ_LEN}]')
    print(f'  Window len: {WINDOW_LEN}')
    print(f'  Pile: {args.pile_max:,}, TempoPFN: {args.tempopfn_max:,}')
    print('='*60)

    datasets = []
    if args.pile_max > 0:
        datasets.append(StreamingPileDataset(max_samples=args.pile_max))
    if args.tempopfn_max > 0:
        datasets.append(TempoPFNDataset(max_samples=args.tempopfn_max))

    total = sum(len(d) for d in datasets)
    print(f'\nTotal: {total:,} windows')

    model = ModelV3NoDecomp().to(DEVICE)
    n = sum(p.numel() for p in model.parameters())
    print(f'\nModel: {n/1e6:.1f}M params')

    save_path = f'checkpoints/v3_{args.tag}.pth'
    os.makedirs('checkpoints', exist_ok=True)

    train(model, datasets, save_path, epochs=20, lr=3e-4, batch_size=48)
    print('\nDONE')
