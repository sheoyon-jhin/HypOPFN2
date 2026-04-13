"""
32.5M Architecture + Longer Context (NO last value residual)

32.5M 그대로:
  ✓ Shared PatchAttn encoder + 3 HyperDiverse trunks (Fourier/Poly/RBF)
  ✓ Informed query (FFT top-5)
  ✓ True OP point-wise loss
  ✓ RevIN OFF
  ✗ Last value residual (없음 — V2와의 차이)

변경:
  ✓ SEQ_LEN: 96 → 192 (or 384 via arg)

Usage:
  CUDA_VISIBLE_DEVICES=0 python experiments/exp_32M_longctx.py --seq 192 --tag seq192
  CUDA_VISIBLE_DEVICES=2 python experiments/exp_32M_longctx.py --seq 384 --tag seq384
"""
import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, math, time, pyarrow as pa
from torch import optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset

_dev = os.environ.get('CUDA_DEV', 'cuda')
DEVICE = torch.device(_dev)

# Parsed later
SEQ_LEN = None
PATCH_SIZE = 16
HIDDEN = 512
N_LAYERS = 6        # 원본 32.5M과 동일
WIDTH = 192          # 원본과 동일
N_FREQ = 32
N_RBF = 20
RBF_BW = 20
TOP_K_IQ = 5
INFORMED_DIM = 1 + 2*TOP_K_IQ + 2  # 13


# ============================================================
# Trunks + Informed query (원본 32.5M과 동일)
# ============================================================
def extract_freq(x, top_k=TOP_K_IQ):
    fft = torch.fft.rfft(x, dim=-1)
    mag = fft.abs(); mag[:, 0] = 0
    idx = torch.topk(mag, top_k, dim=-1).indices
    phase = torch.angle(torch.gather(fft, 1, idx))
    return idx.float(), phase, x[:, -1:].detach(), (x[:, -1:] - x[:, -2:-1]).detach()


class HyperTrunk(nn.Module):
    def __init__(self, w, btype, nf=N_FREQ, deg=6, nc=N_RBF):
        super().__init__(); self.w=w; self.btype=btype
        if btype=='fourier': self.nf=nf; self.idim=1+2*nf
        elif btype=='poly': self.deg=deg; self.idim=deg+1
        elif btype=='rbf':
            self.register_buffer('centers', torch.linspace(0, 2, nc))
            self.idim = 1 + nc
        self.informed_dim = INFORMED_DIM
        self.full_idim = self.idim + self.informed_dim
        self.pc = self.full_idim * w + w
        self.odim = self.pc + w

    def base_feat(self, t):
        t = t.unsqueeze(-1) if t.dim()==1 else t
        if self.btype=='fourier':
            f = torch.arange(1, self.nf+1, device=t.device, dtype=t.dtype)
            return torch.cat([t, torch.sin(2*math.pi*f*t), torch.cos(2*math.pi*f*t)], dim=-1)
        elif self.btype=='poly':
            return torch.cat([t**i for i in range(self.deg+1)], dim=-1)
        elif self.btype=='rbf':
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


# ============================================================
# Encoder
# ============================================================
class PatchAttnEncoder(nn.Module):
    def __init__(self, seq_len):
        super().__init__()
        self.patch_size = PATCH_SIZE
        n_patches = seq_len // PATCH_SIZE
        self.proj = nn.Linear(PATCH_SIZE, HIDDEN)
        self.pos_emb = nn.Parameter(torch.randn(1, n_patches, HIDDEN) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=HIDDEN, nhead=8, dim_feedforward=HIDDEN*4,
            dropout=0.1, activation='gelu', batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=N_LAYERS)
        self.norm = nn.LayerNorm(HIDDEN)

    def forward(self, x):
        B = x.shape[0]
        patches = x.view(B, -1, self.patch_size)
        z = self.proj(patches) + self.pos_emb
        z = self.transformer(z)
        return self.norm(z.mean(dim=1))


# ============================================================
# Model (원본 32.5M과 동일 구조, SEQ_LEN만 변경)
# ============================================================
class Model32MLongCtx(nn.Module):
    def __init__(self, seq_len):
        super().__init__()
        self.seq_len = seq_len
        self.encoder = PatchAttnEncoder(seq_len)
        self.trunks = nn.ModuleList([
            HyperTrunk(WIDTH, 'fourier'),
            HyperTrunk(WIDTH, 'poly'),
            HyperTrunk(WIDTH, 'rbf'),
        ])
        self.heads = nn.ModuleList([nn.Linear(HIDDEN, t.odim) for t in self.trunks])
        for h in self.heads:
            nn.init.xavier_normal_(h.weight, gain=0.1)
            nn.init.constant_(h.bias, 0)
        self.biases = nn.ParameterList([nn.Parameter(torch.zeros(1)) for _ in range(3)])

    def _build_iq(self, ctx_n, qt):
        freqs, phases, lv, ls = extract_freq(ctx_n)
        B, nq = qt.shape
        t_exp = qt.unsqueeze(-1)
        f = freqs.unsqueeze(1)
        p = phases.unsqueeze(1)
        ang = 2 * math.pi * f * t_exp + p
        return torch.cat([
            t_exp,
            torch.sin(ang), torch.cos(ang),
            lv.unsqueeze(1).expand(-1, nq, -1),
            ls.unsqueeze(1).expand(-1, nq, -1),
        ], dim=-1)

    def _forward_query(self, z, qt, ctx_for_iq):
        iq = self._build_iq(ctx_for_iq, qt)
        t_flat = qt.reshape(-1)
        out = torch.zeros(qt.shape[0], qt.shape[1], device=qt.device)
        for head, trunk, bias in zip(self.heads, self.trunks, self.biases):
            out = out + hyper_forward_iq(trunk, t_flat, head(z), iq) + bias
        return out

    def forward_train(self, ctx, qt):
        """ctx: [B, seq_len] already normalized."""
        z = self.encoder(ctx)
        return self._forward_query(z, qt, ctx)

    def forecast(self, ctx, n=None):
        if n is None: n = self.seq_len
        t = torch.linspace(1, 2, n, device=ctx.device).unsqueeze(0).expand(ctx.shape[0], -1)
        z = self.encoder(ctx)
        return self._forward_query(z, t, ctx)

    def impute(self, ctx, n=None):
        if n is None: n = self.seq_len
        t = torch.linspace(0, 1, n, device=ctx.device).unsqueeze(0).expand(ctx.shape[0], -1)
        z = self.encoder(ctx)
        return self._forward_query(z, t, ctx)


# ============================================================
# Datasets
# ============================================================
def make_datasets(seq_len, pile_max=1000000, tempopfn_max=200000):
    window_len = seq_len * 2
    datasets = []

    # Pile
    from data_provider.pile_dataset import PilePretrainDataset
    print(f'Loading Pile (window={window_len}, skip_anomaly)...')
    pile = PilePretrainDataset(seq_len=window_len, stride=window_len//2,
                                pile_root='./dataset/time_series_pile', skip_anomaly=True)
    n_pile = min(pile_max, len(pile))
    indices = np.random.choice(len(pile), n_pile, replace=False)

    class PileSubset(Dataset):
        def __init__(self):
            self.ds = pile; self.indices = indices; self.wl = window_len
        def __len__(self): return len(self.indices)
        def __getitem__(self, idx):
            w = self.ds[self.indices[idx]]
            if isinstance(w, torch.Tensor): w = w.numpy().flatten()
            else: w = np.array(w).flatten()
            if len(w) >= self.wl: w = w[:self.wl]
            else: w = np.pad(w, (0, self.wl-len(w)))
            s = w.std()
            if s > 1e-6: w = np.clip((w - w.mean()) / s, -10, 10)
            return torch.tensor(w, dtype=torch.float32)

    datasets.append(PileSubset())
    print(f'  Pile: {len(pile):,} → {n_pile:,}')

    # TempoPFN
    path = 'tempopfn_15k_1024.arrow'
    if os.path.exists(path):
        print(f'Loading TempoPFN (window={window_len})...')
        table = pa.ipc.open_file(path).read_all()
        windows = []
        for i in range(len(table)):
            ts = np.array(table.column('target')[i].as_py(), dtype=np.float32)
            if len(ts) >= window_len:
                n_w = min(10, len(ts) // window_len)
                for _ in range(n_w):
                    start = np.random.randint(0, len(ts)-window_len)
                    w = ts[start:start+window_len]
                    s = np.std(w)
                    if s > 1e-6:
                        windows.append(np.clip((w-np.mean(w))/s, -10, 10).astype(np.float32))
            if len(windows) >= tempopfn_max: break
        windows = np.array(windows, dtype=np.float32)
        print(f'  TempoPFN: {len(windows):,}')

        class TempoPFNDs(Dataset):
            def __init__(self): self.w = windows
            def __len__(self): return len(self.w)
            def __getitem__(self, idx): return torch.tensor(self.w[idx])
        datasets.append(TempoPFNDs())

    return datasets


# ============================================================
# Training (원본 32.5M과 거의 동일)
# ============================================================
def collate_batch(windows, seq_len, mode='forecast', n_query=16, mr=0.375):
    ctxs, qts, qvs = [], [], []
    window_len = seq_len * 2
    for w in windows:
        w = w.numpy() if isinstance(w, torch.Tensor) else w
        if mode == 'forecast' and len(w) >= window_len:
            ctx = w[:seq_len]
            future = w[seq_len:]
            qi = np.random.choice(seq_len, n_query, replace=False)
            qt = 1.0 + qi.astype(np.float32) / seq_len
            qv = future[qi]
        else:
            full = w[:seq_len] if len(w) >= seq_len else np.pad(w, (0, seq_len-len(w)))
            mask = np.random.rand(seq_len) > mr
            qi = np.where(~mask)[0]
            if len(qi) == 0: continue
            if len(qi) > n_query:
                qi = np.random.choice(qi, n_query, replace=False)
            elif len(qi) < n_query:
                qi = np.tile(qi, n_query // len(qi) + 1)[:n_query]
            ctx = full * mask.astype(np.float32)
            qt = qi.astype(np.float32) / seq_len
            qv = full[qi]
        ctxs.append(ctx); qts.append(qt); qvs.append(qv)
    if not ctxs: return None
    return (torch.tensor(np.stack(ctxs), dtype=torch.float32),
            torch.tensor(np.stack(qts), dtype=torch.float32),
            torch.tensor(np.stack(qvs), dtype=torch.float32))


def train(model, datasets, save_path, seq_len, epochs=20, lr=3e-4, batch_size=64):
    n = sum(p.numel() for p in model.parameters())
    print(f'\n{"="*60}')
    print(f'32.5M Long Context (SEQ={seq_len})')
    print(f'  Params: {n/1e6:.1f}M')
    print(f'  NO last value residual')
    print(f'  Informed query: ON')
    print(f'{"="*60}')

    combined = ConcatDataset(datasets)
    print(f'Total data: {len(combined):,}')
    dl = DataLoader(combined, batch_size=batch_size, shuffle=True,
                    num_workers=4, drop_last=True, pin_memory=True)
    steps = len(dl)
    print(f'Steps/epoch: {steps:,}')

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        losses, fc_l, imp_l = [], [], []
        t0 = time.time()
        for i, batch_windows in enumerate(dl):
            if np.random.rand() < 0.5:
                batch = collate_batch(batch_windows, seq_len, 'forecast')
                task = 'fc'
            else:
                batch = collate_batch(batch_windows, seq_len, 'impute')
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
            if task == 'fc': fc_l.append(loss.item())
            else: imp_l.append(loss.item())

            if (i+1) % 500 == 0:
                print(f'  iter {i+1}/{steps}: loss={np.mean(losses[-500:]):.4f} '
                      f'fc={np.mean(fc_l[-500:]) if fc_l else 0:.4f} '
                      f'imp={np.mean(imp_l[-500:]) if imp_l else 0:.4f}')

        scheduler.step()
        el = time.time() - t0
        avg = np.mean(losses)
        print(f'Epoch {epoch+1}/{epochs}: loss={avg:.4f} '
              f'(fc={np.mean(fc_l) if fc_l else 0:.4f} imp={np.mean(imp_l) if imp_l else 0:.4f}) '
              f'lr={scheduler.get_last_lr()[0]:.6f} ({el:.0f}s)')
        if avg < best_loss:
            best_loss = avg
            torch.save(model.state_dict(), save_path)
            print(f'  Saved (best={best_loss:.4f})')

    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq', type=int, default=192)
    parser.add_argument('--tag', type=str, default='seq192')
    parser.add_argument('--pile_max', type=int, default=1000000)
    parser.add_argument('--tempopfn_max', type=int, default=200000)
    args = parser.parse_args()

    SEQ_LEN = args.seq
    np.random.seed(42); torch.manual_seed(42)

    print('='*60)
    print(f'32.5M Long Context [SEQ={SEQ_LEN}]')
    print('='*60)

    datasets = make_datasets(SEQ_LEN, args.pile_max, args.tempopfn_max)
    total = sum(len(d) for d in datasets)
    print(f'\nTotal: {total:,} windows')

    model = Model32MLongCtx(SEQ_LEN).to(DEVICE)
    n = sum(p.numel() for p in model.parameters())
    print(f'Model: {n/1e6:.1f}M params')

    save_path = f'checkpoints/32M_{args.tag}.pth'
    os.makedirs('checkpoints', exist_ok=True)

    train(model, datasets, save_path, SEQ_LEN, epochs=20, lr=3e-4)
    print('\nDONE')
