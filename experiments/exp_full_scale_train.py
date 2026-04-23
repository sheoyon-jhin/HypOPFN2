"""
Full Scale Pre-training: 우리가 찾은 모든 것을 합쳐서 MOMENT급 스케일

확정된 설정:
  ✓ Encoder: PatchAttention (scale up에서 best)
  ✓ Trunk: HyperNetwork Diverse (Fourier/Poly/RBF)
  ✓ Loss: True OP point-wise + Multi-task (FC + Imputation)
  ✓ RevIN: OFF
  ✓ Informed Query: ON (slight help)
  ✓ Spectral Loss: OFF (효과 없음)
  ✓ Query: Random point-wise

목표: ~50M 모델, ~1M+ windows
시간: 10~20시간

CUDA_VISIBLE_DEVICES=X python experiments/exp_full_scale_train.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, math, time, pyarrow as pa
from torch import optim
from torch.utils.data import Dataset, DataLoader, IterableDataset

DEVICE = torch.device('cuda')
SEQ_LEN = 96
WIDTH = 192
HIDDEN = 512
N_LAYERS = 6
PATCH_SIZE = 16
N_FREQ = 32
N_RBF = 20


# ============================================================
# Streaming Dataset (메모리 친화적)
# ============================================================
class StreamingPileDataset(Dataset):
    """Pile 데이터를 lazy loading."""
    def __init__(self, max_samples=1000000):
        from data_provider.pile_dataset import PilePretrainDataset
        print('Initializing PilePretrainDataset...')
        self.ds = PilePretrainDataset(seq_len=192, stride=96,
                                       pile_root='./dataset/time_series_pile')
        self.max_samples = min(max_samples, len(self.ds))
        self.indices = np.random.choice(len(self.ds), self.max_samples, replace=False)
        print(f'Pile: {len(self.ds):,} → using {self.max_samples:,}')

    def __len__(self): return self.max_samples

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        w = self.ds[real_idx]
        if isinstance(w, torch.Tensor): w = w.numpy().flatten()
        else: w = np.array(w).flatten()
        # Pad or truncate to 192
        if len(w) >= 192: w = w[:192]
        else: w = np.pad(w, (0, 192-len(w)))
        # Normalize
        s = w.std()
        if s > 1e-6: w = np.clip((w - w.mean()) / s, -10, 10)
        return torch.tensor(w, dtype=torch.float32)


class TempoPFNDataset(Dataset):
    """TempoPFN 15K — 메모리에 한 번만 올림."""
    def __init__(self, max_samples=200000):
        path = 'tempopfn_15k_1024.arrow'
        print(f'Loading {path}...')
        table = pa.ipc.open_file(path).read_all()
        self.windows = []
        for i in range(len(table)):
            ts = np.array(table.column('target')[i].as_py(), dtype=np.float32)
            if len(ts) > 1024:
                # Multiple windows per series
                n_w = min(20, len(ts) // 192)
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
        self.windows = np.array(self.windows, dtype=np.float32)
        print(f'TempoPFN: {len(self.windows)} windows')

    def __len__(self): return len(self.windows)

    def __getitem__(self, idx):
        return torch.tensor(self.windows[idx], dtype=torch.float32)


# ============================================================
# Model
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
        # x: [B, 96]
        B = x.shape[0]
        patches = x.view(B, -1, self.patch_size)  # [B, n_patches, patch_size]
        z = self.proj(patches) + self.pos_emb
        z = self.transformer(z)
        return self.norm(z.mean(dim=1))  # [B, d_model]


class HyperTrunk(nn.Module):
    def __init__(self, w, btype, n_freq=N_FREQ, deg=6, nc=N_RBF):
        super().__init__()
        self.w = w
        self.btype = btype
        if btype == 'fourier':
            self.n_freq = n_freq
            self.idim = 1 + 2*n_freq
        elif btype == 'poly':
            self.deg = deg
            self.idim = deg + 1
        elif btype == 'rbf':
            self.register_buffer('centers', torch.linspace(0, 2, nc))
            self.idim = 1 + nc

        # Add informed query dim
        self.informed_dim = 1 + 2*5 + 2  # t + sin/cos(5) + lv + ls = 13
        self.full_idim = self.idim + self.informed_dim
        self.pc = self.full_idim * w + w
        self.odim = self.pc + w

    def base_feat(self, t):
        t = t.unsqueeze(-1) if t.dim() == 1 else t
        if self.btype == 'fourier':
            f = torch.arange(1, self.n_freq+1, device=t.device, dtype=t.dtype)
            return torch.cat([t, torch.sin(2*math.pi*f*t), torch.cos(2*math.pi*f*t)], dim=-1)
        elif self.btype == 'poly':
            return torch.cat([t**i for i in range(self.deg+1)], dim=-1)
        elif self.btype == 'rbf':
            return torch.cat([t, torch.exp(-20*(t-self.centers.unsqueeze(0))**2)], dim=-1)


def extract_freq(x, top_k=5):
    fft = torch.fft.rfft(x, dim=-1)
    mag = fft.abs(); mag[:, 0] = 0
    idx = torch.topk(mag, top_k, dim=-1).indices
    phase = torch.angle(torch.gather(fft, 1, idx))
    return idx.float(), phase, x[:, -1:].detach(), (x[:, -1:]-x[:, -2:-1]).detach()


def build_iq(t_col, freqs, phases, lv, ls):
    """t_col: [T, 1], freqs/phases: [B, K], lv/ls: [B, 1]"""
    B = freqs.shape[0]
    T = t_col.shape[0]
    te = t_col.squeeze(-1).unsqueeze(0).unsqueeze(-1)  # [1, T, 1]
    f = freqs.unsqueeze(1)  # [B, 1, K]
    p = phases.unsqueeze(1)  # [B, 1, K]
    ang = 2*math.pi*f*te + p  # [B, T, K]
    return torch.cat([
        t_col.squeeze(-1).unsqueeze(0).expand(B, -1).unsqueeze(-1),
        torch.sin(ang), torch.cos(ang),
        lv.unsqueeze(1).expand(-1, T, -1),
        ls.unsqueeze(1).expand(-1, T, -1)
    ], dim=-1)


def hyper_forward_with_iq(trunk, t_flat, head_output, iq_features):
    """trunk: HyperTrunk, t_flat: [B*nq], head_output: [B, odim], iq_features: [B, nq, informed_dim]"""
    B = head_output.shape[0]
    base_feat = trunk.base_feat(t_flat)  # [B*nq, idim]
    nq = base_feat.shape[0] // B
    base_feat = base_feat.view(B, nq, trunk.idim)
    full_feat = torch.cat([base_feat, iq_features], dim=-1)  # [B, nq, full_idim]

    tp = head_output[:, :trunk.pc] * 0.01
    W = tp[:, :trunk.full_idim*trunk.w].view(B, trunk.full_idim, trunk.w)
    b = tp[:, trunk.full_idim*trunk.w:].view(B, trunk.w)
    Phi = F.gelu(torch.bmm(full_feat, W) + b.unsqueeze(1))
    Bc = head_output[:, trunk.pc:]
    return torch.einsum('bw,bqw->bq', Bc, Phi)


class FullScaleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = PatchAttnEncoder()
        self.trunks = nn.ModuleList([
            HyperTrunk(WIDTH, 'fourier'),
            HyperTrunk(WIDTH, 'poly'),
            HyperTrunk(WIDTH, 'rbf')
        ])
        self.heads = nn.ModuleList([
            nn.Linear(HIDDEN, t.odim) for t in self.trunks
        ])
        for h in self.heads:
            nn.init.xavier_normal_(h.weight, gain=0.1)
            nn.init.constant_(h.bias, 0)
        self.biases = nn.ParameterList([nn.Parameter(torch.zeros(1)) for _ in range(3)])

    def _forward_query(self, z, query_t, x_for_iq):
        """z: [B, h], query_t: [B, nq], x_for_iq: [B, 96]"""
        B, nq = query_t.shape
        t_flat = query_t.reshape(-1)

        # Build informed query
        freqs, phases, lv, ls = extract_freq(x_for_iq)
        # Use unique time points for query
        t_col = torch.linspace(query_t.min().item(), query_t.max().item(), nq,
                               device=z.device, dtype=z.dtype).unsqueeze(-1)
        iq = build_iq(t_col, freqs, phases, lv, ls)

        out = torch.zeros(B, nq, device=z.device, dtype=z.dtype)
        for head, trunk, bias in zip(self.heads, self.trunks, self.biases):
            ho = head(z)
            out = out + hyper_forward_with_iq(trunk, t_flat, ho, iq) + bias
        return out

    def forward_train(self, ctx, qt):
        """ctx: [B, 96] (no RevIN!), qt: [B, nq] → pred: [B, nq]"""
        z = self.encoder(ctx)
        return self._forward_query(z, qt, ctx)

    def forecast(self, ctx, n=96):
        z = self.encoder(ctx)
        t = torch.linspace(1, 2, n, device=ctx.device).unsqueeze(0).expand(ctx.shape[0], -1)
        return self._forward_query(z, t, ctx)

    def impute(self, ctx, n=96):
        z = self.encoder(ctx)
        t = torch.linspace(0, 1, n, device=ctx.device).unsqueeze(0).expand(ctx.shape[0], -1)
        return self._forward_query(z, t, ctx)


# ============================================================
# Sampling
# ============================================================
def sample_forecast_targets(window, nq=16):
    """window: [192], return: (ctx [96], qt [nq], qv [nq])"""
    ctx = window[:96]
    future = window[96:]
    qi = np.random.choice(96, nq, replace=False)
    qt = (1.0 + qi.astype(np.float32) / 96)
    qv = future[qi]
    return ctx, qt, qv

def sample_imputation_targets(window, nq=16, mr=0.375):
    """window: [192] or [96], return: (masked_ctx [96], qt [nq], qv [nq])"""
    if len(window) >= 96:
        full = window[:96]
    else:
        full = np.pad(window, (0, 96-len(window)))

    mask = np.random.rand(96) > mr
    qi = np.where(~mask)[0]
    if len(qi) == 0: return None
    if len(qi) > nq:
        qi = np.random.choice(qi, nq, replace=False)
    elif len(qi) < nq:
        # Repeat to fill
        qi = np.tile(qi, nq // len(qi) + 1)[:nq]

    masked_ctx = full * mask.astype(np.float32)
    qt = qi.astype(np.float32) / 96
    qv = full[qi]
    return masked_ctx, qt, qv


def collate_batch(windows, mode='forecast'):
    """windows: list of [192] or [96] arrays"""
    ctxs, qts, qvs = [], [], []
    for w in windows:
        w = w.numpy() if isinstance(w, torch.Tensor) else w
        if mode == 'forecast' and len(w) >= 192:
            ctx, qt, qv = sample_forecast_targets(w)
        else:
            result = sample_imputation_targets(w)
            if result is None: continue
            ctx, qt, qv = result
        ctxs.append(ctx); qts.append(qt); qvs.append(qv)
    if not ctxs: return None
    return (torch.tensor(np.stack(ctxs), dtype=torch.float32),
            torch.tensor(np.stack(qts), dtype=torch.float32),
            torch.tensor(np.stack(qvs), dtype=torch.float32))


# ============================================================
# Training Loop
# ============================================================
def train(model, datasets, save_path, epochs=20, lr=0.0003, batch_size=64):
    print(f'\n{"="*60}')
    print('Full Scale Pre-training')
    print(f'  Encoder: PatchAttention (6-layer, d={HIDDEN})')
    print(f'  Trunks: 3 HyperNet (Fourier/Poly/RBF, w={WIDTH})')
    print(f'  RevIN: OFF')
    print(f'  Informed Query: ON')
    print(f'  Multi-task: Forecast + Imputation')
    print(f'{"="*60}')

    n = sum(p.numel() for p in model.parameters())
    print(f'Model: {n/1e6:.1f}M params')

    # Combine datasets
    from torch.utils.data import ConcatDataset
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
            # Random task: 50% forecast, 50% imputation
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
              f'(fc={np.mean(fc_losses):.4f} imp={np.mean(imp_losses):.4f}) '
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
    np.random.seed(42)
    torch.manual_seed(42)

    print('Loading datasets...')
    pile_ds = StreamingPileDataset(max_samples=1000000)  # 1M from Pile
    tempopfn_ds = TempoPFNDataset(max_samples=200000)     # 200K from TempoPFN

    print(f'\nTotal data: {len(pile_ds):,} (Pile) + {len(tempopfn_ds):,} (TempoPFN) = {len(pile_ds)+len(tempopfn_ds):,}')

    model = FullScaleModel().to(DEVICE)
    n = sum(p.numel() for p in model.parameters())
    n_enc = sum(p.numel() for p in model.encoder.parameters())
    n_heads = sum(sum(p.numel() for p in h.parameters()) for h in model.heads)
    print(f'\nModel: {n/1e6:.1f}M total')
    print(f'  Encoder: {n_enc/1e6:.1f}M')
    print(f'  Heads: {n_heads/1e6:.1f}M')

    save_path = 'checkpoints/full_scale_run.pth'
    os.makedirs('checkpoints', exist_ok=True)

    model = train(model, [pile_ds, tempopfn_ds], save_path, epochs=20, lr=0.0003)
    print('\nDONE')
