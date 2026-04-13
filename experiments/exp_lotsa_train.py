"""
LOTSA Training: eval benchmark 제외한 공정한 학습

Data: LOTSA 156 datasets (weather/m1/m3/m4 제외) + Diverse Synthetic
Model: 32.5M (0.1%) or 80M (1%)

True zero-shot eval: ETT, Weather, M4, Solar, PEMS

Usage:
  CUDA_VISIBLE_DEVICES=3 python experiments/exp_lotsa_train.py --scale 0.1 --tag lotsa_01pct
  CUDA_VISIBLE_DEVICES=3 python experiments/exp_lotsa_train.py --scale 1.0 --model_size 80M --tag lotsa_1pct
"""
import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, math, time
import pyarrow as pa, pyarrow.ipc as ipc
from torch import optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset

DEVICE = torch.device(os.environ.get('CUDA_DEV', 'cuda'))
LOTSA_DIR = '/workspace/HypOPFN/dataset/lotsa'

SEQ_LEN = 96
PATCH_SIZE = 16
TOP_K_IQ = 5
INFORMED_DIM = 1 + 2 * TOP_K_IQ + 2


# ============================================================
# Model (same architecture, scalable)
# ============================================================
def extract_freq(x, top_k=TOP_K_IQ):
    fft = torch.fft.rfft(x, dim=-1)
    mag = fft.abs(); mag[:, 0] = 0
    idx = torch.topk(mag, top_k, dim=-1).indices
    phase = torch.angle(torch.gather(fft, 1, idx))
    return idx.float(), phase, x[:, -1:].detach(), (x[:, -1:] - x[:, -2:-1]).detach()


class HyperTrunk(nn.Module):
    def __init__(self, w, btype, nf=32, deg=6, nc=20):
        super().__init__(); self.w = w; self.btype = btype
        if btype == 'fourier': self.nf = nf; self.idim = 1 + 2*nf
        elif btype == 'poly': self.deg = deg; self.idim = deg + 1
        elif btype == 'rbf':
            self.register_buffer('centers', torch.linspace(0, 2, nc))
            self.idim = 1 + nc
        self.full_idim = self.idim + INFORMED_DIM
        self.pc = self.full_idim * w + w
        self.odim = self.pc + w

    def base_feat(self, t):
        t = t.unsqueeze(-1) if t.dim() == 1 else t
        if self.btype == 'fourier':
            f = torch.arange(1, self.nf+1, device=t.device, dtype=t.dtype)
            return torch.cat([t, torch.sin(2*math.pi*f*t), torch.cos(2*math.pi*f*t)], dim=-1)
        elif self.btype == 'poly':
            return torch.cat([t**i for i in range(self.deg+1)], dim=-1)
        elif self.btype == 'rbf':
            return torch.cat([t, torch.exp(-20*(t-self.centers.unsqueeze(0))**2)], dim=-1)


def hyper_fwd_iq(trunk, t_flat, head_out, iq):
    B = head_out.shape[0]
    base = trunk.base_feat(t_flat)
    nq = base.shape[0] // B
    base = base.view(B, nq, trunk.idim)
    full = torch.cat([base, iq], dim=-1)
    tp = head_out[:, :trunk.pc] * 0.01
    W = tp[:, :trunk.full_idim*trunk.w].view(B, trunk.full_idim, trunk.w)
    b = tp[:, trunk.full_idim*trunk.w:].view(B, trunk.w)
    Phi = F.gelu(torch.bmm(full, W) + b.unsqueeze(1))
    Bc = head_out[:, trunk.pc:]
    return torch.einsum('bw,bqw->bq', Bc, Phi)


class PatchAttnEncoder(nn.Module):
    def __init__(self, d_model, n_layers, nhead=8):
        super().__init__()
        n_patches = SEQ_LEN // PATCH_SIZE
        self.proj = nn.Linear(PATCH_SIZE, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, n_patches, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4,
            dropout=0.1, activation='gelu', batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        B = x.shape[0]
        z = self.proj(x.view(B, -1, PATCH_SIZE)) + self.pos_emb
        return self.norm(self.transformer(z).mean(dim=1))


class OperatorModel(nn.Module):
    def __init__(self, d_model=512, n_layers=6, trunk_w=192):
        super().__init__()
        self.encoder = PatchAttnEncoder(d_model, n_layers)
        self.trunks = nn.ModuleList([
            HyperTrunk(trunk_w, 'fourier'),
            HyperTrunk(trunk_w, 'poly'),
            HyperTrunk(trunk_w, 'rbf'),
        ])
        self.heads = nn.ModuleList([nn.Linear(d_model, t.odim) for t in self.trunks])
        for h in self.heads:
            nn.init.xavier_normal_(h.weight, gain=0.1)
            nn.init.constant_(h.bias, 0)
        self.biases = nn.ParameterList([nn.Parameter(torch.zeros(1)) for _ in self.trunks])
        self.d_model = d_model

    def _build_iq(self, ctx, qt):
        freqs, phases, lv, ls = extract_freq(ctx)
        B, nq = qt.shape
        t_exp = qt.unsqueeze(-1)
        f, p = freqs.unsqueeze(1), phases.unsqueeze(1)
        ang = 2*math.pi*f*t_exp + p
        return torch.cat([t_exp, torch.sin(ang), torch.cos(ang),
                          lv.unsqueeze(1).expand(-1,nq,-1),
                          ls.unsqueeze(1).expand(-1,nq,-1)], dim=-1)

    def forward_train(self, ctx, qt):
        z = self.encoder(ctx)
        iq = self._build_iq(ctx, qt)
        t_flat = qt.reshape(-1)
        out = sum(hyper_fwd_iq(t, t_flat, h(z), iq) + b
                  for t, h, b in zip(self.trunks, self.heads, self.biases))
        return out

    def forecast(self, ctx, n=96):
        t = torch.linspace(1, 2, n, device=ctx.device).unsqueeze(0).expand(ctx.shape[0], -1)
        return self.forward_train(ctx, t)


def make_model(size='32M'):
    if size == '32M':
        return OperatorModel(d_model=512, n_layers=6, trunk_w=192)
    elif size == '80M':
        return OperatorModel(d_model=640, n_layers=8, trunk_w=256)
    elif size == '200M':
        return OperatorModel(d_model=768, n_layers=12, trunk_w=320)


# ============================================================
# LOTSA Dataset
# ============================================================
class LOTSADataset(Dataset):
    def __init__(self, lotsa_dir, windows_per_dataset=1000, seq_len=192):
        self.windows = []
        self.seq_len = seq_len

        if not os.path.exists(lotsa_dir):
            print(f'LOTSA dir not found: {lotsa_dir}')
            return

        datasets = sorted([d for d in os.listdir(lotsa_dir)
                          if os.path.isdir(os.path.join(lotsa_dir, d))])
        print(f'Loading LOTSA: {len(datasets)} datasets, {windows_per_dataset} windows each')

        for i, ds_name in enumerate(datasets):
            ds_path = os.path.join(lotsa_dir, ds_name)
            arrow_files = [f for f in os.listdir(ds_path) if f.endswith('.arrow')]
            if not arrow_files:
                continue

            try:
                fp = os.path.join(ds_path, arrow_files[0])
                table = ipc.open_file(fp).read_all()

                # Try to find value columns
                col_names = table.column_names
                # LOTSA format: typically has 'target' or numeric columns
                target_col = None
                for cn in ['target', 'values', 'value']:
                    if cn in col_names:
                        target_col = cn
                        break

                count = 0
                if target_col:
                    for row_idx in range(min(len(table), windows_per_dataset * 2)):
                        try:
                            vals = table.column(target_col)[row_idx].as_py()
                            if isinstance(vals, list):
                                if isinstance(vals[0], list):
                                    # Multi-variate: take first channel
                                    ts = np.array(vals[0], dtype=np.float32)
                                else:
                                    ts = np.array(vals, dtype=np.float32)
                            else:
                                continue

                            # Extract windows
                            if len(ts) >= seq_len:
                                for start in range(0, len(ts) - seq_len + 1,
                                                   max(1, (len(ts) - seq_len) // 3)):
                                    w = ts[start:start + seq_len]
                                    s = np.std(w)
                                    if s > 1e-6:
                                        w = np.clip((w - np.mean(w)) / s, -10, 10).astype(np.float32)
                                        self.windows.append(w)
                                        count += 1
                                        if count >= windows_per_dataset:
                                            break
                        except:
                            continue
                        if count >= windows_per_dataset:
                            break
                else:
                    # Try numeric columns directly
                    for cn in col_names:
                        try:
                            col = table.column(cn)
                            if hasattr(col[0], 'as_py'):
                                vals = [col[j].as_py() for j in range(min(len(col), 10000))]
                                ts = np.array(vals, dtype=np.float32)
                                if len(ts) >= seq_len:
                                    for start in range(0, len(ts)-seq_len+1, seq_len//2):
                                        w = ts[start:start+seq_len]
                                        s = np.std(w)
                                        if s > 1e-6:
                                            w = np.clip((w-np.mean(w))/s, -10, 10).astype(np.float32)
                                            self.windows.append(w)
                                            count += 1
                                            if count >= windows_per_dataset:
                                                break
                        except:
                            continue
                        if count >= windows_per_dataset:
                            break

                if (i+1) % 20 == 0 or i < 5:
                    print(f'  [{i+1}/{len(datasets)}] {ds_name}: {count} windows (total: {len(self.windows)})')
            except Exception as e:
                if i < 10:
                    print(f'  [{i+1}] {ds_name}: ERROR ({type(e).__name__}: {e})')

        self.windows = np.array(self.windows, dtype=np.float32) if self.windows else np.zeros((0, seq_len), dtype=np.float32)
        print(f'LOTSA loaded: {len(self.windows)} windows from {len(datasets)} datasets')

    def __len__(self): return len(self.windows)
    def __getitem__(self, idx):
        return torch.tensor(self.windows[idx], dtype=torch.float32)


# ============================================================
# Diverse Synthetic (weather/M4 패턴 보충)
# ============================================================
class SyntheticGapFiller(Dataset):
    def __init__(self, n_samples=50000, seq_len=192):
        self.windows = []
        np.random.seed(42)
        for _ in range(n_samples):
            n = seq_len
            t = np.linspace(0, 2, n)
            gen_type = np.random.choice(['weather', 'seasonal', 'financial', 'step', 'compositional'])

            if gen_type == 'weather':
                annual = np.sin(2*np.pi*t) * np.random.uniform(5, 20)
                daily = np.sin(2*np.pi*t*np.random.uniform(5, 30)) * np.random.uniform(1, 5)
                noise = np.cumsum(np.random.randn(n) * 0.3) * 0.1
                y = annual + daily + noise
            elif gen_type == 'seasonal':
                y = sum(np.random.uniform(0.3, 1.5) * np.sin(2*np.pi*np.random.uniform(0.5, 10)*t + np.random.uniform(0, 6.28))
                        for _ in range(np.random.randint(1, 4)))
            elif gen_type == 'financial':
                y = np.cumsum(np.random.randn(n) * np.random.uniform(0.01, 0.1))
            elif gen_type == 'step':
                y = np.zeros(n)
                for bp in sorted(np.random.choice(range(10, n-10), np.random.randint(1, 4), replace=False)):
                    y[bp:] += np.random.uniform(-2, 2)
                y += np.random.randn(n) * 0.3
            else:
                y = (np.random.uniform(-1, 1) * t +
                     np.random.uniform(0.5, 1.5) * np.sin(2*np.pi*np.random.uniform(1, 5)*t) +
                     np.random.randn(n) * 0.2)

            s = np.std(y)
            if s > 1e-6:
                y = np.clip((y - np.mean(y)) / s, -10, 10).astype(np.float32)
                self.windows.append(y)

        self.windows = np.array(self.windows, dtype=np.float32)
        print(f'Synthetic gap filler: {len(self.windows)} windows')

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
            full = w[:96] if len(w) >= 96 else np.pad(w, (0, 96-len(w)))
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


def train(model, datasets, save_path, epochs=20, lr=3e-4, batch_size=64):
    n = sum(p.numel() for p in model.parameters())
    combined = ConcatDataset(datasets)
    dl = DataLoader(combined, batch_size=batch_size, shuffle=True,
                    num_workers=4, drop_last=True, pin_memory=True)
    steps = len(dl)

    print(f'\n{"="*60}')
    print(f'LOTSA Training (FAIR — eval benchmarks excluded)')
    print(f'  Model: {n/1e6:.1f}M')
    print(f'  Data: {len(combined):,} windows')
    print(f'  Steps/epoch: {steps}')
    print(f'{"="*60}')

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        losses = []
        t0 = time.time()
        for i, batch_windows in enumerate(dl):
            batch = collate_batch(batch_windows)
            if batch is None: continue
            ctx, qt, qv = [x.to(DEVICE) for x in batch]
            optimizer.zero_grad()
            loss = F.mse_loss(model.forward_train(ctx, qt), qv)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
            if (i+1) % 500 == 0:
                print(f'  iter {i+1}/{steps}: loss={np.mean(losses[-500:]):.4f}')
        scheduler.step()
        avg = np.mean(losses)
        print(f'Epoch {epoch+1}/{epochs}: loss={avg:.4f} ({time.time()-t0:.0f}s)')
        if avg < best_loss:
            best_loss = avg
            torch.save(model.state_dict(), save_path)
            print(f'  Saved (best={best_loss:.4f})')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scale', type=float, default=0.1, help='% of LOTSA (0.1=0.1%, 1.0=1%)')
    parser.add_argument('--model_size', type=str, default='32M', choices=['32M', '80M', '200M'])
    parser.add_argument('--tag', type=str, default='lotsa_01pct')
    parser.add_argument('--synth_ratio', type=float, default=0.3, help='Synthetic gap filler ratio')
    args = parser.parse_args()

    np.random.seed(42); torch.manual_seed(42)

    # Windows per dataset based on scale
    # 0.1% of 231B / 156 datasets / avg 1000 length ≈ 1500 windows/dataset
    # 1% ≈ 15000 windows/dataset
    wpd = int(1500 * args.scale)
    wpd = max(100, wpd)

    print(f'LOTSA Training [{args.tag}]')
    print(f'  Scale: {args.scale}%, ~{wpd} windows/dataset')
    print(f'  Model: {args.model_size}')

    # Load LOTSA
    lotsa_ds = LOTSADataset(LOTSA_DIR, windows_per_dataset=wpd, seq_len=192)

    # Synthetic gap filler
    n_synth = int(len(lotsa_ds) * args.synth_ratio)
    synth_ds = SyntheticGapFiller(n_samples=max(10000, n_synth), seq_len=192)

    datasets = [lotsa_ds, synth_ds]
    total = sum(len(d) for d in datasets)
    print(f'Total: {total:,} windows (LOTSA: {len(lotsa_ds):,}, Synth: {len(synth_ds):,})')

    model = make_model(args.model_size).to(DEVICE)
    n = sum(p.numel() for p in model.parameters())
    print(f'Model: {n/1e6:.1f}M params')

    save_path = f'checkpoints/{args.tag}.pth'
    os.makedirs('checkpoints', exist_ok=True)
    train(model, datasets, save_path, epochs=20, lr=3e-4)
    print('\nDONE')
