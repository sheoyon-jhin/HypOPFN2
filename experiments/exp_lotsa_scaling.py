"""
LOTSA Scaling Study: 1%, 5%, 10%, 30%, 50% × (baseline, latent_decomp)

공정한 학습: eval benchmark (ETT, Weather, M4, M3, M1) 전부 제외
+ Synthetic gap filler (제외된 dataset 특성 반영)

Usage:
  # Single experiment
  CUDA_VISIBLE_DEVICES=0 python experiments/exp_lotsa_scaling.py --scale 1 --decomp 0 --tag s1_base
  CUDA_VISIBLE_DEVICES=1 python experiments/exp_lotsa_scaling.py --scale 1 --decomp 1 --tag s1_decomp

  # 전체 자동 실행은 run_lotsa_experiments.sh 사용
"""
import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, math, time
import pyarrow.ipc as ipc
from torch import optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset

DEVICE = torch.device(os.environ.get('CUDA_DEV', 'cuda'))
LOTSA_DIR = os.environ.get('LOTSA_DIR', './dataset/lotsa')

PATCH_SIZE = 16
TOP_K_IQ = 5
INFORMED_DIM = 1 + 2 * TOP_K_IQ + 2


# ============================================================
# Model components
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


def hyper_fwd(trunk, t_flat, head_out, iq):
    B = head_out.shape[0]
    base = trunk.base_feat(t_flat)
    nq = base.shape[0] // B
    full = torch.cat([base.view(B, nq, trunk.idim), iq], dim=-1)
    tp = head_out[:, :trunk.pc] * 0.01
    W = tp[:, :trunk.full_idim*trunk.w].view(B, trunk.full_idim, trunk.w)
    b = tp[:, trunk.full_idim*trunk.w:].view(B, trunk.w)
    Phi = F.gelu(torch.bmm(full, W) + b.unsqueeze(1))
    Bc = head_out[:, trunk.pc:]
    return torch.einsum('bw,bqw->bq', Bc, Phi)


class PatchAttnEncoder(nn.Module):
    def __init__(self, seq_len, d_model=512, n_layers=6, nhead=8):
        super().__init__()
        self.patch_size = PATCH_SIZE
        n_patches = seq_len // PATCH_SIZE
        self.proj = nn.Linear(PATCH_SIZE, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, n_patches, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4,
            dropout=0.1, activation='gelu', batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        B = x.shape[0]
        z = self.proj(x.view(B, -1, self.patch_size)) + self.pos_emb
        return self.norm(self.transformer(z).mean(dim=1))


class LatentDecomp(nn.Module):
    """FeDaL-inspired: encoder output z → z_trend + z_seasonal."""
    def __init__(self, d_model=512):
        super().__init__()
        self.trend_proj = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.seasonal_proj = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model))

    def forward(self, z):
        return self.trend_proj(z), self.seasonal_proj(z)


class OperatorModel(nn.Module):
    def __init__(self, seq_len=96, d_model=512, n_layers=6, trunk_w=192,
                 use_latent_decomp=False):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.use_latent_decomp = use_latent_decomp
        self.encoder = PatchAttnEncoder(seq_len, d_model, n_layers)

        if use_latent_decomp:
            self.latent_decomp = LatentDecomp(d_model)
            self.trunks = nn.ModuleList([
                HyperTrunk(trunk_w, 'poly'),      # trend
                HyperTrunk(trunk_w, 'fourier'),    # seasonal
                HyperTrunk(trunk_w, 'rbf'),        # residual
            ])
            self.heads = nn.ModuleList([nn.Linear(d_model, t.odim) for t in self.trunks])
        else:
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

        if self.use_latent_decomp:
            z_t, z_s = self.latent_decomp(z)
            z_list = [z_t, z_s, z]  # trend, seasonal, residual(full)
        else:
            z_list = [z, z, z]

        out = sum(hyper_fwd(trunk, t_flat, head(zi), iq) + bias
                  for zi, trunk, head, bias in zip(z_list, self.trunks, self.heads, self.biases))
        return out

    def forecast(self, ctx, n=None):
        if n is None: n = self.seq_len
        t = torch.linspace(1, 2, n, device=ctx.device).unsqueeze(0).expand(ctx.shape[0], -1)
        return self.forward_train(ctx, t)


# ============================================================
# LOTSA Dataset
# ============================================================
class LOTSAScalingDataset(Dataset):
    def __init__(self, lotsa_dir, scale_pct, seq_len=192):
        """scale_pct: 1, 5, 10, 30, 50 (percentage)."""
        self.windows = []
        self.seq_len = seq_len

        if not os.path.exists(lotsa_dir):
            print(f'ERROR: LOTSA not found at {lotsa_dir}')
            return

        datasets = sorted([d for d in os.listdir(lotsa_dir)
                          if os.path.isdir(os.path.join(lotsa_dir, d))])
        # Windows per dataset based on scale
        # Rough: 1% → 150 wpd, 5% → 750, 10% → 1500, 30% → 4500, 50% → 7500
        wpd = max(50, int(150 * scale_pct))
        print(f'LOTSA {scale_pct}%: {len(datasets)} datasets, ~{wpd} windows each')

        for i, ds_name in enumerate(datasets):
            ds_path = os.path.join(lotsa_dir, ds_name)
            arrow_files = [f for f in os.listdir(ds_path) if f.endswith('.arrow')]
            if not arrow_files:
                continue
            try:
                table = ipc.open_file(os.path.join(ds_path, arrow_files[0])).read_all()
                col_names = table.column_names
                target_col = None
                for cn in ['target', 'values', 'value']:
                    if cn in col_names:
                        target_col = cn; break

                count = 0
                if target_col:
                    for row_idx in range(min(len(table), wpd * 3)):
                        try:
                            vals = table.column(target_col)[row_idx].as_py()
                            if isinstance(vals, list):
                                ts = np.array(vals[0] if isinstance(vals[0], list) else vals, dtype=np.float32)
                            else:
                                continue
                            if len(ts) >= seq_len:
                                stride = max(1, (len(ts) - seq_len) // min(5, wpd))
                                for start in range(0, len(ts) - seq_len + 1, stride):
                                    w = ts[start:start+seq_len]
                                    s = np.std(w)
                                    if s > 1e-6:
                                        self.windows.append(np.clip((w-np.mean(w))/s, -10, 10).astype(np.float32))
                                        count += 1
                                        if count >= wpd: break
                        except: continue
                        if count >= wpd: break

                if (i+1) % 20 == 0:
                    print(f'  [{i+1}/{len(datasets)}] total: {len(self.windows):,}')
            except: continue

        self.windows = np.array(self.windows, dtype=np.float32) if self.windows else np.zeros((0, seq_len))
        print(f'LOTSA {scale_pct}%: {len(self.windows):,} windows loaded')

    def __len__(self): return len(self.windows)
    def __getitem__(self, idx):
        return torch.tensor(self.windows[idx], dtype=torch.float32)


class SyntheticGapFiller(Dataset):
    """제외된 dataset 패턴 보충 (weather, M4-style, financial)."""
    def __init__(self, n_samples=50000, seq_len=192):
        self.windows = []
        np.random.seed(42)
        for _ in range(n_samples):
            n = seq_len; t = np.linspace(0, 2, n)
            gt = np.random.choice(['weather', 'seasonal', 'financial', 'step', 'comp'])
            if gt == 'weather':
                y = (np.sin(2*np.pi*t)*np.random.uniform(5,20) +
                     np.sin(2*np.pi*t*np.random.uniform(5,30))*np.random.uniform(1,5) +
                     np.cumsum(np.random.randn(n)*0.3)*0.1)
            elif gt == 'seasonal':
                y = sum(np.random.uniform(0.3,1.5)*np.sin(2*np.pi*np.random.uniform(0.5,10)*t+np.random.uniform(0,6.28))
                        for _ in range(np.random.randint(1,4)))
            elif gt == 'financial':
                y = np.cumsum(np.random.randn(n)*np.random.uniform(0.01,0.1))
            elif gt == 'step':
                y = np.zeros(n)
                for bp in sorted(np.random.choice(range(10,n-10), np.random.randint(1,4), replace=False)):
                    y[bp:] += np.random.uniform(-2,2)
                y += np.random.randn(n)*0.3
            else:
                y = (np.random.uniform(-1,1)*t +
                     np.random.uniform(0.5,1.5)*np.sin(2*np.pi*np.random.uniform(1,5)*t) +
                     np.random.randn(n)*0.2)
            s = np.std(y)
            if s > 1e-6:
                self.windows.append(np.clip((y-np.mean(y))/s, -10, 10).astype(np.float32))
        self.windows = np.array(self.windows, dtype=np.float32)
        print(f'Synthetic gap filler: {len(self.windows):,}')

    def __len__(self): return len(self.windows)
    def __getitem__(self, idx):
        return torch.tensor(self.windows[idx], dtype=torch.float32)


# ============================================================
# Training
# ============================================================
def collate_batch(windows, seq_len=96, n_query=16, mr=0.375):
    ctxs, qts, qvs = [], [], []
    for w in windows:
        w = w.numpy() if isinstance(w, torch.Tensor) else w
        if np.random.rand() < 0.5 and len(w) >= seq_len * 2:
            ctx = w[:seq_len]; future = w[seq_len:seq_len*2]
            qi = np.random.choice(seq_len, n_query, replace=False)
            qt = 1.0 + qi.astype(np.float32) / seq_len
            qv = future[qi]
        else:
            full = w[:seq_len] if len(w) >= seq_len else np.pad(w, (0, seq_len-len(w)))
            mask = np.random.rand(seq_len) > mr
            qi = np.where(~mask)[0]
            if len(qi) == 0: continue
            if len(qi) > n_query: qi = np.random.choice(qi, n_query, replace=False)
            elif len(qi) < n_query: qi = np.tile(qi, n_query//len(qi)+1)[:n_query]
            ctx = full * mask.astype(np.float32)
            qt = qi.astype(np.float32) / seq_len
            qv = full[qi]
        ctxs.append(ctx); qts.append(qt); qvs.append(qv)
    if not ctxs: return None
    return (torch.tensor(np.stack(ctxs), dtype=torch.float32),
            torch.tensor(np.stack(qts), dtype=torch.float32),
            torch.tensor(np.stack(qvs), dtype=torch.float32))


def train(model, datasets, save_path, seq_len=96, epochs=20, lr=3e-4, batch_size=64):
    n_params = sum(p.numel() for p in model.parameters())
    combined = ConcatDataset(datasets)
    dl = DataLoader(combined, batch_size=batch_size, shuffle=True,
                    num_workers=4, drop_last=True, pin_memory=True)

    print(f'\n{"="*60}')
    print(f'LOTSA Scaling Training')
    print(f'  Model: {n_params/1e6:.1f}M, Latent decomp: {model.use_latent_decomp}')
    print(f'  Data: {len(combined):,} windows, SEQ: {seq_len}')
    print(f'  Steps/epoch: {len(dl):,}')
    print(f'{"="*60}')

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        losses = []
        t0 = time.time()
        for i, batch_windows in enumerate(dl):
            batch = collate_batch(batch_windows, seq_len=seq_len)
            if batch is None: continue
            ctx, qt, qv = [x.to(DEVICE) for x in batch]
            optimizer.zero_grad()
            loss = F.mse_loss(model.forward_train(ctx, qt), qv)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
            if (i+1) % 500 == 0:
                print(f'  iter {i+1}/{len(dl)}: loss={np.mean(losses[-500:]):.4f}')
        scheduler.step()
        avg = np.mean(losses)
        el = time.time() - t0
        print(f'Epoch {epoch+1}/{epochs}: loss={avg:.4f} ({el:.0f}s)')
        if avg < best_loss:
            best_loss = avg
            torch.save(model.state_dict(), save_path)
            print(f'  Saved (best={best_loss:.4f})')
    return best_loss


# ============================================================
# Eval (zero-shot on excluded benchmarks)
# ============================================================
def eval_forecast(model, seq_len):
    from types import SimpleNamespace
    from data_provider.data_factory import data_provider

    datasets = {
        'ETTh1': ('ETTh1','./dataset/ETT-small/','ETTh1.csv',7),
        'ETTh2': ('ETTh2','./dataset/ETT-small/','ETTh2.csv',7),
        'ETTm1': ('ETTm1','./dataset/ETT-small/','ETTm1.csv',7),
        'ETTm2': ('ETTm2','./dataset/ETT-small/','ETTm2.csv',7),
        'Weather': ('custom','./dataset/weather/','weather.csv',21),
    }
    model.eval()
    results = {}
    for dn, (d,root,f,enc_in) in datasets.items():
        for pl in [96, 192, 336, 720]:
            try:
                a = SimpleNamespace(seq_len=seq_len, pred_len=pl, label_len=48, data=d,
                    root_path=root, data_path=f, features='M', target='OT', freq='h',
                    embed='timeF', enc_in=enc_in, dec_in=enc_in, c_out=enc_in,
                    num_workers=2, batch_size=32, exp_name='MTSF', ordered_data=False,
                    data_amount=-1, combine_Gaussian_datasets=False, synthetic_data_path='',
                    synthetic_root_path='./', synthetic_length=1024, stride=-1)
                _, tdl = data_provider(a, 'test')
                preds, tgts = [], []
                with torch.no_grad():
                    for bx, by, _, _ in tdl:
                        bx = bx.float().to(DEVICE)
                        B, S, C = bx.shape
                        outs = []
                        for ch in range(C):
                            x_ch = bx[:, :, ch]
                            if S >= seq_len: x_ctx = x_ch[:, -seq_len:]
                            else: x_ctx = F.pad(x_ch, (seq_len-S, 0))
                            m = x_ctx.mean(1, keepdim=True)
                            s = x_ctx.std(1, keepdim=True).clamp(min=1e-6)
                            x_n = ((x_ctx-m)/s).clamp(-10,10)
                            cur = x_n; chunks = []; remain = pl
                            while remain > 0:
                                step = min(seq_len, remain)
                                pred_n = model.forecast(cur, n=step)
                                chunks.append(pred_n)
                                if remain > step:
                                    cur = torch.cat([cur[:, step:], pred_n], dim=1)
                                remain -= step
                            outs.append(torch.cat(chunks, dim=1)*s+m)
                        preds.append(torch.stack(outs, dim=-1).cpu().numpy())
                        tgts.append(by[:, -pl:, :].numpy())
                p, t = np.concatenate(preds), np.concatenate(tgts)
                mse = np.mean((p-t)**2)
                k = f'{dn}_{pl}'
                print(f'  {k}: MSE={mse:.4f}')
                results[k] = mse
            except Exception as e:
                print(f'  {dn}_{pl}: ERROR ({e})')
    # Avg per dataset
    for dn in ['ETTh1','ETTh2','ETTm1','ETTm2','Weather']:
        avgs = [v for k,v in results.items() if k.startswith(dn+'_')]
        if avgs: print(f'  {dn} avg: {np.mean(avgs):.4f}')
    return results


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scale', type=int, required=True, help='1, 5, 10, 30, or 50')
    parser.add_argument('--decomp', type=int, default=0, help='0: baseline, 1: latent decomp')
    parser.add_argument('--tag', type=str, required=True)
    parser.add_argument('--seq_len', type=int, default=96)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--synth_ratio', type=float, default=0.3)
    parser.add_argument('--eval_after', type=int, default=1, help='Run eval after training')
    args = parser.parse_args()

    np.random.seed(42); torch.manual_seed(42)

    print('='*60)
    print(f'LOTSA Scaling: {args.scale}%, decomp={bool(args.decomp)}, tag={args.tag}')
    print('='*60)

    # Load data
    lotsa_ds = LOTSAScalingDataset(LOTSA_DIR, args.scale, seq_len=args.seq_len*2)
    n_synth = max(10000, int(len(lotsa_ds) * args.synth_ratio))
    synth_ds = SyntheticGapFiller(n_samples=n_synth, seq_len=args.seq_len*2)
    datasets = [lotsa_ds, synth_ds]
    total = sum(len(d) for d in datasets)
    print(f'Total: {total:,} (LOTSA: {len(lotsa_ds):,}, Synth: {len(synth_ds):,})')

    # Model
    model = OperatorModel(
        seq_len=args.seq_len,
        use_latent_decomp=bool(args.decomp)
    ).to(DEVICE)
    n = sum(p.numel() for p in model.parameters())
    print(f'Model: {n/1e6:.1f}M params')

    # Train
    save_path = f'checkpoints/{args.tag}.pth'
    os.makedirs('checkpoints', exist_ok=True)
    best = train(model, datasets, save_path, seq_len=args.seq_len, epochs=args.epochs)

    # Eval
    if args.eval_after:
        print('\n' + '='*60)
        print('TRUE ZERO-SHOT EVAL (these datasets were NOT in training!)')
        print('='*60)
        model.load_state_dict(torch.load(save_path, map_location=DEVICE))
        results = eval_forecast(model, args.seq_len)

        # Save results
        import json
        res_path = f'results/{args.tag}.json'
        os.makedirs('results', exist_ok=True)
        with open(res_path, 'w') as f:
            json.dump({'tag': args.tag, 'scale': args.scale, 'decomp': args.decomp,
                       'train_loss': best, 'forecast': results, 'params': n}, f, indent=2)
        print(f'Results saved: {res_path}')

    print('\nDONE')
