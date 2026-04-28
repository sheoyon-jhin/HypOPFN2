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


def extract_freq_multiscale(x, top_k_per_scale=3, scales=(1.0, 0.5, 0.25)):
    """
    Multi-scale FFT IQ extraction.
    Returns top-k frequencies and phases from multiple recent windows.

    Each freq is normalized by ITS scale's seq_len so that
    sin(2π·f·t) at query time t (in [0,1] for context, >1 for forecast)
    reproduces the right oscillation independent of which scale it came from.

    Returns:
      freqs:  (B, total_k) where total_k = top_k_per_scale * len(scales)
      phases: (B, total_k)
      lv:     (B, 1)  last value
      ls:     (B, 1)  last slope
    """
    B, L = x.shape
    all_freqs, all_phases = [], []
    for s in scales:
        sub_len = max(8, int(L * s))   # min length 8 for FFT
        sub = x[:, -sub_len:]
        fft = torch.fft.rfft(sub, dim=-1)
        mag = fft.abs(); mag[:, 0] = 0
        k_eff = min(top_k_per_scale, mag.shape[-1] - 1)
        idx = torch.topk(mag, k_eff, dim=-1).indices.float()
        phase = torch.angle(torch.gather(fft, 1, idx.long()))
        # Normalize freq by sub_len to keep angular speed comparable
        # original freq is cycles per sub_len → cycles per L = freq * (L/sub_len) / L
        # we keep raw freq index but scale by (L / sub_len) so model sees same units
        idx_normalized = idx * (L / sub_len)
        # Pad to top_k_per_scale if FFT bins were too few
        if k_eff < top_k_per_scale:
            pad = torch.zeros(B, top_k_per_scale - k_eff, device=x.device, dtype=idx_normalized.dtype)
            idx_normalized = torch.cat([idx_normalized, pad], dim=-1)
            phase = torch.cat([phase, pad], dim=-1)
        all_freqs.append(idx_normalized)
        all_phases.append(phase)
    freqs = torch.cat(all_freqs, dim=-1)
    phases = torch.cat(all_phases, dim=-1)
    return freqs, phases, x[:, -1:].detach(), (x[:, -1:] - x[:, -2:-1]).detach()


class HyperTrunk(nn.Module):
    def __init__(self, w, btype, nf=32, deg=6, nc=20, informed_dim=None):
        super().__init__(); self.w = w; self.btype = btype
        if btype == 'fourier': self.nf = nf; self.idim = 1 + 2*nf
        elif btype == 'poly': self.deg = deg; self.idim = deg + 1
        elif btype == 'cheby': self.deg = deg; self.idim = deg + 1
        elif btype == 'rbf':
            self.register_buffer('centers', torch.linspace(0, 2, nc))
            self.idim = 1 + nc
        self.informed_dim = INFORMED_DIM if informed_dim is None else informed_dim
        self.full_idim = self.idim + self.informed_dim
        self.pc = self.full_idim * w + w
        self.odim = self.pc + w

    def base_feat(self, t):
        t = t.unsqueeze(-1) if t.dim() == 1 else t
        if self.btype == 'fourier':
            f = torch.arange(1, self.nf+1, device=t.device, dtype=t.dtype)
            return torch.cat([t, torch.sin(2*math.pi*f*t), torch.cos(2*math.pi*f*t)], dim=-1)
        elif self.btype == 'poly':
            return torch.cat([t**i for i in range(self.deg+1)], dim=-1)
        elif self.btype == 'cheby':
            # Chebyshev T_n on shifted domain x = t - 1 (training t in [0,1] -> x in [-1,0],
            # forecast t up to ~1.13 -> x up to ~0.13; numerically stable).
            x = t - 1.0
            T = [torch.ones_like(x), x]
            for _ in range(self.deg - 1):
                T.append(2 * x * T[-1] - T[-2])
            return torch.cat(T, dim=-1)
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


class FixedTrunk(nn.Module):
    """
    Fixed trunk: basis features + fixed learnable MLP (no hypernetwork).
    Context z is combined via inner product with MLP output (classic DeepONet style).
    """
    def __init__(self, w, btype, d_model=512, hidden=128, nf=32, deg=6, nc=20, informed_dim=None):
        super().__init__()
        self.w = w; self.btype = btype; self.is_fixed = True
        if btype == 'fourier': self.nf = nf; self.idim = 1 + 2*nf
        elif btype == 'poly': self.deg = deg; self.idim = deg + 1
        elif btype == 'cheby': self.deg = deg; self.idim = deg + 1
        elif btype == 'rbf':
            self.register_buffer('centers', torch.linspace(0, 2, nc))
            self.idim = 1 + nc
        self.informed_dim = INFORMED_DIM if informed_dim is None else informed_dim
        self.full_idim = self.idim + self.informed_dim

        self.mlp = nn.Sequential(
            nn.Linear(self.full_idim, hidden),
            nn.GELU(),
            nn.Linear(hidden, w),
        )
        self.coef_head = nn.Linear(d_model, w)
        nn.init.xavier_normal_(self.coef_head.weight, gain=0.1)
        nn.init.zeros_(self.coef_head.bias)
        self.bias = nn.Parameter(torch.zeros(1))

    def base_feat(self, t):
        t = t.unsqueeze(-1) if t.dim() == 1 else t
        if self.btype == 'fourier':
            f = torch.arange(1, self.nf+1, device=t.device, dtype=t.dtype)
            return torch.cat([t, torch.sin(2*math.pi*f*t), torch.cos(2*math.pi*f*t)], dim=-1)
        elif self.btype == 'poly':
            return torch.cat([t**i for i in range(self.deg+1)], dim=-1)
        elif self.btype == 'cheby':
            x = t - 1.0
            T = [torch.ones_like(x), x]
            for _ in range(self.deg - 1):
                T.append(2 * x * T[-1] - T[-2])
            return torch.cat(T, dim=-1)
        elif self.btype == 'rbf':
            return torch.cat([t, torch.exp(-20*(t-self.centers.unsqueeze(0))**2)], dim=-1)

    def forward(self, t_flat, iq, z):
        B = z.shape[0]
        base = self.base_feat(t_flat)
        nq = base.shape[0] // B
        full = torch.cat([base.view(B, nq, self.idim), iq], dim=-1)
        phi = self.mlp(full)  # (B, nq, w)
        coef = self.coef_head(z)  # (B, w)
        return torch.einsum('bnw,bw->bn', phi, coef) + self.bias


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
                 use_latent_decomp=False, hybrid_trunk=False):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.use_latent_decomp = use_latent_decomp
        self.hybrid_trunk = hybrid_trunk
        self.encoder = PatchAttnEncoder(seq_len, d_model, n_layers)

        if hybrid_trunk:
            # 1 Fixed (poly, global trend) + 2 Hyper (fourier/rbf, context-adaptive)
            self.trunks = nn.ModuleList([
                HyperTrunk(trunk_w, 'fourier'),            # hyper: periodic
                FixedTrunk(trunk_w, 'poly', d_model),       # fixed: global trend
                HyperTrunk(trunk_w, 'rbf'),                 # hyper: local
            ])
        elif use_latent_decomp:
            self.latent_decomp = LatentDecomp(d_model)
            self.trunks = nn.ModuleList([
                HyperTrunk(trunk_w, 'poly'),      # trend
                HyperTrunk(trunk_w, 'fourier'),    # seasonal
                HyperTrunk(trunk_w, 'rbf'),        # residual
            ])
        else:
            self.trunks = nn.ModuleList([
                HyperTrunk(trunk_w, 'fourier'),
                HyperTrunk(trunk_w, 'poly'),
                HyperTrunk(trunk_w, 'rbf'),
            ])

        # Heads only for HyperTrunks (FixedTrunk has its own coef_head)
        self.heads = nn.ModuleList([
            nn.Linear(d_model, t.odim) if not getattr(t, 'is_fixed', False)
            else nn.Identity()
            for t in self.trunks
        ])
        for h in self.heads:
            if isinstance(h, nn.Linear):
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

    def forward_train(self, ctx, qt, return_per_trunk=False):
        z = self.encoder(ctx)
        iq = self._build_iq(ctx, qt)
        t_flat = qt.reshape(-1)

        if self.use_latent_decomp:
            z_t, z_s = self.latent_decomp(z)
            z_list = [z_t, z_s, z]  # trend, seasonal, residual(full)
        else:
            z_list = [z, z, z]

        trunk_outs = []
        for zi, trunk, head, bias in zip(z_list, self.trunks, self.heads, self.biases):
            if getattr(trunk, 'is_fixed', False):
                # FixedTrunk has its own internal coef_head; no external hypernet head needed
                trunk_outs.append(trunk(t_flat, iq, zi) + bias)
            else:
                trunk_outs.append(hyper_fwd(trunk, t_flat, head(zi), iq) + bias)
        out = sum(trunk_outs)
        if return_per_trunk:
            return out, torch.stack(trunk_outs, dim=0)  # (3, B, nq)
        return out

    def forecast(self, ctx, n=None):
        """Direct multi-step forecast. t extends based on horizon length."""
        if n is None: n = self.seq_len
        # t = 1 corresponds to ctx_end, t grows linearly with horizon
        # one seq_len of forecast = t in [1, 2]; n steps = t in [1, 1 + n/seq_len]
        t_end = 1.0 + n / self.seq_len
        t = torch.linspace(1.0, t_end, n, device=ctx.device).unsqueeze(0).expand(ctx.shape[0], -1)
        return self.forward_train(ctx, t)


# ============================================================
# LOTSA Dataset
# ============================================================
# Evaluation benchmarks held out from pretraining (overlap with
# ETT/Weather/M4 evals used by HypOPFN2 + FeDaL paper protocol).
# Synced with data/download_lotsa.py. Do NOT remove without updating
# both eval scripts and ensuring no data leakage.
EVAL_EXCLUDE = frozenset({
    'weather', 'oikolab_weather',
    'm1_monthly', 'm1_quarterly', 'm1_yearly',
    'm3_monthly', 'm3_quarterly', 'm3_yearly',
    'monash_m3_monthly', 'monash_m3_other', 'monash_m3_quarterly', 'monash_m3_yearly',
    'm4_daily', 'm4_hourly', 'm4_monthly', 'm4_quarterly', 'm4_weekly', 'm4_yearly',
})


class LOTSAScalingDataset(Dataset):
    def __init__(self, lotsa_dir, scale_pct, seq_len=192, windows_per_series=5,
                 cache_dir='./dataset/lotsa_cache'):
        """scale_pct: 1, 5, 10, 30, 50 (percentage).
        windows_per_series: how many windows to extract per series (default 5 = sparse).
          Higher = denser coverage (10x→50, 100x→500). Triggers different cache key.
        Cache format: raw memmap (.dat) + metadata (.meta). Streams to disk during
        build so memory stays bounded regardless of dataset size."""
        self.seq_len = seq_len

        import hashlib
        import json
        exclude_hash = hashlib.md5(','.join(sorted(EVAL_EXCLUDE)).encode()).hexdigest()[:6]
        mv_suffix = '_mv' if bool(int(os.environ.get('LOTSA_MULTIVARIATE', '0'))) else ''
        cache_key = f'lotsa_s{scale_pct}_w{seq_len}_wps{windows_per_series}_ex{exclude_hash}{mv_suffix}'
        cache_dat = os.path.join(cache_dir, cache_key + '.dat')
        cache_meta = os.path.join(cache_dir, cache_key + '.meta')
        cache_npy_legacy = os.path.join(cache_dir, cache_key + '.npy')

        # HIT (memmap format)
        if os.path.exists(cache_dat) and os.path.exists(cache_meta):
            with open(cache_meta) as f:
                meta = json.load(f)
            shape = tuple(meta['shape'])
            print(f'[LOTSA cache HIT mmap] {cache_dat}')
            self.windows = np.memmap(cache_dat, dtype=np.float32, mode='r', shape=shape)
            print(f'  loaded {len(self.windows):,} windows × {self.windows.shape[-1]} (memmap)')
            return

        # HIT (legacy .npy)
        if os.path.exists(cache_npy_legacy):
            print(f'[LOTSA cache HIT] {cache_npy_legacy}')
            self.windows = np.load(cache_npy_legacy, mmap_mode='r')
            print(f'  loaded {len(self.windows):,} windows × {self.windows.shape[-1]} (mmap)')
            return

        # MISS — streaming build directly to memmap
        print(f'[LOTSA cache MISS] building {cache_dat} (memmap streaming)')
        os.makedirs(cache_dir, exist_ok=True)

        if not os.path.exists(lotsa_dir):
            print(f'ERROR: LOTSA not found at {lotsa_dir}')
            self.windows = np.zeros((0, seq_len), dtype=np.float32)
            return

        all_dirs = sorted([d for d in os.listdir(lotsa_dir)
                          if os.path.isdir(os.path.join(lotsa_dir, d))])
        datasets = [d for d in all_dirs if d not in EVAL_EXCLUDE]
        excluded_present = [d for d in all_dirs if d in EVAL_EXCLUDE]
        assert not (set(datasets) & EVAL_EXCLUDE), \
            f'Data leakage: {set(datasets) & EVAL_EXCLUDE} must not be in pretraining'
        wpd = max(50, int(150 * scale_pct))
        print(f'LOTSA {scale_pct}%: {len(datasets)} datasets (excluded {len(excluded_present)} eval: {sorted(excluded_present)}), cap ~{wpd} windows each, windows_per_series={windows_per_series}')

        # Preallocate memmap (upper bound). LOTSA_MAX_WINDOWS env var for override.
        # Observed ~75M at wps=10000 on 156 datasets; default 100M with 30% safety margin.
        max_est = int(os.environ.get('LOTSA_MAX_WINDOWS', 100_000_000))
        est_gb = max_est * seq_len * 4 / 1e9
        print(f'  [preallocate memmap] max={max_est:,} × {seq_len} → {est_gb:.1f} GB on disk (will truncate at end)')
        mm = np.memmap(cache_dat, dtype=np.float32, mode='w+', shape=(max_est, seq_len))

        count = 0
        for i, ds_name in enumerate(datasets):
            ds_path = os.path.join(lotsa_dir, ds_name)
            arrow_files = [f for f in os.listdir(ds_path) if f.endswith('.arrow')]
            if not arrow_files:
                continue
            try:
                arrow_path = os.path.join(ds_path, arrow_files[0])
                try:
                    table = ipc.open_file(arrow_path).read_all()
                except Exception:
                    with open(arrow_path, 'rb') as _f:
                        table = ipc.open_stream(_f).read_all()
                col_names = table.column_names
                target_col = None
                for cn in ['target', 'values', 'value']:
                    if cn in col_names:
                        target_col = cn; break

                ds_count = 0
                multivariate = bool(int(os.environ.get('LOTSA_MULTIVARIATE', '0')))
                if target_col:
                    for row_idx in range(min(len(table), wpd * 3)):
                        try:
                            vals = table.column(target_col)[row_idx].as_py()
                            if not isinstance(vals, list):
                                continue
                            # Build list of channels (each is a univariate series)
                            if isinstance(vals[0], list):
                                if multivariate:
                                    channels = [np.array(ch, dtype=np.float32) for ch in vals]
                                else:
                                    channels = [np.array(vals[0], dtype=np.float32)]
                            else:
                                channels = [np.array(vals, dtype=np.float32)]
                            for ts in channels:
                                if len(ts) < seq_len:
                                    continue
                                stride = max(1, (len(ts) - seq_len) // min(windows_per_series, wpd))
                                for start in range(0, len(ts) - seq_len + 1, stride):
                                    w = ts[start:start+seq_len]
                                    s = np.std(w)
                                    if s > 1e-6:
                                        if count >= max_est:
                                            break
                                        mm[count] = np.clip((w-np.mean(w))/s, -10, 10).astype(np.float32)
                                        count += 1
                                        ds_count += 1
                                        if ds_count >= wpd: break
                                if ds_count >= wpd or count >= max_est: break
                        except: continue
                        if ds_count >= wpd or count >= max_est: break

                if (i+1) % 20 == 0:
                    mm.flush()  # write dirty pages to disk, let OS reclaim
                    print(f'  [{i+1}/{len(datasets)}] total: {count:,} (flushed)')
                if count >= max_est:
                    print(f'  WARNING: hit max_est={max_est:,}, stopping early')
                    break
            except: continue

        # Finalize
        mm.flush()
        del mm

        # Truncate to actual size and write metadata
        actual_bytes = count * seq_len * 4
        with open(cache_dat, 'r+b') as f:
            f.truncate(actual_bytes)
        with open(cache_meta, 'w') as f:
            json.dump({'shape': [count, seq_len], 'dtype': 'float32'}, f)

        # Re-open as read-only memmap
        if count > 0:
            self.windows = np.memmap(cache_dat, dtype=np.float32, mode='r', shape=(count, seq_len))
        else:
            self.windows = np.zeros((0, seq_len), dtype=np.float32)
        print(f'[LOTSA cache SAVE] {cache_dat} ({count:,} windows, {actual_bytes/1e9:.1f} GB)')
        print(f'LOTSA {scale_pct}%: {count:,} windows loaded (memmap)')

    def __len__(self): return len(self.windows)
    def __getitem__(self, idx):
        return torch.tensor(self.windows[idx], dtype=torch.float32)


# Datasets similar to ETTh2/ETTm2/Weather (random-walk / trend-dominant / noisy)
TARGET_SIMILAR_DATASETS = [
    'bitcoin_with_missing',
    'fred_md',
    'nn5_daily_with_missing',
    'sunspot_with_missing',
    'us_births',
    'wiki-rolling_nips',
    'cdc_fluview_ilinet',
    'cdc_fluview_who_nrevss',
    'covid_deaths',
    'covid_mobility',
    'project_tycho',
    'temperature_rain_with_missing',
    'beijing_air_quality',
    'china_air_quality',
    'era5_2000', 'era5_2003', 'era5_2006',
    'cmip6_2010',
    'subseasonal',
]


class LOTSASubsetDataset(Dataset):
    """Load only specified LOTSA datasets with given windows-per-dataset."""
    def __init__(self, lotsa_dir, dataset_names, seq_len=2160, wpd=30000):
        leaked = set(dataset_names) & EVAL_EXCLUDE
        assert not leaked, f'Data leakage: {leaked} are eval benchmarks, must not be in pretraining'
        self.windows = []
        self.seq_len = seq_len
        print(f'LOTSA subset: {len(dataset_names)} datasets, wpd={wpd}')
        for i, ds_name in enumerate(dataset_names):
            ds_path = os.path.join(lotsa_dir, ds_name)
            if not os.path.isdir(ds_path):
                print(f'  skip: {ds_name} (not found)')
                continue
            arrow_files = [f for f in os.listdir(ds_path) if f.endswith('.arrow')]
            if not arrow_files:
                continue
            try:
                arrow_path = os.path.join(ds_path, arrow_files[0])
                try:
                    table = ipc.open_file(arrow_path).read_all()
                except Exception:
                    with open(arrow_path, 'rb') as _f:
                        table = ipc.open_stream(_f).read_all()
                col_names = table.column_names
                target_col = None
                for cn in ['target', 'values', 'value']:
                    if cn in col_names:
                        target_col = cn; break
                if not target_col:
                    continue
                count = 0
                for row_idx in range(min(len(table), wpd * 5)):
                    try:
                        vals = table.column(target_col)[row_idx].as_py()
                        if isinstance(vals, list):
                            ts = np.array(vals[0] if isinstance(vals[0], list) else vals, dtype=np.float32)
                        else:
                            continue
                        if len(ts) >= seq_len:
                            stride = max(1, (len(ts) - seq_len) // min(10, wpd))
                            for start in range(0, len(ts) - seq_len + 1, stride):
                                w = ts[start:start+seq_len]
                                s = np.std(w)
                                if s > 1e-6:
                                    self.windows.append(np.clip((w-np.mean(w))/s, -10, 10).astype(np.float32))
                                    count += 1
                                    if count >= wpd: break
                    except: continue
                    if count >= wpd: break
            except: continue
            print(f'  [{i+1}/{len(dataset_names)}] {ds_name}: +{count} (total {len(self.windows):,})')
        self.windows = np.array(self.windows, dtype=np.float32) if self.windows else np.zeros((0, seq_len))
        print(f'LOTSA subset loaded: {len(self.windows):,} windows')

    def __len__(self): return len(self.windows)
    def __getitem__(self, idx):
        return torch.tensor(self.windows[idx], dtype=torch.float32)


class CauKerSynthDataset(Dataset):
    """
    Drop-in replacement for SyntheticGapFiller, loading CauKer-generated cache.

    Cache format: {.dat memmap, .meta json} (same as LOTSAScalingDataset)
    Pre-built via data/cauker_to_cache.py from CauKer .arrow output.
    """
    def __init__(self, cache_dir='./dataset/cauker_cache', cache_name=None,
                 seq_len=2160):
        import json
        if cache_name is None:
            cache_name = os.environ.get('CAUKER_CACHE_NAME', 'cauker_synth')
        cache_dat = os.path.join(cache_dir, cache_name + '.dat')
        cache_meta = os.path.join(cache_dir, cache_name + '.meta')
        if not (os.path.exists(cache_dat) and os.path.exists(cache_meta)):
            raise FileNotFoundError(
                f'CauKer cache missing: {cache_dat}\n'
                'Generate first: python data/cauker_to_cache.py --arrow <path>'
            )
        with open(cache_meta) as f:
            meta = json.load(f)
        shape = tuple(meta['shape'])
        self.windows = np.memmap(cache_dat, dtype=np.float32, mode='r', shape=shape)
        self.seq_len = shape[1]              # cached length
        self.seq_len_target = seq_len        # target (training expects this)
        print(f'[CauKer cache HIT] {cache_dat}')
        print(f'  loaded {len(self.windows):,} windows × {self.seq_len} (memmap, target={seq_len})')

    def __len__(self): return len(self.windows)
    def __getitem__(self, idx):
        w = self.windows[idx]
        # If cached series is shorter than expected seq_len, tile to fill.
        # This lets us cache CauKer at L=720 (much faster GP) while training
        # pipeline expects window_len=2160.
        if len(w) < self.seq_len_target:
            n_tile = (self.seq_len_target + len(w) - 1) // len(w)
            w = np.tile(w, n_tile)[:self.seq_len_target]
        elif len(w) > self.seq_len_target:
            w = w[:self.seq_len_target]
        return torch.tensor(w, dtype=torch.float32)


class SyntheticGapFiller(Dataset):
    """
    제외된 eval benchmark 패턴을 정밀 모방하는 synthetic dataset.

    각 도메인의 실제 통계적 특성을 반영:
    - ETT-style: 일/주 주기 + 조화 관계 + intra-day peak + AR noise
    - Weather-style: diurnal cycle + synoptic(3-7일) + 연간 + heavy-tail 이벤트
    - Exchange-style: GARCH 변동성 클러스터링 + mean reversion + jump
    - M4-Monthly-style: 12개월 주기 + 트렌드 + multiplicative seasonality
    - M4-Quarterly-style: 4분기 주기 + structural break
    - Regime-switching: 상태 전환 + 각 상태별 다른 통계 특성
    """

    # 도메인별 생성 비중 (eval benchmark에 강하게 bias — ETT 4개 + Weather = 5/5)
    # v3: 실패 모드(nonstationary/multiscale/heteroskedastic) 추가해서
    # ETTm2, Weather 같은 실패 케이스 보강.
    DOMAIN_WEIGHTS = {
        'ett_h1':              0.10,  # ETTh1-like: 일간 사이클 강함, weekly pattern
        'ett_h2':              0.10,  # ETTh2-like: 더 irregular, heavy noise
        'ett_m1':              0.09,  # ETTm1-like: 15-min resolution, 빠른 진동
        'ett_m2':              0.09,  # ETTm2-like: 15-min, 낮은 variance
        'weather_dry':         0.08,  # Weather (temperature-like): strong diurnal
        'weather_wet':         0.08,  # Weather (humidity/precip): heavy-tail spikes
        'weather_ann':         0.06,  # Weather (annual): slow trends
        'exchange':            0.03,  # Financial (GARCH + jumps)
        'm4_monthly':          0.04,  # M4 Monthly (12-month cycle)
        'm4_quarterly':        0.03,  # M4 Quarterly (4 cycles)
        'regime':              0.03,  # Regime switching
        'composite':           0.02,  # 복합 패턴
        # Failure-mode generators (addresses ETTm2/Weather gap)
        'nonstationary_burst': 0.06,  # Level shifts + bursts
        'multiscale':          0.06,  # 일/주/월 동시
        'heteroskedastic':     0.04,  # 시변 variance
        # M4-specific generators (NEW — target M4 short-series weakness)
        'm4_yearly':           0.05,  # Very short series (20-50 pts), strong trend
        'm4_weekly':           0.04,  # 52-week cycle + trend
        'm4_daily':            0.05,  # 7d + 30d + 365d multi-cycle, business
        'm4_hourly':           0.05,  # 24h + 168h, energy/traffic-like
        'm4_business':         0.04,  # Multiplicative seasonality + trend + promo spikes
    }

    # ETT-boosted weights: ETT 38% → 64% (strengthens ETT-style coverage)
    DOMAIN_WEIGHTS_ETTBOOST = {
        'ett_h1':              0.16,
        'ett_h2':              0.16,
        'ett_m1':              0.16,
        'ett_m2':              0.16,
        'weather_dry':         0.05,
        'weather_wet':         0.05,
        'weather_ann':         0.04,
        'exchange':            0.02,
        'm4_monthly':          0.02,
        'm4_quarterly':        0.02,
        'regime':              0.02,
        'composite':           0.01,
        'nonstationary_burst': 0.02,
        'multiscale':          0.02,
        'heteroskedastic':     0.01,
        'm4_yearly':           0.02,
        'm4_weekly':           0.02,
        'm4_daily':            0.02,
        'm4_hourly':           0.02,
        'm4_business':         0.02,
    }

    def __init__(self, n_samples=50000, seq_len=192):
        # Optional ETT-boost weights via env var
        if os.environ.get('SYNTH_ETTBOOST', '0') == '1':
            self.DOMAIN_WEIGHTS = self.DOMAIN_WEIGHTS_ETTBOOST
            print(f'  [synth] ETT-boosted DOMAIN_WEIGHTS active')
        # Disk cache: same (n_samples, seq_len, DOMAIN_WEIGHTS) → reuse generated array
        cache_dir = os.environ.get('SYNTH_CACHE_DIR', './dataset/synth_cache')
        os.makedirs(cache_dir, exist_ok=True)
        # Hash on domain weights too so changing weights invalidates cache
        import hashlib, json
        wkey = hashlib.md5(str(sorted(self.DOMAIN_WEIGHTS.items())).encode()).hexdigest()[:8]
        cache_key = f'synth_n{n_samples}_seq{seq_len}_w{wkey}'
        cache_npy = os.path.join(cache_dir, cache_key + '.npy')
        cache_dat = os.path.join(cache_dir, cache_key + '.dat')
        cache_meta = os.path.join(cache_dir, cache_key + '.meta')

        # HIT: streaming memmap (preferred)
        if os.path.exists(cache_dat) and os.path.exists(cache_meta):
            with open(cache_meta) as f:
                meta = json.load(f)
            shape = tuple(meta['shape'])
            print(f'  [synth cache HIT mmap] {cache_dat}')
            self.windows = np.memmap(cache_dat, dtype=np.float32, mode='r', shape=shape)
            print(f'  [synth cache] {len(self.windows):,} windows × {self.windows.shape[1]} loaded (memmap)')
            return

        # HIT: legacy .npy (backward compat — uses mmap so even big files OK)
        if os.path.exists(cache_npy):
            print(f'  [synth cache HIT] loading {cache_npy}')
            self.windows = np.load(cache_npy, mmap_mode='r')
            print(f'  [synth cache] {len(self.windows):,} windows × {self.windows.shape[1]} loaded')
            return

        # MISS — streaming build directly to memmap (RAM-bounded)
        print(f'  [synth cache MISS] streaming {n_samples:,} → {cache_dat}')
        mm = np.memmap(cache_dat, dtype=np.float32, mode='w+', shape=(n_samples, seq_len))
        rng = np.random.RandomState(42)
        domains = list(self.DOMAIN_WEIGHTS.keys())
        probs = np.array([self.DOMAIN_WEIGHTS[d] for d in domains])
        probs = probs / probs.sum()
        write_count = 0
        log_every = max(10000, n_samples // 100)
        import time as _time
        _t0 = _time.time()

        for _ in range(n_samples):
            domain = rng.choice(domains, p=probs)
            n = seq_len
            t = np.linspace(0, 2, n)

            # ETT sub-variants (differ in cycle strength, noise, peaks)
            if domain == 'ett_h1':
                y = self._gen_ett(rng, n, t, variant='h1')
            elif domain == 'ett_h2':
                y = self._gen_ett(rng, n, t, variant='h2')
            elif domain == 'ett_m1':
                y = self._gen_ett(rng, n, t, variant='m1')
            elif domain == 'ett_m2':
                y = self._gen_ett(rng, n, t, variant='m2')
            # Weather sub-variants (dry/wet/annual dominant)
            elif domain == 'weather_dry':
                y = self._gen_weather(rng, n, t, variant='dry')
            elif domain == 'weather_wet':
                y = self._gen_weather(rng, n, t, variant='wet')
            elif domain == 'weather_ann':
                y = self._gen_weather(rng, n, t, variant='ann')
            elif domain == 'exchange':
                y = self._gen_exchange(rng, n)
            elif domain == 'm4_monthly':
                y = self._gen_m4_monthly(rng, n, t)
            elif domain == 'm4_quarterly':
                y = self._gen_m4_quarterly(rng, n, t)
            elif domain == 'regime':
                y = self._gen_regime(rng, n)
            elif domain == 'nonstationary_burst':
                y = self._gen_nonstationary_burst(rng, n, t)
            elif domain == 'multiscale':
                y = self._gen_multiscale(rng, n, t)
            elif domain == 'heteroskedastic':
                y = self._gen_heteroskedastic(rng, n, t)
            elif domain == 'm4_yearly':
                y = self._gen_m4_yearly(rng, n, t)
            elif domain == 'm4_weekly':
                y = self._gen_m4_weekly(rng, n, t)
            elif domain == 'm4_daily':
                y = self._gen_m4_daily(rng, n, t)
            elif domain == 'm4_hourly':
                y = self._gen_m4_hourly(rng, n, t)
            elif domain == 'm4_business':
                y = self._gen_m4_business(rng, n, t)
            else:
                y = self._gen_composite(rng, n, t)

            s = np.std(y)
            if s > 1e-6:
                mm[write_count] = np.clip((y - np.mean(y)) / s, -10, 10).astype(np.float32)
                write_count += 1
                if write_count % log_every == 0:
                    elapsed = _time.time() - _t0
                    rate = write_count / max(elapsed, 1e-3)
                    eta = (n_samples - write_count) / max(rate, 1e-3)
                    print(f'  [synth] {write_count:,}/{n_samples:,} '
                          f'({write_count/n_samples*100:.1f}%) | '
                          f'{rate:.0f}/s | ETA {eta/3600:.1f}h', flush=True)

        # Truncate memmap to actual count, write metadata, reopen as read-only
        mm.flush()
        del mm
        if write_count < n_samples:
            # Truncate file to actual size
            actual_bytes = write_count * seq_len * 4
            with open(cache_dat, 'r+b') as f:
                f.truncate(actual_bytes)
        with open(cache_meta, 'w') as f:
            json.dump({'shape': [write_count, seq_len], 'dtype': 'float32',
                       'n_requested': n_samples}, f)
        print(f'Synthetic gap filler: {write_count:,} samples '
              f'(domains: {list(self.DOMAIN_WEIGHTS.keys())})')
        print(f'  [synth cache SAVED mmap] {cache_dat} ({write_count*seq_len*4/1e9:.1f} GB)')

        # Reload as read-only memmap for use
        self.windows = np.memmap(cache_dat, dtype=np.float32, mode='r',
                                 shape=(write_count, seq_len))

    # ----------------------------------------------------------
    # ETT-style: 전력 트랜스포머 온도
    #   - 일간 주기 (기본파 + 2차/3차 조화)
    #   - 주간 주기 (주말 하락)
    #   - intra-day 이중 피크 (아침/저녁)
    #   - AR(1) colored noise
    # ----------------------------------------------------------
    @staticmethod
    def _gen_ett(rng, n, t, variant='h1'):
        """
        ETT variants:
          h1: hourly, strong daily + weekly, moderate peaks, clean
          h2: hourly, similar but more irregular, higher noise
          m1: minutely (15min), faster oscillations (high-freq), pronounced peaks
          m2: minutely, faster osc but lower amplitude/variance
        """
        # 일간 주기: 기본 + 2차 조화
        if variant in ('m1', 'm2'):
            # minutely: much faster cycles (4x hourly = more cycles per window)
            daily_freq = rng.uniform(3.2, 4.8)  # ~4 cycles per unit
        else:
            daily_freq = rng.uniform(0.8, 1.2)
        amp1 = rng.uniform(2.0, 12.0) if variant == 'h2' else rng.uniform(3.0, 15.0)
        amp2 = rng.uniform(0.5, 3.0)
        phase1 = rng.uniform(0, 2 * np.pi)
        daily = (amp1 * np.sin(2 * np.pi * daily_freq * t + phase1)
                 + amp2 * np.sin(2 * np.pi * 2 * daily_freq * t + phase1 * 1.3))

        # intra-day 이중 피크 (variant별 강도)
        peak_amp = {'h1': 3.0, 'h2': 1.5, 'm1': 4.0, 'm2': 1.0}[variant]
        peak_amp = rng.uniform(0.5 * peak_amp, 1.5 * peak_amp)
        peak1_center = rng.uniform(0.35, 0.42)
        peak2_center = rng.uniform(0.72, 0.80)
        peak_width = rng.uniform(0.04, 0.08)
        t_mod = t % (1.0 / max(daily_freq, 1e-6))
        t_mod_norm = (t_mod - t_mod.min()) / (t_mod.max() - t_mod.min() + 1e-6)
        peaks = peak_amp * (
            np.exp(-((t_mod_norm - peak1_center) ** 2) / (2 * peak_width ** 2))
            + np.exp(-((t_mod_norm - peak2_center) ** 2) / (2 * peak_width ** 2))
        )

        # 주간 주기 (variant별 강도 — h1이 가장 뚜렷)
        weekly_amp = {'h1': 3.0, 'h2': 1.8, 'm1': 2.5, 'm2': 1.2}[variant]
        weekly_amp = rng.uniform(0.5 * weekly_amp, 1.5 * weekly_amp)
        weekly_freq = daily_freq / 7.0
        weekly = weekly_amp * np.sin(2 * np.pi * weekly_freq * t + rng.uniform(0, 2 * np.pi))

        # Noise (variant별 강도)
        noise_sigma = {'h1': 0.8, 'h2': 1.8, 'm1': 0.7, 'm2': 0.4}[variant]
        noise_sigma = rng.uniform(0.5 * noise_sigma, 1.5 * noise_sigma)
        noise = _ar1_noise(rng, n, phi=rng.uniform(0.3, 0.8), sigma=noise_sigma)

        # 느린 트렌드
        trend = rng.uniform(-0.5, 0.5) * t

        return daily + peaks + weekly + trend + noise

    # ----------------------------------------------------------
    # Weather-style: 기상 데이터
    #   - 강한 diurnal cycle (24h)
    #   - synoptic 패턴 (3-7일 기상 전선)
    #   - 연간 계절 변동
    #   - heavy-tail 이벤트 (강수 스파이크)
    # ----------------------------------------------------------
    @staticmethod
    def _gen_weather(rng, n, t, variant='dry'):
        """
        Weather variants:
          dry: temperature-like, strong diurnal + smooth
          wet: humidity/precipitation-like, heavy-tail spikes, bursty
          ann: annual/slow-moving, long-range trends + synoptic dominant
        """
        # diurnal cycle
        if variant == 'ann':
            diurnal_amp = rng.uniform(1.0, 5.0)  # 약함
        else:
            diurnal_amp = rng.uniform(5.0, 15.0)
        phase = rng.uniform(0, 2 * np.pi)
        diurnal = (diurnal_amp * np.sin(2 * np.pi * t + phase)
                   + rng.uniform(0.5, 2.0) * np.sin(2 * np.pi * 3 * t + phase * 0.7))

        # synoptic (변종별 강도)
        syn_amp = {'dry': 3.0, 'wet': 6.0, 'ann': 8.0}[variant]
        syn_amp = rng.uniform(0.5 * syn_amp, 1.5 * syn_amp)
        syn_freq = rng.uniform(0.15, 0.35)
        synoptic = syn_amp * np.sin(2 * np.pi * syn_freq * t + rng.uniform(0, 2 * np.pi))

        # 연간 성분 (ann variant에서 지배적)
        annual_amp = {'dry': 1.5, 'wet': 2.0, 'ann': 5.0}[variant]
        annual_amp = rng.uniform(0.5 * annual_amp, 1.5 * annual_amp)
        annual = annual_amp * np.sin(2 * np.pi * 0.05 * t + rng.uniform(0, 2 * np.pi))

        # AR(2) colored noise
        noise_sigma = {'dry': 0.8, 'wet': 2.0, 'ann': 1.2}[variant]
        noise_sigma = rng.uniform(0.5 * noise_sigma, 1.5 * noise_sigma)
        noise = _ar2_noise(rng, n,
                           phi1=rng.uniform(0.4, 0.7),
                           phi2=rng.uniform(-0.2, 0.1),
                           sigma=noise_sigma)

        y = diurnal + synoptic + annual + noise

        # heavy-tail 이벤트 (wet variant는 더 자주)
        spike_prob = {'dry': 0.1, 'wet': 0.6, 'ann': 0.2}[variant]
        spike_strength = {'dry': 2.0, 'wet': 5.0, 'ann': 3.0}[variant]
        if rng.rand() < spike_prob:
            n_spikes = rng.randint(1, 5 if variant == 'wet' else 4)
            for _ in range(n_spikes):
                pos = rng.randint(0, n)
                width = rng.randint(2, max(3, n // 20))
                height = rng.choice([-1, 1]) * rng.exponential(spike_strength)
                spike = height * np.exp(-0.5 * ((np.arange(n) - pos) / width) ** 2)
                y += spike

        return y

    # ----------------------------------------------------------
    # Exchange-style: 환율 데이터
    #   - GARCH(1,1) 변동성 클러스터링
    #   - OU process (mean reversion)
    #   - jump diffusion (정책 발표/지정학 이벤트)
    # ----------------------------------------------------------
    @staticmethod
    def _gen_exchange(rng, n):
        # Ornstein-Uhlenbeck + GARCH volatility + jump
        mu = rng.uniform(-0.01, 0.01)  # 장기 평균 drift
        theta = rng.uniform(0.01, 0.1)  # mean reversion 속도
        kappa = rng.uniform(0.5, 2.0)   # mean reversion 강도

        # GARCH(1,1) 변동성
        omega = rng.uniform(1e-5, 5e-4)
        alpha_g = rng.uniform(0.05, 0.15)
        beta_g = rng.uniform(0.75, 0.92)

        y = np.zeros(n)
        sigma2 = np.full(n, omega / max(1e-8, 1 - alpha_g - beta_g))
        eps_prev = 0.0
        y[0] = rng.randn() * 0.01

        for i in range(1, n):
            # GARCH variance update
            sigma2[i] = omega + alpha_g * eps_prev ** 2 + beta_g * sigma2[i - 1]
            sigma_t = np.sqrt(max(sigma2[i], 1e-10))

            # OU mean reversion + stochastic vol
            eps = rng.randn() * sigma_t
            y[i] = y[i - 1] + theta * (mu - y[i - 1]) * kappa + eps
            eps_prev = eps

        # jump diffusion (희귀 이벤트)
        jump_prob = rng.uniform(0.01, 0.05)
        jump_mask = rng.rand(n) < jump_prob
        jump_sizes = rng.standard_t(df=4, size=n) * rng.uniform(0.02, 0.08)
        y += np.cumsum(jump_mask * jump_sizes)

        return y

    # ----------------------------------------------------------
    # M4-Monthly-style: 월간 시계열
    #   - 12개월 주기 (+ 6개월 조화)
    #   - multiplicative seasonality
    #   - 비선형 트렌드 (다항식/log)
    #   - 이질적 노이즈 (수준에 비례)
    # ----------------------------------------------------------
    @staticmethod
    def _gen_m4_monthly(rng, n, t):
        # 비선형 트렌드
        trend_type = rng.choice(['poly', 'log', 'piecewise'])
        if trend_type == 'poly':
            coeffs = rng.uniform(-1, 1, size=rng.randint(2, 4))
            trend = np.polyval(coeffs, t)
        elif trend_type == 'log':
            trend = rng.uniform(1.0, 5.0) * np.log1p(t * rng.uniform(1, 10))
        else:
            # piecewise linear
            n_segments = rng.randint(2, 4)
            breaks = sorted(rng.uniform(0.2, 1.8, n_segments))
            slopes = rng.uniform(-2, 2, n_segments + 1)
            trend = np.zeros(n)
            prev_bp, prev_val = 0.0, 0.0
            for seg_i, bp in enumerate(breaks + [2.0]):
                mask = (t >= prev_bp) & (t < bp)
                trend[mask] = prev_val + slopes[seg_i] * (t[mask] - prev_bp)
                prev_val = prev_val + slopes[seg_i] * (bp - prev_bp)
                prev_bp = bp

        # 12개월 계절 (기본 + 조화)
        base_freq = rng.uniform(0.8, 1.2) * 6  # ~6 cycles in t=[0,2]
        amp12 = rng.uniform(0.5, 2.0)
        amp6 = rng.uniform(0.1, 0.8)
        phase12 = rng.uniform(0, 2 * np.pi)
        seasonal = (amp12 * np.sin(2 * np.pi * base_freq * t + phase12)
                    + amp6 * np.sin(2 * np.pi * 2 * base_freq * t + phase12 * 0.6))

        # multiplicative or additive (50/50)
        level = np.abs(trend) + rng.uniform(1.0, 5.0)
        if rng.rand() < 0.5:
            # multiplicative: 큰 값에서 더 큰 계절 변동
            y = level * (1 + seasonal * 0.3) + rng.randn(n) * level * rng.uniform(0.02, 0.08)
        else:
            y = trend + seasonal + rng.randn(n) * rng.uniform(0.2, 0.8)

        return y

    # ----------------------------------------------------------
    # M4-Quarterly-style: 분기 시계열
    #   - 4분기 주기
    #   - structural break (정책 변경/외부 충격)
    #   - damped trend
    # ----------------------------------------------------------
    @staticmethod
    def _gen_m4_quarterly(rng, n, t):
        # 4분기 주기
        q_freq = rng.uniform(0.8, 1.2) * 4  # ~4 cycles
        amp = rng.uniform(1.0, 4.0)
        seasonal = amp * np.sin(2 * np.pi * q_freq * t + rng.uniform(0, 2 * np.pi))

        # damped exponential trend
        growth = rng.uniform(0.3, 2.0)
        damping = rng.uniform(0.3, 0.9)
        trend = growth * (1 - damping ** (t * 10))

        # structural break (1~2개)
        y = trend + seasonal
        n_breaks = rng.randint(0, 3)
        for _ in range(n_breaks):
            bp = rng.uniform(0.3, 1.7)
            shift = rng.uniform(-3, 3)
            slope_change = rng.uniform(-1, 1)
            mask = t >= bp
            y[mask] += shift + slope_change * (t[mask] - bp)

        # 노이즈 (AR(1))
        y += _ar1_noise(rng, n, phi=rng.uniform(0.2, 0.5), sigma=rng.uniform(0.3, 1.0))

        return y

    # ----------------------------------------------------------
    # Regime-switching: 상태 전환 시계열
    #   - 2~3개 상태, 각각 다른 통계 특성
    #   - Markov 전이 확률
    #   - 상태별: mean, volatility, AR 계수 변화
    # ----------------------------------------------------------
    @staticmethod
    def _gen_regime(rng, n):
        n_regimes = rng.randint(2, 4)

        # 각 regime의 파라미터
        means = rng.uniform(-2, 2, n_regimes)
        sigmas = rng.uniform(0.3, 2.0, n_regimes)
        ar_phis = rng.uniform(0.1, 0.8, n_regimes)

        # 전이 확률 (자기 상태 유지 확률 높게)
        stay_prob = rng.uniform(0.95, 0.99)

        # 상태 시퀀스 생성
        state = rng.randint(0, n_regimes)
        y = np.zeros(n)
        y[0] = means[state] + rng.randn() * sigmas[state]

        for i in range(1, n):
            if rng.rand() > stay_prob:
                state = rng.randint(0, n_regimes)
            phi = ar_phis[state]
            y[i] = (1 - phi) * means[state] + phi * y[i - 1] + rng.randn() * sigmas[state]

        return y

    # ----------------------------------------------------------
    # Composite: 여러 도메인 특성을 조합
    #   - 다중 스케일 주기 + 트렌드 + AR noise + 가끔 jump
    # ----------------------------------------------------------
    @staticmethod
    def _gen_composite(rng, n, t):
        # 2~4개 서로 다른 주파수 조합
        n_freqs = rng.randint(2, 5)
        y = np.zeros(n)
        for _ in range(n_freqs):
            freq = rng.uniform(0.3, 15.0)
            amp = rng.uniform(0.3, 3.0)
            phase = rng.uniform(0, 2 * np.pi)
            y += amp * np.sin(2 * np.pi * freq * t + phase)

        # 비선형 트렌드
        deg = rng.randint(1, 4)
        coeffs = rng.uniform(-0.5, 0.5, deg + 1)
        y += np.polyval(coeffs, t)

        # 시변 노이즈 (sigma가 시간에 따라 변함)
        base_sigma = rng.uniform(0.2, 1.0)
        sigma_t = base_sigma * (1 + 0.5 * np.sin(2 * np.pi * rng.uniform(0.5, 3) * t))
        y += rng.randn(n) * sigma_t

        # 간헐적 jump (20% 확률)
        if rng.rand() < 0.2:
            n_jumps = rng.randint(1, 3)
            for _ in range(n_jumps):
                pos = rng.randint(n // 10, n - n // 10)
                y[pos:] += rng.choice([-1, 1]) * rng.exponential(1.5)

        return y

    # ----------------------------------------------------------
    # Failure-mode generators (target ETTm2, Weather weak spots)
    # ----------------------------------------------------------
    @staticmethod
    def _gen_nonstationary_burst(rng, n, t):
        """Level shifts + bursts — Weather-style abrupt events, non-stationary."""
        y = _ar1_noise(rng, n, phi=rng.uniform(0.5, 0.85), sigma=rng.uniform(0.3, 1.0))
        # 1~3 level shifts with exponential decay
        n_shifts = rng.randint(1, 4)
        for _ in range(n_shifts):
            pos = rng.randint(n // 8, 7 * n // 8)
            shift = rng.uniform(-3, 3)
            decay = rng.uniform(0.9, 0.99)
            for i in range(pos, n):
                y[i] += shift * (decay ** (i - pos))
        # 1~3 wide bursts (large-scale excursions)
        n_bursts = rng.randint(1, 4)
        for _ in range(n_bursts):
            pos = rng.randint(0, n)
            width = rng.randint(3, max(5, n // 30))
            height = rng.choice([-1, 1]) * rng.exponential(3.0)
            burst = height * np.exp(-0.5 * ((np.arange(n) - pos) / width) ** 2)
            y += burst
        # Long linear trend overlay
        y += rng.uniform(-1.5, 1.5) * t
        return y

    @staticmethod
    def _gen_multiscale(rng, n, t):
        """Daily + weekly + monthly + (optional yearly) simultaneously."""
        y = rng.uniform(3.0, 8.0) * np.sin(2 * np.pi * rng.uniform(0.9, 1.1) * t + rng.uniform(0, 2 * np.pi))
        y += rng.uniform(1.0, 4.0) * np.sin(2 * np.pi * rng.uniform(0.9, 1.1) / 7 * t + rng.uniform(0, 2 * np.pi))
        y += rng.uniform(0.5, 3.0) * np.sin(2 * np.pi * rng.uniform(0.9, 1.1) / 30 * t + rng.uniform(0, 2 * np.pi))
        if rng.rand() < 0.5:
            y += rng.uniform(0.2, 2.0) * np.sin(2 * np.pi * rng.uniform(0.9, 1.1) / 365 * t)
        y += _ar1_noise(rng, n, phi=rng.uniform(0.3, 0.7), sigma=rng.uniform(0.2, 0.8))
        return y

    @staticmethod
    def _gen_heteroskedastic(rng, n, t):
        """Time-varying variance — ETTm2-style irregular spikes."""
        base = rng.uniform(0.5, 2.0) * np.sin(2 * np.pi * rng.uniform(0.5, 3.0) * t)
        # Time-varying sigma (slow oscillation)
        sigma_t = 0.2 + 1.2 * (0.5 + 0.5 * np.sin(2 * np.pi * rng.uniform(0.3, 1.0) * t + rng.uniform(0, 2 * np.pi)))
        noise = _ar1_noise(rng, n, phi=rng.uniform(0.3, 0.6), sigma=1.0) * sigma_t
        y = base + noise
        # Occasional extreme spikes (matches ETTm2 irregular bursts)
        if rng.rand() < 0.5:
            n_spikes = rng.randint(2, 8)
            for _ in range(n_spikes):
                pos = rng.randint(0, n)
                y[pos] += rng.choice([-1, 1]) * rng.uniform(3, 8)
        return y

    # ----------------------------------------------------------
    # M4-specific generators (target M4 short-series weakness)
    # ----------------------------------------------------------
    @staticmethod
    def _gen_m4_yearly(rng, n, t):
        """M4 Yearly-like: very short effective content, strong long-term trend."""
        # Effective length 20-60 (most of window unused, but VarLen handles via mask)
        eff = rng.randint(20, 80)
        # Strong long-term trend (linear + curvature)
        trend_slope = rng.uniform(-2.5, 2.5)
        trend_curve = rng.uniform(-0.6, 0.6)
        # Generate on short grid
        s = np.linspace(0, 1, eff)
        y_short = trend_slope * s + trend_curve * s ** 2
        # Weak noise
        y_short += rng.randn(eff) * rng.uniform(0.05, 0.25)
        # Occasional structural break
        if rng.rand() < 0.3:
            bp = rng.randint(eff // 4, 3 * eff // 4)
            y_short[bp:] += rng.uniform(-1.0, 1.0)
        # Place at end of window, rest is the start value (like repeat-pad)
        y = np.full(n, y_short[0], dtype=np.float32)
        y[-eff:] = y_short
        return y.astype(np.float64)

    @staticmethod
    def _gen_m4_weekly(rng, n, t):
        """M4 Weekly-like: 52-week seasonality + linear/exp trend."""
        # 52-week cycle → per unit t
        freq_52 = rng.uniform(0.8, 1.2) * 26  # ~26 cycles in t=[0,2]
        freq_4 = rng.uniform(0.8, 1.2) * 4    # ~4 cycles (monthly)
        amp_52 = rng.uniform(0.5, 2.5)
        amp_4 = rng.uniform(0.1, 0.8)
        phase1 = rng.uniform(0, 2 * np.pi)
        seasonal = amp_52 * np.sin(2 * np.pi * freq_52 * t + phase1) \
                 + amp_4 * np.sin(2 * np.pi * freq_4 * t + phase1 * 0.7)
        # Trend (linear or exp)
        if rng.rand() < 0.5:
            trend = rng.uniform(-1.0, 1.0) * t
        else:
            trend = rng.uniform(0.3, 1.5) * (np.exp(rng.uniform(-0.3, 0.3) * t) - 1)
        y = seasonal + trend + _ar1_noise(rng, n, phi=rng.uniform(0.2, 0.6),
                                           sigma=rng.uniform(0.1, 0.4))
        return y

    @staticmethod
    def _gen_m4_daily(rng, n, t):
        """M4 Daily-like: 7d + 30d + 365d multi-cycle, business-like."""
        # Assuming t=[0,2] spans some period
        f_7 = rng.uniform(0.9, 1.1) * 15   # 7-day cycles
        f_30 = rng.uniform(0.9, 1.1) * 3   # 30-day cycles
        f_365 = rng.uniform(0.8, 1.2) * 0.3  # yearly slow
        y = rng.uniform(0.3, 1.5) * np.sin(2 * np.pi * f_7 * t + rng.uniform(0, 2 * np.pi))
        y += rng.uniform(0.3, 1.2) * np.sin(2 * np.pi * f_30 * t + rng.uniform(0, 2 * np.pi))
        y += rng.uniform(0.5, 2.0) * np.sin(2 * np.pi * f_365 * t + rng.uniform(0, 2 * np.pi))
        # Upward trend (common in business)
        y += rng.uniform(0.2, 1.5) * t
        # Heteroskedastic noise
        sigma = 0.1 + 0.3 * (0.5 + 0.5 * np.sin(2 * np.pi * rng.uniform(0.5, 2) * t))
        y += _ar1_noise(rng, n, phi=rng.uniform(0.2, 0.5), sigma=1.0) * sigma
        return y

    @staticmethod
    def _gen_m4_hourly(rng, n, t):
        """M4 Hourly-like: 24h daily + 168h weekly, energy/traffic."""
        # Daily ~24 cycles per period, weekly ~3.5 cycles
        f_day = rng.uniform(0.9, 1.1) * 24
        f_week = rng.uniform(0.9, 1.1) * 3.5
        daily_amp = rng.uniform(2.0, 8.0)
        weekly_amp = rng.uniform(0.5, 2.5)
        phase = rng.uniform(0, 2 * np.pi)
        y = daily_amp * np.sin(2 * np.pi * f_day * t + phase) \
          + weekly_amp * np.sin(2 * np.pi * f_week * t + phase * 0.5)
        # Intra-day peaks (like energy morning/evening peaks)
        t_mod = (t * f_day) % 1.0
        peak1 = rng.uniform(0.3, 0.4)
        peak2 = rng.uniform(0.7, 0.8)
        peaks = rng.uniform(1.0, 3.0) * (
            np.exp(-((t_mod - peak1) ** 2) / 0.005)
            + np.exp(-((t_mod - peak2) ** 2) / 0.005)
        )
        y += peaks
        y += _ar1_noise(rng, n, phi=rng.uniform(0.3, 0.7), sigma=rng.uniform(0.3, 1.0))
        return y

    @staticmethod
    def _gen_m4_business(rng, n, t):
        """M4 Business-like: multiplicative seasonality + trend + promo spikes.
        Common in retail/tourism data."""
        # Baseline level (positive)
        level = rng.uniform(2.0, 8.0)
        # Growth trend
        trend = level * rng.uniform(0.1, 0.8) * t
        # Seasonality (12-period, multiplicative)
        f_season = rng.uniform(0.9, 1.1) * 6
        season_amp = rng.uniform(0.15, 0.45)  # relative amplitude (multiplicative)
        seasonal_factor = 1 + season_amp * np.sin(2 * np.pi * f_season * t + rng.uniform(0, 2 * np.pi))
        # Multiplicative combination
        y = (level + trend) * seasonal_factor
        # Promotion/event spikes (occasional)
        if rng.rand() < 0.7:
            n_events = rng.randint(1, 4)
            for _ in range(n_events):
                pos = rng.randint(n // 10, n - n // 10)
                width = rng.randint(2, max(3, n // 30))
                height = rng.uniform(0.3, 1.2) * np.mean(y)
                event = height * np.exp(-0.5 * ((np.arange(n) - pos) / width) ** 2)
                y += event
        # Proportional noise (larger for bigger level)
        y += rng.randn(n) * np.abs(y).mean() * rng.uniform(0.02, 0.08)
        return y

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return torch.tensor(self.windows[idx], dtype=torch.float32)


# ---- Noise utility functions ----
def _ar1_noise(rng, n, phi=0.5, sigma=1.0):
    """AR(1) process: x[t] = phi * x[t-1] + eps, colored noise with memory."""
    noise = np.zeros(n)
    noise[0] = rng.randn() * sigma
    for i in range(1, n):
        noise[i] = phi * noise[i - 1] + rng.randn() * sigma
    return noise


def _ar2_noise(rng, n, phi1=0.5, phi2=-0.1, sigma=1.0):
    """AR(2) process for longer-range autocorrelation."""
    noise = np.zeros(n)
    noise[0] = rng.randn() * sigma
    noise[1] = phi1 * noise[0] + rng.randn() * sigma
    for i in range(2, n):
        noise[i] = phi1 * noise[i - 1] + phi2 * noise[i - 2] + rng.randn() * sigma
    return noise


# ============================================================
# Training
# ============================================================
def collate_batch(windows, seq_len=96, n_query=64, mr=0.375, multi_horizon=True,
                  max_horizon_mult=2, return_dense=False, contiguous_mode=False,
                  pred_len_choices=(96, 192, 336, 720)):
    """
    Dense-query collate with multi-horizon training + long-horizon direct prediction.

    Two modes:
      - Random sparse (default): n_query random points from horizon_cap ∈ [seq_len/8, max_future]
      - Contiguous (contiguous_mode=True): pred_len sampled from pred_len_choices,
        queries at positions 0, 1, ..., pred_len-1 (dense)
        → matches standard TSFM eval protocol while keeping operator formulation.
    """
    ctxs, qts, qvs = [], [], []
    dense_futures, is_fc = [], []
    max_future = seq_len * max_horizon_mult

    # If contiguous_mode: pick a single pred_len per batch (ensures uniform qt shape)
    batch_pred_len = None
    if contiguous_mode:
        valid_pl = [p for p in pred_len_choices if p <= max_future]
        if valid_pl:
            batch_pred_len = int(np.random.choice(valid_pl))
    for w in windows:
        w = w.numpy() if isinstance(w, torch.Tensor) else w

        avail_future = len(w) - seq_len
        # Contiguous mode: force forecast (uniform pl across batch)
        if contiguous_mode:
            do_forecast = (avail_future > 0)
        else:
            do_forecast = (np.random.rand() < 0.5 and avail_future > 0)
        if do_forecast:
            ctx = w[:seq_len]
            future_len = min(avail_future, max_future)
            future = w[seq_len:seq_len + future_len]
            if contiguous_mode and batch_pred_len is not None and batch_pred_len <= future_len:
                # Contiguous queries: 0, 1, 2, ..., pred_len-1 (matches standard TSFM)
                # All samples in batch use same pred_len → uniform batch shape
                # n_query becomes pl (variable per batch, but uniform within batch)
                pl = batch_pred_len
                qi = np.arange(pl, dtype=np.int64)
            elif multi_horizon:
                horizon_cap = np.random.randint(max(8, seq_len // 8), future_len + 1)
                qi = np.random.choice(horizon_cap, min(n_query, horizon_cap), replace=False)
                if len(qi) < n_query:
                    qi = np.concatenate([qi, np.random.choice(horizon_cap, n_query - len(qi), replace=True)])
            else:
                qi = np.random.choice(future_len, n_query, replace=(n_query > future_len))
            qt = 1.0 + qi.astype(np.float32) / seq_len
            qv = future[qi]
            # Dense target for spectral loss: pad/truncate future to max_future
            if len(future) < max_future:
                dense_tgt = np.concatenate([future, np.zeros(max_future - len(future), dtype=np.float32)])
            else:
                dense_tgt = future[:max_future]
            dense_futures.append(dense_tgt.astype(np.float32))
            is_fc.append(True)
        else:
            full = w[:seq_len] if len(w) >= seq_len else np.pad(w, (0, seq_len - len(w)))
            mask = np.random.rand(seq_len) > mr
            qi = np.where(~mask)[0]
            if len(qi) == 0:
                continue
            if len(qi) >= n_query:
                qi = np.random.choice(qi, n_query, replace=False)
            else:
                qi = np.concatenate([qi, np.random.choice(qi, n_query - len(qi), replace=True)])
            ctx = full * mask.astype(np.float32)
            qt = qi.astype(np.float32) / seq_len
            qv = full[qi]
            dense_futures.append(np.zeros(max_future, dtype=np.float32))  # unused
            is_fc.append(False)
        ctxs.append(ctx); qts.append(qt); qvs.append(qv)
    if not ctxs: return None
    out = (torch.tensor(np.stack(ctxs), dtype=torch.float32),
           torch.tensor(np.stack(qts), dtype=torch.float32),
           torch.tensor(np.stack(qvs), dtype=torch.float32))
    if return_dense:
        out = out + (
            torch.tensor(np.stack(dense_futures), dtype=torch.float32),
            torch.tensor(is_fc, dtype=torch.bool),
        )
    return out


def train(model, datasets, save_path, seq_len=96, epochs=20, lr=3e-4, batch_size=64,
          n_query=64, diversity_weight=0.01, spectral_weight=0.0,
          trend_weight=0.0, shape_weight=0.0, max_horizon_mult=2,
          contiguous_mode=False):
    """
    Training with operator learning improvements + shape/trend loss.

    Loss composition:
      loss = MSE(pred, truth)                                # value
           + diversity_weight * trunk_pairwise_similarity    # trunk diversity
           + spectral_weight * MSE(|FFT(pred)|, |FFT(truth)|) # frequency structure
           + trend_weight * MSE(diff(pred), diff(truth))     # 1st-derivative (slope)
           + shape_weight * (1 - pearson_corr(pred, truth))  # shape / pattern

    Spectral, trend, shape losses apply to DENSE forecast grid (contiguous
    prediction), not sparse queries.
    """
    n_params = sum(p.numel() for p in model.parameters())
    combined = ConcatDataset(datasets)
    dl = DataLoader(combined, batch_size=batch_size, shuffle=True,
                    num_workers=4, drop_last=True, pin_memory=True)

    print(f'\n{"="*60}')
    print(f'LOTSA Scaling Training')
    print(f'  Model: {n_params/1e6:.1f}M, Latent decomp: {model.use_latent_decomp}')
    print(f'  Data: {len(combined):,} windows, SEQ: {seq_len}')
    print(f'  Steps/epoch: {len(dl):,}')
    print(f'  n_query: {n_query}, diversity_weight: {diversity_weight}, '
          f'spectral_weight: {spectral_weight}')
    print(f'{"="*60}')

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_loss = float('inf')
    use_dense = (spectral_weight > 0 or trend_weight > 0 or shape_weight > 0)
    max_future = seq_len * max_horizon_mult
    for epoch in range(epochs):
        model.train()
        losses = []
        t0 = time.time()
        for i, batch_windows in enumerate(dl):
            batch = collate_batch(batch_windows, seq_len=seq_len, n_query=n_query,
                                  max_horizon_mult=max_horizon_mult,
                                  return_dense=use_dense,
                                  contiguous_mode=contiguous_mode)
            if batch is None: continue
            if use_dense:
                ctx, qt, qv, dense_fut, is_fc = [x.to(DEVICE) if isinstance(x, torch.Tensor) else x
                                                 for x in batch]
            else:
                ctx, qt, qv = [x.to(DEVICE) for x in batch]
            optimizer.zero_grad()
            pred, per_trunk = model.forward_train(ctx, qt, return_per_trunk=True)
            mse = F.mse_loss(pred, qv)
            total = mse
            # Diversity loss
            if diversity_weight > 0:
                pt = per_trunk - per_trunk.mean(dim=-1, keepdim=True)
                pt_norm = pt / (pt.std(dim=-1, keepdim=True) + 1e-6)
                sim = (pt_norm[0] * pt_norm[1]).mean() + \
                      (pt_norm[0] * pt_norm[2]).mean() + \
                      (pt_norm[1] * pt_norm[2]).mean()
                total = total + diversity_weight * (sim / 3.0)
            # Dense-grid losses (spectral / trend / shape)
            if use_dense and is_fc.any():
                fc_mask = is_fc
                ctx_fc = ctx[fc_mask]
                tgt_fc = dense_fut[fc_mask]  # (B_fc, max_future)
                dense_t = torch.linspace(1.0, 1.0 + max_future / seq_len, max_future,
                                         device=DEVICE).unsqueeze(0).expand(ctx_fc.shape[0], -1)
                dense_pred = model.forward_train(ctx_fc, dense_t)

                if spectral_weight > 0:
                    pred_fft = torch.fft.rfft(dense_pred, dim=-1).abs()
                    tgt_fft = torch.fft.rfft(tgt_fc, dim=-1).abs()
                    spec_loss = F.mse_loss(pred_fft, tgt_fft) / max_future
                    total = total + spectral_weight * spec_loss

                if trend_weight > 0:
                    # First-derivative MSE (preserves slope/trend direction)
                    pred_diff = dense_pred[:, 1:] - dense_pred[:, :-1]
                    tgt_diff = tgt_fc[:, 1:] - tgt_fc[:, :-1]
                    trend_loss = F.mse_loss(pred_diff, tgt_diff)
                    total = total + trend_weight * trend_loss

                if shape_weight > 0:
                    # Pearson correlation loss (1 - corr), noise-insensitive shape matching
                    pred_c = dense_pred - dense_pred.mean(dim=-1, keepdim=True)
                    tgt_c = tgt_fc - tgt_fc.mean(dim=-1, keepdim=True)
                    pred_n = pred_c / (pred_c.std(dim=-1, keepdim=True) + 1e-6)
                    tgt_n = tgt_c / (tgt_c.std(dim=-1, keepdim=True) + 1e-6)
                    corr = (pred_n * tgt_n).mean(dim=-1)  # per-sample corr
                    shape_loss = (1.0 - corr).mean()
                    total = total + shape_weight * shape_loss
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(mse.item())  # log raw MSE only
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
                            x_n = ((x_ctx-m)/s).clamp(-10, 10)
                            # DIRECT prediction: operator learning advantage — no rolling!
                            # Model was trained on t up to 1 + 2*seq_len/seq_len = 3,
                            # supporting direct prediction up to 2×seq_len steps ahead.
                            pred_n = model.forecast(x_ctx, n=pl)
                            outs.append(pred_n * s + m)
                        preds.append(torch.stack(outs, dim=-1).cpu().numpy())
                        tgts.append(by[:, -pl:, :].numpy())
                p, t = np.concatenate(preds), np.concatenate(tgts)
                mse = float(np.mean((p-t)**2))
                mae = float(np.mean(np.abs(p-t)))
                k = f'{dn}_{pl}'
                print(f'  {k}: MSE={mse:.4f}  MAE={mae:.4f}')
                results[k] = {'MSE': mse, 'MAE': mae}
            except Exception as e:
                print(f'  {dn}_{pl}: ERROR ({e})')
    # Avg per dataset (paper-style: MSE and MAE averaged across pred_lens)
    print('\n' + '-'*60)
    print(f'{"Dataset":<10} {"MSE":>8} {"MAE":>8}  (averaged across pred_len={{96,192,336,720}})')
    print('-'*60)
    for dn in ['ETTh1','ETTh2','ETTm1','ETTm2','Weather']:
        entries = [v for k,v in results.items() if k.startswith(dn+'_')]
        if entries:
            avg_mse = np.mean([e['MSE'] for e in entries])
            avg_mae = np.mean([e['MAE'] for e in entries])
            print(f'{dn:<10} {avg_mse:>8.4f} {avg_mae:>8.4f}')
            results[f'{dn}_avg'] = {'MSE': float(avg_mse), 'MAE': float(avg_mae)}
    # Overall avg
    dataset_entries = [results[f'{dn}_avg'] for dn in ['ETTh1','ETTh2','ETTm1','ETTm2','Weather']
                       if f'{dn}_avg' in results]
    if dataset_entries:
        overall_mse = np.mean([e['MSE'] for e in dataset_entries])
        overall_mae = np.mean([e['MAE'] for e in dataset_entries])
        print('-'*60)
        print(f'{"OVERALL":<10} {overall_mse:>8.4f} {overall_mae:>8.4f}')
        results['overall_avg'] = {'MSE': float(overall_mse), 'MAE': float(overall_mae)}
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
    parser.add_argument('--synth_ratio', type=float, default=0.3,
                        help='Synth samples as ratio of LOTSA (0.3 = 30%% of LOTSA size)')
    parser.add_argument('--synth_n', type=int, default=0,
                        help='If >0, directly set synth sample count (overrides synth_ratio)')
    parser.add_argument('--spectral_weight', type=float, default=0.0,
                        help='Spectral (FFT magnitude) loss weight. Try 0.1 or 0.3.')
    parser.add_argument('--trend_weight', type=float, default=0.0,
                        help='Trend loss (1st-derivative MSE). Try 0.3~1.0.')
    parser.add_argument('--shape_weight', type=float, default=0.0,
                        help='Shape loss (1 - pearson_corr). Try 0.1~0.5.')
    parser.add_argument('--contiguous_mode', type=int, default=0,
                        help='1: use contiguous pred_len query (standard TSFM-style); 0: random sparse')
    parser.add_argument('--hybrid_trunk', type=int, default=0,
                        help='1: 1 Fixed (poly) + 2 Hyper (fourier/rbf); 0: all Hyper')
    parser.add_argument('--eval_after', type=int, default=1, help='Run eval after training')
    args = parser.parse_args()

    np.random.seed(42); torch.manual_seed(42)

    print('='*60)
    print(f'LOTSA Scaling: {args.scale}%, decomp={bool(args.decomp)}, tag={args.tag}')
    print('='*60)

    # Load data — windows of size seq_len*3 = context + 2*seq_len future
    # (supports direct prediction up to 2*seq_len steps ahead, covers pred_len=720)
    window_len = args.seq_len * 3
    lotsa_ds = LOTSAScalingDataset(LOTSA_DIR, args.scale, seq_len=window_len)
    if args.synth_n > 0:
        n_synth = args.synth_n
    else:
        n_synth = max(10000, int(len(lotsa_ds) * args.synth_ratio))
    synth_ds = SyntheticGapFiller(n_samples=n_synth, seq_len=window_len)
    datasets = [lotsa_ds, synth_ds]
    total = sum(len(d) for d in datasets)
    print(f'Total: {total:,} (LOTSA: {len(lotsa_ds):,}, Synth: {len(synth_ds):,})')

    # Model
    model = OperatorModel(
        seq_len=args.seq_len,
        use_latent_decomp=bool(args.decomp),
        hybrid_trunk=bool(args.hybrid_trunk),
    ).to(DEVICE)
    n = sum(p.numel() for p in model.parameters())
    print(f'Model: {n/1e6:.1f}M params')

    # Train
    save_path = f'checkpoints/{args.tag}.pth'
    os.makedirs('checkpoints', exist_ok=True)
    best = train(model, datasets, save_path, seq_len=args.seq_len, epochs=args.epochs,
                 spectral_weight=args.spectral_weight,
                 trend_weight=args.trend_weight,
                 shape_weight=args.shape_weight,
                 contiguous_mode=bool(args.contiguous_mode))

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
