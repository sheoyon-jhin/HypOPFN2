"""
Paper Figure: Architecture Comparison - Which is the True Function-to-Function Learner?

Compares 3 of our architectures using identical function-learning diagnostics:
  - 32.5M full_scale       (1 shared encoder + 3 trunks + informed query)
  - 82.7M ConfigB+method   (1 shared encoder + 4 MLP trunks + informed query)
  - 41M L7 mixed           (4 split encoders + 4 matched trunks, NO informed query)

Tests:
  (a) Imputation accuracy + density invariance (all 3 can do)
  (b) Forecast accuracy + smoothness (all 3, 82.7M limited range)
  (c) Per-trunk decomposition
  (d) Response to input perturbation

CUDA_VISIBLE_DEVICES=X python for_figure/fig_architecture_compare.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch, torch.nn.functional as F
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from types import SimpleNamespace

from data_provider.data_factory import data_provider

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAVE_DIR = 'for_figure/figures'
os.makedirs(SAVE_DIR, exist_ok=True)

SEQ_LEN = 96


# ============================================================
# Unified inference interface for each model
# ============================================================
def load_32M():
    from experiments.exp_full_scale_train import (
        FullScaleModel, extract_freq, build_iq, hyper_forward_with_iq
    )
    model = FullScaleModel().to(DEVICE)
    model.load_state_dict(torch.load('checkpoints/full_scale_run.pth', map_location=DEVICE))
    model.eval()

    @torch.no_grad()
    def query(ctx, t_values):
        """ctx: [B, 96] raw (will be normalized), t_values: [B, nq]"""
        m = ctx.mean(dim=1, keepdim=True)
        s = ctx.std(dim=1, keepdim=True).clamp(min=1e-6)
        ctx_n = ((ctx - m) / s).clamp(-10, 10)
        z = model.encoder(ctx_n)
        freqs, phases, lv, ls = extract_freq(ctx_n)
        t_col = t_values[0:1].t()
        iq = build_iq(t_col, freqs, phases, lv, ls)
        t_flat = t_values.reshape(-1)
        out = torch.zeros(ctx.shape[0], t_values.shape[1], device=ctx.device)
        per_trunk_outs = []
        for head, trunk, bias in zip(model.heads, model.trunks, model.biases):
            ho = head(z)
            o = hyper_forward_with_iq(trunk, t_flat, ho, iq) + bias
            out = out + o
            per_trunk_outs.append(o)
        # De-normalize
        out_denorm = out * s + m
        return out_denorm, [p * s + m/3 for p in per_trunk_outs]  # each trunk contributes

    return model, query, '32.5M (shared enc + 3 trunks + IQ)', ['Fourier', 'Poly', 'RBF']


def load_82M():
    from experiments.exp_configB_revinoff_trueop import ConfigBNoRevIN, query_at_vectorized
    model = ConfigBNoRevIN(width=192, branch_hidden=768, trunk_depth=2, top_k_freq=5).to(DEVICE)
    model.load_state_dict(torch.load('checkpoints/configB_revinoff_trueop.pth', map_location=DEVICE))
    model.eval()

    @torch.no_grad()
    def query(ctx, t_values, mode='forecast'):
        """Same interface as 32M but uses ConfigB query_at_vectorized."""
        m = ctx.mean(dim=1, keepdim=True)
        s = ctx.std(dim=1, keepdim=True).clamp(min=1e-6)
        ctx_n = ((ctx - m) / s).clamp(-10, 10)
        x_cross = ctx_n
        out = query_at_vectorized(model, ctx_n, x_cross, t_values, mode=mode)
        out_denorm = out * s + m
        return out_denorm, None  # per-trunk not exposed cleanly

    return model, query, '82.7M (ConfigB 4 MLP trunks + IQ)', ['Trunk 0','Trunk 1','Trunk 2','Trunk 3']


def load_L7():
    from experiments.exp_full_scale_L7 import FullScaleL7Model, decompose_4way
    model = FullScaleL7Model().to(DEVICE)
    model.load_state_dict(torch.load('checkpoints/full_scale_L7_run.pth', map_location=DEVICE))
    model.eval()

    @torch.no_grad()
    def query(ctx, t_values):
        m = ctx.mean(dim=1, keepdim=True)
        s = ctx.std(dim=1, keepdim=True).clamp(min=1e-6)
        ctx_n = ((ctx - m) / s).clamp(-10, 10)
        trend, season, cycle, resid = decompose_4way(ctx_n)
        t_flat = t_values.reshape(-1)
        B, nq = t_values.shape
        blocks = [model.block_T, model.block_S, model.block_C, model.block_R]
        inputs = [trend, season, cycle, resid]
        per_trunk_outs = []
        total = torch.zeros(B, nq, device=ctx.device)
        for blk, inp in zip(blocks, inputs):
            z = blk.enc(inp)
            # L7 uses single trunk per block
            from experiments.exp_full_scale_L7 import hfwd
            head_out = blk.head(z)
            o = hfwd(blk.trunk, t_flat, head_out) + blk.bias
            total = total + o
            per_trunk_outs.append(o)
        total_denorm = total * s + m
        return total_denorm, [p * s + m/4 for p in per_trunk_outs]

    return model, query, '41M L7 mixed (4 split enc + matched trunks, NO IQ)', ['Poly (T)','Fourier (S)','Chirplet (C)','RBF (R)']


# ============================================================
# Get real sample
# ============================================================
def get_sample(dataset_name='ETTh1'):
    datasets = {
        'ETTh1': ('ETTh1', './dataset/ETT-small/', 'ETTh1.csv', 7),
    }
    d, root, f, enc_in = datasets[dataset_name]
    a = SimpleNamespace(seq_len=96, pred_len=96, label_len=48, data=d,
        root_path=root, data_path=f, features='M', target='OT',
        freq='h', embed='timeF', enc_in=enc_in, dec_in=enc_in, c_out=enc_in,
        num_workers=2, batch_size=1, exp_name='MTSF', ordered_data=False,
        data_amount=-1, combine_Gaussian_datasets=False,
        synthetic_data_path='', synthetic_root_path='./',
        synthetic_length=1024, stride=-1)
    _, tdl = data_provider(a, 'test')
    samples = []
    for i, (bx, by, _, _) in enumerate(tdl):
        x = bx[0, :, 0].numpy()
        y = by[0, -96:, 0].numpy()
        samples.append((x, y))
        if len(samples) >= 5: break
    return samples


# ============================================================
# Run comparison
# ============================================================
def run_comparison(sample_idx=0):
    print('Loading models...')
    models = {}
    models['32.5M'] = load_32M()
    models['82.7M'] = load_82M()
    models['L7 41M'] = load_L7()

    samples = get_sample('ETTh1')
    ctx_np, future_np = samples[sample_idx]
    print(f'Using sample {sample_idx}')

    ctx = torch.tensor(ctx_np, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    full_gt = np.concatenate([ctx_np, future_np])

    fig = plt.figure(figsize=(18, 14))

    # ============================================================
    # Row 1: Forecast density invariance per model
    # ============================================================
    densities = [10, 48, 96, 500]
    cmap = plt.cm.viridis(np.linspace(0.2, 0.9, len(densities)))

    for col, (name, (model, query_fn, full_name, trunk_names)) in enumerate(models.items()):
        ax = plt.subplot(4, 3, col + 1)
        ax.plot(range(96), ctx_np, 'k-', linewidth=2, label='context')
        ax.plot(range(96, 192), future_np, 'k--', linewidth=1, alpha=0.5, label='gt future')

        for n, c in zip(densities, cmap):
            t = torch.linspace(1, 2, n, device=DEVICE).unsqueeze(0)
            try:
                if name == '82.7M':
                    # ConfigB's forecast range is limited - clamp to training range
                    out, _ = query_fn(ctx, t, mode='forecast')
                else:
                    out, _ = query_fn(ctx, t)
                pred = out.cpu().numpy()[0]
                t_abs = np.linspace(96, 192, n)
                ax.plot(t_abs, pred, '-', color=c, alpha=0.7, linewidth=1.2,
                        label=f'n={n}' if n in [10, 500] else None)
            except Exception as e:
                ax.text(0.5, 0.5, f'ERROR\n{type(e).__name__}', ha='center', va='center',
                        transform=ax.transAxes, fontsize=9)
        ax.axvline(96, color='gray', linestyle=':', alpha=0.5)
        ax.set_title(f'{name}: Forecast density invariance\n{full_name}', fontsize=9)
        ax.legend(fontsize=7); ax.grid(alpha=0.3)
        if col == 0: ax.set_ylabel('Forecast', fontsize=10)

    # ============================================================
    # Row 2: Imputation density invariance per model
    # ============================================================
    for col, (name, (model, query_fn, full_name, _)) in enumerate(models.items()):
        ax = plt.subplot(4, 3, 3 + col + 1)
        ax.plot(range(96), ctx_np, 'k-', linewidth=1.5, alpha=0.6, label='full context')
        for n, c in zip(densities, cmap):
            t = torch.linspace(0, 1, n, device=DEVICE).unsqueeze(0)
            try:
                if name == '82.7M':
                    out, _ = query_fn(ctx, t, mode='recon')
                else:
                    out, _ = query_fn(ctx, t)
                pred = out.cpu().numpy()[0]
                t_abs = np.linspace(0, 96, n)
                ax.plot(t_abs, pred, '-', color=c, alpha=0.7, linewidth=1.2)
            except Exception as e:
                ax.text(0.5, 0.5, f'ERROR', ha='center', va='center',
                        transform=ax.transAxes, fontsize=9)
        ax.set_title(f'{name}: Imputation density invariance', fontsize=9)
        ax.grid(alpha=0.3)
        if col == 0: ax.set_ylabel('Imputation', fontsize=10)

    # ============================================================
    # Row 3: Input perturbation robustness
    # ============================================================
    rng = np.random.RandomState(42)
    for col, (name, (model, query_fn, full_name, _)) in enumerate(models.items()):
        ax = plt.subplot(4, 3, 6 + col + 1)
        t = torch.linspace(1, 2, 96, device=DEVICE).unsqueeze(0)
        try:
            if name == '82.7M':
                y_orig, _ = query_fn(ctx, t, mode='forecast')
            else:
                y_orig, _ = query_fn(ctx, t)
            y_orig = y_orig.cpu().numpy()[0]
            ax.plot(range(96), y_orig, 'k-', linewidth=2, label='y(x)')
            for noise_scale in [0.01, 0.05, 0.1]:
                noise = rng.randn(96).astype(np.float32) * noise_scale * ctx_np.std()
                ctx_noisy = torch.tensor(ctx_np + noise, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                if name == '82.7M':
                    y_noisy, _ = query_fn(ctx_noisy, t, mode='forecast')
                else:
                    y_noisy, _ = query_fn(ctx_noisy, t)
                y_noisy = y_noisy.cpu().numpy()[0]
                diff = np.mean(np.abs(y_noisy - y_orig))
                ax.plot(range(96), y_noisy, '--', alpha=0.7,
                        label=f'σ={noise_scale} (Δ={diff:.2f})')
        except Exception as e:
            ax.text(0.5, 0.5, f'ERROR', ha='center', va='center',
                    transform=ax.transAxes, fontsize=9)
        ax.set_title(f'{name}: Perturbation robustness (forecast)', fontsize=9)
        ax.legend(fontsize=7); ax.grid(alpha=0.3)
        if col == 0: ax.set_ylabel('Perturb.', fontsize=10)

    # ============================================================
    # Row 4: Per-trunk decomposition (where available)
    # ============================================================
    trunk_colors = ['#3b82f6', '#10b981', '#f59e0b', '#a855f7']
    for col, (name, (model, query_fn, full_name, trunk_names)) in enumerate(models.items()):
        ax = plt.subplot(4, 3, 9 + col + 1)
        t = torch.linspace(1, 2, 96, device=DEVICE).unsqueeze(0)
        try:
            if name == '82.7M':
                # Per-trunk not exposed
                out, per_trunk = query_fn(ctx, t, mode='forecast')
                pred = out.cpu().numpy()[0]
                ax.plot(range(96), future_np, 'k--', alpha=0.4, label='gt')
                ax.plot(range(96), pred, 'k-', linewidth=1.5, label='sum')
                ax.text(0.5, 0.5, '(per-trunk not exposed)',
                        ha='center', va='center', transform=ax.transAxes,
                        fontsize=8, color='gray', alpha=0.7)
            else:
                out, per_trunk = query_fn(ctx, t)
                pred = out.cpu().numpy()[0]
                ax.plot(range(96), future_np, 'k--', alpha=0.4, label='gt')
                ax.plot(range(96), pred, 'k-', linewidth=1.5, label='sum')
                if per_trunk is not None:
                    for pt, tn, tc in zip(per_trunk, trunk_names, trunk_colors):
                        ax.plot(range(96), pt.cpu().numpy()[0], color=tc, alpha=0.7,
                                linewidth=1, label=tn)
        except Exception as e:
            ax.text(0.5, 0.5, f'ERROR', ha='center', va='center',
                    transform=ax.transAxes, fontsize=9)
        ax.set_title(f'{name}: Per-trunk decomposition', fontsize=9)
        ax.legend(fontsize=6); ax.grid(alpha=0.3)
        if col == 0: ax.set_ylabel('Decomp.', fontsize=10)

    plt.suptitle('Architecture Comparison: Function-to-Function Learning Diagnostics (ETTh1)',
                 fontsize=13, y=1.00)
    plt.tight_layout()
    save_path = f'{SAVE_DIR}/fig_architecture_compare.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {save_path}')


if __name__ == '__main__':
    run_comparison(sample_idx=0)
    print('DONE')
