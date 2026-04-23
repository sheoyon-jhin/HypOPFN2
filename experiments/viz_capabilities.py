"""
Model capability demonstration:
  1. One forward → arbitrary horizon (96, 192, 336, 720 동시)
  2. Same model → forecast AND imputation
  3. Variable-length context (192, 384, 720 all work)
  4. Trunk decomposition (what Fourier/Poly/RBF each contribute)
  5. Per-channel independent processing (7ch ETT, 21ch Weather)

Shows what operator learning UNIQUELY enables.
"""
import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch, torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from types import SimpleNamespace

from experiments.exp_v1_varlen_ext import OperatorModelVarLen, OperatorModelDecomp

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_real_data(ds_key, seq_len=720, pred_len=720):
    from data_provider.data_factory import data_provider
    cfg = {
        'ETTh1': ('ETTh1', './dataset/ETT-small/', 'ETTh1.csv', 7),
        'ETTh2': ('ETTh2', './dataset/ETT-small/', 'ETTh2.csv', 7),
        'ETTm1': ('ETTm1', './dataset/ETT-small/', 'ETTm1.csv', 7),
        'ETTm2': ('ETTm2', './dataset/ETT-small/', 'ETTm2.csv', 7),
        'Weather': ('custom', './dataset/weather/', 'weather.csv', 21),
    }
    d, root, f, enc_in = cfg[ds_key]
    a = SimpleNamespace(seq_len=seq_len, pred_len=pred_len, label_len=48, data=d,
        root_path=root, data_path=f, features='M', target='OT', freq='h',
        embed='timeF', enc_in=enc_in, dec_in=enc_in, c_out=enc_in,
        num_workers=0, batch_size=1, exp_name='MTSF', ordered_data=False,
        data_amount=-1, combine_Gaussian_datasets=False, synthetic_data_path='',
        synthetic_root_path='./', synthetic_length=1024, stride=-1)
    _, tdl = data_provider(a, 'test')
    return tdl, enc_in


@torch.no_grad()
def predict(model, ctx_tensor, horizon, seq_len_ref=None):
    ps = model.encoder.patch_size
    eff = (ctx_tensor.shape[-1] // ps) * ps
    ctx = ctx_tensor[..., -eff:]
    m = ctx.mean(-1, keepdim=True)
    s = ctx.std(-1, keepdim=True).clamp(min=1e-6)
    x_n = ((ctx - m) / s).clamp(-10, 10)
    if x_n.dim() == 1: x_n = x_n.unsqueeze(0)
    ref = seq_len_ref if seq_len_ref else eff
    pred = model.forecast(x_n, n=horizon, seq_len_ref=ref).squeeze(0)
    return (pred * s.squeeze(-1) + m.squeeze(-1)).cpu().numpy()


@torch.no_grad()
def predict_with_trunks(model, ctx_tensor, horizon):
    ps = model.encoder.patch_size
    eff = (ctx_tensor.shape[-1] // ps) * ps
    ctx = ctx_tensor[..., -eff:]
    m = ctx.mean(-1, keepdim=True)
    s = ctx.std(-1, keepdim=True).clamp(min=1e-6)
    x_n = ((ctx - m) / s).clamp(-10, 10)
    if x_n.dim() == 1: x_n = x_n.unsqueeze(0)
    t_end = 1.0 + horizon / eff
    qt = torch.linspace(1.0, t_end, horizon, device=ctx.device).unsqueeze(0)
    out, per_trunk = model.forward_train(x_n, qt, return_per_trunk=True)
    out_d = (out * s.squeeze(-1) + m.squeeze(-1)).squeeze(0).cpu().numpy()
    per_d = []
    for k in range(per_trunk.shape[0]):
        p = (per_trunk[k] * s.squeeze(-1) + m.squeeze(-1)).squeeze(0).cpu().numpy()
        per_d.append(p)
    return out_d, per_d


@torch.no_grad()
def impute(model, full_signal, mask_rate=0.375, seed=42):
    L = len(full_signal)
    m, s = full_signal.mean(), full_signal.std().clip(min=1e-6)
    x_n = np.clip((full_signal - m) / s, -10, 10)
    rng = np.random.RandomState(seed)
    mask = rng.rand(L) > mask_rate
    x_masked = x_n * mask.astype(np.float32)
    mi = np.where(~mask)[0]
    ctx = torch.tensor(x_masked, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    qt = torch.tensor(mi / L, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    pred_n = model.forward_train(ctx, qt).squeeze(0).cpu().numpy()
    pred = pred_n * s + m
    return mask, mi, pred, full_signal[mi]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--use_nll', type=int, default=0)
    p.add_argument('--hybrid_trunk', type=int, default=0)
    p.add_argument('--all_fixed', type=int, default=0)
    p.add_argument('--highfreq_nf', type=int, default=0)
    p.add_argument('--fourier_nf', type=int, default=32)
    p.add_argument('--multi_scale_fourier', type=int, default=0)
    p.add_argument('--multi_scale_iq', type=int, default=0)
    p.add_argument('--pool_type', type=str, default='mean')
    p.add_argument('--d_model', type=int, default=512)
    p.add_argument('--n_layers', type=int, default=6)
    p.add_argument('--trunk_w', type=int, default=192)
    p.add_argument('--tag', type=str, default='viz')
    p.add_argument('--dataset', type=str, default='ETTm1',
                   choices=['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'Weather'])
    p.add_argument('--model_type', type=str, default='varlen',
                   choices=['varlen', 'decomp'])
    p.add_argument('--decomp_kernels', type=str, default='49,25,7')
    args = p.parse_args()

    if args.model_type == 'decomp':
        decomp_k = tuple(int(x) for x in args.decomp_kernels.split(','))
        model = OperatorModelDecomp(
            max_seq_len=720,
            d_model=args.d_model,
            n_layers=args.n_layers,
            trunk_w=args.trunk_w,
            fourier_nf=args.fourier_nf,
            pool_type=args.pool_type,
            highfreq_nf=args.highfreq_nf,
            all_fixed=bool(args.all_fixed),
            decomp_kernels=decomp_k,
        ).to(DEVICE)
    else:
        model = OperatorModelVarLen(
            max_seq_len=720,
            d_model=args.d_model,
            n_layers=args.n_layers,
            trunk_w=args.trunk_w,
            hybrid_trunk=bool(args.hybrid_trunk),
            use_nll=bool(args.use_nll),
            fourier_nf=args.fourier_nf,
            multi_scale_fourier=bool(args.multi_scale_fourier),
            multi_scale_iq=bool(args.multi_scale_iq),
            pool_type=args.pool_type,
            highfreq_nf=args.highfreq_nf,
            all_fixed=bool(args.all_fixed),
        ).to(DEVICE)
    state = torch.load(args.ckpt, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    out_dir = f'results/figures/capabilities_{args.tag}'
    os.makedirs(out_dir, exist_ok=True)

    # Pick one good test sample from chosen dataset
    ds_name = args.dataset
    tdl, C = load_real_data(ds_name, seq_len=720, pred_len=720)
    sample = None
    for i, (bx, by, _, _) in enumerate(tdl):
        if i == 300:
            sample = (bx[0].numpy(), by[0].numpy())
            break
    ctx_raw = sample[0][-720:, 0]   # ch 0
    future_raw = sample[1][-720:, 0]
    ctx_tensor = torch.tensor(ctx_raw, dtype=torch.float32, device=DEVICE)

    # ========================================
    # FIG 1: Arbitrary Horizon — one context, 4 horizons
    # ========================================
    fig, axes = plt.subplots(1, 4, figsize=(20, 3.5), sharey=True)
    for j, h in enumerate([96, 192, 336, 720]):
        ax = axes[j]
        pred = predict(model, ctx_tensor, h)
        true = future_raw[:h]
        show_ctx = 200
        t_c = np.arange(show_ctx)
        t_f = np.arange(show_ctx, show_ctx + h)
        ax.plot(t_c, ctx_raw[-show_ctx:], 'k-', alpha=0.4, linewidth=1)
        ax.plot(t_f, true, 'g-', linewidth=1.5, label='truth')
        ax.plot(t_f, pred, 'r--', linewidth=1.5, label='ours')
        ax.axvline(show_ctx, color='gray', linestyle=':', alpha=0.5)
        ax.set_title(f'h={h}', fontsize=11, fontweight='bold')
        ax.grid(alpha=0.3)
        if j == 0: ax.legend(fontsize=9)
    fig.suptitle(f'Capability 1: One model → arbitrary horizon ({ds_name}, single forward each)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    fig.savefig(f'{out_dir}/cap1_arbitrary_horizon.png', dpi=100, bbox_inches='tight')
    plt.close()
    print(f'Saved cap1_arbitrary_horizon.png')

    # ========================================
    # FIG 2: Forecast + Imputation (same model, same weights)
    # ========================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 3.5))
    # Left: forecast
    ax = axes[0]
    pred = predict(model, ctx_tensor, 192)
    true = future_raw[:192]
    show_ctx = 200
    t_c = np.arange(show_ctx)
    t_f = np.arange(show_ctx, show_ctx + 192)
    ax.plot(t_c, ctx_raw[-show_ctx:], 'k-', alpha=0.4, linewidth=1)
    ax.plot(t_f, true, 'g-', linewidth=1.5, label='truth')
    ax.plot(t_f, pred, 'r--', linewidth=1.5, label='ours')
    ax.axvline(show_ctx, color='gray', linestyle=':', alpha=0.5)
    ax.set_title('FORECAST (h=192)', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Right: imputation
    ax = axes[1]
    mask, mi, pred_imp, true_imp = impute(model, ctx_raw, mask_rate=0.375)
    win = 300
    t = np.arange(len(ctx_raw))
    ax.plot(t[-win:], ctx_raw[-win:], 'k-', alpha=0.25, linewidth=0.8)
    visible = ctx_raw.copy()
    visible[~mask] = np.nan
    ax.plot(t[-win:], visible[-win:], 'b.', markersize=2, alpha=0.6, label='visible')
    mi_win = mi[mi >= len(ctx_raw) - win]
    pred_win = pred_imp[np.isin(mi, mi_win)]
    true_win = ctx_raw[mi_win]
    ax.scatter(mi_win, true_win, c='g', s=10, alpha=0.8, zorder=3, label='masked truth')
    ax.scatter(mi_win, pred_win, c='r', marker='x', s=15, zorder=4, label='ours')
    ax.set_title('IMPUTATION (mask=37.5%)', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(alpha=0.3)

    fig.suptitle('Capability 2: Same model → forecast AND imputation (unified operator)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    fig.savefig(f'{out_dir}/cap2_unified_tasks.png', dpi=100, bbox_inches='tight')
    plt.close()
    print(f'Saved cap2_unified_tasks.png')

    # ========================================
    # FIG 3: Variable-length context
    # ========================================
    fig, axes = plt.subplots(1, 3, figsize=(16, 3.5))
    for j, sl in enumerate([192, 384, 720]):
        ax = axes[j]
        ctx_sl = ctx_raw[-sl:]
        pred = predict(model, torch.tensor(ctx_sl, dtype=torch.float32, device=DEVICE), 96, seq_len_ref=sl)
        true = future_raw[:96]
        show = min(sl, 200)
        t_c = np.arange(show)
        t_f = np.arange(show, show + 96)
        ax.plot(t_c, ctx_sl[-show:], 'k-', alpha=0.4, linewidth=1)
        ax.plot(t_f, true, 'g-', linewidth=1.5, label='truth')
        ax.plot(t_f, pred, 'r--', linewidth=1.5, label='ours')
        ax.axvline(show, color='gray', linestyle=':', alpha=0.5)
        mse = ((pred - true) ** 2).mean()
        ax.set_title(f'ctx_len={sl} → h=96 (MSE={mse:.3f})', fontsize=10, fontweight='bold')
        ax.grid(alpha=0.3)
        if j == 0: ax.legend(fontsize=9)
    fig.suptitle('Capability 3: Variable-length context (sinusoidal PE + padding mask)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    fig.savefig(f'{out_dir}/cap3_variable_length.png', dpi=100, bbox_inches='tight')
    plt.close()
    print(f'Saved cap3_variable_length.png')

    # ========================================
    # FIG 4: Trunk decomposition
    # ========================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 7))
    pred_all, per_trunk = predict_with_trunks(model, ctx_tensor, 192)
    true = future_raw[:192]
    trunk_names = ['Fourier', 'Polynomial', 'RBF']
    show = 200
    t_c = np.arange(show)
    t_f = np.arange(show, show + 192)

    ax = axes[0, 0]
    ax.plot(t_c, ctx_raw[-show:], 'k-', alpha=0.4, linewidth=1)
    ax.plot(t_f, true, 'g-', linewidth=1.5, label='truth')
    ax.plot(t_f, pred_all, 'r--', linewidth=1.5, label='combined')
    ax.axvline(show, color='gray', linestyle=':', alpha=0.5)
    ax.set_title('Combined (all 3 trunks)', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    for k in range(3):
        ax = axes[(k+1)//2, (k+1)%2]
        ax.plot(t_f, per_trunk[k], '-', linewidth=1.5, color=['#e74c3c', '#3498db', '#2ecc71'][k])
        ax.axhline(0, color='gray', linestyle=':', alpha=0.3)
        ax.set_title(f'Trunk {k+1}: {trunk_names[k]}', fontsize=11, fontweight='bold')
        ax.grid(alpha=0.3)

    fig.suptitle(f'Capability 4: Trunk decomposition — what each basis contributes ({ds_name} h=192)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    fig.savefig(f'{out_dir}/cap4_trunk_decomposition.png', dpi=100, bbox_inches='tight')
    plt.close()
    print(f'Saved cap4_trunk_decomposition.png')

    # ========================================
    # FIG 5: Multi-channel processing
    # ========================================
    fig, axes = plt.subplots(2, 4, figsize=(18, 6))
    channels_to_show = min(4, C)
    for ch in range(channels_to_show):
        ctx_ch = sample[0][-720:, ch]
        true_ch = sample[1][-192:, ch]
        pred_ch = predict(model,
            torch.tensor(ctx_ch, dtype=torch.float32, device=DEVICE), 192)
        ax_top = axes[0, ch]
        ax_bot = axes[1, ch]
        show = 200
        t_c = np.arange(show)
        t_f = np.arange(show, show + 192)
        ax_top.plot(t_c, ctx_ch[-show:], 'k-', alpha=0.4, linewidth=1)
        ax_top.plot(t_f, true_ch, 'g-', linewidth=1.5)
        ax_top.plot(t_f, pred_ch, 'r--', linewidth=1.5)
        ax_top.axvline(show, color='gray', linestyle=':', alpha=0.5)
        mse = ((pred_ch - true_ch) ** 2).mean()
        ax_top.set_title(f'Ch {ch} (MSE={mse:.3f})', fontsize=10)
        ax_top.grid(alpha=0.3)

        # Imputation for same channel
        mask, mi, pred_imp, true_imp = impute(model, ctx_ch, mask_rate=0.25, seed=ch)
        win = 250
        t = np.arange(len(ctx_ch))
        ax_bot.plot(t[-win:], ctx_ch[-win:], 'k-', alpha=0.25, linewidth=0.8)
        mi_win = mi[mi >= len(ctx_ch) - win]
        pred_win = pred_imp[np.isin(mi, mi_win)]
        ax_bot.scatter(mi_win, ctx_ch[mi_win], c='g', s=8, alpha=0.7, zorder=3)
        ax_bot.scatter(mi_win, pred_win, c='r', marker='x', s=12, zorder=4)
        ax_bot.set_title(f'Ch {ch} impute (mr=25%)', fontsize=10)
        ax_bot.grid(alpha=0.3)

    axes[0, 0].legend(['ctx', 'truth', 'ours'], fontsize=8)
    axes[1, 0].legend(['original', 'truth', 'ours'], fontsize=8)
    fig.suptitle(f'Capability 5: Per-channel forecast (top) + imputation (bottom) — {ds_name} {C}ch',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    fig.savefig(f'{out_dir}/cap5_multichannel.png', dpi=100, bbox_inches='tight')
    plt.close()
    print(f'Saved cap5_multichannel.png')

    print(f'\nAll saved to {out_dir}/')


if __name__ == '__main__':
    main()
