"""
Trunk decomposition: good case (ETTm1) vs bad case (ETTh2) side by side.
What does each trunk do when model succeeds vs fails?
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import torch, numpy as np, math
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from types import SimpleNamespace
from experiments.exp_v1_varlen_ext import OperatorModelVarLen

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_sample(ds_key, seq_len=720, pred_len=192, sample_idx=300):
    from data_provider.data_factory import data_provider
    cfg = {
        'ETTm1': ('ETTm1', './dataset/ETT-small/', 'ETTm1.csv', 7),
        'ETTh2': ('ETTh2', './dataset/ETT-small/', 'ETTh2.csv', 7),
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
    for i, (bx, by, _, _) in enumerate(tdl):
        if i == sample_idx:
            return bx[0, -seq_len:, 0].numpy(), by[0, -pred_len:, 0].numpy()
    return None, None

@torch.no_grad()
def get_trunk_outputs(model, ctx_np, horizon):
    ctx = torch.tensor(ctx_np, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    ps = model.encoder.patch_size
    eff = (ctx.shape[-1] // ps) * ps
    ctx = ctx[..., -eff:]
    m = ctx.mean(-1, keepdim=True)
    s = ctx.std(-1, keepdim=True).clamp(min=1e-6)
    x_n = ((ctx - m) / s).clamp(-10, 10)
    t_end = 1.0 + horizon / eff
    qt = torch.linspace(1.0, t_end, horizon, device=DEVICE).unsqueeze(0)
    out, per_trunk = model.forward_train(x_n, qt, return_per_trunk=True)
    out_d = (out * s.squeeze(-1) + m.squeeze(-1)).squeeze(0).cpu().numpy()
    trunks = []
    for k in range(per_trunk.shape[0]):
        # Per-trunk contribution (denormalized)
        t_val = per_trunk[k].squeeze(0).cpu().numpy() * s.item()
        trunks.append(t_val)
    return out_d, trunks

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--use_nll', type=int, default=0)
    args = p.parse_args()

    model = OperatorModelVarLen(max_seq_len=720, use_nll=bool(args.use_nll)).to(DEVICE)
    state = torch.load(args.ckpt, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    datasets = ['ETTm1', 'ETTh2', 'ETTm2', 'Weather']
    labels = [
        'ETTm1 (periodic, +56%)\nOur model TRACKS pattern',
        'ETTh2 (noisy, +7%)\nOur model goes FLAT',
        'ETTm2 (low var, +18%)\nOur model goes FLAT',
        'Weather (trend, +10%)\nMixed behavior',
    ]
    trunk_names = ['Fourier', 'Polynomial', 'RBF']
    trunk_colors = ['#e74c3c', '#3498db', '#2ecc71']
    pred_len = 192

    fig, axes = plt.subplots(len(datasets), 5, figsize=(22, 3.2 * len(datasets)))

    for row, (ds, label) in enumerate(zip(datasets, labels)):
        ctx, true = load_sample(ds, pred_len=pred_len, sample_idx=500)
        if ctx is None: continue
        pred, trunks = get_trunk_outputs(model, ctx, pred_len)

        show = 200
        t_c = np.arange(show)
        t_f = np.arange(show, show + pred_len)

        # Col 0: full prediction
        ax = axes[row, 0]
        ax.plot(t_c, ctx[-show:], 'k-', alpha=0.4, linewidth=1)
        ax.plot(t_f, true, 'g-', linewidth=1.5, label='truth')
        ax.plot(t_f, pred, 'r--', linewidth=1.5, label='ours')
        ax.axvline(show, color='gray', linestyle=':', alpha=0.3)
        mse = ((pred - true) ** 2).mean()
        ax.set_title(f'Combined (MSE={mse:.3f})', fontsize=9)
        ax.set_ylabel(label, fontsize=9, fontweight='bold')
        if row == 0: ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        # Col 1-3: per-trunk
        for k in range(3):
            ax = axes[row, k+1]
            ax.plot(t_f, trunks[k], '-', color=trunk_colors[k], linewidth=1.5)
            ax.axhline(0, color='gray', linestyle=':', alpha=0.3)
            energy = np.std(trunks[k])
            ax.set_title(f'{trunk_names[k]} (energy={energy:.3f})', fontsize=9)
            ax.grid(alpha=0.3)

        # Col 4: energy bar chart
        ax = axes[row, 4]
        energies = [np.std(t) for t in trunks]
        bars = ax.bar(trunk_names, energies, color=trunk_colors, edgecolor='black')
        ax.set_title('Trunk Energy (std)', fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        for b, e in zip(bars, energies):
            ax.text(b.get_x() + b.get_width()/2, e + 0.005, f'{e:.3f}',
                    ha='center', fontsize=8)

    fig.suptitle('Trunk Decomposition: Good case (ETTm1) vs Bad cases (ETTh2/ETTm2/Weather)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    out = 'results/figures/capabilities/trunk_good_vs_bad.png'
    plt.savefig(out, dpi=100, bbox_inches='tight')
    print(f'Saved {out}')

if __name__ == '__main__':
    main()