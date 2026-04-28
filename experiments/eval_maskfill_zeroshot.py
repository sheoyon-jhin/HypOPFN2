"""
Zero-shot mask-filling forecast eval for mask-recon-trained checkpoints.

Reframes forecasting as masked imputation (FeDaL/MOMENT style):
  - ctx[0:H]   = clean history
  - ctx[H:L]   = 0 (masked future region)
  - query t    = (H+i)/L for i in 0..pred_len-1   (within [0,1] like training)

Matches the training distribution of mask_recon_only collate so the model
never has to extrapolate t > 1.

Usage:
  CUDA_VISIBLE_DEVICES=0 python experiments/eval_maskfill_zeroshot.py \
      --tag hyper4_uv10pct_5trunk_maskrecon --max_seq_len 720 --highfreq_nf 256 --highfreq2_nf 512
"""
import sys, os, argparse, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
from types import SimpleNamespace

from experiments.exp_v1_varlen_ext import OperatorModelVarLen, OperatorModelDecomp, DEVICE


def maskfill_forecast(model, x_ctx, pred_len, max_seq_len):
    """
    x_ctx: (B, H) — clean history, already normalized
    pred_len: int — # future steps to predict
    max_seq_len: int — model's training max_seq_len (used as L)

    Strategy: build full-length ctx of size L = max_seq_len with last `pred_len`
    positions zeroed (masked). Query at the masked positions.

    If H + pred_len > L, we shift: keep last (L - pred_len) of history.
    If pred_len >= L, fall back to using all-zero ctx (heavy OOD).
    """
    B, H = x_ctx.shape
    L = max_seq_len
    pl = pred_len

    # H_use is bounded by both available history H and (L - pl) so future fits
    H_use = max(min(H, L - pl), 1) if pl < L else max(min(H, L // 4), 1)
    masked_len = L - H_use

    ctx = torch.zeros(B, L, device=x_ctx.device, dtype=x_ctx.dtype)
    ctx[:, :H_use] = x_ctx[:, -H_use:]

    if pl <= masked_len:
        pos = torch.arange(H_use, H_use + pl, device=x_ctx.device, dtype=torch.float32)
    else:
        # Distribute pl query points across [H_use, L) — saturate at last position
        idx = torch.arange(pl, device=x_ctx.device, dtype=torch.float32).clamp(max=masked_len - 1)
        pos = idx + H_use
    qt = (pos / float(L)).unsqueeze(0).expand(B, -1)
    with torch.no_grad():
        pred = model.forward_train(ctx, qt)
    return pred


def eval_maskfill(model, max_seq_len, tag=''):
    from data_provider.data_factory import data_provider

    datasets = {
        'ETTh1':   ('ETTh1',  './dataset/ETT-small/', 'ETTh1.csv', 7),
        'ETTh2':   ('ETTh2',  './dataset/ETT-small/', 'ETTh2.csv', 7),
        'ETTm1':   ('ETTm1',  './dataset/ETT-small/', 'ETTm1.csv', 7),
        'ETTm2':   ('ETTm2',  './dataset/ETT-small/', 'ETTm2.csv', 7),
        'Weather': ('custom', './dataset/weather/',  'weather.csv', 21),
    }
    model.eval()
    results = {}
    patch_size = model.encoder.patch_size

    for dn, (d, root, f, enc_in) in datasets.items():
        for pl in [96, 192, 336, 720]:
            try:
                a = SimpleNamespace(seq_len=max_seq_len, pred_len=pl, label_len=48, data=d,
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
                        # Use up to max_seq_len - pl of history (so future fits inside L)
                        H_target = max(patch_size, max_seq_len - pl)
                        H_target = (H_target // patch_size) * patch_size
                        if H_target < patch_size:
                            H_target = patch_size
                        H_use = min(S, H_target)
                        H_use = (H_use // patch_size) * patch_size
                        if H_use < patch_size:
                            continue
                        outs = []
                        for ch in range(C):
                            x_ch = bx[:, -H_use:, ch]
                            m = x_ch.mean(1, keepdim=True)
                            s = x_ch.std(1, keepdim=True).clamp(min=1e-6)
                            x_n = ((x_ch - m) / s).clamp(-10, 10)
                            pred_n = maskfill_forecast(model, x_n, pl, max_seq_len)
                            outs.append(pred_n * s + m)
                        preds.append(torch.stack(outs, dim=-1).cpu().numpy())
                        tgts.append(by[:, -pl:, :].numpy())
                p, t = np.concatenate(preds), np.concatenate(tgts)
                mse = float(np.mean((p - t) ** 2))
                mae = float(np.mean(np.abs(p - t)))
                k = f'{dn}_{pl}'
                print(f'  {k}: MSE={mse:.4f}  MAE={mae:.4f}')
                results[k] = {'MSE': mse, 'MAE': mae}
            except Exception as e:
                import traceback
                print(f'  {dn}_{pl}: ERROR ({e})')
                traceback.print_exc()

    print('\n' + '-' * 60)
    print(f'{"Dataset":<10} {"MSE":>8} {"MAE":>8}')
    print('-' * 60)
    for dn in ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'Weather']:
        entries = [v for k, v in results.items() if k.startswith(dn + '_') and 'avg' not in k]
        if entries:
            avg_mse = np.mean([e['MSE'] for e in entries])
            avg_mae = np.mean([e['MAE'] for e in entries])
            print(f'{dn:<10} {avg_mse:>8.4f} {avg_mae:>8.4f}')
            results[f'{dn}_avg'] = {'MSE': float(avg_mse), 'MAE': float(avg_mae)}
    ds_entries = [results[f'{dn}_avg'] for dn in ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'Weather']
                  if f'{dn}_avg' in results]
    if ds_entries:
        ov_mse = np.mean([e['MSE'] for e in ds_entries])
        ov_mae = np.mean([e['MAE'] for e in ds_entries])
        print('-' * 60)
        print(f'{"OVERALL":<10} {ov_mse:>8.4f} {ov_mae:>8.4f}')
        results['overall_avg'] = {'MSE': float(ov_mse), 'MAE': float(ov_mae)}
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tag', required=True, help='checkpoint tag (loads checkpoints/{tag}.pth)')
    p.add_argument('--max_seq_len', type=int, default=720)
    p.add_argument('--d_model', type=int, default=512)
    p.add_argument('--n_layers', type=int, default=6)
    p.add_argument('--trunk_w', type=int, default=192)
    p.add_argument('--fourier_nf', type=int, default=32)
    p.add_argument('--highfreq_nf', type=int, default=0)
    p.add_argument('--highfreq2_nf', type=int, default=0)
    p.add_argument('--pool_type', type=str, default='mean')
    p.add_argument('--model_type', type=str, default='varlen', choices=['varlen', 'decomp'])
    p.add_argument('--decomp_kernels', type=str, default='49,25,7')
    p.add_argument('--results_suffix', type=str, default='_maskfill_zs',
                   help='Saved as results/{tag}{suffix}.json')
    args = p.parse_args()

    if args.model_type == 'decomp':
        decomp_k = tuple(int(x) for x in args.decomp_kernels.split(','))
        model = OperatorModelDecomp(
            max_seq_len=args.max_seq_len, d_model=args.d_model, n_layers=args.n_layers,
            trunk_w=args.trunk_w, nhead=8, fourier_nf=args.fourier_nf,
            pool_type=args.pool_type, highfreq_nf=args.highfreq_nf,
            decomp_kernels=decomp_k,
        ).to(DEVICE)
    else:
        model = OperatorModelVarLen(
            max_seq_len=args.max_seq_len, d_model=args.d_model, n_layers=args.n_layers,
            trunk_w=args.trunk_w, fourier_nf=args.fourier_nf, pool_type=args.pool_type,
            highfreq_nf=args.highfreq_nf, highfreq2_nf=args.highfreq2_nf,
        ).to(DEVICE)

    ckpt_path = f'checkpoints/{args.tag}.pth'
    print(f'Loading: {ckpt_path}')
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))

    n = sum(p.numel() for p in model.parameters())
    print(f'Model: {n/1e6:.1f}M params')
    print('=' * 60)
    print(f'ZERO-SHOT MASK-FILL EVAL: {args.tag}')
    print(f'  max_seq_len={args.max_seq_len}, history = max_seq_len - pred_len')
    print('=' * 60)

    results = eval_maskfill(model, args.max_seq_len, tag=args.tag)

    os.makedirs('results', exist_ok=True)
    out_path = f'results/{args.tag}{args.results_suffix}.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved: {out_path}')


if __name__ == '__main__':
    main()
