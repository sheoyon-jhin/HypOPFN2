"""
Fine-tuning for VarLen checkpoints.

Unfreezes entire model, trains on target dataset train split.

Usage:
  CUDA_VISIBLE_DEVICES=1 python experiments/eval_ft_varlen.py \
      --ckpt checkpoints/v1_varlen_nll_failmode.pth --max_seq_len 720 \
      --use_nll 1 --tag ft_varlen_nll
"""
import sys, os, json, argparse, copy
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from torch import optim
from types import SimpleNamespace

from experiments.exp_v1_varlen_ext import OperatorModelVarLen

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATASETS = {
    'ETTh1':   ('ETTh1',  './dataset/ETT-small/', 'ETTh1.csv', 7),
    'ETTh2':   ('ETTh2',  './dataset/ETT-small/', 'ETTh2.csv', 7),
    'ETTm1':   ('ETTm1',  './dataset/ETT-small/', 'ETTm1.csv', 7),
    'ETTm2':   ('ETTm2',  './dataset/ETT-small/', 'ETTm2.csv', 7),
    'Weather': ('custom', './dataset/weather/',  'weather.csv', 21),
}

FEDAL_FT = {'ETTh1': 0.380, 'ETTh2': 0.334, 'ETTm1': 0.319, 'ETTm2': 0.261, 'Weather': 0.213}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--hybrid_trunk', type=int, default=0)
    p.add_argument('--use_nll', type=int, default=0)
    p.add_argument('--max_seq_len', type=int, default=720)
    p.add_argument('--ft_epochs', type=int, default=10)
    p.add_argument('--ft_lr', type=float, default=5e-5)
    p.add_argument('--tag', type=str, required=True)
    args = p.parse_args()

    from data_provider.data_factory import data_provider

    print('=' * 70)
    print(f'FT (VarLen): {args.ckpt}')
    print(f'  max_seq_len={args.max_seq_len}, hybrid={bool(args.hybrid_trunk)}, nll={bool(args.use_nll)}')
    print(f'  ft_epochs={args.ft_epochs}, ft_lr={args.ft_lr}')
    print('=' * 70)

    patch_size = None
    results = {}

    for dn, (d, root, f, enc_in) in DATASETS.items():
        for pl in [96, 192, 336, 720]:
            print(f'\n--- {dn} pred_len={pl} ---')
            try:
                # Fresh model from checkpoint each time (avoid cross-dataset interference)
                model = OperatorModelVarLen(
                    max_seq_len=args.max_seq_len,
                    hybrid_trunk=bool(args.hybrid_trunk),
                    use_nll=bool(args.use_nll),
                ).to(DEVICE)
                state = torch.load(args.ckpt, map_location=DEVICE, weights_only=True)
                model.load_state_dict(state)
                patch_size = model.encoder.patch_size

                a = SimpleNamespace(seq_len=args.max_seq_len, pred_len=pl, label_len=48, data=d,
                    root_path=root, data_path=f, features='M', target='OT', freq='h',
                    embed='timeF', enc_in=enc_in, dec_in=enc_in, c_out=enc_in,
                    num_workers=2, batch_size=32, exp_name='MTSF', ordered_data=False,
                    data_amount=-1, combine_Gaussian_datasets=False, synthetic_data_path='',
                    synthetic_root_path='./', synthetic_length=1024, stride=-1)
                _, train_dl = data_provider(a, 'train')
                _, test_dl = data_provider(a, 'test')

                opt = optim.AdamW(model.parameters(), lr=args.ft_lr, weight_decay=0.01)
                scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.ft_epochs)

                # Fine-tune
                for ep in range(args.ft_epochs):
                    model.train()
                    total = 0; nb = 0
                    for bx, by, _, _ in train_dl:
                        bx = bx.float().to(DEVICE); by = by.float().to(DEVICE)
                        B, S, C = bx.shape
                        # Channel-independent loss
                        loss_accum = 0.0
                        for ch in range(C):
                            x_ch = bx[:, -args.max_seq_len:, ch]
                            if x_ch.shape[1] < args.max_seq_len:
                                x_ch = F.pad(x_ch, (args.max_seq_len - x_ch.shape[1], 0))
                            eff = (args.max_seq_len // patch_size) * patch_size
                            x_ch = x_ch[:, -eff:]
                            m = x_ch.mean(1, keepdim=True)
                            s = x_ch.std(1, keepdim=True).clamp(min=1e-6)
                            x_n = ((x_ch - m) / s).clamp(-10, 10)
                            # Forecast with pred_len
                            t_end = 1.0 + pl / eff
                            t = torch.linspace(1.0, t_end, pl, device=DEVICE).unsqueeze(0).expand(B, -1)
                            pred_n = model.forward_train(x_n, t)
                            true = by[:, -pl:, ch]
                            true_n = ((true - m) / s).clamp(-10, 10)
                            loss_accum = loss_accum + F.mse_loss(pred_n, true_n)
                        loss = loss_accum / C
                        opt.zero_grad(); loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        opt.step()
                        total += loss.item(); nb += 1
                    scheduler.step()
                    if (ep + 1) % 3 == 0 or ep == 0:
                        print(f'    FT epoch {ep+1}/{args.ft_epochs}: loss={total/max(nb,1):.4f}')

                # Eval
                model.eval()
                mse_list, mae_list = [], []
                with torch.no_grad():
                    for bx, by, _, _ in test_dl:
                        bx = bx.float().to(DEVICE); by = by.float().to(DEVICE)
                        B, S, C = bx.shape
                        for ch in range(C):
                            x_ch = bx[:, -args.max_seq_len:, ch]
                            if x_ch.shape[1] < args.max_seq_len:
                                x_ch = F.pad(x_ch, (args.max_seq_len - x_ch.shape[1], 0))
                            eff = (args.max_seq_len // patch_size) * patch_size
                            x_ch = x_ch[:, -eff:]
                            m = x_ch.mean(1, keepdim=True); s = x_ch.std(1, keepdim=True).clamp(min=1e-6)
                            x_n = ((x_ch - m) / s).clamp(-10, 10)
                            pred_n = model.forecast(x_n, n=pl, seq_len_ref=eff)
                            pred = pred_n * s + m
                            true = by[:, -pl:, ch]
                            mse_list.append(((pred - true) ** 2).mean().item())
                            mae_list.append((pred - true).abs().mean().item())
                mse = float(np.mean(mse_list)); mae = float(np.mean(mae_list))
                print(f'  FT eval: MSE={mse:.4f}  MAE={mae:.4f}')
                results[f'{dn}_{pl}'] = {'MSE': mse, 'MAE': mae}
                del model
                torch.cuda.empty_cache()
            except Exception as e:
                print(f'  ERROR: {e}')

    # Summary
    print('\n' + '=' * 70)
    print(f'{"Dataset":<10} {"Ours FT":<14} {"FeDaL FT":<10}')
    print('-' * 70)
    for dn in DATASETS:
        entries = [results[f'{dn}_{pl}'] for pl in [96, 192, 336, 720] if f'{dn}_{pl}' in results]
        if entries:
            avg_mse = np.mean([e['MSE'] for e in entries])
            avg_mae = np.mean([e['MAE'] for e in entries])
            results[f'{dn}_avg'] = {'MSE': float(avg_mse), 'MAE': float(avg_mae)}
            print(f'{dn:<10} {avg_mse:.3f}/{avg_mae:.3f}   {FEDAL_FT[dn]:.3f}')
    ds_e = [results[f'{dn}_avg'] for dn in DATASETS if f'{dn}_avg' in results]
    if ds_e:
        ov_mse = np.mean([e['MSE'] for e in ds_e]); ov_mae = np.mean([e['MAE'] for e in ds_e])
        results['overall_avg'] = {'MSE': float(ov_mse), 'MAE': float(ov_mae)}
        print('-' * 70)
        print(f'{"OVERALL":<10} {ov_mse:.3f}/{ov_mae:.3f}')

    os.makedirs('results', exist_ok=True)
    with open(f'results/{args.tag}.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved: results/{args.tag}.json')


if __name__ == '__main__':
    main()
