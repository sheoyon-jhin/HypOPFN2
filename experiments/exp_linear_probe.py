"""
Linear Probing: MOMENT_LP와 동일한 세팅
  - Backbone(encoder) freeze
  - Forecast head만 학습 (Trunk 통해서 출력)
  - 기존 Hyper 63M checkpoint 활용

사용법:
  CUDA_VISIBLE_DEVICES=1 python experiments/exp_linear_probe.py 2>&1 | tee log/linear_probe.log
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import optim
from types import SimpleNamespace
from data_provider.data_factory import data_provider
import time

from model.DeepONetHyperMoE import Model


def linear_probe_forecasting(model, device):
    """Freeze encoder + router, train only forecast_head + trunk-related params."""
    print(f'\n{"="*60}')
    print('Linear Probing: Forecasting')
    print('(backbone frozen, forecast_head trainable, through Trunk)')
    print(f'{"="*60}')

    datasets = {
        'ETTh1': ('ETTh1', './dataset/ETT-small/', 'ETTh1.csv', 7),
        'ETTh2': ('ETTh2', './dataset/ETT-small/', 'ETTh2.csv', 7),
        'ETTm1': ('ETTm1', './dataset/ETT-small/', 'ETTm1.csv', 7),
        'ETTm2': ('ETTm2', './dataset/ETT-small/', 'ETTm2.csv', 7),
        'Weather': ('custom', './dataset/weather/', 'weather.csv', 21),
        'Exchange': ('custom', './dataset/exchange_rate/', 'exchange_rate.csv', 8),
    }

    results = {}

    for dname, (data, root, fpath, enc_in) in datasets.items():
        for pl in [96, 192, 336, 720]:
            # Reload checkpoint each time (fresh head)
            ckpt_path = 'checkpoints/scaleup_pile_pretrain.pth'
            if not os.path.exists(ckpt_path):
                print(f'  Checkpoint not found: {ckpt_path}')
                continue

            args = SimpleNamespace(
                seq_len=96, pred_len=pl, use_norm=True,
                deeponet_width=128, n_experts=4, branch_depth=4, trunk_depth=2,
                activation='gelu', dropout=0.1, branch_hidden=512,
                spectral_branch=False, skip_mode='none',
                use_cross_channel=False, trunk_basis='mixed',
                encoder_type='patch_attn', loss='MSE',
            )
            model = Model(args).to(device)
            model.load_state_dict(torch.load(ckpt_path, map_location=device), strict=False)

            # Freeze everything
            for p in model.parameters():
                p.requires_grad = False

            # Unfreeze only forecast_head (per expert)
            trainable_params = []
            for name, p in model.named_parameters():
                if 'forecast_head' in name:
                    p.requires_grad = True
                    trainable_params.append(p)

            n_trainable = sum(p.numel() for p in trainable_params)
            n_total = sum(p.numel() for p in model.parameters())

            data_args = SimpleNamespace(
                seq_len=96, pred_len=pl, label_len=48,
                data=data, root_path=root, data_path=fpath,
                features='M', target='OT', freq='h', embed='timeF',
                enc_in=enc_in, dec_in=enc_in, c_out=enc_in,
                num_workers=2, batch_size=32,
                exp_name='MTSF', ordered_data=False, data_amount=-1,
                combine_Gaussian_datasets=False,
                synthetic_data_path='', synthetic_root_path='./',
                synthetic_length=1024, stride=-1,
            )

            _, train_dl = data_provider(data_args, 'train')
            _, test_dl = data_provider(data_args, 'test')

            optimizer = optim.Adam(trainable_params, lr=0.001)
            best_loss, patience, best_state = float('inf'), 0, None

            for epoch in range(20):
                model.train()
                losses = []
                for bx, by, _, _ in train_dl:
                    bx, by = bx.float().to(device), by.float().to(device)
                    optimizer.zero_grad()
                    out = model(bx, None, None, None, target_pred_len=pl)
                    if isinstance(out, tuple): out = out[0]
                    loss = F.mse_loss(out, by[:, -pl:, :])
                    loss.backward()
                    optimizer.step()
                    losses.append(loss.item())

                tl = np.mean(losses)
                if tl < best_loss:
                    best_loss = tl
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    patience = 0
                else:
                    patience += 1
                    if patience >= 5: break

            # Test
            if best_state:
                model.load_state_dict(best_state)
            model.to(device).eval()
            preds, trues = [], []
            with torch.no_grad():
                for bx, by, _, _ in test_dl:
                    bx = bx.float().to(device)
                    out = model(bx, None, None, None, target_pred_len=pl)
                    if isinstance(out, tuple): out = out[0]
                    preds.append(out.cpu().numpy())
                    trues.append(by[:, -pl:, :].numpy())
            p, t = np.concatenate(preds), np.concatenate(trues)
            mse = np.mean((p - t) ** 2)
            mae = np.mean(np.abs(p - t))

            key = f'{dname}_pl{pl}'
            results[key] = {'mse': mse, 'mae': mae}
            print(f'  {key}: MSE={mse:.4f} MAE={mae:.4f}  (trainable: {n_trainable:,}/{n_total:,})')

            # Unfreeze for next round
            for p in model.parameters():
                p.requires_grad = True

    return results


if __name__ == '__main__':
    device = torch.device('cuda')

    # Check checkpoint exists
    ckpt_path = 'checkpoints/scaleup_pile_pretrain.pth'
    if not os.path.exists(ckpt_path):
        print(f'ERROR: {ckpt_path} not found!')
        print('Run exp_scaleup_pretrain.py first.')
        sys.exit(1)

    results = linear_probe_forecasting(None, device)

    print(f'\n{"="*60}')
    print('FINAL: Linear Probe Forecasting (63M Hyper + Pile)')
    print(f'{"="*60}')
    print(f'\n{"Dataset":<20} {"MSE":>8} {"MAE":>8} {"MOMENT_LP":>12}')
    print('-' * 50)

    moment_lp = {
        'ETTh1_pl96': 0.387, 'ETTh1_pl192': 0.410, 'ETTh1_pl336': 0.422, 'ETTh1_pl720': 0.454,
        'ETTh2_pl96': 0.288, 'ETTh2_pl192': 0.349, 'ETTh2_pl336': 0.369, 'ETTh2_pl720': 0.403,
        'ETTm1_pl96': 0.293, 'ETTm1_pl192': 0.326, 'ETTm1_pl336': 0.352, 'ETTm1_pl720': 0.405,
        'ETTm2_pl96': 0.170, 'ETTm2_pl192': 0.227, 'ETTm2_pl336': 0.275, 'ETTm2_pl720': 0.363,
        'Weather_pl96': 0.154, 'Weather_pl192': 0.197, 'Weather_pl336': 0.246, 'Weather_pl720': 0.315,
    }

    for k, v in results.items():
        mlp = moment_lp.get(k, '-')
        mlp_str = f'{mlp:.3f}' if isinstance(mlp, float) else mlp
        print(f'  {k:<20} {v["mse"]:>8.4f} {v["mae"]:>8.4f} {mlp_str:>12}')
