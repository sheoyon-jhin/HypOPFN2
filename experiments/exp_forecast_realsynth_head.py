"""
Real+Synth checkpoint → Forecasting task-specific head
Frozen backbone + 2층 head + strong regularization
+ 기존 forecast_head 구조 활용 (Trunk 통해 출력)

사용법:
  CUDA_VISIBLE_DEVICES=3 python experiments/exp_forecast_realsynth_head.py 2>&1 | tee log/eval/forecast_realsynth_head.log
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
import math

from model.DeepONetHyperMoE import Model


def train_forecast_head(model, device, data, root, fpath, enc_in, pl):
    """Frozen backbone + forecast_head만 학습 (기존 구조 그대로, lr/epoch 조절)."""

    args = SimpleNamespace(
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
    _, train_dl = data_provider(args, 'train')
    _, test_dl = data_provider(args, 'test')

    # Freeze everything
    for p in model.parameters():
        p.requires_grad = False

    # Unfreeze forecast_head only (per expert)
    trainable = []
    for name, p in model.named_parameters():
        if 'forecast_head' in name:
            p.requires_grad = True
            trainable.append(p)

    n_train = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in model.parameters())
    print(f'    Trainable: {n_train:,} / {n_total:,} ({100*n_train/n_total:.1f}%)')

    optimizer = optim.Adam(trainable, lr=0.0005, weight_decay=0.01)
    best_loss, patience, best_state = float('inf'), 0, None

    for epoch in range(50):
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
        if epoch % 5 == 0 or tl < best_loss:
            print(f'    epoch {epoch+1}: loss={tl:.4f} (best={best_loss:.4f})')
        if tl < best_loss:
            best_loss = tl
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= 7:
                print(f'    early stopping at epoch {epoch+1}')
                break

    if best_state:
        model.load_state_dict(best_state)
    model.to(device).eval()

    # Test
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

    # Unfreeze for next round
    for p in model.parameters():
        p.requires_grad = True

    return mse, mae


if __name__ == '__main__':
    device = torch.device('cuda')

    # Real+Synth checkpoint
    ckpt_path = 'checkpoints/real_synth_combined.pth'
    if not os.path.exists(ckpt_path):
        print(f'ERROR: {ckpt_path} not found!')
        sys.exit(1)

    args = SimpleNamespace(
        seq_len=96, pred_len=96, use_norm=True,
        deeponet_width=128, n_experts=4, branch_depth=4, trunk_depth=2,
        activation='gelu', dropout=0.1, branch_hidden=512,
        spectral_branch=True, skip_mode='none',
        use_cross_channel=False, trunk_basis='mixed',
        encoder_type='patch_attn', loss='MSE',
    )
    model = Model(args).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device), strict=False)
    print(f'Loaded Real+Synth checkpoint')
    print(f'Forecast head training (frozen backbone, lr=0.0005, patience=7)')

    datasets = {
        'ETTh1': ('ETTh1', './dataset/ETT-small/', 'ETTh1.csv', 7),
        'ETTh2': ('ETTh2', './dataset/ETT-small/', 'ETTh2.csv', 7),
        'ETTm1': ('ETTm1', './dataset/ETT-small/', 'ETTm1.csv', 7),
        'ETTm2': ('ETTm2', './dataset/ETT-small/', 'ETTm2.csv', 7),
        'Weather': ('custom', './dataset/weather/', 'weather.csv', 21),
        'Exchange': ('custom', './dataset/exchange_rate/', 'exchange_rate.csv', 8),
    }

    moment_lp = {
        'ETTh1_pl96': 0.387, 'ETTh1_pl192': 0.410, 'ETTh1_pl336': 0.422, 'ETTh1_pl720': 0.454,
        'ETTh2_pl96': 0.288, 'ETTh2_pl192': 0.349, 'ETTh2_pl336': 0.369, 'ETTh2_pl720': 0.403,
        'ETTm1_pl96': 0.293, 'ETTm1_pl192': 0.326, 'ETTm1_pl336': 0.352, 'ETTm1_pl720': 0.405,
        'ETTm2_pl96': 0.170, 'ETTm2_pl192': 0.227, 'ETTm2_pl336': 0.275, 'ETTm2_pl720': 0.363,
        'Weather_pl96': 0.154, 'Weather_pl192': 0.197, 'Weather_pl336': 0.246, 'Weather_pl720': 0.315,
    }

    all_results = {}
    print(f'\n{"="*60}')
    print('Forecasting: Real+Synth backbone + forecast_head training')
    print(f'{"="*60}')

    for dname, (data, root, fpath, enc_in) in datasets.items():
        for pl in [96, 192, 336, 720]:
            print(f'\n--- {dname} pl={pl} ---')
            # Reload checkpoint each time (fresh head)
            model.load_state_dict(torch.load(ckpt_path, map_location=device), strict=False)
            mse, mae = train_forecast_head(model, device, data, root, fpath, enc_in, pl)

            key = f'{dname}_pl{pl}'
            all_results[key] = {'mse': mse, 'mae': mae}
            mlp = moment_lp.get(key, '-')
            mlp_str = f'{mlp:.3f}' if isinstance(mlp, float) else mlp
            print(f'  {key}: MSE={mse:.4f} MAE={mae:.4f}  (MOMENT_LP={mlp_str})')

    print(f'\n{"="*60}')
    print('SUMMARY: Real+Synth + forecast_head LP')
    print(f'{"="*60}')
    print(f'{"Dataset":<20} {"Ours":>8} {"MOMENT_LP":>10}')
    print('-' * 40)
    for k, v in all_results.items():
        mlp = moment_lp.get(k, '-')
        mlp_str = f'{mlp:.3f}' if isinstance(mlp, float) else mlp
        print(f'  {k:<20} {v["mse"]:>8.4f} {mlp_str:>10}')

    # ============================================================
    # GIFT-eval
    # ============================================================
    print(f'\n{"="*60}')
    print('GIFT-eval (zero-shot probabilistic forecasting)')
    print(f'{"="*60}')

    gift_eval_script = '/workspace/gift-eval/eval_deeponet.py'
    if os.path.exists(gift_eval_script):
        import subprocess
        gpu_id = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
        tag = 'realsynth_head'
        cmd = [
            sys.executable, gift_eval_script,
            '--checkpoint', ckpt_path,
            '--device', f'cuda:0',
            '--output_tag', tag,
            '--model_type', 'DeepONetHyperMoE',
            '--spectral_branch',
            '--branch_depth', '4', '--trunk_depth', '2',
            '--deeponet_width', '128', '--branch_hidden', '512',
            '--n_experts', '4',
            '--loss', 'MSE',
            '--seq_len', '96', '--pred_len', '96',
        ]
        print(f'  Running: {" ".join(cmd[-8:])} ...')
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
        print(result.stdout[-2000:] if result.stdout else '')
        if result.returncode != 0:
            print(f'  GIFT-eval failed: {result.stderr[-500:]}')
        else:
            print(f'  GIFT-eval done! Results: /workspace/gift-eval/results/DeepONetHyper/all_results_{tag}.csv')
    else:
        print(f'  SKIP: {gift_eval_script} not found')
