"""
Grid Search Best Config → 전체 Forecasting 데이터셋 평가
Best: lr=0.0001, wd=0.001, bs=64 (ETTh1 grid search에서 발견)

사용법:
  CUDA_VISIBLE_DEVICES=0 python experiments/exp_forecast_gridsearch_full.py 2>&1 | tee log/eval/forecast_gridsearch_full.log
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
from model.DeepONetHyperMoE import Model


# Grid search best config
BEST_LR = 0.0001
BEST_WD = 0.001
BEST_BS = 64


def train_and_eval(model, device, ckpt_path, data, root, fpath, enc_in, pl):
    """Best config로 forecast head 학습 + eval."""
    model.load_state_dict(torch.load(ckpt_path, map_location=device), strict=False)

    args = SimpleNamespace(
        seq_len=96, pred_len=pl, label_len=48,
        data=data, root_path=root, data_path=fpath,
        features='M', target='OT', freq='h', embed='timeF',
        enc_in=enc_in, dec_in=enc_in, c_out=enc_in,
        num_workers=2, batch_size=BEST_BS,
        exp_name='MTSF', ordered_data=False, data_amount=-1,
        combine_Gaussian_datasets=False, synthetic_data_path='', synthetic_root_path='./',
        synthetic_length=1024, stride=-1,
    )
    _, train_dl = data_provider(args, 'train')
    _, test_dl = data_provider(args, 'test')

    # Freeze all, unfreeze forecast_head only
    for p in model.parameters():
        p.requires_grad = False
    trainable = []
    for name, p in model.named_parameters():
        if 'forecast_head' in name:
            p.requires_grad = True
            trainable.append(p)

    optimizer = optim.Adam(trainable, lr=BEST_LR, weight_decay=BEST_WD)
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
        if tl < best_loss:
            best_loss = tl
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= 7: break

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

    for p in model.parameters():
        p.requires_grad = True
    return mse, mae


if __name__ == '__main__':
    device = torch.device('cuda')

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

    datasets = {
        'ETTh1': ('ETTh1', './dataset/ETT-small/', 'ETTh1.csv', 7),
        'ETTh2': ('ETTh2', './dataset/ETT-small/', 'ETTh2.csv', 7),
        'ETTm1': ('ETTm1', './dataset/ETT-small/', 'ETTm1.csv', 7),
        'ETTm2': ('ETTm2', './dataset/ETT-small/', 'ETTm2.csv', 7),
        'Weather': ('custom', './dataset/weather/', 'weather.csv', 21),
        'Exchange': ('custom', './dataset/exchange_rate/', 'exchange_rate.csv', 8),
    }

    # MOMENT LP baselines
    moment_lp = {
        'ETTh1_96': 0.387, 'ETTh1_192': 0.410, 'ETTh1_336': 0.422, 'ETTh1_720': 0.454,
        'ETTh2_96': 0.288, 'ETTh2_192': 0.349, 'ETTh2_336': 0.369, 'ETTh2_720': 0.403,
        'ETTm1_96': 0.293, 'ETTm1_192': 0.326, 'ETTm1_336': 0.352, 'ETTm1_720': 0.405,
        'ETTm2_96': 0.170, 'ETTm2_192': 0.227, 'ETTm2_336': 0.275, 'ETTm2_720': 0.363,
        'Weather_96': 0.154, 'Weather_192': 0.197, 'Weather_336': 0.246, 'Weather_720': 0.315,
    }

    # Old config (lr=0.0005, wd=0.01) baselines for comparison
    old_config = {
        'ETTh1_96': 0.406, 'ETTh2_96': 0.295, 'ETTm1_96': 0.379, 'ETTm2_96': 0.188,
    }

    all_results = {}
    print(f'{"="*70}')
    print(f'Grid Search Best → Full Forecasting Eval')
    print(f'Config: lr={BEST_LR}, wd={BEST_WD}, bs={BEST_BS}')
    print(f'Checkpoint: {ckpt_path}')
    print(f'{"="*70}')

    for dname, (data, root, fpath, enc_in) in datasets.items():
        for pl in [96, 192, 336, 720]:
            print(f'\n--- {dname} pl={pl} ---')
            mse, mae = train_and_eval(model, device, ckpt_path, data, root, fpath, enc_in, pl)

            key = f'{dname}_{pl}'
            all_results[key] = {'mse': mse, 'mae': mae}

            mlp = moment_lp.get(key, None)
            old = old_config.get(key, None)
            mlp_str = f'{mlp:.3f}' if mlp else '-'
            old_str = f'{old:.3f}' if old else '-'
            gap_str = f'{(mse/mlp - 1)*100:+.1f}%' if mlp else '-'
            print(f'  {key}: MSE={mse:.4f} MAE={mae:.4f}  MOMENT_LP={mlp_str} Gap={gap_str}  Old={old_str}')

    # Summary table
    print(f'\n{"="*70}')
    print(f'SUMMARY: Grid Search Best (lr={BEST_LR}, wd={BEST_WD}, bs={BEST_BS})')
    print(f'{"="*70}')
    print(f'{"Dataset":<18} {"Ours":>8} {"Old cfg":>8} {"MOMENT_LP":>10} {"Gap":>8}')
    print('-' * 56)

    for dname in datasets:
        for pl in [96, 192, 336, 720]:
            key = f'{dname}_{pl}'
            v = all_results.get(key)
            if not v: continue
            mlp = moment_lp.get(key, None)
            old = old_config.get(key, None)
            mlp_str = f'{mlp:.3f}' if mlp else '-'
            old_str = f'{old:.3f}' if old else '-'
            gap_str = f'{(v["mse"]/mlp - 1)*100:+.1f}%' if mlp else '-'
            print(f'  {key:<18} {v["mse"]:>8.4f} {old_str:>8} {mlp_str:>10} {gap_str:>8}')
        # Dataset average
        keys = [f'{dname}_{pl}' for pl in [96, 192, 336, 720]]
        vals = [all_results[k]['mse'] for k in keys if k in all_results]
        mlps = [moment_lp[k] for k in keys if k in moment_lp]
        if vals:
            avg = np.mean(vals)
            avg_mlp = np.mean(mlps) if mlps else None
            avg_gap = f'{(avg/avg_mlp - 1)*100:+.1f}%' if avg_mlp else '-'
            print(f'  {"["+dname+" avg]":<18} {avg:>8.4f} {"":>8} {np.mean(mlps):>10.3f} {avg_gap:>8}' if avg_mlp else f'  {"["+dname+" avg]":<18} {avg:>8.4f}')

    # GIFT-eval
    print(f'\n{"="*70}')
    print('GIFT-eval (zero-shot probabilistic forecasting)')
    print(f'{"="*70}')

    import subprocess
    gift_eval_script = '/workspace/gift-eval/eval_deeponet.py'
    if os.path.exists(gift_eval_script):
        tag = 'gridsearch_best_full'
        cmd = [
            sys.executable, gift_eval_script,
            '--checkpoint', ckpt_path,
            '--device', 'cuda:0',
            '--output_tag', tag,
            '--model_type', 'DeepONetHyperMoE',
            '--spectral_branch',
            '--branch_depth', '4', '--trunk_depth', '2',
            '--deeponet_width', '128', '--branch_hidden', '512',
            '--n_experts', '4',
            '--loss', 'MSE',
            '--seq_len', '96', '--pred_len', '96',
        ]
        print(f'  Running GIFT-eval with tag={tag} ...')
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
        print(result.stdout[-2000:] if result.stdout else '')
        if result.returncode != 0:
            print(f'  GIFT-eval failed: {result.stderr[-500:]}')
        else:
            print(f'  GIFT-eval done! Results: /workspace/gift-eval/results/DeepONetHyper/all_results_{tag}.csv')
    else:
        print(f'  SKIP: {gift_eval_script} not found')
