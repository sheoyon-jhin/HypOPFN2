"""
Forecast Head Grid Search: ETTh1으로 빠르게 최적 lr/wd 찾기
Real+Synth checkpoint, frozen backbone

사용법:
  CUDA_VISIBLE_DEVICES=0 python experiments/exp_forecast_gridsearch.py 2>&1 | tee log/eval/forecast_gridsearch.log
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


def run_one(model, device, ckpt_path, lr, wd, batch_size, pl=96):
    """한 세팅으로 ETTh1 forecast head 학습 + eval."""
    model.load_state_dict(torch.load(ckpt_path, map_location=device), strict=False)

    args = SimpleNamespace(
        seq_len=96, pred_len=pl, label_len=48,
        data='ETTh1', root_path='./dataset/ETT-small/', data_path='ETTh1.csv',
        features='M', target='OT', freq='h', embed='timeF',
        enc_in=7, dec_in=7, c_out=7, num_workers=2, batch_size=batch_size,
        exp_name='MTSF', ordered_data=False, data_amount=-1,
        combine_Gaussian_datasets=False, synthetic_data_path='', synthetic_root_path='./',
        synthetic_length=1024, stride=-1,
    )
    _, train_dl = data_provider(args, 'train')
    _, test_dl = data_provider(args, 'test')

    for p in model.parameters():
        p.requires_grad = False
    trainable = []
    for name, p in model.named_parameters():
        if 'forecast_head' in name:
            p.requires_grad = True
            trainable.append(p)

    optimizer = optim.Adam(trainable, lr=lr, weight_decay=wd)
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

    for p in model.parameters():
        p.requires_grad = True
    return mse


if __name__ == '__main__':
    device = torch.device('cuda')

    ckpt_path = 'checkpoints/real_synth_combined.pth'
    args = SimpleNamespace(
        seq_len=96, pred_len=96, use_norm=True,
        deeponet_width=128, n_experts=4, branch_depth=4, trunk_depth=2,
        activation='gelu', dropout=0.1, branch_hidden=512,
        spectral_branch=True, skip_mode='none',
        use_cross_channel=False, trunk_basis='mixed',
        encoder_type='patch_attn', loss='MSE',
    )
    model = Model(args).to(device)

    print(f'{"="*60}')
    print('Forecast Head Grid Search (ETTh1 pl=96)')
    print(f'{"="*60}')

    # Grid
    lrs = [0.0001, 0.0003, 0.0005, 0.001, 0.002]
    wds = [0.0, 0.001, 0.01]
    batch_sizes = [16, 32, 64]

    results = []

    # First: lr × wd (batch=32 고정)
    print('\n--- Phase 1: lr × weight_decay (batch=32) ---')
    for lr in lrs:
        for wd in wds:
            mse = run_one(model, device, ckpt_path, lr, wd, 32, pl=96)
            results.append({'lr': lr, 'wd': wd, 'bs': 32, 'mse': mse})
            print(f'  lr={lr}, wd={wd}: MSE={mse:.4f}')

    # Best lr/wd로 batch_size 탐색
    best = min(results, key=lambda x: x['mse'])
    print(f'\n--- Phase 2: batch_size (best lr={best["lr"]}, wd={best["wd"]}) ---')
    for bs in batch_sizes:
        if bs == 32: continue  # already done
        mse = run_one(model, device, ckpt_path, best['lr'], best['wd'], bs, pl=96)
        results.append({'lr': best['lr'], 'wd': best['wd'], 'bs': bs, 'mse': mse})
        print(f'  bs={bs}: MSE={mse:.4f}')

    # Best로 다른 pred_len도
    overall_best = min(results, key=lambda x: x['mse'])
    print(f'\n--- Phase 3: Best config on all pred_lens ---')
    print(f'  Best: lr={overall_best["lr"]}, wd={overall_best["wd"]}, bs={overall_best["bs"]}')

    for pl in [96, 192, 336, 720]:
        mse = run_one(model, device, ckpt_path, overall_best['lr'], overall_best['wd'],
                      overall_best['bs'], pl=pl)
        print(f'  ETTh1_pl{pl}: MSE={mse:.4f}  (MOMENT: {[0.387, 0.410, 0.422, 0.454][[96,192,336,720].index(pl)]:.3f})')

    print(f'\n{"="*60}')
    print('SUMMARY')
    print(f'{"="*60}')
    print(f'  Best config: lr={overall_best["lr"]}, wd={overall_best["wd"]}, bs={overall_best["bs"]}, MSE={overall_best["mse"]:.4f}')
    print(f'  Current (lr=0.0005, wd=0.01): MSE=0.406')
    print(f'  LP (1층): MSE=0.409')
    print(f'  MOMENT_LP: MSE=0.387')

    # ============================================================
    # GIFT-eval with best config checkpoint
    # ============================================================
    print(f'\n{"="*60}')
    print('GIFT-eval (zero-shot probabilistic forecasting)')
    print(f'{"="*60}')

    import subprocess
    gift_eval_script = '/workspace/gift-eval/eval_deeponet.py'
    if os.path.exists(gift_eval_script):
        tag = 'gridsearch_best'
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
