"""
Full Fine-tune: 32.5M seq96 → 각 target dataset에 5-10 epoch fine-tune

LP (linear head만 학습)보다 훨씬 강력:
  - Encoder도 target dataset에 적응
  - 작은 lr (1e-5)로 살짝만 update → catastrophic forgetting 방지

Protocol: FeDaL Table 3과 동일 ("Full-shot")

Usage:
  CUDA_VISIBLE_DEVICES=2 python experiments/exp_full_finetune.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from torch import optim
from types import SimpleNamespace

from data_provider.data_factory import data_provider
from experiments.exp_full_scale_train import FullScaleModel, SEQ_LEN, HIDDEN

DEVICE = torch.device(os.environ.get('CUDA_DEV', 'cuda'))
CKPT = 'checkpoints/full_scale_run.pth'


DATASETS = {
    'ETTh1': ('ETTh1', './dataset/ETT-small/', 'ETTh1.csv', 7),
    'ETTh2': ('ETTh2', './dataset/ETT-small/', 'ETTh2.csv', 7),
    'ETTm1': ('ETTm1', './dataset/ETT-small/', 'ETTm1.csv', 7),
    'ETTm2': ('ETTm2', './dataset/ETT-small/', 'ETTm2.csv', 7),
    'Weather': ('custom', './dataset/weather/', 'weather.csv', 21),
    'Exchange': ('custom', './dataset/exchange_rate/', 'exchange_rate.csv', 8),
}
MOMENT_LP = {
    'ETTh1_96':0.387,'ETTh1_192':0.410,'ETTh1_336':0.422,'ETTh1_720':0.454,
    'ETTh2_96':0.288,'ETTh2_192':0.349,'ETTh2_336':0.369,'ETTh2_720':0.403,
    'ETTm1_96':0.293,'ETTm1_192':0.326,'ETTm1_336':0.352,'ETTm1_720':0.405,
    'ETTm2_96':0.170,'ETTm2_192':0.227,'ETTm2_336':0.275,'ETTm2_720':0.363,
    'Weather_96':0.154,'Weather_192':0.197,'Weather_336':0.246,'Weather_720':0.315,
}
FEDAL = {
    'ETTh1':0.407, 'ETTh2':0.361, 'ETTm1':0.360, 'ETTm2':0.292, 'Weather':0.255,
}


@torch.no_grad()
def forecast_channel(model, x_ch, target_pred_len):
    """Per-channel forecast with normalization."""
    B = x_ch.shape[0]
    if x_ch.shape[1] >= SEQ_LEN:
        x_ctx = x_ch[:, -SEQ_LEN:]
    else:
        x_ctx = F.pad(x_ch, (SEQ_LEN - x_ch.shape[1], 0))
    m = x_ctx.mean(dim=1, keepdim=True)
    s = x_ctx.std(dim=1, keepdim=True).clamp(min=1e-6)
    x_n = ((x_ctx - m) / s).clamp(-10, 10)

    # Iterative roll-out
    cur = x_n
    chunks = []
    remain = target_pred_len
    while remain > 0:
        step = min(SEQ_LEN, remain)
        pred_n = model.forecast(cur, n=step)
        chunks.append(pred_n)
        if remain > step:
            cur = torch.cat([cur[:, step:], pred_n], dim=1)
        remain -= step
    pred_n_full = torch.cat(chunks, dim=1)
    return pred_n_full * s + m


def forecast_channel_train(model, x_ch, target_pred_len):
    """Same but with gradients for fine-tuning."""
    B = x_ch.shape[0]
    if x_ch.shape[1] >= SEQ_LEN:
        x_ctx = x_ch[:, -SEQ_LEN:]
    else:
        x_ctx = F.pad(x_ch, (SEQ_LEN - x_ch.shape[1], 0))
    m = x_ctx.mean(dim=1, keepdim=True).detach()
    s = x_ctx.std(dim=1, keepdim=True).clamp(min=1e-6).detach()
    x_n = ((x_ctx - m) / s).clamp(-10, 10)
    pred_n = model.forecast(x_n, n=min(SEQ_LEN, target_pred_len))
    return pred_n * s + m


def eval_dataset(model, dn, d, root, f, enc_in, pl):
    """Evaluate on test set."""
    a = SimpleNamespace(seq_len=SEQ_LEN, pred_len=pl, label_len=48, data=d,
        root_path=root, data_path=f, features='M', target='OT', freq='h',
        embed='timeF', enc_in=enc_in, dec_in=enc_in, c_out=enc_in,
        num_workers=2, batch_size=32, exp_name='MTSF', ordered_data=False,
        data_amount=-1, combine_Gaussian_datasets=False, synthetic_data_path='',
        synthetic_root_path='./', synthetic_length=1024, stride=-1)
    _, tdl = data_provider(a, 'test')

    model.eval()
    preds, tgts = [], []
    with torch.no_grad():
        for bx, by, _, _ in tdl:
            bx = bx.float().to(DEVICE)
            B, S, C = bx.shape
            outs = []
            for ch in range(C):
                outs.append(forecast_channel(model, bx[:, :, ch], pl))
            preds.append(torch.stack(outs, dim=-1).cpu().numpy())
            tgts.append(by[:, -pl:, :].numpy())
    p = np.concatenate(preds); t = np.concatenate(tgts)
    mse = np.mean((p - t)**2)
    mae = np.mean(np.abs(p - t))
    return mse, mae


def finetune_and_eval(base_ckpt, dn, d, root, f, enc_in, pl,
                       ft_epochs=5, ft_lr=1e-5):
    """Load base model, fine-tune on train split, eval on test split."""
    # Load fresh copy
    model = FullScaleModel().to(DEVICE)
    model.load_state_dict(torch.load(base_ckpt, map_location=DEVICE))

    # Get train loader
    a = SimpleNamespace(seq_len=SEQ_LEN, pred_len=pl, label_len=48, data=d,
        root_path=root, data_path=f, features='M', target='OT', freq='h',
        embed='timeF', enc_in=enc_in, dec_in=enc_in, c_out=enc_in,
        num_workers=2, batch_size=32, exp_name='MTSF', ordered_data=False,
        data_amount=-1, combine_Gaussian_datasets=False, synthetic_data_path='',
        synthetic_root_path='./', synthetic_length=1024, stride=-1)
    _, dl_train = data_provider(a, 'train')

    # Fine-tune: all params, small lr
    optimizer = optim.AdamW(model.parameters(), lr=ft_lr, weight_decay=0.01)

    for ep in range(ft_epochs):
        model.train()
        epoch_loss = []
        for bx, by, _, _ in dl_train:
            bx = bx.float().to(DEVICE)
            by = by.float().to(DEVICE)[:, -pl:, :]
            B, S, C = bx.shape

            # Per-channel forecast (single step, no roll-out for training stability)
            step = min(SEQ_LEN, pl)
            all_pred = []
            for ch in range(C):
                pred = forecast_channel_train(model, bx[:, :, ch], step)
                all_pred.append(pred)
            pred_full = torch.stack(all_pred, dim=-1)

            # MSE on first `step` positions
            loss = F.mse_loss(pred_full, by[:, :step, :])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss.append(loss.item())

    # Eval
    mse, mae = eval_dataset(model, dn, d, root, f, enc_in, pl)
    return mse, mae


if __name__ == '__main__':
    print('='*60)
    print('Full Fine-tune: 32.5M seq96 → target datasets')
    print(f'  Base: {CKPT}')
    print(f'  FT epochs: 5, lr: 1e-5')
    print('='*60)

    results = {}
    for dn, (d, root, f, enc_in) in DATASETS.items():
        for pl in [96, 192, 336, 720]:
            try:
                mse, mae = finetune_and_eval(CKPT, dn, d, root, f, enc_in, pl,
                                              ft_epochs=5, ft_lr=1e-5)
                k = f'{dn}_{pl}'
                m_lp = MOMENT_LP.get(k)
                gap = f'{(mse/m_lp-1)*100:+.0f}%' if m_lp else '-'
                print(f'  {k:<14}: MSE={mse:.4f} MAE={mae:.4f} | M-LP={m_lp or "N/A"} gap={gap}')
                results[k] = mse
            except Exception as e:
                print(f'  {dn}_{pl}: ERROR ({e})')

    # Summary
    print('\n' + '='*60)
    print('SUMMARY — Full Fine-tune')
    print('='*60)
    for k in ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'Weather', 'Exchange']:
        avgs = [v for k_, v in results.items() if k_.startswith(k + '_')]
        if avgs:
            avg = np.mean(avgs)
            fed = FEDAL.get(k)
            fed_str = f' | FeDaL={fed} gap={(avg/fed-1)*100:+.0f}%' if fed else ''
            print(f'  {k:<10}: {avg:.4f}{fed_str}')

    # Overall
    five = []
    for k in ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'Weather']:
        avgs = [v for k_, v in results.items() if k_.startswith(k + '_')]
        if avgs: five.append(np.mean(avgs))
    if five:
        avg5 = np.mean(five)
        print(f'\n  ** 5-dataset Avg: {avg5:.4f} **')
        print(f'  ** Our LP (no FT): 0.355 **')
        print(f'  ** FeDaL: 0.335 **')
        print(f'  ** MOMENT-LP: 0.320 **')
    print('='*60)
