"""
32.5M eval with 3 LVR methods for forecast improvement

Method 1: Post-hoc LVR at inference (no retraining)
Method 3: LP with LVR head (freeze encoder, train LP)

Usage:
  CUDA_VISIBLE_DEVICES=X python experiments/eval_32M_methods.py --seq 96 --tag orig --ckpt full_scale_run
  CUDA_VISIBLE_DEVICES=X python experiments/eval_32M_methods.py --seq 192 --tag seq192 --ckpt 32M_seq192
  CUDA_VISIBLE_DEVICES=X python experiments/eval_32M_methods.py --seq 384 --tag seq384 --ckpt 32M_seq384
"""
import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from types import SimpleNamespace
from torch import optim
from data_provider.data_factory import data_provider

DEVICE = torch.device(os.environ.get('CUDA_DEV', 'cuda'))

MOMENT_LP = {
    'ETTh1_96':0.387,'ETTh1_192':0.410,'ETTh1_336':0.422,'ETTh1_720':0.454,
    'ETTh2_96':0.288,'ETTh2_192':0.349,'ETTh2_336':0.369,'ETTh2_720':0.403,
    'ETTm1_96':0.293,'ETTm1_192':0.326,'ETTm1_336':0.352,'ETTm1_720':0.405,
    'ETTm2_96':0.170,'ETTm2_192':0.227,'ETTm2_336':0.275,'ETTm2_720':0.363,
    'Weather_96':0.154,'Weather_192':0.197,'Weather_336':0.246,'Weather_720':0.315,
}
MOMENT_IMP = {'ETTh1':(0.402,0.139),'ETTh2':(0.125,0.061),
              'ETTm1':(0.202,0.074),'ETTm2':(0.078,0.031),'Weather':(0.082,0.035)}

DATASETS_FC = {
    'ETTh1': ('ETTh1','./dataset/ETT-small/','ETTh1.csv',7),
    'ETTh2': ('ETTh2','./dataset/ETT-small/','ETTh2.csv',7),
    'ETTm1': ('ETTm1','./dataset/ETT-small/','ETTm1.csv',7),
    'ETTm2': ('ETTm2','./dataset/ETT-small/','ETTm2.csv',7),
    'Weather': ('custom','./dataset/weather/','weather.csv',21),
    'Exchange': ('custom','./dataset/exchange_rate/','exchange_rate.csv',8),
}


def get_args(seq_len, dn, d, root, f, enc_in, pl):
    return SimpleNamespace(seq_len=seq_len, pred_len=pl, label_len=48, data=d,
        root_path=root, data_path=f, features='M', target='OT',
        freq='h', embed='timeF', enc_in=enc_in, dec_in=enc_in, c_out=enc_in,
        num_workers=2, batch_size=32, exp_name='MTSF', ordered_data=False,
        data_amount=-1, combine_Gaussian_datasets=False,
        synthetic_data_path='', synthetic_root_path='./',
        synthetic_length=1024, stride=-1)


# ============================================================
# Per-channel forecast (shared utility)
# ============================================================
@torch.no_grad()
def forecast_channel(model, x_ch, target_pred_len, seq_len, use_lvr=False):
    """x_ch: [B, S] raw single channel. Returns [B, target_pred_len]."""
    B = x_ch.shape[0]
    if x_ch.shape[1] >= seq_len:
        x_ctx = x_ch[:, -seq_len:]
    else:
        x_ctx = F.pad(x_ch, (seq_len - x_ch.shape[1], 0))

    # Normalize
    m = x_ctx.mean(dim=1, keepdim=True)
    s = x_ctx.std(dim=1, keepdim=True).clamp(min=1e-6)
    x_n = ((x_ctx - m) / s).clamp(-10, 10)

    # Iterative roll-out in normalized space
    cur = x_n
    chunks = []
    remain = target_pred_len
    while remain > 0:
        step = min(seq_len, remain)
        t = 1.0 + torch.arange(step, device=cur.device, dtype=torch.float32) / seq_len
        t = t.unsqueeze(0).expand(B, -1)
        pred_n = model.forward_train(cur, t)
        chunks.append(pred_n)
        if remain > step:
            cur = torch.cat([cur[:, step:], pred_n], dim=1)
        remain -= step

    pred_n_full = torch.cat(chunks, dim=1)
    pred = pred_n_full * s + m

    # Method 1: Post-hoc LVR
    if use_lvr:
        last = x_ch[:, -1:]  # original last value
        shift = last - pred[:, 0:1]
        pred = pred + shift

    return pred


def forecast_mv(model, x_enc, target_pred_len, seq_len, use_lvr=False):
    B, S, C = x_enc.shape
    outs = []
    for ch in range(C):
        outs.append(forecast_channel(model, x_enc[:, :, ch], target_pred_len, seq_len, use_lvr))
    return torch.stack(outs, dim=-1)


# ============================================================
# Method 0: Standard ZS forecast (baseline)
# Method 1: ZS forecast + post-hoc LVR
# ============================================================
def eval_forecast(model, seq_len, methods=['standard', 'lvr']):
    results = {}
    for method in methods:
        use_lvr = method == 'lvr'
        print(f'\n{"="*60}\nForecast ZS {"+ LVR" if use_lvr else "(standard)"}\n{"="*60}')
        res = {}
        for dn, (d, root, f, enc_in) in DATASETS_FC.items():
            for pl in [96, 192, 336, 720]:
                try:
                    a = get_args(seq_len, dn, d, root, f, enc_in, pl)
                    _, tdl = data_provider(a, 'test')
                    preds, tgts = [], []
                    for bx, by, _, _ in tdl:
                        bx = bx.float().to(DEVICE)
                        p = forecast_mv(model, bx, pl, seq_len, use_lvr)
                        preds.append(p.cpu().numpy())
                        tgts.append(by[:, -pl:, :].numpy())
                    p = np.concatenate(preds); t = np.concatenate(tgts)
                    mse = np.mean((p - t)**2)
                    k = f'{dn}_{pl}'
                    m_lp = MOMENT_LP.get(k)
                    gap = f'{(mse/m_lp-1)*100:+.0f}%' if m_lp else '-'
                    print(f'  {k:<14}: MSE={mse:.4f} | M-LP={m_lp or "N/A"} gap={gap}')
                    res[k] = mse
                except Exception as e:
                    print(f'  {dn}_{pl}: ERROR ({e})')
        results[method] = res
    return results


# ============================================================
# Imputation (standard, no LVR)
# ============================================================
def eval_imputation(model, seq_len):
    print(f'\n{"="*60}\nImputation (standard)\n{"="*60}')
    results = {}
    for dn, (d, root, f, enc_in) in list(DATASETS_FC.items())[:5]:
        try:
            a = get_args(seq_len, dn, d, root, f, enc_in, 96)
            a.pred_len = 96; a.label_len = 0
            _, tdl = data_provider(a, 'test')
            mr_mses = []
            for mr in [0.125, 0.25, 0.375, 0.5]:
                torch.manual_seed(2021)
                preds, tgts, masks = [], [], []
                for bx, by, _, _ in tdl:
                    bx = bx.float().to(DEVICE)
                    B, S, C = bx.shape
                    mk = (torch.rand_like(bx) > mr).float()
                    outs = []
                    for ch in range(C):
                        x_ch = bx[:, :, ch] * mk[:, :, ch]
                        if S > seq_len: x_ch = x_ch[:, -seq_len:]
                        elif S < seq_len: x_ch = F.pad(x_ch, (seq_len - S, 0))
                        mask_ch = mk[:, -seq_len:, ch] if S >= seq_len else F.pad(mk[:, :, ch], (seq_len-S, 0))
                        vis = mask_ch.sum(dim=1, keepdim=True).clamp(min=1)
                        m_v = (x_ch * mask_ch).sum(dim=1, keepdim=True) / vis
                        s_v = (((x_ch - m_v)*mask_ch)**2).sum(dim=1, keepdim=True) / vis
                        s_v = s_v.sqrt().clamp(min=1e-6)
                        x_n = ((x_ch - m_v) / s_v).clamp(-10, 10) * mask_ch
                        t = torch.arange(seq_len, device=DEVICE, dtype=torch.float32) / seq_len
                        t = t.unsqueeze(0).expand(B, -1)
                        with torch.no_grad():
                            rec_n = model.forward_train(x_n, t)
                        rec = rec_n * s_v + m_v
                        outs.append(rec)
                    rec_full = torch.stack(outs, dim=-1)
                    preds.append(rec_full.cpu().numpy())
                    tgts.append(bx.cpu().numpy())
                    masks.append(mk.cpu().numpy())
                p = np.concatenate(preds); t = np.concatenate(tgts); m = np.concatenate(masks)
                mse = np.mean((p[m==0] - t[m==0])**2)
                mr_mses.append(mse)
            avg = np.mean(mr_mses)
            m0, m_lp = MOMENT_IMP.get(dn, (None, None))
            print(f'  {dn:<8}: Avg={avg:.4f} | M_0={m0} LP={m_lp}')
            results[dn] = avg
        except Exception as e:
            print(f'  {dn}: ERROR ({e})')
    return results


# ============================================================
# Method 3: LP with LVR head (forecast only)
# ============================================================
def eval_forecast_lp_lvr(model, seq_len, hidden_dim):
    print(f'\n{"="*60}\nForecast LP + LVR\n{"="*60}')
    results = {}
    model.eval()
    for p in model.parameters(): p.requires_grad = False

    for dn, (d, root, f, enc_in) in DATASETS_FC.items():
        for pl in [96, 192, 336, 720]:
            try:
                a_train = get_args(seq_len, dn, d, root, f, enc_in, pl)
                a_test = get_args(seq_len, dn, d, root, f, enc_in, pl)
                _, dl_train = data_provider(a_train, 'train')
                _, dl_test = data_provider(a_test, 'test')

                # LP head: z → delta (pred_len) per channel, then + last
                head = nn.Linear(hidden_dim, pl).to(DEVICE)
                nn.init.xavier_normal_(head.weight, gain=0.1)
                nn.init.zeros_(head.bias)
                opt = optim.AdamW(head.parameters(), lr=1e-3, weight_decay=0.01)

                # Train LP
                for ep in range(30):
                    head.train()
                    for bx, by, _, _ in dl_train:
                        bx = bx.float().to(DEVICE)
                        by = by.float().to(DEVICE)[:, -pl:, :]
                        B, S, C = bx.shape
                        # Per-channel: encode → LP → delta + last
                        all_pred = []
                        for ch in range(C):
                            x_ch = bx[:, :, ch]
                            if S >= seq_len: x_ctx = x_ch[:, -seq_len:]
                            else: x_ctx = F.pad(x_ch, (seq_len - S, 0))
                            last = x_ch[:, -1:]
                            m = x_ctx.mean(dim=1, keepdim=True)
                            s = x_ctx.std(dim=1, keepdim=True).clamp(min=1e-6)
                            x_n = ((x_ctx - m) / s).clamp(-10, 10)
                            with torch.no_grad():
                                z = model.encoder(x_n)
                            delta = head(z)  # [B, pl] normalized delta
                            pred = delta * s + last  # LVR: last + scaled delta
                            all_pred.append(pred)
                        pred_full = torch.stack(all_pred, dim=-1)  # [B, pl, C]
                        loss = F.mse_loss(pred_full, by)
                        opt.zero_grad(); loss.backward()
                        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
                        opt.step()

                # Eval LP
                head.eval()
                preds, tgts = [], []
                with torch.no_grad():
                    for bx, by, _, _ in dl_test:
                        bx = bx.float().to(DEVICE)
                        by = by.float().to(DEVICE)[:, -pl:, :]
                        B, S, C = bx.shape
                        all_pred = []
                        for ch in range(C):
                            x_ch = bx[:, :, ch]
                            if S >= seq_len: x_ctx = x_ch[:, -seq_len:]
                            else: x_ctx = F.pad(x_ch, (seq_len - S, 0))
                            last = x_ch[:, -1:]
                            m = x_ctx.mean(dim=1, keepdim=True)
                            s = x_ctx.std(dim=1, keepdim=True).clamp(min=1e-6)
                            x_n = ((x_ctx - m) / s).clamp(-10, 10)
                            z = model.encoder(x_n)
                            delta = head(z)
                            pred = delta * s + last
                            all_pred.append(pred)
                        preds.append(torch.stack(all_pred, dim=-1).cpu().numpy())
                        tgts.append(by.cpu().numpy())
                p = np.concatenate(preds); t = np.concatenate(tgts)
                mse = np.mean((p - t)**2)
                k = f'{dn}_{pl}'
                m_lp = MOMENT_LP.get(k)
                gap = f'{(mse/m_lp-1)*100:+.0f}%' if m_lp else '-'
                print(f'  {k:<14}: MSE={mse:.4f} | M-LP={m_lp or "N/A"} gap={gap}')
                results[k] = mse
            except Exception as e:
                print(f'  {dn}_{pl}: ERROR ({e})')

    for p in model.parameters(): p.requires_grad = True
    return results


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq', type=int, required=True)
    parser.add_argument('--tag', type=str, required=True)
    parser.add_argument('--ckpt', type=str, required=True)
    args = parser.parse_args()

    ckpt_path = f'checkpoints/{args.ckpt}.pth'
    print('='*60)
    print(f'32.5M Methods Eval [{args.tag}, SEQ={args.seq}]')
    print(f'Checkpoint: {ckpt_path}')
    print('='*60)

    # Load model
    if args.seq == 96:
        from experiments.exp_full_scale_train import FullScaleModel
        model = FullScaleModel().to(DEVICE)
        hidden = 512
    else:
        from experiments.exp_32M_longctx import Model32MLongCtx
        model = Model32MLongCtx(args.seq).to(DEVICE)
        hidden = 512

    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    n = sum(p.numel() for p in model.parameters())
    print(f'Model: {n/1e6:.1f}M, SEQ={args.seq}')

    # Method 0 + Method 1: ZS forecast (standard + LVR)
    fc_results = eval_forecast(model, args.seq, methods=['standard', 'lvr'])

    # Imputation (no LVR)
    imp_results = eval_imputation(model, args.seq)

    # Method 3: LP + LVR
    lp_results = eval_forecast_lp_lvr(model, args.seq, hidden)

    # Summary
    print('\n' + '='*60)
    print(f'SUMMARY — {args.tag} (SEQ={args.seq})')
    print('='*60)
    for method_name, res in fc_results.items():
        print(f'\nForecast {method_name}:')
        for k in ['ETTh1','ETTh2','ETTm1','ETTm2','Weather','Exchange']:
            avgs = [v for k_, v in res.items() if k_.startswith(k+'_')]
            if avgs: print(f'  {k:<10}: {np.mean(avgs):.4f}')

    print(f'\nForecast LP+LVR:')
    for k in ['ETTh1','ETTh2','ETTm1','ETTm2','Weather','Exchange']:
        avgs = [v for k_, v in lp_results.items() if k_.startswith(k+'_')]
        if avgs: print(f'  {k:<10}: {np.mean(avgs):.4f}')

    print(f'\nImputation (no LVR):')
    for k, v in imp_results.items():
        print(f'  {k:<10}: {v:.4f}')
    print('='*60)
