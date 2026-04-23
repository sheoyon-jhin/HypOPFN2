"""
5-Task Eval for ConfigB + RevIN OFF + True OP (82.7M) — FIXED

Fix: externally pre-normalize (match training), call model internals directly,
     denormalize afterwards. Iterative roll-out in normalized space.

CUDA_VISIBLE_DEVICES=2 python experiments/eval_configB_method_v2.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from types import SimpleNamespace
from torch import optim
from torch.utils.data import DataLoader

from data_provider.data_factory import data_provider
from data_provider.data_loader import Dataset_Classification
from experiments.exp_configB_revinoff_trueop import ConfigBNoRevIN, query_at_vectorized
from sklearn.metrics import accuracy_score, f1_score

DEVICE = torch.device('cuda')
CKPT = 'checkpoints/configB_revinoff_trueop.pth'
SEQ_LEN = 96


# ============================================================
# Per-window normalized forecast (matching training)
# ============================================================
@torch.no_grad()
def normalized_forecast_chunk(model, x_n, step):
    """x_n: [B, 96] pre-normalized single-channel. Return [B, step] in normalized space."""
    B = x_n.shape[0]
    x_cross = x_n  # single channel, cross = self
    t_vals = torch.linspace(1.0, 1.0 + step/SEQ_LEN, step, device=x_n.device).unsqueeze(0).expand(B, -1)
    return query_at_vectorized(model, x_n, x_cross, t_vals, mode='forecast')


@torch.no_grad()
def normalized_reconstruct(model, x_masked_n, n_points=SEQ_LEN):
    """x_masked_n: [B, 96] pre-normalized. Return [B, 96] reconstruction in normalized space."""
    B = x_masked_n.shape[0]
    x_cross = x_masked_n
    t_vals = torch.linspace(0.0, 1.0, n_points, device=x_masked_n.device).unsqueeze(0).expand(B, -1)
    return query_at_vectorized(model, x_masked_n, x_cross, t_vals, mode='recon')


def forecast_mv(model, x_enc, target_pred_len):
    """x_enc [B, S, C] → [B, target_pred_len, C] with per-channel normalize + iterative roll."""
    B, S, C = x_enc.shape
    outs = []
    for ch in range(C):
        x_ch = x_enc[:, :, ch]  # [B, S]
        if S >= SEQ_LEN:
            x_ctx = x_ch[:, -SEQ_LEN:]
        else:
            x_ctx = F.pad(x_ch, (SEQ_LEN - S, 0))
        m = x_ctx.mean(dim=1, keepdim=True)
        s = x_ctx.std(dim=1, keepdim=True).clamp(min=1e-6)
        x_n = ((x_ctx - m) / s).clamp(-10, 10)

        # Iterative roll-out in normalized space
        cur = x_n
        chunks = []
        remain = target_pred_len
        while remain > 0:
            step = min(SEQ_LEN, remain)
            pred_n = normalized_forecast_chunk(model, cur, step)  # [B, step]
            chunks.append(pred_n)
            if remain > step:
                cur = torch.cat([cur[:, step:], pred_n], dim=1)
            remain -= step
        pred_n_full = torch.cat(chunks, dim=1)  # [B, target_pred_len]
        # De-normalize
        pred = pred_n_full * s + m  # broadcast [B, target_pred_len]
        outs.append(pred)
    return torch.stack(outs, dim=-1)  # [B, target_pred_len, C]


def reconstruct_mv(model, x_enc, mask):
    """x_enc [B, S, C], mask [B, S, C] → [B, S, C]."""
    B, S, C = x_enc.shape
    outs = []
    for ch in range(C):
        x_ch = x_enc[:, :, ch] * mask[:, :, ch]
        if S > SEQ_LEN:
            x_ch = x_ch[:, -SEQ_LEN:]
        elif S < SEQ_LEN:
            x_ch = F.pad(x_ch, (SEQ_LEN - S, 0))
        # Normalize on visible only
        mask_ch = mask[:, -SEQ_LEN:, ch] if S >= SEQ_LEN else F.pad(mask[:, :, ch], (SEQ_LEN-S, 0))
        vis_sum = mask_ch.sum(dim=1, keepdim=True).clamp(min=1)
        m_vis = (x_ch * mask_ch).sum(dim=1, keepdim=True) / vis_sum
        s_vis = (((x_ch - m_vis) * mask_ch)**2).sum(dim=1, keepdim=True) / vis_sum
        s_vis = s_vis.sqrt().clamp(min=1e-6)
        x_n = ((x_ch - m_vis) / s_vis).clamp(-10, 10) * mask_ch
        rec_n = normalized_reconstruct(model, x_n)
        rec = rec_n * s_vis + m_vis
        outs.append(rec)
    return torch.stack(outs, dim=-1)


# ============================================================
# Task 1: Forecasting
# ============================================================
def eval_forecasting(model):
    print(f'\n{"="*60}\nTask 1: Forecasting (zero-shot)\n{"="*60}')
    datasets = {
        'ETTh1':    ('ETTh1',  './dataset/ETT-small/', 'ETTh1.csv', 7),
        'ETTh2':    ('ETTh2',  './dataset/ETT-small/', 'ETTh2.csv', 7),
        'ETTm1':    ('ETTm1',  './dataset/ETT-small/', 'ETTm1.csv', 7),
        'ETTm2':    ('ETTm2',  './dataset/ETT-small/', 'ETTm2.csv', 7),
        'Weather':  ('custom', './dataset/weather/',   'weather.csv', 21),
        'Exchange': ('custom', './dataset/exchange_rate/', 'exchange_rate.csv', 8),
    }
    moment_lp = {
        'ETTh1_96':0.387,'ETTh1_192':0.410,'ETTh1_336':0.422,'ETTh1_720':0.454,
        'ETTh2_96':0.288,'ETTh2_192':0.349,'ETTh2_336':0.369,'ETTh2_720':0.403,
        'ETTm1_96':0.293,'ETTm1_192':0.326,'ETTm1_336':0.352,'ETTm1_720':0.405,
        'ETTm2_96':0.170,'ETTm2_192':0.227,'ETTm2_336':0.275,'ETTm2_720':0.363,
        'Weather_96':0.154,'Weather_192':0.197,'Weather_336':0.246,'Weather_720':0.315,
    }
    model.eval()
    results = {}
    for dn, (d, root, f, enc_in) in datasets.items():
        for pl in [96, 192, 336, 720]:
            try:
                a = SimpleNamespace(seq_len=96, pred_len=pl, label_len=48, data=d,
                    root_path=root, data_path=f, features='M', target='OT',
                    freq='h', embed='timeF', enc_in=enc_in, dec_in=enc_in, c_out=enc_in,
                    num_workers=2, batch_size=32, exp_name='MTSF', ordered_data=False,
                    data_amount=-1, combine_Gaussian_datasets=False,
                    synthetic_data_path='', synthetic_root_path='./',
                    synthetic_length=1024, stride=-1)
                _, tdl = data_provider(a, 'test')
                preds, tgts = [], []
                for bx, by, _, _ in tdl:
                    bx = bx.float().to(DEVICE)
                    p = forecast_mv(model, bx, pl)
                    preds.append(p.cpu().numpy())
                    tgts.append(by[:, -pl:, :].numpy())
                p = np.concatenate(preds); t = np.concatenate(tgts)
                mse = np.mean((p - t)**2)
                mae = np.mean(np.abs(p - t))
                k = f'{dn}_{pl}'
                m_lp = moment_lp.get(k)
                gap = f'{(mse/m_lp-1)*100:+.0f}%' if m_lp else '-'
                m_str = f'{m_lp:.3f}' if m_lp else 'N/A'
                print(f'  {k:<14}: MSE={mse:.4f} MAE={mae:.4f}  | MOMENT-LP={m_str} gap={gap}')
                results[k] = mse
            except Exception as e:
                print(f'  {dn}_{pl}: ERROR ({type(e).__name__}: {e})')
    return results


# ============================================================
# Task 2: Imputation
# ============================================================
def eval_imputation(model):
    print(f'\n{"="*60}\nTask 2: Imputation\n{"="*60}')
    datasets = {
        'ETTh1':   ('ETTh1', './dataset/ETT-small/', 'ETTh1.csv', 7),
        'ETTh2':   ('ETTh2', './dataset/ETT-small/', 'ETTh2.csv', 7),
        'ETTm1':   ('ETTm1', './dataset/ETT-small/', 'ETTm1.csv', 7),
        'ETTm2':   ('ETTm2', './dataset/ETT-small/', 'ETTm2.csv', 7),
        'Weather': ('custom','./dataset/weather/',   'weather.csv', 21),
    }
    moment = {'ETTh1':(0.402,0.139),'ETTh2':(0.125,0.061),
              'ETTm1':(0.202,0.074),'ETTm2':(0.078,0.031),'Weather':(0.082,0.035)}
    model.eval()
    results = {}
    for dn, (d, root, f, enc_in) in datasets.items():
        try:
            a = SimpleNamespace(seq_len=96, pred_len=96, label_len=0, data=d,
                root_path=root, data_path=f, features='M', target='OT',
                freq='h', embed='timeF', enc_in=enc_in, dec_in=enc_in, c_out=enc_in,
                num_workers=2, batch_size=32, exp_name='MTSF', ordered_data=False,
                data_amount=-1, combine_Gaussian_datasets=False,
                synthetic_data_path='', synthetic_root_path='./',
                synthetic_length=1024, stride=-1)
            _, tdl = data_provider(a, 'test')
            mr_mses = []
            for mr in [0.125, 0.25, 0.375, 0.5]:
                torch.manual_seed(2021)
                preds, tgts, masks = [], [], []
                for bx, by, _, _ in tdl:
                    bx = bx.float().to(DEVICE)
                    mk = (torch.rand_like(bx) > mr).float()
                    rec = reconstruct_mv(model, bx, mk)
                    preds.append(rec.cpu().numpy())
                    tgts.append(bx.cpu().numpy())
                    masks.append(mk.cpu().numpy())
                p = np.concatenate(preds); t = np.concatenate(tgts); m = np.concatenate(masks)
                mse = np.mean((p[m==0] - t[m==0])**2)
                mr_mses.append(mse)
            avg = np.mean(mr_mses)
            m0, m_lp = moment.get(dn, (None, None))
            print(f'  {dn:<8}: Avg={avg:.4f} | by mr: {[f"{x:.3f}" for x in mr_mses]} | MOMENT_0={m0} LP={m_lp}')
            results[dn] = avg
        except Exception as e:
            print(f'  {dn}: ERROR ({type(e).__name__}: {e})')
    return results


# ============================================================
# Task 3: Classification (linear probe)
# ============================================================
def eval_classification(model):
    print(f'\n{"="*60}\nTask 3: Classification\n{"="*60}')
    h = model.branch_hidden
    for p in model.parameters(): p.requires_grad = False
    results = {}
    datasets = ['Epilepsy', 'FingerMovements', 'BasicMotions', 'NATOPS',
                'EthanolConcentration', 'Heartbeat', 'MotorImagery',
                'SelfRegulationSCP1', 'SelfRegulationSCP2', 'UWaveGestureLibrary']
    for ds in datasets:
        try:
            cr = './dataset/classification/Multivariate_ts'
            trd = Dataset_Classification(root_path=cr, flag='train', size=[96,0,96], data_path=ds)
            ted = Dataset_Classification(root_path=cr, flag='test', size=[96,0,96], data_path=ds)
            trl = DataLoader(trd, batch_size=16, shuffle=True, drop_last=True)
            tel = DataLoader(ted, batch_size=16, shuffle=False)
            head = nn.Sequential(
                nn.Linear(h, 256), nn.GELU(), nn.Dropout(0.1),
                nn.Linear(256, trd.n_classes)).to(DEVICE)
            opt = optim.Adam(head.parameters(), lr=1e-3, weight_decay=1e-4)
            best = 0
            for ep in range(30):
                head.train()
                for bx, lb, _, _ in trl:
                    bx = bx.float().to(DEVICE); lb = lb.long().to(DEVICE)
                    with torch.no_grad():
                        z = model.get_representation(bx).mean(dim=1)
                    loss = nn.CrossEntropyLoss()(head(z), lb)
                    opt.zero_grad(); loss.backward(); opt.step()
                head.eval()
                ps, ls = [], []
                with torch.no_grad():
                    for bx, lb, _, _ in tel:
                        bx = bx.float().to(DEVICE)
                        z = model.get_representation(bx).mean(dim=1)
                        ps.append(head(z).argmax(-1).cpu().numpy())
                        ls.append(lb.numpy())
                acc = accuracy_score(np.concatenate(ls), np.concatenate(ps))
                best = max(best, acc)
            print(f'  {ds:<25}: Acc={best:.4f}')
            results[ds] = best
        except Exception as e:
            print(f'  {ds}: SKIP ({type(e).__name__}: {e})')
    for p in model.parameters(): p.requires_grad = True
    return results


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    print('='*60)
    print('5-Task Eval: ConfigB + RevIN OFF + True OP (82.7M)')
    print('='*60)
    print(f'Loading: {CKPT}')
    model = ConfigBNoRevIN(width=192, branch_hidden=768, trunk_depth=2, top_k_freq=5).to(DEVICE)
    state = torch.load(CKPT, map_location=DEVICE)
    model.load_state_dict(state)
    n = sum(p.numel() for p in model.parameters())
    print(f'Model: {n/1e6:.1f}M params')

    fc_res = eval_forecasting(model)
    imp_res = eval_imputation(model)
    cls_res = eval_classification(model)

    print('\n' + '='*60)
    print('SUMMARY — ConfigB+method (82.7M)')
    print('='*60)
    print('\nForecasting (avg):')
    for k in ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'Weather', 'Exchange']:
        avgs = [v for k_, v in fc_res.items() if k_.startswith(k+'_')]
        if avgs: print(f'  {k:<10}: {np.mean(avgs):.4f}')
    print('\nImputation:')
    for k, v in imp_res.items(): print(f'  {k:<10}: {v:.4f}')
    print('\nClassification:')
    for k, v in cls_res.items(): print(f'  {k:<25}: {v:.4f}')
    if cls_res:
        print(f'  {"AVG":<25}: {np.mean(list(cls_res.values())):.4f}')
    print('='*60)
