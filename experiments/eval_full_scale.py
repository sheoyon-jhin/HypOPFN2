"""
Full Scale 50M Zero-shot Evaluation

체크포인트 로드 → ETTh/Weather/Exchange 표준 forecast + imputation eval
모델: FullScaleModel (50M, PatchAttn + 3 trunks, RevIN OFF)
체크포인트: checkpoints/full_scale_run.pth

CUDA_VISIBLE_DEVICES=1 python experiments/eval_full_scale.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch, torch.nn.functional as F
import numpy as np
from types import SimpleNamespace

from data_provider.data_factory import data_provider
from experiments.exp_full_scale_train import FullScaleModel, SEQ_LEN

DEVICE = torch.device('cuda')
CKPT = 'checkpoints/full_scale_run.pth'


# Multi-variate wrapper: per-channel forecast/impute with per-window normalization
def forecast_mv(model, x_enc, target_pred_len=96):
    """x_enc: [B, S, C] → [B, target_pred_len, C]"""
    B, S, C = x_enc.shape
    outs = []
    for ch in range(C):
        x_ch = x_enc[:, :, ch]  # [B, S]
        # per-sample normalize
        m = x_ch.mean(dim=1, keepdim=True)
        s = x_ch.std(dim=1, keepdim=True).clamp(min=1e-6)
        x_n = ((x_ch - m) / s).clamp(-10, 10)
        # crop/pad to SEQ_LEN
        if S > SEQ_LEN:
            x_n = x_n[:, -SEQ_LEN:]
        elif S < SEQ_LEN:
            x_n = F.pad(x_n, (SEQ_LEN - S, 0))
        with torch.no_grad():
            pred_n = model.forecast(x_n, n=target_pred_len)  # [B, target_pred_len]
        # de-normalize
        pred = pred_n * s + m
        outs.append(pred)
    return torch.stack(outs, dim=-1)  # [B, target_pred_len, C]


def reconstruct_mv(model, x_enc, mask):
    """x_enc: [B, S, C], mask: [B, S, C] (1=visible, 0=masked) → [B, S, C]"""
    B, S, C = x_enc.shape
    outs = []
    for ch in range(C):
        x_ch = x_enc[:, :, ch] * mask[:, :, ch]
        # crop/pad
        if S > SEQ_LEN:
            x_ch = x_ch[:, -SEQ_LEN:]
        elif S < SEQ_LEN:
            x_ch = F.pad(x_ch, (SEQ_LEN - S, 0))
        # normalize on visible only
        visible = mask[:, -SEQ_LEN:, ch] if S >= SEQ_LEN else F.pad(mask[:, :, ch], (SEQ_LEN-S, 0))
        m_vis = (x_ch * visible).sum(dim=1, keepdim=True) / visible.sum(dim=1, keepdim=True).clamp(min=1)
        s_vis = (((x_ch - m_vis) * visible)**2).sum(dim=1, keepdim=True) / visible.sum(dim=1, keepdim=True).clamp(min=1)
        s_vis = s_vis.sqrt().clamp(min=1e-6)
        x_n = ((x_ch - m_vis) / s_vis).clamp(-10, 10) * visible
        with torch.no_grad():
            recon_n = model.impute(x_n, n=SEQ_LEN)
        recon = recon_n * s_vis + m_vis
        outs.append(recon)
    return torch.stack(outs, dim=-1)


# ============================================================
# Forecast eval
# ============================================================
def eval_forecast(model):
    print(f'\n{"="*60}\nForecasting Eval (zero-shot)\n{"="*60}')
    datasets = {
        'ETTh1':    ('ETTh1',  './dataset/ETT-small/', 'ETTh1.csv', 7),
        'ETTh2':    ('ETTh2',  './dataset/ETT-small/', 'ETTh2.csv', 7),
        'ETTm1':    ('ETTm1',  './dataset/ETT-small/', 'ETTm1.csv', 7),
        'ETTm2':    ('ETTm2',  './dataset/ETT-small/', 'ETTm2.csv', 7),
        'Weather':  ('custom', './dataset/weather/',   'weather.csv', 21),
        'Exchange': ('custom', './dataset/exchange_rate/', 'exchange_rate.csv', 8),
    }
    # MOMENT-LP baseline (linear probe of MOMENT-base)
    moment_lp = {
        'ETTh1_96':0.387,'ETTh1_192':0.410,'ETTh1_336':0.422,'ETTh1_720':0.454,
        'ETTh2_96':0.288,'ETTh2_192':0.349,'ETTh2_336':0.369,'ETTh2_720':0.403,
        'ETTm1_96':0.293,'ETTm1_192':0.326,'ETTm1_336':0.352,'ETTm1_720':0.405,
        'ETTm2_96':0.170,'ETTm2_192':0.227,'ETTm2_336':0.275,'ETTm2_720':0.363,
        'Weather_96':0.154,'Weather_192':0.197,'Weather_336':0.246,'Weather_720':0.315,
    }
    model.eval()
    all_results = {}
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

                # For pl > SEQ_LEN, do iterative roll-out
                preds, tgts = [], []
                for bx, by, _, _ in tdl:
                    bx = bx.float().to(DEVICE)
                    if pl <= SEQ_LEN:
                        p = forecast_mv(model, bx, target_pred_len=pl)
                    else:
                        # iterative: forecast 96, append, forecast next 96
                        cur_input = bx
                        chunks = []
                        remain = pl
                        while remain > 0:
                            step = min(SEQ_LEN, remain)
                            p_step = forecast_mv(model, cur_input, target_pred_len=step)
                            chunks.append(p_step)
                            # roll input: drop oldest step, append predicted
                            if remain > step:
                                cur_input = torch.cat([cur_input[:, step:, :], p_step], dim=1)
                            remain -= step
                        p = torch.cat(chunks, dim=1)
                    preds.append(p.cpu().numpy())
                    tgts.append(by[:, -pl:, :].numpy())

                preds = np.concatenate(preds)
                tgts = np.concatenate(tgts)
                mse = np.mean((preds - tgts)**2)
                mae = np.mean(np.abs(preds - tgts))
                k = f'{dn}_{pl}'
                m_lp = moment_lp.get(k)
                gap = f'{(mse/m_lp-1)*100:+.0f}%' if m_lp else '-'
                m_str = f'{m_lp:.3f}' if m_lp else 'N/A'
                print(f'  {k:<14}: MSE={mse:.4f} MAE={mae:.4f}  | MOMENT-LP={m_str}  gap={gap}')
                all_results[k] = mse
            except Exception as e:
                print(f'  {dn}_{pl}: ERROR ({e})')
    return all_results


# ============================================================
# Imputation eval
# ============================================================
def eval_imputation(model):
    print(f'\n{"="*60}\nImputation Eval (zero-shot)\n{"="*60}')
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
            print(f'  {dn:<8}: Avg={avg:.4f}  |  by mr: {[f"{x:.3f}" for x in mr_mses]}  '
                  f'| MOMENT_0={m0} LP={m_lp}')
            results[dn] = avg
        except Exception as e:
            print(f'  {dn}: ERROR ({e})')
    return results


if __name__ == '__main__':
    print('='*60)
    print('Full Scale 50M Zero-shot Evaluation')
    print('='*60)
    print(f'Loading checkpoint: {CKPT}')
    model = FullScaleModel().to(DEVICE)
    state = torch.load(CKPT, map_location=DEVICE)
    model.load_state_dict(state)
    n = sum(p.numel() for p in model.parameters())
    print(f'Model: {n/1e6:.1f}M params')

    fc_res = eval_forecast(model)
    imp_res = eval_imputation(model)

    print('\n' + '='*60)
    print('SUMMARY')
    print('='*60)
    print(f'Forecast MSE (avg over horizons):')
    for k in ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'Weather', 'Exchange']:
        avg = np.mean([v for k_, v in fc_res.items() if k_.startswith(k+'_')])
        if not np.isnan(avg): print(f'  {k:<10}: {avg:.4f}')

    print(f'\nImputation MSE (avg over mask rates):')
    for k, v in imp_res.items():
        print(f'  {k:<10}: {v:.4f}')
    print('='*60)
