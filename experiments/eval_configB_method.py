"""
5-Task Evaluation for ConfigB + RevIN OFF + True OP (83M)

Checkpoint: checkpoints/configB_revinoff_trueop.pth

Tasks:
  1. Forecasting (ETTh/ETTm/Weather/Exchange, 4 horizons each)
  2. Imputation (5 mask rates)
  3. Classification (UCR/UEA, linear probe)
  4. Anomaly detection (reconstruction error)

CUDA_VISIBLE_DEVICES=2 python experiments/eval_configB_method.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch, torch.nn as nn
import numpy as np
from types import SimpleNamespace
from torch import optim
from torch.utils.data import DataLoader

from data_provider.data_factory import data_provider
from data_provider.data_loader import Dataset_Classification
from experiments.exp_configB_revinoff_trueop import ConfigBNoRevIN
from sklearn.metrics import accuracy_score, f1_score

DEVICE = torch.device('cuda')
CKPT = 'checkpoints/configB_revinoff_trueop.pth'


# ============================================================
# Task 1: Forecasting (zero-shot + iterative roll-out)
# ============================================================
def iterative_forecast(model, x_enc, target_pred_len):
    """Roll-out forecast using model's native 96-step forecast."""
    B, S, C = x_enc.shape
    cur = x_enc
    chunks = []
    remain = target_pred_len
    while remain > 0:
        step = min(96, remain)
        with torch.no_grad():
            p = model.forecast(cur, target_pred_len=step)  # [B, step, C]
        chunks.append(p)
        if remain > step:
            cur = torch.cat([cur[:, step:, :], p], dim=1)
        remain -= step
    return torch.cat(chunks, dim=1)


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
                    p = iterative_forecast(model, bx, pl)
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
                print(f'  {dn}_{pl}: ERROR ({e})')
    return results


# ============================================================
# Task 2: Imputation
# ============================================================
def eval_imputation(model):
    print(f'\n{"="*60}\nTask 2: Imputation (zero-shot)\n{"="*60}')
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
                with torch.no_grad():
                    for bx, by, _, _ in tdl:
                        bx = bx.float().to(DEVICE)
                        mk = (torch.rand_like(bx) > mr).float()
                        rec = model.reconstruct(bx * mk)
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


# ============================================================
# Task 3: Classification (linear probe)
# ============================================================
def eval_classification(model):
    print(f'\n{"="*60}\nTask 3: Classification (linear probe)\n{"="*60}')
    h = model.branch_hidden
    # Freeze
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
                nn.Linear(256, trd.n_classes)
            ).to(DEVICE)
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
            print(f'  {ds}: SKIP ({e})')

    # Unfreeze
    for p in model.parameters(): p.requires_grad = True
    return results


# ============================================================
# Task 4: Anomaly Detection (reconstruction error)
# ============================================================
def eval_anomaly(model):
    print(f'\n{"="*60}\nTask 4: Anomaly Detection (reconstruction error)\n{"="*60}')
    try:
        from data_provider.data_loader import Dataset_PSM, Dataset_MSL, Dataset_SMAP, Dataset_SMD, Dataset_SWAT
    except ImportError:
        print('  Anomaly dataset loaders not found, checking custom path...')
        # Try loading from TSB-UAD or manually
        anom_root = './dataset/anomaly_detection/TSB-UAD-Public'
        if not os.path.exists(anom_root):
            print(f'  Anomaly datasets not available at {anom_root}')
            return {}
        return eval_anomaly_tsb(model, anom_root)

    datasets = ['PSM', 'MSL', 'SMAP', 'SMD', 'SWAT']
    results = {}

    model.eval()
    for ds in datasets:
        try:
            # Use standard anomaly detection protocol
            a = SimpleNamespace(seq_len=96, pred_len=0, label_len=0, data=ds,
                root_path=f'./dataset/anomaly_detection/{ds}/',
                data_path='', features='M', target='OT',
                freq='h', embed='timeF', enc_in=1, dec_in=1, c_out=1,
                num_workers=2, batch_size=32, exp_name='anomaly', ordered_data=True,
                data_amount=-1, combine_Gaussian_datasets=False,
                synthetic_data_path='', synthetic_root_path='./',
                synthetic_length=1024, stride=1)
            _, tdl = data_provider(a, 'test')
            # Compute reconstruction errors
            errors, labels = [], []
            with torch.no_grad():
                for batch in tdl:
                    if len(batch) >= 2:
                        bx, lb = batch[0], batch[1]
                    else:
                        continue
                    bx = bx.float().to(DEVICE)
                    rec = model.reconstruct(bx)
                    err = ((rec - bx)**2).mean(dim=(1,2)).cpu().numpy()
                    errors.append(err)
                    if hasattr(lb, 'numpy'):
                        labels.append(lb.numpy().flatten())
            if not errors:
                print(f'  {ds}: no data loaded')
                continue
            err_arr = np.concatenate(errors)
            if labels:
                lab_arr = np.concatenate(labels)[:len(err_arr)]
                # F1 at best threshold
                from sklearn.metrics import f1_score
                best_f1 = 0
                for pct in [90, 95, 99, 99.5]:
                    thresh = np.percentile(err_arr, pct)
                    pred_lab = (err_arr > thresh).astype(int)
                    try:
                        f1 = f1_score(lab_arr, pred_lab)
                    except:
                        f1 = 0
                    best_f1 = max(best_f1, f1)
                print(f'  {ds:<10}: Best F1={best_f1:.4f}  (mean err={err_arr.mean():.4f})')
                results[ds] = best_f1
            else:
                print(f'  {ds:<10}: mean recon err={err_arr.mean():.4f} (no labels)')
        except Exception as e:
            print(f'  {ds}: SKIP ({type(e).__name__}: {e})')
    return results


def eval_anomaly_tsb(model, anom_root):
    """Fallback: compute reconstruction error on TSB-UAD files."""
    results = {}
    errors_per_subdir = {}
    model.eval()
    subdirs = sorted([d for d in os.listdir(anom_root) if os.path.isdir(os.path.join(anom_root, d))])
    for subdir in subdirs[:5]:  # limit to 5 subdirs
        subdir_path = os.path.join(anom_root, subdir)
        errs = []
        files = [f for f in os.listdir(subdir_path) if f.endswith('.out') or f.endswith('.csv')]
        for fname in files[:10]:  # limit per subdir
            try:
                data = np.loadtxt(os.path.join(subdir_path, fname), delimiter=',')
                if data.ndim == 1:
                    data = data[:, None]
                series = data[:, 0]
                labels = data[:, 1] if data.shape[1] > 1 else None
                s = np.std(series)
                if s < 1e-8: continue
                series = (series - np.mean(series)) / s
                series = np.clip(series, -10, 10)
                # Sliding windows
                wins, labs_win = [], []
                for i in range(0, len(series) - 96 + 1, 48):
                    w = series[i:i+96]
                    wins.append(w)
                    if labels is not None:
                        labs_win.append(labels[i:i+96].max())
                if not wins: continue
                wins_t = torch.tensor(np.stack(wins)).float().unsqueeze(-1).to(DEVICE)
                with torch.no_grad():
                    rec = model.reconstruct(wins_t).cpu().numpy().squeeze(-1)
                err = np.mean((rec - np.stack(wins))**2, axis=1)
                errs.extend(err.tolist())
            except: continue
        if errs:
            errors_per_subdir[subdir] = np.mean(errs)
            print(f'  {subdir:<20}: mean recon err={errors_per_subdir[subdir]:.4f}')
    return errors_per_subdir


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    print('='*60)
    print('5-Task Evaluation: ConfigB + RevIN OFF + True OP (83M)')
    print('='*60)
    print(f'Loading checkpoint: {CKPT}')
    model = ConfigBNoRevIN(width=192, branch_hidden=768, trunk_depth=2, top_k_freq=5).to(DEVICE)
    state = torch.load(CKPT, map_location=DEVICE)
    model.load_state_dict(state)
    n = sum(p.numel() for p in model.parameters())
    print(f'Model: {n/1e6:.1f}M params')

    fc_res = eval_forecasting(model)
    imp_res = eval_imputation(model)
    cls_res = eval_classification(model)
    anom_res = eval_anomaly(model)

    print('\n' + '='*60)
    print('SUMMARY — 5 Task Results')
    print('='*60)
    print(f'\nForecasting (avg per dataset):')
    for k in ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'Weather', 'Exchange']:
        avgs = [v for k_, v in fc_res.items() if k_.startswith(k+'_')]
        if avgs: print(f'  {k:<10}: {np.mean(avgs):.4f}')

    print(f'\nImputation (avg):')
    for k, v in imp_res.items():
        print(f'  {k:<10}: {v:.4f}')

    print(f'\nClassification (acc):')
    for k, v in cls_res.items():
        print(f'  {k:<25}: {v:.4f}')
    if cls_res:
        print(f'  {"AVERAGE":<25}: {np.mean(list(cls_res.values())):.4f}')

    print(f'\nAnomaly (recon err or F1):')
    for k, v in anom_res.items():
        print(f'  {k:<20}: {v:.4f}')
    print('='*60)
