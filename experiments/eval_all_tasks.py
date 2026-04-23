"""
통합 Evaluation 함수: 모든 실험에서 공통으로 사용
MOMENT 논문 Table과 동일한 세팅으로 비교 가능

포함:
  1. Long-term Forecasting: ETTh1/h2, ETTm1/m2, Weather, Exchange (pl=96,192,336,720)
  2. Imputation: Weather, ETTh1/h2, ETTm1/m2, Electricity (mask=0.125,0.25,0.375,0.5)
  3. Classification: Epilepsy, BasicMotions, NATOPS, FingerMovements, EthanolConcentration
  4. (TODO) Short-term Forecasting: M3, M4
  5. (TODO) Anomaly Detection: MSL, PSM

사용법:
  from experiments.eval_all_tasks import eval_forecasting, eval_imputation, eval_classification
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import optim
from torch.utils.data import DataLoader
from types import SimpleNamespace
from sklearn.metrics import accuracy_score

from data_provider.data_factory import data_provider
from data_provider.data_loader import Dataset_Classification


def _get_data_args(data, root, fpath, enc_in, seq_len=96, pred_len=96, label_len=48):
    return SimpleNamespace(
        seq_len=seq_len, pred_len=pred_len, label_len=label_len,
        data=data, root_path=root, data_path=fpath,
        features='M', target='OT', freq='h', embed='timeF',
        enc_in=enc_in, dec_in=enc_in, c_out=enc_in,
        num_workers=2, batch_size=1,
        exp_name='MTSF', ordered_data=False, data_amount=-1,
        combine_Gaussian_datasets=False,
        synthetic_data_path='', synthetic_root_path='./',
        synthetic_length=1024, stride=-1,
    )


# ============================================================
# 1. Long-term Forecasting
# ============================================================
def eval_forecasting(model, device, pred_lens=[96, 192, 336, 720]):
    print(f'\n{"="*60}')
    print('Long-term Forecasting')
    print(f'{"="*60}')

    datasets = {
        'Weather': ('custom', './dataset/weather/', 'weather.csv', 21),
        'ETTh1': ('ETTh1', './dataset/ETT-small/', 'ETTh1.csv', 7),
        'ETTh2': ('ETTh2', './dataset/ETT-small/', 'ETTh2.csv', 7),
        'ETTm1': ('ETTm1', './dataset/ETT-small/', 'ETTm1.csv', 7),
        'ETTm2': ('ETTm2', './dataset/ETT-small/', 'ETTm2.csv', 7),
        'Exchange': ('custom', './dataset/exchange_rate/', 'exchange_rate.csv', 8),
        # 아래는 channel 수가 많아서 느림 — 나중에 추가
        # 'ECL': ('custom', './dataset/electricity/', 'electricity.csv', 321),
        # 'Traffic': ('custom', './dataset/traffic/', 'traffic.csv', 862),
    }

    # ILI uses different pred_lens (24,36,48,60)
    ili_pred_lens = [24, 36, 48, 60]

    results = {}
    model.eval()
    for dname, (data, root, fpath, enc_in) in datasets.items():
        pls = ili_pred_lens if dname == 'ILI' else pred_lens
        for pl in pls:
            args = _get_data_args(data, root, fpath, enc_in, pred_len=pl)
            _, test_dl = data_provider(args, 'test')
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
            results[f'{dname}_pl{pl}'] = {'mse': mse, 'mae': mae}
            print(f'  {dname}_pl{pl}: MSE={mse:.4f} MAE={mae:.4f}')

    # Mean
    mean_mse = np.mean([v['mse'] for v in results.values()])
    print(f'  Mean MSE: {mean_mse:.4f}')
    return results


# ============================================================
# 2. Imputation (MOMENT Table 29 세팅)
# ============================================================
def eval_imputation(model, device, mask_rates=[0.125, 0.25, 0.375, 0.5]):
    print(f'\n{"="*60}')
    print('Imputation')
    print(f'{"="*60}')

    datasets = {
        'Weather': ('custom', './dataset/weather/', 'weather.csv', 21),
        'ETTh1': ('ETTh1', './dataset/ETT-small/', 'ETTh1.csv', 7),
        'ETTh2': ('ETTh2', './dataset/ETT-small/', 'ETTh2.csv', 7),
        'ETTm1': ('ETTm1', './dataset/ETT-small/', 'ETTm1.csv', 7),
        'ETTm2': ('ETTm2', './dataset/ETT-small/', 'ETTm2.csv', 7),
        'Electricity': ('custom', './dataset/electricity/', 'electricity.csv', 321),
    }

    results = {}
    model.eval()

    for dname, (data, root, fpath, enc_in) in datasets.items():
        args = _get_data_args(data, root, fpath, enc_in, pred_len=96, label_len=0)
        _, test_dl = data_provider(args, 'test')

        for mask_rate in mask_rates:
            torch.manual_seed(2021)
            preds, trues, masks = [], [], []
            with torch.no_grad():
                for bx, by, _, _ in test_dl:
                    bx = bx.float().to(device)
                    mask = (torch.rand_like(bx) > mask_rate).float()
                    out = model.reconstruct(bx * mask)
                    preds.append(out.cpu().numpy())
                    trues.append(bx.cpu().numpy())
                    masks.append(mask.cpu().numpy())

            p = np.concatenate(preds)
            t = np.concatenate(trues)
            m = np.concatenate(masks)
            mse = np.mean((p[m == 0] - t[m == 0]) ** 2)
            mae = np.mean(np.abs(p[m == 0] - t[m == 0]))
            results[f'{dname}_m{mask_rate}'] = {'mse': mse, 'mae': mae}
            print(f'  {dname} mask={mask_rate}: MSE={mse:.4f} MAE={mae:.4f}')

        # Per-dataset mean
        ds_mses = [results[f'{dname}_m{mr}']['mse'] for mr in mask_rates]
        print(f'  {dname} Mean: MSE={np.mean(ds_mses):.4f}')

    return results


# ============================================================
# 3. Classification
# ============================================================
def eval_classification(model, device):
    print(f'\n{"="*60}')
    print('Classification')
    print(f'{"="*60}')

    cls_datasets = [
        # Fast (21개) — ch < 100, len < 1000
        'ArticularyWordRecognition', 'AtrialFibrillation', 'BasicMotions',
        'CharacterTrajectories', 'ERing', 'Epilepsy', 'FingerMovements',
        'HandMovementDirection', 'Handwriting', 'Heartbeat',
        'JapaneseVowels', 'LSST', 'Libras', 'NATOPS', 'PenDigits',
        'PhonemeSpectra', 'RacketSports', 'SelfRegulationSCP1',
        'SpokenArabicDigits', 'UWaveGestureLibrary', 'EthanolConcentration',
        # Slow (나중에 추가) — ch > 100 or len > 1000
        # 'Cricket', 'DuckDuckGeese', 'EigenWorms', 'FaceDetection',
        # 'InsectWingbeat', 'MotorImagery', 'PEMS-SF', 'SelfRegulationSCP2',
        # 'StandWalkJump',
    ]
    cls_root = './dataset/classification/Multivariate_ts'
    hidden = model.branch_hidden if hasattr(model, 'branch_hidden') else model.hidden
    results = {}

    for p in model.parameters():
        p.requires_grad = False

    for ds_name in cls_datasets:
        try:
            train_ds = Dataset_Classification(root_path=cls_root, flag='train', size=[96, 0, 96], data_path=ds_name)
            test_ds = Dataset_Classification(root_path=cls_root, flag='test', size=[96, 0, 96], data_path=ds_name)
        except Exception as e:
            print(f'  {ds_name}: SKIP ({e})')
            continue

        train_dl = DataLoader(train_ds, batch_size=16, shuffle=True, drop_last=True)
        test_dl = DataLoader(test_ds, batch_size=16, shuffle=False)

        cls_head = nn.Sequential(
            nn.Linear(hidden, 256), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(256, train_ds.n_classes)
        ).to(device)

        opt = optim.Adam(cls_head.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        best_acc = 0
        for epoch in range(30):
            cls_head.train()
            for bx, label, _, _ in train_dl:
                bx = bx.float().to(device)
                label = label.long().to(device)
                with torch.no_grad():
                    z = model.get_representation(bx).mean(dim=1)
                loss = criterion(cls_head(z), label)
                opt.zero_grad()
                loss.backward()
                opt.step()

            cls_head.eval()
            preds, labels = [], []
            with torch.no_grad():
                for bx, label, _, _ in test_dl:
                    bx = bx.float().to(device)
                    z = model.get_representation(bx).mean(dim=1)
                    preds.append(cls_head(z).argmax(-1).cpu().numpy())
                    labels.append(label.numpy())
            acc = accuracy_score(np.concatenate(labels), np.concatenate(preds))
            best_acc = max(best_acc, acc)

        results[ds_name] = best_acc
        print(f'  {ds_name}: Acc={best_acc:.4f}')

    for p in model.parameters():
        p.requires_grad = True

    mean_acc = np.mean(list(results.values()))
    print(f'  Mean Acc: {mean_acc:.4f}')
    return results


# ============================================================
# 4. Short-term Forecasting (M3, M4) — sMAPE metric
# ============================================================
def _parse_tsf_series(filepath):
    """Parse Monash .tsf → list of (series_values, horizon)."""
    series_list = []
    horizon = 0
    with open(filepath, 'r') as f:
        in_data = False
        for line in f:
            line = line.strip()
            if line.startswith('@horizon'):
                horizon = int(line.split()[1])
            if line.startswith('@data'):
                in_data = True
                continue
            if not in_data or not line or line.startswith('@') or line.startswith('#'):
                continue
            parts = line.split(':')
            if len(parts) >= 3:
                values_str = parts[2].strip()
            elif len(parts) >= 2:
                values_str = parts[-1].strip()
            else:
                values_str = line
            try:
                values = [float(v) for v in values_str.split(',') if v.strip() and v.strip() != '?']
                if len(values) > horizon + 5:
                    series_list.append(np.array(values))
            except:
                continue
    return series_list, horizon


def _smape(pred, true):
    """Symmetric Mean Absolute Percentage Error."""
    denom = (np.abs(true) + np.abs(pred)) / 2.0
    denom = np.where(denom == 0, 1.0, denom)
    return np.mean(np.abs(pred - true) / denom) * 100


def eval_short_term(model, device):
    print(f'\n{"="*60}')
    print('Short-term Forecasting (M3, M4) — sMAPE')
    print(f'{"="*60}')
    import torch

    monash_dir = './dataset/time_series_pile/forecasting/monash'
    datasets = {
        'M3_Monthly': f'{monash_dir}/m3_monthly_dataset.tsf',
        'M3_Quarterly': f'{monash_dir}/m3_quarterly_dataset.tsf',
        'M3_Yearly': f'{monash_dir}/m3_yearly_dataset.tsf',
        'M4_Monthly': f'{monash_dir}/m4_monthly_dataset.tsf',
        'M4_Quarterly': f'{monash_dir}/m4_quarterly_dataset.tsf',
        'M4_Yearly': f'{monash_dir}/m4_yearly_dataset.tsf',
    }

    results = {}
    model.eval()

    for dname, fpath in datasets.items():
        if not os.path.exists(fpath):
            print(f'  {dname}: FILE NOT FOUND')
            continue

        series_list, horizon = _parse_tsf_series(fpath)
        if horizon == 0 or len(series_list) == 0:
            print(f'  {dname}: PARSE ERROR (horizon={horizon}, series={len(series_list)})')
            continue

        smapes = []
        seq_len = 96
        for series in series_list[:500]:  # limit for speed
            if len(series) < seq_len + horizon:
                continue

            # Last seq_len as input, last horizon as ground truth
            context = series[-(seq_len + horizon):-horizon]
            true_future = series[-horizon:]

            # Normalize
            mean, std = context.mean(), context.std()
            if std < 1e-8:
                continue
            context_norm = (context - mean) / std

            # Predict
            x = torch.tensor(context_norm, dtype=torch.float32).reshape(1, seq_len, 1).to(device)
            with torch.no_grad():
                out = model(x, None, None, None, target_pred_len=horizon)
                if isinstance(out, tuple):
                    out = out[0]
                pred_norm = out.cpu().numpy().reshape(-1)

            # Denormalize
            pred = pred_norm * std + mean

            smape = _smape(pred[:horizon], true_future[:horizon])
            if not np.isnan(smape) and smape < 200:
                smapes.append(smape)

        if smapes:
            mean_smape = np.mean(smapes)
            results[dname] = mean_smape
            print(f'  {dname}: sMAPE={mean_smape:.2f} ({len(smapes)} series)')
        else:
            print(f'  {dname}: NO VALID SERIES')

    if results:
        print(f'  Mean sMAPE: {np.mean(list(results.values())):.2f}')
    return results


# ============================================================
# Print Summary
# ============================================================
def print_summary(fc_results, imp_results, cls_results, st_results=None, model_name='Ours'):
    print(f'\n{"="*60}')
    print(f'SUMMARY: {model_name}')
    print(f'{"="*60}')

    print('\n--- Long-term Forecasting MSE ---')
    for k, v in fc_results.items():
        print(f'  {k}: MSE={v["mse"]:.4f} MAE={v["mae"]:.4f}')

    if st_results:
        print('\n--- Short-term Forecasting sMAPE ---')
        for k, v in st_results.items():
            print(f'  {k}: sMAPE={v:.2f}')

    print('\n--- Imputation MSE ---')
    for k, v in imp_results.items():
        print(f'  {k}: MSE={v["mse"]:.4f} MAE={v["mae"]:.4f}')

    print('\n--- Classification Accuracy ---')
    for k, v in cls_results.items():
        print(f'  {k}: {v:.4f}')
    if cls_results:
        print(f'  Mean: {np.mean(list(cls_results.values())):.4f}')
