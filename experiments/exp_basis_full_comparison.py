"""
Basis 함수 확장 비교: 유망한 3개 basis × 다양한 Task
  A: Fourier only (forecasting best)
  C: Fourier+Wavelet+Sigmoid+Decay (classification best)
  E: Learnable (ETTh2 best)

Tasks:
  - Forecasting: ETTh1, ETTh2, Weather, Exchange (pl=96, 336)
  - Classification: Epilepsy, BasicMotions, NATOPS, FingerMovements, EthanolConc
  - Imputation: ETTh1 (m=0.125, 0.25, 0.5)

사용법:
  CUDA_VISIBLE_DEVICES=3 python experiments/exp_basis_full_comparison.py 2>&1 | tee log/basis_full.log
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
from data_provider.data_loader import Dataset_Classification
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

# Reuse model from basis exploration
from experiments.exp_basis_exploration import BasisTestModel


def train_and_eval_forecasting(basis_type, device):
    datasets = {
        'ETTh1': ('ETTh1', './dataset/ETT-small/', 'ETTh1.csv', 7),
        'ETTh2': ('ETTh2', './dataset/ETT-small/', 'ETTh2.csv', 7),
        'Weather': ('custom', './dataset/weather/', 'weather.csv', 21),
        'Exchange': ('custom', './dataset/exchange_rate/', 'exchange_rate.csv', 8),
    }
    results = {}
    for dname, (data, root, fpath, enc_in) in datasets.items():
        for pl in [96, 336]:
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
            model = BasisTestModel(96, pl, 64, 256, basis_type).to(device)
            _, train_dl = data_provider(args, 'train')
            _, test_dl = data_provider(args, 'test')
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            best_loss, patience, best_state = float('inf'), 0, None

            for epoch in range(20):
                model.train()
                losses = []
                for bx, by, _, _ in train_dl:
                    bx, by = bx.float().to(device), by.float().to(device)
                    optimizer.zero_grad()
                    out = model(bx, target_pred_len=pl)
                    loss = F.mse_loss(out, by[:, -pl:, :])
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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

            model.load_state_dict(best_state); model.to(device).eval()
            preds, trues = [], []
            with torch.no_grad():
                for bx, by, _, _ in test_dl:
                    bx = bx.float().to(device)
                    out = model(bx, target_pred_len=pl)
                    preds.append(out.cpu().numpy())
                    trues.append(by[:, -pl:, :].numpy())
            preds, trues = np.concatenate(preds), np.concatenate(trues)
            mse = np.mean((preds - trues) ** 2)
            key = f'{dname}_pl{pl}'
            results[key] = mse
            print(f'    {key}: MSE={mse:.4f}')
    return results


def train_and_eval_classification(basis_type, device):
    results = {}
    cls_root = './dataset/classification/Multivariate_ts'
    for ds_name in ['Epilepsy', 'BasicMotions', 'NATOPS', 'FingerMovements', 'EthanolConcentration']:
        model = BasisTestModel(96, 96, 64, 256, basis_type).to(device)
        train_ds = Dataset_Classification(root_path=cls_root, flag='train', size=[96,0,96], data_path=ds_name)
        test_ds = Dataset_Classification(root_path=cls_root, flag='test', size=[96,0,96], data_path=ds_name)
        train_dl = DataLoader(train_ds, batch_size=16, shuffle=True, drop_last=True)
        test_dl = DataLoader(test_ds, batch_size=16, shuffle=False)

        cls_head = nn.Sequential(nn.Linear(256, 128), nn.GELU(), nn.Dropout(0.1),
                                 nn.Linear(128, train_ds.n_classes)).to(device)
        optimizer = optim.Adam(list(model.parameters()) + list(cls_head.parameters()), lr=0.001)
        best_acc = 0
        for epoch in range(50):
            model.train(); cls_head.train()
            for bx, label, _, _ in train_dl:
                bx = bx.float().to(device); label = label.long().to(device)
                z = model.get_representation(bx).mean(dim=1)
                loss = nn.CrossEntropyLoss()(cls_head(z), label)
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            model.eval(); cls_head.eval()
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
        print(f'    {ds_name}: Acc={best_acc:.4f}')
    return results


def eval_imputation(basis_type, device):
    results = {}
    for mask_rate in [0.125, 0.25, 0.5]:
        args = SimpleNamespace(
            seq_len=96, pred_len=96, label_len=0,
            data='ETTh1', root_path='./dataset/ETT-small/', data_path='ETTh1.csv',
            features='M', target='OT', freq='h', embed='timeF',
            enc_in=7, dec_in=7, c_out=7, num_workers=2, batch_size=32,
            exp_name='MTSF', ordered_data=False, data_amount=-1,
            combine_Gaussian_datasets=False, synthetic_data_path='', synthetic_root_path='./',
            synthetic_length=1024, stride=-1,
        )
        model = BasisTestModel(96, 96, 64, 256, basis_type).to(device)
        _, train_dl = data_provider(args, 'train')
        _, test_dl = data_provider(args, 'test')

        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss(reduction='none')
        best_loss, patience, best_state = float('inf'), 0, None

        for epoch in range(20):
            model.train()
            losses = []
            for bx, by, _, _ in train_dl:
                bx = bx.float().to(device)
                mask = (torch.rand_like(bx) > mask_rate).float()
                optimizer.zero_grad()
                out = model.reconstruct(bx * mask)
                loss_mat = criterion(out, bx)
                inv_mask = 1.0 - mask
                loss = (loss_mat * inv_mask).sum() / inv_mask.sum().clamp(min=1)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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

        model.load_state_dict(best_state); model.to(device).eval()
        torch.manual_seed(2021)
        preds, trues, masks_all = [], [], []
        with torch.no_grad():
            for bx, by, _, _ in test_dl:
                bx = bx.float().to(device)
                mask = (torch.rand_like(bx) > mask_rate).float()
                out = model.reconstruct(bx * mask)
                preds.append(out.cpu().numpy()); trues.append(bx.cpu().numpy()); masks_all.append(mask.cpu().numpy())
        p, t, m = np.concatenate(preds), np.concatenate(trues), np.concatenate(masks_all)
        mse = np.mean((p[m==0] - t[m==0])**2)
        results[f'm={mask_rate}'] = mse
        print(f'    mask={mask_rate}: MSE={mse:.4f}')
    return results


if __name__ == '__main__':
    device = torch.device('cuda')

    basis_configs = {
        'A_fourier': 'fourier',
        'C_wavelet_step': 'fourier_wavelet_sigmoid_decay',
        'E_learnable': 'learnable',
    }

    all_results = {}

    for name, btype in basis_configs.items():
        print(f'\n{"="*60}')
        print(f'Basis: {name} ({btype})')
        print(f'{"="*60}')

        print('\n  Forecasting:')
        fc = train_and_eval_forecasting(btype, device)

        print('\n  Classification:')
        cls = train_and_eval_classification(btype, device)

        print('\n  Imputation:')
        imp = eval_imputation(btype, device)

        all_results[name] = {**fc, **{f'cls_{k}': v for k, v in cls.items()}, **imp}

    # Summary
    print(f'\n{"="*60}')
    print('FULL COMPARISON')
    print(f'{"="*60}')

    metrics = ['ETTh1_pl96', 'ETTh1_pl336', 'ETTh2_pl96', 'Weather_pl96', 'Exchange_pl96',
               'cls_Epilepsy', 'cls_BasicMotions', 'cls_NATOPS', 'cls_FingerMovements',
               'm=0.125', 'm=0.25', 'm=0.5']

    header = f'{"Metric":<20}' + ''.join(f'{n:>15}' for n in basis_configs.keys())
    print(header)
    print('-' * len(header))
    for metric in metrics:
        row = f'{metric:<20}'
        for name in basis_configs.keys():
            val = all_results[name].get(metric, 0)
            row += f'{val:>15.4f}'
        print(row)
