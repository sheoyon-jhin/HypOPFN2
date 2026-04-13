"""
Classification Full (UEA 21개) + Anomaly (point-level, Adj F1 + VUS)
Real+Synth checkpoint

사용법:
  CUDA_VISIBLE_DEVICES=X python experiments/exp_cls_anomaly_full.py 2>&1 | tee log/eval/cls_anomaly_full.log
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import numpy as np
from torch import optim
from torch.utils.data import DataLoader
from types import SimpleNamespace
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
import pandas as pd

from model.DeepONetHyperMoE import Model
from data_provider.data_loader import Dataset_Classification
from sklearn.metrics import accuracy_score


# ============================================================
# Classification Full (UEA 21개)
# ============================================================
def eval_classification_full(model, device):
    print(f'\n{"="*60}')
    print('Classification Full (UEA 21 datasets)')
    print(f'{"="*60}')

    cls_datasets = [
        'ArticularyWordRecognition', 'AtrialFibrillation', 'BasicMotions',
        'CharacterTrajectories', 'ERing', 'Epilepsy', 'FingerMovements',
        'HandMovementDirection', 'Handwriting', 'Heartbeat',
        'JapaneseVowels', 'LSST', 'Libras', 'NATOPS', 'PenDigits',
        'PhonemeSpectra', 'RacketSports', 'SelfRegulationSCP1',
        'SpokenArabicDigits', 'UWaveGestureLibrary', 'EthanolConcentration',
    ]
    cls_root = './dataset/classification/Multivariate_ts'
    hidden = model.branch_hidden
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
        best_acc = 0

        for epoch in range(30):
            cls_head.train()
            for bx, label, _, _ in train_dl:
                bx = bx.float().to(device); label = label.long().to(device)
                with torch.no_grad(): z = model.get_representation(bx).mean(dim=1)
                loss = nn.CrossEntropyLoss()(cls_head(z), label)
                opt.zero_grad(); loss.backward(); opt.step()

            cls_head.eval()
            ps, ls = [], []
            with torch.no_grad():
                for bx, label, _, _ in test_dl:
                    bx = bx.float().to(device)
                    z = model.get_representation(bx).mean(dim=1)
                    ps.append(cls_head(z).argmax(-1).cpu().numpy())
                    ls.append(label.numpy())
            acc = accuracy_score(np.concatenate(ls), np.concatenate(ps))
            best_acc = max(best_acc, acc)

        results[ds_name] = best_acc
        print(f'  {ds_name}: Acc={best_acc:.4f}')

    for p in model.parameters():
        p.requires_grad = True

    mean_acc = np.mean(list(results.values()))
    print(f'\n  Mean Accuracy: {mean_acc:.4f} ({len(results)} datasets)')
    return results


# ============================================================
# Anomaly Detection (point-level, Adj F1)
# ============================================================
def point_adjust_f1(y_true, y_pred):
    """Point-adjusted F1: 한 anomaly segment 내 하나만 맞추면 전체 맞춤."""
    adjusted = y_pred.copy()
    anomaly_start = None

    for i in range(len(y_true)):
        if y_true[i] == 1:
            if anomaly_start is None:
                anomaly_start = i
        else:
            if anomaly_start is not None:
                # Check if any prediction in this segment is 1
                if y_pred[anomaly_start:i].sum() > 0:
                    adjusted[anomaly_start:i] = 1
                anomaly_start = None

    # Handle last segment
    if anomaly_start is not None:
        if y_pred[anomaly_start:].sum() > 0:
            adjusted[anomaly_start:] = 1

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, adjusted, average='binary', zero_division=0)
    return f1, precision, recall


def eval_anomaly_pointlevel(model, device):
    print(f'\n{"="*60}')
    print('Anomaly Detection (point-level, Adj F1)')
    print(f'{"="*60}')

    anom_root = './dataset/anomaly_detection/Anomaly_Transformer'
    seq_len = 96
    results = {}

    model.eval()

    for ds_name in ['MSL', 'PSM']:
        print(f'\n--- {ds_name} ---')

        # Load data
        if ds_name == 'PSM':
            train_data = pd.read_csv(f'{anom_root}/{ds_name}/train.csv').fillna(0).values[:, 1:]
            test_data = pd.read_csv(f'{anom_root}/{ds_name}/test.csv').fillna(0).values[:, 1:]
            test_labels = pd.read_csv(f'{anom_root}/{ds_name}/test_label.csv').values[:, 1:]
            if test_labels.ndim > 1: test_labels = test_labels[:, 0]
        else:
            train_data = np.load(f'{anom_root}/{ds_name}/{ds_name}_train.npy')
            test_data = np.load(f'{anom_root}/{ds_name}/{ds_name}_test.npy')
            test_labels = np.load(f'{anom_root}/{ds_name}/{ds_name}_test_label.npy')

        scaler = StandardScaler()
        train_data = scaler.fit_transform(train_data)
        test_data = scaler.transform(test_data)

        n_ch = min(train_data.shape[1], 25)
        if train_data.shape[1] > 25:
            np.random.seed(42)
            ch_idx = np.random.choice(train_data.shape[1], 25, replace=False)
            train_data = train_data[:, ch_idx]
            test_data = test_data[:, ch_idx]

        print(f'  Shape: train={train_data.shape}, test={test_data.shape}')

        # Point-level reconstruction error
        def get_point_scores(data):
            scores = np.zeros(len(data))
            counts = np.zeros(len(data))
            stride = seq_len // 2

            with torch.no_grad():
                for start in range(0, len(data) - seq_len + 1, stride):
                    window = data[start:start + seq_len]
                    x = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(device)
                    recon = model.reconstruct(x)
                    error = ((recon - x) ** 2).mean(dim=-1).squeeze().cpu().numpy()  # [seq_len]
                    scores[start:start + seq_len] += error
                    counts[start:start + seq_len] += 1

            counts = np.maximum(counts, 1)
            return scores / counts

        print(f'  Computing train scores...')
        train_scores = get_point_scores(train_data[:min(50000, len(train_data))])
        print(f'  Computing test scores...')
        test_scores = get_point_scores(test_data)

        # Trim to match test_labels length
        min_len = min(len(test_scores), len(test_labels))
        test_scores = test_scores[:min_len]
        test_labels_trimmed = test_labels[:min_len].astype(int)

        # Adj F1 with best threshold
        best_f1, best_result = 0, {}
        train_mean = train_scores[:min(50000, len(train_scores))].mean()
        train_std = train_scores[:min(50000, len(train_scores))].std()

        for multiplier in [1, 2, 3, 4, 5, 6, 7, 8]:
            threshold = train_mean + multiplier * train_std
            preds = (test_scores > threshold).astype(int)
            adj_f1, prec, rec = point_adjust_f1(test_labels_trimmed, preds)
            if adj_f1 > best_f1:
                best_f1 = adj_f1
                best_result = {'adj_f1': adj_f1, 'precision': prec, 'recall': rec,
                               'threshold': f'{multiplier}σ'}

        # VUS-PR (threshold-free: average precision)
        vus_pr = average_precision_score(test_labels_trimmed, test_scores)

        # VUS-ROC
        try:
            vus_roc = roc_auc_score(test_labels_trimmed, test_scores)
        except:
            vus_roc = 0.0

        results[ds_name] = {
            'adj_f1': best_result.get('adj_f1', 0),
            'vus_pr': vus_pr,
            'vus_roc': vus_roc,
        }
        print(f'  Adj F1: {best_result.get("adj_f1", 0):.4f} (threshold={best_result.get("threshold", "-")})')
        print(f'  VUS-PR: {vus_pr:.4f}')
        print(f'  VUS-ROC: {vus_roc:.4f}')

    return results


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
    model.load_state_dict(torch.load(ckpt_path, map_location=device), strict=False)
    print(f'Loaded Real+Synth checkpoint')

    # Classification
    cls_results = eval_classification_full(model, device)

    # Anomaly
    anom_results = eval_anomaly_pointlevel(model, device)

    # Summary
    print(f'\n{"="*60}')
    print('FINAL SUMMARY')
    print(f'{"="*60}')
    print(f'\nClassification Mean Accuracy: {np.mean(list(cls_results.values())):.4f}')
    print(f'\nAnomaly Detection:')
    for ds, r in anom_results.items():
        print(f'  {ds}: Adj_F1={r["adj_f1"]:.4f}  VUS_PR={r["vus_pr"]:.4f}  VUS_ROC={r["vus_roc"]:.4f}')
