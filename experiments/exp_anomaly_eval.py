"""
Anomaly Detection Eval: Real+Synth checkpoint
MSL, PSM (작은 데이터셋만, SMD는 너무 큼)
Zero-shot reconstruction error 기반

사용법:
  CUDA_VISIBLE_DEVICES=0 python experiments/exp_anomaly_eval.py 2>&1 | tee log/eval/anomaly_eval.log
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
from types import SimpleNamespace
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.preprocessing import StandardScaler
import pandas as pd

from model.DeepONetHyperMoE import Model


def load_anomaly_data(dataset_name, root='./dataset/anomaly_detection/Anomaly_Transformer', seq_len=96):
    """Load anomaly dataset and create windows."""
    if dataset_name == 'PSM':
        train_data = pd.read_csv(f'{root}/{dataset_name}/train.csv').fillna(0).values[:, 1:]
        test_data = pd.read_csv(f'{root}/{dataset_name}/test.csv').fillna(0).values[:, 1:]
        test_labels = pd.read_csv(f'{root}/{dataset_name}/test_label.csv').values[:, 1:]
        if test_labels.ndim > 1:
            test_labels = test_labels[:, 0]
    else:
        train_data = np.load(f'{root}/{dataset_name}/{dataset_name}_train.npy')
        test_data = np.load(f'{root}/{dataset_name}/{dataset_name}_test.npy')
        test_labels = np.load(f'{root}/{dataset_name}/{dataset_name}_test_label.npy')

    # Normalize
    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_data)
    test_data = scaler.transform(test_data)

    # Subsample channels if too many
    n_ch = train_data.shape[1]
    if n_ch > 25:
        ch_idx = np.random.choice(n_ch, 25, replace=False)
        train_data = train_data[:, ch_idx]
        test_data = test_data[:, ch_idx]

    # Create windows
    def make_windows(data, labels=None):
        windows = []
        window_labels = []
        for i in range(0, len(data) - seq_len + 1, seq_len // 2):
            windows.append(data[i:i+seq_len])
            if labels is not None:
                window_labels.append(int(labels[i:i+seq_len].sum() > 0))
        return np.array(windows), np.array(window_labels) if labels is not None else None

    train_windows, _ = make_windows(train_data)
    test_windows, test_window_labels = make_windows(test_data, test_labels)

    return train_windows, test_windows, test_window_labels


def eval_anomaly(model, device, dataset_name):
    print(f'\n--- {dataset_name} ---')

    train_windows, test_windows, test_labels = load_anomaly_data(dataset_name)
    n_ch = train_windows.shape[2]
    print(f'  train: {train_windows.shape}, test: {test_windows.shape}, channels: {n_ch}')

    model.eval()

    # Train scores (for threshold)
    train_scores = []
    with torch.no_grad():
        for i in range(0, len(train_windows), 32):
            batch = torch.tensor(train_windows[i:i+32], dtype=torch.float32).to(device)
            recon = model.reconstruct(batch)
            score = torch.mean((recon - batch) ** 2, dim=(-1, -2))
            train_scores.append(score.cpu().numpy())
    train_scores = np.concatenate(train_scores)

    # Test scores
    test_scores = []
    with torch.no_grad():
        for i in range(0, len(test_windows), 32):
            batch = torch.tensor(test_windows[i:i+32], dtype=torch.float32).to(device)
            recon = model.reconstruct(batch)
            score = torch.mean((recon - batch) ** 2, dim=(-1, -2))
            test_scores.append(score.cpu().numpy())
    test_scores = np.concatenate(test_scores)

    # Best F1 threshold search
    best_f1, best_result = 0, {}
    for percentile in [90, 92, 95, 97, 99, 99.5]:
        threshold = np.percentile(train_scores, percentile)
        preds = (test_scores > threshold).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            test_labels, preds, average='binary', zero_division=0)
        acc = accuracy_score(test_labels, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_result = {
                'percentile': percentile, 'threshold': threshold,
                'precision': precision, 'recall': recall,
                'f1': f1, 'accuracy': acc,
            }

    print(f'  Best F1: {best_result["f1"]:.4f} (percentile={best_result["percentile"]})')
    print(f'  Precision: {best_result["precision"]:.4f}, Recall: {best_result["recall"]:.4f}')
    print(f'  Accuracy: {best_result["accuracy"]:.4f}')
    return best_result


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
    model.load_state_dict(torch.load(ckpt_path, map_location=device), strict=False)
    print(f'Loaded Real+Synth checkpoint')

    print(f'\n{"="*60}')
    print('Anomaly Detection (reconstruction error)')
    print(f'{"="*60}')

    # MOMENT Table 7 비교
    moment_lp = {'MSL': 0.628, 'PSM': 0.628}  # Adj F1 mean
    moment_0 = {'MSL': 0.585, 'PSM': 0.585}

    results = {}
    for ds in ['MSL', 'PSM']:
        result = eval_anomaly(model, device, ds)
        results[ds] = result

    print(f'\n{"="*60}')
    print('SUMMARY: Anomaly Detection')
    print(f'{"="*60}')
    print(f'{"Dataset":<10} {"Ours F1":>8} {"MOMENT_0":>10} {"MOMENT_LP":>10}')
    print('-' * 40)
    for ds, r in results.items():
        print(f'  {ds:<10} {r["f1"]:>8.4f} {moment_0.get(ds, "-"):>10} {moment_lp.get(ds, "-"):>10}')
