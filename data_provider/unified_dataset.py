"""
Unified Dataset: All real-world time series for pre-training.
No labels needed — masked reconstruction only.

Sources:
  - Forecasting: ETTh1/h2, ETTm1/m2, Weather, Exchange, Electricity
  - Anomaly: SMD, MSL, SMAP, PSM (train split only)
  - Classification: UEA 30 datasets (train split only, labels ignored)

Each sample: [seq_len] from one channel of one dataset.
Channel-independent: each channel is a separate training sample.
"""
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


class UnifiedPretrainDataset(Dataset):
    """Collects all real time series into channel-independent windows."""

    def __init__(self, seq_len=96, stride=48, root='./dataset'):
        self.seq_len = seq_len
        self.windows = []  # list of [seq_len] numpy arrays

        # 1. Forecasting datasets (train split: first 70%)
        self._add_csv_datasets(root, stride)

        # 2. Anomaly datasets (train split)
        self._add_anomaly_datasets(root, stride)

        # 3. Classification datasets (train split, labels ignored)
        self._add_classification_datasets(root)

        self.windows = np.array(self.windows, dtype=np.float32)
        print(f'UnifiedPretrainDataset: {len(self.windows)} windows, seq_len={seq_len}')

    def _add_csv_datasets(self, root, stride):
        csv_datasets = [
            ('ETT-small/ETTh1.csv', None),
            ('ETT-small/ETTh2.csv', None),
            ('ETT-small/ETTm1.csv', None),
            ('ETT-small/ETTm2.csv', None),
            ('weather/weather.csv', None),
            ('exchange_rate/exchange_rate.csv', None),
            # Electricity is too large (321ch), subsample channels
            ('electricity/electricity.csv', 20),
        ]

        for fpath, max_ch in csv_datasets:
            full_path = os.path.join(root, fpath)
            if not os.path.exists(full_path):
                continue

            df = pd.read_csv(full_path)
            # Drop date column
            data = df.iloc[:, 1:].values  # [T, C]

            # Train split: first 70%
            n_train = int(len(data) * 0.7)
            data = data[:n_train]

            # Normalize per channel
            scaler = StandardScaler()
            data = scaler.fit_transform(data)

            # Subsample channels if needed
            n_ch = data.shape[1]
            if max_ch and n_ch > max_ch:
                ch_idx = np.random.choice(n_ch, max_ch, replace=False)
                data = data[:, ch_idx]

            # Extract windows per channel
            n_ch = data.shape[1]
            for ch in range(n_ch):
                series = data[:, ch]
                for start in range(0, len(series) - self.seq_len + 1, stride):
                    window = series[start:start + self.seq_len]
                    self.windows.append(window)

            name = fpath.split('/')[-1]
            print(f'  {name}: {n_ch}ch, {n_train} rows → {len(range(0, n_train - self.seq_len + 1, stride)) * n_ch} windows')

    def _add_anomaly_datasets(self, root, stride):
        anom_root = os.path.join(root, 'anomaly_detection/Anomaly_Transformer')

        for ds in ['SMD', 'MSL', 'SMAP', 'PSM']:
            ds_path = os.path.join(anom_root, ds)
            if not os.path.exists(ds_path):
                continue

            if ds == 'PSM':
                data = pd.read_csv(f'{ds_path}/train.csv').fillna(0).values[:, 1:]
            else:
                data = np.load(f'{ds_path}/{ds}_train.npy')

            # Normalize
            scaler = StandardScaler()
            data = scaler.fit_transform(data)

            # Subsample if too many rows (SMD has 700K+)
            if len(data) > 100000:
                data = data[:100000]

            # Subsample channels if too many
            n_ch = data.shape[1]
            if n_ch > 20:
                ch_idx = np.random.choice(n_ch, 20, replace=False)
                data = data[:, ch_idx]
                n_ch = 20

            for ch in range(n_ch):
                series = data[:, ch]
                for start in range(0, len(series) - self.seq_len + 1, stride):
                    self.windows.append(series[start:start + self.seq_len])

            print(f'  {ds}: {n_ch}ch → {len(range(0, len(data) - self.seq_len + 1, stride)) * n_ch} windows')

    def _add_classification_datasets(self, root):
        from data_provider.data_loader import _parse_ts_file

        cls_root = os.path.join(root, 'classification/Multivariate_ts')
        if not os.path.exists(cls_root):
            return

        total = 0
        for ds_name in sorted(os.listdir(cls_root)):
            train_file = os.path.join(cls_root, ds_name, f'{ds_name}_TRAIN.ts')
            if not os.path.exists(train_file):
                continue

            try:
                arr, _, _ = _parse_ts_file(train_file)  # [N, C, T]
            except:
                continue

            n_samples, n_ch, T = arr.shape

            # Skip datasets with very short or very long series
            if T < self.seq_len // 2:
                continue

            # Normalize per channel across all samples
            for ch in range(n_ch):
                ch_data = arr[:, ch, :]  # [N, T]
                mean = np.nanmean(ch_data)
                std = np.nanstd(ch_data) + 1e-8
                ch_data = (ch_data - mean) / std

                for sample in range(n_samples):
                    series = ch_data[sample]  # [T]
                    # Pad if shorter than seq_len
                    if len(series) < self.seq_len:
                        padded = np.zeros(self.seq_len)
                        padded[:len(series)] = series
                        self.windows.append(padded)
                    else:
                        # Slide window
                        for start in range(0, len(series) - self.seq_len + 1, max(1, self.seq_len // 2)):
                            self.windows.append(series[start:start + self.seq_len])
                    total += 1

        print(f'  Classification: {total} channel-samples added')

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        window = self.windows[idx]  # [seq_len]
        # Return as [seq_len, 1] to match model expectation
        x = torch.tensor(window, dtype=torch.float32).unsqueeze(-1)  # [seq_len, 1]
        return x
