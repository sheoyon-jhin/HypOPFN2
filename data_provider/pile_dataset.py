"""
Time Series Pile Dataset for pre-training.
Loads all data from MOMENT's Time Series Pile (HuggingFace).

Sources:
  - forecasting/autoformer: ETT, Weather, Traffic, Electricity, Exchange, ILI (CSV)
  - forecasting/monash: 48 datasets (TSF format)
  - classification/UCR: 158 datasets (TS format)
  - anomaly_detection/TSB-UAD: 19 subdirs (CSV)
"""
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


def parse_tsf_file(filepath, max_length=2048):
    """Parse Monash .tsf format → list of numpy arrays."""
    series_list = []
    with open(filepath, 'r') as f:
        in_data = False
        for line in f:
            line = line.strip()
            if line.startswith('@data'):
                in_data = True
                continue
            if not in_data or not line or line.startswith('@') or line.startswith('#'):
                continue
            # Parse: values separated by comma, last might be frequency info
            parts = line.split(':')
            if len(parts) >= 2:
                values_str = parts[-1].strip()
            else:
                values_str = line
            try:
                values = [float(v) for v in values_str.split(',') if v.strip() and v.strip() != '?']
                if len(values) > 10:  # skip very short
                    if len(values) > max_length:
                        values = values[:max_length]
                    series_list.append(np.array(values))
            except:
                continue
    return series_list


def parse_ts_file_simple(filepath):
    """Parse UCR/UEA .ts format → list of numpy arrays (one per channel)."""
    series_list = []
    with open(filepath, 'r') as f:
        in_data = False
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('@'):
                if '@data' in line.lower():
                    in_data = True
                continue
            if not in_data:
                continue
            # Each line: dim1:dim2:...:label
            parts = line.split(':')
            for dim_str in parts[:-1]:  # skip label
                try:
                    values = [float(v) for v in dim_str.strip().split(',') if v.strip()]
                    if len(values) > 5:
                        series_list.append(np.array(values))
                except:
                    continue
    return series_list


class PilePretrainDataset(Dataset):
    """Time Series Pile for pre-training via masked reconstruction."""

    def __init__(self, seq_len=96, stride=48, pile_root='./dataset/time_series_pile',
                 max_series_per_source=50000, skip_anomaly=False):
        self.seq_len = seq_len
        self.windows = []

        print('Loading Time Series Pile...')

        # 1. Forecasting - autoformer (CSV)
        self._load_csv_forecasting(pile_root, stride)

        # 2. Forecasting - monash (TSF)
        self._load_monash(pile_root, stride, max_series_per_source)

        # 3. Classification - UCR (TS)
        self._load_ucr(pile_root)

        # 4. Anomaly - TSB-UAD (CSV) — optional
        if not skip_anomaly:
            self._load_anomaly(pile_root, stride)
        else:
            print('  Anomaly TSB-UAD: SKIPPED')

        self.windows = np.array(self.windows, dtype=np.float32)
        print(f'\nTotal: {len(self.windows)} windows')

    def _normalize_and_window(self, data_1d, stride):
        """Normalize a 1D series and extract windows."""
        if len(data_1d) < self.seq_len:
            return
        # Z-normalize
        mean, std = np.mean(data_1d), np.std(data_1d)
        if std < 1e-8:
            return
        data_1d = (data_1d - mean) / std
        # Replace nan/inf
        data_1d = np.nan_to_num(data_1d, nan=0.0, posinf=3.0, neginf=-3.0)
        # Clip outliers
        data_1d = np.clip(data_1d, -10, 10)
        # Extract windows
        for start in range(0, len(data_1d) - self.seq_len + 1, stride):
            self.windows.append(data_1d[start:start + self.seq_len])

    def _load_csv_forecasting(self, pile_root, stride):
        csv_dir = os.path.join(pile_root, 'forecasting/autoformer')
        if not os.path.exists(csv_dir):
            return
        count = 0
        for fname in os.listdir(csv_dir):
            if not fname.endswith('.csv'):
                continue
            df = pd.read_csv(os.path.join(csv_dir, fname))
            data = df.iloc[:, 1:].values  # skip date column
            # Train split: first 70%
            n_train = int(len(data) * 0.7)
            data = data[:n_train]
            n_ch = min(data.shape[1], 50)  # limit channels
            for ch in range(n_ch):
                self._normalize_and_window(data[:, ch], stride)
                count += 1
        print(f'  Forecasting CSV: {count} channels loaded')

    def _load_monash(self, pile_root, stride, max_series):
        monash_dir = os.path.join(pile_root, 'forecasting/monash')
        if not os.path.exists(monash_dir):
            return
        count = 0
        for fname in sorted(os.listdir(monash_dir)):
            if not fname.endswith('.tsf'):
                continue
            try:
                series_list = parse_tsf_file(os.path.join(monash_dir, fname))
                for series in series_list[:2000]:  # limit per file
                    self._normalize_and_window(series, stride)
                    count += 1
                    if count >= max_series:
                        break
            except:
                continue
            if count >= max_series:
                break
        print(f'  Monash: {count} series loaded')

    def _load_ucr(self, pile_root):
        ucr_dir = os.path.join(pile_root, 'classification/UCR')
        if not os.path.exists(ucr_dir):
            return
        count = 0
        for ds_name in sorted(os.listdir(ucr_dir)):
            ds_path = os.path.join(ucr_dir, ds_name)
            if not os.path.isdir(ds_path):
                continue
            for split_file in os.listdir(ds_path):
                if not split_file.endswith('_TRAIN.ts'):
                    continue
                try:
                    series_list = parse_ts_file_simple(os.path.join(ds_path, split_file))
                    for series in series_list:
                        if len(series) >= self.seq_len:
                            self._normalize_and_window(series, max(1, self.seq_len // 2))
                        elif len(series) > 10:
                            # Pad short series
                            padded = np.zeros(self.seq_len)
                            padded[:len(series)] = (series - np.mean(series)) / (np.std(series) + 1e-8)
                            padded = np.nan_to_num(padded, nan=0.0)
                            padded = np.clip(padded, -10, 10)
                            self.windows.append(padded)
                        count += 1
                except:
                    continue
        print(f'  UCR Classification: {count} channel-samples loaded')

    def _load_anomaly(self, pile_root, stride):
        anom_dir = os.path.join(pile_root, 'anomaly_detection/TSB-UAD-Public')
        if not os.path.exists(anom_dir):
            return
        count = 0
        for subdir in sorted(os.listdir(anom_dir)):
            subdir_path = os.path.join(anom_dir, subdir)
            if not os.path.isdir(subdir_path):
                continue
            for fname in os.listdir(subdir_path):
                if not fname.endswith('.out') and not fname.endswith('.csv'):
                    continue
                try:
                    data = np.loadtxt(os.path.join(subdir_path, fname), delimiter=',')
                    if data.ndim == 1:
                        self._normalize_and_window(data, stride)
                    else:
                        for ch in range(min(data.shape[1], 10)):
                            self._normalize_and_window(data[:, ch], stride)
                    count += 1
                except:
                    continue
        print(f'  Anomaly TSB-UAD: {count} files loaded')

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        window = self.windows[idx]
        x = torch.tensor(window, dtype=torch.float32).unsqueeze(-1)  # [seq_len, 1]
        return x
