import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings
import pyarrow as pa
import pyarrow.dataset as ds
warnings.filterwarnings('ignore')
from torch.utils.data import Sampler


class RoundRobinDataLoader: 
    def __init__(self, *loaders):
        self.loaders_ = loaders
        self.loaders = [iter(loader) for loader in loaders]

    def __iter__(self):
        return self

    def __next__(self):
        # Cycle through each loader
        for loader in self.loaders:
            try:
                return next(loader)
            except StopIteration:
                # Restart the loader if it's exhausted
                self.loaders[self.loaders.index(loader)] = iter(loader.dataset)
                yield next(loader)

    def __len__(self):
        # Optional, can return the sum or the minimum of lengths depending on the exact use case
        return sum(len(loader) for loader in self.loaders)

class IterativeDataLoader: 
    def __init__(self, *loaders):
        self.loaders_ = loaders
        self.loaders = [iter(loader) for loader in loaders]
        self.active_loaders = list(range(len(self.loaders)))  # Keep track of non-exhausted loaders


    def __iter__(self):
        return self

    def __next__(self):
        if not self.active_loaders:
            self.loaders = [iter(loader) for loader in self.loaders_]
            self.active_loaders = list(range(len(self.loaders)))  # Keep track of non-exhausted loaders
            raise StopIteration
        # Cycle through each loader
        loader = self.loaders[self.active_loaders[0]]
        try:
            return next(loader)
        except StopIteration:
                self.active_loaders.remove(self.active_loaders[0])
                return self.__next__()

    def __len__(self):
        # Optional, can return the sum or the minimum of lengths depending on the exact use case
        return sum(len(loader) for loader in self.loaders)





class RandomCombinedDataLoader:
    def __init__(self, *loaders):
        self.loaders_ = loaders
        self.loaders = [iter(loader) for loader in loaders]
        self.active_loaders = list(range(len(self.loaders)))  # Keep track of non-exhausted loaders

    def __iter__(self):
        return self

    def __next__(self):
        if not self.active_loaders:
            self.loaders = [iter(loader) for loader in self.loaders_]
            self.active_loaders = list(range(len(self.loaders)))  # Keep track of non-exhausted loaders
            raise StopIteration
        
        # Randomly select a DataLoader from which to draw the next batch
        choice = np.random.choice(self.active_loaders)
        try:
            return next(self.loaders[choice])
            
        except StopIteration:
            self.active_loaders.remove(choice)
            return self.__next__()  # Recursively try to fetch next batch

    def __len__(self):
        return sum(len(loader) for loader in self.loaders)




class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        #cast everything to torch tensors if they are not 
        if isinstance(seq_x, np.ndarray):
            seq_x = torch.from_numpy(seq_x)
        
        if isinstance(seq_y, np.ndarray):
            seq_y = torch.from_numpy(seq_y)
        
        if isinstance(seq_x_mark, np.ndarray):
            seq_x_mark = torch.from_numpy(seq_x_mark)
        if isinstance(seq_y_mark, np.ndarray):
            seq_y_mark = torch.from_numpy(seq_y_mark)
        


        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
        
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]




        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_PEMS(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        data_file = os.path.join(self.root_path, self.data_path)
        data = np.load(data_file, allow_pickle=True)
        data = data['data'][:, :, 0]

        train_ratio = 0.6
        valid_ratio = 0.2
        train_data = data[:int(train_ratio * len(data))]
        valid_data = data[int(train_ratio * len(data)): int((train_ratio + valid_ratio) * len(data))]
        test_data = data[int((train_ratio + valid_ratio) * len(data)):]
        total_data = [train_data, valid_data, test_data]
        data = total_data[self.set_type]

        if self.scale:
            self.scaler.fit(train_data)
            data = self.scaler.transform(data)

        df = pd.DataFrame(data)
        df = df.fillna(method='ffill', limit=len(df)).fillna(method='bfill', limit=len(df)).values

        self.data_x = df
        self.data_y = df

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = torch.zeros((seq_x.shape[0], 1))
        seq_y_mark = torch.zeros((seq_x.shape[0], 1))




        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Solar(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = []
        with open(os.path.join(self.root_path, self.data_path), "r", encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip('\n').split(',')
                data_line = np.stack([float(i) for i in line])
                df_raw.append(data_line)
        df_raw = np.stack(df_raw, 0)
        df_raw = pd.DataFrame(df_raw)

        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_valid = int(len(df_raw) * 0.1)
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_valid, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        df_data = df_raw.values

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data)
            data = self.scaler.transform(df_data)
        else:
            data = df_data

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = torch.zeros((seq_x.shape[0], 1))
        seq_y_mark = torch.zeros((seq_x.shape[0], 1))



        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Pred(Dataset):
    def __init__(self, root_path, flag='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, inverse=False, timeenc=0, freq='15min', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['pred']

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        if self.cols:
            cols = self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns)
            cols.remove(self.target)
            cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        border1 = len(df_raw) - self.seq_len
        border2 = len(df_raw)

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        tmp_stamp = df_raw[['date']][border1:border2]
        tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)
        pred_dates = pd.date_range(tmp_stamp.date.values[-1], periods=self.pred_len + 1, freq=self.freq)

        df_stamp = pd.DataFrame(columns=['date'])
        df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = self.data_x[r_begin:r_begin + self.label_len]
        else:
            seq_y = self.data_y[r_begin:r_begin + self.label_len]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)





class Dataset_GaussianP(Dataset):
    def __init__(self, root_path=None, flag='train', size=None,
                 features='S', data_path=None,
                 target='OT', scale=True, timeenc=0, freq='h', n_variables = 10, seq_len = 192, pred_len = 96, stride = -1, noise = 0.1, synthetic_length=-1):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.features = features
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.root_path = root_path
        self.data_path = data_path
        self.n_variables = n_variables
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.noise = noise
        self.synthetic_length = synthetic_length
        if stride == -1:
            self.stride = self.n_variables//2
            self.stride = min(self.stride, 20)
        else:
            self.stride = stride
        self._read_data()


    def _read_data(self):
        # read data — supports comma-separated multiple arrow files
        if self.data_path is not None:
            paths = [p.strip() for p in self.data_path.split(",")]
            tables = []
            for p in paths:
                all_path = os.path.join(self.root_path, p)
                t = ds.dataset(all_path, format="arrow").to_table()
                tables.append(t)
                print(f"  Loaded {all_path}: {len(t)} rows")
            import pyarrow as pa
            combined = pa.concat_tables(tables)
            self.dataset = combined.to_pandas()
            self.dataset = self.dataset[:self.synthetic_length]
        else:
            raise ValueError("Data path is not provided")

    def __getitem__(self, index):
        index_var = index // (1024 - self.seq_len - self.pred_len + 1)
        index_interior = index % (1024 - self.seq_len - self.pred_len + 1)
        
        s_begin = index_interior
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len


        # get data
        data_x = self.dataset["target"][index_var*self.stride:index_var*self.stride+self.n_variables].to_list()
        #convert to numpy
        data_x = np.asarray(data_x)
        #transpose
        data_x = data_x.T
        
        seq_x = data_x[s_begin:s_end]
        seq_y = data_x[r_begin:r_end]

        #create a multiplicative noise with a gaussian distribution
        noise_x = np.random.normal(1, self.noise, seq_x.shape)

        #apply the noise to the data
        seq_x = seq_x*noise_x

        noise_y = np.random.normal(1, self.noise, seq_y.shape)
        seq_y = seq_y*noise_y

     

        #append seq_y with 0's at the beginning with the size of args.label_len


        seq_x_mark = torch.zeros((seq_x.shape[0], 4)) #TODO fix this just for trying
        seq_y_mark = torch.zeros((seq_y.shape[0], 4))
        
        if isinstance(seq_x, np.ndarray):
            seq_x = torch.from_numpy(seq_x)
        
        if isinstance(seq_y, np.ndarray):
            seq_y = torch.from_numpy(seq_y)
        


        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return ((self.dataset.shape[0] - self.n_variables)//self.stride + 1)*(1024 - self.seq_len - self.pred_len + 1)


class Dataset_GaussianPCoregionalization(Dataset):
    def __init__(self, root_path=None, flag='train', size=None,
                 features='S', data_path=None,
                 target='OT', scale=True, timeenc=0, freq='h', n_variables = 10, seq_len = 192, pred_len = 96, stride = -1, noise = 0.1, synthetic_length = 1024):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.features = features
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.root_path = root_path
        self.data_path = data_path
        self.n_variables = n_variables
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.noise = noise
        self.synthetic_length = synthetic_length
        if stride == -1:
            self.stride = 1
        else:
            self.stride = stride
        self._read_data()


    def _read_data(self):
        # read data — supports comma-separated multiple arrow files
        if self.data_path is not None:
            paths = [p.strip() for p in self.data_path.split(",")]
            tables = []
            for p in paths:
                all_path = os.path.join(self.root_path, p)
                t = ds.dataset(all_path, format="arrow").to_table()
                tables.append(t)
                print(f"  Loaded {all_path}: {len(t)} rows")
            import pyarrow as pa
            # Unify schema: keep only 'start' and 'target' columns, cast start to same type
            unified = []
            for t in tables:
                cols_to_keep = [c for c in ['start', 'target'] if c in t.column_names]
                unified.append(t.select(cols_to_keep).cast(unified[0].schema) if unified else t.select(cols_to_keep))
            combined = pa.concat_tables(unified, promote_options="permissive")
            self.dataset = combined.to_pandas()
        else:
            raise ValueError("Data path is not provided")



    def __getitem__(self, index):
        index_var = index % self.dataset.shape[0]
        index_interior = index // self.dataset.shape[0]
        
        s_begin = index_interior*self.stride
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len


        # get data
        data_x = self.dataset["target"][index_var]
        #convert to numpy
        data_x = data_x.reshape(-1,self.synthetic_length)

        #replace the nan values with the mean of the column
        col_mean = np.nanmean(data_x, axis=0)
        inds = np.where(np.isnan(data_x))
        data_x[inds] = np.take(col_mean, inds[1]) + 0.1*np.random.normal(0, 1, len(inds[1]))
        #if inds is not empty, print
        if len(inds[0]) > 0:
            print(len(inds[0]), len(inds[1]) )
            print(inds)


        #transpose
        data_x = data_x.T
        
        seq_x = data_x[s_begin:s_end]
        seq_y = data_x[r_begin:r_end]

        #create a multiplicative noise with a gaussian distribution
        noise_x = np.random.normal(1, self.noise, seq_x.shape)
        seq_x = seq_x*noise_x

        noise_y = np.random.normal(1, self.noise, seq_y.shape)
        seq_y = seq_y*noise_y

        seq_x_mark = torch.zeros((seq_x.shape[0], 4)) 
        seq_y_mark = torch.zeros((seq_y.shape[0], 4))
        
        if isinstance(seq_x, np.ndarray):
            seq_x = torch.from_numpy(seq_x)
        
        if isinstance(seq_y, np.ndarray):
            seq_y = torch.from_numpy(seq_y)
        

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return self.dataset.shape[0]*((self.synthetic_length - self.seq_len - self.pred_len)//self.stride + 1)


class Dataset_Anomaly(Dataset):
    """Anomaly detection dataset for SMD, MSL, SMAP, PSM."""
    def __init__(self, root_path, flag='train', size=None,
                 features='M', data_path='SMD', target='OT',
                 scale=True, timeenc=0, freq='h'):
        self.seq_len = size[0]
        self.pred_len = size[0]  # reconstruction: output same length as input
        self.flag = flag
        self.scale = scale
        self.root_path = root_path
        self.data_path = data_path  # SMD, MSL, SMAP, PSM
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        dataset = self.data_path  # e.g., 'SMD'

        if dataset == 'PSM':
            train_df = pd.read_csv(os.path.join(self.root_path, dataset, 'train.csv'))
            train_df = train_df.fillna(0)
            test_df = pd.read_csv(os.path.join(self.root_path, dataset, 'test.csv'))
            test_df = test_df.fillna(0)
            label_df = pd.read_csv(os.path.join(self.root_path, dataset, 'test_label.csv'))
            train_data = train_df.values[:, 1:]  # skip timestamp
            test_data = test_df.values[:, 1:]
            test_labels = label_df.values[:, 1:]
            if test_labels.ndim > 1:
                test_labels = test_labels[:, 0]
        else:
            train_data = np.load(os.path.join(self.root_path, dataset, f'{dataset}_train.npy'))
            test_data = np.load(os.path.join(self.root_path, dataset, f'{dataset}_test.npy'))
            test_labels = np.load(os.path.join(self.root_path, dataset, f'{dataset}_test_label.npy'))

        # Normalize using train statistics
        if self.scale:
            self.scaler.fit(train_data)
            train_data = self.scaler.transform(train_data)
            test_data = self.scaler.transform(test_data)

        if self.flag == 'train':
            self.data = train_data
            self.labels = np.zeros(len(train_data))
        elif self.flag == 'val':
            # Use last 20% of train as validation
            val_start = int(len(train_data) * 0.8)
            self.data = train_data[val_start:]
            self.labels = np.zeros(len(self.data))
        else:  # test
            self.data = test_data
            self.labels = test_labels

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len

        seq_x = self.data[s_begin:s_end]
        seq_y = self.data[s_begin:s_end]  # reconstruction target = input
        label = self.labels[s_begin:s_end]

        # Dummy time marks
        seq_x_mark = np.zeros((self.seq_len, 4))
        seq_y_mark = np.zeros((self.seq_len, 4))

        return seq_x, seq_y, seq_x_mark, seq_y_mark, label

    def __len__(self):
        return len(self.data) - self.seq_len + 1


def _parse_ts_file(filepath):
    """Parse UEA .ts file format into numpy arrays."""
    data = []
    labels = []
    metadata = {}

    with open(filepath, 'r') as f:
        in_data = False
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if line.startswith('@'):
                key_val = line[1:].split(' ', 1)
                key = key_val[0].lower()
                if key == 'data':
                    in_data = True
                    continue
                if len(key_val) > 1:
                    metadata[key] = key_val[1]
                continue
            if not in_data:
                continue

            # Parse data line: dim1:dim2:...:dimN
            parts = line.split(':')
            label = parts[-1].strip()
            dims = []
            for dim_str in parts[:-1]:
                values = [float(v) for v in dim_str.strip().split(',') if v.strip()]
                dims.append(values)
            data.append(dims)
            labels.append(label)

    # Convert to numpy: [n_samples, n_channels, seq_len]
    # Handle variable length by padding
    max_len = max(len(dims[0]) for dims in data)
    n_channels = len(data[0])
    n_samples = len(data)

    arr = np.zeros((n_samples, n_channels, max_len))
    for i, dims in enumerate(data):
        for j, vals in enumerate(dims):
            arr[i, j, :len(vals)] = vals

    # Encode labels as integers
    unique_labels = sorted(set(labels))
    label_map = {l: idx for idx, l in enumerate(unique_labels)}
    label_arr = np.array([label_map[l] for l in labels])

    return arr, label_arr, len(unique_labels)


class Dataset_Classification(Dataset):
    """UEA multivariate time series classification dataset."""
    def __init__(self, root_path, flag='train', size=None,
                 features='M', data_path='EthanolConcentration',
                 target='OT', scale=True, timeenc=0, freq='h'):
        self.seq_len = size[0] if size else 96
        self.flag = flag
        self.scale = scale
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        dataset_name = self.data_path
        base = os.path.join(self.root_path, dataset_name)

        train_file = os.path.join(base, f'{dataset_name}_TRAIN.ts')
        test_file = os.path.join(base, f'{dataset_name}_TEST.ts')

        train_data, train_labels, self.n_classes = _parse_ts_file(train_file)
        test_data, test_labels, _ = _parse_ts_file(test_file)

        self.n_channels = train_data.shape[1]
        self.raw_seq_len = train_data.shape[2]

        if self.flag == 'train':
            self.data = train_data  # [N, C, T]
            self.labels = train_labels
        elif self.flag == 'val':
            # Use last 20% of train as val
            n_val = max(1, int(len(train_data) * 0.2))
            self.data = train_data[-n_val:]
            self.labels = train_labels[-n_val:]
        else:  # test
            self.data = test_data
            self.labels = test_labels

        # Normalize per-channel using train stats
        if self.scale:
            # Compute stats from all train data
            train_flat = train_data.reshape(-1, train_data.shape[-1])
            self.mean = np.nanmean(train_data, axis=(0, 2), keepdims=True)  # [1, C, 1]
            self.std = np.nanstd(train_data, axis=(0, 2), keepdims=True) + 1e-8
            self.data = (self.data - self.mean) / self.std

    def __getitem__(self, index):
        sample = self.data[index]  # [C, T]
        label = self.labels[index]

        # Transpose to [T, C] to match model convention
        sample = sample.T  # [T, C]

        # Pad or truncate to seq_len
        T = sample.shape[0]
        if T >= self.seq_len:
            sample = sample[:self.seq_len]
        else:
            pad = np.zeros((self.seq_len - T, sample.shape[1]))
            sample = np.concatenate([sample, pad], axis=0)

        # Dummy marks
        mark = np.zeros((self.seq_len, 4))

        return sample, label, mark, mark

    def __len__(self):
        return len(self.data)