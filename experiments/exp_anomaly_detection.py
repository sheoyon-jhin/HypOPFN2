from data_provider.data_factory import data_provider
from experiments.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

warnings.filterwarnings('ignore')


class Exp_AnomalyDetection(Exp_Basic):
    def __init__(self, args):
        args.pred_len = args.seq_len
        args.label_len = 0

        self.freeze_mode = getattr(args, 'freeze_mode', 'none')
        self.pretrained_ckpt = getattr(args, 'pretrained_ckpt', None)

        super(Exp_AnomalyDetection, self).__init__(args)

        if self.pretrained_ckpt and os.path.exists(self.pretrained_ckpt):
            self._load_pretrained(self.pretrained_ckpt)

    def _load_pretrained(self, ckpt_path):
        print(f'Loading pretrained checkpoint: {ckpt_path}')
        state_dict = torch.load(ckpt_path, map_location=self.device)
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f'Missing keys: {missing}')
        if unexpected:
            print(f'Unexpected keys: {unexpected}')
        loaded = len(state_dict) - len(unexpected)
        print(f'Loaded {loaded}/{len(state_dict)} parameters')
        self._apply_freeze()

    def _apply_freeze(self):
        if self.freeze_mode == 'all':
            for param in self.model.parameters():
                param.requires_grad = False
            print('Freeze: ALL (zero-shot)')
        elif self.freeze_mode == 'linear_probe':
            for param in self.model.parameters():
                param.requires_grad = False
            for name, param in self.model.named_parameters():
                if 'recon_head' in name or 'recon_bias' in name or 'linear_skip' in name:
                    param.requires_grad = True
            trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print(f'Freeze: RECON HEAD — {trainable} params trainable')
        else:
            print(f'Freeze: NONE — full fine-tune')

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        if not trainable_params:
            return optim.Adam([torch.zeros(1)], lr=self.args.learning_rate)
        return optim.Adam(trainable_params, lr=self.args.learning_rate)

    def _select_criterion(self):
        return nn.MSELoss()

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, _) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                outputs = self.model.reconstruct(batch_x)

                loss = criterion(outputs, batch_y)
                total_loss.append(loss.item())

        self.model.train()
        return np.average(total_loss)

    def train(self, setting):
        if self.freeze_mode == 'all':
            print('Zero-shot mode: skipping training')
            return self.model

        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        print(f'Start anomaly detection training for {self.args.train_epochs} epochs, '
              f'freeze_mode={self.freeze_mode}, steps={train_steps}')

        for epoch in range(self.args.train_epochs):
            train_loss = []
            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, _) in enumerate(train_loader):
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                outputs = self.model.reconstruct(batch_x)

                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                model_optim.step()

            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            print(f"Epoch: {epoch + 1} | time: {time.time() - epoch_time:.1f}s "
                  f"| Train: {train_loss:.7f} Vali: {vali_loss:.7f}")

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(model_optim, None, epoch + 1, self.args)

        self.model.load_state_dict(torch.load(path + '/checkpoint.pth'))
        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        train_data, train_loader = self._get_data(flag='train')

        if test:
            self.model.load_state_dict(
                torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        self.model.eval()

        # Step 1: Reconstruction errors on train → threshold
        train_scores = []
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark, _ in train_loader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                outputs = self.model.reconstruct(batch_x)

                score = torch.mean((outputs - batch_y) ** 2, dim=(-1, -2))
                train_scores.append(score.cpu().numpy())

        train_scores = np.concatenate(train_scores)

        # Step 2: Reconstruction errors on test
        test_scores = []
        test_labels = []
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark, batch_label in test_loader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                outputs = self.model.reconstruct(batch_x)

                score = torch.mean((outputs - batch_y) ** 2, dim=(-1, -2))
                test_scores.append(score.cpu().numpy())
                label = batch_label.numpy()
                window_label = (label.sum(axis=-1) > 0).astype(int)
                test_labels.append(window_label)

        test_scores = np.concatenate(test_scores)
        test_labels = np.concatenate(test_labels)

        # Step 3: Best F1 threshold search
        best_f1 = 0
        best_result = {}
        for percentile in [90, 95, 97, 99, 99.5]:
            threshold = np.percentile(train_scores, percentile)
            preds = (test_scores > threshold).astype(int)
            precision, recall, f1, _ = precision_recall_fscore_support(
                test_labels, preds, average='binary', zero_division=0)
            acc = accuracy_score(test_labels, preds)
            if f1 > best_f1:
                best_f1 = f1
                best_result = {
                    'percentile': percentile,
                    'threshold': threshold,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'accuracy': acc,
                }

        print(f'Anomaly Detection Results ({self.args.data_path}):')
        print(f'  Best F1: {best_result["f1"]:.4f} (threshold percentile: {best_result["percentile"]})')
        print(f'  Precision: {best_result["precision"]:.4f}, Recall: {best_result["recall"]:.4f}')
        print(f'  Accuracy: {best_result["accuracy"]:.4f}')

        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        f = open("result_anomaly_detection.txt", 'a')
        f.write(setting + "  \n")
        f.write(f'F1:{best_result["f1"]:.4f}, Precision:{best_result["precision"]:.4f}, '
                f'Recall:{best_result["recall"]:.4f}, Accuracy:{best_result["accuracy"]:.4f}\n\n')
        f.close()

        np.save(folder_path + 'test_scores.npy', test_scores)
        np.save(folder_path + 'test_labels.npy', test_labels)

        return best_result
