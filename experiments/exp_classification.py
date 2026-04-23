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
from sklearn.metrics import accuracy_score, f1_score

warnings.filterwarnings('ignore')


class ClassificationHead(nn.Module):
    """Classification head on top of encoder representation."""
    def __init__(self, input_dim, n_classes):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        return self.head(x)


class Exp_Classification(Exp_Basic):
    def __init__(self, args):
        args.pred_len = args.seq_len
        args.label_len = 0

        self.freeze_mode = getattr(args, 'freeze_mode', 'none')
        self.pretrained_ckpt = getattr(args, 'pretrained_ckpt', None)

        super(Exp_Classification, self).__init__(args)

        # Get dataset info
        train_data, _ = self._get_data(flag='train')
        self.n_classes = train_data.n_classes
        self.n_channels = train_data.n_channels

        # Get encoder hidden dim from model
        branch_hidden = self.model.branch_hidden

        # Classification head: encoder repr pooled over channels → classes
        # Input: [B, n_channels * branch_hidden] (flattened) or [B, branch_hidden] (channel-pooled)
        # Use channel-pooled for simplicity and channel-independence
        self.cls_head = ClassificationHead(branch_hidden, self.n_classes).to(self.device)

        # Load pretrained if available
        if self.pretrained_ckpt and os.path.exists(self.pretrained_ckpt):
            self._load_pretrained(self.pretrained_ckpt)

    def _load_pretrained(self, ckpt_path):
        print(f'Loading pretrained checkpoint: {ckpt_path}')
        state_dict = torch.load(ckpt_path, map_location=self.device)
        # Use model's custom load_state_dict which handles legacy branch_net → encoder mapping
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f'Missing keys: {missing}')
        if unexpected:
            print(f'Unexpected keys: {unexpected}')
        loaded = len(state_dict) - len(unexpected)
        print(f'Loaded {loaded}/{len(state_dict)} parameters')
        self._apply_freeze()

    def _apply_freeze(self):
        if self.freeze_mode == 'all' or self.freeze_mode == 'linear_probe':
            for param in self.model.parameters():
                param.requires_grad = False
            trainable = sum(p.numel() for p in self.cls_head.parameters())
            print(f'Freeze: backbone frozen, cls_head trainable ({trainable} params)')
        else:
            total = sum(p.numel() for p in self.model.parameters())
            head = sum(p.numel() for p in self.cls_head.parameters())
            print(f'Freeze: NONE — backbone ({total}) + cls_head ({head}) trainable')

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        params = []
        backbone_params = [p for p in self.model.parameters() if p.requires_grad]
        if backbone_params:
            params.append({'params': backbone_params, 'lr': self.args.learning_rate * 0.1})
        params.append({'params': self.cls_head.parameters(), 'lr': self.args.learning_rate})
        return optim.Adam(params)

    def _extract_features(self, batch_x):
        """Extract encoder representation for classification.

        Uses model.get_representation() → [B, n_channels, branch_hidden]
        Then pool over channels → [B, branch_hidden]
        """
        z = self.model.get_representation(batch_x)  # [B, C, hidden]
        # Mean pool over channels
        features = z.mean(dim=1)  # [B, hidden]
        return features

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        all_preds = []
        all_labels = []
        self.model.eval()
        self.cls_head.eval()

        with torch.no_grad():
            for batch_x, batch_label, batch_x_mark, _ in vali_loader:
                batch_x = batch_x.float().to(self.device)
                batch_label = batch_label.long().to(self.device)

                features = self._extract_features(batch_x)
                logits = self.cls_head(features)
                loss = criterion(logits, batch_label)

                total_loss.append(loss.item())
                all_preds.append(logits.argmax(dim=-1).cpu().numpy())
                all_labels.append(batch_label.cpu().numpy())

        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        acc = accuracy_score(all_labels, all_preds)

        self.model.train()
        self.cls_head.train()
        return np.average(total_loss), acc

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()
        criterion = nn.CrossEntropyLoss()

        print(f'Classification training: {self.args.data_path}, '
              f'n_classes={self.n_classes}, n_channels={self.n_channels}, '
              f'freeze_mode={self.freeze_mode}')

        for epoch in range(self.args.train_epochs):
            train_loss = []
            self.model.train()
            self.cls_head.train()
            epoch_time = time.time()

            for i, (batch_x, batch_label, batch_x_mark, _) in enumerate(train_loader):
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_label = batch_label.long().to(self.device)

                features = self._extract_features(batch_x)
                logits = self.cls_head(features)
                loss = criterion(logits, batch_label)

                train_loss.append(loss.item())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.model.parameters()) + list(self.cls_head.parameters()), 1.0)
                model_optim.step()

            train_loss = np.average(train_loss)
            vali_loss, vali_acc = self.vali(vali_data, vali_loader, criterion)
            print(f"Epoch: {epoch + 1} | time: {time.time() - epoch_time:.1f}s "
                  f"| Train: {train_loss:.4f} Vali: {vali_loss:.4f} Acc: {vali_acc:.4f}")

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            torch.save(self.cls_head.state_dict(), path + '/cls_head.pth')
            adjust_learning_rate(model_optim, None, epoch + 1, self.args)

        self.model.load_state_dict(torch.load(path + '/checkpoint.pth'))
        if os.path.exists(path + '/cls_head.pth'):
            self.cls_head.load_state_dict(torch.load(path + '/cls_head.pth'))
        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            ckpt_dir = './checkpoints/' + setting
            self.model.load_state_dict(torch.load(os.path.join(ckpt_dir, 'checkpoint.pth')))
            if os.path.exists(os.path.join(ckpt_dir, 'cls_head.pth')):
                self.cls_head.load_state_dict(torch.load(os.path.join(ckpt_dir, 'cls_head.pth')))

        self.model.eval()
        self.cls_head.eval()

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch_x, batch_label, batch_x_mark, _ in test_loader:
                batch_x = batch_x.float().to(self.device)

                features = self._extract_features(batch_x)
                logits = self.cls_head(features)

                all_preds.append(logits.argmax(dim=-1).cpu().numpy())
                all_labels.append(batch_label.numpy())

        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)

        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

        print(f'Classification Results ({self.args.data_path}):')
        print(f'  Accuracy: {acc:.4f}, Macro F1: {f1:.4f}')

        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        f = open("result_classification.txt", 'a')
        f.write(setting + "  \n")
        f.write(f'Accuracy:{acc:.4f}, F1:{f1:.4f}\n\n')
        f.close()

        np.save(folder_path + 'preds.npy', all_preds)
        np.save(folder_path + 'labels.npy', all_labels)

        return acc, f1
