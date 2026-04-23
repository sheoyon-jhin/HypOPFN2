from data_provider.data_factory import data_provider
from experiments.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np

warnings.filterwarnings('ignore')


class Exp_Imputation(Exp_Basic):
    def __init__(self, args):
        self.original_pred_len = getattr(args, 'pred_len', args.seq_len)
        args.pred_len = args.seq_len
        args.label_len = 0

        self.freeze_mode = getattr(args, 'freeze_mode', 'none')
        self.pretrained_ckpt = getattr(args, 'pretrained_ckpt', None)

        super(Exp_Imputation, self).__init__(args)

        if self.pretrained_ckpt and os.path.exists(self.pretrained_ckpt):
            self._load_pretrained(self.pretrained_ckpt)

    def _load_pretrained(self, ckpt_path):
        print(f'Loading pretrained checkpoint: {ckpt_path}')
        state_dict = torch.load(ckpt_path, map_location=self.device)
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f'Missing keys (randomly initialized): {missing}')
        if unexpected:
            print(f'Unexpected keys: {unexpected}')
        loaded = len(state_dict) - len(unexpected)
        print(f'Loaded {loaded}/{len(state_dict)} parameters')
        self._apply_freeze()

    def _apply_freeze(self):
        if self.freeze_mode == 'all':
            for param in self.model.parameters():
                param.requires_grad = False
            print('Freeze mode: ALL parameters frozen (zero-shot)')
        elif self.freeze_mode == 'linear_probe':
            # Freeze encoder, unfreeze recon_head + linear_skip
            for param in self.model.parameters():
                param.requires_grad = False
            for name, param in self.model.named_parameters():
                if 'recon_head' in name or 'recon_bias' in name or 'linear_skip' in name:
                    param.requires_grad = True
            trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.model.parameters())
            print(f'Freeze mode: RECON HEAD — {trainable}/{total} params trainable')
        else:
            trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print(f'Freeze mode: NONE — all {trainable} params trainable')

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
        return nn.MSELoss(reduction='none')

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)

                mask = torch.rand_like(batch_x) > self.args.mask_rate
                mask = mask.float()
                masked_input = batch_x * mask

                # Use reconstruct() with recon_head
                outputs = self.model.reconstruct(masked_input)

                loss_matrix = criterion(outputs, batch_x)
                loss = (loss_matrix * (1 - mask)).sum() / (1 - mask).sum()
                total_loss.append(loss.item())

        self.model.train()
        return np.average(total_loss)

    def train(self, setting):
        if self.freeze_mode == 'all':
            print('Zero-shot mode: skipping training')
            return self.model

        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        print(f'Start imputation training for {self.args.train_epochs} epochs, '
              f'mask_rate={self.args.mask_rate}, freeze_mode={self.freeze_mode}, '
              f'train_steps={train_steps}')

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)

                mask = torch.rand_like(batch_x) > self.args.mask_rate
                mask = mask.float()
                masked_input = batch_x * mask

                # Use reconstruct() with recon_head
                outputs = self.model.reconstruct(masked_input)

                loss_matrix = criterion(outputs, batch_x)
                loss = (loss_matrix * (1 - mask)).sum() / (1 - mask).sum()

                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print(f"\titers: {i + 1}, epoch: {epoch + 1} | loss: {loss.item():.7f}")
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print(f'\tspeed: {speed:.4f}s/iter; left time: {left_time:.4f}s')
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                model_optim.step()

            print(f"Epoch: {epoch + 1} cost time: {time.time() - epoch_time:.1f}s")
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print(f"Epoch: {epoch + 1} | Train Loss: {train_loss:.7f} "
                  f"Vali Loss: {vali_loss:.7f} Test Loss: {test_loss:.7f}")

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, None, epoch + 1, self.args)

        best_model_path = path + '/checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(
                torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        masks = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        torch.manual_seed(2021)
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)

                mask = torch.rand_like(batch_x) > self.args.mask_rate
                mask = mask.float()
                masked_input = batch_x * mask

                # Use reconstruct() with recon_head
                outputs = self.model.reconstruct(masked_input)

                outputs = outputs.detach().cpu().numpy()
                batch_x_np = batch_x.detach().cpu().numpy()
                mask_np = mask.detach().cpu().numpy()

                preds.append(outputs)
                trues.append(batch_x_np)
                masks.append(mask_np)

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        masks = np.concatenate(masks, axis=0)
        print(f'test shape: preds={preds.shape}, trues={trues.shape}, masks={masks.shape}')

        masked_preds = preds[masks == 0]
        masked_trues = trues[masks == 0]
        mse = np.mean((masked_preds - masked_trues) ** 2)
        mae = np.mean(np.abs(masked_preds - masked_trues))

        print(f'Imputation results — mse:{mse:.6f}, mae:{mae:.6f}')

        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        f = open("result_imputation.txt", 'a')
        f.write(setting + "  \n")
        f.write(f'mse:{mse}, mae:{mae}\n\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mse, mae]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return mse, mae
