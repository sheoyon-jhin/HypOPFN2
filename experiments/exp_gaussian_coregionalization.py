from data_provider.data_factory import data_provider
from experiments.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import os
import time
import warnings
import numpy as np
import random
from torch.optim import lr_scheduler


class QuantileLoss(nn.Module):
    """Quantile regression loss over multiple quantile levels.
    Trains the model to output the median (0.5 quantile) while
    being evaluated on the full quantile grid {0.1, ..., 0.9}.
    This encourages calibrated uncertainty via asymmetric penalties.
    """
    def __init__(self, quantiles=None):
        super().__init__()
        if quantiles is None:
            quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        self.quantiles = quantiles

    def forward(self, pred, target):
        losses = []
        for q in self.quantiles:
            errors = target - pred
            losses.append(torch.max((q - 1) * errors, q * errors))
        return torch.mean(torch.stack(losses))
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
warnings.filterwarnings('ignore')


class Exp_GaussianPCoregionalization(Exp_Basic):
    def __init__(self, args):
        super(Exp_GaussianPCoregionalization, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        if self.args.optimizer == 'adam':
            model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        elif self.args.optimizer == 'adamw':
            model_optim = optim.AdamW(self.model.parameters(), lr=self.args.learning_rate)
        elif self.args.optimizer == 'sgd':
            model_optim = optim.SGD(self.model.parameters(), lr=self.args.learning_rate, momentum=0.9)
        else:
            raise ValueError('Invalid optimizer')
        return model_optim

    def _select_criterion(self):
        if getattr(self.args, 'loss', 'MSE') == 'quantile':
            return QuantileLoss()
        if getattr(self.args, 'loss', 'MSE') == 'gaussian_nll':
            return nn.GaussianNLLLoss()
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        preds = []
        trues = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        model_out = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    model_out = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                # Handle Gaussian NLL: model returns (mean, log_var)
                if isinstance(model_out, tuple):
                    outputs, log_var = model_out
                else:
                    outputs = model_out
                    if self.args.output_attention:
                        outputs = outputs[0] if isinstance(outputs, tuple) else outputs
                    log_var = None

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                if log_var is not None:
                    log_var = log_var[:, -self.args.pred_len:, f_dim:]
                    loss = criterion(pred, true, torch.exp(log_var.detach().cpu()))
                else:
                    loss = criterion(pred, true)

                total_loss.append(loss)

                preds.append(pred.numpy())
                trues.append(true.numpy())
        total_loss = np.average(total_loss)
        preds = np.array(preds)
        trues = np.array(trues)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

        mae, mse, rmse, mape, mspe = metric(preds, trues)


        self.model.train()
        return total_loss, mae, mse, rmse, mape, mspe

    def train(self, setting):
        print("Started training with gaussian coregionalization")
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)
        
                #load the model from the path
        if self.args.load:
            self.model.load_state_dict(torch.load(self.args.load_path))
            #get the test and vali loss
            criterion = self._select_criterion()
            vali_loss, vali_mae, vali_mse, vali_rmse, vali_mape, vali_mspe = self.vali(vali_data, vali_loader, criterion)
            test_loss, test_mae, test_mse, test_rmse, test_mape, test_mspe = self.vali(test_data, test_loader, criterion)
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                0, 0, 0, vali_loss, test_loss))
            f = open("result_long_term_forecast.txt", 'a')
            f.write("zero-shot-" + setting + "  \n")
            f.write('mse:{}, mae:{}'.format(test_loss, test_mae))
            f.write('\n')
            f.write('\n')
            f.close()
        
        print("Loaded data loaders")
        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()


        if self.args.lradj == 'synthetic':
            scheduler = lr_scheduler.OneCycleLR(optimizer = model_optim,
                                                steps_per_epoch = train_steps,
                                                pct_start = self.args.pct_start,
                                                epochs = self.args.train_epochs,
                                                max_lr = self.args.learning_rate)
        else:
            scheduler = None

        # Operator learning training modes
        multi_res = getattr(self.args, 'multi_resolution', False)
        irregular_query = getattr(self.args, 'irregular_query', False)
        cross_freq = getattr(self.args, 'cross_frequency', False)
        var_seq = getattr(self.args, 'variable_seq_len', False)
        pretrain_mask_rate = getattr(self.args, 'pretrain_mask_rate', 0.0)

        if multi_res:
            pred_len_candidates = [24, 48, 96, 192, 336, 720]
            max_target = self.args.label_len + self.args.pred_len
            pred_len_candidates = [p for p in pred_len_candidates if p <= max_target - self.args.label_len]
            print(f'Multi-resolution training enabled: pred_len candidates = {pred_len_candidates}')
        else:
            pred_len_candidates = None

        if irregular_query:
            print(f'Irregular query training enabled: random time points per batch')
        if cross_freq:
            subsample_factors = [1, 2, 3, 4]
            print(f'Cross-frequency training enabled: subsample factors = {subsample_factors}')
        if var_seq:
            seq_len_candidates = [s for s in [48, 64, 96, 128, 192] if s <= self.args.seq_len]
            print(f'Variable seq_len training enabled: candidates = {seq_len_candidates}')
        if pretrain_mask_rate > 0:
            mask_criterion = nn.MSELoss(reduction='none')
            print(f'Masked reconstruction enabled: mask_rate={pretrain_mask_rate}')

        print(f'Start training for {self.args.train_epochs} epochs with train_steps: {train_steps}')

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                # --- Operator learning augmentations ---

                # 1. Multi-resolution: random pred_len
                if multi_res and pred_len_candidates:
                    cur_pred_len = random.choice(pred_len_candidates)
                else:
                    cur_pred_len = self.args.pred_len

                # 2. Cross-frequency: subsample input and target
                if cross_freq:
                    factor = random.choice(subsample_factors)
                    if factor > 1:
                        # Subsample batch_x: take every factor-th step, then pad back to seq_len
                        subsampled_x = batch_x[:, ::factor, :]  # [batch, seq_len//factor, ch]
                        pad_len = self.args.seq_len - subsampled_x.shape[1]
                        if pad_len > 0:
                            batch_x = F.pad(subsampled_x, (0, 0, pad_len, 0))  # left-pad with zeros
                        else:
                            batch_x = subsampled_x[:, -self.args.seq_len:, :]
                        # Subsample batch_y similarly
                        subsampled_y = batch_y[:, ::factor, :]
                        # Ensure enough target length
                        needed = self.args.label_len + cur_pred_len
                        if subsampled_y.shape[1] < needed:
                            cur_pred_len = max(subsampled_y.shape[1] - self.args.label_len, 12)
                        batch_y = subsampled_y

                # 3. Variable seq_len: randomly truncate input
                if var_seq:
                    cur_seq_len = random.choice(seq_len_candidates)
                    if cur_seq_len < self.args.seq_len:
                        # Zero-pad the beginning, keep last cur_seq_len values
                        mask = torch.zeros_like(batch_x)
                        mask[:, -cur_seq_len:, :] = 1.0
                        batch_x = batch_x * mask

                # 4. Irregular query: random time points
                query_points = None
                if irregular_query:
                    n_query = cur_pred_len
                    query_points = torch.sort(torch.rand(n_query, device=self.device))[0]

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # --- Forward pass ---
                f_dim = -1 if self.args.features == 'MS' else 0

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,
                                             target_pred_len=cur_pred_len, query_points=query_points)
                        if self.args.output_attention:
                            outputs = outputs[0]
                        outputs = outputs[:, -cur_pred_len:, f_dim:]

                        if irregular_query and query_points is not None:
                            # GT at irregular points: interpolate from regular grid
                            target_full = batch_y[:, self.args.label_len:self.args.label_len + cur_pred_len, f_dim:]
                            # query_points are in [0,1], map to indices
                            indices = (query_points * (cur_pred_len - 1)).long().clamp(0, target_full.shape[1] - 1)
                            target_y = target_full[:, indices, :]
                        else:
                            target_y = batch_y[:, self.args.label_len:self.args.label_len + cur_pred_len, f_dim:].to(self.device)

                        loss = criterion(outputs, target_y)
                        train_loss.append(loss.item())
                else:
                    model_out = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,
                                         target_pred_len=cur_pred_len, query_points=query_points)

                    # Handle Gaussian NLL: model returns (mean, log_var)
                    if isinstance(model_out, tuple):
                        outputs, log_var = model_out
                        outputs = outputs[:, -cur_pred_len:, f_dim:]
                        log_var = log_var[:, -cur_pred_len:, f_dim:]
                    else:
                        outputs = model_out
                        if self.args.output_attention:
                            outputs = outputs[0]
                        outputs = outputs[:, -cur_pred_len:, f_dim:]
                        log_var = None

                    if irregular_query and query_points is not None:
                        target_full = batch_y[:, self.args.label_len:self.args.label_len + cur_pred_len, f_dim:]
                        indices = (query_points * (cur_pred_len - 1)).long().clamp(0, target_full.shape[1] - 1)
                        target_y = target_full[:, indices, :]
                    else:
                        target_y = batch_y[:, self.args.label_len:self.args.label_len + cur_pred_len, f_dim:].to(self.device)

                    if log_var is not None:
                        # GaussianNLLLoss expects (input, target, var)
                        loss = criterion(outputs, target_y, torch.exp(log_var))
                    else:
                        loss = criterion(outputs, target_y)
                    # Add MoE auxiliary loss if available
                    if hasattr(self.model, 'get_auxiliary_loss'):
                        aux_loss = self.model.get_auxiliary_loss()
                        loss = loss + aux_loss

                    # --- Masked Reconstruction Loss ---
                    if pretrain_mask_rate > 0:
                        # Mask random positions in the input
                        mask = (torch.rand_like(batch_x) > pretrain_mask_rate).float()  # 1=observed, 0=masked
                        masked_input = batch_x * mask

                        # Forward with masked input, reconstruct full sequence (pred_len=seq_len)
                        dec_inp_mask = torch.zeros_like(masked_input)
                        recon_out = self.model(masked_input, batch_x_mark, dec_inp_mask, batch_x_mark,
                                               target_pred_len=self.args.seq_len)
                        if isinstance(recon_out, tuple):
                            recon_out = recon_out[0]
                        recon_out = recon_out[:, -self.args.seq_len:, f_dim:]

                        # MSE only at masked positions
                        recon_target = batch_x[:, :self.args.seq_len, f_dim:]
                        recon_loss_matrix = mask_criterion(recon_out, recon_target)
                        inv_mask = 1.0 - mask[:, :self.args.seq_len, f_dim:]
                        n_masked = inv_mask.sum().clamp(min=1.0)
                        recon_loss = (recon_loss_matrix * inv_mask).sum() / n_masked

                        loss = loss + recon_loss

                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if (i + 1)% 2500 == 0:
                    self.test(setting, test=0)
                    train_loss_ = np.average(train_loss)
                    vali_loss, vali_mae, vali_mse, vali_rmse, vali_mape, vali_mspe  = self.vali(vali_data, vali_loader, criterion)
                    test_loss, test_mae, test_mse, test_rmse, test_mape, test_mspe = self.vali(test_data, test_loader, criterion)
                    print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                        epoch + 1, train_steps, train_loss_, vali_loss, test_loss))
                    early_stopping(vali_loss, self.model, path)
                    if early_stopping.early_stop:
                        print("Early stopping")
                        break


                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    model_optim.step()
                
                if self.args.lradj == 'synthetic' and scheduler is not None:
                    scheduler.step()
            
            if early_stopping.early_stop:
                print("Early stopping")
                break

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss, vali_mae, vali_mse, vali_rmse, vali_mape, vali_mspe  = self.vali(vali_data, vali_loader, criterion)
            test_loss, test_mae, test_mse, test_rmse, test_mape, test_mspe = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            # 매 epoch 마지막 체크포인트도 항상 저장 (사전학습 시 real data val loss가 불안정하므로)
            torch.save(self.model.state_dict(), path + '/last_epoch_checkpoint.pth')
            print(f"Last epoch checkpoint saved to {path}/last_epoch_checkpoint.pth")
            if WANDB_AVAILABLE and wandb.run is not None:
                wandb.log({"train_loss": train_loss , "vali_loss": vali_loss, "test_loss": test_loss , "vali_mae": vali_mae, "vali_mse": vali_mse, "vali_rmse": vali_rmse, "vali_mape": vali_mape, "vali_mspe": vali_mspe, "test_mae": test_mae, "test_mse": test_mse, "test_rmse": test_rmse, "test_mape": test_mape, "test_mspe": test_mspe})
            print(f"train_loss: {train_loss} vali_loss: {vali_loss} test_loss: {test_loss} vali_mae: {vali_mae} vali_mse: {vali_mse} vali_rmse: {vali_rmse} vali_mape: {vali_mape} vali_mspe: {vali_mspe} test_mae: {test_mae} test_mse: {test_mse} test_rmse: {test_rmse} test_mape: {test_mape} test_mspe: {test_mspe}")

            if early_stopping.early_stop:
                print("Early stopping")
                break

            if self.args.lradj != 'synthetic':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))


        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        model_out = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    model_out = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                # Handle Gaussian NLL: model returns (mean, log_var)
                if isinstance(model_out, tuple):
                    outputs, log_var = model_out
                else:
                    outputs = model_out
                    if self.args.output_attention:
                        outputs = outputs[0] if isinstance(outputs, tuple) else outputs

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = test_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.squeeze(0)).reshape(shape)

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.squeeze(0)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()


        return


    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                outputs = outputs.detach().cpu().numpy()
                if pred_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = pred_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                preds.append(outputs)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return