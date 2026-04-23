"""
독립 실험: Cross-channel attention 효과 테스트
기존 코드 안 건드리고 따로 실행

사용법:
  source /opt/miniforge3/etc/profile.d/conda.sh && conda activate timefound
  CUDA_VISIBLE_DEVICES=0 python experiments/exp_cross_channel_test.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import optim
from data_provider.data_factory import data_provider
from types import SimpleNamespace
from model.DeepONetHyperMoE import Model as BaseModel


class CrossChannelModel(nn.Module):
    """BaseModel + Cross-channel attention layer."""
    def __init__(self, base_args):
        super().__init__()
        self.base = BaseModel(base_args)
        hidden = self.base.branch_hidden  # 256

        # Cross-channel attention: channels as tokens
        self.cross_attn = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=4, dim_feedforward=hidden * 2,
            dropout=0.1, activation='gelu', batch_first=True, norm_first=True
        )
        self.use_cross = True

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None,
                target_pred_len=None, query_points=None):
        if target_pred_len is None:
            target_pred_len = self.base.pred_len

        batch_size, seq_len, n_channels = x_enc.shape

        # RevIN
        if self.base.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc = x_enc / stdev

        x_cross_mean = x_enc.mean(dim=-1)  # not used if use_cross=False in base

        # Step 1: Per-channel encoder → z [B, C, hidden]
        z_list = []
        branch_inputs = []
        for ch in range(n_channels):
            x_ch = x_enc[:, :, ch]
            branch_input = self.base._build_branch_input(x_ch, x_cross_mean)
            branch_inputs.append(branch_input)

            # Get router-weighted representation
            router_logits = self.base.router(branch_input)
            expert_weights = F.softmax(router_logits, dim=-1)
            z_ch = torch.zeros(batch_size, self.base.branch_hidden,
                               dtype=x_ch.dtype, device=x_ch.device)
            for i, expert in enumerate(self.base.experts):
                z_expert = expert.get_representation(branch_input)
                weight = expert_weights[:, i].unsqueeze(-1)
                z_ch = z_ch + weight * z_expert
            z_list.append(z_ch)

        z_all = torch.stack(z_list, dim=1)  # [B, C, hidden]

        # Step 2: Cross-channel attention
        if self.use_cross:
            z_all = self.cross_attn(z_all)  # [B, C, hidden] — channels attend to each other

        # Step 3: Per-channel forecast head + trunk
        outputs = []
        for ch in range(n_channels):
            x_ch = x_enc[:, :, ch]
            z_ch = z_all[:, ch, :]  # enriched representation

            # Skip
            if self.base.use_skip:
                base = self.base.linear_skip(x_ch)
                if target_pred_len != self.base.pred_len:
                    base = F.interpolate(base.unsqueeze(1), size=target_pred_len,
                                         mode='linear', align_corners=True).squeeze(1)
            else:
                base = torch.zeros(batch_size, target_pred_len,
                                   dtype=x_ch.dtype, device=x_ch.device)

            # Use router weights for weighted expert output
            branch_input = branch_inputs[ch]
            router_logits = self.base.router(branch_input)
            expert_weights = F.softmax(router_logits, dim=-1)

            out_ch = torch.zeros(batch_size, target_pred_len,
                                 dtype=x_ch.dtype, device=x_ch.device)
            for i, expert in enumerate(self.base.experts):
                # Use enriched z instead of re-encoding
                branch_output = expert.forecast_head(z_ch)
                trunk_params = branch_output[:, :expert.trunk_param_count] * 0.01
                B = branch_output[:, expert.trunk_param_count:]

                trunk_weights = []
                idx = 0
                for in_dim, out_dim, bias_size in expert.trunk_param_shapes:
                    w_size = in_dim * out_dim
                    w = trunk_params[:, idx:idx+w_size].view(batch_size, in_dim, out_dim)
                    idx += w_size
                    b = trunk_params[:, idx:idx+bias_size].view(batch_size, out_dim)
                    idx += bias_size
                    trunk_weights.append((w, b))

                import math
                t_output = torch.linspace(0, 1, target_pred_len,
                                          dtype=x_ch.dtype, device=x_ch.device).unsqueeze(-1)
                t_features = expert._get_trunk_features(t_output)
                Phi = t_features.unsqueeze(0).expand(batch_size, -1, -1)

                act_fn = F.gelu
                for j, (w, b) in enumerate(trunk_weights):
                    Phi = torch.bmm(Phi, w) + b.unsqueeze(1)
                    if j < len(trunk_weights) - 1:
                        Phi = act_fn(Phi)

                expert_out = base + torch.einsum('bp,bqp->bq', B, Phi) + expert.bias
                weight = expert_weights[:, i].unsqueeze(-1)
                out_ch = out_ch + weight * expert_out

            outputs.append(out_ch)

        output = torch.stack(outputs, dim=-1)  # [B, pred_len, C]

        if self.base.use_norm:
            output = output * stdev + means

        return output


def run_experiment():
    # Dataset configs
    datasets = {
        'ETTh1': ('ETTh1', './dataset/ETT-small/', 'ETTh1.csv', 7),
        'ETTh2': ('ETTh2', './dataset/ETT-small/', 'ETTh2.csv', 7),
        'Weather': ('custom', './dataset/weather/', 'weather.csv', 21),
        'Exchange': ('custom', './dataset/exchange_rate/', 'exchange_rate.csv', 8),
    }

    results = {}

    for dname, (data, root, fpath, enc_in) in datasets.items():
        for pl in [96, 336]:
            print(f'\n{"="*50}')
            print(f'{dname} pred_len={pl}')
            print(f'{"="*50}')

            args = SimpleNamespace(
                seq_len=96, pred_len=pl, label_len=48,
                use_norm=True, deeponet_width=64,
                n_experts=4, branch_depth=4, trunk_depth=2,
                activation='gelu', dropout=0.1, branch_hidden=-1,
                spectral_branch=False, skip_mode='none',
                use_cross_channel=False, trunk_basis='mixed',
                encoder_type='patch_attn', loss='MSE',
                # data
                data=data, root_path=root, data_path=fpath,
                features='M', target='OT', freq='h', embed='timeF',
                enc_in=enc_in, dec_in=enc_in, c_out=enc_in,
                num_workers=2, batch_size=32,
                exp_name='MTSF', ordered_data=False, data_amount=-1,
                combine_Gaussian_datasets=False,
                synthetic_data_path='', synthetic_root_path='./',
                synthetic_length=1024, stride=-1,
            )

            # Build model
            model = CrossChannelModel(args).cuda()
            n_params = sum(p.numel() for p in model.parameters())
            print(f'Params: {n_params:,}')

            # Data
            train_ds, train_dl = data_provider(args, 'train')
            val_ds, val_dl = data_provider(args, 'val')
            test_ds, test_dl = data_provider(args, 'test')

            # Train
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            best_val = float('inf')
            patience_counter = 0

            for epoch in range(20):
                model.train()
                train_losses = []
                for batch_x, batch_y, bxm, bym in train_dl:
                    batch_x = batch_x.float().cuda()
                    batch_y = batch_y.float().cuda()

                    optimizer.zero_grad()
                    out = model(batch_x, target_pred_len=pl)
                    target = batch_y[:, -pl:, :]
                    loss = F.mse_loss(out, target)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    train_losses.append(loss.item())

                # Validate
                model.eval()
                val_losses = []
                with torch.no_grad():
                    for batch_x, batch_y, bxm, bym in val_dl:
                        batch_x = batch_x.float().cuda()
                        batch_y = batch_y.float().cuda()
                        out = model(batch_x, target_pred_len=pl)
                        target = batch_y[:, -pl:, :]
                        loss = F.mse_loss(out, target)
                        val_losses.append(loss.item())

                val_loss = np.mean(val_losses)
                train_loss = np.mean(train_losses)
                print(f'  Epoch {epoch+1}: train={train_loss:.4f} val={val_loss:.4f}')

                if val_loss < best_val:
                    best_val = val_loss
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= 5:
                        print(f'  Early stopping at epoch {epoch+1}')
                        break

            # Test
            model.load_state_dict(best_state)
            model.cuda().eval()
            preds, trues = [], []
            with torch.no_grad():
                for batch_x, batch_y, bxm, bym in test_dl:
                    batch_x = batch_x.float().cuda()
                    batch_y = batch_y.float().cuda()
                    out = model(batch_x, target_pred_len=pl)
                    target = batch_y[:, -pl:, :]
                    preds.append(out.cpu().numpy())
                    trues.append(target.cpu().numpy())

            preds = np.concatenate(preds)
            trues = np.concatenate(trues)
            mse = np.mean((preds - trues) ** 2)
            mae = np.mean(np.abs(preds - trues))

            key = f'{dname}_pl{pl}'
            results[key] = {'mse': mse, 'mae': mae}
            print(f'  TEST: MSE={mse:.4f} MAE={mae:.4f}')

    # Summary
    print(f'\n{"="*50}')
    print('RESULTS SUMMARY (Cross-Channel Attention)')
    print(f'{"="*50}')
    for k, v in results.items():
        print(f'  {k}: MSE={v["mse"]:.4f} MAE={v["mae"]:.4f}')


if __name__ == '__main__':
    run_experiment()
