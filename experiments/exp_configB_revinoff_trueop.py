"""
Config B + RevIN OFF + True OP Loss

같은 모델 (ConfigBModel 83M), 같은 데이터 (Pile + Synth),
단:
  ✓ RevIN OFF (forecast/reconstruct에서 mean/std norm 제거)
  ✓ True OP loss (point-wise query, seq2seq 대신)

목적: 우리가 발견한 두 가지 (RevIN off, True OP) 가 GPU 0의 Config B보다 좋은가?

CUDA_VISIBLE_DEVICES=2 python experiments/exp_configB_revinoff_trueop.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, math, time
from torch import optim
from torch.utils.data import DataLoader, ConcatDataset, random_split

from data_provider.pile_dataset import PilePretrainDataset
from experiments.exp_configB_pretrain import (
    ConfigBModel, SyntheticWindowDataset,
    extract_freq_info, build_informed_query,
)

DEVICE = torch.device('cuda')
SEQ_LEN = 96


# ============================================================
# Subclass: NO RevIN + arbitrary query points
# ============================================================
class ConfigBNoRevIN(ConfigBModel):
    """
    Pre-normalized 데이터를 받음. 내부 RevIN 안 씀.
    + query_at(): 임의 t에서 출력 (True OP)
    """
    def query_at(self, x_ch, x_cross, t_values, mode='forecast'):
        """
        x_ch: [B, S]   (이미 normalized)
        x_cross: [B, S]
        t_values: [B, nq]   (각 샘플별 query t)
        return: [B, nq]
        """
        z = self.encoder(self._branch(x_ch, x_cross))
        freqs, phases, lv, ls = extract_freq_info(x_ch, self.top_k_freq)

        # build informed query per-sample (same t per batch for simplicity)
        # build_informed_query expects t: [T, 1]
        # But here t differs per sample. Use a unified set of t and reindex.
        B, nq = t_values.shape

        outs = []
        for b in range(B):
            t_b = t_values[b].unsqueeze(-1)  # [nq, 1]
            iq = build_informed_query(t_b, freqs[b:b+1], phases[b:b+1], lv[b:b+1], ls[b:b+1])
            heads = self.forecast_heads if mode == 'forecast' else self.recon_heads
            out_b = sum(trunk(head(z[b:b+1]), iq) for trunk, head in zip(self.trunks, heads))
            outs.append(out_b)
        return torch.cat(outs, dim=0)  # [B, nq]


# ============================================================
# Vectorized query (faster than per-sample loop)
# ============================================================
def query_at_vectorized(model, x_ch, x_cross, t_values, mode='forecast'):
    """
    x_ch: [B, S], x_cross: [B, S], t_values: [B, nq]
    return: [B, nq]
    """
    z = model.encoder(model._branch(x_ch, x_cross))
    freqs, phases, lv, ls = extract_freq_info(x_ch, model.top_k_freq)
    B, nq = t_values.shape

    # Build per-sample informed query
    # build_informed_query inner shape: t_exp = [1, T, 1], freqs.unsqueeze(1) = [B, 1, K]
    # We want: t_exp = [B, nq, 1] (per-sample t)
    t_exp = t_values.unsqueeze(-1)  # [B, nq, 1]
    f = freqs.unsqueeze(1)  # [B, 1, K]
    p = phases.unsqueeze(1)  # [B, 1, K]
    angle = 2 * math.pi * f * t_exp + p  # [B, nq, K]
    iq = torch.cat([
        t_exp,
        torch.sin(angle), torch.cos(angle),
        lv.unsqueeze(1).expand(-1, nq, -1),
        ls.unsqueeze(1).expand(-1, nq, -1)
    ], dim=-1)  # [B, nq, query_dim]

    heads = model.forecast_heads if mode == 'forecast' else model.recon_heads
    return sum(trunk(head(z), iq) for trunk, head in zip(model.trunks, heads))


# ============================================================
# Dataset wrapper: pre-normalize per window (RevIN OFF in model)
# ============================================================
class PreNormDataset(torch.utils.data.Dataset):
    def __init__(self, base_ds):
        self.base = base_ds

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        item = self.base[idx]
        # PilePretrainDataset returns tensor or array; SyntheticWindowDataset returns [seq_len, 1]
        if isinstance(item, torch.Tensor):
            x = item.float()
        else:
            x = torch.tensor(np.array(item), dtype=torch.float32)
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        # Per-window z-score (pre-normalize)
        m = x.mean(dim=0, keepdim=True)
        s = x.std(dim=0, keepdim=True).clamp(min=1e-6)
        x = ((x - m) / s).clamp(-10, 10)
        return x


# ============================================================
# Training
# ============================================================
def train(model, save_path, epochs=20, lr=0.0003, batch_size=64, n_query=16):
    print(f'\n{"="*60}')
    print('Config B + RevIN OFF + True OP Loss')
    print(f'  Model: {sum(p.numel() for p in model.parameters())/1e6:.1f}M')
    print(f'  RevIN: OFF (data pre-normalized per window)')
    print(f'  Loss: True OP point-wise (n_query={n_query})')
    print(f'{"="*60}')

    real_base = PilePretrainDataset(seq_len=SEQ_LEN, stride=48,
                                     pile_root='./dataset/time_series_pile')
    print(f'Real (Pile): {len(real_base)} windows')
    synth_base = SyntheticWindowDataset('tempopfn_15k_1024.arrow',
                                         seq_len=SEQ_LEN, stride=48, max_samples=100000)

    real_ds = PreNormDataset(real_base)
    synth_ds = PreNormDataset(synth_base)
    combined = ConcatDataset([real_ds, synth_ds])
    n_val = min(10000, len(combined) // 10)
    n_train = len(combined) - n_val
    train_ds, _ = random_split(combined, [n_train, n_val])

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=4, drop_last=True, pin_memory=True)
    print(f'Train: {n_train}, Steps/epoch: {len(train_dl)}')

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        losses, fc_l, imp_l = [], [], []
        t0 = time.time()

        for i, batch_x in enumerate(train_dl):
            batch_x = batch_x.float().to(DEVICE)  # [B, S, C=1]
            B, S, C = batch_x.shape
            ch = 0
            x_ch = batch_x[:, :, ch]  # [B, S]
            x_cross = batch_x.mean(dim=-1)  # [B, S]

            optimizer.zero_grad()

            if np.random.rand() < 0.5:
                # ============= Forecast (True OP) =============
                # split context, query points in future region
                split = np.random.randint(48, 80)
                ctx = x_ch[:, :split]
                ctx_pad = F.pad(ctx, (0, S - split))
                cross_pad = F.pad(x_cross[:, :split], (0, S - split))

                # Sample query points in [split, S)
                future_len = S - split
                if future_len < n_query:
                    qi = np.arange(future_len)
                else:
                    qi = np.random.choice(future_len, n_query, replace=False)
                qi_t = torch.tensor(qi, device=DEVICE).long()

                # t in normalized [t_s, t_e] using SAME convention as model: forecast t in [1, 1+pred_len/seq_len]
                # ConfigBModel uses linspace(1, 1+tlen/seq_len, tlen) but we want point-wise
                # Match the original: t = 1.0 + (qi / seq_len) -- but here pred starts at split
                # Actually original normalizes such that t represents step / seq_len + 1 for forecast region.
                # We use: for the i-th future step, t_norm = 1.0 + (i+1)/seq_len  (matching linspace start≈1)
                # But our query is offset within [split, S), so:
                # i-th future step (0..future_len-1) corresponds to (split+i)-th absolute position
                # In original linspace(1, 1+tlen/S, tlen), step k -> 1 + k*tlen/S/(tlen-1) ~ 1 + k/S
                # So: t_val = 1.0 + qi / S
                t_val = 1.0 + qi_t.float() / S  # [n_query]
                t_val = t_val.unsqueeze(0).expand(B, -1)  # [B, n_query]

                pred = query_at_vectorized(model, ctx_pad, cross_pad, t_val, mode='forecast')
                # Targets: x_ch at positions [split + qi]
                tgt = x_ch[:, split:][:, :future_len].gather(1, qi_t.unsqueeze(0).expand(B, -1))
                loss = F.mse_loss(pred, tgt)
                fc_l.append(loss.item())

            else:
                # ============= Imputation (True OP) =============
                mask = (torch.rand(B, S, device=DEVICE) > 0.4).float()
                x_masked = x_ch * mask
                x_cross_masked = x_masked  # single channel
                # Query: random subset of MASKED positions
                # Find masked indices per sample
                qis = []
                for b in range(B):
                    masked_pos = (mask[b] == 0).nonzero(as_tuple=True)[0]
                    if len(masked_pos) == 0:
                        qis.append(torch.zeros(n_query, dtype=torch.long, device=DEVICE))
                    elif len(masked_pos) >= n_query:
                        sel = masked_pos[torch.randperm(len(masked_pos), device=DEVICE)[:n_query]]
                        qis.append(sel)
                    else:
                        # Repeat
                        rep = masked_pos.repeat((n_query + len(masked_pos) - 1) // len(masked_pos))[:n_query]
                        qis.append(rep)
                qis = torch.stack(qis)  # [B, n_query]
                t_val = qis.float() / S  # [B, n_query]

                pred = query_at_vectorized(model, x_masked, x_cross_masked, t_val, mode='recon')
                tgt = x_ch.gather(1, qis)
                loss = F.mse_loss(pred, tgt)
                imp_l.append(loss.item())

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())

            if (i+1) % 200 == 0:
                fc_avg = np.mean(fc_l[-200:]) if fc_l else 0
                imp_avg = np.mean(imp_l[-200:]) if imp_l else 0
                print(f'  iter {i+1}/{len(train_dl)}: loss={np.mean(losses[-200:]):.4f} '
                      f'fc={fc_avg:.4f} imp={imp_avg:.4f}')

        scheduler.step()
        elapsed = time.time() - t0
        print(f'Epoch {epoch+1}/{epochs}: loss={np.mean(losses):.4f} '
              f'(fc={np.mean(fc_l) if fc_l else 0:.4f} imp={np.mean(imp_l) if imp_l else 0:.4f}) '
              f'lr={scheduler.get_last_lr()[0]:.6f} ({elapsed:.0f}s)')

        if np.mean(losses) < best_loss:
            best_loss = np.mean(losses)
            torch.save(model.state_dict(), save_path)
            print(f'  Saved (best={best_loss:.4f})')

    return model


if __name__ == '__main__':
    print('='*60)
    print('ConfigB + RevIN OFF + True OP — Pre-training')
    print('  같은 모델 (83M), 같은 데이터 (Pile + Synth)')
    print('='*60)

    model = ConfigBNoRevIN(width=192, branch_hidden=768, trunk_depth=2, top_k_freq=5).to(DEVICE)
    n = sum(p.numel() for p in model.parameters())
    print(f'Model: {n/1e6:.1f}M params')

    save_path = 'checkpoints/configB_revinoff_trueop.pth'
    os.makedirs('checkpoints', exist_ok=True)

    train(model, save_path, epochs=20, lr=0.0003)
    print('\nDONE')
