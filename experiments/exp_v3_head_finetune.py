"""
v3 Informed Query: Head fine-tune eval
- v3 backbone (3.7M, pre-trained) + forecast/recon head fine-tune
- 비교: zero-shot v3 vs head-tuned v3 vs 63M MoE head-tuned

CUDA_VISIBLE_DEVICES=3 python experiments/exp_v3_head_finetune.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, math
from torch import optim
from torch.utils.data import DataLoader
from types import SimpleNamespace
from sklearn.metrics import accuracy_score
from data_provider.data_factory import data_provider
from data_provider.data_loader import Dataset_Classification

DEVICE = torch.device('cuda')
CKPT = 'checkpoints/v3_informed_query.pth'


def load_v3():
    from analysis.v3_informed_query_test import InformedQueryDeepONet
    model = InformedQueryDeepONet(
        seq_len=96, pred_len=96, width=96, branch_hidden=384,
        trunk_depth=2, top_k_freq=5, spectral_branch=True, dropout=0.1
    ).to(DEVICE)
    if os.path.exists(CKPT):
        model.load_state_dict(torch.load(CKPT, map_location=DEVICE))
        print(f'Loaded v3 checkpoint: {CKPT}')
    else:
        print(f'WARNING: {CKPT} not found, using random weights!')
    return model


# ============================================================
# 1. Forecasting (head fine-tune)
# ============================================================
def eval_forecasting():
    print(f'\n{"="*60}')
    print('1. Forecasting (v3, head fine-tune)')
    print(f'{"="*60}')

    datasets = {
        'ETTh1': ('ETTh1', './dataset/ETT-small/', 'ETTh1.csv', 7),
        'ETTh2': ('ETTh2', './dataset/ETT-small/', 'ETTh2.csv', 7),
        'ETTm1': ('ETTm1', './dataset/ETT-small/', 'ETTm1.csv', 7),
        'ETTm2': ('ETTm2', './dataset/ETT-small/', 'ETTm2.csv', 7),
        'Weather': ('custom', './dataset/weather/', 'weather.csv', 21),
        'Exchange': ('custom', './dataset/exchange_rate/', 'exchange_rate.csv', 8),
    }
    mlp = {'ETTh1_96':0.387,'ETTh1_192':0.410,'ETTh1_336':0.422,'ETTh1_720':0.454,
           'ETTh2_96':0.288,'ETTh2_192':0.349,'ETTh2_336':0.369,'ETTh2_720':0.403,
           'ETTm1_96':0.293,'ETTm1_192':0.326,'ETTm1_336':0.352,'ETTm1_720':0.405,
           'ETTm2_96':0.170,'ETTm2_192':0.227,'ETTm2_336':0.275,'ETTm2_720':0.363,
           'Weather_96':0.154,'Weather_192':0.197,'Weather_336':0.246,'Weather_720':0.315}

    for dname, (data, root, fpath, enc_in) in datasets.items():
        for pl in [96, 192, 336, 720]:
            model = load_v3()

            args = SimpleNamespace(
                seq_len=96, pred_len=pl, label_len=48,
                data=data, root_path=root, data_path=fpath,
                features='M', target='OT', freq='h', embed='timeF',
                enc_in=enc_in, dec_in=enc_in, c_out=enc_in,
                num_workers=2, batch_size=64,
                exp_name='MTSF', ordered_data=False, data_amount=-1,
                combine_Gaussian_datasets=False,
                synthetic_data_path='', synthetic_root_path='./',
                synthetic_length=1024, stride=-1)
            _, train_dl = data_provider(args, 'train')
            _, test_dl = data_provider(args, 'test')

            # Freeze encoder, unfreeze forecast_head only
            for p in model.parameters(): p.requires_grad = False
            trainable = []
            for name, p in model.named_parameters():
                if 'forecast_head' in name:
                    p.requires_grad = True; trainable.append(p)
            n_train = sum(p.numel() for p in trainable)
            if dname == 'ETTh1' and pl == 96:
                print(f'  Trainable: {n_train:,} params')

            optimizer = optim.Adam(trainable, lr=0.0001, weight_decay=0.001)
            best_loss, patience, best_state = float('inf'), 0, None

            for epoch in range(50):
                model.train(); losses = []
                for bx, by, _, _ in train_dl:
                    bx, by = bx.float().to(DEVICE), by.float().to(DEVICE)
                    optimizer.zero_grad()
                    out = model.forecast(bx, target_pred_len=pl)
                    loss = F.mse_loss(out, by[:, -pl:, :])
                    loss.backward(); optimizer.step(); losses.append(loss.item())
                tl = np.mean(losses)
                if tl < best_loss:
                    best_loss = tl
                    best_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}
                    patience = 0
                else:
                    patience += 1
                    if patience >= 7: break

            if best_state: model.load_state_dict(best_state)
            model.to(DEVICE).eval()

            preds, trues = [], []
            with torch.no_grad():
                for bx, by, _, _ in test_dl:
                    bx = bx.float().to(DEVICE)
                    out = model.forecast(bx, target_pred_len=pl)
                    preds.append(out.cpu().numpy()); trues.append(by[:, -pl:, :].numpy())
            p, t = np.concatenate(preds), np.concatenate(trues)
            mse = np.mean((p - t) ** 2)
            mae = np.mean(np.abs(p - t))
            key = f'{dname}_{pl}'
            m = mlp.get(key)
            g = f'{(mse/m-1)*100:+.1f}%' if m else '-'
            print(f'  {key}: MSE={mse:.4f} MAE={mae:.4f}  MOMENT_LP={m or "-"}  gap={g}')
            del model; torch.cuda.empty_cache()


# ============================================================
# 2. Imputation (head fine-tune)
# ============================================================
def eval_imputation():
    print(f'\n{"="*60}')
    print('2. Imputation (v3, recon head fine-tune)')
    print(f'{"="*60}')

    datasets = {
        'ETTh1': ('ETTh1', './dataset/ETT-small/', 'ETTh1.csv', 7),
        'ETTh2': ('ETTh2', './dataset/ETT-small/', 'ETTh2.csv', 7),
        'ETTm1': ('ETTm1', './dataset/ETT-small/', 'ETTm1.csv', 7),
        'ETTm2': ('ETTm2', './dataset/ETT-small/', 'ETTm2.csv', 7),
        'Weather': ('custom', './dataset/weather/', 'weather.csv', 21),
    }
    moment = {'ETTh1':(0.402,0.139),'ETTh2':(0.125,0.061),
              'ETTm1':(0.202,0.074),'ETTm2':(0.078,0.031),'Weather':(0.082,0.035)}

    for dname, (data, root, fpath, enc_in) in datasets.items():
        model = load_v3()
        args = SimpleNamespace(
            seq_len=96, pred_len=96, label_len=48,
            data=data, root_path=root, data_path=fpath,
            features='M', target='OT', freq='h', embed='timeF',
            enc_in=enc_in, dec_in=enc_in, c_out=enc_in,
            num_workers=2, batch_size=32,
            exp_name='MTSF', ordered_data=False, data_amount=-1,
            combine_Gaussian_datasets=False,
            synthetic_data_path='', synthetic_root_path='./',
            synthetic_length=1024, stride=-1)
        _, train_dl = data_provider(args, 'train')
        _, test_dl = data_provider(args, 'test')

        for p in model.parameters(): p.requires_grad = False
        trainable = []
        for name, p in model.named_parameters():
            if 'recon_head' in name or 'recon_bias' in name:
                p.requires_grad = True; trainable.append(p)

        optimizer = optim.Adam(trainable, lr=0.0001, weight_decay=0.001)
        best_loss, patience, best_state = float('inf'), 0, None

        for epoch in range(30):
            model.train(); losses = []
            torch.manual_seed(epoch)
            for bx, _, _, _ in train_dl:
                bx = bx.float().to(DEVICE)
                mask = (torch.rand_like(bx) > 0.375).float()
                optimizer.zero_grad()
                out = model.reconstruct(bx * mask)
                lm = F.mse_loss(out, bx, reduction='none')
                inv = 1.0 - mask
                loss = (lm * inv).sum() / inv.sum().clamp(min=1)
                loss.backward(); optimizer.step(); losses.append(loss.item())
            tl = np.mean(losses)
            if tl < best_loss:
                best_loss = tl; best_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}; patience = 0
            else:
                patience += 1
                if patience >= 5: break

        if best_state: model.load_state_dict(best_state)
        model.to(DEVICE).eval()

        all_mse = []
        for mr in [0.125, 0.25, 0.375, 0.5]:
            torch.manual_seed(2021)
            preds, trues, masks = [], [], []
            with torch.no_grad():
                for bx, _, _, _ in test_dl:
                    bx = bx.float().to(DEVICE)
                    mk = (torch.rand_like(bx) > mr).float()
                    preds.append(model.reconstruct(bx * mk).cpu().numpy())
                    trues.append(bx.cpu().numpy()); masks.append(mk.cpu().numpy())
            p, t, m = np.concatenate(preds), np.concatenate(trues), np.concatenate(masks)
            mse = np.mean((p[m==0] - t[m==0])**2); all_mse.append(mse)
            print(f'  {dname} mask={mr}: MSE={mse:.4f}')
        avg = np.mean(all_mse)
        m0, ml = moment.get(dname, (None, None))
        print(f'  {dname} Mean: MSE={avg:.4f}  (MOMENT_0={m0}, LP={ml})')
        del model; torch.cuda.empty_cache()


# ============================================================
# 3. Classification
# ============================================================
def eval_classification():
    print(f'\n{"="*60}')
    print('3. Classification (v3, linear probe)')
    print(f'{"="*60}')

    model = load_v3(); model.eval()
    hidden = model.branch_hidden
    for p in model.parameters(): p.requires_grad = False

    for ds_name in ['Epilepsy', 'FingerMovements', 'BasicMotions',
                    'NATOPS', 'EthanolConcentration', 'Heartbeat',
                    'ArticularyWordRecognition', 'ERing']:
        try:
            cr = './dataset/classification/Multivariate_ts'
            trd = Dataset_Classification(root_path=cr, flag='train', size=[96,0,96], data_path=ds_name)
            ted = Dataset_Classification(root_path=cr, flag='test', size=[96,0,96], data_path=ds_name)
            trl = DataLoader(trd, batch_size=16, shuffle=True, drop_last=True)
            tel = DataLoader(ted, batch_size=16, shuffle=False)

            ch = nn.Sequential(
                nn.Linear(hidden, 256), nn.GELU(), nn.Dropout(0.1),
                nn.Linear(256, trd.n_classes)).to(DEVICE)
            op = optim.Adam(ch.parameters(), lr=0.001)
            ba = 0
            for epoch in range(30):
                ch.train()
                for bx, lb, _, _ in trl:
                    bx = bx.float().to(DEVICE); lb = lb.long().to(DEVICE)
                    with torch.no_grad(): z = model.get_representation(bx).mean(dim=1)
                    l = nn.CrossEntropyLoss()(ch(z), lb)
                    op.zero_grad(); l.backward(); op.step()
                ch.eval(); ps, ls = [], []
                with torch.no_grad():
                    for bx, lb, _, _ in tel:
                        bx = bx.float().to(DEVICE)
                        z = model.get_representation(bx).mean(dim=1)
                        ps.append(ch(z).argmax(-1).cpu().numpy()); ls.append(lb.numpy())
                ac = accuracy_score(np.concatenate(ls), np.concatenate(ps))
                ba = max(ba, ac)
            print(f'  {ds_name}: Acc={ba:.4f}')
        except Exception as e:
            print(f'  {ds_name}: SKIP ({e})')


if __name__ == '__main__':
    print('='*60)
    print('v3 Informed Query: Head Fine-tune Eval')
    print(f'Checkpoint: {CKPT}')
    print('='*60)

    eval_forecasting()
    eval_imputation()
    eval_classification()

    print(f'\n{"="*60}\nALL DONE\n{"="*60}')
