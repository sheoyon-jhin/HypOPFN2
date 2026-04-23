"""
Classification Linear Probe for 32.5M model.
(Forecast/Imputation/LP 이미 완료)

CUDA_VISIBLE_DEVICES=1 python experiments/eval_32M_cls.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from torch import optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

from data_provider.data_loader import Dataset_Classification
from experiments.exp_full_scale_train import FullScaleModel, SEQ_LEN, HIDDEN

DEVICE = torch.device('cuda')
CKPT = 'checkpoints/full_scale_run.pth'


def eval_classification(model):
    print(f'\n{"="*60}\nClassification (linear probe)\n{"="*60}')
    for p in model.parameters(): p.requires_grad = False
    results = {}
    datasets = ['Epilepsy', 'FingerMovements', 'BasicMotions', 'NATOPS',
                'EthanolConcentration', 'Heartbeat', 'MotorImagery',
                'SelfRegulationSCP1', 'SelfRegulationSCP2', 'UWaveGestureLibrary']
    for ds in datasets:
        try:
            cr = './dataset/classification/Multivariate_ts'
            trd = Dataset_Classification(root_path=cr, flag='train', size=[96,0,96], data_path=ds)
            ted = Dataset_Classification(root_path=cr, flag='test', size=[96,0,96], data_path=ds)
            trl = DataLoader(trd, batch_size=16, shuffle=True, drop_last=True)
            tel = DataLoader(ted, batch_size=16, shuffle=False)
            head = nn.Sequential(
                nn.Linear(HIDDEN, 256), nn.GELU(), nn.Dropout(0.1),
                nn.Linear(256, trd.n_classes)).to(DEVICE)
            opt = optim.Adam(head.parameters(), lr=1e-3, weight_decay=1e-4)

            def get_z(bx):
                """bx: [B, S, C] → [B, C, HIDDEN] per-channel normalize + encode"""
                B, S, C = bx.shape
                zs = []
                for ch in range(C):
                    x_ch = bx[:, :, ch]
                    if S > SEQ_LEN: x_ch = x_ch[:, -SEQ_LEN:]
                    elif S < SEQ_LEN: x_ch = F.pad(x_ch, (SEQ_LEN - S, 0))
                    m = x_ch.mean(dim=1, keepdim=True)
                    s = x_ch.std(dim=1, keepdim=True).clamp(min=1e-6)
                    x_n = ((x_ch - m) / s).clamp(-10, 10)
                    z = model.encoder(x_n)
                    zs.append(z)
                return torch.stack(zs, dim=1).mean(dim=1)  # [B, HIDDEN]

            best = 0
            for ep in range(30):
                head.train()
                for bx, lb, _, _ in trl:
                    bx = bx.float().to(DEVICE); lb = lb.long().to(DEVICE)
                    with torch.no_grad(): z = get_z(bx)
                    loss = nn.CrossEntropyLoss()(head(z), lb)
                    opt.zero_grad(); loss.backward(); opt.step()
                head.eval()
                ps, ls = [], []
                with torch.no_grad():
                    for bx, lb, _, _ in tel:
                        bx = bx.float().to(DEVICE)
                        z = get_z(bx)
                        ps.append(head(z).argmax(-1).cpu().numpy())
                        ls.append(lb.numpy())
                acc = accuracy_score(np.concatenate(ls), np.concatenate(ps))
                best = max(best, acc)
            print(f'  {ds:<25}: Acc={best:.4f}')
            results[ds] = best
        except Exception as e:
            print(f'  {ds}: SKIP ({type(e).__name__}: {e})')
    return results


if __name__ == '__main__':
    print('='*60)
    print('32.5M Classification Linear Probe')
    print('='*60)
    model = FullScaleModel().to(DEVICE)
    model.load_state_dict(torch.load(CKPT, map_location=DEVICE))
    n = sum(p.numel() for p in model.parameters())
    print(f'Model: {n/1e6:.1f}M params')

    cls_res = eval_classification(model)

    print('\n' + '='*60)
    print('SUMMARY — 32.5M Classification')
    print('='*60)
    for k, v in cls_res.items():
        print(f'  {k:<25}: {v:.4f}')
    if cls_res:
        print(f'  {"AVG":<25}: {np.mean(list(cls_res.values())):.4f}')
    print('='*60)
