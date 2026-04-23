"""
V2 5-task Eval (no-decomp or decomp variant)

Usage:
  CUDA_VISIBLE_DEVICES=X python experiments/eval_v2.py --tag nodecomp
  CUDA_VISIBLE_DEVICES=X python experiments/eval_v2.py --tag decomp
"""
import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from types import SimpleNamespace
from torch import optim
from torch.utils.data import DataLoader

from data_provider.data_factory import data_provider
from data_provider.data_loader import Dataset_Classification
from experiments.exp_v2 import ModelV2NoDecomp, ModelV2Decomp, SEQ_LEN, HIDDEN
from sklearn.metrics import accuracy_score

DEVICE = torch.device('cuda')


@torch.no_grad()
def forecast_step(model, ctx, n):
    """ctx: [B, SEQ_LEN] raw. Returns [B, n] in same scale.
    Uses training-consistent t spacing: t = 1 + i/SEQ_LEN for i in [0, n-1]."""
    t = 1.0 + torch.arange(n, device=ctx.device, dtype=torch.float32) / SEQ_LEN
    t = t.unsqueeze(0).expand(ctx.shape[0], -1)
    return model.forward_train(ctx, t)


@torch.no_grad()
def reconstruct(model, ctx_masked, n_points=SEQ_LEN):
    """Imputation query: t in [0, 1] with training-consistent spacing."""
    t = torch.arange(n_points, device=ctx_masked.device, dtype=torch.float32) / SEQ_LEN
    t = t.unsqueeze(0).expand(ctx_masked.shape[0], -1)
    return model.forward_train(ctx_masked, t)


def forecast_mv(model, x_enc, target_pred_len):
    """x_enc [B, S, C] → [B, target_pred_len, C]"""
    B, S, C = x_enc.shape
    outs = []
    for ch in range(C):
        x_ch = x_enc[:, :, ch]
        if S >= SEQ_LEN:
            x_ctx = x_ch[:, -SEQ_LEN:]
        else:
            x_ctx = F.pad(x_ch, (SEQ_LEN - S, 0))
        # Iterative roll-out
        cur = x_ctx
        chunks = []
        remain = target_pred_len
        while remain > 0:
            step = min(SEQ_LEN, remain)
            p = forecast_step(model, cur, step)
            chunks.append(p)
            if remain > step:
                cur = torch.cat([cur[:, step:], p], dim=1)
            remain -= step
        outs.append(torch.cat(chunks, dim=1))
    return torch.stack(outs, dim=-1)


def reconstruct_mv(model, x_enc, mask):
    B, S, C = x_enc.shape
    outs = []
    for ch in range(C):
        x_ch = x_enc[:, :, ch] * mask[:, :, ch]
        if S > SEQ_LEN: x_ch = x_ch[:, -SEQ_LEN:]
        elif S < SEQ_LEN: x_ch = F.pad(x_ch, (SEQ_LEN - S, 0))
        rec = reconstruct(model, x_ch)
        outs.append(rec)
    return torch.stack(outs, dim=-1)


def eval_forecast(model, tag):
    print(f'\n{"="*60}\nForecast (V2 {tag})\n{"="*60}')
    datasets = {
        'ETTh1':    ('ETTh1',  './dataset/ETT-small/', 'ETTh1.csv', 7),
        'ETTh2':    ('ETTh2',  './dataset/ETT-small/', 'ETTh2.csv', 7),
        'ETTm1':    ('ETTm1',  './dataset/ETT-small/', 'ETTm1.csv', 7),
        'ETTm2':    ('ETTm2',  './dataset/ETT-small/', 'ETTm2.csv', 7),
        'Weather':  ('custom', './dataset/weather/',   'weather.csv', 21),
        'Exchange': ('custom', './dataset/exchange_rate/', 'exchange_rate.csv', 8),
    }
    moment_lp = {
        'ETTh1_96':0.387,'ETTh1_192':0.410,'ETTh1_336':0.422,'ETTh1_720':0.454,
        'ETTh2_96':0.288,'ETTh2_192':0.349,'ETTh2_336':0.369,'ETTh2_720':0.403,
        'ETTm1_96':0.293,'ETTm1_192':0.326,'ETTm1_336':0.352,'ETTm1_720':0.405,
        'ETTm2_96':0.170,'ETTm2_192':0.227,'ETTm2_336':0.275,'ETTm2_720':0.363,
        'Weather_96':0.154,'Weather_192':0.197,'Weather_336':0.246,'Weather_720':0.315,
    }
    model.eval()
    results = {}
    for dn, (d, root, f, enc_in) in datasets.items():
        for pl in [96, 192, 336, 720]:
            try:
                a = SimpleNamespace(seq_len=SEQ_LEN, pred_len=pl, label_len=48, data=d,
                    root_path=root, data_path=f, features='M', target='OT',
                    freq='h', embed='timeF', enc_in=enc_in, dec_in=enc_in, c_out=enc_in,
                    num_workers=2, batch_size=32, exp_name='MTSF', ordered_data=False,
                    data_amount=-1, combine_Gaussian_datasets=False,
                    synthetic_data_path='', synthetic_root_path='./',
                    synthetic_length=1024, stride=-1)
                _, tdl = data_provider(a, 'test')
                preds, tgts = [], []
                for bx, by, _, _ in tdl:
                    bx = bx.float().to(DEVICE)
                    p = forecast_mv(model, bx, pl)
                    preds.append(p.cpu().numpy())
                    tgts.append(by[:, -pl:, :].numpy())
                p = np.concatenate(preds); t = np.concatenate(tgts)
                mse = np.mean((p - t)**2)
                mae = np.mean(np.abs(p - t))
                k = f'{dn}_{pl}'
                m_lp = moment_lp.get(k)
                gap = f'{(mse/m_lp-1)*100:+.0f}%' if m_lp else '-'
                m_str = f'{m_lp:.3f}' if m_lp else 'N/A'
                print(f'  {k:<14}: MSE={mse:.4f} MAE={mae:.4f} | MOMENT-LP={m_str} gap={gap}')
                results[k] = mse
            except Exception as e:
                print(f'  {dn}_{pl}: ERROR ({type(e).__name__}: {e})')
    return results


def eval_imputation(model, tag):
    print(f'\n{"="*60}\nImputation (V2 {tag})\n{"="*60}')
    datasets = {
        'ETTh1':   ('ETTh1', './dataset/ETT-small/', 'ETTh1.csv', 7),
        'ETTh2':   ('ETTh2', './dataset/ETT-small/', 'ETTh2.csv', 7),
        'ETTm1':   ('ETTm1', './dataset/ETT-small/', 'ETTm1.csv', 7),
        'ETTm2':   ('ETTm2', './dataset/ETT-small/', 'ETTm2.csv', 7),
        'Weather': ('custom','./dataset/weather/',   'weather.csv', 21),
    }
    moment = {'ETTh1':(0.402,0.139),'ETTh2':(0.125,0.061),
              'ETTm1':(0.202,0.074),'ETTm2':(0.078,0.031),'Weather':(0.082,0.035)}
    model.eval()
    results = {}
    for dn, (d, root, f, enc_in) in datasets.items():
        try:
            a = SimpleNamespace(seq_len=SEQ_LEN, pred_len=96, label_len=0, data=d,
                root_path=root, data_path=f, features='M', target='OT',
                freq='h', embed='timeF', enc_in=enc_in, dec_in=enc_in, c_out=enc_in,
                num_workers=2, batch_size=32, exp_name='MTSF', ordered_data=False,
                data_amount=-1, combine_Gaussian_datasets=False,
                synthetic_data_path='', synthetic_root_path='./',
                synthetic_length=1024, stride=-1)
            _, tdl = data_provider(a, 'test')
            mr_mses = []
            for mr in [0.125, 0.25, 0.375, 0.5]:
                torch.manual_seed(2021)
                preds, tgts, masks = [], [], []
                for bx, by, _, _ in tdl:
                    bx = bx.float().to(DEVICE)
                    mk = (torch.rand_like(bx) > mr).float()
                    rec = reconstruct_mv(model, bx, mk)
                    preds.append(rec.cpu().numpy())
                    tgts.append(bx.cpu().numpy())
                    masks.append(mk.cpu().numpy())
                p = np.concatenate(preds); t = np.concatenate(tgts); m = np.concatenate(masks)
                mse = np.mean((p[m==0] - t[m==0])**2)
                mr_mses.append(mse)
            avg = np.mean(mr_mses)
            m0, m_lp = moment.get(dn, (None, None))
            print(f'  {dn:<8}: Avg={avg:.4f} | by mr: {[f"{x:.3f}" for x in mr_mses]} | M_0={m0} LP={m_lp}')
            results[dn] = avg
        except Exception as e:
            print(f'  {dn}: ERROR ({type(e).__name__}: {e})')
    return results


def get_z_v2(model, bx, decomp=False):
    """Get encoder representation for classification LP."""
    B, S, C = bx.shape
    zs = []
    for ch in range(C):
        x_ch = bx[:, :, ch]
        if S > SEQ_LEN: x_ch = x_ch[:, -SEQ_LEN:]
        elif S < SEQ_LEN: x_ch = F.pad(x_ch, (SEQ_LEN - S, 0))
        last = x_ch[:, -1:]
        x_c = x_ch - last
        s = x_c.std(dim=1, keepdim=True).clamp(min=1e-6)
        x_n = (x_c / s).clamp(-10, 10)
        if decomp:
            from experiments.exp_v2 import decompose_4way
            trend, season, cycle, resid = decompose_4way(x_n)
            z_t = model.block_T.encoder(trend)
            z_s = model.block_S.encoder(season)
            z_c = model.block_C.encoder(cycle)
            z_r = model.block_R.encoder(resid)
            z = torch.cat([z_t, z_s, z_c, z_r], dim=-1)
        else:
            z = model.block.encoder(x_n)
        zs.append(z)
    return torch.stack(zs, dim=1).mean(dim=1)


def eval_classification(model, tag, is_decomp):
    print(f'\n{"="*60}\nClassification LP (V2 {tag})\n{"="*60}')
    for p in model.parameters(): p.requires_grad = False
    # Determine feature dim
    if is_decomp:
        feat_dim = 384 * 4  # HIDDEN_DECOMP * 4
    else:
        feat_dim = HIDDEN
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
                nn.Linear(feat_dim, 256), nn.GELU(), nn.Dropout(0.1),
                nn.Linear(256, trd.n_classes)).to(DEVICE)
            opt = optim.Adam(head.parameters(), lr=1e-3, weight_decay=1e-4)
            best = 0
            for ep in range(30):
                head.train()
                for bx, lb, _, _ in trl:
                    bx = bx.float().to(DEVICE); lb = lb.long().to(DEVICE)
                    with torch.no_grad(): z = get_z_v2(model, bx, decomp=is_decomp)
                    loss = nn.CrossEntropyLoss()(head(z), lb)
                    opt.zero_grad(); loss.backward(); opt.step()
                head.eval()
                ps, ls = [], []
                with torch.no_grad():
                    for bx, lb, _, _ in tel:
                        bx = bx.float().to(DEVICE)
                        z = get_z_v2(model, bx, decomp=is_decomp)
                        ps.append(head(z).argmax(-1).cpu().numpy())
                        ls.append(lb.numpy())
                acc = accuracy_score(np.concatenate(ls), np.concatenate(ps))
                best = max(best, acc)
            print(f'  {ds:<25}: Acc={best:.4f}')
            results[ds] = best
        except Exception as e:
            print(f'  {ds}: SKIP ({type(e).__name__}: {e})')
    for p in model.parameters(): p.requires_grad = True
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', type=str, required=True, choices=['nodecomp', 'decomp'])
    args = parser.parse_args()

    ckpt = f'checkpoints/v2_{args.tag}.pth'
    print('='*60)
    print(f'V2 5-Task Eval [{args.tag}]')
    print('='*60)
    print(f'Loading: {ckpt}')

    is_decomp = args.tag == 'decomp'
    model = ModelV2Decomp() if is_decomp else ModelV2NoDecomp()
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    model = model.to(DEVICE)
    n = sum(p.numel() for p in model.parameters())
    print(f'Model: {n/1e6:.1f}M params')

    fc_res = eval_forecast(model, args.tag)
    imp_res = eval_imputation(model, args.tag)
    cls_res = eval_classification(model, args.tag, is_decomp)

    print('\n' + '='*60)
    print(f'SUMMARY — V2 {args.tag}')
    print('='*60)
    print('\nForecast (avg):')
    for k in ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'Weather', 'Exchange']:
        avgs = [v for k_, v in fc_res.items() if k_.startswith(k+'_')]
        if avgs: print(f'  {k:<10}: {np.mean(avgs):.4f}')
    print('\nImputation:')
    for k, v in imp_res.items(): print(f'  {k:<10}: {v:.4f}')
    print('\nClassification:')
    for k, v in cls_res.items(): print(f'  {k:<25}: {v:.4f}')
    if cls_res: print(f'  AVG: {np.mean(list(cls_res.values())):.4f}')
    print('='*60)
