"""
Eval for 32.5M long context models (seq192, seq384)
Usage: CUDA_VISIBLE_DEVICES=X python experiments/eval_32M_longctx.py --seq 192 --tag seq192
"""
import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.environ.setdefault('CUDA_DEV', os.environ.get('CUDA_DEV', 'cuda'))

import torch, torch.nn.functional as F
import numpy as np
from types import SimpleNamespace
from data_provider.data_factory import data_provider
from experiments.exp_32M_longctx import Model32MLongCtx

_dev = os.environ.get('CUDA_DEV', 'cuda')
DEVICE = torch.device(_dev)

def forecast_mv(model, x_enc, target_pred_len, seq_len):
    B, S, C = x_enc.shape
    outs = []
    for ch in range(C):
        x_ch = x_enc[:, :, ch]
        if S >= seq_len: x_ctx = x_ch[:, -seq_len:]
        else: x_ctx = F.pad(x_ch, (seq_len - S, 0))
        m = x_ctx.mean(dim=1, keepdim=True)
        s = x_ctx.std(dim=1, keepdim=True).clamp(min=1e-6)
        x_n = ((x_ctx - m) / s).clamp(-10, 10)
        cur = x_n
        chunks = []
        remain = target_pred_len
        while remain > 0:
            step = min(seq_len, remain)
            t = 1.0 + torch.arange(step, device=cur.device, dtype=torch.float32) / seq_len
            t = t.unsqueeze(0).expand(B, -1)
            with torch.no_grad():
                pred_n = model.forward_train(cur, t)
            chunks.append(pred_n)
            if remain > step:
                cur = torch.cat([cur[:, step:], pred_n], dim=1)
            remain -= step
        pred_n_full = torch.cat(chunks, dim=1)
        pred = pred_n_full * s + m
        outs.append(pred)
    return torch.stack(outs, dim=-1)

def eval_forecast(model, seq_len, tag):
    print(f'\n{"="*60}\nForecast ZS (32.5M {tag})\n{"="*60}')
    datasets = {
        'ETTh1': ('ETTh1','./dataset/ETT-small/','ETTh1.csv',7),
        'ETTh2': ('ETTh2','./dataset/ETT-small/','ETTh2.csv',7),
        'ETTm1': ('ETTm1','./dataset/ETT-small/','ETTm1.csv',7),
        'ETTm2': ('ETTm2','./dataset/ETT-small/','ETTm2.csv',7),
        'Weather': ('custom','./dataset/weather/','weather.csv',21),
        'Exchange': ('custom','./dataset/exchange_rate/','exchange_rate.csv',8),
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
    for dn, (d,root,f,enc_in) in datasets.items():
        for pl in [96,192,336,720]:
            try:
                a = SimpleNamespace(seq_len=seq_len,pred_len=pl,label_len=48,data=d,
                    root_path=root,data_path=f,features='M',target='OT',
                    freq='h',embed='timeF',enc_in=enc_in,dec_in=enc_in,c_out=enc_in,
                    num_workers=2,batch_size=32,exp_name='MTSF',ordered_data=False,
                    data_amount=-1,combine_Gaussian_datasets=False,
                    synthetic_data_path='',synthetic_root_path='./',
                    synthetic_length=1024,stride=-1)
                _,tdl = data_provider(a,'test')
                preds,tgts = [],[]
                for bx,by,_,_ in tdl:
                    bx = bx.float().to(DEVICE)
                    p = forecast_mv(model,bx,pl,seq_len)
                    preds.append(p.cpu().numpy())
                    tgts.append(by[:,-pl:,:].numpy())
                p=np.concatenate(preds);t=np.concatenate(tgts)
                mse=np.mean((p-t)**2);mae=np.mean(np.abs(p-t))
                k=f'{dn}_{pl}'
                m_lp=moment_lp.get(k)
                gap=f'{(mse/m_lp-1)*100:+.0f}%' if m_lp else '-'
                print(f'  {k:<14}: MSE={mse:.4f} MAE={mae:.4f} | MOMENT-LP={m_lp or "N/A"} gap={gap}')
                results[k]=mse
            except Exception as e:
                print(f'  {dn}_{pl}: ERROR ({e})')
    return results

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--seq',type=int,required=True)
    parser.add_argument('--tag',type=str,required=True)
    args=parser.parse_args()
    ckpt=f'checkpoints/32M_{args.tag}.pth'
    print(f'Loading {ckpt}')
    model=Model32MLongCtx(args.seq).to(DEVICE)
    model.load_state_dict(torch.load(ckpt,map_location=DEVICE))
    n=sum(p.numel() for p in model.parameters())
    print(f'Model: {n/1e6:.1f}M, SEQ={args.seq}')
    fc=eval_forecast(model,args.seq,args.tag)
    print('\n'+'='*60)
    print(f'SUMMARY — 32.5M {args.tag}')
    print('='*60)
    for k in ['ETTh1','ETTh2','ETTm1','ETTm2','Weather','Exchange']:
        avgs=[v for k_,v in fc.items() if k_.startswith(k+'_')]
        if avgs: print(f'  {k:<10}: {np.mean(avgs):.4f}')
    print('='*60)
