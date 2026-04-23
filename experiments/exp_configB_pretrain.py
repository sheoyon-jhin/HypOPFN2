"""
Config B Full Pre-training: Additive Trunk + Informed Query (83M)
- Quick test에서 주파수 보존 최강 (all freq corr>0.43, phase 0.87)
- Pile + Synth, 20 epochs, MSE + Spectral Loss

CUDA_VISIBLE_DEVICES=0 python experiments/exp_configB_pretrain.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, math, time
from torch import optim
from torch.utils.data import DataLoader, Dataset, ConcatDataset, random_split
from types import SimpleNamespace
from sklearn.metrics import accuracy_score

from data_provider.pile_dataset import PilePretrainDataset
from data_provider.data_factory import data_provider
from data_provider.data_loader import Dataset_Classification, Dataset_GaussianPCoregionalization
from model.DeepONetHyperMoE import build_encoder

DEVICE = torch.device('cuda')

# ============================================================
# Shared components
# ============================================================
def extract_freq_info(x_ch, top_k=5):
    fft = torch.fft.rfft(x_ch, dim=-1)
    mag = fft.abs(); mag[:, 0] = 0
    idx = torch.topk(mag, top_k, dim=-1).indices
    phase = torch.angle(torch.gather(fft, 1, idx))
    return idx.float(), phase, x_ch[:, -1:].detach(), (x_ch[:, -1:] - x_ch[:, -2:-1]).detach()

def build_informed_query(t, freqs, phases, lv, ls):
    B, T = freqs.shape[0], t.shape[0]
    t_exp = t.squeeze(-1).unsqueeze(0).unsqueeze(-1)
    f, p = freqs.unsqueeze(1), phases.unsqueeze(1)
    angle = 2 * math.pi * f * t_exp + p
    return torch.cat([
        t.squeeze(-1).unsqueeze(0).expand(B, -1).unsqueeze(-1),
        torch.sin(angle), torch.cos(angle),
        lv.unsqueeze(1).expand(-1, T, -1),
        ls.unsqueeze(1).expand(-1, T, -1)
    ], dim=-1)

def spectral_loss(pred, target):
    pf, tf = torch.fft.rfft(pred, dim=1), torch.fft.rfft(target, dim=1)
    ml = F.mse_loss(pf.abs(), tf.abs())
    w = tf.abs().detach() + 1e-8
    pl = (w * (1 - torch.cos(torch.angle(pf) - torch.angle(tf)))).mean()
    return ml + 0.1 * pl

class TrunkOperator(nn.Module):
    def __init__(self, query_dim, width, trunk_depth):
        super().__init__()
        self.width = width
        tp_count, tp_shapes = 0, []
        tp_count += query_dim * width + width
        tp_shapes.append((query_dim, width, width))
        for _ in range(2, trunk_depth):
            tp_count += width * width + width
            tp_shapes.append((width, width, width))
        self.trunk_param_count = tp_count
        self.trunk_param_shapes = tp_shapes
        self.head_output_dim = tp_count + width
        self.bias = nn.Parameter(torch.zeros([1]))

    def forward(self, head_output, query):
        B = head_output.shape[0]
        tp = head_output[:, :self.trunk_param_count] * 0.01
        B_coeff = head_output[:, self.trunk_param_count:]
        weights = []
        idx = 0
        for in_dim, out_dim, bs in self.trunk_param_shapes:
            ws = in_dim * out_dim
            w = tp[:, idx:idx+ws].view(B, in_dim, out_dim); idx += ws
            b = tp[:, idx:idx+bs].view(B, out_dim); idx += bs
            weights.append((w, b))
        Phi = query
        for i, (w, b) in enumerate(weights):
            Phi = torch.bmm(Phi, w) + b.unsqueeze(1)
            if i < len(weights) - 1: Phi = F.gelu(Phi)
        return torch.einsum('bp,bqp->bq', B_coeff, Phi) + self.bias

# ============================================================
# Config B Model
# ============================================================
class ConfigBModel(nn.Module):
    def __init__(self, seq_len=96, pred_len=96, width=192, branch_hidden=768,
                 trunk_depth=2, top_k_freq=5, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len; self.pred_len = pred_len
        self.top_k_freq = top_k_freq; self.branch_hidden = branch_hidden
        self.query_dim = 1 + 2 * top_k_freq + 2

        branch_dim = seq_len * 2 + (seq_len // 2 + 1) * 2
        self.encoder = build_encoder('patch_attn', branch_dim, branch_hidden,
                                     seq_len=seq_len, depth=6, activation='gelu', dropout=dropout)

        # 4 diverse trunks (width/depth vary)
        configs = [
            {'width': width, 'depth': 2},
            {'width': width, 'depth': 3},
            {'width': width // 2, 'depth': 2},
            {'width': width, 'depth': 2},
        ]
        self.trunks = nn.ModuleList([
            TrunkOperator(self.query_dim, c['width'], c['depth']) for c in configs])
        self.forecast_heads = nn.ModuleList([
            nn.Linear(branch_hidden, t.head_output_dim) for t in self.trunks])
        self.recon_heads = nn.ModuleList([
            nn.Linear(branch_hidden, t.head_output_dim) for t in self.trunks])
        for hs in [self.forecast_heads, self.recon_heads]:
            for h in hs:
                nn.init.xavier_normal_(h.weight, gain=0.1)
                nn.init.constant_(h.bias, 0)

    def _branch(self, x_ch, x_cross):
        fft = torch.fft.rfft(x_ch, dim=-1)
        return torch.cat([x_ch, x_cross, fft.real, fft.imag], dim=-1)

    def _forward_ch(self, x_ch, x_cross, tlen, mode='forecast'):
        z = self.encoder(self._branch(x_ch, x_cross))
        freqs, phases, lv, ls = extract_freq_info(x_ch, self.top_k_freq)
        t_s = 1.0 if mode == 'forecast' else 0.0
        t_e = t_s + tlen / self.seq_len if mode == 'forecast' else 1.0
        t = torch.linspace(t_s, t_e, tlen, device=x_ch.device, dtype=x_ch.dtype).unsqueeze(-1)
        iq = build_informed_query(t, freqs, phases, lv, ls)
        heads = self.forecast_heads if mode == 'forecast' else self.recon_heads
        return sum(trunk(head(z), iq) for trunk, head in zip(self.trunks, heads))

    def get_representation(self, x_enc):
        B, S, C = x_enc.shape
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = (x_enc - means) / torch.sqrt(
            torch.var(x_enc - means, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_cross = x_enc.mean(dim=-1)
        return torch.stack([self.encoder(self._branch(x_enc[:,:,ch], x_cross))
                           for ch in range(C)], dim=1)

    def forecast(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None,
                 target_pred_len=None, **kw):
        if target_pred_len is None: target_pred_len = self.pred_len
        B, S, C = x_enc.shape
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev
        x_cross = x_enc.mean(dim=-1)
        out = torch.stack([self._forward_ch(x_enc[:,:,ch], x_cross, target_pred_len, 'forecast')
                           for ch in range(C)], dim=-1)
        return out * stdev + means

    def reconstruct(self, x_enc):
        B, S, C = x_enc.shape
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev
        x_cross = x_enc.mean(dim=-1)
        out = torch.stack([self._forward_ch(x_enc[:,:,ch], x_cross, S, 'recon')
                           for ch in range(C)], dim=-1)
        return out * stdev + means

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None,
                target_pred_len=None, **kw):
        return self.forecast(x_enc, target_pred_len=target_pred_len)

# ============================================================
# Synthetic Dataset
# ============================================================
class SyntheticWindowDataset(Dataset):
    def __init__(self, arrow_path, seq_len=96, stride=48, max_samples=100000):
        self.windows = []
        ds = Dataset_GaussianPCoregionalization(
            root_path='./', data_path=arrow_path,
            n_variables=160, seq_len=seq_len, pred_len=seq_len,
            size=[seq_len, 0, seq_len], synthetic_length=1024, stride=stride)
        for i in range(min(len(ds), max_samples)):
            x, y, _, _ = ds[i]
            if isinstance(x, torch.Tensor): x = x.numpy()
            if x.ndim == 1: x = x.reshape(-1, 1)
            ch = np.random.randint(0, x.shape[1])
            w = x[:, ch].astype(np.float32)
            s = np.std(w)
            if s > 1e-8: self.windows.append(np.clip((w - np.mean(w)) / s, -10, 10))
        self.windows = np.array(self.windows, dtype=np.float32)
        print(f'SyntheticWindowDataset: {len(self.windows)} windows')
    def __len__(self): return len(self.windows)
    def __getitem__(self, idx):
        return torch.tensor(self.windows[idx], dtype=torch.float32).unsqueeze(-1)

# ============================================================
# Pre-training
# ============================================================
def pretrain(model, device, save_path, epochs=20, lr=0.0003, mask_rate=0.4,
             spectral_weight=0.01):
    print(f'\n{"="*60}')
    print(f'Config B Full Pre-training (83M, Pile+Synth)')
    print(f'  spectral_weight={spectral_weight}')
    print(f'{"="*60}')

    real_ds = PilePretrainDataset(seq_len=96, stride=48,
                                   pile_root='./dataset/time_series_pile')
    print(f'Real: {len(real_ds)} windows')
    synth_ds = SyntheticWindowDataset('tempopfn_15k_1024.arrow',
                                      seq_len=96, stride=48, max_samples=100000)
    combined = ConcatDataset([real_ds, synth_ds])
    n_val = min(10000, len(combined) // 10)
    n_train = len(combined) - n_val
    train_ds, _ = random_split(combined, [n_train, n_val])
    train_dl = DataLoader(train_ds, batch_size=64, shuffle=True,
                          num_workers=4, drop_last=True, pin_memory=True)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    print(f'Train: {n_train}, Steps/epoch: {len(train_dl)}')

    best_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        losses, rls, nls, sls = [], [], [], []
        t0 = time.time()
        for i, batch_x in enumerate(train_dl):
            batch_x = batch_x.float().to(device)
            B, S, C = batch_x.shape
            optimizer.zero_grad()

            mask = (torch.rand_like(batch_x) > mask_rate).float()
            recon_out = model.reconstruct(batch_x * mask)
            lm = F.mse_loss(recon_out, batch_x, reduction='none')
            im = 1.0 - mask
            rl = (lm * im).sum() / im.sum().clamp(min=1)

            split = torch.randint(24, 72, (1,)).item()
            ctx = batch_x[:, :split, :]
            tgt = batch_x[:, split:, :]
            tl = S - split
            cp = F.pad(ctx, (0, 0, 0, S - split))
            no = model.forecast(cp, target_pred_len=tl)
            nl = F.mse_loss(no, tgt)

            sr = spectral_loss(recon_out.squeeze(-1), batch_x.squeeze(-1))
            sn = spectral_loss(no.squeeze(-1), tgt.squeeze(-1))
            sl = sr + sn

            loss = rl + nl + spectral_weight * sl
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            losses.append(loss.item()); rls.append(rl.item())
            nls.append(nl.item()); sls.append(sl.item())

            if (i + 1) % 200 == 0:
                print(f'  iter {i+1}/{len(train_dl)}: '
                      f'recon={np.mean(rls[-200:]):.4f} '
                      f'nt={np.mean(nls[-200:]):.4f} '
                      f'spec={np.mean(sls[-200:]):.2f}')

        scheduler.step()
        el = time.time() - t0
        print(f'Epoch {epoch+1}/{epochs}: loss={np.mean(losses):.4f} '
              f'(recon={np.mean(rls):.4f} nt={np.mean(nls):.4f} spec={np.mean(sls):.2f}) '
              f'lr={scheduler.get_last_lr()[0]:.6f} ({el:.0f}s)')

        if np.mean(losses) < best_loss:
            best_loss = np.mean(losses)
            torch.save(model.state_dict(), save_path)
            print(f'  Saved (best={best_loss:.4f})')

    model.load_state_dict(torch.load(save_path))
    return model

# ============================================================
# Eval
# ============================================================
def eval_forecasting(model, device):
    print(f'\n{"="*60}\nForecasting Eval\n{"="*60}')
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
    model.eval()
    for dn, (d, r, f, e) in datasets.items():
        for pl in [96, 192, 336, 720]:
            a = SimpleNamespace(seq_len=96,pred_len=pl,label_len=48,data=d,root_path=r,
                data_path=f,features='M',target='OT',freq='h',embed='timeF',
                enc_in=e,dec_in=e,c_out=e,num_workers=2,batch_size=32,
                exp_name='MTSF',ordered_data=False,data_amount=-1,
                combine_Gaussian_datasets=False,synthetic_data_path='',
                synthetic_root_path='./',synthetic_length=1024,stride=-1)
            _, tdl = data_provider(a, 'test')
            ps, ts = [], []
            with torch.no_grad():
                for bx, by, _, _ in tdl:
                    bx = bx.float().to(device)
                    ps.append(model.forecast(bx, target_pred_len=pl).cpu().numpy())
                    ts.append(by[:, -pl:, :].numpy())
            p, t = np.concatenate(ps), np.concatenate(ts)
            mse = np.mean((p-t)**2)
            k = f'{dn}_{pl}'
            m = mlp.get(k)
            g = f'{(mse/m-1)*100:+.1f}%' if m else '-'
            print(f'  {k}: MSE={mse:.4f}  MOMENT_LP={m or "-"}  gap={g}')

def eval_imputation(model, device):
    print(f'\n{"="*60}\nImputation Eval\n{"="*60}')
    datasets = {
        'ETTh1': ('ETTh1','./dataset/ETT-small/','ETTh1.csv',7),
        'ETTh2': ('ETTh2','./dataset/ETT-small/','ETTh2.csv',7),
        'ETTm1': ('ETTm1','./dataset/ETT-small/','ETTm1.csv',7),
        'ETTm2': ('ETTm2','./dataset/ETT-small/','ETTm2.csv',7),
        'Weather': ('custom','./dataset/weather/','weather.csv',21),
    }
    moment = {'ETTh1':(0.402,0.139),'ETTh2':(0.125,0.061),
              'ETTm1':(0.202,0.074),'ETTm2':(0.078,0.031),'Weather':(0.082,0.035)}
    model.eval()
    for dn,(d,r,f,e) in datasets.items():
        a = SimpleNamespace(seq_len=96,pred_len=96,label_len=0,data=d,root_path=r,
            data_path=f,features='M',target='OT',freq='h',embed='timeF',
            enc_in=e,dec_in=e,c_out=e,num_workers=2,batch_size=32,
            exp_name='MTSF',ordered_data=False,data_amount=-1,
            combine_Gaussian_datasets=False,synthetic_data_path='',
            synthetic_root_path='./',synthetic_length=1024,stride=-1)
        _, tdl = data_provider(a, 'test')
        ams = []
        for mr in [0.125, 0.25, 0.375, 0.5]:
            torch.manual_seed(2021)
            ps, ts, ms = [], [], []
            with torch.no_grad():
                for bx, by, _, _ in tdl:
                    bx = bx.float().to(device)
                    mk = (torch.rand_like(bx)>mr).float()
                    ps.append(model.reconstruct(bx*mk).cpu().numpy())
                    ts.append(bx.cpu().numpy()); ms.append(mk.cpu().numpy())
            p,t,m = np.concatenate(ps),np.concatenate(ts),np.concatenate(ms)
            mse = np.mean((p[m==0]-t[m==0])**2); ams.append(mse)
        avg = np.mean(ams)
        m0,ml = moment.get(dn,(None,None))
        print(f'  {dn}: Mean MSE={avg:.4f}  (MOMENT_0={m0}, LP={ml})')

def eval_classification(model, device):
    print(f'\n{"="*60}\nClassification Eval\n{"="*60}')
    h = model.branch_hidden
    for p in model.parameters(): p.requires_grad = False
    for ds in ['Epilepsy','FingerMovements','BasicMotions','NATOPS','EthanolConcentration']:
        try:
            cr = './dataset/classification/Multivariate_ts'
            trd = Dataset_Classification(root_path=cr,flag='train',size=[96,0,96],data_path=ds)
            ted = Dataset_Classification(root_path=cr,flag='test',size=[96,0,96],data_path=ds)
            trl = DataLoader(trd,batch_size=16,shuffle=True,drop_last=True)
            tel = DataLoader(ted,batch_size=16,shuffle=False)
            ch = nn.Sequential(nn.Linear(h,256),nn.GELU(),nn.Dropout(0.1),
                               nn.Linear(256,trd.n_classes)).to(device)
            op = optim.Adam(ch.parameters(),lr=0.001)
            ba = 0
            for ep in range(30):
                ch.train()
                for bx,lb,_,_ in trl:
                    bx=bx.float().to(device); lb=lb.long().to(device)
                    with torch.no_grad(): z=model.get_representation(bx).mean(dim=1)
                    l=nn.CrossEntropyLoss()(ch(z),lb); op.zero_grad(); l.backward(); op.step()
                ch.eval(); ps,ls=[],[]
                with torch.no_grad():
                    for bx,lb,_,_ in tel:
                        bx=bx.float().to(device); z=model.get_representation(bx).mean(dim=1)
                        ps.append(ch(z).argmax(-1).cpu().numpy()); ls.append(lb.numpy())
                ac=accuracy_score(np.concatenate(ls),np.concatenate(ps)); ba=max(ba,ac)
            print(f'  {ds}: Acc={ba:.4f}')
        except Exception as e: print(f'  {ds}: SKIP ({e})')
    for p in model.parameters(): p.requires_grad = True

# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    print('='*60)
    print('Config B: Additive Trunk + Informed Query — Full Pre-training')
    print('  83M params, Pile+Synth, 20 epochs')
    print('  spectral_weight=0.01')
    print('='*60)

    model = ConfigBModel(width=192, branch_hidden=768, trunk_depth=2, top_k_freq=5).to(DEVICE)
    n = sum(p.numel() for p in model.parameters())
    print(f'Model: {n/1e6:.1f}M params')

    save_path = 'checkpoints/configB_full.pth'
    os.makedirs('checkpoints', exist_ok=True)

    model = pretrain(model, DEVICE, save_path, epochs=20, lr=0.0003, spectral_weight=0.01)
    eval_forecasting(model, DEVICE)
    eval_imputation(model, DEVICE)
    eval_classification(model, DEVICE)

    print(f'\n{"="*60}\nALL DONE\n{"="*60}')
