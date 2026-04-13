"""
Imputation Table 완성: best checkpoint으로 6 datasets × 4 mask rates
Transformer 63M checkpoint (ETTh1 m=0.125 = 0.115)

사용법:
  CUDA_VISIBLE_DEVICES=3 python experiments/exp_imputation_full_eval.py 2>&1 | tee log/eval/imputation_full.log
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
from types import SimpleNamespace
from experiments.exp_architecture_comparison import TransformerOperatorModel
from data_provider.data_factory import data_provider


def eval_imputation_full(model, device):
    datasets = {
        'Weather': ('custom', './dataset/weather/', 'weather.csv', 21),
        'ETTh1': ('ETTh1', './dataset/ETT-small/', 'ETTh1.csv', 7),
        'ETTh2': ('ETTh2', './dataset/ETT-small/', 'ETTh2.csv', 7),
        'ETTm1': ('ETTm1', './dataset/ETT-small/', 'ETTm1.csv', 7),
        'ETTm2': ('ETTm2', './dataset/ETT-small/', 'ETTm2.csv', 7),
    }
    mask_rates = [0.125, 0.25, 0.375, 0.5]

    model.eval()
    results = {}

    for dname, (data, root, fpath, enc_in) in datasets.items():
        args = SimpleNamespace(
            seq_len=96, pred_len=96, label_len=0,
            data=data, root_path=root, data_path=fpath,
            features='M', target='OT', freq='h', embed='timeF',
            enc_in=enc_in, dec_in=enc_in, c_out=enc_in,
            num_workers=2, batch_size=1,
            exp_name='MTSF', ordered_data=False, data_amount=-1,
            combine_Gaussian_datasets=False,
            synthetic_data_path='', synthetic_root_path='./',
            synthetic_length=1024, stride=-1,
        )
        _, test_dl = data_provider(args, 'test')

        for mr in mask_rates:
            torch.manual_seed(2021)
            preds, trues, masks = [], [], []
            with torch.no_grad():
                for bx, by, _, _ in test_dl:
                    bx = bx.float().to(device)
                    mask = (torch.rand_like(bx) > mr).float()
                    out = model.reconstruct(bx * mask)
                    preds.append(out.cpu().numpy())
                    trues.append(bx.cpu().numpy())
                    masks.append(mask.cpu().numpy())

            p = np.concatenate(preds)
            t = np.concatenate(trues)
            m = np.concatenate(masks)
            mse = np.mean((p[m == 0] - t[m == 0]) ** 2)
            mae = np.mean(np.abs(p[m == 0] - t[m == 0]))
            key = f'{dname}_m{mr}'
            results[key] = {'mse': mse, 'mae': mae}
            print(f'  {dname} mask={mr}: MSE={mse:.4f} MAE={mae:.4f}')

        # Mean
        ds_mses = [results[f'{dname}_m{mr}']['mse'] for mr in mask_rates]
        ds_maes = [results[f'{dname}_m{mr}']['mae'] for mr in mask_rates]
        print(f'  {dname} Mean: MSE={np.mean(ds_mses):.4f} MAE={np.mean(ds_maes):.4f}')
        print()

    return results


if __name__ == '__main__':
    device = torch.device('cuda')

    ckpt_path = 'checkpoints/scaleup_transformer_pile.pth'

    model = TransformerOperatorModel(
        seq_len=96, pred_len=96,
        width=128, hidden=512,
        n_heads=8, n_layers=4, trunk_depth=4
    ).to(device)

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    n_params = sum(p.numel() for p in model.parameters())
    print(f'Model: {n_params/1e6:.1f}M (Transformer + Fixed Trunk + Cross-channel)')
    print(f'Checkpoint: {ckpt_path}')

    print(f'\n{"="*60}')
    print('Imputation Full Eval (MOMENT Table 3 format)')
    print(f'{"="*60}\n')

    results = eval_imputation_full(model, device)

    # MOMENT comparison
    moment_0 = {
        'Weather': 0.082, 'ETTh1': 0.402, 'ETTh2': 0.125,
        'ETTm1': 0.202, 'ETTm2': 0.078,
    }
    moment_lp = {
        'Weather': 0.035, 'ETTh1': 0.139, 'ETTh2': 0.061,
        'ETTm1': 0.074, 'ETTm2': 0.031,
    }

    print(f'\n{"="*60}')
    print('Mean MSE comparison (averaged over mask rates)')
    print(f'{"="*60}')
    print(f'{"Dataset":<12} {"Ours":>8} {"MOMENT_0":>10} {"MOMENT_LP":>10}')
    print('-' * 42)

    datasets_list = ['Weather', 'ETTh1', 'ETTh2', 'ETTm1', 'ETTm2']
    for dname in datasets_list:
        ours = np.mean([results[f'{dname}_m{mr}']['mse'] for mr in [0.125, 0.25, 0.375, 0.5]])
        m0 = moment_0.get(dname, '-')
        mlp = moment_lp.get(dname, '-')
        print(f'  {dname:<12} {ours:>8.4f} {m0:>10} {mlp:>10}')
