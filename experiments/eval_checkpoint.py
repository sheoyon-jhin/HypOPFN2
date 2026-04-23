"""
Generic eval for any trained checkpoint (FeDaL-style report).

Usage:
  CUDA_VISIBLE_DEVICES=2 python experiments/eval_checkpoint.py \
      --ckpt checkpoints/synth_full_base.pth --decomp 0 --seq_len 192 \
      --tag synth_full_base_eval
"""
import sys, os, json, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np

from experiments.exp_lotsa_scaling import OperatorModel, eval_forecast

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

REFERENCE = {
    'ETTh1':   {'MSE': 0.380, 'MAE': 0.409},
    'ETTh2':   {'MSE': 0.334, 'MAE': 0.377},
    'ETTm1':   {'MSE': 0.319, 'MAE': 0.365},
    'ETTm2':   {'MSE': 0.261, 'MAE': 0.319},
    'Weather': {'MSE': 0.213, 'MAE': 0.255},
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--decomp', type=int, default=0)
    p.add_argument('--seq_len', type=int, default=192)
    p.add_argument('--tag', type=str, required=True)
    args = p.parse_args()

    print('=' * 70)
    print(f'EVAL: {args.ckpt}')
    print(f'  seq_len={args.seq_len}, decomp={bool(args.decomp)}')
    print('=' * 70)

    model = OperatorModel(seq_len=args.seq_len, use_latent_decomp=bool(args.decomp)).to(DEVICE)
    state = torch.load(args.ckpt, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    results = eval_forecast(model, args.seq_len)

    # Compare with FeDaL
    print('\n' + '=' * 70)
    print(f'{"Dataset":<10} {"Ours (MSE/MAE)":<20} {"FeDaL (MSE/MAE)":<20} {"ΔMSE":>8}')
    print('-' * 70)
    for dn in ['ETTh1','ETTh2','ETTm1','ETTm2','Weather']:
        ours = results.get(f'{dn}_avg')
        ref = REFERENCE[dn]
        if ours:
            gap = (ours['MSE'] - ref['MSE']) / ref['MSE'] * 100
            print(f'{dn:<10} {ours["MSE"]:.4f} / {ours["MAE"]:.4f}   '
                  f'{ref["MSE"]:.3f} / {ref["MAE"]:.3f}        {gap:>+6.1f}%')
    if 'overall_avg' in results:
        ov = results['overall_avg']
        print('-' * 70)
        print(f'{"OVERALL":<10} {ov["MSE"]:.4f} / {ov["MAE"]:.4f}')

    os.makedirs('results', exist_ok=True)
    with open(f'results/{args.tag}.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved: results/{args.tag}.json')


if __name__ == '__main__':
    main()
