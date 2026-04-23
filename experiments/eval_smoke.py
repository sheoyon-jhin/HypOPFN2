"""
Smoke test checkpoint eval (FeDaL/MOMENT 호환 포맷).

논문 기준:
- FeDaL Table 3: ETTh1/h2/m1/m2, Weather @ pred_len {96,192,336,720}, MSE+MAE
- MOMENT: 같은 프로토콜 (GPT4TS)

참조 수치 (FeDaL Table 3, full-shot):
  ETTh1: MSE 0.380, MAE 0.409
  ETTh2: MSE 0.334, MAE 0.377
  ETTm1: MSE 0.319, MAE 0.365
  ETTm2: MSE 0.261, MAE 0.319
  Weather: MSE 0.213, MAE 0.255

Usage:
  CUDA_VISIBLE_DEVICES=1 python experiments/eval_smoke.py
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np

from experiments.exp_lotsa_scaling import OperatorModel, eval_forecast

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Paper reference (FeDaL Table 3, full-shot long-term forecasting)
REFERENCE = {
    'ETTh1':   {'MSE': 0.380, 'MAE': 0.409},
    'ETTh2':   {'MSE': 0.334, 'MAE': 0.377},
    'ETTm1':   {'MSE': 0.319, 'MAE': 0.365},
    'ETTm2':   {'MSE': 0.261, 'MAE': 0.319},
    'Weather': {'MSE': 0.213, 'MAE': 0.255},
}


def main():
    ckpt_path = 'checkpoints/synth_smoke.pth'
    seq_len = 192
    use_decomp = True

    print('=' * 70)
    print('EVAL PIPELINE VALIDATION (synth_smoke checkpoint)')
    print(f'  checkpoint: {ckpt_path}, seq_len: {seq_len}, decomp: {use_decomp}')
    print('  NOTE: MSE/MAE will be HIGH (synthetic-only, 5 epochs only).')
    print('        Purpose: validate eval pipeline correctness, NOT scores.')
    print('=' * 70)

    model = OperatorModel(seq_len=seq_len, use_latent_decomp=use_decomp).to(DEVICE)
    state = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    print('\n[1/2] Running eval (ETT × 4 + Weather) × pred_len {96,192,336,720}')
    print('-' * 70)
    results = eval_forecast(model, seq_len)

    # Compare with FeDaL reference
    print('\n[2/2] Comparison with FeDaL reference (Table 3, ICLR 2026)')
    print('=' * 70)
    print(f'{"Dataset":<10} {"Ours (MSE)":>12} {"FeDaL":>10} {"Δ%":>8}  |  '
          f'{"Ours (MAE)":>12} {"FeDaL":>10} {"Δ%":>8}')
    print('-' * 70)
    for dn in ['ETTh1','ETTh2','ETTm1','ETTm2','Weather']:
        ours = results.get(f'{dn}_avg')
        ref = REFERENCE[dn]
        if ours:
            mse_gap = (ours['MSE'] - ref['MSE']) / ref['MSE'] * 100
            mae_gap = (ours['MAE'] - ref['MAE']) / ref['MAE'] * 100
            print(f'{dn:<10} {ours["MSE"]:>12.4f} {ref["MSE"]:>10.3f} {mse_gap:>+7.1f}%  |  '
                  f'{ours["MAE"]:>12.4f} {ref["MAE"]:>10.3f} {mae_gap:>+7.1f}%')

    os.makedirs('results', exist_ok=True)
    with open('results/eval_smoke.json', 'w') as f:
        json.dump(results, f, indent=2)
    print('\nSaved: results/eval_smoke.json')

    # Pipeline health
    expected = [f'{d}_{p}' for d in ['ETTh1','ETTh2','ETTm1','ETTm2','Weather']
                for p in [96, 192, 336, 720]]
    missing = [k for k in expected if k not in results]
    if missing:
        print(f'\n⚠ MISSING: {missing}')
    else:
        print('\n✓ All 20 (dataset × pred_len) evaluated. Pipeline OK.')


if __name__ == '__main__':
    main()
