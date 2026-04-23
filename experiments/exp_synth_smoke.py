"""
Synthetic-only smoke test (LOTSA 다운로드 대기 중 파이프라인 검증)

목적:
- seq_len=192 모델 생성/학습 검증
- 개선된 SyntheticGapFiller 동작 확인
- Loss 감소 확인 (학습 가능성)
- 약 30-60분 완료 (GPU 1개)

Usage:
  CUDA_VISIBLE_DEVICES=0 python experiments/exp_synth_smoke.py
"""
import sys, os, time, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch, torch.nn.functional as F
import numpy as np
from torch import optim
from torch.utils.data import DataLoader

from experiments.exp_lotsa_scaling import (
    OperatorModel, SyntheticGapFiller, collate_batch
)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seq_len', type=int, default=192)
    p.add_argument('--n_samples', type=int, default=30000)
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--decomp', type=int, default=1)
    p.add_argument('--tag', type=str, default='synth_smoke')
    args = p.parse_args()

    torch.manual_seed(42); np.random.seed(42)

    print('=' * 60)
    print(f'Synthetic-only smoke test')
    print(f'  seq_len={args.seq_len}, n_samples={args.n_samples}')
    print(f'  epochs={args.epochs}, decomp={bool(args.decomp)}')
    print('=' * 60)

    # Synthetic data
    t0 = time.time()
    synth_ds = SyntheticGapFiller(n_samples=args.n_samples, seq_len=args.seq_len * 2)
    print(f'Synth gen: {time.time()-t0:.1f}s')

    dl = DataLoader(synth_ds, batch_size=args.batch_size, shuffle=True,
                    num_workers=4, drop_last=True, pin_memory=True)

    # Model
    model = OperatorModel(
        seq_len=args.seq_len,
        use_latent_decomp=bool(args.decomp)
    ).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'Model: {n_params/1e6:.1f}M params, seq_len={args.seq_len}')
    print(f'Steps/epoch: {len(dl):,}')

    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    os.makedirs('checkpoints', exist_ok=True)
    save_path = f'checkpoints/{args.tag}.pth'

    print('\n--- Training ---')
    best_loss = float('inf')
    epoch_losses = []
    for epoch in range(args.epochs):
        model.train()
        losses = []
        t_ep = time.time()
        for i, batch_windows in enumerate(dl):
            batch = collate_batch(batch_windows, seq_len=args.seq_len)
            if batch is None:
                continue
            ctx, qt, qv = [x.to(DEVICE) for x in batch]
            optimizer.zero_grad()
            loss = F.mse_loss(model.forward_train(ctx, qt), qv)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())

            if (i + 1) % 100 == 0:
                print(f'  epoch {epoch+1} iter {i+1}/{len(dl)}: loss={np.mean(losses[-100:]):.4f}')

        scheduler.step()
        avg = float(np.mean(losses))
        epoch_losses.append(avg)
        elapsed = time.time() - t_ep
        print(f'Epoch {epoch+1}/{args.epochs}: loss={avg:.4f}, {elapsed:.1f}s')

        if avg < best_loss:
            best_loss = avg
            torch.save(model.state_dict(), save_path)

    print('\n' + '=' * 60)
    print(f'DONE. Best loss: {best_loss:.4f}')
    print(f'Loss progression: {[f"{l:.4f}" for l in epoch_losses]}')
    print(f'Checkpoint saved: {save_path}')
    print('=' * 60)

    # Sanity check: loss should decrease
    if len(epoch_losses) >= 2:
        improvement = (epoch_losses[0] - epoch_losses[-1]) / epoch_losses[0] * 100
        print(f'Loss improvement: {improvement:.1f}%')
        if improvement > 10:
            print('✓ Model is learning properly')
        else:
            print('⚠ Loss not decreasing as expected — investigate')


if __name__ == '__main__':
    main()
