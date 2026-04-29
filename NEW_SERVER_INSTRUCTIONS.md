# HypOPFN2 — New Server Instructions

**Goal:** 100% LOTSA fresh pretraining of 3 models (hyper4, decomp, hyper4+short).

**Note (2026-04-29 update):** `--learnable_alpha` is now default ON (=1). The
hypernet output scale α used to be hard-coded to 0.01 — making it learnable
yields ~1.5% MSE improvement overall (Weather -9%, ETT -2%) and the trained α
values land in [0.05, 0.5], i.e. 5–50× larger than the old constant. No flag
needed; runs automatically include this fix.
**Hardware:** 2× A100 + 4 TB disk + 80 GB RAM.
**Approach:** No checkpoint transfer — fresh training from scratch, code pulled from GitHub.

---

## Step 1 — Clone + venv (5–10 min)

```bash
cd /workspace
git clone --depth 1 https://github.com/sheoyon-jhin/HypOPFN2.git
cd HypOPFN2

python3 -m venv /workspace/timefound
source /workspace/timefound/bin/activate
pip install --upgrade pip
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# Sanity check
python -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available(), 'GPUs:', torch.cuda.device_count())"
```

Expected:
```
torch: 2.5.1+cu121 cuda: True GPUs: 2
```

---

## Step 2 — LOTSA download (background, 3–5 h)

```bash
cd /workspace/HypOPFN2
source /workspace/timefound/bin/activate
mkdir -p log
nohup python data/download_lotsa.py > log/lotsa_dl.log 2>&1 &
echo "LOTSA DL PID: $!"
# Progress: tail -f log/lotsa_dl.log
```

This creates `dataset/lotsa/` (~860 GB). The first training run auto-builds the cache.

---

## Step 3 — Launch 2 models in parallel when LOTSA download is ≥ 90% done

The first command builds the wps=10000 cache (~15 h, CPU-bound); the second run shares the same cache (cache HIT → instant start).

### GPU 0 — hyper4 100% (generalist, imputation specialist candidate)

```bash
cd /workspace/HypOPFN2
source /workspace/timefound/bin/activate
CUDA_VISIBLE_DEVICES=0 nohup python experiments/exp_v1_varlen_ext.py \
  --scale 9999 --max_seq_len 720 --epochs 40 --synth_n 500000 \
  --use_nll 0 --highfreq_nf 256 \
  --windows_per_series 10000 --batch_size 1024 --amp 1 --lr 1.2e-3 \
  --tag hyper4_100pct \
  > log/hyper4_100pct.log 2>&1 &
echo "hyper4_100pct PID: $!"
```

### GPU 1 — decomp 100% (generalist, M4 + classification specialist candidate)

```bash
cd /workspace/HypOPFN2
source /workspace/timefound/bin/activate
CUDA_VISIBLE_DEVICES=1 nohup python experiments/exp_v1_varlen_ext.py \
  --scale 9999 --max_seq_len 720 --epochs 40 --synth_n 500000 \
  --use_nll 0 --highfreq_nf 256 \
  --model_type decomp --decomp_kernels 49,25,7 \
  --windows_per_series 10000 --batch_size 1024 --amp 1 --lr 1.2e-3 \
  --tag decomp_100pct \
  > log/decomp_100pct.log 2>&1 &
echo "decomp_100pct PID: $!"
```

**Note:** Wait for the cache to finish building before launching the second one. The first run will print `[LOTSA cache MISS] building ...` followed by `[LOTSA cache HIT]` eventually. Then launch the second.

Easiest: launch hyper4 first, wait until log shows `AMP: bfloat16 autocast enabled` and an `iter` line. Then launch decomp — it will share the cache via mmap.

---

## Step 4 — Queue the 3rd model (hyper4+short, M4 specialist)

Run this **after** hyper4_100pct OR decomp_100pct finishes (~30 h later) on the freed GPU.

```bash
cd /workspace/HypOPFN2
source /workspace/timefound/bin/activate
CUDA_VISIBLE_DEVICES=<FREE_GPU> nohup python experiments/exp_v1_varlen_ext.py \
  --scale 9999 --max_seq_len 720 --epochs 40 --synth_n 500000 \
  --use_nll 0 --highfreq_nf 256 \
  --windows_per_series 10000 --batch_size 1024 --amp 1 --lr 1.2e-3 \
  --seq_choices 64,128,192,192,384,384,512,512,720,720,720 \
  --lotsa_short_aux 64 --short_pred 1 \
  --tag hyper4_100pct_short \
  > log/hyper4_100pct_short.log 2>&1 &
```

Automatic queue wrapper (optional — runs when hyper4_100pct finishes):

```bash
cd /workspace/HypOPFN2
nohup bash -c '
while kill -0 <HYPER4_PID> 2>/dev/null; do sleep 300; done
source /workspace/timefound/bin/activate
CUDA_VISIBLE_DEVICES=0 python experiments/exp_v1_varlen_ext.py \
  --scale 9999 --max_seq_len 720 --epochs 40 --synth_n 500000 \
  --use_nll 0 --highfreq_nf 256 \
  --windows_per_series 10000 --batch_size 1024 --amp 1 --lr 1.2e-3 \
  --seq_choices 64,128,192,192,384,384,512,512,720,720,720 \
  --lotsa_short_aux 64 --short_pred 1 \
  --tag hyper4_100pct_short \
  > log/hyper4_100pct_short.log 2>&1
' > log/queue_short.log 2>&1 &
```

Replace `<HYPER4_PID>` with the actual PID printed in Step 3.

---

## Step 5 — Monitoring

```bash
# Check all GPUs
nvidia-smi

# Pretrain progress (live)
tail -f /workspace/HypOPFN2/log/hyper4_100pct.log
tail -f /workspace/HypOPFN2/log/decomp_100pct.log

# Summary at a glance
for tag in hyper4_100pct decomp_100pct hyper4_100pct_short; do
  log=/workspace/HypOPFN2/log/${tag}.log
  [ -f $log ] && echo "=== $tag ===" && grep 'iter ' $log | tail -1
done
```

Expected pace once cache is built: ~11 iter/sec per A100 → ~30 h for 40 epochs at wps=10000, bs=1024.

---

## Checkpoints will save to

```
checkpoints/hyper4_100pct_full231B.pth
checkpoints/decomp_100pct_full231B.pth
checkpoints/hyper4_100pct_short_full231B.pth
```

Each ~125 MB. Send these back to the main server for FT evaluation.

---

## Expected full timeline

| Phase | Duration |
|---|---|
| Setup + LOTSA download | 5–6 h |
| Cache build (first training run) | ~15 h |
| hyper4_100pct + decomp_100pct in parallel (after cache) | ~30 h |
| hyper4_100pct_short (queued, on freed GPU) | ~30 h |
| **Total end-to-end** | **~3 days** |

---

## If anything fails

1. **CUDA mismatch**: check `nvidia-smi` driver version. If it's CUDA < 12.1, install torch with `cu118` index instead of `cu121`.
2. **Disk full**: `df -h`. LOTSA (860 GB) + cache (~480 GB at wps=10000) = 1.3 TB. 4 TB is plenty.
3. **OOM**: reduce `--batch_size` to 512 or 256.
4. **Slow first iters**: the cache is being built (~15 h first time only). Subsequent runs will be cache HIT.
5. **`git clone` fails for public repo**: fall back to `wget https://github.com/sheoyon-jhin/HypOPFN2/archive/refs/heads/main.tar.gz`.

---

## Questions? Ping the main server.
