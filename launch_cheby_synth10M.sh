#!/bin/bash
# Two parallel runs:
#   GPU 0: 5-trunk + synth 10M (default DOMAIN_WEIGHTS, 81GB cache build first)
#   GPU 1: 5-trunk + Chebyshev poly + synth 5M (existing cache)
# Usage: bash launch_cheby_synth10M.sh

cd /workspace/HypOPFN2
source .venv/bin/activate
NVIDIA_LIB_DIRS=$(find .venv -path "*/nvidia/*/lib" -type d | tr '\n' ':')
export LD_LIBRARY_PATH="$NVIDIA_LIB_DIRS:$LD_LIBRARY_PATH"
export TORCHDYNAMO_DISABLE=1

mkdir -p log

echo "[$(date '+%F %T')] Launching hyper4_uv10pct_5trunk_synth10M on GPU 0"
LOTSA_MULTIVARIATE=0 CUDA_VISIBLE_DEVICES=0 nohup \
python experiments/exp_v1_varlen_ext.py \
  --scale 9999 --max_seq_len 720 --epochs 5 \
  --synth_n 10000000 \
  --use_nll 0 --highfreq_nf 256 --highfreq2_nf 512 \
  --windows_per_series 1000 --batch_size 1024 --amp 1 --lr 1.2e-3 \
  --tag hyper4_uv10pct_5trunk_synth10M \
  > log/hyper4_uv10pct_5trunk_synth10M.log 2>&1 < /dev/null & disown
echo "  PID: $!"

sleep 5

echo "[$(date '+%F %T')] Launching hyper4_uv10pct_5trunk_cheby_synth5M on GPU 1"
LOTSA_MULTIVARIATE=0 CUDA_VISIBLE_DEVICES=1 nohup \
python experiments/exp_v1_varlen_ext.py \
  --scale 9999 --max_seq_len 720 --epochs 5 \
  --synth_n 5000000 \
  --use_nll 0 --highfreq_nf 256 --highfreq2_nf 512 \
  --use_cheby 1 \
  --windows_per_series 1000 --batch_size 1024 --amp 1 --lr 1.2e-3 \
  --tag hyper4_uv10pct_5trunk_cheby_synth5M \
  > log/hyper4_uv10pct_5trunk_cheby_synth5M.log 2>&1 < /dev/null & disown
echo "  PID: $!"

echo "[$(date '+%F %T')] Both launched"

# Restart auto-eval watcher (uses new tags + cheby/12layer flags)
echo "[$(date '+%F %T')] Starting auto_eval_watcher"
nohup bash auto_eval_watcher.sh > log/auto_eval.log 2>&1 < /dev/null & disown
echo "  Watcher PID: $!"
