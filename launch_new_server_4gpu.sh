#!/bin/bash
# 4 parallel experiments on new server (assumes 4 GPUs visible as 0,1,2,3).
# Run from /workspace/HypOPFN2 after extracting bundle.
#
# Setup (first run on new server):
#   bash setup_server.sh         # installs deps, creates .venv
#   # ensure dataset/ETT-small/ and dataset/weather/ are present
#   # ensure dataset/lotsa_cache/ has the .dat/.meta (or rebuild from LOTSA raw)
#   # ensure dataset/synth_cache/synth_n5000000_seq2160_w56c0c04d.npy (or wait ~30min for build)
#
# Usage: bash launch_new_server_4gpu.sh

set -e
cd /workspace/HypOPFN2
source .venv/bin/activate
NVIDIA_LIB_DIRS=$(find .venv -path "*/nvidia/*/lib" -type d | tr '\n' ':')
export LD_LIBRARY_PATH="$NVIDIA_LIB_DIRS:$LD_LIBRARY_PATH"
export TORCHDYNAMO_DISABLE=1

mkdir -p log

COMMON="--scale 9999 --max_seq_len 720 --epochs 5 \
  --use_nll 0 --highfreq_nf 256 --highfreq2_nf 512 \
  --windows_per_series 1000 --batch_size 1024 --amp 1 --lr 1.2e-3"

# ============================================================
# GPU 0 — Performance: cheby + synth10M (best 2 combined)
# ============================================================
echo "[$(date '+%F %T')] GPU 0: hyper4_uv10pct_5trunk_cheby_synth10M"
LOTSA_MULTIVARIATE=0 CUDA_VISIBLE_DEVICES=0 nohup \
python experiments/exp_v1_varlen_ext.py $COMMON \
  --synth_n 10000000 --use_cheby 1 \
  --tag hyper4_uv10pct_5trunk_cheby_synth10M \
  > log/hyper4_uv10pct_5trunk_cheby_synth10M.log 2>&1 < /dev/null & disown
echo "  PID: $!"

sleep 5

# ============================================================
# GPU 1 — Ablation: LOTSA-only (synth=0)
# Tests: how much synthetic prior contributes
# ============================================================
echo "[$(date '+%F %T')] GPU 1: hyper4_uv10pct_5trunk_lotsa_only"
LOTSA_MULTIVARIATE=0 CUDA_VISIBLE_DEVICES=1 nohup \
python experiments/exp_v1_varlen_ext.py $COMMON \
  --synth_n 1 \
  --tag hyper4_uv10pct_5trunk_lotsa_only \
  > log/hyper4_uv10pct_5trunk_lotsa_only.log 2>&1 < /dev/null & disown
echo "  PID: $!"

sleep 5

# ============================================================
# GPU 2 — Ablation: synth-only (LOTSA scale=0)
# Tests: pure operator-learning-style prior fitting
# Note: --scale 0 means 0% LOTSA → only synth dataset survives
# ============================================================
echo "[$(date '+%F %T')] GPU 2: hyper4_5trunk_synth5M_only (LOTSA=0)"
LOTSA_MULTIVARIATE=0 CUDA_VISIBLE_DEVICES=2 nohup \
python experiments/exp_v1_varlen_ext.py \
  --scale 1 --max_seq_len 720 --epochs 5 \
  --use_nll 0 --highfreq_nf 256 --highfreq2_nf 512 \
  --windows_per_series 1 --batch_size 1024 --amp 1 --lr 1.2e-3 \
  --synth_n 5000000 \
  --tag hyper4_5trunk_synth5M_only \
  > log/hyper4_5trunk_synth5M_only.log 2>&1 < /dev/null & disown
echo "  PID: $!"

sleep 5

# ============================================================
# GPU 3 — Ablation: 4-trunk (drop the highfreq2 nf=512)
# Tests: does the 5th trunk really help, or is 4-trunk enough?
# ============================================================
echo "[$(date '+%F %T')] GPU 3: hyper4_uv10pct_4trunk_synth5M (no highfreq2)"
LOTSA_MULTIVARIATE=0 CUDA_VISIBLE_DEVICES=3 nohup \
python experiments/exp_v1_varlen_ext.py \
  --scale 9999 --max_seq_len 720 --epochs 5 \
  --use_nll 0 --highfreq_nf 256 \
  --windows_per_series 1000 --batch_size 1024 --amp 1 --lr 1.2e-3 \
  --synth_n 5000000 \
  --tag hyper4_uv10pct_4trunk_synth5M_repro \
  > log/hyper4_uv10pct_4trunk_synth5M_repro.log 2>&1 < /dev/null & disown
echo "  PID: $!"

echo ""
echo "[$(date '+%F %T')] All 4 launched. Tail logs:"
echo "  tail -f log/hyper4_uv10pct_5trunk_cheby_synth10M.log"
echo "  tail -f log/hyper4_uv10pct_5trunk_lotsa_only.log"
echo "  tail -f log/hyper4_5trunk_synth5M_only.log"
echo "  tail -f log/hyper4_uv10pct_4trunk_synth5M_repro.log"
echo ""
echo "Restart watcher: nohup bash auto_eval_watcher.sh > log/auto_eval.log 2>&1 & disown"
