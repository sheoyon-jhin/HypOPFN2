#!/bin/bash
"""
LOTSA Scaling Study — 4 GPU 병렬 실행

10 experiments: 5 scales × 2 variants (baseline, latent_decomp)
4 GPU에서 병렬로 효율적 실행

Usage: bash run_lotsa_experiments.sh
"""
set +e
cd "$(dirname "$0")"

echo "============================================================"
echo "LOTSA Scaling Study — $(date)"
echo "GPUs: $(nvidia-smi --query-gpu=count --format=csv,noheader | head -1)"
echo "============================================================"

PYTHON=${PYTHON:-python}
RESULTS_DIR=results
mkdir -p $RESULTS_DIR checkpoints log

# ============================================================
# Phase 1: Small scales (1%, 5%) — fast, all 4 GPUs
# ============================================================
echo ""
echo "=== Phase 1: 1% and 5% (4 experiments, ~4-6h) ==="

CUDA_VISIBLE_DEVICES=0 nohup $PYTHON experiments/exp_lotsa_scaling.py \
    --scale 1 --decomp 0 --tag lotsa_s1_base > log/lotsa_s1_base.log 2>&1 &
P1=$!; echo "GPU 0: 1% baseline (PID $P1)"

CUDA_VISIBLE_DEVICES=1 nohup $PYTHON experiments/exp_lotsa_scaling.py \
    --scale 1 --decomp 1 --tag lotsa_s1_decomp > log/lotsa_s1_decomp.log 2>&1 &
P2=$!; echo "GPU 1: 1% + latent decomp (PID $P2)"

CUDA_VISIBLE_DEVICES=2 nohup $PYTHON experiments/exp_lotsa_scaling.py \
    --scale 5 --decomp 0 --tag lotsa_s5_base > log/lotsa_s5_base.log 2>&1 &
P3=$!; echo "GPU 2: 5% baseline (PID $P3)"

CUDA_VISIBLE_DEVICES=3 nohup $PYTHON experiments/exp_lotsa_scaling.py \
    --scale 5 --decomp 1 --tag lotsa_s5_decomp > log/lotsa_s5_decomp.log 2>&1 &
P4=$!; echo "GPU 3: 5% + latent decomp (PID $P4)"

echo "Waiting for Phase 1..."
wait $P1 $P2 $P3 $P4
echo "Phase 1 complete — $(date)"

# ============================================================
# Phase 2: Medium scales (10%) — 4 GPUs
# ============================================================
echo ""
echo "=== Phase 2: 10% (2 experiments, ~7h) ==="

CUDA_VISIBLE_DEVICES=0 nohup $PYTHON experiments/exp_lotsa_scaling.py \
    --scale 10 --decomp 0 --tag lotsa_s10_base > log/lotsa_s10_base.log 2>&1 &
P5=$!; echo "GPU 0: 10% baseline (PID $P5)"

CUDA_VISIBLE_DEVICES=1 nohup $PYTHON experiments/exp_lotsa_scaling.py \
    --scale 10 --decomp 1 --tag lotsa_s10_decomp > log/lotsa_s10_decomp.log 2>&1 &
P6=$!; echo "GPU 1: 10% + latent decomp (PID $P6)"

# GPU 2,3 can start 30% while 10% runs
CUDA_VISIBLE_DEVICES=2 nohup $PYTHON experiments/exp_lotsa_scaling.py \
    --scale 30 --decomp 0 --tag lotsa_s30_base > log/lotsa_s30_base.log 2>&1 &
P7=$!; echo "GPU 2: 30% baseline (PID $P7)"

CUDA_VISIBLE_DEVICES=3 nohup $PYTHON experiments/exp_lotsa_scaling.py \
    --scale 30 --decomp 1 --tag lotsa_s30_decomp > log/lotsa_s30_decomp.log 2>&1 &
P8=$!; echo "GPU 3: 30% + latent decomp (PID $P8)"

echo "Waiting for Phase 2..."
wait $P5 $P6
echo "10% complete — $(date)"

# ============================================================
# Phase 3: Large scales (50%) — use freed GPUs
# ============================================================
echo ""
echo "=== Phase 3: 50% (2 experiments, ~20h) ==="

CUDA_VISIBLE_DEVICES=0 nohup $PYTHON experiments/exp_lotsa_scaling.py \
    --scale 50 --decomp 0 --tag lotsa_s50_base > log/lotsa_s50_base.log 2>&1 &
P9=$!; echo "GPU 0: 50% baseline (PID $P9)"

CUDA_VISIBLE_DEVICES=1 nohup $PYTHON experiments/exp_lotsa_scaling.py \
    --scale 50 --decomp 1 --tag lotsa_s50_decomp > log/lotsa_s50_decomp.log 2>&1 &
P10=$!; echo "GPU 1: 50% + latent decomp (PID $P10)"

echo "Waiting for all remaining..."
wait $P7 $P8 $P9 $P10
echo "All experiments complete — $(date)"

# ============================================================
# Summary
# ============================================================
echo ""
echo "============================================================"
echo "ALL DONE — $(date)"
echo "============================================================"
echo ""
echo "Results:"
for f in $RESULTS_DIR/lotsa_*.json; do
    if [ -f "$f" ]; then
        echo "  $(basename $f): $(python -c "import json; d=json.load(open('$f')); print(f'scale={d[\"scale\"]}% decomp={d[\"decomp\"]} loss={d[\"train_loss\"]:.4f}')" 2>/dev/null)"
    fi
done
echo ""
echo "Logs: log/lotsa_*.log"
echo "Checkpoints: checkpoints/lotsa_*.pth"
