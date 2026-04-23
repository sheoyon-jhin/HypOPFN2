#!/bin/bash
# Orchestrate FT matrix: 3 tasks × 9 ckpts × 2 configs, round-robin across GPU 0/1/3.
# GPU 2 is saturated by hyper4 20% training — skip.
# GPU 1 eventually gets freed after mask_recon — run classification there primarily.
#
# Usage: bash experiments/run_ft_matrix.sh <task> <gpu>
#   task ∈ {cls, impute, m4, all}
#   gpu  ∈ {0, 1, 3}
# Example:
#   bash experiments/run_ft_matrix.sh impute 0
#   bash experiments/run_ft_matrix.sh m4 3
#   bash experiments/run_ft_matrix.sh cls 1

set -u
TASK=${1:-all}
GPU=${2:-0}
cd /workspace/HypOPFN2

PY=/workspace/timefound/bin/python
mkdir -p log/ft results/ft

# Ckpt list: name | path | extra_args
CKPTS=(
  "hyper4_base|checkpoints/hyper4_full231B.pth|--highfreq_nf 256"
  "hyper4_1pct|checkpoints/hyper4_1pct_full231B.pth|--highfreq_nf 256"
  "hyper4_10pct|checkpoints/hyper4_10pct_full231B.pth|--highfreq_nf 256"
  "hyper4_spec2|checkpoints/hyper4_spec2_full231B.pth|--highfreq_nf 256"
  "decomp_base|checkpoints/decomp_hyper4_full231B.pth|--highfreq_nf 256 --model_type decomp --decomp_kernels 49,25,7"
  "decomp_1pct|checkpoints/decomp_hyper4_1pct_full231B.pth|--highfreq_nf 256 --model_type decomp --decomp_kernels 49,25,7"
  "allfixed|checkpoints/allfixed_hf256_full231B.pth|--highfreq_nf 256 --all_fixed 1"
  "msiq|checkpoints/allfixed_hf256_msiq_full231B.pth|--highfreq_nf 256 --all_fixed 1 --multi_scale_iq 1"
  "msiq_1pct|checkpoints/msiq_1pct_full231B.pth|--highfreq_nf 256 --multi_scale_iq 1"
)

run_cls() {
  local name=$1 ckpt=$2 extra=$3
  # Config A: ft_epochs=3, lr=1e-3, linear head
  CUDA_VISIBLE_DEVICES=$GPU $PY experiments/ft_classification.py \
    --ckpt $ckpt $extra --ft_epochs 3 --lr 1e-3 --head_type linear \
    --tag ft/${name}_cls_A > log/ft/${name}_cls_A.log 2>&1
  # Config B: ft_epochs=5, lr=5e-4, mlp head
  CUDA_VISIBLE_DEVICES=$GPU $PY experiments/ft_classification.py \
    --ckpt $ckpt $extra --ft_epochs 5 --lr 5e-4 --head_type mlp \
    --tag ft/${name}_cls_B > log/ft/${name}_cls_B.log 2>&1
}

run_impute() {
  local name=$1 ckpt=$2 extra=$3
  # Config A: ft_epochs=3, lr=1e-4
  CUDA_VISIBLE_DEVICES=$GPU $PY experiments/ft_imputation.py \
    --ckpt $ckpt $extra --ft_epochs 3 --lr 1e-4 --max_steps_per_epoch 300 \
    --tag ft/${name}_impute_A > log/ft/${name}_impute_A.log 2>&1
  # Config B: ft_epochs=5, lr=5e-5
  CUDA_VISIBLE_DEVICES=$GPU $PY experiments/ft_imputation.py \
    --ckpt $ckpt $extra --ft_epochs 5 --lr 5e-5 --max_steps_per_epoch 300 \
    --tag ft/${name}_impute_B > log/ft/${name}_impute_B.log 2>&1
}

run_m4() {
  local name=$1 ckpt=$2 extra=$3
  # Config A: ft_epochs=3, lr=1e-4
  CUDA_VISIBLE_DEVICES=$GPU $PY experiments/ft_m4.py \
    --ckpt $ckpt $extra --ft_epochs 3 --lr 1e-4 \
    --tag ft/${name}_m4_A > log/ft/${name}_m4_A.log 2>&1
  # Config B: ft_epochs=5, lr=5e-4
  CUDA_VISIBLE_DEVICES=$GPU $PY experiments/ft_m4.py \
    --ckpt $ckpt $extra --ft_epochs 5 --lr 5e-4 \
    --tag ft/${name}_m4_B > log/ft/${name}_m4_B.log 2>&1
}

echo "[$(date)] Starting FT matrix: task=$TASK gpu=$GPU"
for entry in "${CKPTS[@]}"; do
  IFS='|' read -r name ckpt extra <<< "$entry"
  if [ ! -f "$ckpt" ]; then
    echo "[$(date)] SKIP $name (no ckpt: $ckpt)"
    continue
  fi
  echo "[$(date)] === $name ==="
  case "$TASK" in
    cls)    run_cls    $name $ckpt "$extra" ;;
    impute) run_impute $name $ckpt "$extra" ;;
    m4)     run_m4     $name $ckpt "$extra" ;;
    all)
      run_cls    $name $ckpt "$extra"
      run_impute $name $ckpt "$extra"
      run_m4     $name $ckpt "$extra"
      ;;
    *) echo "Unknown task: $TASK"; exit 1 ;;
  esac
  echo "[$(date)] === $name done ==="
done
echo "[$(date)] FT matrix complete: task=$TASK gpu=$GPU"
