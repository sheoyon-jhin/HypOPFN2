#!/bin/bash
# 학습 중 체크포인트 파일이 갱신될 때마다 자동으로 eval 실행
# Usage: nohup bash auto_eval_watcher.sh > log/auto_eval.log 2>&1 & disown

cd /workspace/HypOPFN2
source .venv/bin/activate

CKPT_DIR=/workspace/HypOPFN2/checkpoints
LOG_DIR=/workspace/HypOPFN2/log
RESULTS_DIR=/workspace/HypOPFN2/log/eval_history
mkdir -p $RESULTS_DIR

# 모델별 마지막 eval 시각 (체크포인트 mtime) 추적
declare -A LAST_MTIME

echo "[$(date '+%F %T')] auto_eval_watcher started (PID $$)"

while true; do
  for tag in hyper4_100pct decomp_100pct hyper4_mv decomp_mv hyper4_mv10pct_synth700K hyper4_mv50pct_synth3500K hyper4_cauker decomp_cauker hyper4_uv10pct_synth2M hyper4_uv10pct_synth5M hyper4_uv10pct_5trunk_synth5M hyper4_uv10pct_5trunk_ettboost_synth10M hyper4_uv10pct_5trunk_maskrecon hyper4_uv10pct_5trunk_12layer hyper4_uv10pct_5trunk_synth10M hyper4_uv10pct_5trunk_cheby_synth5M; do
    ckpt="$CKPT_DIR/${tag}.pth"
    if [ ! -f "$ckpt" ]; then continue; fi

    # 현재 체크포인트 mtime
    mt=$(stat -c "%Y" "$ckpt" 2>/dev/null)
    last="${LAST_MTIME[$tag]:-0}"
    if [ "$mt" -le "$last" ]; then continue; fi

    # 새 체크포인트 감지 → eval 실행 (파일 쓰기 안정화 위해 10초 대기)
    sleep 10

    # 현재 학습 진행 epoch 추출 (Saved 라인 기반)
    epoch_done=$(grep -cE "^  Saved" $LOG_DIR/${tag}.log 2>/dev/null)
    timestamp=$(date '+%F_%H%M')
    out_log="$RESULTS_DIR/${tag}_ep${epoch_done}_${timestamp}.log"

    echo "[$(date '+%F %T')] new checkpoint for ${tag} (epoch $epoch_done) → running eval"

    # 모델별 eval 명령어 다름
    if [[ "$tag" == decomp* ]]; then
      MODEL_FLAGS="--model_type decomp --decomp_kernels 49,25,7"
    else
      MODEL_FLAGS=""
    fi
    # 5-trunk variants need highfreq2_nf=512
    if [[ "$tag" == *_5trunk_* ]]; then
      MODEL_FLAGS="$MODEL_FLAGS --highfreq2_nf 512"
    fi
    # Chebyshev variant
    if [[ "$tag" == *_cheby_* ]]; then
      MODEL_FLAGS="$MODEL_FLAGS --use_cheby 1"
    fi
    # 12-layer variant
    if [[ "$tag" == *_12layer ]]; then
      MODEL_FLAGS="$MODEL_FLAGS --n_layers 12"
    fi

    # MV cache 사용 시 env var 전달
    if [[ "$tag" == *_mv ]]; then
      MV_ENV="LOTSA_MULTIVARIATE=1"
    else
      MV_ENV=""
    fi

    # GPU 0 공유 (학습이랑 같이) — VRAM 80GB 중 9GB 사용 중이라 여유 충분
    env $MV_ENV CUDA_VISIBLE_DEVICES=0 python experiments/exp_v1_varlen_ext.py \
      --scale 9999 --max_seq_len 720 \
      --highfreq_nf 256 \
      $MODEL_FLAGS \
      --tag $tag \
      --eval_only 1 \
      > "$out_log" 2>&1

    # 결과 요약 추출 (OVERALL line)
    if grep -q "OVERALL" "$out_log"; then
      echo "[$(date '+%F %T')] ${tag} ep${epoch_done} eval done →" >> $LOG_DIR/auto_eval.log
      grep -E "Dataset|^ETT|^Weather|^OVERALL" "$out_log" | sed 's/^/  /' >> $LOG_DIR/auto_eval.log
      echo "" >> $LOG_DIR/auto_eval.log
    else
      echo "[$(date '+%F %T')] ${tag} ep${epoch_done} eval FAILED — see $out_log" >> $LOG_DIR/auto_eval.log
    fi

    LAST_MTIME[$tag]=$mt
  done

  sleep 30
done
