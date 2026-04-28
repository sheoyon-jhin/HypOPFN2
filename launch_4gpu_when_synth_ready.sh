#!/bin/bash
# Wait for 30M synth cache to finish building, then launch all 4 ablations.
# Run as: nohup bash launch_4gpu_when_synth_ready.sh > log/auto_launch_4gpu.log 2>&1 &

cd /workspace/HypOPFN2
PY=/workspace/timefound/bin/python
SYNTH_FILE=dataset/synth_cache/synth_n30000000_seq2160_w56c0c04d.dat
SYNTH_META=dataset/synth_cache/synth_n30000000_seq2160_w56c0c04d.meta

echo "[$(date)] Waiting for 30M synth cache: $SYNTH_FILE"

# Phase 1: wait for both .dat and .meta to exist (meta is written ONLY at end)
echo "[$(date)] Waiting for $SYNTH_META (written only when build completes)"
while [ ! -f "$SYNTH_META" ]; do
  if [ -f "$SYNTH_FILE" ]; then
    cur_size=$(stat -c %s "$SYNTH_FILE" 2>/dev/null || echo 0)
    echo "[$(date)] dat: $(numfmt --to=iec $cur_size) (still building)"
  else
    echo "[$(date)] still waiting for file..."
  fi
  sleep 120
done
echo "[$(date)] meta detected. Build complete."

echo "[$(date)] === LAUNCHING 4 ABLATIONS ==="
mkdir -p log

# GPU 0: Synth-only + 4-trunk (30M)
echo "[$(date)] GPU 0: hyper4_synth30M_only (4-trunk, lr=1.2e-3)"
CUDA_VISIBLE_DEVICES=0 nohup $PY experiments/exp_v1_varlen_ext.py \
  --scale 1 --max_seq_len 720 --epochs 5 \
  --use_nll 0 --highfreq_nf 256 \
  --windows_per_series 1 --batch_size 1024 --amp 1 --lr 1.2e-3 \
  --synth_n 30000000 \
  --tag hyper4_synth30M_only \
  > log/hyper4_synth30M_only.log 2>&1 < /dev/null & disown
echo "  PID: $!"
sleep 10

# GPU 1: LOTSA10% + synth30M + 4-trunk
echo "[$(date)] GPU 1: hyper4_uv10pct_synth30M (4-trunk)"
CUDA_VISIBLE_DEVICES=1 nohup $PY experiments/exp_v1_varlen_ext.py \
  --scale 9999 --max_seq_len 720 --epochs 5 \
  --use_nll 0 --highfreq_nf 256 \
  --windows_per_series 1000 --batch_size 1024 --amp 1 --lr 1.2e-3 \
  --synth_n 30000000 \
  --tag hyper4_uv10pct_synth30M \
  > log/hyper4_uv10pct_synth30M.log 2>&1 < /dev/null & disown
echo "  PID: $!"
sleep 10

# GPU 2: Synth-only + 5-trunk (30M)
echo "[$(date)] GPU 2: hyper4_5trunk_synth30M_only (5-trunk + highfreq2_nf=512)"
CUDA_VISIBLE_DEVICES=2 nohup $PY experiments/exp_v1_varlen_ext.py \
  --scale 1 --max_seq_len 720 --epochs 5 \
  --use_nll 0 --highfreq_nf 256 --highfreq2_nf 512 \
  --windows_per_series 1 --batch_size 1024 --amp 1 --lr 1.2e-3 \
  --synth_n 30000000 \
  --tag hyper4_5trunk_synth30M_only \
  > log/hyper4_5trunk_synth30M_only.log 2>&1 < /dev/null & disown
echo "  PID: $!"
sleep 10

# GPU 3: LOTSA10% + synth30M + 5-trunk
echo "[$(date)] GPU 3: hyper4_uv10pct_5trunk_synth30M"
CUDA_VISIBLE_DEVICES=3 nohup $PY experiments/exp_v1_varlen_ext.py \
  --scale 9999 --max_seq_len 720 --epochs 5 \
  --use_nll 0 --highfreq_nf 256 --highfreq2_nf 512 \
  --windows_per_series 1000 --batch_size 1024 --amp 1 --lr 1.2e-3 \
  --synth_n 30000000 \
  --tag hyper4_uv10pct_5trunk_synth30M \
  > log/hyper4_uv10pct_5trunk_synth30M.log 2>&1 < /dev/null & disown
echo "  PID: $!"

echo "[$(date)] All 4 launched. Watching first iters..."
sleep 60
for tag in hyper4_synth30M_only hyper4_uv10pct_synth30M hyper4_5trunk_synth30M_only hyper4_uv10pct_5trunk_synth30M; do
  echo "--- $tag ---"
  tail -3 log/${tag}.log 2>/dev/null
done
