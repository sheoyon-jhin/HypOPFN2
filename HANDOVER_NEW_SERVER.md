# HypOPFN — New Server Handover Guide

## Target Server
- 8x A100 SXM4 80GB
- 2TB RAM, 25TB SSD
- Vast.ai, ~$9.19/hr

---

## Step 1: Transfer Files

On current server:
```bash
cd /workspace
tar czf HypOPFN2_transfer.tar.gz \
    HypOPFN2/experiments/ \
    HypOPFN2/data/ \
    HypOPFN2/data_provider/ \
    HypOPFN2/checkpoints/v1_varlen_nll_failmode.pth \
    HypOPFN2/checkpoints/v1_varlen_mse_allfixed.pth \
    HypOPFN2/checkpoints/v1_varlen_mse_4trunk.pth \
    HypOPFN2/results/ \
    HypOPFN2/dataset/ETT-small/ \
    HypOPFN2/dataset/weather/ \
    HypOPFN2/dataset/synth_cache/ \
    HypOPFN2/*.md \
    HypOPFN2/requirements.txt
# ~5-10GB
```

Transfer:
```bash
scp HypOPFN2_transfer.tar.gz user@new_server:/workspace/
```

On new server:
```bash
cd /workspace && tar xzf HypOPFN2_transfer.tar.gz
```

---

## Step 2: Environment Setup

```bash
cd /workspace/HypOPFN2
python -m venv /workspace/timefound
source /workspace/timefound/bin/activate
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install gluonts huggingface_hub pyarrow
```

---

## Step 3: Download Full LOTSA (231B, 174 datasets)

```bash
source /workspace/timefound/bin/activate
python data/download_lotsa.py
# Takes several hours. Downloads to dataset/lotsa/
# Excludes eval benchmarks (ETT, Weather, M4, M1, M3)
```

Verify:
```bash
ls dataset/lotsa/ | wc -l  # Should be ~159 (174 - 15 excluded)
du -sh dataset/lotsa/       # Should be ~500-600GB
```

---

## Step 4: Generate Synth Cache (if not transferred)

```bash
python -c "
import sys; sys.path.insert(0, '.')
from experiments.exp_lotsa_scaling import SyntheticGapFiller
SyntheticGapFiller(n_samples=500000, seq_len=2160)
"
# Creates dataset/synth_cache/ (~4.3GB), takes ~25 min first time
```

---

## Step 5: Run Experiments

### Priority 1: Decomp + 4 HyperTrunk + Full LOTSA
```bash
CUDA_VISIBLE_DEVICES=0 python experiments/exp_v1_varlen_ext.py \
    --scale 9999 --max_seq_len 720 --epochs 40 --synth_n 500000 \
    --use_nll 0 --highfreq_nf 256 \
    --model_type decomp --decomp_kernels 49,25,7 \
    --tag decomp_hyper4_full231B
```

### Priority 2: AllFixed + hf256 + Full LOTSA
```bash
CUDA_VISIBLE_DEVICES=1 python experiments/exp_v1_varlen_ext.py \
    --scale 9999 --max_seq_len 720 --epochs 40 --synth_n 500000 \
    --use_nll 0 --all_fixed 1 --highfreq_nf 256 \
    --tag allfixed_hf256_full231B
```

### Priority 3: Ablations on remaining GPUs
```bash
# GPU 2: Decomp 3-way (no highfreq)
--model_type decomp --decomp_kernels 49,25 --highfreq_nf 0

# GPU 3: Decomp + spectral
--model_type decomp --decomp_kernels 49,25,7 --highfreq_nf 256 --spectral_weight 10

# GPU 4-7: further variations
```

---

## Key Files

| File | Purpose |
|------|---------|
| experiments/exp_v1_varlen_ext.py | Main training code (all models) |
| experiments/exp_lotsa_scaling.py | Base components (trunks, data, synth) |
| experiments/eval_m4_varlen.py | M4 evaluation |
| experiments/eval_imputation_varlen.py | Imputation evaluation |
| experiments/viz_*.py | Visualization scripts |
| data/download_lotsa.py | LOTSA download |
| EXPERIMENTS_LOG.md | All trial-and-error |
| MODEL_ARCHITECTURE.md | Architecture reference |

---

## Critical Notes

1. **NLL loss UNSTABLE** — use `--use_nll 0` (MSE only) for all experiments
2. **Synth cache**: auto-generated on first run, reused after
3. **LOTSA loading**: scale=9999 with 174 datasets = 30-60 min load time
4. **Best checkpoint**: `v1_varlen_mse_allfixed.pth` (MSE 0.398, 19.3M)
5. **model_type=decomp**: input decomposition model
6. **all_fixed=1**: FixedTrunk (stable, no HyperNet collapse)

---

## Current Experiments (old server, completing in 1-3 days)

| GPU | Experiment | Status |
|-----|-----------|--------|
| 0 | allfixed+hf256 scale=2000 | training |
| 1 | decomp_hyper4 scale=2000 | training |
| 2 | allfixed+hf256 scale=9999 | training |
| 3 | decomp_hyper4 scale=9999 | training |

Transfer checkpoints after completion.