#!/bin/bash
# Minimal venv + deps setup. Run this AFTER `git clone` on the new server.
#
# Usage:
#   cd HypOPFN2
#   bash setup_venv.sh

set -e

# 1) Verify we're inside the repo
if [ ! -f "requirements.txt" ]; then
  echo "ERROR: run from inside HypOPFN2/ (requirements.txt not found)"
  exit 1
fi

VENV_DIR=${VENV_DIR:-/workspace/timefound}
echo "[$(date)] Creating venv at $VENV_DIR"

# 2) Check Python
PYTHON=${PYTHON:-python3}
$PYTHON --version || { echo "python3 not found"; exit 1; }

# 3) Create venv
if [ ! -d "$VENV_DIR" ]; then
  $PYTHON -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

# 4) Upgrade pip/wheel
pip install --upgrade pip wheel setuptools

# 5) PyTorch 2.5.1 + CUDA 12.1 (matches current server)
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu121

# 6) Project deps
pip install -r requirements.txt

# 7) Extra deps used by data pipeline + FT eval
pip install datasets  # HF arrow loader (for some scripts)

# 8) Quick sanity check
python - <<'PY'
import torch
print('torch :', torch.__version__)
print('cuda  :', torch.cuda.is_available())
print('n_gpu :', torch.cuda.device_count())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
import pyarrow, numpy, pandas, sklearn
print('pyarrow:', pyarrow.__version__)
print('numpy :', numpy.__version__)
print('pandas:', pandas.__version__)
print('sklearn:', sklearn.__version__)
PY

echo ""
echo "======================================================================"
echo " venv READY at $VENV_DIR"
echo " Activate later with:  source $VENV_DIR/bin/activate"
echo "======================================================================"
