#!/bin/bash
"""
새 서버 (A100/H100) 세팅 스크립트
1. 패키지 설치
2. LOTSA 다운로드
3. 실험 실행

Usage: bash setup_server.sh
"""

set -e
echo "=== HypOPFN Server Setup ==="

# 1. 패키지 설치
echo "[1/3] Installing packages..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install pyarrow scikit-learn matplotlib huggingface_hub gluonts pandas

# 2. LOTSA 다운로드 (eval benchmark 제외)
echo "[2/3] Downloading LOTSA (excluding eval benchmarks)..."
python data/download_lotsa.py

# 3. 확인
echo "[3/3] Verifying..."
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
"

echo ""
echo "=== Setup Complete ==="
echo "Run experiments with: bash run_lotsa_experiments.sh"
