#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$ROOT_DIR/.venv"
PYTHON_BIN="$VENV_DIR/bin/python"

echo "[1/6] Installing system packages"
sudo apt-get update
sudo apt-get install -y \
  git \
  python3 \
  python3-pip \
  python3-venv \
  libatlas-base-dev \
  libopenblas-dev \
  libgl1 \
  libglib2.0-0 \
  libjpeg-dev \
  libopenjp2-7 \
  libtiff6 \
  zlib1g-dev

echo "[2/6] Creating virtual environment"
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

echo "[3/6] Upgrading pip tooling"
python -m pip install --upgrade pip setuptools wheel

echo "[4/6] Installing app dependencies"
python -m pip install \
  fastapi \
  "uvicorn[standard]" \
  jinja2 \
  python-multipart \
  pillow \
  numpy \
  opencv-python-headless

echo "[5/6] Installing PyTorch and Ultralytics"
if ! python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu; then
  echo "Torch installation from the CPU wheel index failed."
  echo "Install a Raspberry Pi compatible torch/torchvision build manually, then rerun this script."
  exit 1
fi

python -m pip install ultralytics

echo "[6/6] Verifying imports"
python - <<'PY'
import cv2
import jinja2
import multipart
import numpy
import torch
import ultralytics
import uvicorn
from PIL import Image
from fastapi import FastAPI

print("Environment check passed.")
print(f"torch={torch.__version__}")
print(f"ultralytics={ultralytics.__version__}")
PY

echo
echo "Setup complete."
echo "If you have trained weights, copy them to:"
echo "  $ROOT_DIR/rtdetr/runs/solder_defects_rtdetr/weights/best.pt"
echo
echo "Launch on Raspberry Pi with:"
echo "  cd $ROOT_DIR"
echo "  source .venv/bin/activate"
echo "  export RTDETR_HOST=0.0.0.0"
echo "  export RTDETR_OPEN_BROWSER=0"
echo "  python deployed_app/launch_rtdetr_ai.py"