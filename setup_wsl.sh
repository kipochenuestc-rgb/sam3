#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python3 restore_weights.py
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip 'setuptools<81'
.venv/bin/python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
.venv/bin/python -m pip install -e . einops pycocotools decord pillow psutil
.venv/bin/python -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
