#!/bin/bash
set -e

source ~/.zshrc

# Activate the mamba environment
mamba activate yolo

# Install pinned requirements with CUDA 12.1 support
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121

echo "✅ YOLO pose environment setup complete!"

