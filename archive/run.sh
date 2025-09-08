#!/bin/bash
#SBATCH --job-name=python_job
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00

# Activate mamba env
source ~/.zshrc
mamba activate yolo

# Usage: sbatch run_python.sh your_script.py arg1 arg2 ...
python "$@"

