#!/bin/bash
# Phase 0: Train pretrained and random encoder baselines.
# Run from ~/ELF_pytorch/
#
# Usage:
#   bash run_phase0.sh pretrained   # run pretrained encoder
#   bash run_phase0.sh random       # run random encoder
#   bash run_phase0.sh both         # run both sequentially (takes 2x time)

set -e
CONDA_PYTHON=~/miniconda3/envs/elf/bin/python
N_GPUS=2
MAX_STEPS=20000

MODE=${1:-"pretrained"}

run_exp() {
    local cfg=$1
    local name=$2
    echo "========================================"
    echo "Starting Phase 0: $name"
    echo "Config: $cfg"
    echo "GPUs: $N_GPUS"
    echo "Max steps: $MAX_STEPS (manual stop after ~$(($MAX_STEPS * 128 / 1000000))M tokens)"
    echo "========================================"
    cd ~/ELF_pytorch/src
    torchrun --nproc_per_node=$N_GPUS \
        train.py \
        --config "configs/training_configs/$cfg" \
        --config_override "use_wandb=false" \
        2>&1 | tee "../logs/${name}_$(date +%Y%m%d_%H%M).log"
}

mkdir -p ~/ELF_pytorch/logs

if [[ "$MODE" == "pretrained" || "$MODE" == "both" ]]; then
    run_exp "train_owt_ELF-B_pretrained.yml" "phase0_pretrained"
fi

if [[ "$MODE" == "random" || "$MODE" == "both" ]]; then
    run_exp "train_owt_ELF-B_random.yml" "phase0_random"
fi

echo "Done. Checkpoints in ~/ELF_pytorch/src/outputs/"
