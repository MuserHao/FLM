#!/bin/bash
# Phase 0: 2×2 encoder design — train all conditions.
# Run from ~/ELF_pytorch/
#
# 2×2 design:
#                   Pretrained          Not pretrained
#   Contextual    | pretrained (A)    | (Phase 2 scratch, not here)
#   Non-contextual| t5_embed   (C)    | random       (D)
#
# Usage:
#   bash run_phase0.sh pretrained   # Condition A: pretrained contextual T5
#   bash run_phase0.sh t5_embed     # Condition C: pretrained non-contextual (token embeddings)
#   bash run_phase0.sh random       # Condition D: random non-contextual
#   bash run_phase0.sh all          # run all three sequentially

set -e
CONDA_PYTHON=~/miniconda3/envs/elf/bin/python
N_GPUS=2
MAX_STEPS=20000

MODE=${1:-"pretrained"}

# Each run gets a unique port so parallel tmux windows don't collide.
declare -A PORTS=(
    ["phase0_pretrained"]=29500
    ["phase0_random"]=29501
    ["phase0_t5_embed"]=29502
)

run_exp() {
    local cfg=$1
    local name=$2
    local port=${PORTS[$name]:-29500}
    echo "========================================"
    echo "Starting Phase 0: $name"
    echo "Config: $cfg"
    echo "GPUs: $N_GPUS  Port: $port"
    echo "Max steps: $MAX_STEPS (~$(($MAX_STEPS * 128 / 1000000))M tokens)"
    echo "========================================"
    cd ~/ELF_pytorch/src
    torchrun --nproc_per_node=$N_GPUS \
        --master_port=$port \
        train.py \
        --config "configs/training_configs/$cfg" \
        --config_override "use_wandb=false" \
        2>&1 | tee "../logs/${name}_$(date +%Y%m%d_%H%M).log"
}

mkdir -p ~/ELF_pytorch/logs

if [[ "$MODE" == "pretrained" || "$MODE" == "all" ]]; then
    run_exp "train_owt_ELF-B_pretrained.yml" "phase0_pretrained"
fi

if [[ "$MODE" == "t5_embed" || "$MODE" == "all" ]]; then
    run_exp "train_owt_ELF-B_t5_embed.yml" "phase0_t5_embed"
fi

if [[ "$MODE" == "random" || "$MODE" == "all" ]]; then
    run_exp "train_owt_ELF-B_random.yml" "phase0_random"
fi

echo "Done. Checkpoints in ~/ELF_pytorch/src/outputs/"
