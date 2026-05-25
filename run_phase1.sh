#!/bin/bash
# Phase 1: Geometric analysis — pretrained vs random encoder.
# Runs both experiments and prints a side-by-side comparison.
#
# Usage:
#   bash run_phase1.sh [--n_samples 1000] [--device cuda]

set -e
CONDA_PYTHON=~/miniconda3/envs/elf/bin/python
N_SAMPLES=1000
DEVICE="cuda"

while [[ $# -gt 0 ]]; do
    case $1 in
        --n_samples) N_SAMPLES="$2"; shift 2;;
        --device)    DEVICE="$2"; shift 2;;
        *) echo "Unknown arg: $1"; exit 1;;
    esac
done

mkdir -p ~/ELF_pytorch/results

cd ~/ELF_pytorch/src

echo "=== Phase 1: Pretrained T5 Encoder ==="
$CONDA_PYTHON phase1_geometry.py \
    --encoder t5 \
    --n_samples $N_SAMPLES \
    --device $DEVICE \
    --output ../results/phase1_t5.json

echo ""
echo "=== Phase 1: Random Embedding Encoder ==="
$CONDA_PYTHON phase1_geometry.py \
    --encoder random_embedding \
    --n_samples $N_SAMPLES \
    --device $DEVICE \
    --output ../results/phase1_random.json

echo ""
echo "=== Comparison ==="
$CONDA_PYTHON - << 'PYEOF'
import json

with open("../results/phase1_t5.json") as f:
    pre = json.load(f)
with open("../results/phase1_random.json") as f:
    ran = json.load(f)

print(f"{'t':>5}  {'δ_pretrained':>14}  {'δ_random':>10}  {'ratio':>8}  {'Var_pre':>10}  {'Var_ran':>10}")
print("-" * 70)
for t in ["0.1","0.2","0.3","0.4","0.5","0.6","0.7","0.8","0.9"]:
    d_pre = pre["delta_t"][t]["mean"]
    d_ran = ran["delta_t"][t]["mean"]
    v_pre = pre["cond_variance"][t]["mean"]
    v_ran = ran["cond_variance"][t]["mean"]
    ratio = d_ran / d_pre if d_pre > 0 else float('inf')
    print(f"{float(t):>5.1f}  {d_pre:>14.4f}  {d_ran:>10.4f}  {ratio:>8.3f}x  {v_pre:>10.4f}  {v_ran:>10.4f}")

print()
print(f"Curvature κ  pretrained={pre['curvature']['kappa_mean']:.4f}  random={ran['curvature']['kappa_mean']:.4f}")
PYEOF
