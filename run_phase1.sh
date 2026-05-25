#!/bin/bash
# Phase 1: Geometric analysis — 2×2 encoder design.
#
# Conditions:
#   A) t5              — pretrained T5-small contextual
#   C) t5_token_embed  — pretrained T5 token embeddings, non-contextual
#   D) random_embedding— frozen random lookup, non-contextual
#
# Comparison axes:
#   A vs C → contextuality effect (pretraining held constant)
#   C vs D → pretraining geometry effect (non-contextual held constant)
#   A vs D → total geometric structure (our main upper-bound comparison)
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

echo "=== Phase 1 Condition A: Pretrained T5 (contextual) ==="
$CONDA_PYTHON phase1_geometry.py \
    --encoder t5 \
    --n_samples $N_SAMPLES \
    --device $DEVICE \
    --output ../results/phase1_t5.json

echo ""
echo "=== Phase 1 Condition C: T5 Token Embeddings (non-contextual, pretrained) ==="
$CONDA_PYTHON phase1_geometry.py \
    --encoder t5_token_embed \
    --n_samples $N_SAMPLES \
    --device $DEVICE \
    --output ../results/phase1_t5_embed.json

echo ""
echo "=== Phase 1 Condition D: Random Embeddings (non-contextual, random) ==="
$CONDA_PYTHON phase1_geometry.py \
    --encoder random_embedding \
    --n_samples $N_SAMPLES \
    --device $DEVICE \
    --output ../results/phase1_random.json

echo ""
echo "=== 2×2 Comparison ==="
$CONDA_PYTHON - << 'PYEOF'
import json, os

def load(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)

pre  = load("../results/phase1_t5.json")
emb  = load("../results/phase1_t5_embed.json")
ran  = load("../results/phase1_random.json")

T_VALS = ["0.1","0.2","0.3","0.4","0.5","0.6","0.7","0.8","0.9"]

# ── Off-manifold distance ──────────────────────────────────────────────
print("\n── Off-manifold distance δ(t) ──")
print(f"{'t':>5}  {'A:T5(ctx)':>10}  {'C:T5(emb)':>10}  {'D:random':>10}"
      f"  {'A/C (ctx)':>10}  {'C/D (pre)':>10}  {'A/D (tot)':>10}")
print("-" * 78)
for t in T_VALS:
    da = pre["delta_t"][t]["mean"]  if pre  else float('nan')
    dc = emb["delta_t"][t]["mean"]  if emb  else float('nan')
    dd = ran["delta_t"][t]["mean"]  if ran  else float('nan')
    r_ctx = dc/da if da>0 else float('inf')
    r_pre = dd/dc if dc>0 else float('inf')
    r_tot = dd/da if da>0 else float('inf')
    print(f"{float(t):>5.1f}  {da:>10.4f}  {dc:>10.4f}  {dd:>10.4f}"
          f"  {r_ctx:>10.3f}x  {r_pre:>10.3f}x  {r_tot:>10.3f}x")

# ── Conditional variance ───────────────────────────────────────────────
print("\n── Conditional variance Var[x₀|zₜ] ──")
print(f"{'t':>5}  {'A:T5(ctx)':>10}  {'C:T5(emb)':>10}  {'D:random':>10}"
      f"  {'C/A (ctx↓)':>11}  {'D/C (pre↓)':>11}  {'D/A (tot↓)':>11}")
print("-" * 84)
for t in T_VALS:
    va = pre["cond_variance"][t]["mean"]  if pre  else float('nan')
    vc = emb["cond_variance"][t]["mean"]  if emb  else float('nan')
    vd = ran["cond_variance"][t]["mean"]  if ran  else float('nan')
    r_ctx = vc/va if va>0 else float('inf')   # how much worse is non-ctx?
    r_pre = vd/vc if vc>0 else float('inf')   # how much worse is random?
    r_tot = vd/va if va>0 else float('inf')
    print(f"{float(t):>5.1f}  {va:>10.4f}  {vc:>10.4f}  {vd:>10.4f}"
          f"  {r_ctx:>11.2f}x  {r_pre:>11.2f}x  {r_tot:>11.2f}x")

# ── Curvature ──────────────────────────────────────────────────────────
print("\n── Manifold curvature κ ──")
for label, res in [("A: T5 contextual", pre),
                   ("C: T5 token embed", emb),
                   ("D: random        ", ran)]:
    if res:
        k = res["curvature"]
        print(f"  {label}  κ={k['kappa_mean']:.4f} ± {k['kappa_std']:.4f}")
PYEOF
