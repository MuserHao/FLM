#!/usr/bin/env python
"""Phase 1: Geometric analysis of pretrained vs random T5 embedding space.

Measures three quantities from the research plan:
  1. Off-manifold distance delta(t): distance of interpolated z_t to nearest
     token embedding, as a function of t in {0.1, 0.2, ..., 0.9}.
  2. Conditional variance proxy Var[x | z_t]: k-NN variance of token embeddings
     near each z_t, approximating the local uncertainty in x given z_t.
  3. Manifold curvature proxy kappa: ratio of discrete geodesic path length to
     Euclidean distance between pairs of token embeddings.

Usage:
  # Run on server (GPU):
  python phase1_geometry.py --encoder t5               --n_samples 1000 --output results/phase1_t5.json
  python phase1_geometry.py --encoder random_embedding --n_samples 1000 --output results/phase1_random.json

  # Run locally (CPU, fast smoke test):
  python phase1_geometry.py --encoder t5 --n_samples 100 --device cpu --output results/phase1_t5.json
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from modules.t5_encoder import get_encoder


# ── helpers ─────────────────────────────────────────────────────────────────

def get_token_embeddings(encoder, device) -> torch.Tensor:
    """Extract the token embedding matrix E ∈ R^{V × d} from the encoder."""
    # T5Encoder: embeddings are in model.shared
    # RandomEmbeddingEncoder: embeddings are in self.embedding
    if hasattr(encoder, "model") and hasattr(encoder.model, "shared"):
        emb = encoder.model.shared.weight.detach().to(device)
    elif hasattr(encoder, "embedding"):
        emb = encoder.embedding.weight.detach().to(device)
    else:
        raise AttributeError(f"Cannot extract embedding matrix from {type(encoder)}")
    return emb


def nearest_neighbor_dist(z: torch.Tensor, E: torch.Tensor, batch_size: int = 512) -> torch.Tensor:
    """For each vector in z (N, d), find L2 distance to nearest row in E (V, d).

    Returns shape (N,).  Chunked to avoid OOM on large V.
    """
    N, d = z.shape
    V = E.shape[0]
    dists = torch.full((N,), float("inf"), device=z.device)
    for start in range(0, V, batch_size):
        chunk = E[start : start + batch_size]          # (chunk, d)
        # (N, chunk)
        diff = z.unsqueeze(1) - chunk.unsqueeze(0)     # broadcast
        d2 = diff.pow(2).sum(-1)                       # (N, chunk)
        min_d2, _ = d2.min(dim=1)                      # (N,)
        dists = torch.minimum(dists, min_d2.sqrt())
    return dists


def knn_embeddings(z: torch.Tensor, E: torch.Tensor, k: int = 10, batch_size: int = 512) -> torch.Tensor:
    """Return k nearest token embeddings for each z ∈ (N, d).

    Returns shape (N, k, d).
    """
    N, d = z.shape
    V = E.shape[0]
    # Compute full distance matrix in chunks over V
    all_d2 = torch.zeros(N, V, device=z.device)
    for start in range(0, V, batch_size):
        chunk = E[start : start + batch_size]
        diff = z.unsqueeze(1) - chunk.unsqueeze(0)
        all_d2[:, start : start + batch_size] = diff.pow(2).sum(-1)
    topk_idx = all_d2.topk(k, dim=1, largest=False).indices  # (N, k)
    return E[topk_idx]  # (N, k, d)


# ── Experiment 1.1: Off-manifold distance delta(t) ──────────────────────────

def measure_delta_t(
    x: torch.Tensor,         # (N, d) clean embeddings
    E: torch.Tensor,          # (V, d) token embedding matrix
    t_values: list,
    noise_scale: float = 2.0,
    n_noise_samples: int = 5,
) -> dict:
    """Measure E[dist(z_t, manifold)] for each t.

    Saved per t:
      mean / std   — scalar summaries for quick comparison
      per_noise_means — list of length n_noise_samples; each entry is the mean
                        distance across N samples for one noise draw.  Allows
                        bootstrap CI computation in post-processing.
    """
    results = {}
    N, d = x.shape
    for t in t_values:
        per_noise_means = []
        flat_dists = []
        for _ in range(n_noise_samples):
            eps = torch.randn_like(x)
            z_t = t * x + (1 - t) * eps * noise_scale
            d_t = nearest_neighbor_dist(z_t, E)   # (N,)
            per_noise_means.append(float(d_t.mean()))
            flat_dists.append(d_t.cpu())
        stacked = torch.stack(flat_dists, dim=0)    # (n_noise, N)
        results[t] = {
            "mean": float(stacked.mean()),
            "std":  float(stacked.std()),
            # per_noise_means: one scalar per noise draw → use for bootstrap CI
            "per_noise_means": per_noise_means,
        }
        print(f"  δ(t={t:.1f})  mean={results[t]['mean']:.4f}  std={results[t]['std']:.4f}")
    return results


# ── Experiment 1.2: Conditional variance proxy Var[x | z_t] ─────────────────

def measure_cond_variance(
    x: torch.Tensor,
    E: torch.Tensor,
    t_values: list,
    k: int = 10,
    noise_scale: float = 2.0,
    n_noise_samples: int = 3,
) -> dict:
    """k-NN variance of token embeddings near z_t as proxy for Var[x | z_t].

    Saved per t:
      mean / std        — scalar summaries
      per_noise_means   — list of n_noise_samples means for bootstrap CI
      per_sample_means  — list of N per-sample variances (from the last noise
                          draw); allows per-token distribution analysis.
    """
    results = {}
    for t in t_values:
        per_noise_means = []
        last_per_sample = None
        for _ in range(n_noise_samples):
            eps = torch.randn_like(x)
            z_t = t * x + (1 - t) * eps * noise_scale
            neighbors = knn_embeddings(z_t, E, k=k)           # (N, k, d)
            var_per_sample = neighbors.var(dim=1).mean(dim=-1) # (N,)
            per_noise_means.append(float(var_per_sample.mean()))
            last_per_sample = var_per_sample.cpu()
        all_means = torch.tensor(per_noise_means)
        results[t] = {
            "mean": float(all_means.mean()),
            "std":  float(all_means.std()),
            "per_noise_means": per_noise_means,
            # Distribution across individual token positions (last noise draw)
            "per_sample_mean": float(last_per_sample.mean()),
            "per_sample_std":  float(last_per_sample.std()),
            "per_sample_p25":  float(last_per_sample.quantile(0.25)),
            "per_sample_p75":  float(last_per_sample.quantile(0.75)),
        }
        print(f"  Var[x|z_t={t:.1f}]  mean={results[t]['mean']:.4f}  std={results[t]['std']:.4f}")
    return results


# ── Experiment 1.3: Manifold curvature proxy kappa ──────────────────────────

def measure_curvature(
    E: torch.Tensor,          # (V, d)
    n_pairs: int = 500,
    n_midpoints: int = 5,
    seed: int = 0,
) -> dict:
    """kappa = E[d_geo / d_euc - 1] over random pairs of token embeddings.

    d_geo is approximated by projecting midpoints along the linear path to
    the nearest token embedding and summing segment lengths.
    """
    torch.manual_seed(seed)
    V = E.shape[0]
    idx = torch.randint(0, V, (n_pairs, 2))
    # avoid self-pairs
    same = idx[:, 0] == idx[:, 1]
    idx[same, 1] = (idx[same, 1] + 1) % V

    x1 = E[idx[:, 0]]  # (P, d)
    x2 = E[idx[:, 1]]  # (P, d)

    d_euc = (x2 - x1).norm(dim=-1)  # (P,)

    # Geodesic approximation: sample midpoints, project each to nearest token
    ts = torch.linspace(0, 1, n_midpoints + 2)[1:-1]  # interior only
    projected = []
    endpoints = [x1]
    for t_val in ts:
        mid = (1 - t_val) * x1 + t_val * x2   # (P, d)
        # nearest token embedding for each midpoint
        proj_idx = torch.cdist(mid, E).argmin(dim=-1)  # (P,)
        projected.append(E[proj_idx])
    endpoints.append(x2)

    # Build path: x1 -> proj(t1) -> ... -> proj(tn) -> x2
    waypoints = [x1] + projected + [x2]
    d_geo = sum(
        (waypoints[i + 1] - waypoints[i]).norm(dim=-1)
        for i in range(len(waypoints) - 1)
    )

    ratio = d_geo / d_euc.clamp(min=1e-8)
    kappa_vals = (ratio - 1).cpu()
    kappa = float(kappa_vals.mean())
    kappa_std = float(kappa_vals.std())
    print(f"  κ = {kappa:.4f} ± {kappa_std:.4f}")
    return {
        "kappa_mean": kappa,
        "kappa_std": kappa_std,
        "ratio_mean": float(ratio.mean()),
        "ratio_p25": float(kappa_vals.quantile(0.25)),
        "ratio_p75": float(kappa_vals.quantile(0.75)),
        # Per-pair ratios (for histogram / distribution plot)
        "per_pair_ratios": kappa_vals.tolist(),
    }


# ── Data loading ─────────────────────────────────────────────────────────────

def load_sample_sequences(n_samples: int, max_length: int = 64, seed: int = 42) -> torch.Tensor:
    """Load n_samples token sequences from OWT (HuggingFace, pre-tokenised)."""
    from datasets import load_dataset
    print(f"Loading {n_samples} sequences from OWT...")
    ds = load_dataset("embedded-language-flows/openwebtext-t5", split="train", streaming=True)
    rng = np.random.default_rng(seed)
    seqs = []
    for ex in ds:
        ids = ex["input_ids"][:max_length]
        if len(ids) < 4:
            continue
        seqs.append(ids)
        if len(seqs) >= n_samples:
            break
    # pad to max_length
    padded = np.zeros((len(seqs), max_length), dtype=np.int64)
    masks = np.zeros((len(seqs), max_length), dtype=np.int64)
    for i, s in enumerate(seqs):
        padded[i, : len(s)] = s
        masks[i, : len(s)] = 1
    return torch.tensor(padded), torch.tensor(masks)


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--encoder", choices=["t5", "random_embedding"], default="t5",
                   help="'t5' = pretrained T5-small; 'random_embedding' = frozen random lookup")
    p.add_argument("--n_samples", type=int, default=500)
    p.add_argument("--max_seq_len", type=int, default=64)
    p.add_argument("--k_neighbors", type=int, default=10)
    p.add_argument("--n_pairs_curvature", type=int, default=500)
    p.add_argument("--noise_scale", type=float, default=2.0,
                   help="ELF denoiser_noise_scale (default 2.0)")
    p.add_argument("--n_noise_samples", type=int, default=5,
                   help="Number of noise realisations per t value")
    p.add_argument("--device", type=str, default=None,
                   help="cuda / cpu (auto-detected if not set)")
    p.add_argument("--output", type=str, default="results/phase1.json")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}  |  encoder: {args.encoder}  |  n_samples: {args.n_samples}")

    # Load encoder
    enc_cfg, encoder = get_encoder("t5-small", torch.float32, encoder_type=args.encoder)
    encoder = encoder.to(device).eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    print(f"Encoder loaded. d_model={enc_cfg.d_model}")

    # Token embedding matrix
    E = get_token_embeddings(encoder, device)
    print(f"Token embedding matrix: {E.shape}  (V={E.shape[0]}, d={E.shape[1]})")

    # Load sequences and encode
    input_ids, attention_mask = load_sample_sequences(
        args.n_samples, max_length=args.max_seq_len, seed=args.seed
    )
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    print(f"Encoding {args.n_samples} sequences (seq_len={args.max_seq_len})...")
    with torch.no_grad():
        x_all = encoder(input_ids, attention_mask=attention_mask.float())
        # x_all: (N, S, d) — flatten to (N*S, d) for valid tokens only
    valid = attention_mask.bool().reshape(-1)
    x_flat = x_all.reshape(-1, enc_cfg.d_model)[valid]  # (M, d)
    print(f"Valid token embeddings: {x_flat.shape[0]}")

    # Normalise by latent_std=0.2 (matching ELF training)
    x_flat = x_flat / 0.2

    t_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    results = {
        # ── Experiment metadata (for reproducibility) ──────────────────────
        "encoder": args.encoder,
        "n_sequences": args.n_samples,
        "n_token_embeddings": x_flat.shape[0],
        "max_seq_len": args.max_seq_len,
        "k_neighbors": args.k_neighbors,
        "noise_scale": args.noise_scale,
        "n_noise_samples": args.n_noise_samples,
        "n_pairs_curvature": args.n_pairs_curvature,
        "latent_std": 0.2,
        "t_values": t_values,
        "seed": args.seed,
        "device": str(device),
    }

    # 1.1 Off-manifold distance
    print("\n=== Experiment 1.1: Off-manifold distance δ(t) ===")
    t0 = time.time()
    results["delta_t"] = measure_delta_t(
        x_flat, E / 0.2,   # normalise E too
        t_values=t_values,
        noise_scale=args.noise_scale,
        n_noise_samples=args.n_noise_samples,
    )
    print(f"  Done in {time.time()-t0:.1f}s")

    # 1.2 Conditional variance proxy
    print("\n=== Experiment 1.2: Conditional variance proxy ===")
    t0 = time.time()
    results["cond_variance"] = measure_cond_variance(
        x_flat, E / 0.2,
        t_values=t_values,
        k=args.k_neighbors,
        noise_scale=args.noise_scale,
        n_noise_samples=args.n_noise_samples,
    )
    print(f"  Done in {time.time()-t0:.1f}s")

    # 1.3 Manifold curvature
    print("\n=== Experiment 1.3: Manifold curvature proxy κ ===")
    t0 = time.time()
    results["curvature"] = measure_curvature(
        E / 0.2,
        n_pairs=args.n_pairs_curvature,
        n_midpoints=5,
        seed=args.seed,
    )
    print(f"  Done in {time.time()-t0:.1f}s")

    # Save
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
