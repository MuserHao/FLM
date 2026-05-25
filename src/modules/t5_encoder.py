#!/usr/bin/env python
"""Frozen T5 text embedder, wrapping `transformers.T5EncoderModel`.

Two encoder modes are supported (controlled by ``encoder_type`` in config):
  - ``"t5"`` (default): frozen pretrained T5 contextual encoder.
  - ``"random_embedding"``: frozen random static embedding lookup table with
    the same vocabulary size and dimension as T5-small.  Each token id maps
    to a fixed random vector; there is no transformer and no contextualisation.
    This is the clean Phase-0 baseline: it tests whether pretrained *geometric
    structure* drives ELF's data efficiency, not whether the encoder was
    "trained" as a feature extractor.
"""

from typing import Any, Optional

import torch
import torch.nn as nn

from utils.logging_utils import log_for_0


class T5EncoderConfig:
    """Configuration class for T5Encoder."""

    def __init__(self, model_name: str, dtype: Any):
        self.model_name = model_name
        self.dtype = dtype
        self.vocab_size: int = 0
        self.d_model: int = 0
        self.d_kv: int = 0
        self.d_ff: int = 0
        self.num_layers: int = 0
        self.num_heads: int = 0
        self.is_gated_act: bool = False

    @classmethod
    def from_pretrained(cls, model_name: str, dtype: Any = torch.float32) -> "T5EncoderConfig":
        cfg = cls(model_name, dtype)
        defaults = {
            "t5-small": dict(vocab_size=32128, d_model=512, d_kv=64, d_ff=2048,
                             num_layers=6, num_heads=8, is_gated_act=False),
            "t5-base":  dict(vocab_size=32128, d_model=768, d_kv=64, d_ff=3072,
                             num_layers=12, num_heads=12, is_gated_act=False),
            "t5-large": dict(vocab_size=32128, d_model=1024, d_kv=64, d_ff=4096,
                             num_layers=24, num_heads=16, is_gated_act=False),
        }
        if model_name in defaults:
            for k, v in defaults[model_name].items():
                setattr(cfg, k, v)
        return cfg


class T5Encoder(nn.Module):
    """T5 encoder used as a frozen text embedder."""

    def __init__(self, config: T5EncoderConfig, *, pretrained: bool = True):
        super().__init__()
        from transformers import T5EncoderModel, T5Config

        if pretrained:
            self.model = T5EncoderModel.from_pretrained(config.model_name)
        else:
            hf_config = T5Config.from_pretrained(config.model_name)
            self.model = T5EncoderModel(hf_config)

        hf = self.model.config
        config.vocab_size = hf.vocab_size
        config.d_model = hf.d_model
        config.d_kv = hf.d_kv
        config.d_ff = hf.d_ff
        config.num_layers = hf.num_layers
        config.num_heads = hf.num_heads
        config.is_gated_act = bool(getattr(hf, "is_gated_act", False))
        self.config = config

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        deterministic: bool = True,
    ) -> torch.Tensor:
        was_training = self.model.training
        if deterministic:
            self.model.eval()
        try:
            out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        finally:
            if not deterministic and was_training:
                self.model.train()
        return out.last_hidden_state


class RandomEmbeddingEncoder(nn.Module):
    """Frozen random static embedding lookup — Phase-0 control baseline.

    Each token id maps to a fixed random 512-d vector drawn from N(0, 0.2²)
    at init time, matching pretrained T5-small's per-dimension std so that
    both encoders occupy the same scale in ELF's training space after the
    latent_std=0.2 normalisation.  No transformer, no contextualisation,
    no trainable parameters.

    Why this is the right baseline:
    - Eliminates the "encoder wasn't trained" confound: T5-small's value is
      its *geometry* (semantic clusters, smooth manifold), not its role as a
      feature extractor.  A random *contextual* T5 still runs an attention
      transformer over random weights, which can impose spurious structure
      or mask the pure-geometry comparison.
    - Much faster forward pass (single embedding lookup, no transformer).
    - Directly tests the hypothesis: structured semantic geometry → low
      conditional variance Var[x|z_t] → better flow-matching data efficiency.
    """

    def __init__(self, vocab_size: int = 32128, d_model: int = 512,
                 embed_std: float = 0.2, seed: int = 0):
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        # Match pretrained T5-small's per-dimension std (~0.2) so that after
        # ELF's latent normalisation (divide by latent_std=0.2) both encoders
        # have std≈1.0 per dimension in the training space.  Unit-norm init
        # would give std≈0.044 → 5× scale mismatch → unfair noise-to-signal ratio.
        weight = torch.randn(vocab_size, d_model, generator=gen) * embed_std
        self.embedding = nn.Embedding(vocab_size, d_model, _weight=weight)
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        deterministic: bool = True,
    ) -> torch.Tensor:
        # (B, S) → (B, S, d) — same output shape as T5Encoder
        return self.embedding(input_ids)


def get_encoder(model_name: str, dtype: Any, encoder_type: str = "t5"):
    """Return ``(config, encoder_module)``.

    Args:
        model_name: HuggingFace T5 model name (e.g. ``"t5-small"``).
            Only used when ``encoder_type="t5"``.
        dtype: torch dtype for the encoder weights.
        encoder_type: ``"t5"`` (default, pretrained) or
            ``"random_embedding"`` (frozen random lookup table, Phase-0 baseline).
    """
    if encoder_type == "random_embedding":
        log_for_0("Loading RandomEmbeddingEncoder (frozen random lookup, Phase-0 baseline)...")
        # Use T5-small dimensions so the rest of the model is unchanged
        config = T5EncoderConfig.from_pretrained(model_name, dtype=dtype)
        model = RandomEmbeddingEncoder(
            vocab_size=config.vocab_size, d_model=config.d_model,
        )
        if dtype is not None:
            model = model.to(dtype)
        return config, model

    if encoder_type == "t5":
        log_for_0(f"Loading T5 Encoder: {model_name} (pretrained)...")
        config = T5EncoderConfig.from_pretrained(model_name, dtype=dtype)
        model = T5Encoder(config, pretrained=True)
        if dtype is not None:
            model = model.to(dtype)
        return config, model

    raise ValueError(f"Unknown encoder_type: {encoder_type!r}. Choose 't5' or 'random_embedding'.")
