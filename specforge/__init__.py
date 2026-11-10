"""SpecForge: distilled speculative decoding with rollback-capable KV caches."""

from .decoding import autoregressive_generate, speculative_generate
from .model import TinyTransformerLM

__all__ = ["TinyTransformerLM", "autoregressive_generate", "speculative_generate"]
__version__ = "1.0.0"
