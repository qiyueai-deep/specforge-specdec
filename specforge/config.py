from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class ModelConfig:
    vocab_size: int
    max_seq_len: int = 128
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    d_ff: int = 384
    dropout: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ModelConfig":
        return cls(**data)


@dataclass
class TrainConfig:
    seed: int = 20260820
    seq_len: int = 48
    batch_size: int = 24
    target_steps: int = 220
    draft_sft_steps: int = 120
    distill_steps: int = 220
    learning_rate: float = 2.5e-3
    min_learning_rate: float = 2.5e-4
    weight_decay: float = 0.01
    temperature: float = 2.0
    distill_alpha: float = 0.75
    two_stage_distillation: bool = False
    ce_refinement_fraction: float = 0.0
    final_distill_alpha: float = 0.05
    eval_batches: int = 12
    torch_threads: int = 4
    # v3: distillation objective and corpus selection.
    # "kl_ce" reproduces the original loss exactly.
    distill_objective: str = "kl_ce"
    tv_weight: float = 0.75
    tv_scale: float | None = None
    corpus_mode: str = "template"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "TrainConfig":
        return cls(**data)


def target_config(vocab_size: int) -> ModelConfig:
    # Matches the published artifact: 4,828,928 parameters at vocab_size=247.
    return ModelConfig(vocab_size=vocab_size, max_seq_len=128, d_model=256, n_layers=6, n_heads=8, d_ff=1024)


def draft_config(vocab_size: int) -> ModelConfig:
    # Matches the published artifact: 159,440 parameters, compression 30.29x.
    return ModelConfig(vocab_size=vocab_size, max_seq_len=128, d_model=80, n_layers=2, n_heads=4, d_ff=240)
