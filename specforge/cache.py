from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class KVCache:
    """Per-layer KV tensors with an explicit rollback operation."""

    n_layers: int
    keys: list[torch.Tensor | None] = field(init=False)
    values: list[torch.Tensor | None] = field(init=False)

    def __post_init__(self) -> None:
        self.keys = [None] * self.n_layers
        self.values = [None] * self.n_layers

    @property
    def length(self) -> int:
        first = self.keys[0]
        return 0 if first is None else int(first.shape[2])

    def get(self, layer: int) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        return self.keys[layer], self.values[layer]

    def append(self, layer: int, key: torch.Tensor, value: torch.Tensor) -> None:
        old_key, old_value = self.get(layer)
        self.keys[layer] = key if old_key is None else torch.cat([old_key, key], dim=2)
        self.values[layer] = value if old_value is None else torch.cat([old_value, value], dim=2)

    def truncate(self, new_length: int) -> None:
        if new_length < 0 or new_length > self.length:
            raise ValueError(f"cannot truncate cache of length {self.length} to {new_length}")
        for layer in range(self.n_layers):
            if self.keys[layer] is not None:
                self.keys[layer] = self.keys[layer][:, :, :new_length, :].contiguous()
                self.values[layer] = self.values[layer][:, :, :new_length, :].contiguous()

    def clone(self) -> "KVCache":
        result = KVCache(self.n_layers)
        result.keys = [None if tensor is None else tensor.clone() for tensor in self.keys]
        result.values = [None if tensor is None else tensor.clone() for tensor in self.values]
        return result
