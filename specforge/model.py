from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .cache import KVCache
from .config import ModelConfig


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        if config.d_model % config.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.out = nn.Linear(config.d_model, config.d_model, bias=False)

    def forward(self, x: torch.Tensor, cache: KVCache | None, layer_index: int) -> torch.Tensor:
        batch, seq_len, width = x.shape
        qkv = self.qkv(x).view(batch, seq_len, 3, self.n_heads, self.head_dim)
        query, key, value = qkv.unbind(dim=2)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        past_key, past_value = (None, None) if cache is None else cache.get(layer_index)
        past_len = 0 if past_key is None else past_key.shape[2]
        all_key = key if past_key is None else torch.cat([past_key, key], dim=2)
        all_value = value if past_value is None else torch.cat([past_value, value], dim=2)
        scores = query @ all_key.transpose(-2, -1) / math.sqrt(self.head_dim)
        query_positions = past_len + torch.arange(seq_len, device=x.device)
        key_positions = torch.arange(past_len + seq_len, device=x.device)
        causal_mask = key_positions.unsqueeze(0) > query_positions.unsqueeze(1)
        scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        attention = F.softmax(scores, dim=-1)
        output = attention @ all_value
        output = output.transpose(1, 2).contiguous().view(batch, seq_len, width)
        if cache is not None:
            cache.append(layer_index, key, value)
        return self.out(output)

    def forward_functional(
        self,
        x: torch.Tensor,
        past_key: torch.Tensor,
        past_value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Functional attention used by the torch.compile path.

        Same math as ``forward``, but the KV state is passed in and returned as
        plain tensors instead of mutating a KVCache object, so the whole pass is
        traceable by torch.compile. ``past_key``/``past_value`` have shape
        [B, H, T, D]; T == 0 means no history yet. Returns the block output
        together with the *concatenated* keys/values including the new tokens.
        """
        batch, seq_len, width = x.shape
        qkv = self.qkv(x).view(batch, seq_len, 3, self.n_heads, self.head_dim)
        query, key, value = qkv.unbind(dim=2)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        past_len = past_key.shape[2]
        all_key = torch.cat([past_key, key], dim=2)
        all_value = torch.cat([past_value, value], dim=2)
        scores = query @ all_key.transpose(-2, -1) / math.sqrt(self.head_dim)
        query_positions = past_len + torch.arange(seq_len, device=x.device)
        key_positions = torch.arange(past_len + seq_len, device=x.device)
        causal_mask = key_positions.unsqueeze(0) > query_positions.unsqueeze(1)
        scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        attention = F.softmax(scores, dim=-1)
        output = attention @ all_value
        output = output.transpose(1, 2).contiguous().view(batch, seq_len, width)
        return self.out(output), all_key, all_value


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.d_model)
        self.attention = CausalSelfAttention(config)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff), nn.GELU(),
            nn.Linear(config.d_ff, config.d_model),
        )

    def forward(self, x: torch.Tensor, cache: KVCache | None, layer_index: int) -> torch.Tensor:
        x = x + self.attention(self.norm1(x), cache, layer_index)
        return x + self.mlp(self.norm2(x))

    def forward_functional(
        self,
        x: torch.Tensor,
        past_key: torch.Tensor,
        past_value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h, new_key, new_value = self.attention.forward_functional(self.norm1(x), past_key, past_value)
        x = x + h
        return x + self.mlp(self.norm2(x)), new_key, new_value


class TinyTransformerLM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.final_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def new_cache(self) -> KVCache:
        return KVCache(self.config.n_layers)

    def forward(self, input_ids: torch.Tensor, cache: KVCache | None = None) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        past_len = 0 if cache is None else cache.length
        if past_len + input_ids.shape[1] > self.config.max_seq_len:
            raise ValueError("sequence exceeds max_seq_len")
        positions = torch.arange(past_len, past_len + input_ids.shape[1], device=input_ids.device)
        x = self.token_embedding(input_ids) + self.position_embedding(positions).unsqueeze(0)
        for index, block in enumerate(self.blocks):
            x = block(x, cache, index)
        return self.lm_head(self.final_norm(x))

    def forward_functional(
        self,
        input_ids: torch.Tensor,
        past_keys: torch.Tensor,
        past_values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Functional forward used by the torch.compile path.

        Mirrors ``forward`` but the KV state is passed in and returned as plain
        *stacked* tensors of shape [n_layers, H, T, D] instead of mutating a
        KVCache, so the whole pass is traceable by torch.compile. T == 0 means
        no history yet; the returned tensors hold the concatenated keys/values
        including the new tokens. Callers own the state and implement rollback
        by slicing along dim 2. Stacking (instead of a list of tensors) lets
        torch.compile treat T as a single dynamic dimension without
        recompiling on every cache growth step.
        """
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        batch, seq_len = input_ids.shape
        past_len = past_keys.shape[3]
        if past_len + seq_len > self.config.max_seq_len:
            raise ValueError("sequence exceeds max_seq_len")
        positions = torch.arange(past_len, past_len + seq_len, device=input_ids.device)
        x = self.token_embedding(input_ids) + self.position_embedding(positions).unsqueeze(0)
        new_keys: list[torch.Tensor] = []
        new_values: list[torch.Tensor] = []
        for index, block in enumerate(self.blocks):
            x, new_key, new_value = block.forward_functional(
                x, past_keys[index], past_values[index]
            )
            new_keys.append(new_key)
            new_values.append(new_value)
        return self.lm_head(self.final_norm(x)), torch.stack(new_keys), torch.stack(new_values)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
