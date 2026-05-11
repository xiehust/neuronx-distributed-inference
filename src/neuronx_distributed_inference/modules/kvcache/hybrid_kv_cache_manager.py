# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Hybrid KV cache manager for models that mix standard full-attention layers
with SSM-style linear-attention layers (Qwen3.5 / Qwen3.6).

The stock ``KVCacheManager`` treats the per-layer cache as a fixed
``(K, V)`` pair — see ``kv_cache_manager.py:KVCacheManager.__init__`` where
``past_key_values`` is a flat ``ParameterList`` of length ``2 *
num_hidden_layers``. That contract breaks for linear-attention layers,
which carry:

  * ``conv_state``: ``[B, conv_dim, kernel_size - 1]`` — the last K-1
    pre-activations feeding the depthwise conv1d.
  * ``recurrent_state``: ``[B, num_v_heads, head_k_dim, head_v_dim]`` —
    the Gated Delta Rule matrix state.

This module introduces ``HybridKVCacheManager``, an aggregate that
delegates to the *existing* ``KVCacheManager`` for full-attention layers
and holds dedicated parameter lists for the linear-attention state. It
preserves the public ``get_cache`` / ``update_cache`` surface the rest
of NXDI calls, but returns per-layer state tuples whose **shape varies
by layer type**.

**Status:** scaffold + unit-tested tensor allocation. The buffers are
correctly sized per layer type. What remains to make this production-
ready:

  1. ``get_cache`` / ``update_cache`` in the base ``KVCacheManager`` are
     called with layer indices that assume a flat ``[(K,V), (K,V), …]``
     structure. The model-level forward (``NeuronBaseModel.get_model_output``)
     currently iterates ``past_key_values`` uniformly. To integrate this
     class we either (a) change the iteration to pass a **layer-type
     tag** to the decoder-layer forward, or (b) have the hybrid manager
     synthesise zero-shaped K/V for linear layers and zero-shaped
     (conv_state, recurrent_state) for full layers, so iteration stays
     uniform. (a) is cleaner but touches model_base.py.
  2. Tracing requires the conv_state + recurrent_state tensors to appear
     in the traced kernel signature alongside K/V. That plumbing follows
     from step (1).

The scaffold below is framework-compatible in the sense that
``HybridKVCacheManager`` is a proper ``nn.Module`` with parameters
registered for all four tensor types — meaning a future tracer will
find them during module walk.
"""
from __future__ import annotations

from typing import Iterable, List, Optional, Tuple

import torch
from torch import Tensor, nn

from neuronx_distributed_inference.models.config import InferenceConfig
from neuronx_distributed_inference.modules.kvcache.kv_cache_manager import (
    KVCacheManager,
)


class HybridKVCacheManager(nn.Module):
    """Per-layer cache manager that supports full + linear-attention layers.

    The caller supplies ``layer_types`` (list of ``"full_attention"`` or
    ``"linear_attention"``, one per decoder layer). For full-attention
    layers, an internal ``KVCacheManager`` holds the standard K/V pair.
    For linear-attention layers, this class owns two buffers:
    ``conv_states`` and ``recurrent_states``.

    Args:
        config: InferenceConfig with ``layer_types``, ``linear_*`` dims,
            ``linear_conv_kernel_dim``, etc. set (i.e. a Qwen3_5-style
            config).
        num_kv_head: KV heads for the full-attention layers.
        **kwargs: Forwarded to the inner ``KVCacheManager``.
    """

    FULL = "full_attention"
    LINEAR = "linear_attention"

    def __init__(
        self,
        config: InferenceConfig,
        num_kv_head: int,
        **kwargs,
    ):
        super().__init__()
        layer_types: List[str] = list(getattr(config, "layer_types"))
        self.layer_types = layer_types

        self.full_layer_indices: List[int] = [
            i for i, t in enumerate(layer_types) if t == self.FULL
        ]
        self.linear_layer_indices: List[int] = [
            i for i, t in enumerate(layer_types) if t == self.LINEAR
        ]
        if len(self.full_layer_indices) + len(self.linear_layer_indices) != len(layer_types):
            unknown = [t for t in layer_types if t not in (self.FULL, self.LINEAR)]
            raise ValueError(
                f"layer_types contains unknown entries: {unknown!r}. "
                f"Expected only {self.FULL!r} or {self.LINEAR!r}."
            )

        # Full-attention sub-manager: wraps the stock KV cache, pretending
        # that only the full layers exist. We swap ``config.num_hidden_layers``
        # for that subset so the inner manager allocates the right count.
        if self.full_layer_indices:
            full_config = _shallow_clone_config_with_layer_count(
                config, len(self.full_layer_indices)
            )
            self._full_manager = KVCacheManager(
                full_config, num_kv_head=num_kv_head, **kwargs
            )
        else:
            self._full_manager = None

        # Linear-attention buffers: shape is determined by the SSM
        # hyperparameters in the config. Every linear layer gets a pair
        # (conv_state, recurrent_state).
        dtype = config.neuron_config.torch_dtype
        batch = config.neuron_config.kv_cache_batch_size + config.neuron_config.kv_cache_padding_size
        num_v_heads = getattr(config, "linear_num_value_heads")
        num_k_heads = getattr(config, "linear_num_key_heads")
        head_k_dim = getattr(config, "linear_key_head_dim")
        head_v_dim = getattr(config, "linear_value_head_dim")
        conv_k = getattr(config, "linear_conv_kernel_dim")
        tp = config.neuron_config.tp_degree
        if num_v_heads % tp != 0 or num_k_heads % tp != 0:
            raise ValueError(
                f"tp_degree={tp} must divide both linear_num_value_heads="
                f"{num_v_heads} and linear_num_key_heads={num_k_heads}"
            )
        num_v_heads_per_tp = num_v_heads // tp
        # conv_dim = 2 * key_dim + value_dim  (per TP rank, sharded along
        # num_k_heads == num_v_heads // group_ratio).
        conv_dim_per_tp = 2 * (num_k_heads // tp) * head_k_dim + num_v_heads_per_tp * head_v_dim

        conv_shape = (batch, conv_dim_per_tp, max(conv_k - 1, 0))
        recurrent_shape = (batch, num_v_heads_per_tp, head_k_dim, head_v_dim)

        self._conv_states = nn.ParameterList([
            nn.Parameter(torch.zeros(conv_shape, dtype=dtype), requires_grad=False)
            for _ in self.linear_layer_indices
        ])
        # Recurrent state is carried in fp32 to match the HF reference
        # (``mamba_ssm_dtype: float32``), independent of the model dtype.
        self._recurrent_states = nn.ParameterList([
            nn.Parameter(torch.zeros(recurrent_shape, dtype=torch.float32), requires_grad=False)
            for _ in self.linear_layer_indices
        ])

        # Build a fast lookup: layer_idx -> (kind, sub_index).
        self._layer_map: List[Tuple[str, int]] = []
        full_ctr = linear_ctr = 0
        for t in layer_types:
            if t == self.FULL:
                self._layer_map.append((self.FULL, full_ctr))
                full_ctr += 1
            else:
                self._layer_map.append((self.LINEAR, linear_ctr))
                linear_ctr += 1

    # ------------------------------------------------------------------
    # Read / write API — mirrors KVCacheManager.get_cache / update_cache
    # but returns heterogeneous per-layer state tuples.
    # ------------------------------------------------------------------

    def get_cache_for_layer(
        self,
        layer_idx: int,
        seq_len: Optional[int] = None,
        **kwargs,
    ):
        """Return the per-layer cache as a tuple whose shape depends on layer type.

        For full-attention layers: ``(K, V)`` (same as KVCacheManager).
        For linear-attention layers: ``(conv_state, recurrent_state)``.
        """
        kind, sub_idx = self._layer_map[layer_idx]
        if kind == self.FULL:
            assert self._full_manager is not None
            k_cache, v_cache = self._full_manager.get_kv_by_layer_id(
                idx=sub_idx, seq_len=seq_len, **kwargs
            )
            return (k_cache, v_cache)
        # Linear-attention layer.
        return (self._conv_states[sub_idx], self._recurrent_states[sub_idx])

    def update_linear_cache_for_layer(
        self,
        layer_idx: int,
        new_conv_state: Tensor,
        new_recurrent_state: Tensor,
    ) -> None:
        """In-place update for a linear-attention layer's buffers."""
        kind, sub_idx = self._layer_map[layer_idx]
        if kind != self.LINEAR:
            raise ValueError(
                f"update_linear_cache_for_layer called on a {kind!r} layer (idx={layer_idx})"
            )
        self._conv_states[sub_idx].data.copy_(new_conv_state)
        # recurrent_state is fp32; cast if caller passed a different dtype.
        self._recurrent_states[sub_idx].data.copy_(
            new_recurrent_state.to(self._recurrent_states[sub_idx].dtype)
        )

    def update_full_cache_for_layer(self, layer_idx: int, **kwargs):
        """Thin delegation to the underlying full-attention KV cache."""
        kind, sub_idx = self._layer_map[layer_idx]
        if kind != self.FULL:
            raise ValueError(
                f"update_full_cache_for_layer called on a {kind!r} layer (idx={layer_idx})"
            )
        assert self._full_manager is not None
        return self._full_manager.update_kv_by_layer_id(idx=sub_idx, **kwargs)

    # ------------------------------------------------------------------
    # Introspection helpers.
    # ------------------------------------------------------------------

    def layer_kind(self, layer_idx: int) -> str:
        return self._layer_map[layer_idx][0]

    def num_layers(self) -> int:
        return len(self._layer_map)

    def iter_layers(self) -> Iterable[Tuple[int, str]]:
        return ((i, kind) for i, (kind, _) in enumerate(self._layer_map))


def _shallow_clone_config_with_layer_count(
    config: InferenceConfig, new_num_layers: int
) -> InferenceConfig:
    """Create a minimal copy of ``config`` whose ``num_hidden_layers`` is overridden.

    KVCacheManager reads ``config.num_hidden_layers`` to size its
    ``ParameterList``. We need to give it a count that reflects only the
    full-attention layers, without mutating the caller's config.
    """
    # InferenceConfig objects in NXDI accept kwargs in __setattr__; we use
    # object.__setattr__ on a shallow copy to avoid triggering any custom
    # load_config hooks the subclass may have.
    import copy

    clone = copy.copy(config)
    object.__setattr__(clone, "num_hidden_layers", new_num_layers)
    return clone
