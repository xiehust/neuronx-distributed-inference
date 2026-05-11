# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Shape / dispatch tests for HybridKVCacheManager.

We do not stand up a real full-attention KVCacheManager (it requires a
distributed init); instead we drive the linear-attention side directly,
which is the novel piece, and verify the layer-kind dispatch maps
correctly.
"""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "src")
)


def _make_fake_config(
    num_layers: int = 4,
    num_k_heads: int = 4,
    num_v_heads: int = 8,
    head_k_dim: int = 16,
    head_v_dim: int = 16,
    conv_kernel: int = 4,
    tp_degree: int = 1,
    batch: int = 1,
    layer_types=None,
):
    if layer_types is None:
        # 3 linear + 1 full pattern, tiled.
        layer_types = []
        for i in range(num_layers):
            layer_types.append(
                "full_attention" if (i + 1) % 4 == 0 else "linear_attention"
            )
    neuron_config = SimpleNamespace(
        torch_dtype=torch.float32,
        kv_cache_batch_size=batch,
        kv_cache_padding_size=0,
        tp_degree=tp_degree,
        max_length=128,
        is_medusa=False,
        num_medusa_heads=0,
        padding_side="right",
        is_continuous_batching=False,
        flash_decoding_enabled=False,
        attention_dp_degree=1,
        kv_cache_tiling=False,
        attention_dtype=None,
        kv_quant_config=None,
        k_cache_transposed=False,
    )
    return SimpleNamespace(
        num_hidden_layers=num_layers,
        layer_types=layer_types,
        linear_num_key_heads=num_k_heads,
        linear_num_value_heads=num_v_heads,
        linear_key_head_dim=head_k_dim,
        linear_value_head_dim=head_v_dim,
        linear_conv_kernel_dim=conv_kernel,
        hidden_size=64,
        num_attention_heads=num_k_heads,
        neuron_config=neuron_config,
        num_cores_per_group=1,
    )


def test_linear_state_shapes_no_full_layers():
    """Pure-linear hybrid: no full-attention sub-manager is built."""
    cfg = _make_fake_config(
        num_layers=3, layer_types=["linear_attention"] * 3
    )
    from neuronx_distributed_inference.modules.kvcache.hybrid_kv_cache_manager import (
        HybridKVCacheManager,
    )
    mgr = HybridKVCacheManager(cfg, num_kv_head=cfg.linear_num_key_heads)
    assert mgr._full_manager is None
    assert len(mgr._conv_states) == 3
    assert len(mgr._recurrent_states) == 3
    conv_shape = mgr._conv_states[0].shape
    rec_shape = mgr._recurrent_states[0].shape
    # conv_dim = 2 * key_dim + value_dim (per TP=1 this is everything)
    key_dim = cfg.linear_num_key_heads * cfg.linear_key_head_dim
    value_dim = cfg.linear_num_value_heads * cfg.linear_value_head_dim
    expected_conv_dim = 2 * key_dim + value_dim
    assert conv_shape == (1, expected_conv_dim, cfg.linear_conv_kernel_dim - 1)
    assert rec_shape == (
        1,
        cfg.linear_num_value_heads,
        cfg.linear_key_head_dim,
        cfg.linear_value_head_dim,
    )
    # Recurrent buffer is fp32 regardless of model dtype.
    assert mgr._recurrent_states[0].dtype == torch.float32


def test_layer_kind_lookup():
    cfg = _make_fake_config(num_layers=4)  # default 3linear+1full pattern
    # Full-attention sub-manager init requires a distributed context; skip
    # it by monkey-patching HybridKVCacheManager to not build it.
    from neuronx_distributed_inference.modules.kvcache import hybrid_kv_cache_manager as m
    class _NoopFullManager:
        def __init__(self, *a, **kw): pass
    real_kv_mgr = m.KVCacheManager
    m.KVCacheManager = _NoopFullManager  # type: ignore[attr-defined]
    try:
        mgr = m.HybridKVCacheManager(cfg, num_kv_head=cfg.linear_num_key_heads)
    finally:
        m.KVCacheManager = real_kv_mgr  # type: ignore[attr-defined]

    assert mgr.num_layers() == 4
    assert mgr.layer_kind(0) == "linear_attention"
    assert mgr.layer_kind(1) == "linear_attention"
    assert mgr.layer_kind(2) == "linear_attention"
    assert mgr.layer_kind(3) == "full_attention"
    # 3 linear layers, 1 full layer.
    assert len(mgr._conv_states) == 3
    assert len(mgr._recurrent_states) == 3


def test_update_linear_cache_writes_in_place():
    cfg = _make_fake_config(
        num_layers=2, layer_types=["linear_attention", "linear_attention"]
    )
    from neuronx_distributed_inference.modules.kvcache.hybrid_kv_cache_manager import (
        HybridKVCacheManager,
    )
    mgr = HybridKVCacheManager(cfg, num_kv_head=cfg.linear_num_key_heads)

    conv0 = torch.randn_like(mgr._conv_states[0])
    rec0 = torch.randn(
        mgr._recurrent_states[0].shape, dtype=torch.float32
    )
    mgr.update_linear_cache_for_layer(0, conv0, rec0)
    torch.testing.assert_close(mgr._conv_states[0], conv0)
    torch.testing.assert_close(mgr._recurrent_states[0], rec0)
    # Layer 1 should be untouched.
    assert torch.all(mgr._conv_states[1] == 0)


def test_update_linear_rejects_full_layer():
    cfg = _make_fake_config(
        num_layers=2, layer_types=["full_attention", "linear_attention"]
    )
    # Stub full mgr to skip dist init.
    from neuronx_distributed_inference.modules.kvcache import hybrid_kv_cache_manager as m
    class _NoopFullManager:
        def __init__(self, *a, **kw): pass
    real = m.KVCacheManager
    m.KVCacheManager = _NoopFullManager  # type: ignore[attr-defined]
    try:
        mgr = m.HybridKVCacheManager(cfg, num_kv_head=cfg.linear_num_key_heads)
    finally:
        m.KVCacheManager = real

    with pytest.raises(ValueError, match="full_attention.*layer"):
        mgr.update_linear_cache_for_layer(
            0, torch.zeros(1), torch.zeros(1)
        )


def test_rejects_bad_layer_type():
    cfg = _make_fake_config(num_layers=1, layer_types=["gibberish"])
    from neuronx_distributed_inference.modules.kvcache.hybrid_kv_cache_manager import (
        HybridKVCacheManager,
    )
    with pytest.raises(ValueError, match="unknown entries"):
        HybridKVCacheManager(cfg, num_kv_head=cfg.linear_num_key_heads)


def test_tp_divisibility_check():
    cfg = _make_fake_config(
        num_layers=1,
        layer_types=["linear_attention"],
        tp_degree=3,  # does not divide 4
    )
    from neuronx_distributed_inference.modules.kvcache.hybrid_kv_cache_manager import (
        HybridKVCacheManager,
    )
    with pytest.raises(ValueError, match="tp_degree"):
        HybridKVCacheManager(cfg, num_kv_head=cfg.linear_num_key_heads)
