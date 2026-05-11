# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""
End-to-end integration test for the FP8 shim: it must find PR #140's base
adapter, dequantize an FP8 state dict, and produce a state dict that PR
#140's own weight-conversion logic accepts.

This test exercises the real path — PR #140's
``NeuronQwen35ForCausalLM.convert_hf_to_neuron_state_dict`` is called
after our dequant step.
"""
from __future__ import annotations

import os
import sys

import pytest
import torch

# Our shim needs its own ``src`` dir on sys.path; PR #140's base adapter is
# discovered via the shim's heuristic.
_FP8_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if _FP8_SRC not in sys.path:
    sys.path.insert(0, _FP8_SRC)


def _load_base_adapter():
    """Duplicate of shim's heuristic — used by the test to build a config."""
    import importlib
    this_dir = os.path.dirname(os.path.abspath(__file__))
    sibling = os.path.abspath(os.path.join(this_dir, "..", "..", "Qwen3.6-27B"))
    if not os.path.isdir(os.path.join(sibling, "src")):
        pytest.skip("PR #140's Qwen3.6-27B contrib not on disk")
    if sibling not in sys.path:
        sys.path.insert(0, sibling)
    for name in list(sys.modules):
        if name == "src" or name.startswith("src."):
            del sys.modules[name]
    importlib.invalidate_caches()
    return importlib.import_module("src.modeling_qwen35")


def _make_mini_config(base, tp_degree: int = 1):
    from neuronx_distributed_inference.models.config import NeuronConfig
    neuron_config = NeuronConfig(
        tp_degree=tp_degree,
        batch_size=1,
        seq_len=128,
        torch_dtype=torch.bfloat16,
        fused_qkv=True,
    )
    cfg = base.Qwen35InferenceConfig(
        neuron_config=neuron_config,
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        intermediate_size=512,
        vocab_size=1000,
        rms_norm_eps=1e-6,
        max_position_embeddings=4096,
        rope_theta=10000,
        hidden_act="silu",
        linear_num_value_heads=8,
        linear_num_key_heads=4,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        linear_conv_kernel_dim=4,
    )
    # Pretend this is the FP8 checkpoint: attach quantization_config that
    # our shim will consume.
    cfg.quantization_config = {
        "activation_scheme": "dynamic",
        "fmt": "e4m3",
        "quant_method": "fp8",
        "weight_block_size": [128, 128],
        "modules_to_not_convert": [],
    }
    return cfg


def _build_fp8_state_dict(config) -> dict:
    """Build an HF-style state dict with FP8 weight/scale pairs on a subset of
    dense tensors, plus the other tensors left as bf16."""
    import copy
    # Borrow PR #140's own mini-state-dict helper rather than re-implementing.
    from src.modeling_qwen35 import NeuronQwen35ForCausalLM  # noqa: F401
    # Re-use the test helper in PR #140's test module.
    _CONTRIB_ROOT = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "Qwen3.6-27B")
    )
    sys.path.insert(0, os.path.join(_CONTRIB_ROOT, "test", "unit"))
    from test_weight_conversion import _make_mini_state_dict  # type: ignore
    sd = _make_mini_state_dict(config)

    # Inject FP8 _scale_inv pairs for a *subset* of dense weights: take
    # each layer's MLP down_proj (present in both full and linear layers).
    H = config.hidden_size
    I = config.intermediate_size
    block = config.quantization_config["weight_block_size"][0]  # 128
    for i in range(config.num_hidden_layers):
        key = f"layers.{i}.mlp.down_proj.weight"
        if key not in sd:
            continue
        w = sd[key]
        # Round shape up to block for the scale tensor; dequant handles
        # tail truncation.
        rows_blocks = (w.shape[0] + block - 1) // block
        cols_blocks = (w.shape[1] + block - 1) // block
        # Pick a deterministic scale = 0.5 so we can verify exact values later.
        scale = torch.full(
            (rows_blocks, cols_blocks), 0.5, dtype=torch.float32
        )
        sd[f"layers.{i}.mlp.down_proj.weight_scale_inv"] = scale
    # Take a copy of the *pre-dequant* FP8 weights for later comparison.
    pre_dequant = {
        k: v.clone() for k, v in sd.items() if k.endswith(".weight") and "down_proj" in k
    }
    return sd, pre_dequant


def test_shim_end_to_end_converts_fp8_state_dict():
    base = _load_base_adapter()
    config = _make_mini_config(base)
    sd, pre_dequant = _build_fp8_state_dict(config)

    from modeling_qwen3_6_fp8 import (
        convert_qwen3_6_fp8_hf_to_neuron_state_dict,
    )

    out = convert_qwen3_6_fp8_hf_to_neuron_state_dict(sd, config)

    # 1. No _scale_inv keys leak through.
    leaked = [k for k in out if k.endswith("_scale_inv")]
    assert not leaked, f"_scale_inv keys not cleaned up: {leaked}"

    # 2. Dequantized down_proj is bf16 and equals 0.5 * original fp8 (our scale
    #    was 0.5 everywhere). NB: PR #140's converter may rename / reshape keys
    #    (fused_qkv handling runs on attention layers only — down_proj should
    #    survive). Compare first layer only.
    key = "layers.0.mlp.down_proj.weight"
    assert key in out, f"{key} missing after conversion"
    got = out[key]
    want = pre_dequant[key].to(torch.float32) * 0.5
    # Cast wanted to bf16 to match the dequantizer's target_dtype.
    want_bf16 = want.to(torch.bfloat16)
    torch.testing.assert_close(got, want_bf16, atol=5e-3, rtol=5e-3)

    # 3. Non-quantized tensors still present (rank_util should be injected by
    #    PR #140's converter).
    assert "rank_util.rank" in out, "PR #140 converter did not inject rank_util"


def test_shim_no_op_without_quantization_config():
    """If config has no quantization_config, shim just delegates to PR #140's converter."""
    base = _load_base_adapter()
    config = _make_mini_config(base)
    # Remove the quantization hint.
    del config.quantization_config

    # Build a plain bf16 state dict (no _scale_inv).
    _CONTRIB_ROOT = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "Qwen3.6-27B")
    )
    if os.path.join(_CONTRIB_ROOT, "test", "unit") not in sys.path:
        sys.path.insert(0, os.path.join(_CONTRIB_ROOT, "test", "unit"))
    from test_weight_conversion import _make_mini_state_dict  # type: ignore
    sd = _make_mini_state_dict(config)

    from modeling_qwen3_6_fp8 import convert_qwen3_6_fp8_hf_to_neuron_state_dict

    out = convert_qwen3_6_fp8_hf_to_neuron_state_dict(sd, config)
    assert "rank_util.rank" in out
    assert not any(k.endswith("_scale_inv") for k in out)
