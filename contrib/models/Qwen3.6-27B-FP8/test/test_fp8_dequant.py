# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Unit test for the FP8 block-scaled dequantization recipe.

This does NOT depend on PR #140's base adapter; it exercises only the
self-contained numerics that this contrib adapter owns.
"""
from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "src")
)

from modeling_qwen3_6_fp8 import dequantize_fp8_block_scaled  # noqa: E402


def test_dequantize_matches_manual_math():
    """Round-trip: quantize bf16 weight block-wise, dequantize, compare."""
    block = 4
    rows, cols = 8, 8
    torch.manual_seed(0)
    original = torch.randn(rows, cols, dtype=torch.float32)

    # Quantize to fake FP8 by block: find each 4x4 block's scale, divide.
    blocks_r, blocks_c = rows // block, cols // block
    scale = torch.zeros(blocks_r, blocks_c, dtype=torch.float32)
    fp8_fake = torch.zeros_like(original)
    for br_i in range(blocks_r):
        for bc_i in range(blocks_c):
            sub = original[br_i * block:(br_i + 1) * block,
                           bc_i * block:(bc_i + 1) * block]
            s = sub.abs().max().item() + 1e-8
            scale[br_i, bc_i] = s
            fp8_fake[br_i * block:(br_i + 1) * block,
                     bc_i * block:(bc_i + 1) * block] = sub / s

    sd = {
        "weights.a.weight": fp8_fake.clone(),
        "weights.a.weight_scale_inv": scale.clone(),
    }
    dequantize_fp8_block_scaled(
        sd, weight_block_size=[block, block], target_dtype=torch.float32
    )
    assert "weights.a.weight_scale_inv" not in sd
    torch.testing.assert_close(sd["weights.a.weight"], original, atol=1e-5, rtol=1e-5)


def test_modules_to_not_convert_preserves_raw_fp8():
    """Weights listed in modules_to_not_convert are left as-is and their scale dropped."""
    block = 2
    rows, cols = 4, 4
    fake_fp8 = torch.randn(rows, cols)
    scale = torch.ones(rows // block, cols // block)

    sd = {
        "visual.weight": fake_fp8.clone(),
        "visual.weight_scale_inv": scale.clone(),
    }
    dequantize_fp8_block_scaled(
        sd,
        weight_block_size=[block, block],
        target_dtype=torch.bfloat16,
        modules_to_not_convert=["visual"],
    )
    # weight_scale_inv is dropped either way (it has no consumer after load).
    assert "visual.weight_scale_inv" not in sd
    # The weight itself must NOT have been cast or scaled.
    torch.testing.assert_close(sd["visual.weight"], fake_fp8, atol=0, rtol=0)


def test_missing_scale_is_noop():
    """Weights without a matching _scale_inv pass through untouched."""
    w = torch.randn(4, 4)
    sd = {"layers.0.self_attn.q_proj.weight": w.clone()}
    dequantize_fp8_block_scaled(
        sd, weight_block_size=[2, 2], target_dtype=torch.bfloat16
    )
    torch.testing.assert_close(sd["layers.0.self_attn.q_proj.weight"], w, atol=0, rtol=0)


def test_tail_block_truncation():
    """Weights whose dims aren't a clean multiple of the block still work."""
    # 5x5 weight, block [2,2] -> scale is 3x3 (expanded to 6x6, truncated to 5x5).
    rows, cols = 5, 5
    block = 2
    w = torch.randn(rows, cols)
    # Build a trivial scale of 3x3 ones so dequantize returns w unchanged.
    scale = torch.ones(3, 3)
    sd = {"x.weight": w.clone(), "x.weight_scale_inv": scale.clone()}
    dequantize_fp8_block_scaled(
        sd, weight_block_size=[block, block], target_dtype=torch.float32
    )
    torch.testing.assert_close(sd["x.weight"], w, atol=1e-6, rtol=1e-6)
