# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""
FP8 loader delta for ``Qwen/Qwen3.6-27B-FP8``.

This module is **only** the FP8-specific layer on top of the base
Qwen3.6-27B adapter introduced by PR #140
(https://github.com/aws-neuron/neuronx-distributed-inference/pull/140).
It relies on the upstream adapter for the model architecture (hybrid
GatedDeltaNet + GQA, mRoPE, attn_output_gate, NKI kernels, etc.) and
adds two responsibilities of its own:

  1. Block-wise FP8 → bf16 dequantization on load. Qwen3.6-27B-FP8
     stores per-weight ``*_scale_inv`` tensors with
     ``weight_block_size=[128, 128]`` (E4M3, dynamic activation scheme).
     The same recipe used by ``qwen3_moe`` and ``DeepSeek-V3`` applies
     unchanged.
  2. Honoring ``quantization_config.modules_to_not_convert`` so the
     vision tower, embeddings, and various SSM scalars stay in bf16.

The base adapter in PR #140 is BF16-only by design; this file exists
specifically to bridge the published FP8 checkpoint to that adapter.
"""
from __future__ import annotations

import importlib
from typing import Any, Dict

import torch


def _import_base_adapter():
    """Locate the base Qwen3.6-27B adapter introduced by PR #140.

    PR #140's ``contrib/models/Qwen3.6-27B/src/__init__.py`` itself
    imports ``from src.modeling_qwen35 import ...`` — which means that
    the ``Qwen3.6-27B/`` directory (its parent) needs to be on
    ``sys.path`` so the ``src`` name resolves. Because the folder name
    contains a ``.`` it is *not* importable as a Python package, so we
    can't use ``contrib.models.Qwen3_6_27B.src.modeling_qwen35``.

    This helper tries, in order:
      1. ``modeling_qwen35`` — if the user already added
         ``contrib/models/Qwen3.6-27B/src`` to sys.path.
      2. ``src.modeling_qwen35`` — if the user added
         ``contrib/models/Qwen3.6-27B`` to sys.path (PR #140's README
         convention).
      3. A repo-layout heuristic: walk up from this file and inject
         the sibling ``Qwen3.6-27B`` directory if we find it.
    """
    import os

    for candidate in ("modeling_qwen35", "src.modeling_qwen35"):
        try:
            return importlib.import_module(candidate)
        except ImportError:
            continue

    # Heuristic: our shim lives at
    #   contrib/models/Qwen3.6-27B-FP8/src/modeling_qwen3_6_fp8.py
    # so the PR #140 contrib lives at ../../Qwen3.6-27B relative to this
    # file's parent directory. The two contribs both call their package
    # ``src``, so whichever gets imported first wins in ``sys.modules``.
    # We invalidate any existing ``src`` entry and then re-resolve under
    # the PR's contrib dir.
    this_dir = os.path.dirname(os.path.abspath(__file__))
    sibling = os.path.abspath(os.path.join(this_dir, "..", "..", "Qwen3.6-27B"))
    if os.path.isdir(os.path.join(sibling, "src")):
        import sys as _sys
        if sibling not in _sys.path:
            _sys.path.insert(0, sibling)
        # Drop any cached ``src`` / ``src.*`` modules so the PR's ``src``
        # is what gets imported next.
        for name in list(_sys.modules):
            if name == "src" or name.startswith("src."):
                del _sys.modules[name]
        importlib.invalidate_caches()
        try:
            return importlib.import_module("src.modeling_qwen35")
        except ImportError:
            pass

    raise ImportError(
        "Could not import the base Qwen3.6-27B adapter. This FP8 delta "
        "depends on PR #140 "
        "(https://github.com/aws-neuron/neuronx-distributed-inference/pull/140). "
        "Either (a) merge / check out that PR (our repo already has this "
        "on main if you ran `git cherry-pick pr-140`), or (b) add "
        "contrib/models/Qwen3.6-27B to PYTHONPATH."
    )


# ---------------------------------------------------------------------------
# FP8 state-dict dequantization.
#
# Identical recipe to qwen3_moe.maybe_dequantize_layer and
# DeepSeek-V3's _dequantize_fp8_state_dict: for every "*.weight" that
# has a matching "*.weight_scale_inv" tensor, expand the scale with
# repeat_interleave per weight_block_size, multiply in fp32, cast back
# to the neuron target dtype.
# ---------------------------------------------------------------------------


def dequantize_fp8_block_scaled(
    state_dict: Dict[str, torch.Tensor],
    weight_block_size: list[int],
    target_dtype: torch.dtype = torch.bfloat16,
    modules_to_not_convert: list[str] | None = None,
) -> None:
    """In-place: replace every FP8 weight with its bf16 dequantization.

    Args:
        state_dict: Mutable HF state dict; mutated in place. Keys that
            end in ``_scale_inv`` are consumed and deleted once applied.
        weight_block_size: ``[block_row, block_col]`` from the HF
            ``quantization_config``. Qwen3.6-27B-FP8 uses ``[128, 128]``.
        target_dtype: Output dtype for the dequantized weights
            (typically ``torch.bfloat16``).
        modules_to_not_convert: List of *exact* module prefixes whose
            weights should be left alone. Populate from
            ``quantization_config["modules_to_not_convert"]``.
    """
    br, bc = weight_block_size
    skip = set(modules_to_not_convert or [])
    to_delete: list[str] = []

    for key in list(state_dict.keys()):
        if not key.endswith("_scale_inv"):
            continue
        weight_key = key[: -len("_scale_inv")]
        if weight_key not in state_dict:
            continue
        # Respect modules_to_not_convert: the key for exclusion checks is
        # the *module prefix* that the HF config uses (the part of the
        # dotted path before the terminal ".weight").
        module_prefix = weight_key.rsplit(".", 1)[0]
        if module_prefix in skip:
            to_delete.append(key)
            continue

        w = state_dict[weight_key]
        s = state_dict[key]
        expanded = s.repeat_interleave(br, dim=0).repeat_interleave(bc, dim=1)
        # Align shapes: if the weight isn't a clean multiple of the
        # block size along either axis, the tail block is truncated.
        expanded = expanded[: w.shape[0], : w.shape[1]]
        dequantized = w.to(torch.float32) * expanded.to(torch.float32)
        state_dict[weight_key] = dequantized.to(target_dtype)
        to_delete.append(key)

    for key in to_delete:
        del state_dict[key]


def convert_qwen3_6_fp8_hf_to_neuron_state_dict(
    state_dict: Dict[str, torch.Tensor],
    config: Any,
) -> Dict[str, torch.Tensor]:
    """Dequantize FP8 weights in-place, then delegate to PR #140's class-
    level converter (which does the VL/language_model prefix stripping and
    the actual NxDI-format conversion).

    ``config`` is expected to be an instance of the base adapter's
    ``Qwen35InferenceConfig`` (from PR #140).
    """
    qcfg = getattr(config, "quantization_config", None)
    if isinstance(qcfg, dict) and qcfg.get("quant_method") == "fp8":
        target_dtype = config.neuron_config.torch_dtype
        dequantize_fp8_block_scaled(
            state_dict,
            weight_block_size=qcfg["weight_block_size"],
            target_dtype=target_dtype,
            modules_to_not_convert=qcfg.get("modules_to_not_convert"),
        )

    base = _import_base_adapter()
    base_cls = getattr(base, "NeuronQwen35ForCausalLM", None)
    if base_cls is None:
        raise AttributeError(
            "Base Qwen3.6-27B adapter (PR #140) does not expose "
            "`NeuronQwen35ForCausalLM`. If the upstream PR renamed the "
            "class, update this shim accordingly."
        )
    # NeuronQwen35ForCausalLM.convert_hf_to_neuron_state_dict is a
    # @staticmethod that (a) strips `language_model.` / `model.` / `visual`
    # / `mtp.` prefixes and (b) calls the module-level
    # convert_qwen35_hf_to_neuron_state_dict helper.
    return base_cls.convert_hf_to_neuron_state_dict(state_dict, config)


# ---------------------------------------------------------------------------
# Entry-point class: reuse the base adapter, only overriding the state-dict
# converter so FP8 is transparent to downstream users.
# ---------------------------------------------------------------------------


def build_fp8_for_causal_lm_class():
    """Factory that returns a subclass of the PR #140 for-causal-LM class
    with the FP8 loader wired in. Kept as a factory (instead of a top-
    level subclass) so the module imports cleanly even before the base
    adapter is available on the path.
    """
    base = _import_base_adapter()
    base_cls = getattr(base, "NeuronQwen35ForCausalLM", None)
    if base_cls is None:
        raise AttributeError(
            "Base Qwen3.6-27B adapter (PR #140) does not expose "
            "`NeuronQwen35ForCausalLM`. If the upstream PR renamed the "
            "class, update this shim accordingly."
        )

    class NeuronQwen3_6_27B_FP8_ForCausalLM(base_cls):  # type: ignore[misc,valid-type]
        """PR #140 base + FP8 loader."""

        @staticmethod
        def convert_hf_to_neuron_state_dict(
            state_dict: Dict[str, torch.Tensor], config: Any
        ) -> Dict[str, torch.Tensor]:
            return convert_qwen3_6_fp8_hf_to_neuron_state_dict(state_dict, config)

    return NeuronQwen3_6_27B_FP8_ForCausalLM


__all__ = [
    "convert_qwen3_6_fp8_hf_to_neuron_state_dict",
    "dequantize_fp8_block_scaled",
    "build_fp8_for_causal_lm_class",
]
