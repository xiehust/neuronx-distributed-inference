# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Vision-Language FP8 delta adapter for ``Qwen/Qwen3.6-27B-FP8``.

Like the text-only FP8 shim (``modeling_qwen3_6_fp8.py``), this does not
implement the architecture itself — it layers onto PR #140's
``NeuronQwen35VLForCausalLM`` with two patches:

  1. **Text decoder.** Replaced with our FP8 subclass so the block-wise
     FP8 weights are dequantized to bf16 on load.
  2. **Vision weight loader.** PR #140's
     ``NeuronQwen35VisionModelWrapper.load_cpu_model`` and
     ``load_vision_weights_from_hf`` iterate ``model*.safetensors`` via
     ``Path(model_path).glob("model*.safetensors")``. The published
     ``Qwen/Qwen3.6-27B-FP8`` checkpoint uses per-layer sharding
     (``layers-N.safetensors``) with vision weights in a separate
     ``outside.safetensors`` file. We patch the glob to include
     ``outside.safetensors`` so vision weights are actually loaded.

Vision tower weights in the FP8 checkpoint are **already bf16** (the HF
``quantization_config.modules_to_not_convert`` list covers every
``visual.blocks.*`` module), so no dequantization is needed for the ViT.

Per PR #140's Caveat #4, the vision encoder runs on CPU at TP=4 on
trn2.3xlarge (HBM is full with the text decoder). Expect ~900 ms per
image on top of the text generation time.
"""
from __future__ import annotations

import importlib
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)


def _import_base_vl_adapter():
    """Resolve PR #140's ``NeuronQwen35VLForCausalLM`` etc.

    Mirrors the text-only shim's import strategy. The ``src`` package name
    collides between our contrib and PR #140, so we clear any cached
    entry **only once** and then cache the resolved modules so subsequent
    calls don't undo patches installed by callers.
    """
    # If both modules are already loaded and they're PR #140's (pointing at
    # the file we expect), short-circuit and return them as-is. This is
    # important: ``patch_vision_wrapper_for_fp8_checkpoint`` mutates the
    # class on the imported module; re-importing fresh would drop the patch.
    already_vl = sys.modules.get("src.modeling_qwen35_vl")
    already_vis = sys.modules.get("src.modeling_qwen35_vision")
    if (already_vl is not None and already_vis is not None
            and "Qwen3.6-27B/src" in (getattr(already_vl, "__file__", "") or "")
            and "Qwen3.6-27B/src" in (getattr(already_vis, "__file__", "") or "")):
        return already_vl, already_vis

    this_dir = os.path.dirname(os.path.abspath(__file__))
    sibling = os.path.abspath(os.path.join(this_dir, "..", "..", "Qwen3.6-27B"))
    if not os.path.isdir(os.path.join(sibling, "src")):
        raise ImportError(
            "Base Qwen3.6-27B adapter (PR #140) not found at "
            f"{sibling}. Either cherry-pick PR #140 or add its contrib "
            "dir to PYTHONPATH."
        )
    if sibling not in sys.path:
        sys.path.insert(0, sibling)
    for name in list(sys.modules):
        if name == "src" or name.startswith("src."):
            del sys.modules[name]
    importlib.invalidate_caches()
    return (
        importlib.import_module("src.modeling_qwen35_vl"),
        importlib.import_module("src.modeling_qwen35_vision"),
    )


# ---------------------------------------------------------------------------
# Vision weight loader patch: scan the FP8 checkpoint's layout.
# ---------------------------------------------------------------------------


def _fp8_checkpoint_safetensors(model_path: str) -> list[Path]:
    """Return every safetensors file that might contain vision or scalar weights.

    PR #140 iterates ``model*.safetensors``; the FP8 checkpoint uses
    ``layers-N.safetensors`` + ``outside.safetensors`` + ``mtp.safetensors``.
    We simply grab everything ending in ``.safetensors`` under the model
    directory, which is safe because the loaders only *copy* keys that
    match their key map — foreign keys are ignored.
    """
    return sorted(Path(model_path).glob("*.safetensors"))


def _extract_on_device_sampled_tokens(output) -> "torch.Tensor":
    """Pull the [B, 1] next-token tensor out of whatever the traced
    model returned.

    NxDI's ``HuggingFaceGenerationAdapter`` wraps model output in a
    ``CausalLMOutputWithPast`` whose ``.logits`` attribute actually
    contains:
      * raw logits ``[B, S, V]`` when on-device sampling is off, or
      * sampled token ids ``[B, S]`` / ``[B, 1]`` when it is on.
    We distinguish by the last-dim size: vocab is ~248k so any tensor
    with last-dim > 1024 is treated as logits.
    """
    import torch as _t
    # NxDI's CausalLMOutputWithPast has BOTH ``logits`` and ``tokens``
    # attributes; whichever of the two is populated depends on whether
    # on-device sampling was compiled in. Check in priority: tokens (ids
    # already sampled) > logits (ids still to be sampled) > sequences
    # (rare, belt-and-braces).
    head = None
    if isinstance(output, tuple):
        head = output[0]
    else:
        for attr in ("tokens", "logits", "sequences"):
            val = getattr(output, attr, None)
            if val is not None:
                head = val
                break
    if head is None:
        raise RuntimeError(
            "Text model returned no usable head tensor (expected sampled "
            f"tokens or logits). Got: {type(output).__name__} "
            f"with fields {getattr(output, '__dict__', None) and list(output.__dict__.keys())}"
        )
    # Logits: last dim is vocab (~248k for Qwen3.6).
    if head.ndim >= 3 and head.shape[-1] > 1024:
        return head[:, -1:, :].argmax(dim=-1).to(_t.int64)
    # Sampled tokens: reshape to [B, 1].
    if head.ndim == 1:
        head = head.unsqueeze(-1)
    if head.ndim >= 3:
        head = head.reshape(head.shape[0], -1)[:, -1:]
    elif head.shape[-1] != 1:
        head = head[:, -1:]
    return head.to(_t.int64)


def patch_vision_wrapper_for_fp8_checkpoint(vision_module) -> None:
    """Monkey-patch ``NeuronQwen35VisionModelWrapper`` loaders to handle
    the FP8 checkpoint's per-layer sharding.

    Called once at import time. Idempotent.
    """
    cls = vision_module.NeuronQwen35VisionModelWrapper
    if getattr(cls, "_fp8_patched", False):
        return

    orig_load_cpu = cls.load_cpu_model
    orig_load_weights = cls.load_vision_weights_from_hf

    def patched_load_cpu_model(self, model_path: str):
        """Same as the original, but scans every *.safetensors (not just model*)."""
        from safetensors import safe_open
        import torch as _torch

        config = self.vision_config
        cpu_model = vision_module.CPUVisionModel(config)

        key_map: Dict[str, str] = {}
        for i in range(config.depth):
            hf_pre = f"model.visual.blocks.{i}"
            loc_pre = f"blocks.{i}"
            for suffix in (
                "attn.qkv.weight", "attn.qkv.bias",
                "attn.proj.weight", "attn.proj.bias",
                "mlp.linear_fc1.weight", "mlp.linear_fc1.bias",
                "mlp.linear_fc2.weight", "mlp.linear_fc2.bias",
                "norm1.weight", "norm1.bias",
                "norm2.weight", "norm2.bias",
            ):
                key_map[f"{hf_pre}.{suffix}"] = f"{loc_pre}.{suffix}"
        for suffix in (
            "norm.weight", "norm.bias",
            "linear_fc1.weight", "linear_fc1.bias",
            "linear_fc2.weight", "linear_fc2.bias",
        ):
            key_map[f"model.visual.merger.{suffix}"] = f"merger_{suffix.replace('.', '_')}".replace(
                "norm_weight", "norm.weight"
            ).replace("norm_bias", "norm.bias").replace(
                "linear_fc1_weight", "fc1.weight"
            ).replace("linear_fc1_bias", "fc1.bias").replace(
                "linear_fc2_weight", "fc2.weight"
            ).replace("linear_fc2_bias", "fc2.bias")
        # The original uses these exact keys; reuse its map verbatim
        # to avoid drift.
        key_map["model.visual.merger.norm.weight"] = "merger_norm.weight"
        key_map["model.visual.merger.norm.bias"] = "merger_norm.bias"
        key_map["model.visual.merger.linear_fc1.weight"] = "merger_fc1.weight"
        key_map["model.visual.merger.linear_fc1.bias"] = "merger_fc1.bias"
        key_map["model.visual.merger.linear_fc2.weight"] = "merger_fc2.weight"
        key_map["model.visual.merger.linear_fc2.bias"] = "merger_fc2.bias"

        state_dict = cpu_model.state_dict()
        loaded = 0
        for sf_path in _fp8_checkpoint_safetensors(model_path):
            with safe_open(str(sf_path), framework="pt") as f:
                for key in f.keys():
                    if key in key_map:
                        local_key = key_map[key]
                        if local_key in state_dict:
                            state_dict[local_key].copy_(f.get_tensor(key))
                            loaded += 1

        cpu_model.load_state_dict(state_dict)
        cpu_model = cpu_model.to(_torch.bfloat16).eval()
        self._cpu_model = cpu_model
        logger.info(
            "Loaded CPU vision encoder (FP8 layout): %d weights, %.1fM params",
            loaded,
            sum(p.numel() for p in cpu_model.parameters()) / 1e6,
        )
        if loaded == 0:
            raise RuntimeError(
                f"FP8 vision loader found no vision weights in {model_path}. "
                "Expected keys like 'model.visual.blocks.N.*' inside "
                "outside.safetensors."
            )

    def patched_load_vision_weights_from_hf(self, model_path: str):
        """Load patch_embed + pos_embed from any .safetensors in the dir."""
        from safetensors import safe_open

        loaded = 0
        for sf_path in _fp8_checkpoint_safetensors(model_path):
            with safe_open(str(sf_path), framework="pt") as f:
                for key in f.keys():
                    if key == "model.visual.patch_embed.proj.weight":
                        self.patch_embed.proj.weight.data.copy_(f.get_tensor(key))
                        loaded += 1
                    elif key == "model.visual.patch_embed.proj.bias":
                        self.patch_embed.proj.bias.data.copy_(f.get_tensor(key))
                        loaded += 1
                    elif key == "model.visual.pos_embed.weight":
                        self.pos_embed.weight.data.copy_(f.get_tensor(key))
                        loaded += 1
        logger.info("Loaded %d CPU-side vision weights (FP8 layout)", loaded)
        if loaded == 0:
            logger.warning(
                "FP8 vision-weight loader found no patch_embed / pos_embed in %s",
                model_path,
            )

    cls.load_cpu_model = patched_load_cpu_model
    cls.load_vision_weights_from_hf = patched_load_vision_weights_from_hf
    cls._fp8_patched = True


# ---------------------------------------------------------------------------
# Factory: build a VL subclass with the FP8 text decoder.
# ---------------------------------------------------------------------------


def build_vl_fp8_class():
    """Return a ``NeuronQwen35VLForCausalLM`` subclass whose ``text_model``
    is the FP8 text subclass built by ``modeling_qwen3_6_fp8``.

    The returned class also runs ``patch_vision_wrapper_for_fp8_checkpoint``
    on import so the CPU vision loader handles this checkpoint's sharding.
    """
    vl_module, vision_module = _import_base_vl_adapter()
    base_vl_cls = vl_module.NeuronQwen35VLForCausalLM
    patch_vision_wrapper_for_fp8_checkpoint(vision_module)

    # Import our FP8 text factory. This lives alongside us in the same
    # ``src`` directory, so an absolute path is safer than a relative one.
    this_dir = os.path.dirname(os.path.abspath(__file__))
    if this_dir not in sys.path:
        sys.path.insert(0, this_dir)
    from modeling_qwen3_6_fp8 import build_fp8_for_causal_lm_class
    fp8_text_cls = build_fp8_for_causal_lm_class()

    class NeuronQwen3_6_27B_FP8_VLForCausalLM(base_vl_cls):
        """PR #140's VL model with FP8 text decoder + FP8 vision loader.

        Key overrides:
          * ``__init__`` swaps in the FP8 text model class.
          * ``load`` uses the CPU vision encoder path (per PR #140's
            Caveat #4, HBM on trn2.3xlarge is consumed by the text
            decoder; the ViT has to run on CPU).
        """

        def __init__(self, model_path, text_config, vision_config=None, processor=None):
            # Short-circuit the base __init__ to avoid building a bf16-only
            # text model. We mirror it manually so we can slot in our FP8
            # text model class.
            self.model_path = model_path
            self.text_config = text_config
            self.vl_config = vision_config
            self.processor = processor

            # FP8 text decoder (dequantizes on load).
            self.text_model = fp8_text_cls(model_path=model_path, config=text_config)

            self.vision_model_wrapper = None
            if vision_config is not None:
                self._init_vision_model(vision_config)

            self.rope_deltas = None

        def generate(
            self,
            input_ids,
            attention_mask=None,
            pixel_values=None,
            image_grid_thw=None,
            video_grid_thw=None,
            max_new_tokens=32,
            temperature=0.0,
            top_p=1.0,
            top_k=0,
            eos_token_ids=None,
            **kwargs,
        ):
            """VL generate that works with on-device sampling.

            PR #140's base ``generate()`` assumes ``text_model(...)`` returns
            raw logits of shape ``[B, S, V]`` and does its own sampling.
            When ``on_device_sampling_config`` is set (which is how PR #140
            itself compiles the text-only model), the traced text model
            instead returns **already-sampled token ids** as ``output[0]``,
            and the legacy path tries to do ``output[0][:, -1, :]`` which
            fails.

            We detect this by checking whether the text model has
            on-device sampling enabled, and consume the returned token id
            directly. Everything else (mRoPE, vision scatter) reuses the
            base implementation.
            """
            import torch as _t
            has_ods = self.text_model.neuron_config.on_device_sampling_config is not None
            if not has_ods:
                return super().generate(
                    input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    video_grid_thw=video_grid_thw,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    eos_token_ids=eos_token_ids,
                    **kwargs,
                )

            if eos_token_ids is None:
                eos_token_ids = self._DEFAULT_EOS_TOKEN_IDS

            self.text_model.reset()
            has_vision = pixel_values is not None and pixel_values.numel() > 0

            # mRoPE / vision scatter — duplicate of base-class logic up to
            # and including the CTE call, with logits handling replaced.
            from src.modeling_qwen35_vl import get_rope_index  # type: ignore
            if has_vision and self.vl_config is not None:
                position_ids, self.rope_deltas = get_rope_index(
                    input_ids,
                    image_grid_thw=image_grid_thw,
                    video_grid_thw=video_grid_thw,
                    attention_mask=attention_mask,
                    image_token_id=self.vl_config.image_token_id,
                    video_token_id=self.vl_config.video_token_id,
                    vision_start_token_id=self.vl_config.vision_start_token_id,
                    spatial_merge_size=self.vl_config.spatial_merge_size,
                )
            else:
                seq_len = input_ids.shape[1]
                position_ids = _t.arange(seq_len).unsqueeze(0)
                self.rope_deltas = None

            llava_args = []
            batch_size = input_ids.shape[0]
            if has_vision and self.vision_model_wrapper is not None:
                vision_embeddings = self.vision_model_wrapper(
                    pixel_values, image_grid_thw
                )
                image_token_id = self.vl_config.image_token_id
                video_token_id = self.vl_config.video_token_id
                vision_bool_mask = (input_ids == image_token_id) | (
                    input_ids == video_token_id
                )
                positions = vision_bool_mask[0].nonzero(as_tuple=False).squeeze(-1)
                n_vis = positions.shape[0]
                hidden_size = vision_embeddings.shape[-1]
                vis_emb = vision_embeddings[:n_vis].unsqueeze(0)
                seq_len = input_ids.shape[1]
                pad_limit = seq_len
                if n_vis < pad_limit:
                    pad_emb = _t.zeros(
                        (1, pad_limit - n_vis, hidden_size), dtype=vis_emb.dtype
                    )
                    vis_emb_padded = _t.cat([vis_emb, pad_emb], dim=1)
                else:
                    vis_emb_padded = vis_emb[:, :pad_limit]
                positions_padded = _t.full(
                    (1, pad_limit, 1),
                    fill_value=pad_limit - 1,
                    dtype=_t.int32,
                )
                positions_padded[0, :n_vis, 0] = positions[:pad_limit].to(_t.int32)
                llava_args = [vis_emb_padded, positions_padded]
                if position_ids.ndim == 3:
                    mrope_pos = position_ids[:, :, :seq_len].to(_t.int32).contiguous()
                    llava_args.append(mrope_pos)

            if attention_mask is None:
                attention_mask = _t.ones_like(input_ids)

            cte_seq_len = input_ids.shape[1]
            cte_position_ids = _t.arange(cte_seq_len, dtype=_t.long).unsqueeze(0)

            generated_ids = input_ids.clone()

            # CTE: on-device sampler returns a single next-token tensor as
            # ``output[0]`` with shape [B, 1].
            with _t.no_grad():
                output = self.text_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=cte_position_ids,
                    output_attentions=False,
                    output_hidden_states=False,
                    return_dict=False,
                    llava_args=llava_args,
                )
            next_token_ids = _extract_on_device_sampled_tokens(output)
            generated_ids = _t.cat([generated_ids, next_token_ids], dim=-1)
            if next_token_ids.squeeze().item() in eos_token_ids:
                return generated_ids

            # TKG loop.
            for _ in range(max_new_tokens - 1):
                pos_ids = _t.tensor([[generated_ids.shape[1] - 1]])
                if self.rope_deltas is not None:
                    pos_ids = pos_ids + self.rope_deltas
                last_token = generated_ids[:, -1:]
                with _t.no_grad():
                    output = self.text_model(
                        input_ids=last_token,
                        position_ids=pos_ids,
                        output_attentions=False,
                        output_hidden_states=False,
                        return_dict=False,
                    )
                next_token_ids = _extract_on_device_sampled_tokens(output)
                generated_ids = _t.cat([generated_ids, next_token_ids], dim=-1)
                if next_token_ids.squeeze().item() in eos_token_ids:
                    break

            return generated_ids

        def load(self, compiled_model_path, vision_compiled_path=None):
            """Load text (compiled) + vision (CPU-only) per PR #140 Caveat #4.

            PR #140's base ``load()`` tries ``load_compiled()`` for the
            vision encoder, which needs a pre-compiled .pt file from a
            separate ``torch_neuronx.trace`` step. On trn2.3xlarge there
            is no HBM left for a compiled ViT, so we use the pure-PyTorch
            CPU vision path instead.
            """
            text_path = os.path.join(compiled_model_path, "text_model")
            if os.path.exists(text_path):
                self.text_model.load(text_path)
            else:
                self.text_model.load(compiled_model_path)

            if self.vision_model_wrapper is not None:
                # Our patched load_cpu_model handles the FP8 checkpoint's
                # layers-N.safetensors + outside.safetensors layout.
                self.vision_model_wrapper.load_cpu_model(self.model_path)
                # Also load the CPU-side patch_embed / pos_embed.
                self.vision_model_wrapper.load_vision_weights_from_hf(self.model_path)
                logger.info(
                    "VL FP8 model loaded (text on Neuron, vision on CPU)"
                )

    return NeuronQwen3_6_27B_FP8_VLForCausalLM


__all__ = [
    "build_vl_fp8_class",
    "patch_vision_wrapper_for_fp8_checkpoint",
]
