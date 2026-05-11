#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""
End-to-end VL integration test for Qwen/Qwen3.6-27B-FP8.

Pipeline:
  1. Build the FP8 VL subclass (text on Neuron, vision on CPU).
  2. Compile the FP8 text decoder (reuses text-only compile cache if
     present at ``QWEN36_FP8_COMPILED_PATH/text_model``).
  3. Load: text from compiled NEFFs, vision from bf16 safetensors.
  4. Run generate() with a real image file.

Environment:
    QWEN36_FP8_MODEL_PATH      Path to downloaded HF weights (required)
    QWEN36_FP8_COMPILED_PATH   Path for compiled artifacts
                               (default /tmp/qwen36_fp8_vl_traced)
    QWEN36_FP8_TP_DEGREE       TP degree (default 4)
    QWEN36_FP8_SEQ_LEN         Max seq len for the text decoder (default 64)
    QWEN36_FP8_IMAGE_PATH      Path to a test image (default: downloads a
                               public Creative Commons sample)

Per PR #140 Caveat #4, the vision encoder runs on CPU (~918 ms / image
extra latency). This test is best-effort accuracy; timing is not
asserted.
"""
from __future__ import annotations

import gc
import json
import os
import sys
import time
from pathlib import Path


def _ensure_neuron_bin_on_path():
    """``torch_xla`` shells out to ``libneuronpjrt-path`` during its
    ``_aws_ec2_inf_trn_init`` hook. That binary lives in the same venv's
    ``bin/`` directory as the Python interpreter, but PATH in this
    process may not include it (e.g. when run from a shell that has not
    activated the venv). Inject it here, before any ``import torch_xla``
    can happen — so this call must come before ``import torch``.
    """
    venv_bin = os.path.dirname(sys.executable)
    current = os.environ.get("PATH", "")
    if venv_bin and venv_bin not in current.split(os.pathsep):
        os.environ["PATH"] = venv_bin + os.pathsep + current


_ensure_neuron_bin_on_path()

import torch  # noqa: E402


def _setup_paths():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    fp8_src = os.path.abspath(os.path.join(this_dir, "..", "src"))
    pr140 = os.path.abspath(os.path.join(this_dir, "..", "..", "Qwen3.6-27B"))
    for p in (fp8_src, pr140):
        if p not in sys.path:
            sys.path.insert(0, p)


def _load_shim():
    _setup_paths()
    from modeling_qwen3_6_vl_fp8 import build_vl_fp8_class
    return build_vl_fp8_class()


def _make_vl_config(model_path: str, tp_degree: int, seq_len: int):
    from neuronx_distributed_inference.models.config import (
        NeuronConfig,
        OnDeviceSamplingConfig,
    )
    from src.modeling_qwen35 import Qwen35InferenceConfig  # type: ignore
    from src.modeling_qwen35_vl import Qwen35VLInferenceConfig  # type: ignore

    with open(os.path.join(model_path, "config.json")) as f:
        full_cfg = json.load(f)
    text_cfg = full_cfg.get("text_config", full_cfg)
    vision_cfg = full_cfg.get("vision_config", {})
    quantization_cfg = full_cfg.get("quantization_config")

    cfg_dict = dict(text_cfg)
    cfg_dict["pad_token_id"] = text_cfg.get("eos_token_id", 248044)
    if "rope_parameters" in text_cfg:
        cfg_dict["rope_theta"] = text_cfg["rope_parameters"].get("rope_theta", 1e7)
    cfg_dict.setdefault("tie_word_embeddings", False)

    neuron_config = NeuronConfig(
        tp_degree=tp_degree,
        batch_size=1,
        ctx_batch_size=1,
        tkg_batch_size=1,
        seq_len=seq_len,
        torch_dtype=torch.bfloat16,
        on_device_sampling_config=OnDeviceSamplingConfig(top_k=1),
        enable_bucketing=False,
        flash_decoding_enabled=False,
        logical_nc_config=2,
        save_sharded_checkpoint=True,
    )
    text_inf_cfg = Qwen35InferenceConfig(neuron_config=neuron_config, **cfg_dict)
    if quantization_cfg is not None:
        text_inf_cfg.quantization_config = quantization_cfg

    vl_inf_cfg = Qwen35VLInferenceConfig(
        text_config=text_inf_cfg,
        vision_config=vision_cfg,
        image_token_id=full_cfg.get("image_token_id", 248056),
        video_token_id=full_cfg.get("video_token_id", 248057),
        vision_start_token_id=full_cfg.get("vision_start_token_id", 248053),
        vision_end_token_id=full_cfg.get("vision_end_token_id", 248054),
        spatial_merge_size=vision_cfg.get("spatial_merge_size", 2),
    )
    return text_inf_cfg, vl_inf_cfg


def _compile_and_load(
    model_path: str, compiled_path: str, tp_degree: int, seq_len: int
):
    vl_cls = _load_shim()
    text_cfg, vl_cfg = _make_vl_config(model_path, tp_degree, seq_len)
    os.makedirs(compiled_path, exist_ok=True)

    model = vl_cls(
        model_path=model_path,
        text_config=text_cfg,
        vision_config=vl_cfg,
    )

    text_path = os.path.join(compiled_path, "text_model")
    neff = os.path.join(text_path, "model.pt")
    if not os.path.exists(neff):
        print(f"Compiling text decoder to {text_path}...", flush=True)
        t0 = time.time()
        model.compile(compiled_path)
        print(f"  Compile took {time.time()-t0:.1f}s", flush=True)

    print(f"Loading VL model from {compiled_path} (vision on CPU)...", flush=True)
    t0 = time.time()
    model.load(compiled_path)
    print(f"  Load took {time.time()-t0:.1f}s", flush=True)
    return model


def _shrink_image_if_needed(image_path: str, seq_len: int) -> str:
    """Resize the image so the full chat-templated prompt fits in ``seq_len``.

    Qwen3.6's image processor emits one patch per ``patch_size * patch_size``
    pixel region, then merges ``spatial_merge_size**2`` patches into one
    vision token. Chat boilerplate adds ~20-30 tokens; we budget
    conservatively and leave ``seq_len - 64`` vision tokens for the ViT.

    If the image is already small enough, return the original path.
    Otherwise, save a downscaled copy and return that.
    """
    from PIL import Image
    patch = 16         # Qwen3.6 patch_size
    merge = 2          # spatial_merge_size
    budget_vision_tokens = max(16, seq_len - 64)
    max_pixels = budget_vision_tokens * (patch * merge) ** 2

    with Image.open(image_path) as im:
        im = im.convert("RGB")
        w, h = im.size
        current_pixels = w * h
        if current_pixels <= max_pixels:
            return image_path
        scale = (max_pixels / current_pixels) ** 0.5
        new_w = max(patch * merge, int(w * scale) // (patch * merge) * (patch * merge))
        new_h = max(patch * merge, int(h * scale) // (patch * merge) * (patch * merge))
        im_resized = im.resize((new_w, new_h), Image.LANCZOS)
        dest = os.path.join(
            "/tmp", f"qwen36_vl_resized_{new_w}x{new_h}.jpg"
        )
        im_resized.save(dest, "JPEG", quality=92)
        print(
            f"  resized {w}x{h} ({current_pixels} px) -> {new_w}x{new_h} "
            f"({new_w*new_h} px) to fit seq_len={seq_len}",
            flush=True,
        )
        return dest


def _resolve_image_path() -> str:
    """Return a local image path.

    If ``QWEN36_FP8_IMAGE_PATH`` points at an existing file, use that.
    Otherwise, synthesise a small local test image (a red circle on a
    white background, labelled "CAT" in bold) so the test runs without
    any network fetches.
    """
    env = os.environ.get("QWEN36_FP8_IMAGE_PATH")
    if env and os.path.isfile(env):
        return env
    dest = os.path.join("/tmp", "qwen36_vl_test_image.jpg")
    if not os.path.exists(dest):
        from PIL import Image, ImageDraw, ImageFont
        img = Image.new("RGB", (224, 224), "white")
        d = ImageDraw.Draw(img)
        d.ellipse((40, 40, 184, 184), fill="red", outline="black", width=3)
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 48
            )
        except Exception:
            font = ImageFont.load_default()
        d.text((80, 90), "CAT", fill="white", font=font)
        img.save(dest, "JPEG", quality=95)
        print(f"Generated synthetic test image at {dest}", flush=True)
    return dest


def main():
    model_path = os.environ.get("QWEN36_FP8_MODEL_PATH", "")
    if not model_path or not os.path.isdir(model_path):
        print(f"QWEN36_FP8_MODEL_PATH not set or invalid: {model_path!r}")
        return 1
    compiled_path = os.environ.get(
        "QWEN36_FP8_COMPILED_PATH", "/tmp/qwen36_fp8_vl_traced"
    )
    tp_degree = int(os.environ.get("QWEN36_FP8_TP_DEGREE", "4"))
    seq_len = int(os.environ.get("QWEN36_FP8_SEQ_LEN", "64"))

    print(f"Model path:  {model_path}")
    print(f"Compiled to: {compiled_path}")
    print(f"TP / seq_len: {tp_degree} / {seq_len}")

    model = _compile_and_load(model_path, compiled_path, tp_degree, seq_len)

    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(model_path)

    image_path = _resolve_image_path()
    print(f"\nImage: {image_path}")

    # Qwen3.6's image processor does NOT auto-cap pixel count (its
    # longest_edge is 2^24). A 756x1032 RGBA PNG produces 3072 patches
    # => 768 vision tokens after spatial merge, which already blows our
    # seq_len=256. We downscale to ~200 vision tokens worth of pixels so
    # the chat-templated prompt fits comfortably inside the compiled
    # bucket, and we cache the resized image to a temp path.
    image_path = _shrink_image_if_needed(image_path, seq_len)
    print(f"  using image: {image_path}")

    # Use PR #140's prepare_input_args to build the multimodal input.
    from src.modeling_qwen35_vl import NeuronQwen35VLForCausalLM  # type: ignore
    input_ids, attention_mask, vision_inputs = (
        NeuronQwen35VLForCausalLM.prepare_input_args(
            text_prompt="What is shown in this image? Answer briefly.",
            image_path=image_path,
            processor=processor,
        )
    )
    print(f"  input_ids.shape: {tuple(input_ids.shape)}")
    print(f"  pixel_values.shape: {tuple(vision_inputs.get('pixel_values').shape) if 'pixel_values' in vision_inputs else None}")

    if input_ids.shape[1] > seq_len:
        raise RuntimeError(
            f"After shrinking, input_ids length {input_ids.shape[1]} still "
            f"exceeds the compiled seq_len {seq_len}. Either recompile with a "
            f"larger seq_len (e.g. 512) or provide a smaller image."
        )

    print("\nGenerating...")
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            input_ids,
            attention_mask=attention_mask,
            pixel_values=vision_inputs.get("pixel_values"),
            image_grid_thw=vision_inputs.get("image_grid_thw"),
            video_grid_thw=vision_inputs.get("video_grid_thw"),
            max_new_tokens=60,
            temperature=0.0,
        )
    elapsed = time.time() - t0

    # Decode only the assistant response.
    response = processor.tokenizer.decode(
        out[0, input_ids.shape[1]:], skip_special_tokens=True
    )
    new_tok = out.shape[1] - input_ids.shape[1]
    print(f"\nGenerated {new_tok} tokens in {elapsed:.2f}s ({new_tok/elapsed:.1f} tok/s)")
    print(f"Response: {response!r}")

    # Basic sanity checks:
    #   - we generated at least a couple of tokens.
    #   - the response is non-empty.
    assert new_tok >= 2, f"expected >= 2 new tokens, got {new_tok}"
    assert response.strip(), f"empty response: {response!r}"
    print("\nAll VL checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
