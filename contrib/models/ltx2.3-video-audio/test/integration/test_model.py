#!/usr/bin/env python3
"""
Integration test for LTX-2.3 video+audio diffusion on Neuron.

This test runs a minimal end-to-end LTX-2.3 pipeline (Neuron DiT + CPU
Gemma3/VAE from the LTX-2 Diffusers repo) and asserts:

  1. The pipeline completes without shape or compilation errors.
  2. The output has the expected number of frames and each frame has
     the expected (H, W, 3) RGB shape.

Unlike the LTX-2 test, this does NOT yet run an SSIM comparison against a
GPU reference — an LTX-2.3 GPU reference needs to be captured offline with
the Lightricks LTX-2 GitHub repo and committed under `samples/gpu/` before
the SSIM assertion can be enabled. That step is out of scope here.

Prerequisites (all from the LTX-2 flow, unchanged):
  - Compiled DiT backbone at $LTX23_DIT_COMPILE_DIR
      default: /home/ubuntu/ltx23_nxdi_compiled_1024/
  - Compiled Gemma3 encoder at $LTX2_GEMMA3_COMPILE_DIR
  - Pre-sharded Gemma3 weights at $LTX2_GEMMA3_SHARDED_DIR

Usage:
  NEURON_FUSE_SOFTMAX=1 NEURON_CUSTOM_SILU=1 NEURON_RT_STOCHASTIC_ROUNDING_EN=0 \
    /opt/aws_neuronx_venv_pytorch_inference_vllm_0_16/bin/python \
      test/integration/test_model.py
"""

import gc
import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import torch

os.environ.setdefault("NEURON_FUSE_SOFTMAX", "1")
os.environ.setdefault("NEURON_CUSTOM_SILU", "1")
os.environ.setdefault("NEURON_RT_STOCHASTIC_ROUNDING_EN", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

CONTRIB_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(CONTRIB_ROOT / "src"))
sys.path.insert(0, str(CONTRIB_ROOT.parent / "ltx2-video-audio" / "src"))

DIT_COMPILE_DIR = os.environ.get(
    "LTX23_DIT_COMPILE_DIR", "/home/ubuntu/ltx23_nxdi_compiled_1024/"
)
GEMMA3_COMPILE_DIR = os.environ.get(
    "LTX2_GEMMA3_COMPILE_DIR", "/home/ubuntu/gemma3_encoder_compiled_1024/"
)
GEMMA3_SHARDED_DIR = os.environ.get(
    "LTX2_GEMMA3_SHARDED_DIR", "/home/ubuntu/gemma3_encoder_sharded/"
)
VARIANT = os.environ.get("LTX23_VARIANT", "distilled")
PIPELINE_REPO = os.environ.get("LTX23_PIPELINE_REPO", "Lightricks/LTX-2")

TP_DEGREE = 4
HEIGHT, WIDTH, NUM_FRAMES = 256, 256, 25
NUM_STEPS = 8
SEED = 42
PROMPT = "A golden retriever puppy runs across a sunny green meadow."


def _prereqs_present() -> bool:
    return (
        os.path.isdir(DIT_COMPILE_DIR)
        and os.path.isdir(GEMMA3_COMPILE_DIR)
        and os.path.isdir(GEMMA3_SHARDED_DIR)
    )


@pytest.mark.skipif(
    not _prereqs_present(), reason="Compiled LTX-2.3 DiT or Gemma3 artifacts not present"
)
def test_ltx23_end_to_end_smoke():
    """Run one LTX-2.3 generation and assert the output shape is correct."""
    from modeling_ltx2_3 import (
        LTX23BackboneInferenceConfig,
        NeuronLTX23BackboneApplication,
        replace_sdpa_with_bmm,
    )
    from recover_config import recover_ltx23_config
    from pipeline import NeuronTransformerWrapper
    from neuronx_distributed_inference.models.config import NeuronConfig
    from huggingface_hub import hf_hub_download
    from diffusers import LTX2Pipeline

    replace_sdpa_with_bmm()

    filename = f"ltx-2.3-22b-{VARIANT}.safetensors"
    local_weights = hf_hub_download(repo_id="Lightricks/LTX-2.3", filename=filename)
    hf_config = recover_ltx23_config(variant=VARIANT, safetensors_path=local_weights)

    num_heads = hf_config["num_attention_heads"]
    head_dim = hf_config["attention_head_dim"]
    inner_dim = num_heads * head_dim
    audio_num_heads = hf_config["audio_num_attention_heads"]
    audio_head_dim = hf_config["audio_attention_head_dim"]
    audio_inner_dim = audio_num_heads * audio_head_dim
    audio_ca_dim = hf_config.get("audio_cross_attention_dim", audio_inner_dim)

    latent_num_frames = (NUM_FRAMES - 1) // 8 + 1
    video_seq = latent_num_frames * (HEIGHT // 32) * (WIDTH // 32)
    audio_num_frames = round((NUM_FRAMES / 24.0) * 24.97)

    backbone_nc = NeuronConfig(
        tp_degree=TP_DEGREE, world_size=TP_DEGREE, torch_dtype=torch.bfloat16
    )
    config = LTX23BackboneInferenceConfig(
        neuron_config=backbone_nc,
        num_layers=hf_config["num_layers"],
        num_attention_heads=num_heads,
        attention_head_dim=head_dim,
        inner_dim=inner_dim,
        audio_num_attention_heads=audio_num_heads,
        audio_attention_head_dim=audio_head_dim,
        audio_inner_dim=audio_inner_dim,
        audio_cross_attention_dim=audio_ca_dim,
        caption_channels=hf_config.get("caption_channels", 3840),
        video_seq=video_seq,
        audio_seq=audio_num_frames,
        text_seq=1024,
        height=HEIGHT,
        width=WIDTH,
        num_frames=NUM_FRAMES,
    )
    config.hf_config_dict = hf_config
    config.ltx23_variant = VARIANT

    pipe = LTX2Pipeline.from_pretrained(PIPELINE_REPO, torch_dtype=torch.bfloat16)
    cpu_transformer = pipe.transformer

    backbone_app = NeuronLTX23BackboneApplication(
        model_path=local_weights, config=config
    )
    backbone_app.load(DIT_COMPILE_DIR)

    wrapper = NeuronTransformerWrapper(
        compiled_backbone=backbone_app, cpu_transformer=cpu_transformer, text_seq=1024
    )
    del cpu_transformer.transformer_blocks
    del cpu_transformer.norm_out, cpu_transformer.proj_out
    del cpu_transformer.audio_norm_out, cpu_transformer.audio_proj_out
    gc.collect()
    pipe.transformer = wrapper

    generator = torch.Generator(device="cpu").manual_seed(SEED)
    t0 = time.time()
    output = pipe(
        prompt=PROMPT,
        height=HEIGHT,
        width=WIDTH,
        num_frames=NUM_FRAMES,
        num_inference_steps=NUM_STEPS,
        generator=generator,
        output_type="pil",
    )
    gen_time = time.time() - t0
    print(f"Generated in {gen_time:.1f}s")

    frames = output.frames[0]
    assert len(frames) == NUM_FRAMES, f"expected {NUM_FRAMES} frames, got {len(frames)}"
    for i, frame in enumerate(frames):
        arr = np.array(frame)
        assert arr.shape == (HEIGHT, WIDTH, 3), (
            f"frame {i} wrong shape: {arr.shape}, expected ({HEIGHT}, {WIDTH}, 3)"
        )
        assert arr.dtype == np.uint8
        assert arr.min() >= 0 and arr.max() <= 255


if __name__ == "__main__":
    test_ltx23_end_to_end_smoke()
    print("OK")
