#!/usr/bin/env python3
"""
LTX-2.3 E2E: Neuron Text Encoder + Neuron DiT Backbone + Neuron VAE Decoder
============================================================================
Mirror of the LTX-2 E2E script, with two differences:

  1. The DiT backbone is LTX-2.3 (22B) — loaded from a monolithic single-file
     safetensors checkpoint published at `Lightricks/LTX-2.3`.
  2. The Diffusers pipeline (text encoder / VAE / vocoder / scheduler) is
     still pulled from `Lightricks/LTX-2` because LTX-2.3 does not yet ship
     a Diffusers multi-folder layout and those components are architecturally
     identical.

The Gemma3 text encoder and VAE decoder compile/shard directories are reused
unchanged from the LTX-2 flow.

Prerequisites:
  1. LTX-2.3 DiT compiled: python ../src/compile_dit.py --variant distilled
  2. Gemma3 compiled:       python ../../ltx2-video-audio/src/compile_gemma3.py
  3. Gemma3 sharded:        python ../../ltx2-video-audio/src/shard_gemma3_weights.py
  4. VAE compiled (opt):    python ../../ltx2-video-audio/src/compile_vae.py

Usage:
  source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_16/bin/activate
  NEURON_FUSE_SOFTMAX=1 NEURON_CUSTOM_SILU=1 NEURON_RT_STOCHASTIC_ROUNDING_EN=0 \
      python neuron_e2e.py
"""

import gc
import json
import os
import sys
import time

import torch

os.environ.setdefault("NEURON_FUSE_SOFTMAX", "1")
os.environ.setdefault("NEURON_CUSTOM_SILU", "1")
os.environ.setdefault("NEURON_RT_STOCHASTIC_ROUNDING_EN", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS_DIR, "..", "src"))
# Also import from the LTX-2 package (pipeline wrapper, Gemma3/VAE helpers).
sys.path.insert(0, os.path.join(_THIS_DIR, "..", "..", "ltx2-video-audio", "src"))

from modeling_ltx2_3 import (  # noqa: E402
    LTX23BackboneInferenceConfig,
    NeuronLTX23BackboneApplication,
    replace_sdpa_with_bmm,
)
from recover_config import recover_ltx23_config  # noqa: E402
from neuronx_distributed_inference.models.config import NeuronConfig  # noqa: E402


# Directories — adjust for your instance layout
DIT_COMPILE_DIR = os.environ.get(
    "LTX23_DIT_COMPILE_DIR", "/home/ubuntu/ltx23_nxdi_compiled_1024_256x256_f25/"
)
GEMMA3_COMPILE_DIR = os.environ.get(
    "LTX2_GEMMA3_COMPILE_DIR", "/home/ubuntu/gemma3_encoder_compiled_1024/"
)
GEMMA3_SHARDED_DIR = os.environ.get(
    "LTX2_GEMMA3_SHARDED_DIR", "/home/ubuntu/gemma3_encoder_sharded/"
)
VAE_COMPILE_DIR = os.environ.get(
    "LTX2_VAE_COMPILE_DIR", "/home/ubuntu/ltx2_vae_tp4_128x512/"
)
VAE_TILE_LATENT_H = int(os.environ.get("LTX2_VAE_TILE_H", "8"))
VAE_TILE_LATENT_W = int(os.environ.get("LTX2_VAE_TILE_W", "8"))
VAE_OVERLAP_H = int(os.environ.get("LTX2_VAE_OVERLAP_H", "2"))
VAE_OVERLAP_W = int(os.environ.get("LTX2_VAE_OVERLAP_W", "2"))
OUTPUT_DIR = os.environ.get("LTX23_OUTPUT_DIR", "/home/ubuntu/ltx23_output/")
LTX23_VARIANT = os.environ.get("LTX23_VARIANT", "distilled")
LTX23_HF_REPO = "Lightricks/LTX-2.3"
PIPELINE_REPO = os.environ.get("LTX23_PIPELINE_REPO", "Lightricks/LTX-2")

TP_DEGREE = 4
HEIGHT = int(os.environ.get( "HEIGHT", "256"))
WIDTH = int(os.environ.get( "WIDTH", "256"))
NUM_FRAMES = int(os.environ.get( "NUM_FRAMES", "25"))
NUM_STEPS = 8
PROMPT = (
    "A close-up shot of a young waitress in a retro 1950s diner, her warm brown eyes "
    "meeting the camera with a gentle smile. She wears a black polka-dot dress with an "
    "elegant cream lace collar, her reddish-brown hair styled in an elaborate updo. Soft "
    "warm light illuminates her features as she says: \"Welcome to Rosie's. What can I "
    "get for you today?\""
)
SEED = 10


# -- Neuron Gemma3 Text Encoder (identical to LTX-2 flow) ---------------------
def load_neuron_gemma3(sharded_dir, compile_dir, tp_degree):
    import torch_neuronx
    from neuronx_distributed.trace.trace import (
        replace_weights,
        TensorParallelNeuronModel,
    )

    models = []
    for rank in range(tp_degree):
        t0 = time.time()
        rank_ckpt_path = os.path.join(sharded_dir, f"rank_{rank}.pt")
        ckpt = torch.load(rank_ckpt_path, weights_only=True)

        neff_path = os.path.join(compile_dir, f"tp_{rank}.pt")
        with torch_neuronx.contexts.disable_nrt_load():
            traced_model = torch.jit.load(neff_path)

        replace_weights(traced_model, ckpt)
        print(f"    [Gemma3 rank {rank}] {time.time() - t0:.1f}s")
        models.append(traced_model)
        del ckpt
        gc.collect()

    compiled = TensorParallelNeuronModel(models)
    print(f"    Gemma3: all {tp_degree} ranks loaded")
    return compiled


class NeuronTextEncoderOutput:
    def __init__(self, hidden_states):
        self.hidden_states = hidden_states


class NeuronTextEncoderWrapper:
    def __init__(self, compiled_gemma3, dtype=torch.bfloat16):
        self.compiled_model = compiled_gemma3
        self.dtype = dtype
        self._device = torch.device("cpu")
        self.config = type("Config", (), {"output_hidden_states": True})()

    def __call__(self, input_ids=None, attention_mask=None, output_hidden_states=True, **kwargs):
        with torch.no_grad():
            stacked = self.compiled_model(input_ids, attention_mask)
            num_states = stacked.shape[-1]
            hidden_states = tuple(stacked[:, :, :, i] for i in range(num_states))
        return NeuronTextEncoderOutput(hidden_states=hidden_states)

    def eval(self):
        return self

    def to(self, *args, **kwargs):
        return self

    @property
    def device(self):
        return self._device


# -- Main ---------------------------------------------------------------------
def main():
    replace_sdpa_with_bmm()

    print("=" * 60)
    print(f"LTX-2.3 ({LTX23_VARIANT}) E2E: Neuron Gemma3 + DiT + VAE (TP={TP_DEGREE})")
    print("=" * 60)
    t_total = time.time()

    use_neuron_vae = os.path.isdir(VAE_COMPILE_DIR) and os.path.exists(
        os.path.join(VAE_COMPILE_DIR, "tp_0.pt")
    )
    if use_neuron_vae:
        print(f"  Neuron VAE: {VAE_COMPILE_DIR}")
    else:
        print(f"  Neuron VAE: NOT FOUND at {VAE_COMPILE_DIR} (CPU fallback)")

    # 1. Locate LTX-2.3 weights and build DiT config
    print("\n[1/7] Resolving LTX-2.3 weights + config...")
    from huggingface_hub import hf_hub_download

    filename = f"ltx-2.3-22b-{LTX23_VARIANT}.safetensors"
    local_weights = hf_hub_download(repo_id=LTX23_HF_REPO, filename=filename)
    print(f"  Weights: {local_weights}")

    hf_config = recover_ltx23_config(variant=LTX23_VARIANT, safetensors_path=local_weights)
    num_heads = hf_config["num_attention_heads"]
    head_dim = hf_config["attention_head_dim"]
    inner_dim = num_heads * head_dim
    audio_num_heads = hf_config["audio_num_attention_heads"]
    audio_head_dim = hf_config["audio_attention_head_dim"]
    audio_inner_dim = audio_num_heads * audio_head_dim
    audio_ca_dim = hf_config.get("audio_cross_attention_dim", audio_inner_dim)

    latent_num_frames = (NUM_FRAMES - 1) // 8 + 1
    latent_height = HEIGHT // 32
    latent_width = WIDTH // 32
    video_seq = latent_num_frames * latent_height * latent_width
    audio_num_frames = round((NUM_FRAMES / 24.0) * 24.97)

    backbone_neuron_config = NeuronConfig(
        tp_degree=TP_DEGREE,
        world_size=TP_DEGREE,
        torch_dtype=torch.bfloat16,
    )

    config = LTX23BackboneInferenceConfig(
        neuron_config=backbone_neuron_config,
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
    config.ltx23_variant = LTX23_VARIANT
    print(f"  {hf_config['num_layers']} blocks, TP={TP_DEGREE}, {HEIGHT}x{WIDTH}, {NUM_FRAMES} frames")

    # 2. Load Diffusers pipeline (from LTX-2 repo for Gemma3/VAE/vocoder/scheduler)
    print(f"\n[2/7] Loading Diffusers LTX2Pipeline from {PIPELINE_REPO} (CPU)...")
    t0 = time.time()
    from diffusers import LTX2Pipeline
    from diffusers.pipelines.ltx2.connectors import LTX2TextConnectors
    from modeling_ltx2_3 import extract_connector_state_dict, LTX23_CONNECTORS_CONFIG  # noqa: E402

    pipe = LTX2Pipeline.from_pretrained(PIPELINE_REPO, torch_dtype=torch.bfloat16)
    print(f"  Loaded in {time.time() - t0:.1f}s")

    # 2a. LTX-2.0's connectors are trained for the 19B DiT and have a
    # different shape than LTX-2.3's. Replace with a freshly constructed
    # LTX2TextConnectors sized for LTX-2.3, loaded with weights extracted
    # from the LTX-2.3 checkpoint. Without this swap, text conditioning on
    # the 22B DiT uses untrained/mismatched projections and output is noise.
    print("  Replacing connectors with LTX-2.3 weights...")
    t0 = time.time()
    ltx23_connectors = LTX2TextConnectors(**LTX23_CONNECTORS_CONFIG).to(torch.bfloat16)
    connector_sd = extract_connector_state_dict(local_weights)
    ltx23_connectors.load_state_dict(connector_sd, strict=True)
    ltx23_connectors.eval()
    pipe.connectors = ltx23_connectors
    print(f"  Connectors swapped in {time.time() - t0:.1f}s")

    # 3. Load Neuron LTX-2.3 DiT backbone (block-loop only).
    #    The compiled Neuron graph runs all 48 transformer blocks in one call.
    #    We install it as pipe.transformer.transformer_blocks so diffusers'
    #    own forward() does the preprocessing (proj_in, time_embed,
    #    prompt_adaln, rope) and output heads (norm_out, proj_out) on CPU.
    print(f"\n[3/7] Loading Neuron LTX-2.3 DiT from {DIT_COMPILE_DIR}...")
    t0 = time.time()
    backbone_app = NeuronLTX23BackboneApplication(
        model_path=local_weights, config=config
    )
    backbone_app.load(DIT_COMPILE_DIR)
    print(f"  DiT loaded in {time.time() - t0:.1f}s")

    from modeling_ltx2_3 import NeuronBlockListShim  # noqa: E402

    num_layers = hf_config["num_layers"]
    pipe.transformer.transformer_blocks = NeuronBlockListShim(
        backbone_app, num_layers=num_layers
    )
    print(f"  Installed NeuronBlockListShim ({num_layers} slots; first runs Neuron, rest no-op)")

    # 4. Swap text encoder: CPU -> Neuron
    print("\n[4/7] Swapping text encoder to Neuron...")
    t0 = time.time()
    del pipe.text_encoder
    gc.collect()
    compiled_gemma3 = load_neuron_gemma3(GEMMA3_SHARDED_DIR, GEMMA3_COMPILE_DIR, TP_DEGREE)
    pipe.text_encoder = NeuronTextEncoderWrapper(compiled_gemma3)
    print(f"  Neuron text encoder loaded in {time.time() - t0:.1f}s")

    # 5. Swap VAE decoder: CPU -> Neuron (if compiled)
    if use_neuron_vae:
        print("\n[5/7] Swapping VAE decoder to Neuron...")
        t0 = time.time()
        from pipeline import NeuronTiledVAEDecoder

        original_decoder = pipe.vae.decoder
        neuron_decoder = NeuronTiledVAEDecoder(
            compiled_dir=VAE_COMPILE_DIR,
            tile_latent_h=VAE_TILE_LATENT_H,
            tile_latent_w=VAE_TILE_LATENT_W,
            overlap_latent_h=VAE_OVERLAP_H,
            overlap_latent_w=VAE_OVERLAP_W,
            original_decoder=original_decoder,
        )
        del original_decoder
        gc.collect()
        pipe.vae.decoder = neuron_decoder
        print(f"  Neuron VAE loaded in {time.time() - t0:.1f}s")
        print("  Warming up VAE...")
        neuron_decoder.warmup(num_frames=NUM_FRAMES)
        print("  VAE warmup done")
    else:
        print("\n[5/7] Skipping Neuron VAE (using CPU fallback)")

    # 6. Generate
    print("\n[6/7] Generating video+audio...")
    print(f"  Prompt: {PROMPT[:80]}...")
    print(f"  {WIDTH}x{HEIGHT}, {NUM_FRAMES} frames, {NUM_STEPS} steps")

    generator = torch.Generator(device="cpu").manual_seed(SEED)

    t0 = time.time()
    output = pipe(
        prompt=PROMPT,
        height=HEIGHT,
        width=WIDTH,
        num_frames=NUM_FRAMES,
        num_inference_steps=NUM_STEPS,
        guidance_scale=3.0,  # LTX-2.3 default (was 4.0 for LTX-2.0)
        generator=generator,
        output_type="pil",
    )
    gen_time = time.time() - t0
    print(f"  Generated in {gen_time:.1f}s")

    # 7. Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    frames = output.frames[0]
    frames_dir = os.path.join(OUTPUT_DIR, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    for i, frame in enumerate(frames):
        frame.save(os.path.join(frames_dir, f"frame_{i:04d}.png"))
    print(f"  Saved {len(frames)} frames to {frames_dir}/")

    try:
        from diffusers.utils import export_to_video

        video_path = os.path.join(OUTPUT_DIR, "output.mp4")
        export_to_video(frames, video_path, fps=24)
        print(f"  Video: {video_path}")
    except Exception as e:
        print(f"  Video export failed: {e}")

    metadata = {
        "model": f"Lightricks/LTX-2.3 ({LTX23_VARIANT})",
        "pipeline_repo": PIPELINE_REPO,
        "prompt": PROMPT,
        "resolution": f"{WIDTH}x{HEIGHT}",
        "num_frames": NUM_FRAMES,
        "num_steps": NUM_STEPS,
        "guidance_scale": 3.0,
        "max_sequence_length": 1024,
        "seed": SEED,
        "generation_time_s": gen_time,
        "total_time_s": time.time() - t_total,
        "text_encoder": f"Neuron Gemma3-12B (TP={TP_DEGREE})",
        "dit": f"Neuron LTX-2.3 DiT {hf_config['num_layers']} blocks (TP={TP_DEGREE})",
        "vae_decoder": (
            f"Neuron tiled (TP={TP_DEGREE})" if use_neuron_vae else "CPU (Diffusers default)"
        ),
    }
    with open(os.path.join(OUTPUT_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    total_time = time.time() - t_total
    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  Total time: {total_time:.1f}s")
    print(f"  Generation time: {gen_time:.1f}s")
    print(f"  Output frames: {len(frames)}")
    print(f"  DiT: Neuron LTX-2.3 {LTX23_VARIANT} ({hf_config['num_layers']} blocks)")
    print(f"  VAE: {'Neuron tiled' if use_neuron_vae else 'CPU fallback'}")
    print(f"  Output dir: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
