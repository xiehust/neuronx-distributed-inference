#!/usr/bin/env python3
"""
LTX-2.3 Video+Audio Generation on AWS Neuron
=============================================
End-to-end LTX-2.3 inference on trn2.3xlarge (TP=4, bf16).

The DiT backbone is LTX-2.3 (22B). The text encoder (Gemma 3-12B), VAE,
vocoder, and scheduler are reused from `Lightricks/LTX-2` — they are
architecturally identical to LTX-2.3 and Lightricks has not yet published
a Diffusers-layout copy for LTX-2.3.

Usage:
  /opt/aws_neuronx_venv_pytorch_inference_vllm_0_16/bin/python generate_ltx2_3.py \
      --prompt "A golden retriever runs across a meadow"
  /opt/aws_neuronx_venv_pytorch_inference_vllm_0_16/bin/python generate_ltx2_3.py --compile-only
  /opt/aws_neuronx_venv_pytorch_inference_vllm_0_16/bin/python generate_ltx2_3.py --load-only
"""

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from application import NeuronLTX23Application
from modeling_ltx2_3 import LTX23BackboneInferenceConfig, replace_sdpa_with_bmm
from recover_config import recover_ltx23_config

try:
    from neuronx_distributed_inference.models.config import NeuronConfig
except ImportError:
    raise ImportError(
        "neuronx_distributed_inference is required. "
        "Activate: source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_16/bin/activate"
    )


DEFAULT_CONFIG = {
    "hf_repo": "Lightricks/LTX-2.3",
    "pipeline_repo": "Lightricks/LTX-2",
    "variant": "distilled",
    "height": 384,
    "width": 512,
    "num_frames": 25,
    "num_inference_steps": 8,
    "tp_degree": 4,
    "world_size": 4,
    "compile_dir": "/tmp/ltx23_nxdi/compiler_workdir/",
    "output_dir": "/tmp/ltx23_nxdi/output/",
    "prompt": (
        "A golden retriever puppy runs across a sunny green meadow, its ears "
        "flapping in the wind. The camera follows from a low angle. Birds chirp."
    ),
    "seed": 42,
    "max_sequence_length": 1024,
    "frame_rate": 24.0,
}


def create_ltx23_config(args, weights_path):
    latent_num_frames = (args.num_frames - 1) // 8 + 1
    latent_height = args.height // 32
    latent_width = args.width // 32
    video_seq = latent_num_frames * latent_height * latent_width
    audio_num_frames = round((args.num_frames / args.frame_rate) * 24.97)

    hf_config = recover_ltx23_config(variant=args.variant, safetensors_path=weights_path)

    num_heads = hf_config["num_attention_heads"]
    head_dim = hf_config["attention_head_dim"]
    inner_dim = num_heads * head_dim
    audio_num_heads = hf_config["audio_num_attention_heads"]
    audio_head_dim = hf_config["audio_attention_head_dim"]
    audio_inner_dim = audio_num_heads * audio_head_dim
    audio_ca_dim = hf_config.get("audio_cross_attention_dim", audio_inner_dim)

    backbone_neuron_config = NeuronConfig(
        tp_degree=args.tp_degree,
        world_size=args.world_size,
        torch_dtype=torch.bfloat16,
    )

    backbone_config = LTX23BackboneInferenceConfig(
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
        text_seq=args.max_sequence_length,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
    )
    backbone_config.hf_config_dict = hf_config
    backbone_config.ltx23_variant = args.variant

    print(
        f"  Config: {hf_config['num_layers']} layers, {num_heads} heads, "
        f"head_dim={head_dim}, inner_dim={inner_dim}"
    )
    print(
        f"  Audio: {audio_num_heads} heads, audio_head_dim={audio_head_dim}, "
        f"audio_inner_dim={audio_inner_dim}"
    )
    print(
        f"  Video: {args.height}x{args.width}, {args.num_frames} frames → "
        f"latent {latent_height}x{latent_width}x{latent_num_frames} = {video_seq} tokens"
    )
    print(f"  Audio: {audio_num_frames} tokens")
    print(f"  TP={args.tp_degree}, world_size={args.world_size}")
    return backbone_config


def _resolve_weights(args):
    """Return a local path to the LTX-2.3 safetensors file."""
    filename = f"ltx-2.3-22b-{args.variant}.safetensors"
    if args.weights_path:
        if os.path.isdir(args.weights_path):
            path = os.path.join(args.weights_path, filename)
        else:
            path = args.weights_path
        if not os.path.isfile(path):
            raise FileNotFoundError(f"LTX-2.3 weights not found at {path}")
        return path
    from huggingface_hub import hf_hub_download

    print(f"  Downloading {filename} from {args.hf_repo}...")
    return hf_hub_download(repo_id=args.hf_repo, filename=filename)


def run_generate(args):
    print("=" * 60)
    print("LTX-2.3 Video+Audio Generation on Neuron")
    print("=" * 60)

    t_total = time.time()
    replace_sdpa_with_bmm()

    print("\n[1/4] Resolving weights and configuration...")
    weights_path = _resolve_weights(args)
    print(f"  Weights: {weights_path}")
    backbone_config = create_ltx23_config(args, weights_path)

    print("\n[2/4] Creating NeuronLTX23Application...")
    app = NeuronLTX23Application(
        model_path=args.hf_repo,
        backbone_config=backbone_config,
        transformer_path=weights_path,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.num_inference_steps,
        pipeline_repo=args.pipeline_repo,
    )

    transformer_pt = os.path.join(args.compile_dir, "transformer/model.pt")
    if args.compile_only or not os.path.exists(transformer_pt):
        print("\n[3/4] Compiling transformer backbone...")
        t0 = time.time()
        app.compile(args.compile_dir)
        print(f"  Compiled in {time.time() - t0:.1f}s")
        if args.compile_only:
            print("\nCompilation complete. Re-run with --load-only to skip.")
            return

    print("\n[3/4] Loading compiled transformer...")
    t0 = time.time()
    app.load(args.compile_dir)
    print(f"  Loaded in {time.time() - t0:.1f}s")

    print("\n[4/4] Generating video+audio...")
    print(f"  Prompt: {args.prompt[:80]}...")
    print(
        f"  Resolution: {args.width}x{args.height}, {args.num_frames} frames, "
        f"{args.num_inference_steps} steps"
    )

    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    t0 = time.time()
    output = app(
        prompt=args.prompt,
        generator=generator,
        output_type="pil",
        max_sequence_length=args.max_sequence_length,
    )
    gen_time = time.time() - t0
    print(f"  Generated in {gen_time:.1f}s")

    os.makedirs(args.output_dir, exist_ok=True)
    frames = output.frames[0]
    frames_dir = os.path.join(args.output_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    for i, frame in enumerate(frames):
        frame.save(os.path.join(frames_dir, f"frame_{i:04d}.png"))
    print(f"  Saved {len(frames)} frames to {frames_dir}/")

    try:
        from diffusers.utils import export_to_video

        video_path = os.path.join(args.output_dir, "output.mp4")
        export_to_video(frames, video_path, fps=int(args.frame_rate))
        print(f"  Video: {video_path}")
    except Exception as e:
        print(f"  Video export failed: {e}")

    total_time = time.time() - t_total
    print("\nSummary:")
    print(f"  Total time: {total_time:.1f}s")
    print(f"  Generation time: {gen_time:.1f}s")
    print(f"  Output frames: {len(frames)}")
    print(f"  Output dir: {args.output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LTX-2.3 Video Generation on Neuron")
    parser.add_argument("-p", "--prompt", type=str, default=DEFAULT_CONFIG["prompt"])
    parser.add_argument(
        "--variant", choices=["dev", "distilled", "distilled-1.1"],
        default=DEFAULT_CONFIG["variant"],
    )
    parser.add_argument("--height", type=int, default=DEFAULT_CONFIG["height"])
    parser.add_argument("--width", type=int, default=DEFAULT_CONFIG["width"])
    parser.add_argument("--num-frames", type=int, default=DEFAULT_CONFIG["num_frames"])
    parser.add_argument("--num-inference-steps", type=int,
                        default=DEFAULT_CONFIG["num_inference_steps"])
    parser.add_argument("--tp-degree", type=int, default=DEFAULT_CONFIG["tp_degree"])
    parser.add_argument("--world-size", type=int, default=DEFAULT_CONFIG["world_size"])
    parser.add_argument("--hf-repo", type=str, default=DEFAULT_CONFIG["hf_repo"])
    parser.add_argument("--pipeline-repo", type=str,
                        default=DEFAULT_CONFIG["pipeline_repo"],
                        help="Diffusers repo that supplies Gemma3/VAE/vocoder/scheduler")
    parser.add_argument("--weights-path", type=str, default=None,
                        help="Optional local path to ltx-2.3-*.safetensors")
    parser.add_argument("--compile-dir", type=str, default=DEFAULT_CONFIG["compile_dir"])
    parser.add_argument("--output-dir", type=str, default=DEFAULT_CONFIG["output_dir"])
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG["seed"])
    parser.add_argument("--max-sequence-length", type=int,
                        default=DEFAULT_CONFIG["max_sequence_length"])
    parser.add_argument("--frame-rate", type=float, default=DEFAULT_CONFIG["frame_rate"])
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--load-only", action="store_true")

    args = parser.parse_args()
    run_generate(args)
