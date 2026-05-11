#!/usr/bin/env python3
"""
Compile LTX-2.3 DiT transformer backbone for Neuron (TP=4).

Produces compiled model files at OUTPUT_DIR that can be loaded by
NeuronLTX23BackboneApplication.load().

Usage:
  NEURON_RT_VISIBLE_CORES=0-3 \
  /opt/aws_neuronx_venv_pytorch_inference_vllm_0_16/bin/python compile_dit.py \
      [--output-dir DIR] [--variant distilled|dev|distilled-1.1] \
      [--height H] [--width W] [--num-frames F] [--text-seq S]

Default output: /home/ubuntu/ltx23_nxdi_compiled_1024/
"""

import argparse
import os
import sys
import time

import torch

os.environ.setdefault("NEURON_FUSE_SOFTMAX", "1")
os.environ.setdefault("NEURON_CUSTOM_SILU", "1")
os.environ.setdefault("NEURON_RT_STOCHASTIC_ROUNDING_EN", "0")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modeling_ltx2_3 import (
    LTX23BackboneInferenceConfig,
    NeuronLTX23BackboneApplication,
    replace_sdpa_with_bmm,
)
from recover_config import recover_ltx23_config
from neuronx_distributed_inference.models.config import NeuronConfig


_HF_REPO = "Lightricks/LTX-2.3"


def main():
    parser = argparse.ArgumentParser(description="Compile LTX-2.3 DiT backbone for Neuron")
    parser.add_argument(
        "--output-dir",
        default="/home/ubuntu/ltx23_nxdi_compiled_1024/",
    )
    parser.add_argument(
        "--variant",
        default="distilled",
        choices=["dev", "distilled", "distilled-1.1"],
        help="Which LTX-2.3 22B variant to compile (default: distilled)",
    )
    parser.add_argument("--tp-degree", type=int, default=4)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--num-frames", type=int, default=121)
    parser.add_argument("--text-seq", type=int, default=1024)
    parser.add_argument(
        "--weights-path",
        default=None,
        help=(
            "Optional local path to the ltx-2.3-*.safetensors file or directory. "
            f"If omitted, downloads from {_HF_REPO}."
        ),
    )
    args = parser.parse_args()

    replace_sdpa_with_bmm()

    print("=" * 60)
    print(
        f"Compiling LTX-2.3 ({args.variant}) DiT backbone "
        f"(TP={args.tp_degree}, {args.width}x{args.height}, {args.num_frames} frames)"
    )
    print("=" * 60)

    # [1/3] Locate weights
    print("\n[1/3] Locating LTX-2.3 weights...")
    filename = f"ltx-2.3-22b-{args.variant}.safetensors"
    if args.weights_path:
        if os.path.isdir(args.weights_path):
            local_weights = os.path.join(args.weights_path, filename)
        else:
            local_weights = args.weights_path
        if not os.path.isfile(local_weights):
            raise FileNotFoundError(f"Weights not found at {local_weights}")
    else:
        from huggingface_hub import hf_hub_download

        print(f"  Downloading {filename} from {_HF_REPO}...")
        local_weights = hf_hub_download(repo_id=_HF_REPO, filename=filename)
    print(f"  Weights: {local_weights}")

    # [2/3] Recover / build DiT config
    print("\n[2/3] Recovering DiT config...")
    hf_config = recover_ltx23_config(variant=args.variant, safetensors_path=local_weights)

    num_heads = hf_config["num_attention_heads"]
    head_dim = hf_config["attention_head_dim"]
    inner_dim = num_heads * head_dim
    audio_num_heads = hf_config["audio_num_attention_heads"]
    audio_head_dim = hf_config["audio_attention_head_dim"]
    audio_inner_dim = audio_num_heads * audio_head_dim
    audio_ca_dim = hf_config.get("audio_cross_attention_dim", audio_inner_dim)

    latent_num_frames = (args.num_frames - 1) // 8 + 1
    latent_height = args.height // 32
    latent_width = args.width // 32
    video_seq = latent_num_frames * latent_height * latent_width
    audio_num_frames = round((args.num_frames / 24.0) * 24.97)

    print(f"  Blocks: {hf_config['num_layers']}")
    print(f"  Video seq: {video_seq} ({latent_num_frames}x{latent_height}x{latent_width})")
    print(f"  Audio seq: {audio_num_frames}")
    print(f"  Text seq: {args.text_seq}")
    print(f"  Inner dim: {inner_dim}, Audio inner dim: {audio_inner_dim}")

    backbone_neuron_config = NeuronConfig(
        tp_degree=args.tp_degree,
        world_size=args.tp_degree,
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
        text_seq=args.text_seq,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
    )
    config.hf_config_dict = hf_config
    config.ltx23_variant = args.variant

    # [3/3] Compile
    print(f"\n[3/3] Compiling to {args.output_dir}...")
    os.makedirs(args.output_dir, exist_ok=True)

    backbone_app = NeuronLTX23BackboneApplication(
        model_path=local_weights, config=config
    )

    t0 = time.time()
    backbone_app.compile(args.output_dir)
    elapsed = time.time() - t0

    print(f"\n  Compiled in {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    print(f"  Output: {args.output_dir}")

    model_file = os.path.join(args.output_dir, "model.pt")
    if os.path.exists(model_file):
        size_gb = os.path.getsize(model_file) / 1e9
        print(f"  model.pt: {size_gb:.2f} GB")
    else:
        files = os.listdir(args.output_dir)
        print(f"  Files: {files[:10]}")

    print("\nDone!")


if __name__ == "__main__":
    main()
