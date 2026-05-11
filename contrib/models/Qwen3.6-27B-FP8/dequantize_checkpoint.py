#!/usr/bin/env python3
"""
Dequantize the downloaded Qwen3.6-27B-FP8 checkpoint to bf16 on disk so
we can compile via PR #140's BF16 path (for sanity-checking the compile
independently of our FP8 shim).

Usage:
    source /opt/aws_neuronx_venv_pytorch_2_9_nxd_inference/bin/activate
    python dequantize_checkpoint.py \
        --src /home/ubuntu/models/Qwen3.6-27B-FP8 \
        --dst /home/ubuntu/models/Qwen3.6-27B-BF16

The output directory is drop-in compatible with PR #140's
NeuronQwen35ForCausalLM -- config.json has its quantization_config
stripped and all weights are bf16.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def dequantize_block_scaled(
    w: torch.Tensor, scale: torch.Tensor, block: tuple[int, int]
) -> torch.Tensor:
    br, bc = block
    expanded = scale.repeat_interleave(br, dim=0).repeat_interleave(bc, dim=1)
    expanded = expanded[: w.shape[0], : w.shape[1]]
    return (w.to(torch.float32) * expanded.to(torch.float32)).to(torch.bfloat16)


def dequantize_dir(src: str, dst: str, block: tuple[int, int],
                   modules_to_not_convert: set[str]) -> None:
    os.makedirs(dst, exist_ok=True)
    # Copy non-safetensors metadata (config.json, tokenizer, etc).
    for name in os.listdir(src):
        if name.endswith(".safetensors") or name.startswith("."):
            continue
        srcp = os.path.join(src, name)
        dstp = os.path.join(dst, name)
        if os.path.isdir(srcp):
            continue
        shutil.copy2(srcp, dstp)

    # Rewrite config.json with quantization_config stripped.
    cfg_path = os.path.join(dst, "config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)
    cfg.pop("quantization_config", None)
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)

    total_files = sum(1 for n in os.listdir(src) if n.endswith(".safetensors"))
    done = 0
    t0 = time.time()
    for name in sorted(os.listdir(src)):
        if not name.endswith(".safetensors"):
            continue
        srcp = os.path.join(src, name)
        dstp = os.path.join(dst, name)
        out: dict[str, torch.Tensor] = {}
        with safe_open(srcp, framework="pt") as f:
            keys = list(f.keys())
            scale_keys = {k for k in keys if k.endswith("_scale_inv")}
            weight_keys = {k.removesuffix("_scale_inv") for k in scale_keys}
            for k in keys:
                if k in scale_keys:
                    continue
                t = f.get_tensor(k)
                if k in weight_keys and t.dtype == torch.float8_e4m3fn:
                    module_prefix = k.rsplit(".", 1)[0]
                    if module_prefix in modules_to_not_convert:
                        # Keep as FP8 (caller intended this); PR #140 doesn't
                        # support that so raise a clear error.
                        raise RuntimeError(
                            f"modules_to_not_convert listed {module_prefix!r} "
                            f"but its weight is FP8; can't leave FP8 for PR #140."
                        )
                    scale = f.get_tensor(k + "_scale_inv")
                    t = dequantize_block_scaled(t, scale, block)
                out[k] = t
        save_file(out, dstp)
        done += 1
        elapsed = time.time() - t0
        eta = elapsed / done * (total_files - done)
        print(f"  [{done}/{total_files}] {name} ({elapsed:.0f}s, eta {eta:.0f}s)",
              flush=True)
    print(f"Done. Output at {dst}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="HF FP8 checkpoint dir")
    parser.add_argument("--dst", required=True, help="Output bf16 checkpoint dir")
    args = parser.parse_args()

    with open(os.path.join(args.src, "config.json")) as f:
        cfg = json.load(f)
    qcfg = cfg.get("quantization_config", {})
    block = tuple(qcfg.get("weight_block_size", [128, 128]))
    skip = set(qcfg.get("modules_to_not_convert", []))
    print(f"Block size: {block}, skip {len(skip)} modules")

    dequantize_dir(args.src, args.dst, block, skip)
    return 0


if __name__ == "__main__":
    sys.exit(main())
