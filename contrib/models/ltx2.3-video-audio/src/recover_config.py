#!/usr/bin/env python3
"""
Recover an LTX-2.3 DiT config dict.

`Lightricks/LTX-2.3` on HuggingFace ships only monolithic safetensors files —
there is no `transformer/config.json` or `model_index.json`. This helper
recovers the required hyperparameters by (in order):

  1. Fetching `packages/ltx-core/configs/ltx-2.3-*.{json,yaml}` from the
     Lightricks/LTX-2 GitHub repo. This is the authoritative source.
  2. Introspecting the safetensors keys and tensor shapes to infer
     num_layers / inner_dim / head count. Used when (1) is unavailable or
     fails. Requires the weights file already downloaded locally.
  3. Falling back to the LTX-2 values (48L / 32H / 128 head_dim / audio 32x64
     / caption 3840) as a last-resort starting point — only safe if the
     user has verified that 2.3 kept the same architecture.

The returned dict matches the LTX-2 HuggingFace `transformer/config.json`
schema so it can be passed directly into `LTX2VideoTransformer3DModel
.from_config(...)` and into `LTX23BackboneInferenceConfig(...)`.

Usage:
  cfg = recover_ltx23_config(safetensors_path="/path/to/ltx-2.3-...safetensors")
  cfg["num_layers"]  # -> int
"""

from __future__ import annotations

import json
import logging
import os
import re
import urllib.request
from typing import Optional

logger = logging.getLogger(__name__)


_LTX2_FALLBACK_CONFIG = {
    "num_layers": 48,
    "num_attention_heads": 32,
    "attention_head_dim": 128,
    "audio_num_attention_heads": 32,
    "audio_attention_head_dim": 64,
    "audio_cross_attention_dim": 2048,
    "caption_channels": 3840,
    "cross_attention_dim": 4096,
}


def _try_hf_ltx2_config() -> Optional[dict]:
    """Fetch the LTX-2 transformer config.json from HuggingFace.

    LTX-2.3 inherits architecture fields (rope_type, qk_norm, activation_fn,
    attention_bias, etc.) from LTX-2. Using the full LTX-2 config as the
    fallback — instead of a hand-curated subset — preserves every field
    that `diffusers.LTX2VideoTransformer3DModel.from_config` consumes. The
    partial fallback caused diffusers to default `rope_type` to
    `"interleaved"` even though the correct value is `"split"`, which
    produced a broadcasting error during tracing.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        return None
    try:
        p = hf_hub_download("Lightricks/LTX-2", "transformer/config.json")
        with open(p) as f:
            return json.load(f)
    except Exception as e:
        logger.debug("HF LTX-2 config fetch failed: %s", e)
        return None


def _try_github_config(variant: str) -> Optional[dict]:
    """Attempt to fetch the ltx-core config from Lightricks/LTX-2 on GitHub.

    `variant` is one of "dev", "distilled", "distilled-1.1" and selects the
    matching config file under `packages/ltx-core/configs/`.
    Returns the parsed dict or None on any failure (network, 404, parse).
    """
    candidates = [
        f"https://raw.githubusercontent.com/Lightricks/LTX-2/main/packages/ltx-core/configs/ltx-2.3-22b-{variant}.json",
        f"https://raw.githubusercontent.com/Lightricks/LTX-2/main/packages/ltx-core/configs/ltx-2.3-{variant}.json",
    ]
    for url in candidates:
        try:
            with urllib.request.urlopen(url, timeout=10) as f:
                body = f.read().decode("utf-8")
            cfg = json.loads(body)
            logger.info("Fetched LTX-2.3 config from %s", url)
            return cfg
        except Exception as e:
            logger.debug("GitHub config fetch failed for %s: %s", url, e)
    return None


def _introspect_safetensors(safetensors_path: str) -> Optional[dict]:
    """Recover num_layers and inner_dim by reading safetensors metadata.

    Opens the file with safe_open and iterates keys without materializing
    tensors — only the first attention projection's shape is read to derive
    inner_dim. The audio sub-network is detected by the `audio_` key prefix.
    """
    try:
        from safetensors import safe_open
    except ImportError:
        logger.warning("safetensors not installed; cannot introspect")
        return None

    if not os.path.isfile(safetensors_path):
        logger.warning("safetensors file not found: %s", safetensors_path)
        return None

    block_re = re.compile(r"transformer_blocks\.(\d+)\.")
    max_block = -1
    inner_dim = None
    audio_inner_dim = None
    caption_channels = None

    with safe_open(safetensors_path, framework="pt") as f:
        keys = list(f.keys())
        for k in keys:
            m = block_re.search(k)
            if m:
                idx = int(m.group(1))
                if idx > max_block:
                    max_block = idx

        for k in keys:
            if inner_dim is None and k.endswith("transformer_blocks.0.attn1.to_q.weight"):
                inner_dim = f.get_slice(k).get_shape()[0]
            if audio_inner_dim is None and k.endswith(
                "transformer_blocks.0.audio_attn1.to_q.weight"
            ):
                audio_inner_dim = f.get_slice(k).get_shape()[0]
            if caption_channels is None and "caption_projection" in k and k.endswith(".weight"):
                caption_channels = f.get_slice(k).get_shape()[1]

    if max_block < 0 or inner_dim is None:
        logger.warning("Could not infer num_layers/inner_dim from %s", safetensors_path)
        return None

    num_layers = max_block + 1
    head_dim = _LTX2_FALLBACK_CONFIG["attention_head_dim"]
    num_heads = inner_dim // head_dim

    audio_head_dim = _LTX2_FALLBACK_CONFIG["audio_attention_head_dim"]
    if audio_inner_dim is None:
        audio_inner_dim = _LTX2_FALLBACK_CONFIG["audio_num_attention_heads"] * audio_head_dim
    audio_num_heads = audio_inner_dim // audio_head_dim

    return {
        "num_layers": num_layers,
        "num_attention_heads": num_heads,
        "attention_head_dim": head_dim,
        "audio_num_attention_heads": audio_num_heads,
        "audio_attention_head_dim": audio_head_dim,
        "audio_cross_attention_dim": audio_inner_dim,
        "caption_channels": caption_channels
        or _LTX2_FALLBACK_CONFIG["caption_channels"],
        "cross_attention_dim": inner_dim,
    }


def recover_ltx23_config(
    variant: str = "distilled",
    safetensors_path: Optional[str] = None,
) -> dict:
    """Return a DiT config dict for LTX-2.3.

    Tries GitHub first, then safetensors introspection, then LTX-2 fallback.
    """
    cfg = _try_github_config(variant)
    if cfg is not None:
        return cfg

    # Fallback: pull the full LTX-2 HF config and optionally patch num_layers
    # / head counts from safetensors introspection. This keeps every non-
    # architectural field (rope_type, qk_norm, activation_fn, etc.) intact,
    # which `diffusers.LTX2VideoTransformer3DModel.from_config` requires.
    hf_cfg = _try_hf_ltx2_config()
    if hf_cfg is not None:
        if safetensors_path:
            probe = _introspect_safetensors(safetensors_path)
            if probe is not None:
                # Merge architecture deltas (num_layers, inner_dim-derived head
                # counts) from the LTX-2.3 checkpoint over the LTX-2 defaults.
                for k in (
                    "num_layers",
                    "num_attention_heads",
                    "attention_head_dim",
                    "audio_num_attention_heads",
                    "audio_attention_head_dim",
                    "audio_cross_attention_dim",
                ):
                    if probe.get(k) is not None:
                        hf_cfg[k] = probe[k]
        # Flip on LTX-2.3 feature flags (diffusers 0.39+, after PR #13217).
        # Without these, the backbone runs the LTX-2.0 compute graph and
        # produces visual noise with LTX-2.3 weights.
        hf_cfg["gated_attn"] = True
        hf_cfg["audio_gated_attn"] = True
        hf_cfg["cross_attn_mod"] = True
        hf_cfg["audio_cross_attn_mod"] = True
        hf_cfg["perturbed_attn"] = True
        # LTX-2.3 does text embedding in the pipeline's connector, so the DiT
        # has no caption_projection. LTX-2.0 had it; switch off for 2.3.
        hf_cfg["use_prompt_embeddings"] = False
        logger.info("Recovered LTX-2.3 config from Lightricks/LTX-2 with LTX-2.3 flags + safetensors patches")
        return hf_cfg

    logger.warning(
        "Falling back to LTX-2 architecture values for LTX-2.3 — verify these "
        "match the checkpoint before running compile."
    )
    return dict(_LTX2_FALLBACK_CONFIG)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="distilled",
                        choices=["dev", "distilled", "distilled-1.1"])
    parser.add_argument("--safetensors", default=None,
                        help="Path to local ltx-2.3-*.safetensors (optional)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    cfg = recover_ltx23_config(variant=args.variant, safetensors_path=args.safetensors)
    print(json.dumps(cfg, indent=2))


if __name__ == "__main__":
    main()
