#!/usr/bin/env python3
"""
Test scaffold for the Qwen3.6-27B-FP8 adapter.

This adapter is a thin FP8 delta on top of PR #140's Qwen3.6-27B base
adapter (https://github.com/aws-neuron/neuronx-distributed-inference/pull/140).
This scaffold only verifies:
  1. Python import of the FP8 loader.
  2. The dequantization recipe runs against a tiny synthetic checkpoint.

End-to-end generation requires PR #140 merged (or its branch checked out
and on PYTHONPATH).
"""
import os
import sys
from pathlib import Path


def test_fp8_imports():
    src = Path(__file__).parent / "src"
    sys.path.insert(0, str(src))
    from modeling_qwen3_6_fp8 import (  # noqa: F401
        convert_qwen3_6_fp8_hf_to_neuron_state_dict,
        dequantize_fp8_block_scaled,
        build_fp8_for_causal_lm_class,
    )
    print("FP8 delta imports: OK")


def test_dequant_smoke():
    import torch
    src = Path(__file__).parent / "src"
    sys.path.insert(0, str(src))
    from modeling_qwen3_6_fp8 import dequantize_fp8_block_scaled

    sd = {
        "layers.0.self_attn.q_proj.weight": torch.ones(4, 4),
        "layers.0.self_attn.q_proj.weight_scale_inv": torch.full((2, 2), 2.0),
    }
    dequantize_fp8_block_scaled(
        sd, weight_block_size=[2, 2], target_dtype=torch.float32
    )
    expected = torch.full((4, 4), 2.0)
    assert torch.allclose(sd["layers.0.self_attn.q_proj.weight"], expected), "dequant math"
    assert "layers.0.self_attn.q_proj.weight_scale_inv" not in sd, "scale dropped"
    print("Dequant smoke: OK")


def main() -> bool:
    try:
        test_fp8_imports()
        test_dequant_smoke()
    except Exception as exc:
        print(f"FAIL: {exc}")
        return False
    print(
        "NOTE: end-to-end tracing / generation requires PR #140 merged or "
        "its branch on PYTHONPATH. See README."
    )
    return True


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
