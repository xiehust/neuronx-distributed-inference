#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""
End-to-end integration test for Qwen/Qwen3.6-27B-FP8 on Neuron.

Mirrors PR #140's ``contrib/models/Qwen3.6-27B/test/integration/test_model.py``
but wires our FP8 shim in as the model class so the FP8 checkpoint is
dequantized to bf16 during load.

Environment:
    QWEN36_FP8_MODEL_PATH     Path to downloaded HF weights (required)
    QWEN36_FP8_COMPILED_PATH  Path for compiled artifacts (default /tmp/qwen36_fp8_traced)
    QWEN36_FP8_TP_DEGREE      TP degree (default 4 for trn2.3xlarge)
    QWEN36_FP8_SEQ_LEN        Max seq len (default 128)

Run with:
    QWEN36_FP8_MODEL_PATH=/home/ubuntu/models/Qwen3.6-27B-FP8 \\
    /opt/aws_neuronx_venv_pytorch_inference_vllm_0_16/bin/python \\
        contrib/models/Qwen3.6-27B-FP8/test/integration_fp8.py
"""
from __future__ import annotations

import gc
import json
import os
import sys
import time


def _ensure_neuron_bin_on_path():
    """``torch_xla`` calls ``libneuronpjrt-path`` during its AWS Neuron
    init. That binary lives next to the venv's Python; make sure it's
    reachable via PATH before the first ``import torch`` triggers
    ``torch_xla`` loading.
    """
    venv_bin = os.path.dirname(sys.executable)
    current = os.environ.get("PATH", "")
    if venv_bin and venv_bin not in current.split(os.pathsep):
        os.environ["PATH"] = venv_bin + os.pathsep + current


_ensure_neuron_bin_on_path()

import torch  # noqa: E402


def _setup_paths():
    """Put both PR #140's contrib and our shim on sys.path."""
    this_dir = os.path.dirname(os.path.abspath(__file__))
    # Our shim lives at ../src; PR #140 lives at ../../Qwen3.6-27B.
    fp8_src = os.path.abspath(os.path.join(this_dir, "..", "src"))
    pr140 = os.path.abspath(os.path.join(this_dir, "..", "..", "Qwen3.6-27B"))
    for p in (fp8_src, pr140):
        if p not in sys.path:
            sys.path.insert(0, p)
    # Our shim's heuristic will also invalidate cached ``src`` modules so
    # PR #140's src resolves; we import the shim last so it runs after
    # both paths are on sys.path.


def _load_shim():
    _setup_paths()
    from modeling_qwen3_6_fp8 import build_fp8_for_causal_lm_class
    return build_fp8_for_causal_lm_class()


def _make_config(model_path: str, tp_degree: int, seq_len: int):
    """Build a Qwen35InferenceConfig from the downloaded HF config.json,
    attaching the HF quantization_config so our shim runs on load."""
    from neuronx_distributed_inference.models.config import (
        NeuronConfig,
        OnDeviceSamplingConfig,
    )
    # Resolve PR #140's config class via our already-populated sys.path.
    from src.modeling_qwen35 import Qwen35InferenceConfig  # type: ignore

    with open(os.path.join(model_path, "config.json")) as f:
        full_cfg = json.load(f)
    text_cfg = full_cfg.get("text_config", full_cfg)
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
    inf_config = Qwen35InferenceConfig(neuron_config=neuron_config, **cfg_dict)
    # Attach FP8 quant info so our shim's convert_hf_to_neuron_state_dict
    # picks it up.
    if quantization_cfg is not None:
        inf_config.quantization_config = quantization_cfg
    return inf_config


def _compile_and_load(model_path: str, compiled_path: str, tp_degree: int, seq_len: int):
    fp8_cls = _load_shim()
    inf_config = _make_config(model_path, tp_degree, seq_len)
    neff_path = os.path.join(compiled_path, "model.pt")
    if not os.path.exists(neff_path):
        print(f"Compiling to {compiled_path}...", flush=True)
        t0 = time.time()
        model = fp8_cls(model_path, inf_config)
        model.compile(compiled_path)
        print(f"  Compile took {time.time()-t0:.1f}s", flush=True)
        del model
        gc.collect()
    print(f"Loading from {compiled_path}...", flush=True)
    model = fp8_cls(compiled_path)
    model.load(compiled_path)
    return model


def _generate(model, tokenizer, gen_cfg, prompt: str, max_new_tokens: int = 20):
    from neuronx_distributed_inference.utils.hf_adapter import (
        HuggingFaceGenerationAdapter,
    )
    inputs = tokenizer(prompt, padding=True, return_tensors="pt")
    adapter = HuggingFaceGenerationAdapter(model)
    t0 = time.time()
    outputs = adapter.generate(
        inputs.input_ids,
        generation_config=gen_cfg,
        attention_mask=inputs.attention_mask,
        max_new_tokens=max_new_tokens,
    )
    elapsed = time.time() - t0
    tokens = outputs[0].tolist()
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    new_tokens = len(tokens) - inputs.input_ids.shape[1]
    return tokens, text, new_tokens, elapsed


def main():
    model_path = os.environ.get("QWEN36_FP8_MODEL_PATH", "")
    if not model_path or not os.path.isdir(model_path):
        print(f"QWEN36_FP8_MODEL_PATH not set or not a directory: {model_path!r}")
        return 1
    compiled_path = os.environ.get("QWEN36_FP8_COMPILED_PATH", "/tmp/qwen36_fp8_traced")
    tp_degree = int(os.environ.get("QWEN36_FP8_TP_DEGREE", "4"))
    seq_len = int(os.environ.get("QWEN36_FP8_SEQ_LEN", "128"))
    os.makedirs(compiled_path, exist_ok=True)

    from transformers import AutoTokenizer, GenerationConfig

    print(f"Model path:   {model_path}")
    print(f"Compiled to:  {compiled_path}")
    print(f"TP degree:    {tp_degree}")
    print(f"Seq len:      {seq_len}")

    model = _compile_and_load(model_path, compiled_path, tp_degree, seq_len)
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    gen_cfg = GenerationConfig(
        do_sample=True,
        top_k=1,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    # ---------------- Smoke ----------------
    print("\n[1/3] Smoke test: basic generation")
    tokens, text, new_tok, elapsed = _generate(
        model, tokenizer, gen_cfg, "Hello, I am a language model", max_new_tokens=20
    )
    print(f"  New tokens: {new_tok}  elapsed: {elapsed:.2f}s  "
          f"throughput: {new_tok/elapsed:.1f} tok/s")
    print(f"  Text: {text[:160]}")
    assert new_tok >= 5, f"Expected >=5 new tokens, got {new_tok}"

    # ---------------- Accuracy -------------
    # Qwen3.6-27B has a "thinking mode" enabled by default that emits a
    # <think>...</think> block before the answer. We disable that by passing
    # enable_thinking=False to the chat template, and give the model enough
    # headroom (90 tokens) to answer even if thinking is accidentally on.
    print("\n[2/3] Accuracy: capital of France via chat template")
    accuracy_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "What is the capital of France? Answer in one word."}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    tokens, text, new_tok, elapsed = _generate(
        model, tokenizer, gen_cfg, accuracy_prompt, max_new_tokens=90
    )
    print(f"  New tokens: {new_tok}  Generated:")
    # Print only the tail (the model response), not the echoed prompt.
    input_len = len(tokenizer.encode(accuracy_prompt))
    response_tokens = tokens[input_len:]
    response_text = tokenizer.decode(response_tokens, skip_special_tokens=True)
    print(f"    {response_text!r}")
    assert "paris" in response_text.lower(), (
        f"Expected 'Paris' in response; got {response_text!r}"
    )

    # ---------------- Perf -----------------
    print("\n[3/3] Performance: throughput on 50 new tokens")
    tokens, text, new_tok, elapsed = _generate(
        model, tokenizer, gen_cfg, "Hello, I am a language model", max_new_tokens=50
    )
    tpot_ms = elapsed / new_tok * 1000 if new_tok else float('inf')
    tput = new_tok / elapsed if elapsed else 0
    print(f"  New tokens: {new_tok}  elapsed: {elapsed:.2f}s")
    print(f"  TPOT: {tpot_ms:.1f} ms   Throughput: {tput:.1f} tok/s")

    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
