# Contrib Model: Qwen3.6-27B-FP8

FP8 delta adapter for
[`Qwen/Qwen3.6-27B-FP8`](https://huggingface.co/Qwen/Qwen3.6-27B-FP8).

## Status: validated end-to-end (text **and** vision)

Last validation: 2026-05-11 on `trn2.3xlarge` (SDK 2.29, 1 Neuron device, 4 cores / 96 GB HBM).
The full pipeline — real checkpoint download → FP8 dequantization on
load → compilation → TP=4 trace → text-only and VL generation — runs
cleanly.

### Text-only (TP=4, seq_len=64)

| Test | Result |
|------|--------|
| Smoke — "Hello, I am a language model" → 20 tokens | **15.5–16.6 tok/s** |
| Accuracy — "What is the capital of France?" (chat-templated) | ✅ `" Paris"` |
| Throughput — 50 tokens | **TPOT 57.6 ms, 17.4 tok/s** |

Compare to PR #140's `Qwen3.5-27B` **BF16** baseline reported on the same
hardware (TPOT 54.2 ms, 18.5 tok/s): FP8 is ~6% slower on both metrics,
because dequantization happens once at weight load (we then store bf16),
not per token.

### Vision-language (TP=4, seq_len=256, vision on CPU)

| Test | Result |
|------|--------|
| Image + prompt → generation | ✅ 60 tokens in 5.10 s = **11.8 tok/s** |
| Visual grounding | Model correctly identifies "a solid red circle" / "text inside" in the supplied synthetic image |

Vision encoder runs on CPU per PR #140's Caveat #4 (HBM is consumed by
the text decoder on trn2.3xlarge); this adds ~900 ms per image.

### Compile-time notes

* Text-only at `seq_len=64` compiled in ~20 min on the first run.
* `seq_len=128` hit a compiler `F139 neuronx-cc terminated abnormally`
  after ~90 min on the text-only path; retry at a different size or
  move to `trn2.12xlarge`.
* VL at `seq_len=256` compiled in ~26 min + ~12 min weight load.

## Depends on PR #140

This adapter **does not** implement the Qwen3.6 architecture itself.
The hybrid GatedDeltaNet + GQA architecture, NKI kernels, attn_output_gate,
and partial RoPE all come from PR #140:
<https://github.com/aws-neuron/neuronx-distributed-inference/pull/140>
(base adapter path: `contrib/models/Qwen3.6-27B/`).

This adapter adds **only** what PR #140 does not:

1. **Block-wise FP8 → bf16 dequantization on state-dict load.** The
   published `Qwen/Qwen3.6-27B-FP8` checkpoint stores weights in E4M3
   FP8 with per-block scale factors (`weight_block_size=[128, 128]`,
   `activation_scheme="dynamic"`). We expand scales via
   `repeat_interleave` and multiply the FP8 weight in fp32 before
   casting to `neuron_config.torch_dtype` (typically bf16).
2. **Respect for `modules_to_not_convert`.** The HF quantization config
   lists the full vision tower, embeddings, `lm_head`, SSM `A_log /
   conv1d / dt_bias / in_proj_{a,b}`, and layernorms as "do not
   dequantize" (they were never quantized). The shim skips them.
3. **VL vision loader for the FP8 checkpoint's sharding.** PR #140's
   `NeuronQwen35VisionModelWrapper.load_cpu_model` iterates
   `model*.safetensors`. The FP8 checkpoint uses `layers-N.safetensors`
   + `outside.safetensors`, so the stock loader finds zero vision
   weights. We patch the wrapper to scan every `*.safetensors` in the
   model directory.
4. **VL on-device-sampling generate loop.** PR #140's
   `NeuronQwen35VLForCausalLM.generate()` assumes the text model returns
   raw logits. When the text model is compiled with
   `on_device_sampling_config=...` (the default), the traced model
   returns sampled token ids in `output.tokens` instead. Our subclass
   detects this and reads `tokens` / `logits` correctly.
5. **VL load path.** Vision on CPU (not compiled) per PR #140's
   Caveat #4: HBM on trn2.3xlarge is consumed by the text decoder, so
   the ViT runs in pure PyTorch. Our subclass's `load()` calls
   `load_cpu_model()` + `load_vision_weights_from_hf()` directly.

The dequant recipe is the same as `qwen3_moe.maybe_dequantize_layer` and
DeepSeek-V3's FP8 path already in this tree; our variant is a 30-line
standalone so the adapter keeps working regardless of refactors in
those modules.

## Reproducing the end-to-end run

### 1. Prerequisites

- A Trainium instance (`trn2.3xlarge` validated, 1 device × 4 cores × 96 GB HBM).
- SDK 2.29+ with the `aws_neuronx_venv_pytorch_inference_vllm_0_16` venv
  (that's what the validation machine used).
- PR #140 on the branch/main. If you just cloned the repo:
  ```bash
  git fetch origin 'pull/140/head:pr-140'
  git cherry-pick pr-140
  ```

### 2. Download the checkpoint (~29 GB)

```bash
/opt/aws_neuronx_venv_pytorch_inference_vllm_0_16/bin/python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='Qwen/Qwen3.6-27B-FP8',
    local_dir='/home/ubuntu/models/Qwen3.6-27B-FP8',
    max_workers=8,
)"
```

Took ~4 min on a fast connection during validation.

### 3a. Text-only integration test

```bash
QWEN36_FP8_MODEL_PATH=/home/ubuntu/models/Qwen3.6-27B-FP8 \
QWEN36_FP8_COMPILED_PATH=/tmp/qwen36_fp8_traced \
QWEN36_FP8_TP_DEGREE=4 \
QWEN36_FP8_SEQ_LEN=64 \
/opt/aws_neuronx_venv_pytorch_inference_vllm_0_16/bin/python \
    contrib/models/Qwen3.6-27B-FP8/test/integration_fp8.py
```

First run: ~25 min (compile + load + generate). Second run: ~20 s
(NEFFs are cached at `QWEN36_FP8_COMPILED_PATH`).

Expected tail of the output:

```
[1/3] Smoke test: basic generation
  New tokens: 20  elapsed: 1.20s  throughput: 16.6 tok/s
  Text: Hello, I am a language model ...

[2/3] Accuracy: capital of France via chat template
  New tokens: 2  Generated:
    ' Paris'

[3/3] Performance: throughput on 50 new tokens
  New tokens: 50  elapsed: 2.88s
  TPOT: 57.6 ms   Throughput: 17.4 tok/s

All checks passed.
```

### 3b. Vision-language integration test

VL needs more seq_len than text-only because the chat template plus
256 vision tokens already pushes past 64. Use `seq_len=256`:

```bash
QWEN36_FP8_MODEL_PATH=/home/ubuntu/models/Qwen3.6-27B-FP8 \
QWEN36_FP8_COMPILED_PATH=/tmp/qwen36_fp8_vl_traced_256 \
QWEN36_FP8_TP_DEGREE=4 \
QWEN36_FP8_SEQ_LEN=256 \
/opt/aws_neuronx_venv_pytorch_inference_vllm_0_16/bin/python \
    contrib/models/Qwen3.6-27B-FP8/test/integration_vl_fp8.py
```

First run: ~26 min compile + ~12 min FP8 weight load. Subsequent runs
reuse the compiled NEFFs. The test generates a synthetic 224×224 JPEG
(a red circle labelled "CAT") at `/tmp/qwen36_vl_test_image.jpg` so it
needs no network access. Supply your own image via
`QWEN36_FP8_IMAGE_PATH=/path/to/image.jpg` to test anything else.

Expected tail:

```
Image: /tmp/qwen36_vl_test_image.jpg
  input_ids.shape: (1, 86)
  pixel_values.shape: (256, 1536)

Generating...

Generated 60 tokens in 5.10s (11.8 tok/s)
Response: "The user wants me to identify the image provided.
  ... It features a solid red circle. Inside the circle, there is text ..."

All VL checks passed.
```

## Usage (programmatic)

### Text-only

```python
from src.modeling_qwen3_6_fp8 import build_fp8_for_causal_lm_class

# Returns a subclass of PR #140's NeuronQwen35ForCausalLM with
# convert_hf_to_neuron_state_dict overridden to dequantize FP8 first.
NeuronQwen3_6_27B_FP8_ForCausalLM = build_fp8_for_causal_lm_class()

model = NeuronQwen3_6_27B_FP8_ForCausalLM(model_path, config)
model.compile(compiled_model_path)
model.load(compiled_model_path)
```

### Vision-language

```python
from src.modeling_qwen3_6_vl_fp8 import build_vl_fp8_class

NeuronQwen3_6_27B_FP8_VL = build_vl_fp8_class()

vl_model = NeuronQwen3_6_27B_FP8_VL(
    model_path=model_path,
    text_config=text_config,
    vision_config=vl_config,   # Qwen35VLInferenceConfig from PR #140
)
vl_model.compile(compiled_path)
vl_model.load(compiled_path)   # text on Neuron, vision on CPU

output_ids = vl_model.generate(
    input_ids,
    attention_mask=attention_mask,
    pixel_values=pixel_values,
    image_grid_thw=image_grid_thw,
    max_new_tokens=60,
    temperature=0.0,
)
```

See `test/integration_fp8.py` and `test/integration_vl_fp8.py` for
complete end-to-end scripts (config construction, chat templating, etc.).

## What is NOT in this adapter

- **Anything architectural.** See PR #140 for the model code, the 3
  NKI DeltaNet kernels (TKG, chunked, fused CTE), the full-attention
  output gate, and the partial-RoPE path.
- **A compiled vision tower on Neuron.** Per PR #140 Caveat #4, HBM on
  `trn2.3xlarge` is full with the text decoder, so the ViT runs on CPU
  (~900 ms per image). `trn2.12xlarge` or larger could host it.
- **A separate hybrid KV cache manager.** PR #140 sidesteps one by
  returning dummy KV tuples for DeltaNet layers. A production-grade
  hybrid cache is tracked at
  `src/neuronx_distributed_inference/modules/kvcache/hybrid_kv_cache_manager.py`
  but is **not** required for this adapter to run.

## Test matrix

| Test | Kind | How to run |
|------|------|------------|
| `test_model.py` | Smoke — imports + 2-line dequant sanity | `python test_model.py` |
| `test/test_fp8_dequant.py` | Unit — round-trip math, `modules_to_not_convert`, edge cases | `pytest test/test_fp8_dequant.py` |
| `test/test_end_to_end_shim.py` | Integration — real shim resolves PR #140, dequantizes, delegates | `pytest test/test_end_to_end_shim.py` |
| `test/integration_fp8.py` | End-to-end text on Neuron — compile, load, generate | See "Reproducing" above |
| `test/integration_vl_fp8.py` | End-to-end VL on Neuron — image + text → generation | See "Reproducing" above |

CPU-only results on the validation machine: **4/4 dequant + 2/2 shim = 6 passing**.
Device tests passed on `trn2.3xlarge` as documented above.

## Compatibility Matrix

| Instance | Modality | seq_len | Status |
|----------|----------|---------|--------|
| trn2.3xlarge (TP=4) | Text | 64  | ✅ Validated 2026-05-11 |
| trn2.3xlarge (TP=4) | Text | 128 | ⚠️ Compile hangs / F139 (~90 min) on this FP8 checkpoint. PR #140's BF16 variant passes at 128. |
| trn2.3xlarge (TP=4, vision on CPU) | Image+Text | 256 | ✅ Validated 2026-05-11 |
| Trn1 | — | — | Not tested |
| Inf2 | — | — | Not tested |

## Maintainer

Community contribution. If PR #140 lands upstream with different
symbol names (`Qwen35InferenceConfig`, `NeuronQwen35ForCausalLM`,
`convert_qwen35_hf_to_neuron_state_dict`), update the imports in
`src/modeling_qwen3_6_fp8.py` accordingly. The shim already checks for
these names and raises a clear error if upstream has renamed them.
