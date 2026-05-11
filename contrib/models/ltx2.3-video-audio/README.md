# LTX-2.3 on AWS Trainium (NxDI)

Neuron-optimized implementation of the **Lightricks LTX-2.3** (22B) DiT audio-video diffusion transformer, running on **trn2.3xlarge** with tensor parallelism (TP=4) in bfloat16.

This package is a focused extension of `contrib/models/ltx2-video-audio/`. The DiT backbone is the new 22B LTX-2.3 checkpoint; the text encoder (Gemma 3-12B), VAE, vocoder, scheduler, and NKI attention kernels are reused unchanged from the LTX-2 package.

## What's different from LTX-2

| Aspect | LTX-2 | LTX-2.3 |
| --- | --- | --- |
| HF repo | `Lightricks/LTX-2` (Diffusers multi-folder) | `Lightricks/LTX-2.3` (monolithic `*.safetensors`) |
| DiT params | 19 B | 22 B |
| Variants | dev / distilled / … | dev / distilled / distilled-1.1 |
| `config.json` | yes, under `transformer/` | **not published** — recovered from Lightricks GitHub or safetensors introspection |
| Text encoder / VAE / scheduler | same | **same — reused from LTX-2 Diffusers repo** |
| Precision | bf16 | bf16 |
| Target | trn2.3xlarge, TP=4, SDK 2.28 | same |

FP8 is **not** used here — we intentionally target the bf16 LTX-2.3 weights, not `Lightricks/LTX-2.3-fp8`, because Neuron SDK 2.28 does not yet expose native fp8 GEMMs.

## Prerequisites

- **Instance:** `trn2.3xlarge` (1 NeuronDevice, 4 logical cores with LNC=2)
- **Neuron SDK:** 2.27 or later (tested against 2.28)
- **Python venv:** `/opt/aws_neuronx_venv_pytorch_inference_vllm_0_16/` (all commands below assume this interpreter)
- **HuggingFace access:** a valid token with access to `Lightricks/LTX-2` and `Lightricks/LTX-2.3` (both gated under the LTX-2 Community License)
- **LTX-2 compile outputs reused as-is:** Gemma3 encoder, Gemma3 sharded weights, (optional) VAE. Compile these **once** from the `ltx2-video-audio` package before running LTX-2.3.

## End-to-end flow

```bash
VENV=/opt/aws_neuronx_venv_pytorch_inference_vllm_0_16/bin
LTX2=contrib/models/ltx2-video-audio/src
LTX23=contrib/models/ltx2.3-video-audio/src

# 1. Reuse the LTX-2 Gemma3 compile + shard (skip if already done for LTX-2)
NEURON_FUSE_SOFTMAX=1 NEURON_CUSTOM_SILU=1 NEURON_RT_STOCHASTIC_ROUNDING_EN=0 \
  $VENV/python $LTX2/compile_gemma3.py
$VENV/python $LTX2/shard_gemma3_weights.py

# 2. (Optional) reuse the LTX-2 VAE compile
$VENV/python $LTX2/compile_vae.py --tp-degree 4 --height 256 --width 256 --num-frames 25

# 3. Compile the LTX-2.3 DiT backbone
NEURON_RT_VISIBLE_CORES=0-3 \
  $VENV/python $LTX23/compile_dit.py \
    --variant distilled \
    --output-dir /home/ubuntu/ltx23_nxdi_compiled_1024_256x256_f25/ \
    --tp-degree 4 --height 256 --width 256 --num-frames 25 --text-seq 1024

# 4. Generate
LTX23_DIT_COMPILE_DIR=/home/ubuntu/ltx23_nxdi_compiled_1024_256x256_f25/ \
NEURON_FUSE_SOFTMAX=1 NEURON_CUSTOM_SILU=1 NEURON_RT_STOCHASTIC_ROUNDING_EN=0 \
  $VENV/python contrib/models/ltx2.3-video-audio/examples/neuron_e2e.py
```

## File layout

```
ltx2.3-video-audio/
├── README.md                              # this file
├── src/
│   ├── __init__.py
│   ├── modeling_ltx2_3.py                 # wraps LTX-2 modeling with a new loader
│   ├── recover_config.py                  # recovers DiT config (GitHub → safetensors → fallback)
│   ├── compile_dit.py                     # compile entry point
│   ├── application.py                     # NeuronLTX23Application (extends LTX-2's)
│   └── generate_ltx2_3.py                 # one-shot generate CLI
├── examples/
│   └── neuron_e2e.py                      # full TP=4 E2E pipeline
└── test/
    └── integration/
        └── test_model.py                  # shape-level smoke test
```

All NKI kernels (`nki_cross_attention_kernel.py`, `attention_cte_bias.py`), the transformer backbone class, the TP sharding helper, and the NxDI application/model-wrapper base classes are **imported** from `contrib/models/ltx2-video-audio/src/` rather than copied, so there is a single source of truth for those.

## Configuration recovery

`Lightricks/LTX-2.3` does not publish a `transformer/config.json`. `src/recover_config.py` tries three strategies in order:

1. Fetch `packages/ltx-core/configs/ltx-2.3-22b-{variant}.json` from [github.com/Lightricks/LTX-2](https://github.com/Lightricks/LTX-2).
2. Open the local `.safetensors` file and introspect `transformer_blocks.*` keys to recover `num_layers`, `inner_dim`, head counts.
3. Fall back to LTX-2's values (48 layers / 32 heads / head_dim=128 / audio 32×64 / caption_channels=3840) with a warning.

Run it standalone to inspect the recovered config:

```bash
$VENV/python contrib/models/ltx2.3-video-audio/src/recover_config.py --variant distilled
```

## Key remapping

The LTX-2.3 monolithic safetensors file may prefix tensor names differently from the `diffusers.LTX2VideoTransformer3DModel` convention that the existing LTX-2 sharding code expects. `modeling_ltx2_3.py` exposes a `_LTX23_KEY_REMAP` tuple — add `("source_prefix.", "diffusers_prefix.")` entries there if compile fails with key-miss errors during `load_state_dict`. Build the remap incrementally by diffing:

```python
from safetensors import safe_open
with safe_open("/path/to/ltx-2.3-22b-distilled.safetensors", framework="pt") as f:
    print("\n".join(sorted(f.keys())[:40]))
```

against the `state_dict()` of `diffusers.LTX2VideoTransformer3DModel.from_pretrained("Lightricks/LTX-2/transformer")`.

## Verification

| Step | Command | Success criterion |
| --- | --- | --- |
| Config recovery | `python src/recover_config.py --variant distilled` | prints JSON with plausible `num_layers`, `num_attention_heads`, `inner_dim` |
| DiT compile (small) | `python src/compile_dit.py --height 256 --width 256 --num-frames 25 --text-seq 1024` | writes `model.pt` to output dir without errors |
| Smoke E2E | `python examples/neuron_e2e.py` | produces 25 RGB frames and optional MP4 |
| Integration test | `pytest test/integration/test_model.py` | passes when compiled artifacts are present |
| Full-shape DiT | rerun compile with `--height 512 --width 768 --num-frames 121` | recompiles successfully |

## Risks and open items

- **Diffusers support for LTX-2.3 is not published yet.** This port reuses `LTX2Pipeline.from_pretrained("Lightricks/LTX-2", ...)` for the non-DiT components. When Lightricks publishes an `LTX23Pipeline` or updates `LTX2Pipeline` to handle 2.3, switch `PIPELINE_REPO` to `Lightricks/LTX-2.3`.
- **Layer count / width deltas.** 22B vs 19B implies LTX-2.3 is deeper or wider than LTX-2. If the actual `num_layers` differs, the existing TP sharder scales; if width differs, a per-tensor shape check may fail and the sharder will need a parallel adjustment.
- **NKI cross-attention kernel** hardcodes `K_seq=1024, head_dim=128`. `head_dim` is almost certainly unchanged (caption_channels=3840 matches Gemma3 hidden). If LTX-2.3 increases the text sequence length beyond 1024, `nki_cross_attention_kernel.py` needs to be reparameterized.
- **License.** LTX-2 Community License Agreement (Lightricks-custom, not OSI). Same license applies to LTX-2.3. Review before productionizing.
