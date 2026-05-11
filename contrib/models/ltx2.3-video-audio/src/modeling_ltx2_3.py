"""
NxDI LTX-2.3 Transformer Model
==============================
Neuron-optimized implementation of the LTX-2.3 22B DiT audio-video
diffusion transformer. Reuses the LTX-2 modeling, sharding, and NKI kernel
integration wholesale — the only differences handled here are:

  1. The HuggingFace layout is a single monolithic safetensors file, not a
     Diffusers multi-folder layout, so the checkpoint loader must download
     one file and iterate its keys.
  2. The 22B DiT likely has a different block count than LTX-2's 48; the
     architecture is otherwise identical (same attention heads, same
     caption_channels=3840, same audio stream dims).
  3. The internal tensor key naming may differ from `diffusers
     .LTX2VideoTransformer3DModel` — an extensible key-remap hook is
     provided so the canonical LTX-2 tooling can consume the state dict.

Everything else — the NKI flash/cross-attn kernels, DistributedRMSNorm,
SPMDRank RoPE slicing, _shard_ltx2_transformer, input_generator,
compiler args — is imported unchanged from the LTX-2 package.

Usage:
  python compile_dit.py [--variant distilled] ...
  python generate_ltx2_3.py [--prompt "..."] ...
"""

from __future__ import annotations

import logging
import math
import os
import sys
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Cross-import the sibling LTX-2 package so we reuse its modeling verbatim.
_LTX2_SRC = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "ltx2-video-audio", "src")
)
if _LTX2_SRC not in sys.path:
    sys.path.insert(0, _LTX2_SRC)

# Re-export everything we need from the LTX-2 modeling module.
from modeling_ltx2 import (  # noqa: E402  (path munging)
    DistributedRMSNorm,
    LTX2BackboneInferenceConfig,
    ModelWrapperLTX2Backbone,
    NeuronLTX2BackboneApplication,
    NeuronLTX2TransformerBackbone,
    _shard_ltx2_transformer,
    replace_sdpa_with_bmm,
)

NEURON_AVAILABLE = True
try:
    from neuronx_distributed.parallel_layers.layers import (  # noqa: E402
        ColumnParallelLinear,
        RowParallelLinear,
        SPMDRank,
    )
    from neuronx_distributed.parallel_layers import parallel_state  # noqa: E402
    from neuronx_distributed_inference.models.model_wrapper import (  # noqa: E402
        BaseModelInstance,
    )
except ImportError:
    NEURON_AVAILABLE = False

# LTX-2.3 block uses 9 modulation rows per-block (vs LTX-2.0's 6) because
# cross_attn_mod adds 3 extra rows (shift_q, scale_q, gate) for the text
# cross-attention path. This means:
#   - time_embed.linear output dim is 9 * inner_dim (was 6 * inner_dim)
#   - Each block's scale_shift_table is [9, inner_dim]
#   - Extra per-block prompt_scale_shift_table is [2, inner_dim]
NUM_MOD_PARAMS_LTX23 = 9
NUM_MOD_PARAMS_LTX20 = 6
PROMPT_MOD_PARAMS = 2  # LTX-2.3-only
A2V_CA_MOD_PARAMS = 4  # a2v scale_shift rows 0-3
A2V_CA_GATE_PARAMS = 1  # a2v scale_shift row 4 (reshaped as [1, D])


# ── LTX-2.3 Backbone ────────────────────────────────────────────────────────
# Diffusers 0.39+ `LTX2VideoTransformer3DModel.forward` delegates to each
# block with 22 tensor kwargs plus a few per-block `None` defaults. We
# collapse the 48-block loop into a single Neuron graph and expose the same
# block-kwargs interface so a `NeuronBlockListShim` in `pipeline` land can
# call this backbone once per forward() (not once per block).

class LTX23BackboneInferenceConfig(LTX2BackboneInferenceConfig):
    """InferenceConfig for the LTX-2.3 DiT block-loop backbone."""


class NeuronLTX23TransformerBackbone(nn.Module):
    """The LTX-2.3 DiT block loop on Neuron.

    The module owns only the 48 transformer blocks + the SPMDRank tensor for
    per-rank RoPE slicing. Input projection, time embedding, caption
    projection, output norm, and proj_out all stay on CPU (diffusers owns
    them and runs them before/after this call).

    forward() signature matches the kwargs that diffusers
    `LTX2VideoTransformer3DModel.forward` passes to each `LTX2VideoTransformerBlock`
    in LTX-2.3 mode (cross_attn_mod=True, perturbed_attn=True). Outputs the
    post-48-blocks `(hidden_states, audio_hidden_states)`.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tp_degree = config.neuron_config.tp_degree

        hf_config_dict = getattr(config, "hf_config_dict", None)
        if hf_config_dict is None:
            raise ValueError(
                "LTX23BackboneInferenceConfig needs an hf_config_dict attribute "
                "with the LTX-2.3 flags set (gated_attn=True, cross_attn_mod=True, "
                "perturbed_attn=True, use_prompt_embeddings=False)."
            )
        self._build_from_diffusers(hf_config_dict)

    def _build_from_diffusers(self, hf_config_dict):
        """Instantiate diffusers' LTX-2.3 transformer, strip preprocessing +
        output heads, apply TP sharding to the 48 blocks."""
        replace_sdpa_with_bmm()

        from diffusers.models.transformers.transformer_ltx2 import (
            LTX2VideoTransformer3DModel,
        )

        hf_model = LTX2VideoTransformer3DModel.from_config(hf_config_dict)
        hf_model = hf_model.to(dtype=self.config.neuron_config.torch_dtype)
        hf_model.eval()

        if self.tp_degree > 1:
            _shard_ltx23_transformer(hf_model, self.tp_degree)

        # Only keep the 48 transformer blocks; preprocessing + output layers
        # stay on the pipeline-side CPU transformer.
        self.transformer_blocks = hf_model.transformer_blocks

        if self.tp_degree > 1 and NEURON_AVAILABLE:
            self.spmd_rank = SPMDRank(self.tp_degree)
        else:
            self.spmd_rank = None

    def _slice_rope(self, rope_tuple):
        """Slice RoPE (cos, sin) to the heads owned by this TP rank."""
        if rope_tuple is None or self.tp_degree <= 1 or self.spmd_rank is None:
            return rope_tuple
        cos, sin = rope_tuple
        h_per_rank = cos.shape[1] // self.tp_degree
        rank = self.spmd_rank.get_rank()
        start = (rank[0] * h_per_rank).to(torch.long)
        indices = start + torch.arange(h_per_rank, device=cos.device, dtype=torch.long)
        return (
            torch.index_select(cos, 1, indices),
            torch.index_select(sin, 1, indices),
        )

    def forward(
        self,
        hidden_states,
        audio_hidden_states,
        encoder_hidden_states,
        audio_encoder_hidden_states,
        temb,
        temb_audio,
        temb_ca_scale_shift,
        temb_ca_audio_scale_shift,
        temb_ca_gate,
        temb_ca_audio_gate,
        temb_prompt,
        temb_prompt_audio,
        video_rot_cos,
        video_rot_sin,
        audio_rot_cos,
        audio_rot_sin,
        ca_video_rot_cos,
        ca_video_rot_sin,
        ca_audio_rot_cos,
        ca_audio_rot_sin,
        encoder_attention_mask,
        audio_encoder_attention_mask,
    ):
        video_rotary_emb = self._slice_rope((video_rot_cos, video_rot_sin))
        audio_rotary_emb = self._slice_rope((audio_rot_cos, audio_rot_sin))
        ca_video_rotary_emb = self._slice_rope((ca_video_rot_cos, ca_video_rot_sin))
        ca_audio_rotary_emb = self._slice_rope((ca_audio_rot_cos, ca_audio_rot_sin))

        for block in self.transformer_blocks:
            hidden_states, audio_hidden_states = block(
                hidden_states=hidden_states,
                audio_hidden_states=audio_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                audio_encoder_hidden_states=audio_encoder_hidden_states,
                temb=temb,
                temb_audio=temb_audio,
                temb_ca_scale_shift=temb_ca_scale_shift,
                temb_ca_audio_scale_shift=temb_ca_audio_scale_shift,
                temb_ca_gate=temb_ca_gate,
                temb_ca_audio_gate=temb_ca_audio_gate,
                temb_prompt=temb_prompt,
                temb_prompt_audio=temb_prompt_audio,
                video_rotary_emb=video_rotary_emb,
                audio_rotary_emb=audio_rotary_emb,
                ca_video_rotary_emb=ca_video_rotary_emb,
                ca_audio_rotary_emb=ca_audio_rotary_emb,
                encoder_attention_mask=encoder_attention_mask,
                audio_encoder_attention_mask=audio_encoder_attention_mask,
            )

        return hidden_states, audio_hidden_states


def _shard_ltx23_transformer(transformer, tp_degree):
    """TP-shard every LTX-2.3 block. Extends `_shard_ltx2_transformer` with
    the LTX-2.3-only `to_gate_logits` linear per attention module."""
    _shard_ltx2_transformer(transformer, tp_degree)

    if not NEURON_AVAILABLE:
        return

    tp_rank = parallel_state.get_tensor_model_parallel_rank()
    tp_size = parallel_state.get_tensor_model_parallel_size()

    def get_shard(data, dim):
        s = data.shape[dim] // tp_size
        if dim == 0:
            return data[s * tp_rank : s * (tp_rank + 1)].clone()
        return data[:, s * tp_rank : s * (tp_rank + 1)].clone()

    def shard_gate_logits(attn):
        gl = getattr(attn, "to_gate_logits", None)
        if gl is None:
            return
        # to_gate_logits maps inner_dim -> num_heads. We shard both ends: the
        # input (inner_dim) is already row-parallel because it comes from the
        # sharded `to_out`; the output (num_heads) is sharded because we
        # divided attn.heads by tp_size in the parent sharder.
        # Use a plain Linear with column-sharded weight along dim 0
        # (out_features axis) since both in and out are divisible.
        col = ColumnParallelLinear(
            gl.in_features,
            gl.out_features,
            bias=gl.bias is not None,
            gather_output=False,
            dtype=gl.weight.dtype,
        )
        # weight shape: [out_features, in_features]; in_features=inner_dim is
        # already sharded (RowParallel would need it sharded dim=1, but here
        # we ColumnParallel-shard out dim 0 = num_heads).
        col.weight.data = get_shard(gl.weight.data, 0)
        if gl.bias is not None:
            col.bias.data = get_shard(gl.bias.data, 0)
        attn.to_gate_logits = col

    for block in transformer.transformer_blocks:
        for name in (
            "attn1", "attn2", "audio_attn1", "audio_attn2",
            "audio_to_video_attn", "video_to_audio_attn",
        ):
            attn = getattr(block, name, None)
            if attn is not None:
                shard_gate_logits(attn)


# ── ModelWrapper ────────────────────────────────────────────────────────────
class ModelWrapperLTX23Backbone(ModelWrapperLTX2Backbone):
    """ModelWrapper for the LTX-2.3 block-loop backbone.

    Overrides only input_generator to match the LTX-2.3 block-kwargs signature
    (22 tensor inputs; adds temb_prompt / temb_prompt_audio, uses 9× temb).
    """

    @staticmethod
    def _make_attention_mask(bs, text_seq, dtype, valid_tokens=84):
        mask = torch.full((bs, 1, text_seq), -10000.0, dtype=dtype)
        mask[:, :, :valid_tokens] = 0.0
        return mask

    def input_generator(self):
        dtype = self.config.neuron_config.torch_dtype
        inner_dim = self.config.inner_dim
        audio_inner_dim = self.config.audio_inner_dim
        audio_ca_dim = self.config.audio_cross_attention_dim
        video_seq = self.config.video_seq
        audio_seq = self.config.audio_seq
        text_seq = self.config.text_seq
        num_heads = self.config.num_attention_heads
        audio_num_heads = self.config.audio_num_attention_heads
        bs = getattr(self.config, "batch_size", 2)

        # RoPE per-head dims — unchanged from LTX-2.0.
        video_rope_dim = inner_dim // num_heads // 2
        audio_rope_dim = audio_inner_dim // audio_num_heads // 2
        ca_video_rope_dim = audio_ca_dim // num_heads // 2
        ca_audio_rope_dim = audio_ca_dim // audio_num_heads // 2

        # Block-level modulation vector counts (LTX-2.3, with cross_attn_mod):
        #   temb        : 9 × inner_dim  (was 6 in LTX-2.0)
        #   temb_prompt : 2 × inner_dim  (new in LTX-2.3, from prompt_adaln)
        #   temb_ca_scale_shift : 4 × inner_dim  (unchanged)
        #   temb_ca_gate        : 1 × inner_dim  (unchanged)
        video_mod = NUM_MOD_PARAMS_LTX23
        audio_mod = NUM_MOD_PARAMS_LTX23
        prompt_mod = PROMPT_MOD_PARAMS
        ca_ss_mod = A2V_CA_MOD_PARAMS
        ca_gate_mod = A2V_CA_GATE_PARAMS

        model_inputs = (
            # Post-proj_in hidden states
            torch.randn(bs, video_seq, inner_dim, dtype=dtype),           # hidden_states
            torch.randn(bs, audio_seq, audio_inner_dim, dtype=dtype),     # audio_hidden_states
            # Post-connectors encoder hidden states (at inner_dim, not caption_channels)
            torch.randn(bs, text_seq, inner_dim, dtype=dtype),            # encoder_hidden_states
            torch.randn(bs, text_seq, audio_inner_dim, dtype=dtype),      # audio_encoder_hidden_states
            # Global timestep embedding — 9× for LTX-2.3
            torch.randn(bs, 1, video_mod * inner_dim, dtype=dtype),       # temb
            torch.randn(bs, 1, audio_mod * audio_inner_dim, dtype=dtype), # temb_audio
            # a2v cross-attention modulation (unchanged)
            torch.randn(bs, 1, ca_ss_mod * inner_dim, dtype=dtype),       # temb_ca_scale_shift
            torch.randn(bs, 1, ca_ss_mod * audio_inner_dim, dtype=dtype), # temb_ca_audio_scale_shift
            torch.randn(bs, 1, ca_gate_mod * inner_dim, dtype=dtype),     # temb_ca_gate
            torch.randn(bs, 1, ca_gate_mod * audio_inner_dim, dtype=dtype), # temb_ca_audio_gate
            # Prompt modulation (LTX-2.3 only)
            torch.randn(bs, 1, prompt_mod * inner_dim, dtype=dtype),      # temb_prompt
            torch.randn(bs, 1, prompt_mod * audio_inner_dim, dtype=dtype), # temb_prompt_audio
            # RoPE (cos, sin) — as separate tensors (NxDI doesn't trace tuples cleanly)
            torch.randn(bs, num_heads, video_seq, video_rope_dim, dtype=dtype),
            torch.randn(bs, num_heads, video_seq, video_rope_dim, dtype=dtype),
            torch.randn(bs, audio_num_heads, audio_seq, audio_rope_dim, dtype=dtype),
            torch.randn(bs, audio_num_heads, audio_seq, audio_rope_dim, dtype=dtype),
            torch.randn(bs, num_heads, video_seq, ca_video_rope_dim, dtype=dtype),
            torch.randn(bs, num_heads, video_seq, ca_video_rope_dim, dtype=dtype),
            torch.randn(bs, audio_num_heads, audio_seq, ca_audio_rope_dim, dtype=dtype),
            torch.randn(bs, audio_num_heads, audio_seq, ca_audio_rope_dim, dtype=dtype),
            # Attention masks (additive bias, [B, 1, text_seq])
            self._make_attention_mask(bs, text_seq, dtype),
            self._make_attention_mask(bs, text_seq, dtype),
        )

        return [model_inputs]


# ── Key remapping (single-file safetensors -> diffusers key namespace) ──────
# The LTX-2.3 monolithic checkpoint stores every pipeline component
# (audio_vae, spatial_vae, vocoder, text_encoder scalers, DiT) in a single
# file with a top-level prefix `model.diffusion_model.` for DiT tensors.
#
# diffusers 0.39+ (after PR #13217 merged 2026-03-19) has first-class LTX-2.3
# support: the LTX2VideoTransformer3DModel accepts gated_attn, cross_attn_mod,
# perturbed_attn, use_prompt_embeddings flags that enable all LTX-2.3-only
# layers (to_gate_logits, prompt_scale_shift_table, 9-row scale_shift_table,
# prompt_adaln). So we no longer drop or slice anything — every LTX-2.3
# tensor has a slot in the diffusers state_dict.
#
# The translation is now:
#   1. Strip the `model.diffusion_model.` prefix on DiT tensors.
#   2. Drop non-DiT tensors (audio_vae/spatial_vae/vocoder/text_encoder
#      scalers) — they are loaded from the LTX-2 Diffusers pipeline.
#   3. Rename LTX-2.3 → diffusers naming:
#         attn*.q_norm         -> attn*.norm_q
#         attn*.k_norm         -> attn*.norm_k
#         scale_shift_table_a2v_ca_video -> video_a2v_cross_attn_scale_shift_table
#         scale_shift_table_a2v_ca_audio -> audio_a2v_cross_attn_scale_shift_table
#         av_ca_video_scale_shift_adaln_single -> av_cross_attn_video_scale_shift
#         av_ca_audio_scale_shift_adaln_single -> av_cross_attn_audio_scale_shift
#         adaln_single -> time_embed
#         audio_adaln_single -> audio_time_embed
#         prompt_adaln_single -> prompt_adaln
#         audio_prompt_adaln_single -> audio_prompt_adaln

_DIT_PREFIX = "model.diffusion_model."

# Suffix-level renames realign LTX-2.3 naming with diffusers.
_SUFFIX_RENAMES: Tuple[Tuple[str, str], ...] = (
    (".q_norm.weight", ".norm_q.weight"),
    (".k_norm.weight", ".norm_k.weight"),
    (".q_norm.bias", ".norm_q.bias"),
    (".k_norm.bias", ".norm_k.bias"),
    (".scale_shift_table_a2v_ca_video", ".video_a2v_cross_attn_scale_shift_table"),
    (".scale_shift_table_a2v_ca_audio", ".audio_a2v_cross_attn_scale_shift_table"),
)

# Prefix-level renames for the top-level adaln/timestep/patchify modules.
# Applied AFTER stripping _DIT_PREFIX.
_PREFIX_RENAMES: Tuple[Tuple[str, str], ...] = (
    ("av_ca_video_scale_shift_adaln_single.", "av_cross_attn_video_scale_shift."),
    ("av_ca_audio_scale_shift_adaln_single.", "av_cross_attn_audio_scale_shift."),
    ("av_ca_a2v_gate_adaln_single.",          "av_cross_attn_video_a2v_gate."),
    ("av_ca_v2a_gate_adaln_single.",          "av_cross_attn_audio_v2a_gate."),
    ("audio_prompt_adaln_single.",            "audio_prompt_adaln."),
    ("prompt_adaln_single.",                  "prompt_adaln."),
    ("audio_adaln_single.",                   "audio_time_embed."),
    ("adaln_single.",                         "time_embed."),
    # LTX-2.3 renamed the latent/audio patchify projections.
    ("audio_patchify_proj.", "audio_proj_in."),
    ("patchify_proj.",       "proj_in."),
)

# LTX-2.3 top-level modules that live on the pipeline, not the DiT, in
# diffusers 0.39+: the {video,audio}_embeddings_connector heads are wrapped
# by LTX2TextConnectors on the pipeline side. They must be dropped from the
# DiT state dict but later re-loaded into pipe.text_connectors.
_DROP_TOPLEVEL_PREFIXES: Tuple[str, ...] = (
    "audio_embeddings_connector.",
    "video_embeddings_connector.",
    "embeddings_connector.",
)


def _remap_key(k: str) -> Optional[str]:
    """Map an LTX-2.3 checkpoint key to its diffusers-0.39+ equivalent.

    Returns None if the key is not a DiT tensor (audio_vae / spatial_vae /
    vocoder / text_encoder scalers / embeddings_connector, etc.). The
    embeddings_connector lives on the pipeline in diffusers 0.39+ and is
    loaded separately, so it is dropped from the DiT state dict."""
    if not k.startswith(_DIT_PREFIX):
        return None
    k = k[len(_DIT_PREFIX):]

    if any(k.startswith(p) for p in _DROP_TOPLEVEL_PREFIXES):
        return None

    for src, dst in _PREFIX_RENAMES:
        if k.startswith(src):
            k = dst + k[len(src):]
            break

    for src, dst in _SUFFIX_RENAMES:
        if k.endswith(src):
            k = k[: -len(src)] + dst
            break
    return k


def _load_single_file_safetensors(path: str) -> Dict[str, torch.Tensor]:
    """Load DiT tensors from a monolithic safetensors file, applying the
    LTX-2.3 → diffusers-0.39+ key remap on the fly. Uses the streaming
    `safe_open` API so the whole file is never held in memory twice.
    Non-DiT components are skipped."""
    from safetensors import safe_open

    sd: Dict[str, torch.Tensor] = {}
    kept = 0
    dropped_non_dit = 0
    with safe_open(path, framework="pt") as f:
        for raw_key in f.keys():
            mapped = _remap_key(raw_key)
            if mapped is None:
                dropped_non_dit += 1
                continue
            t = f.get_tensor(raw_key)
            # LTX-2.3-fp8: fp8 weights have a .to(bf16) cost at first use.
            # The fp8_cast policy at Lightricks/LTX-2 does exactly this —
            # no per-tensor scale tensor exists in the checkpoint.
            if t.dtype == torch.float8_e4m3fn:
                t = t.to(torch.bfloat16)
            sd[mapped] = t
            kept += 1
    logger.info(
        "Loaded %d DiT tensors from %s (dropped %d non-DiT)",
        kept, path, dropped_non_dit,
    )
    return sd


# Connectors: LTX-2.3 ships these weights inside the monolithic DiT file.
# LTX-2.0's HF repo has an older connectors/ safetensors trained with LTX-2.0.
# To match the LTX-2.3 DiT, we extract LTX-2.3's connector weights and
# overwrite pipe.connectors.load_state_dict(...) at pipeline-load time.
#
# LTX-2.3 key → diffusers LTX2TextConnectors key:
#   model.diffusion_model.{video,audio}_embeddings_connector.transformer_1d_blocks.N.*
#     → {video,audio}_connector.transformer_blocks.N.*
#   model.diffusion_model.{video,audio}_embeddings_connector.learnable_registers
#     → {video,audio}_connector.learnable_registers
_CONNECTOR_MODALITY_MAP = (
    ("video_embeddings_connector.", "video_connector."),
    ("audio_embeddings_connector.", "audio_connector."),
)


def extract_connector_state_dict(path: str) -> Dict[str, torch.Tensor]:
    """Extract the video/audio connector weights from an LTX-2.3 monolithic
    safetensors file and remap them to the
    `diffusers.pipelines.ltx2.connectors.LTX2TextConnectors` key namespace.

    Handles:
      - `model.diffusion_model.{video,audio}_embeddings_connector.*`
        → `{video,audio}_connector.*`
      - `transformer_1d_blocks` → `transformer_blocks`
      - attention norm rename `q_norm`/`k_norm` → `norm_q`/`norm_k`
      - the root-level per-modality text projections stored under
        `text_embedding_projection.{video,audio}_aggregate_embed.*`
        → `{video,audio}_text_proj_in.*`
    """
    from safetensors import safe_open

    out: Dict[str, torch.Tensor] = {}
    with safe_open(path, framework="pt") as f:
        for raw in f.keys():
            # Handle the root-level text_embedding_projection → {*}_text_proj_in rename.
            if raw.startswith("text_embedding_projection."):
                suffix = raw[len("text_embedding_projection."):]
                for mod in ("video", "audio"):
                    agg = f"{mod}_aggregate_embed."
                    if suffix.startswith(agg):
                        new_key = f"{mod}_text_proj_in." + suffix[len(agg):]
                        t = f.get_tensor(raw)
                        if t.dtype == torch.float8_e4m3fn:
                            t = t.to(torch.bfloat16)
                        out[new_key] = t
                        break
                continue

            if not raw.startswith(_DIT_PREFIX):
                continue
            k = raw[len(_DIT_PREFIX):]
            matched = False
            for src, dst in _CONNECTOR_MODALITY_MAP:
                if k.startswith(src):
                    k = dst + k[len(src):]
                    matched = True
                    break
            if not matched:
                continue
            k = k.replace("transformer_1d_blocks.", "transformer_blocks.")
            for old, new in (
                (".q_norm.weight", ".norm_q.weight"),
                (".k_norm.weight", ".norm_k.weight"),
                (".q_norm.bias", ".norm_q.bias"),
                (".k_norm.bias", ".norm_k.bias"),
            ):
                if k.endswith(old):
                    k = k[: -len(old)] + new
                    break
            t = f.get_tensor(raw)
            if t.dtype == torch.float8_e4m3fn:
                t = t.to(torch.bfloat16)
            out[k] = t
    logger.info("Extracted %d connector tensors from %s", len(out), path)
    return out


# Reverse-engineered from LTX-2.3 checkpoint introspection — the LTX-2 HF
# repo's connectors/config.json is for the LTX-2.0 connectors, not the 2.3
# ones. Pass this dict as **kwargs to LTX2TextConnectors(...) when building
# the pipeline's connectors module for LTX-2.3.
LTX23_CONNECTORS_CONFIG = dict(
    caption_channels=3840,
    text_proj_in_factor=49,
    video_connector_num_attention_heads=32,
    video_connector_attention_head_dim=128,
    video_connector_num_layers=8,
    video_connector_num_learnable_registers=128,
    video_gated_attn=True,
    audio_connector_num_attention_heads=32,
    audio_connector_attention_head_dim=64,
    audio_connector_num_layers=8,
    audio_connector_num_learnable_registers=128,
    audio_gated_attn=True,
    rope_type="split",
    per_modality_projections=True,
    video_hidden_dim=4096,
    audio_hidden_dim=2048,
    proj_bias=True,
)


# ── Application ─────────────────────────────────────────────────────────────
class NeuronLTX23BackboneApplication(NeuronLTX2BackboneApplication):
    """NxDI Application for the LTX-2.3 block-loop backbone."""

    _model_cls = NeuronLTX23TransformerBackbone

    def get_model_wrapper_cls(self):
        return ModelWrapperLTX23Backbone

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict, config):
        """Keep only the 48-block weights + SPMDRank. Everything else
        (proj_in, time_embed, caption_projection, norm_out, proj_out,
        scale_shift_table, the audio analogs, and the top-level av_cross_attn_*
        adaLN heads) lives on the pipeline-side CPU transformer for LTX-2.3."""
        state_dict["spmd_rank.rank"] = torch.arange(
            0, config.neuron_config.world_size, dtype=torch.int32
        )
        keep_prefixes = ("transformer_blocks.", "spmd_rank.")
        filtered = {}
        skipped = 0
        for k, v in state_dict.items():
            if k.startswith(keep_prefixes):
                filtered[k] = v.clone().detach().contiguous()
            else:
                skipped += 1
        logger.info(
            "Filtered state dict to %d block-level keys (dropped %d CPU-only)",
            len(filtered), skipped,
        )
        return filtered

    def checkpoint_loader_fn(self, mmap: bool = False):
        # NeuronApplicationBase normalizes model_path as a directory and can
        # append a trailing "/". Strip it so path-type dispatch works whether
        # the caller passed a file, a directory, or a HF repo id.
        model_path = self.model_path.rstrip("/")
        logger.info("Loading LTX-2.3 transformer weights from %s", model_path)

        # Case 1: caller passed a path to a specific .safetensors file.
        if model_path.endswith(".safetensors") and os.path.isfile(model_path):
            sd = _load_single_file_safetensors(model_path)
        # Case 2: caller passed a local directory; pick the requested variant file.
        elif os.path.isdir(model_path):
            import glob

            candidates = sorted(
                glob.glob(os.path.join(model_path, "ltx-2.3-*.safetensors"))
            )
            if not candidates:
                raise FileNotFoundError(
                    f"No ltx-2.3-*.safetensors in {model_path}"
                )
            if len(candidates) > 1:
                logger.warning(
                    "Multiple LTX-2.3 safetensors in %s; using %s",
                    model_path,
                    candidates[0],
                )
            sd = _load_single_file_safetensors(candidates[0])
        else:
            # Case 3: HuggingFace repo id (e.g. "Lightricks/LTX-2.3"). Download
            # the requested single-file variant and load it. Variant is read
            # off the config (default: distilled).
            from huggingface_hub import hf_hub_download

            variant = getattr(self.config, "ltx23_variant", "distilled")
            filename = f"ltx-2.3-22b-{variant}.safetensors"
            logger.info("Downloading %s from %s", filename, model_path)
            local = hf_hub_download(repo_id=model_path, filename=filename)
            sd = _load_single_file_safetensors(local)

        sd = self.convert_hf_to_neuron_state_dict(sd, self.config)
        return sd


# ── NeuronBlockListShim ─────────────────────────────────────────────────────
class _NeuronBlockLoop(nn.Module):
    """Exposes the block-level forward() signature expected by diffusers,
    but routes the call to the compiled Neuron backbone (which internally
    loops over all 48 blocks in a single call)."""

    def __init__(self, backbone_app, num_layers: int):
        super().__init__()
        self.backbone_app = backbone_app
        self.num_layers = num_layers
        # These flags exist on real diffusers blocks; some diffusers code
        # checks them (e.g. self.video_cross_attn_adaln). Safe defaults for
        # LTX-2.3 (cross_attn_mod=True).
        self.video_cross_attn_adaln = True
        self.audio_cross_attn_adaln = True
        self.cross_attn_adaln = True
        self.perturbed_attn = True

    def forward(
        self,
        hidden_states,
        audio_hidden_states,
        encoder_hidden_states,
        audio_encoder_hidden_states,
        temb,
        temb_audio,
        temb_ca_scale_shift,
        temb_ca_audio_scale_shift,
        temb_ca_gate,
        temb_ca_audio_gate,
        temb_prompt=None,
        temb_prompt_audio=None,
        video_rotary_emb=None,
        audio_rotary_emb=None,
        ca_video_rotary_emb=None,
        ca_audio_rotary_emb=None,
        encoder_attention_mask=None,
        audio_encoder_attention_mask=None,
        self_attention_mask=None,
        audio_self_attention_mask=None,
        a2v_cross_attention_mask=None,
        v2a_cross_attention_mask=None,
        use_a2v_cross_attention=True,
        use_v2a_cross_attention=True,
        perturbation_mask=None,
        all_perturbed=None,
    ):
        # Flatten rope tuples to separate cos/sin tensors for the compiled graph.
        video_rot_cos, video_rot_sin = video_rotary_emb
        audio_rot_cos, audio_rot_sin = audio_rotary_emb
        ca_video_rot_cos, ca_video_rot_sin = ca_video_rotary_emb
        ca_audio_rot_cos, ca_audio_rot_sin = ca_audio_rotary_emb

        # Cast to bf16 in case diffusers kept anything in f32 (RoPE output
        # is f32 by default for numerical precision).
        bf16 = torch.bfloat16
        video_rot_cos = video_rot_cos.to(bf16)
        video_rot_sin = video_rot_sin.to(bf16)
        audio_rot_cos = audio_rot_cos.to(bf16)
        audio_rot_sin = audio_rot_sin.to(bf16)
        ca_video_rot_cos = ca_video_rot_cos.to(bf16)
        ca_video_rot_sin = ca_video_rot_sin.to(bf16)
        ca_audio_rot_cos = ca_audio_rot_cos.to(bf16)
        ca_audio_rot_sin = ca_audio_rot_sin.to(bf16)

        # Encoder attention masks reach us as [B, 1, T] additive bias.
        if encoder_attention_mask is None:
            b, t = encoder_hidden_states.shape[0], encoder_hidden_states.shape[1]
            encoder_attention_mask = torch.zeros((b, 1, t), dtype=bf16,
                                                 device=encoder_hidden_states.device)
        if audio_encoder_attention_mask is None:
            b, t = audio_encoder_hidden_states.shape[0], audio_encoder_hidden_states.shape[1]
            audio_encoder_attention_mask = torch.zeros((b, 1, t), dtype=bf16,
                                                       device=audio_encoder_hidden_states.device)

        out = self.backbone_app(
            hidden_states.to(bf16),
            audio_hidden_states.to(bf16),
            encoder_hidden_states.to(bf16),
            audio_encoder_hidden_states.to(bf16),
            temb.to(bf16),
            temb_audio.to(bf16),
            temb_ca_scale_shift.to(bf16),
            temb_ca_audio_scale_shift.to(bf16),
            temb_ca_gate.to(bf16),
            temb_ca_audio_gate.to(bf16),
            temb_prompt.to(bf16) if temb_prompt is not None else
                torch.zeros_like(temb[..., : 2 * (temb.shape[-1] // 9)]),
            temb_prompt_audio.to(bf16) if temb_prompt_audio is not None else
                torch.zeros_like(temb_audio[..., : 2 * (temb_audio.shape[-1] // 9)]),
            video_rot_cos, video_rot_sin,
            audio_rot_cos, audio_rot_sin,
            ca_video_rot_cos, ca_video_rot_sin,
            ca_audio_rot_cos, ca_audio_rot_sin,
            encoder_attention_mask.to(bf16),
            audio_encoder_attention_mask.to(bf16),
        )
        return out


class NeuronBlockListShim(nn.ModuleList):
    """Drop-in replacement for `LTX2VideoTransformer3DModel.transformer_blocks`.

    Looks like an `nn.ModuleList` of length `num_layers` so diffusers'
    `for block_idx, block in enumerate(self.transformer_blocks)` iterates as
    expected. Only the *first* block actually executes — it runs the full
    compiled 48-block loop on Neuron. The remaining 47 are no-ops that
    just return their inputs.

    Install with: `pipe.transformer.transformer_blocks = NeuronBlockListShim(...)`.
    """

    def __init__(self, backbone_app, num_layers: int):
        super().__init__([_NeuronBlockLoop(backbone_app, num_layers)])
        # Extend to look like num_layers blocks. We add identity blocks after
        # the first so diffusers' `enumerate(self.transformer_blocks)` loops
        # `num_layers` times — but only the first call does work.
        for _ in range(num_layers - 1):
            self.append(_IdentityBlock())


class _IdentityBlock(nn.Module):
    """A block-shaped no-op used to pad `NeuronBlockListShim` out to 48 entries."""

    def forward(self, hidden_states, audio_hidden_states, **kwargs):
        return hidden_states, audio_hidden_states
