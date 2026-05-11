"""
NxDI LTX-2.3 Application
========================
Top-level orchestrator for the LTX-2.3 video+audio diffusion model on Neuron.

Mirrors `NeuronLTX2Application` exactly except that the DiT backbone is
`NeuronLTX23BackboneApplication` (which handles LTX-2.3's monolithic
single-file safetensors layout). The text encoder (Gemma 3-12B), the VAE,
the vocoder, and the scheduler are pulled from the `Lightricks/LTX-2`
Diffusers repo because LTX-2.3's Diffusers multi-folder layout is not
published yet and the architectures of those components are identical.

If/when Lightricks publishes a Diffusers layout for LTX-2.3, switch
`pipeline_repo` below from "Lightricks/LTX-2" to "Lightricks/LTX-2.3".
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Optional

import torch

# Cross-import from the LTX-2 package for the pipeline wrapper and base app.
_LTX2_SRC = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "ltx2-video-audio", "src")
)
if _LTX2_SRC not in sys.path:
    sys.path.insert(0, _LTX2_SRC)

from application import NeuronLTX2Application  # noqa: E402
from modeling_ltx2_3 import (  # noqa: E402
    LTX23BackboneInferenceConfig,
    NeuronLTX23BackboneApplication,
)

logger = logging.getLogger(__name__)


class NeuronLTX23Application(NeuronLTX2Application):
    """LTX-2.3 top-level app. Only the backbone class differs from LTX-2."""

    def __init__(
        self,
        model_path: str,
        backbone_config: LTX23BackboneInferenceConfig,
        transformer_path: Optional[str] = None,
        height: int = 384,
        width: int = 512,
        num_frames: int = 25,
        num_inference_steps: int = 8,
        instance_type: str = "trn2",
        pipeline_repo: str = "Lightricks/LTX-2",
    ):
        # Skip NeuronLTX2Application.__init__ so we can use the 2.3 backbone
        # (the parent __init__ hardcodes NeuronLTX2BackboneApplication).
        # We inherit compile/load/__call__ unchanged.
        torch.nn.Module.__init__(self)
        self.model_path = model_path
        self.transformer_path = transformer_path or model_path
        self.height = height
        self.width = width
        self.num_frames = num_frames
        self.num_inference_steps = num_inference_steps
        self.instance_type = instance_type
        self.backbone_config = backbone_config

        from pipeline import NeuronLTX2Pipeline  # lazy: diffusers import is heavy

        logger.info(
            "Loading Diffusers LTX2Pipeline from %s (for Gemma3/VAE/vocoder/scheduler)",
            pipeline_repo,
        )
        self.pipe = NeuronLTX2Pipeline.from_pretrained(
            pipeline_repo,
            torch_dtype=torch.bfloat16,
        )

        self.pipe.neuron_backbone = NeuronLTX23BackboneApplication(
            model_path=self.transformer_path,
            config=self.backbone_config,
        )
