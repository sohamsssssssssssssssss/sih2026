"""Production tool adapters. Importing this module loads no model weights."""

from core.tools.change import ChangeTool
from core.tools.grounding_dino import GroundingDINOTool
from core.tools.optical_sar import OpticalSARTool
from core.tools.qwen_vl import QwenVLTool

TOOLS = (QwenVLTool(), GroundingDINOTool(), OpticalSARTool(), ChangeTool())

__all__ = ["TOOLS", "QwenVLTool", "GroundingDINOTool", "OpticalSARTool", "ChangeTool"]
