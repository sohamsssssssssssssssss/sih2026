"""Registry of model instances addressable by stable names."""

from models.base import Model
from models.change import ChangeModel
from models.grounding_dino import GroundingDINOModel
from models.mock import MockModel
from models.optical_sar import OpticalSARModel
from models.qwen_vl import QwenVLModel
from models.qwen_vl.cdvqa import CdvqaChangeModel

_MODELS: dict[str, Model] = {}


def register(name: str, model_instance: Model) -> None:
    if not isinstance(model_instance, Model):
        raise TypeError("model_instance must implement Model")
    _MODELS[name] = model_instance


def get(name: str) -> Model:
    try:
        return _MODELS[name]
    except KeyError as exc:
        available = ", ".join(sorted(_MODELS)) or "none"
        raise KeyError(f"Unknown model '{name}'. Registered models: {available}") from exc


def names() -> list[str]:
    return sorted(_MODELS)


register("mock", MockModel())
register("qwen2.5vl-3b", QwenVLModel())
register("grounding-dino-swint", GroundingDINOModel())
register("optical-sar-deterministic", OpticalSARModel())
register("change-deterministic", ChangeModel())
register("qwen2.5vl-3b-cdvqa", CdvqaChangeModel())
