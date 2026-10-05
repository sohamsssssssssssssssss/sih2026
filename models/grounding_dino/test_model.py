import builtins
import sys
from types import ModuleType, SimpleNamespace

import pytest

import models.grounding_dino.model as grounding_module
from models.grounding_dino import GroundingDINOModel
from models.artifacts import ArtifactStatus


def test_infer_returns_normalized_bounding_box_contract(tmp_path, monkeypatch) -> None:
    image = tmp_path / "scene.png"
    image.touch()
    model = GroundingDINOModel()
    monkeypatch.setattr(
        model,
        "_predict",
        lambda *_: ([[0.5, 0.5, 0.4, 0.2]], [0.87], ["bridge"]),
    )

    result = model.infer([str(image)], "the bridge")

    assert result == {
        "answer": "Found 1 match for 'the bridge'.",
        "evidence": [
            {
                "type": "bounding_box",
                "label": "bridge",
                "coordinates": pytest.approx([0.3, 0.4, 0.7, 0.6]),
                "coordinate_space": "normalized_xyxy",
                "confidence": 0.87,
                "source_scene_id": None,
            }
        ],
    }


@pytest.mark.parametrize(
    "boxes,scores,message",
    [
        ([[float("nan"), 0.5, 0.2, 0.2]], [0.8], "non-finite"),
        ([[0.5, 0.5, -0.2, 0.2]], [0.8], "out-of-range"),
        ([[0.5, 0.5, 0.2, 0.2]], [float("inf")], "non-finite"),
        ([[0.5, 0.5, 0.2]], [0.8], "malformed box"),
    ],
)
def test_infer_rejects_malformed_evidence(tmp_path, monkeypatch, boxes, scores, message) -> None:
    image = tmp_path / "scene.png"
    image.touch()
    model = GroundingDINOModel()
    monkeypatch.setattr(model, "_predict", lambda *_: (boxes, scores, ["bridge"]))

    with pytest.raises(ValueError, match=message):
        model.infer([str(image)], "bridge")


def test_infer_reports_no_match_when_evidence_is_empty(tmp_path, monkeypatch) -> None:
    image = tmp_path / "scene.png"
    image.touch()
    model = GroundingDINOModel()
    monkeypatch.setattr(model, "_predict", lambda *_: ([], [], []))

    assert model.infer([str(image)], "bridge") == {
        "answer": "No match found for 'bridge'.",
        "evidence": [],
    }


def test_missing_grounding_package_fails_closed(monkeypatch) -> None:
    original_import = builtins.__import__

    def missing_grounding(name, *args, **kwargs):
        if name == "groundingdino":
            raise ImportError("not installed")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_grounding)
    with pytest.raises(RuntimeError, match="requires groundingdino"):
        GroundingDINOModel()._load()


def test_missing_cuda_fails_closed(monkeypatch) -> None:
    groundingdino = ModuleType("groundingdino")
    groundingdino.__file__ = __file__
    inference = ModuleType("groundingdino.util.inference")
    inference.load_image = object()
    inference.load_model = object()
    inference.predict = object()
    monkeypatch.setitem(sys.modules, "groundingdino", groundingdino)
    monkeypatch.setitem(sys.modules, "groundingdino.util", ModuleType("groundingdino.util"))
    monkeypatch.setitem(sys.modules, "groundingdino.util.inference", inference)
    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    )

    with pytest.raises(RuntimeError, match="requires a CUDA GPU"):
        GroundingDINOModel()._load()


def test_readiness_fails_closed_without_dependency(monkeypatch) -> None:
    monkeypatch.setattr(grounding_module, "find_spec", lambda name: None if name == "groundingdino" else object())
    readiness = GroundingDINOModel().readiness()
    assert readiness.available is False
    assert readiness.reason_code == "DEPENDENCY_UNAVAILABLE"


def test_readiness_fails_closed_without_cuda(monkeypatch) -> None:
    monkeypatch.setattr(grounding_module, "find_spec", lambda _: object())
    groundingdino = ModuleType("groundingdino")
    groundingdino.__file__ = __file__
    monkeypatch.setitem(sys.modules, "groundingdino", groundingdino)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)))
    readiness = GroundingDINOModel().readiness()
    assert readiness.available is False
    assert readiness.reason_code == "CUDA_UNAVAILABLE"


def test_readiness_fails_closed_without_local_checkpoint(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(grounding_module, "find_spec", lambda _: object())
    package = tmp_path / "groundingdino"
    config = package / "config" / "GroundingDINO_SwinT_OGC.py"
    config.parent.mkdir(parents=True)
    config.touch()
    groundingdino = ModuleType("groundingdino")
    groundingdino.__file__ = str(package / "__init__.py")
    hub = ModuleType("huggingface_hub")
    hub.try_to_load_from_cache = lambda *_args: None
    monkeypatch.setitem(sys.modules, "groundingdino", groundingdino)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True)))
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    readiness = GroundingDINOModel().readiness()
    assert readiness.available is False
    assert readiness.reason_code == "ARTIFACT_UNAVAILABLE"


def test_packaged_config_discovery(tmp_path, monkeypatch) -> None:
    package = tmp_path / "groundingdino"
    config = package / "config" / "GroundingDINO_SwinT_OGC.py"
    config.parent.mkdir(parents=True)
    config.touch()
    groundingdino = ModuleType("groundingdino")
    groundingdino.__file__ = str(package / "__init__.py")
    monkeypatch.setitem(sys.modules, "groundingdino", groundingdino)

    assert GroundingDINOModel()._resolve_config() == config.resolve()


def test_missing_explicit_config_fails_readiness(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(grounding_module, "find_spec", lambda _: object())
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True)),
    )
    readiness = GroundingDINOModel(config_path=tmp_path / "missing.py").readiness()

    assert readiness.available is False
    assert readiness.reason_code == "NOT_CONFIGURED"
    assert readiness.detail == (
        "Grounding DINO configuration is unavailable; set "
        "SATQUERY_GROUNDING_CONFIG or install the packaged configuration."
    )
    assert str(tmp_path) not in readiness.detail


def test_readiness_and_load_use_same_resolved_config(tmp_path, monkeypatch) -> None:
    first_config = tmp_path / "first.py"
    first_config.touch()
    second_config = tmp_path / "second.py"
    second_config.touch()
    checkpoint = tmp_path / "groundingdino_swint_ogc.pth"
    checkpoint.write_bytes(b"weights")
    loaded_with = []

    class FakeModel:
        def to(self, device):
            assert device == "cuda"

        def parameters(self):
            return iter([SimpleNamespace(device=SimpleNamespace(type="cuda"))])

    inference = ModuleType("groundingdino.util.inference")
    inference.load_image = object()
    inference.predict = object()
    inference.load_model = lambda config, weights, device: (
        loaded_with.append((config, weights, device)) or FakeModel()
    )
    monkeypatch.setattr(grounding_module, "find_spec", lambda _: object())
    monkeypatch.setattr(
        grounding_module,
        "validate_artifact",
        lambda _: ArtifactStatus(True, path=checkpoint),
    )
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True)),
    )
    monkeypatch.setitem(sys.modules, "groundingdino", ModuleType("groundingdino"))
    monkeypatch.setitem(sys.modules, "groundingdino.util", ModuleType("groundingdino.util"))
    monkeypatch.setitem(sys.modules, "groundingdino.util.inference", inference)
    monkeypatch.setattr(grounding_module, "_text_encoder_cached", lambda: True)
    monkeypatch.setenv("SATQUERY_GROUNDING_CONFIG", str(first_config))
    model = GroundingDINOModel()

    assert model.readiness().available
    monkeypatch.setenv("SATQUERY_GROUNDING_CONFIG", str(second_config))
    model._load()

    assert loaded_with == [(str(first_config.resolve()), str(checkpoint), "cuda")]


def test_load_fails_if_model_remains_on_cpu(tmp_path, monkeypatch) -> None:
    config = tmp_path / "GroundingDINO_SwinT_OGC.py"
    config.touch()
    checkpoint = tmp_path / "groundingdino_swint_ogc.pth"
    checkpoint.write_bytes(b"weights")

    class CpuModel:
        def to(self, _device):
            return self

        def parameters(self):
            return iter([SimpleNamespace(device=SimpleNamespace(type="cpu"))])

    inference = ModuleType("groundingdino.util.inference")
    inference.load_image = object()
    inference.predict = object()
    inference.load_model = lambda *_args, **_kwargs: CpuModel()
    monkeypatch.setattr(
        grounding_module,
        "validate_artifact",
        lambda _: ArtifactStatus(True, path=checkpoint),
    )
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True)),
    )
    monkeypatch.setitem(sys.modules, "groundingdino", ModuleType("groundingdino"))
    monkeypatch.setitem(sys.modules, "groundingdino.util", ModuleType("groundingdino.util"))
    monkeypatch.setitem(sys.modules, "groundingdino.util.inference", inference)

    with pytest.raises(RuntimeError, match="could not be placed on CUDA"):
        GroundingDINOModel(config_path=config)._load()


def test_load_preserves_bounded_checkpoint_failure_reason(tmp_path, monkeypatch) -> None:
    config = tmp_path / "GroundingDINO_SwinT_OGC.py"
    config.touch()
    checkpoint = tmp_path / "groundingdino_swint_ogc.pth"
    checkpoint.write_bytes(b"weights")
    inference = ModuleType("groundingdino.util.inference")
    inference.load_image = object()
    inference.predict = object()
    inference.load_model = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        RuntimeError("CUDA out of memory\n" + "x" * 500)
    )
    monkeypatch.setattr(
        grounding_module,
        "validate_artifact",
        lambda _: ArtifactStatus(True, path=checkpoint),
    )
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True)),
    )
    monkeypatch.setitem(sys.modules, "groundingdino", ModuleType("groundingdino"))
    monkeypatch.setitem(sys.modules, "groundingdino.util", ModuleType("groundingdino.util"))
    monkeypatch.setitem(sys.modules, "groundingdino.util.inference", inference)

    with pytest.raises(RuntimeError) as raised:
        GroundingDINOModel(config_path=config)._load()

    assert "RuntimeError: CUDA out of memory" in str(raised.value)
    assert "\n" not in str(raised.value)
    assert len(str(raised.value)) < 340


def _ready_except_text_encoder(tmp_path, monkeypatch, cached_files):
    config = tmp_path / "GroundingDINO_SwinT_OGC.py"
    config.touch()
    hub = ModuleType("huggingface_hub")
    hub.try_to_load_from_cache = lambda repo, name: (
        str(tmp_path / name) if repo == "bert-base-uncased" and name in cached_files else None
    )
    monkeypatch.setattr(grounding_module, "find_spec", lambda _: object())
    monkeypatch.setattr(
        grounding_module,
        "validate_artifact",
        lambda _: ArtifactStatus(True, path=tmp_path / "groundingdino_swint_ogc.pth"),
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True)))
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    return GroundingDINOModel(config_path=config).readiness()


def test_readiness_fails_closed_without_cached_text_encoder(tmp_path, monkeypatch) -> None:
    readiness = _ready_except_text_encoder(tmp_path, monkeypatch, {"config.json", "vocab.txt"})
    assert readiness.available is False
    assert readiness.reason_code == "ARTIFACT_UNAVAILABLE"
    assert "bert-base-uncased" in readiness.detail


def test_readiness_passes_with_cached_text_encoder(tmp_path, monkeypatch) -> None:
    readiness = _ready_except_text_encoder(
        tmp_path, monkeypatch, {"config.json", "vocab.txt", "model.safetensors"}
    )
    assert readiness.available is True
