"""B13 contract checks without pretrained weights or an accelerator."""
import sys
from types import SimpleNamespace

import pytest

from models.qwen_vl import b13
from models.qwen_vl.model import QwenVLModel
from models.qwen_vl.test_device_gate import fake_torch
from orchestrator import router, trace
from orchestrator.capabilities import ProviderNotReady
from models.grounding_dino import GroundingDINOModel


@pytest.fixture
def enabled(monkeypatch, tmp_path):
    monkeypatch.setenv(b13.ADAPTER_ENV, str(tmp_path))
    return tmp_path


def test_flag_off_keeps_prompt_and_result(monkeypatch, tmp_path):
    monkeypatch.delenv(b13.ADAPTER_ENV, raising=False)
    image = tmp_path / "image.png"
    image.touch()
    model = QwenVLModel()
    def generate(paths, question):
        assert question == "How many buildings? " + b13.ANSWER_INSTRUCTION
        return " Yes "
    monkeypatch.setattr(model, "_generate_answer", generate)
    assert model.infer([str(image)], "How many buildings?") == {
        "answer": "Yes", "confidence": None, "evidence": []}


def test_prompt_matches_frozen_training_contract(tmp_path):
    image = tmp_path / "scene.tif"
    assert b13.messages(image, "Is it rural?") == [{"role": "user", "content": [
        {"type": "image", "image": image.as_uri(), "resized_height": 392, "resized_width": 392},
        {"type": "text", "text": "Is it rural? " + b13.ANSWER_INSTRUCTION}]}]


@pytest.mark.parametrize("filename", list(b13.ADAPTER_HASHES))
def test_hash_mismatch_refuses_load(enabled, monkeypatch, filename):
    for name in b13.ADAPTER_HASHES:
        (enabled / name).write_bytes(b"corrupt")
    # Both files independently fail, before any heavyweight model can be loaded.
    hashes = dict(b13.ADAPTER_HASHES)
    for name in hashes:
        if name != filename:
            import hashlib
            hashes[name] = hashlib.sha256(b"corrupt").hexdigest()
    monkeypatch.setattr(b13, "ADAPTER_HASHES", hashes)
    with pytest.raises(ValueError, match=filename):
        b13.adapter_identity(enabled)
    assert QwenVLModel().readiness().reason_code == "ADAPTER_INVALID"
    monkeypatch.setitem(sys.modules, "peft", SimpleNamespace(PeftModel=None))
    with pytest.raises(ValueError, match=filename):
        QwenVLModel()._load_adapter(None, None, "/unused")


def test_adapter_loader_pins_base_and_applies_peft_with_mps_fallback(enabled, monkeypatch):
    from scripts import stage1_runtime
    identity = {"name": "B13", "files_sha256": dict(b13.ADAPTER_HASHES)}
    monkeypatch.setattr(b13, "adapter_identity", lambda _: identity)
    monkeypatch.setattr(stage1_runtime, "verify_checkpoint", lambda path, revision, hashes: {
        "passed": revision == b13.BASE_REVISION and len(hashes) == 2})
    placements = []
    class Model:
        def to(self, device):
            placements.append(device)
            if device == "mps":
                raise RuntimeError("MPS unsupported")
            return self
    def load(path, **kwargs):
        assert kwargs == {"torch_dtype": "fp16", "local_files_only": True}
        return Model()
    def attach(base, path, **kwargs):
        assert path == str(enabled)
        assert kwargs == {"local_files_only": True, "is_trainable": False,
                          "autocast_adapter_dtype": False}
        return base
    monkeypatch.setitem(sys.modules, "peft", SimpleNamespace(
        PeftModel=SimpleNamespace(from_pretrained=attach)))
    torch = fake_torch(cuda=False, mps=True)
    torch.float16 = "fp16"
    torch.mps = SimpleNamespace(empty_cache=lambda: None)
    monkeypatch.setenv("SATQUERY_ALLOW_MPS", "1")
    model = QwenVLModel()
    model._load_adapter(torch, SimpleNamespace(from_pretrained=load), "/base")
    assert placements == ["mps", "cpu"]
    assert model.provenance == {**identity, "device": "cpu", "dtype": "float16",
                                "quantization": None, "mps_fallback": "MPS unsupported"}


def test_bad_base_refuses_before_model_load(enabled, monkeypatch):
    from scripts import stage1_runtime
    monkeypatch.setattr(b13, "adapter_identity", lambda _: {})
    monkeypatch.setattr(stage1_runtime, "verify_checkpoint", lambda *a: {"passed": False})
    monkeypatch.setitem(sys.modules, "peft", SimpleNamespace(PeftModel=None))
    with pytest.raises(ValueError, match="pinned base"):
        QwenVLModel()._load_adapter(fake_torch(cuda=False, mps=False), None, "/base")


def test_identity_survives_worker_trace_and_response(enabled, monkeypatch, ready_providers):
    from backend.services import _live_response
    identity = {"name": "B13", "files_sha256": dict(b13.ADAPTER_HASHES), "device": "mps"}
    model = QwenVLModel()
    image = enabled / "scene.png"
    image.touch()
    model.provenance = identity
    monkeypatch.setattr(model, "_generate_answer", lambda paths, question: " Yes ")
    worker_result = model.infer([str(image)], "Is a building present?")
    assert worker_result["answer"] == "Yes"  # runner strips; does not lowercase generated text
    monkeypatch.setattr(router.inference_worker, "run", lambda *a: worker_result)
    result = router.route("single_image_vqa", [str(image)], "Is a building present?",
                          params={"execution_mode": "live", "model_provenance": {"forged": True}},
                          timeout_seconds=120)
    assert result["trace"]["params"]["model_provenance"] == identity
    assert result["trace"]["params"]["answer_source"] == "b13_adapter"
    assert _live_response(result)["model_provenance"] == identity
    assert trace.verify_chain()[0]


@pytest.mark.parametrize("count", [0, 2])
@pytest.mark.parametrize("question", ["How many square buildings are there?",
                                      "What is the amount of square buildings?"])
def test_counts_use_detector_boxes(enabled, monkeypatch, ready_providers, count, question):
    from orchestrator.planner import plan_request, PlanRequest
    plan = plan_request(PlanRequest(question, ("scene",)))
    assert plan.selected_capability == "grounding"
    assert plan.rule_id == "b13_detection_count"
    boxes = [{"type": "bounding_box", "coordinates": [0, 0, 1, 1]} for _ in range(count)]
    def detect(self, image_paths, question):
        assert question == "square buildings"
        return {"answer": "Found objects", "evidence": boxes}
    monkeypatch.setattr(GroundingDINOModel, "infer", detect)
    monkeypatch.setattr(QwenVLModel, "infer", lambda self, **kw: pytest.fail("used adapter for count"))
    result = router.route("single_image_vqa", ["scene"], question, params={"execution_mode": "live"})
    assert result["answer"] == str(count)
    assert result["count"] == count
    assert result["evidence"] == boxes
    assert result["trace"]["model_name"] == "grounding-dino-swint"
    assert result["trace"]["params"]["answer_source"] == "grounding_dino_count"


def test_unavailable_detector_never_falls_back_to_adapter(enabled, monkeypatch):
    from models.base import ModelReadiness
    monkeypatch.setattr(GroundingDINOModel, "readiness", lambda self: ModelReadiness(False, "CUDA_UNAVAILABLE", "No CUDA"))
    monkeypatch.setattr(QwenVLModel, "infer", lambda self, **kw: pytest.fail("used adapter"))
    with pytest.raises(ProviderNotReady, match="No CUDA"):
        router.route("single_image_vqa", ["scene"], "What is the number of buildings?",
                     params={"execution_mode": "live"})
    assert trace.records() == []


@pytest.mark.parametrize("question", ["Are there more buildings than roads?", "Is it rural?"])
def test_non_count_questions_are_not_detection_counts(question):
    assert b13.count_target(question) is None


def test_adapter_generation_uses_eval_resize_prompt_and_decoding(enabled, monkeypatch):
    from contextlib import nullcontext
    captured = {}
    class Inputs(dict):
        input_ids = [[1, 2]]
        def to(self, device):
            assert device == "cpu"
            return self
    class Processor:
        def apply_chat_template(self, messages, **kwargs):
            captured["messages"] = messages
            assert kwargs == {"tokenize": False, "add_generation_prompt": True}
            return "prompt"
        def __call__(self, **kwargs):
            assert kwargs == {"text": ["prompt"], "images": ["image"], "videos": None,
                              "return_tensors": "pt"}
            return Inputs()
        def batch_decode(self, tokens, **kwargs):
            assert tokens == [[3]]
            assert kwargs == {"skip_special_tokens": True, "clean_up_tokenization_spaces": False}
            return ["YES"]
    def generate(**kwargs):
        assert kwargs == {"do_sample": False, "max_new_tokens": 32}
        return [[1, 2, 3]]
    model = QwenVLModel()
    model._processor = Processor()
    model._model = SimpleNamespace(device="cpu", generate=generate)
    model._process_vision_info = lambda messages: (["image"], None)
    monkeypatch.setattr(model, "_load", lambda: None)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(inference_mode=nullcontext))
    assert model._generate_answer([str(enabled / "scene.tif")], "Is it rural?") == "YES"
    assert captured["messages"] == b13.messages(enabled / "scene.tif", "Is it rural?")


def test_generation_mps_failure_retries_on_cpu_and_records_it(enabled, monkeypatch):
    image = enabled / "scene.tif"
    image.touch()
    model = QwenVLModel()
    model.device = "mps"
    model.provenance = {"name": "B13", "device": "mps"}
    moved = []
    model._model = SimpleNamespace(to=lambda device: moved.append(device))
    def generate(*args):
        if model.device == "mps":
            raise RuntimeError("unsupported MPS operation")
        return "rural"
    monkeypatch.setattr(model, "_generate_answer", generate)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(mps=SimpleNamespace(empty_cache=lambda: None)))
    result = model.infer([str(image)], "Is it rural or urban?")
    assert result["answer"] == "rural"
    assert moved == ["cpu"]
    assert result["model_provenance"]["device"] == "cpu"
    assert result["model_provenance"]["mps_fallback"] == "unsupported MPS operation"
