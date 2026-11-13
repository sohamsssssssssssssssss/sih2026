"""Change-VQA route: flag off keeps routing unchanged; flag on runs the CDVQA path with a stub runner."""
import json
from io import BytesIO
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend import services
from backend.main import app
from models.qwen_vl import cdvqa
from orchestrator import router
from orchestrator import trace as trace_store
from orchestrator.planner import PlanRequest, plan_request
from orchestrator.registry import get

EVAL = cdvqa.ROOT / "data/manifests/cdvqa/eval-test1-400.v1.jsonl"
QUESTION = "Have the areas of water decreased?"


@pytest.fixture(autouse=True)
def runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ready_providers):
    trace_store._TRACE.clear()
    trace_store._LOADED_PATH = None
    monkeypatch.setattr(trace_store, "TRACE_PATH", tmp_path / "trace.jsonl")
    for name, folder in (("INGESTED_SCENE_DIR", "scenes"), ("INGESTED_RASTER_DIR", "rasters"),
                         ("SCENE_MANIFEST_DIR", "manifests")):
        monkeypatch.setattr(services, name, tmp_path / folder)
    monkeypatch.delenv(cdvqa.FLAG_ENV, raising=False)
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}")
    (adapter / "adapter_model.safetensors").write_bytes(b"synthetic")
    monkeypatch.setenv(cdvqa.ADAPTER_ENV, str(adapter))
    yield
    trace_store._TRACE.clear()


class Calls(list):
    answer = ["Yes"]


@pytest.fixture
def calls(monkeypatch):
    seen = Calls()
    seen.answer = ["Yes"]
    model = get(cdvqa.CdvqaChangeModel.name)

    def stub(before, after, question, qtype):
        seen.append((before, after, question, qtype))
        return seen.answer[0]
    monkeypatch.setattr(model, "runner", stub)
    # The worker would host the same registry; run it in-process for CPU tests.
    monkeypatch.setattr(router.inference_worker, "run",
                        lambda name, paths, question, timeout: get(name).infer(paths, question))
    return seen


@pytest.fixture
def client():
    return TestClient(app)


def upload(client, name, size=(64, 64), **metadata):
    data = BytesIO()
    Image.new("RGB", size, (20, 80, 140)).save(data, format="PNG")
    response = client.post("/api/scenes", files={"file": (name, data.getvalue(), "image/png")}, data=metadata)
    assert response.status_code == 201, response.text
    return response.json()["scene_id"]


def ask(client, question, *scenes):
    body = {"scene_id": scenes[0], "question": question, "execution_mode": "live"}
    if len(scenes) > 1:
        body["scene_id_2"] = scenes[1]
    return client.post("/api/analyze", json=body)


def test_flag_off_never_plans_the_cdvqa_path(monkeypatch):
    questions = [json.loads(line)["question"] for line in EVAL.read_text().splitlines()]
    for value in (None, "0"):
        if value is None:
            monkeypatch.delenv(cdvqa.FLAG_ENV, raising=False)
        else:
            monkeypatch.setenv(cdvqa.FLAG_ENV, value)
        for question in questions + ["What changed between these images?"]:
            plan = plan_request(PlanRequest(question=question, scene_ids=("a", "b")))
            assert plan.rule_id not in (cdvqa.RULE, cdvqa.UNSUPPORTED_RULE)
            assert plan.provider != cdvqa.CdvqaChangeModel.name


def test_flag_off_api_does_not_call_the_adapter(client, monkeypatch):
    model = get(cdvqa.CdvqaChangeModel.name)
    monkeypatch.setattr(model, "runner", lambda *a: pytest.fail("adapter called with flag off"))
    first, second = upload(client, "before.png"), upload(client, "after.png")
    response = ask(client, QUESTION, first, second)
    assert "change_vqa" not in response.text or response.json().get("change_vqa") is None


def test_flag_on_answers_with_adapter_identity_and_unvalidated_status(client, monkeypatch, calls):
    monkeypatch.setenv(cdvqa.FLAG_ENV, "1")
    first, second = upload(client, "before.png"), upload(client, "after.png")
    response = ask(client, QUESTION, first, second)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["answer"] == "Yes" and body["answer_source"] == "cdvqa_adapter"
    change = body["change_vqa"]
    assert change["question_type"] == "decrease_or_not" and change["answer_status"] == "ok"
    assert change["validation_status"] == "experimental_unvalidated"
    assert change["adapter"]["id"] == "cdvqa-g2-checkpoint-1000"
    assert len(change["adapter"]["adapter_model_sha256"]) == 64
    assert "not validated on Cartosat or RISAT" in change["caveat"]
    assert "experimental_unvalidated" in body["notice"]
    assert body["trace"]["params"]["change_vqa"] == change
    before, after, question, qtype = calls[0]
    assert Path(before).stem == first and Path(after).stem == second  # request order = before, after


def test_template_question_that_does_not_parse_as_change_still_routes(client, monkeypatch, calls):
    monkeypatch.setenv(cdvqa.FLAG_ENV, "1")
    calls.answer[0] = "buildings"
    response = ask(client, "What is the largest change?", upload(client, "a.png"), upload(client, "b.png"))
    assert response.status_code == 200, response.text
    assert response.json()["change_vqa"]["question_type"] == "largest_change"


def test_out_of_vocabulary_answer_is_shown_as_malformed(client, monkeypatch, calls):
    monkeypatch.setenv(cdvqa.FLAG_ENV, "1")
    calls.answer[0] = "probably yes"
    body = ask(client, QUESTION, upload(client, "a.png"), upload(client, "b.png")).json()
    assert body["answer"] == "malformed: probably yes"
    assert body["change_vqa"]["answer_status"] == "malformed"


def test_validated_config_flips_status(client, monkeypatch, calls, tmp_path):
    monkeypatch.setenv(cdvqa.FLAG_ENV, "1")
    config = json.loads(cdvqa.CONFIG_PATH.read_text()) | {"cdvqa_validated": True}
    path = tmp_path / "change_vqa.json"
    path.write_text(json.dumps(config))
    monkeypatch.setattr(cdvqa, "CONFIG_PATH", path)
    body = ask(client, QUESTION, upload(client, "a.png"), upload(client, "b.png")).json()
    assert body["change_vqa"]["validation_status"] == "validated"


@pytest.mark.parametrize("question,names,sizes,code", [
    ("What changed between the two dates?", ("a.png", "b.png"), ((64, 64), (64, 64)), "unsupported_change_question"),
    (QUESTION, ("a.png", "b.png"), ((64, 64), (160, 64)), "pair_refused"),
    (QUESTION, ("a_gsd0.5.png", "b_gsd2.png"), ((64, 64), (64, 64)), "pair_refused"),
])
def test_refusals_never_call_the_model(client, monkeypatch, calls, question, names, sizes, code):
    monkeypatch.setenv(cdvqa.FLAG_ENV, "1")
    scenes = [upload(client, name, size) for name, size in zip(names, sizes)]
    response = ask(client, question, *scenes)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["change_vqa"]["refusal"] == code and body["answer"].startswith("Refused: ")
    assert body["model"]["name"] == cdvqa.GATE
    assert calls == []


def test_one_image_or_sar_upload_is_refused(client, monkeypatch, calls):
    monkeypatch.setenv(cdvqa.FLAG_ENV, "1")
    one = ask(client, QUESTION, upload(client, "a.png"))
    assert one.status_code == 422 and "two scenes" in one.text
    sar = ask(client, QUESTION, upload(client, "a.png"), upload(client, "s.png", modality="sar"))
    assert sar.status_code == 422 and "SAR" in sar.text
    assert calls == []
