"""Inference worker: killable process isolation for heavyweight models.

Spawning a child costs about a second, so these tests share a few workers and
keep model-side delays short. ``WARM`` is a generous ceiling for a request
that may have to spawn a fresh worker; the timeouts under test are short.
"""

import os
import threading
import time
from typing import Any

import pytest

from models.base import Model
from models.change import ChangeModel
from models.grounding_dino import GroundingDINOModel
from models.mock import MockModel
from models.optical_sar import OpticalSARModel
from models.qwen_vl import QwenVLModel
from orchestrator import router, worker
from orchestrator.capabilities import SINGLE_IMAGE_VQA
from orchestrator.worker import (
    InferenceWorker,
    ModelExecutionTimeout,
    RemoteInferenceError,
    WorkerCrashed,
)

FAKES = "orchestrator._test_fakes:get"
WARM = 30.0
LIVE = {"scene_id": "scene_a", "sensor": None, "execution_mode": "live"}


@pytest.fixture
def fake_worker():
    instance = InferenceWorker(factory=FAKES, kill_grace_seconds=0.5)
    yield instance
    instance.shutdown()


def ask(instance: InferenceWorker, question: str, timeout: float = WARM) -> dict:
    return instance.run("qwen2.5vl-3b", ["/tmp/scene.png"], question, timeout)


def pid_of(result: dict) -> int:
    return int(result["answer"].split()[0].removeprefix("pid="))


def test_only_gpu_models_are_isolated() -> None:
    assert Model.isolated is False
    assert QwenVLModel.isolated is True
    assert GroundingDINOModel.isolated is True
    for deterministic in (ChangeModel, OpticalSARModel, MockModel):
        assert deterministic.isolated is False


def test_hung_job_is_killed_and_next_request_runs_promptly(fake_worker) -> None:
    first = ask(fake_worker, "hello")
    assert ask(fake_worker, "hello")["answer"] == f"pid={pid_of(first)} calls=2"

    started = time.monotonic()
    with pytest.raises(ModelExecutionTimeout):
        ask(fake_worker, "sleep:60", timeout=0.3)
    assert time.monotonic() - started < 3.0
    assert fake_worker.pid is None

    started = time.monotonic()
    after = ask(fake_worker, "hello", timeout=10.0)
    assert time.monotonic() - started < 10.0
    assert pid_of(after) != pid_of(first)
    assert after["answer"].endswith("calls=1")  # fresh process, not queued behind the hung one


def test_waiting_for_a_busy_worker_times_out_without_killing_it(fake_worker) -> None:
    ask(fake_worker, "hello")
    owner: dict[str, Any] = {}

    def long_job() -> None:
        owner["result"] = ask(fake_worker, "sleep:1.0", timeout=WARM)

    thread = threading.Thread(target=long_job)
    thread.start()
    time.sleep(0.2)
    with pytest.raises(ModelExecutionTimeout):
        ask(fake_worker, "hello", timeout=0.2)
    thread.join(5)
    assert owner["result"]["answer"].startswith("pid=")


def test_worker_crash_is_reported_and_next_request_respawns(fake_worker) -> None:
    before = pid_of(ask(fake_worker, "hello"))
    with pytest.raises(WorkerCrashed, match=r"exit code 7.*qwen2\.5vl-3b"):
        ask(fake_worker, "crash:7")
    after = pid_of(ask(fake_worker, "hello"))
    assert after != before


def test_infer_exception_message_and_type_propagate(fake_worker) -> None:
    message = "Qwen2.5-VL inference requires a CUDA GPU for this baseline"
    with pytest.raises(RemoteInferenceError) as caught:
        ask(fake_worker, f"raise:{message}")
    assert isinstance(caught.value, RuntimeError)
    assert str(caught.value) == message
    assert caught.value.remote_type == "ValueError"
    assert "ValueError" in str(caught.value.__cause__)
    with pytest.raises(RemoteInferenceError, match="could not be returned"):
        ask(fake_worker, "unpicklable")
    assert ask(fake_worker, "hello")["answer"].endswith("calls=3")  # still warm


def test_unimportable_factory_reports_instead_of_dying() -> None:
    instance = InferenceWorker(factory="orchestrator._no_such_module:get")
    try:
        with pytest.raises(RemoteInferenceError, match="_no_such_module"):
            ask(instance, "hello")
    finally:
        instance.shutdown()


class InProcessModel:
    version = "in-process-test"
    isolated = False

    def __init__(self) -> None:
        self.pids: list[int] = []

    def infer(self, image_paths: list[str], question: str) -> dict:
        self.pids.append(os.getpid())
        if question.startswith("sleep:"):
            time.sleep(float(question.removeprefix("sleep:")))
        return {"answer": "in process", "evidence": []}


@pytest.mark.usefixtures("ready_providers")
def test_non_isolated_model_runs_in_process_and_does_not_block_later_calls(
    monkeypatch,
) -> None:
    model = InProcessModel()
    monkeypatch.setattr(router, "get", lambda _: model)
    monkeypatch.setattr(
        worker, "run", lambda *_, **__: pytest.fail("worker used for in-process model")
    )

    with pytest.raises(ModelExecutionTimeout):
        router.route(SINGLE_IMAGE_VQA, ["/tmp/scene.png"], "sleep:1", LIVE, timeout_seconds=0.05)
    started = time.monotonic()
    result = router.route(SINGLE_IMAGE_VQA, ["/tmp/scene.png"], "go", LIVE, timeout_seconds=0.5)
    assert time.monotonic() - started < 0.5
    assert result["answer"] == "in process"
    assert result["trace"]["model_version"] == "in-process-test"
    assert set(model.pids) == {os.getpid()}


@pytest.mark.usefixtures("ready_providers")
def test_isolated_model_routes_through_worker_with_in_process_metadata(
    monkeypatch, fake_worker
) -> None:
    class Isolated(InProcessModel):
        isolated = True

        def infer(self, image_paths: list[str], question: str) -> dict:
            raise AssertionError("isolated model must not run in the server process")

    monkeypatch.setattr(router, "get", lambda _: Isolated())
    monkeypatch.setattr(worker, "_DEFAULT_WORKER", fake_worker)

    result = router.route(SINGLE_IMAGE_VQA, ["/tmp/scene.png"], "hello", LIVE, timeout_seconds=WARM)
    assert pid_of(result) != os.getpid()
    assert result["evidence"] == [{"model": "qwen2.5vl-3b", "n_images": 1}]
    assert result["trace"]["model_version"] == "in-process-test"
    with pytest.raises(router.ModelExecutionTimeout):
        router.route(SINGLE_IMAGE_VQA, ["/tmp/scene.png"], "sleep:60", LIVE, timeout_seconds=0.3)
