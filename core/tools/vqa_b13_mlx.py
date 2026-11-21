"""Ask adapter for the pinned, out-of-process Mac VQA model."""
from __future__ import annotations

import json
import os
from pathlib import Path
import socket
import subprocess
import time

from core.registry import ROOT, ToolNeeds
from core.router import VQA_MLX
from core.schema import Evidence, ToolResult
from core.tools.base import BaseTool, materialize_paths

MODEL_REV = "66285546d2b821cf421d4f5eb2576359d3770cd3"
ADAPTER_SHA = "7f0b1ebc84106b485784e9a45a96e624dc31a1f7698b76efac174f1db86e393a"
LABEL = ("experimental — RSVQA-LR fresh test (n=400): B13 73.25% vs base 54.50% "
         "(majority 57.25%); trained on 10 m-class imagery")
UNAVAILABLE = "Mac VQA is unavailable. Check the pinned MLX weights and worker startup."


def enabled():
    return os.environ.get("SATQUERY_VQA_MLX") == "1"


class VQAWorker:
    def __init__(self):
        self.process = None
        self.channel = None
        self.reader = None

    @property
    def ready(self):
        return self.process is not None and self.process.poll() is None

    def start(self):
        python = Path.home() / "models/.venv-satquery-mac-vqa/bin/python"
        self.channel, child = socket.socketpair()
        try:
            with child:
                self.process = subprocess.Popen(
                    [str(python), "-m", "scripts.vqa_b13_worker", str(child.fileno())],
                    cwd=ROOT, pass_fds=(child.fileno(),), stdin=subprocess.DEVNULL,
                    env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
                )
            self.channel.settimeout(60)
            self.reader = self.channel.makefile("rb")
            if self._receive() != {"ready": True}:
                raise RuntimeError(UNAVAILABLE)
            self.channel.settimeout(10)
        except (OSError, ValueError, RuntimeError):
            self.close()
            raise

    def _receive(self):
        line = self.reader.readline(8192)
        if not line.endswith(b"\n"):
            raise RuntimeError(UNAVAILABLE)
        return json.loads(line)

    def ask(self, path, question):
        try:
            self.channel.sendall(json.dumps({"path": path, "question": question}).encode() + b"\n")
            response = self._receive()
            answer = response.get("answer")
            if not isinstance(answer, str) or not answer.strip():
                raise RuntimeError(UNAVAILABLE)
            return answer
        except (AttributeError, OSError, ValueError, RuntimeError) as error:
            self.close()
            raise RuntimeError(UNAVAILABLE) from error

    def close(self):
        if self.process is not None:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
            self.process = None
        if self.reader is not None:
            self.reader.close()
            self.reader = None
        if self.channel is not None:
            self.channel.close()
            self.channel = None


class B13MLXTool(BaseTool):
    name = VQA_MLX
    version = MODEL_REV
    needs = ToolNeeds(sensors=("S2", "S1S2", "RGB_HR"))

    def __init__(self, worker=None):
        super().__init__()
        self.worker = worker if worker is not None else VQAWorker()

    @property
    def unavailable_reason(self):
        return None if self.worker.ready else UNAVAILABLE

    def run(self, scene, query, pixels):
        started = time.perf_counter()
        try:
            with materialize_paths(pixels, count=1) as paths:
                answer = self.worker.ask(str(Path(paths[0]).resolve()), query.text)
        except (OSError, RuntimeError) as error:
            return self.error_result(error, started)
        result = self._result(started, {"answer": answer, "label": LABEL}, evidence=(
            Evidence(kind="text", summary={"label": LABEL, "model_revision": MODEL_REV,
                                           "adapter_sha256": ADAPTER_SHA, "runtime": "8-bit MLX"}),
        ))
        return ToolResult.model_validate({**result.model_dump(), "checkpoint_sha": ADAPTER_SHA})
