"""Test-only model factory importable by the spawned inference worker.

Spawned children re-import modules from scratch, so in-test monkeypatching of
the registry is invisible to them. Tests instead point
``InferenceWorker(factory="orchestrator._test_fakes:get")`` here. The question
text scripts the behaviour:

* ``sleep:<seconds>``  - block, then answer
* ``crash:<code>``     - ``os._exit`` without replying (simulates OOM/segfault)
* ``raise:<message>``  - raise ``ValueError(message)``
* ``unpicklable``      - return a result that cannot cross the pipe
* anything else        - answer immediately with the worker's pid and call count
"""

import os
import threading
import time

_CALLS = 0


class ScriptedModel:
    version = "test-fake"
    isolated = True

    def __init__(self, name: str) -> None:
        self.name = name

    def infer(self, image_paths: list[str], question: str) -> dict:
        global _CALLS
        _CALLS += 1
        command, _, argument = question.partition(":")
        if command == "sleep":
            time.sleep(float(argument))
        elif command == "crash":
            os._exit(int(argument or 3))
        elif command == "raise":
            raise ValueError(argument)
        elif command == "unpicklable":
            return {"answer": "lock", "evidence": [threading.Lock()]}
        return {
            "answer": f"pid={os.getpid()} calls={_CALLS}",
            "evidence": [{"model": self.name, "n_images": len(image_paths)}],
        }


def get(name: str) -> ScriptedModel:
    return ScriptedModel(name)
