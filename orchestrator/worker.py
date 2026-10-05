"""Out-of-process execution for heavyweight models that can hang.

A thread cannot be cancelled once a CUDA kernel or a long ``generate`` call is
running, so a hung in-process inference used to hold the only executor slot and
every later request timed out without ever starting. Models that declare
``isolated = True`` instead run in a long-lived child process:

* The child hosts the model registry, so loaded weights stay warm between
  requests.
* One request runs at a time per worker (guarded by a lock). Time spent
  waiting for the lock counts against the caller's timeout; a caller that
  never got the lock times out without touching the worker.
* When the request that owns the worker exceeds its timeout, the child is
  terminated (then killed after a short grace period), which also releases
  its GPU memory. The next request lazily spawns a fresh worker, paying the
  weight-loading cost again.
* If the child dies on its own (segfault, OOM kill, ``os._exit``), the
  in-flight request fails with ``WorkerCrashed`` and the next one respawns.
* An exception raised by ``infer`` in the child is re-raised in the parent as
  ``RemoteInferenceError`` (a ``RuntimeError``) carrying the original message
  verbatim, so callers that classify failures by message substring keep
  working. The original type name is kept on ``remote_type`` and the child's
  traceback is chained as ``__cause__`` for logging.

The ``spawn`` start method is used so the child never inherits a CUDA context
from the parent. This module imports only the standard library; the model
factory (``module:attribute``) is imported in the child.
"""

from __future__ import annotations

import atexit
import importlib
import itertools
import multiprocessing
import signal
import threading
import time
import traceback
from multiprocessing.connection import Connection, wait
from typing import Any, Callable

DEFAULT_FACTORY = "orchestrator.registry:get"
DEFAULT_KILL_GRACE_SECONDS = 1.0


class ModelExecutionTimeout(RuntimeError):
    """The request stopped waiting for an in-flight model execution."""


class WorkerCrashed(RuntimeError):
    """The inference worker process exited while a request was in flight."""


class RemoteInferenceError(RuntimeError):
    """``infer`` raised inside the worker; the message is the original one."""

    def __init__(self, message: str, remote_type: str) -> None:
        super().__init__(message)
        self.remote_type = remote_type


class _RemoteTraceback(Exception):
    """Carries the child's formatted traceback as an exception cause."""

    def __init__(self, text: str) -> None:
        super().__init__(text)
        self.text = text

    def __str__(self) -> str:
        return self.text


def _resolve_factory(spec: str) -> Callable[[str], Any]:
    module_name, _, attribute = spec.partition(":")
    if not module_name or not attribute:
        raise ValueError(f"Model factory must be 'module:attribute', got {spec!r}")
    return getattr(importlib.import_module(module_name), attribute)


def _serve(conn: Connection, factory_spec: str) -> None:
    """Child process loop: receive requests, run ``infer``, send replies."""
    # The parent owns shutdown; a terminal Ctrl-C must not race it.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    factory: Callable[[str], Any] | None = None
    factory_error: tuple[str, str, str] | None = None
    try:
        factory = _resolve_factory(factory_spec)
    except Exception as exc:  # Reported on every request instead of dying silently.
        factory_error = (type(exc).__name__, str(exc), traceback.format_exc())
    while True:
        try:
            message = conn.recv()
        except (EOFError, OSError):
            return  # Parent went away.
        if message is None:
            return
        request_id, model_name, image_paths, question = message
        if factory is None:
            assert factory_error is not None
            reply_error: tuple[Any, ...] = (request_id, "error", *factory_error)
            try:
                conn.send(reply_error)
            except (EOFError, OSError):
                return
            continue
        try:
            model = factory(model_name)
            result = model.infer(image_paths=image_paths, question=question)
            reply: tuple[Any, ...] = (request_id, "ok", result)
        except BaseException as exc:  # noqa: BLE001 - everything is reported back
            reply = (
                request_id,
                "error",
                type(exc).__name__,
                str(exc),
                traceback.format_exc(),
            )
        try:
            conn.send(reply)
        except (EOFError, OSError):
            return
        except Exception as exc:  # e.g. an unpicklable result
            conn.send(
                (
                    request_id,
                    "error",
                    type(exc).__name__,
                    f"Model result could not be returned from the worker: {exc}",
                    traceback.format_exc(),
                )
            )


class InferenceWorker:
    """A lazily spawned, restartable inference process."""

    def __init__(
        self,
        factory: str = DEFAULT_FACTORY,
        kill_grace_seconds: float = DEFAULT_KILL_GRACE_SECONDS,
    ) -> None:
        self._factory = factory
        self._kill_grace_seconds = kill_grace_seconds
        self._context = multiprocessing.get_context("spawn")
        self._lock = threading.Lock()
        self._process: Any | None = None
        self._conn: Connection | None = None
        self._request_ids = itertools.count(1)

    @property
    def pid(self) -> int | None:
        process = self._process
        return process.pid if process is not None and process.is_alive() else None

    def run(
        self,
        model_name: str,
        image_paths: list[str],
        question: str,
        timeout: float,
    ) -> Any:
        """Run ``registry.get(model_name).infer(...)`` in the worker process."""
        deadline = time.monotonic() + timeout
        if not self._lock.acquire(timeout=max(timeout, 0.0)):
            # Someone else's job holds the worker; it is theirs to time out.
            raise ModelExecutionTimeout
        try:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ModelExecutionTimeout
            conn, process = self._ensure_started()
            request_id = next(self._request_ids)
            try:
                conn.send((request_id, model_name, list(image_paths), question))
            except (EOFError, OSError) as exc:
                self._stop()
                raise WorkerCrashed(
                    f"Inference worker became unreachable before running model "
                    f"'{model_name}'."
                ) from exc
            return self._await_reply(conn, process, request_id, model_name, deadline)
        finally:
            self._lock.release()

    def _await_reply(
        self,
        conn: Connection,
        process: Any,
        request_id: int,
        model_name: str,
        deadline: float,
    ) -> Any:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._stop()
                raise ModelExecutionTimeout
            ready = wait([conn, process.sentinel], timeout=remaining)
            if conn in ready:
                try:
                    reply = conn.recv()
                except (EOFError, OSError):
                    reply = None
                if reply is not None:
                    if reply[0] != request_id:
                        continue  # Stale reply; cannot normally happen.
                    if reply[1] == "ok":
                        return reply[2]
                    _, _, remote_type, message, remote_traceback = reply
                    raise RemoteInferenceError(message, remote_type) from _RemoteTraceback(
                        remote_traceback
                    )
            if conn in ready or process.sentinel in ready:
                process.join(self._kill_grace_seconds)
                exitcode = process.exitcode
                self._stop()
                raise WorkerCrashed(
                    f"Inference worker exited unexpectedly (exit code {exitcode}) "
                    f"while running model '{model_name}'."
                )

    def _ensure_started(self) -> tuple[Connection, Any]:
        process = self._process
        if process is not None and self._conn is not None and process.is_alive():
            return self._conn, process
        self._stop()  # Reap a worker that died between requests.
        parent_conn, child_conn = self._context.Pipe()
        process = self._context.Process(
            target=_serve,
            args=(child_conn, self._factory),
            name="satquery-inference-worker",
            daemon=True,
        )
        process.start()
        child_conn.close()
        self._process, self._conn = process, parent_conn
        return parent_conn, process

    def _stop(self) -> None:
        """Terminate the current worker, escalating to kill. Caller holds the lock."""
        process, conn = self._process, self._conn
        self._process = self._conn = None
        if process is not None:
            if process.is_alive():
                process.terminate()
                process.join(self._kill_grace_seconds)
            if process.is_alive():
                process.kill()
                process.join(self._kill_grace_seconds)
            try:
                process.close()
            except ValueError:
                pass  # Still running after kill; leave it to the OS.
        if conn is not None:
            conn.close()

    def shutdown(self, timeout: float = 2.0) -> None:
        """Ask the worker to exit, terminating it if it does not."""
        acquired = self._lock.acquire(timeout=timeout)
        try:
            process, conn = self._process, self._conn
            if process is not None and conn is not None and process.is_alive():
                try:
                    conn.send(None)
                except (EOFError, OSError):
                    pass
                process.join(timeout)
            self._stop()
        finally:
            if acquired:
                self._lock.release()


_DEFAULT_WORKER = InferenceWorker()
atexit.register(_DEFAULT_WORKER.shutdown)


def default_worker() -> InferenceWorker:
    return _DEFAULT_WORKER


def run(model_name: str, image_paths: list[str], question: str, timeout: float) -> Any:
    """Run an isolated model in the shared default worker."""
    return _DEFAULT_WORKER.run(model_name, image_paths, question, timeout)
