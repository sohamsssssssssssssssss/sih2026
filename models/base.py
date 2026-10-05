"""Contract shared by every model used by the orchestrator."""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelReadiness:
    available: bool
    reason_code: str | None = None
    detail: str | None = None


class Model(ABC):
    """Minimal interface exposed to orchestration code."""

    name: str = "base"
    version: str = "0.0.0"
    # True for heavyweight (GPU) models whose inference can hang: under a
    # timeout the orchestrator runs them in a killable worker process
    # (orchestrator.worker) instead of an in-process thread.
    isolated: bool = False

    def readiness(self) -> ModelReadiness:
        """Report local prerequisites without loading weights or running inference."""
        return ModelReadiness(False, "NOT_CONFIGURED", "Provider readiness is not configured.")

    @abstractmethod
    def infer(self, image_paths: list[str], question: str) -> dict:
        """Return answer, confidence, and structured supporting evidence."""
        raise NotImplementedError
