"""Contract shared by every model used by the orchestrator."""

from abc import ABC, abstractmethod


class Model(ABC):
    """Minimal interface exposed to orchestration code."""

    name: str = "base"
    version: str = "0.0.0"

    @abstractmethod
    def infer(self, image_paths: list[str], question: str) -> dict:
        """Return answer, confidence, and structured supporting evidence."""
        raise NotImplementedError
