"""Deterministic placeholder model."""

from models.base import Model


class MockModel(Model):
    """Return canned responses until production models are registered."""

    name = "mock"
    version = "1.0.0"

    def infer(self, image_paths: list[str], question: str) -> dict:
        return {
            "answer": f"mock answer for: {question}",
            "confidence": 0.5,
            "evidence": [],
        }
