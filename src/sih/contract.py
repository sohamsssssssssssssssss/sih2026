"""The model contract. Every track implements this and nothing else.

The orchestrator routes on this interface alone and never knows what is inside
a model. A1, A2, and A3 each ship a class satisfying RSModel; if you need to
change this interface, it is a team decision, not a track decision.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Protocol, runtime_checkable


@dataclass
class Evidence:
    """Why the model gave this answer. Feeds the audit trace the PS requires."""
    kind: str                       # "box" | "band" | "frame" | "region"
    value: Any                      # [x1,y1,x2,y2] normalised | band name | frame idx
    score: float = 1.0


@dataclass
class Prediction:
    answer: str
    confidence: float               # [0,1], CALIBRATED — see Stage 4
    evidence: list[Evidence] = field(default_factory=list)
    model_name: str = ""
    model_version: str = ""
    latency_ms: float = 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["evidence"] = [asdict(e) if not isinstance(e, dict) else e
                         for e in self.evidence]
        return d


@runtime_checkable
class RSModel(Protocol):
    """Implemented by every track's served model."""

    name: str
    version: str

    def supports(self, image_paths: list[str], question: str) -> float:
        """Return routing confidence in [0,1] for this query.

        The orchestrator picks the highest scorer. Be honest here — a model
        that claims everything breaks routing for everyone.
        """
        ...

    def infer(self, image_paths: list[str], question: str) -> Prediction:
        ...
