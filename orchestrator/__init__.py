"""Contract-only model orchestration."""

from orchestrator.registry import get, register
from orchestrator.router import route

__all__ = ["get", "register", "route"]
