"""Capability vocabulary and the deterministic provider registry.

Phase 0 orchestration: every implemented capability resolves to exactly one
registered provider. Known-but-unimplemented capabilities stay listed for
truthful reporting but never resolve to a provider, so unsupported requests
can never fall through to VQA.

Providers carry identity metadata; runtime readiness comes from the model in
the existing name-keyed registry, which remains the execution interface.
"""

from dataclasses import dataclass
from threading import Lock

from core.router import CHANGE_UNSUPPORTED
from models.base import ModelReadiness

# Stable internal capability identifiers.
SINGLE_IMAGE_VQA = "single_image_vqa"
GROUNDING = "grounding"
CHANGE_VQA = "change_vqa"
OPTICAL_SAR = "optical_sar"

# Capabilities with at least one registered provider.
IMPLEMENTED_CAPABILITIES: frozenset[str] = frozenset(
    {SINGLE_IMAGE_VQA, GROUNDING, CHANGE_VQA, OPTICAL_SAR}
)

# Capabilities on the roadmap: known vocabulary, no provider, never resolvable.
UNAVAILABLE_CAPABILITIES: tuple[str, ...] = ()

# The full advertised vocabulary: implemented capabilities first.
KNOWN_CAPABILITIES: tuple[str, ...] = (
    SINGLE_IMAGE_VQA,
    GROUNDING,
    CHANGE_VQA,
    OPTICAL_SAR,
)


class UnknownCapability(RuntimeError):
    """The requested capability is not part of the known vocabulary."""


class CapabilityUnavailable(RuntimeError):
    """A known capability cannot be executed."""


class ProviderNotReady(CapabilityUnavailable):
    def __init__(self, capability: str, provider: str | None, readiness: ModelReadiness):
        self.capability = capability
        self.provider = provider
        self.reason_code = readiness.reason_code or "PROVIDER_UNAVAILABLE"
        self.detail = readiness.detail or "Provider cannot execute now."
        super().__init__(self.detail)


@dataclass(frozen=True)
class Provider:
    """A provider advertised for specific capabilities.

    ``model_name`` is the stable key in the existing model registry, which
    stays the execution interface; this record supplies the capability binding
    and truthful identity metadata for routing and tracing.
    """

    name: str
    version: str
    capabilities: frozenset[str]
    model_name: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("Provider name must be a non-empty string.")
        if not isinstance(self.version, str) or not self.version:
            raise ValueError("Provider version must be a non-empty string.")
        if not isinstance(self.model_name, str) or not self.model_name:
            raise ValueError("Provider model_name must be a non-empty string.")
        unknown = self.capabilities - IMPLEMENTED_CAPABILITIES
        if unknown:
            raise ValueError(
                f"Provider advertises unimplemented capabilities: {sorted(unknown)}"
            )


@dataclass(frozen=True)
class ResolvedProvider:
    """The immutable outcome of a capability resolution."""

    capability: str
    provider_name: str
    model_name: str
    model_version: str


_PROVIDERS: dict[str, Provider] = {}
_CAPABILITY_BINDINGS: dict[str, str] = {}  # capability -> provider name
_LOCK = Lock()


def register_provider(provider: Provider) -> None:
    """Register a provider for its advertised capabilities.

    A capability already bound to a different provider raises instead of
    silently overriding the existing binding. Re-registering an equal
    provider is an idempotent no-op so repeat imports stay deterministic;
    a different definition under a bound name also raises.
    """
    with _LOCK:
        existing = _PROVIDERS.get(provider.name)
        if existing is not None:
            if existing == provider:
                return
            raise ValueError(f"Provider name already registered: {provider.name}")
        for capability in provider.capabilities:
            bound = _CAPABILITY_BINDINGS.get(capability)
            if bound is not None and bound != provider.name:
                raise ValueError(
                    f"Capability already registered to another provider: {capability}"
                )
        _PROVIDERS[provider.name] = provider
        for capability in provider.capabilities:
            _CAPABILITY_BINDINGS[capability] = provider.name


def resolve_provider(capability: str) -> ResolvedProvider:
    """Resolve a capability to its registered provider, failing explicitly."""
    if capability not in KNOWN_CAPABILITIES:
        raise UnknownCapability(f"Unknown capability: {capability}")
    with _LOCK:
        name = _CAPABILITY_BINDINGS.get(capability)
        provider = _PROVIDERS.get(name) if name is not None else None
    if provider is None:
        raise CapabilityUnavailable(
            f"No provider is registered for capability: {capability}"
        )
    return ResolvedProvider(
        capability=capability,
        provider_name=provider.name,
        model_name=provider.model_name,
        model_version=provider.version,
    )


def capabilities_status() -> list[dict[str, object]]:
    """Registration and local runtime readiness from one source of truth."""
    return [{"name": name, **capability_readiness(name)} for name in KNOWN_CAPABILITIES]


def capability_readiness(capability: str) -> dict[str, object]:
    """Check a registered model without loading weights or contacting a service."""
    try:
        resolved = resolve_provider(capability)
    except CapabilityUnavailable:
        return {
            "registered": False, "available": False, "state": "NOT_IMPLEMENTED",
            "provider": None, "reason_code": "NO_PROVIDER",
            "detail": "No real provider is registered for this capability.",
        }
    if capability == CHANGE_VQA:
        return {
            "registered": True, "available": False, "state": "UNAVAILABLE",
            "provider": resolved.provider_name, "reason_code": "not_answerable",
            "detail": CHANGE_UNSUPPORTED,
        }
    from orchestrator.registry import get

    try:
        readiness = get(resolved.model_name).readiness()
        if not isinstance(readiness, ModelReadiness):
            raise TypeError
    except Exception:
        readiness = ModelReadiness(False, "PROVIDER_UNAVAILABLE", "Provider readiness could not be checked.")
    return {
        "registered": True, "available": readiness.available,
        "state": "AVAILABLE" if readiness.available else "UNAVAILABLE",
        "provider": resolved.provider_name,
        "reason_code": None if readiness.available else readiness.reason_code or "PROVIDER_UNAVAILABLE",
        "detail": None if readiness.available else readiness.detail or "Provider cannot execute now.",
    }


def require_provider_ready(capability: str) -> ResolvedProvider:
    resolved = resolve_provider(capability)
    status = capability_readiness(capability)
    if not status["available"]:
        raise ProviderNotReady(
            capability, resolved.provider_name,
            ModelReadiness(False, str(status["reason_code"]), str(status["detail"])),
        )
    return resolved


def reset_registry() -> None:
    """Test-only hook that clears all registrations."""
    with _LOCK:
        _PROVIDERS.clear()
        _CAPABILITY_BINDINGS.clear()


def register_default_providers() -> None:
    """Bind the current production providers to their capabilities.

    The adapter only reads identity metadata from the existing singleton;
    Qwen internals, weights, and scientific behavior are untouched, and the
    name-keyed model registry remains the execution interface.
    """
    from orchestrator.registry import get

    model = get("qwen2.5vl-3b")
    register_provider(
        Provider(
            name=model.name,
            version=model.version,
            capabilities=frozenset({SINGLE_IMAGE_VQA}),
            model_name="qwen2.5vl-3b",
        )
    )
    model = get("optical-sar-deterministic")
    register_provider(
        Provider(
            name=model.name,
            version=model.version,
            capabilities=frozenset({OPTICAL_SAR}),
            model_name="optical-sar-deterministic",
        )
    )
    model = get("change-deterministic")
    register_provider(
        Provider(
            name=model.name,
            version=model.version,
            capabilities=frozenset({CHANGE_VQA}),
            model_name="change-deterministic",
        )
    )
    model = get("grounding-dino-swint")
    register_provider(
        Provider(
            name=model.name,
            version=model.version,
            capabilities=frozenset({GROUNDING}),
            model_name="grounding-dino-swint",
        )
    )


register_default_providers()
