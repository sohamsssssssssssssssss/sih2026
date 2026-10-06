import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import unittest
from unittest.mock import patch

from models.base import ModelReadiness
from models.change import ChangeModel
from models.grounding_dino import GroundingDINOModel
from models.optical_sar import OpticalSARModel
from models.qwen_vl import QwenVLModel

from orchestrator import capabilities
from orchestrator.capabilities import (
    CHANGE_VQA,
    GROUNDING,
    CapabilityUnavailable,
    KNOWN_CAPABILITIES,
    OPTICAL_SAR,
    Provider,
    ProviderNotReady,
    SINGLE_IMAGE_VQA,
    UnknownCapability,
    resolve_provider,
)
from orchestrator.router import route
from orchestrator import trace as trace_store


class CapabilityRegistryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.qwen_ready = patch.object(QwenVLModel, "readiness", lambda _: ModelReadiness(True))
        self.grounding_ready = patch.object(GroundingDINOModel, "readiness", lambda _: ModelReadiness(True))
        self.optical_sar_ready = patch.object(OpticalSARModel, "readiness", lambda _: ModelReadiness(True))
        self.change_ready = patch.object(ChangeModel, "readiness", lambda _: ModelReadiness(True))
        self.qwen_ready.start()
        self.grounding_ready.start()
        self.optical_sar_ready.start()
        self.change_ready.start()
        capabilities.reset_registry()
        capabilities.register_default_providers()

    def tearDown(self) -> None:
        capabilities.reset_registry()
        capabilities.register_default_providers()
        self.grounding_ready.stop()
        self.optical_sar_ready.stop()
        self.change_ready.stop()
        self.qwen_ready.stop()

    def test_known_vocabulary_matches_spec(self) -> None:
        self.assertEqual(
            KNOWN_CAPABILITIES,
            (SINGLE_IMAGE_VQA, GROUNDING, CHANGE_VQA, OPTICAL_SAR),
        )
        self.assertEqual(
            capabilities.IMPLEMENTED_CAPABILITIES,
            {SINGLE_IMAGE_VQA, GROUNDING, CHANGE_VQA, OPTICAL_SAR},
        )

    def test_single_image_vqa_resolves_to_qwen_provider(self) -> None:
        resolved = resolve_provider(SINGLE_IMAGE_VQA)
        self.assertEqual(resolved.capability, SINGLE_IMAGE_VQA)
        self.assertEqual(resolved.provider_name, "qwen2.5vl-3b")
        self.assertEqual(resolved.model_name, "qwen2.5vl-3b")

    def test_provider_metadata_is_truthful(self) -> None:
        resolved = resolve_provider(SINGLE_IMAGE_VQA)
        self.assertEqual(resolved.model_version, "Qwen/Qwen2.5-VL-3B-Instruct")

    def test_grounding_resolves_to_grounding_dino_provider(self) -> None:
        resolved = resolve_provider(GROUNDING)
        self.assertEqual(resolved.capability, GROUNDING)
        self.assertEqual(resolved.provider_name, "grounding-dino-swint")
        self.assertEqual(resolved.model_name, "grounding-dino-swint")
        self.assertEqual(
            resolved.model_version,
            "ShilongLiu/GroundingDINO:groundingdino_swint_ogc.pth",
        )

    def test_optical_sar_resolves_to_deterministic_provider(self) -> None:
        resolved = resolve_provider(OPTICAL_SAR)
        self.assertEqual(resolved.provider_name, "optical-sar-deterministic")
        self.assertEqual(resolved.model_name, "optical-sar-deterministic")
        self.assertEqual(
            resolved.model_version,
            "sentinel2-indices__sentinel1-backscatter-v1",
        )

    def test_change_resolves_to_deterministic_provider(self) -> None:
        resolved = resolve_provider(CHANGE_VQA)
        self.assertEqual(resolved.provider_name, "change-deterministic")
        self.assertEqual(resolved.model_name, "change-deterministic")
        self.assertEqual(resolved.model_version, "bitemporal-difference-v2")

    def test_unknown_capability_fails_clearly(self) -> None:
        with self.assertRaises(UnknownCapability):
            resolve_provider("time_travel")

    def test_unregistered_capability_fails_as_unavailable(self) -> None:
        capabilities.reset_registry()
        with self.assertRaises(CapabilityUnavailable):
            resolve_provider(SINGLE_IMAGE_VQA)

    def test_duplicate_registration_does_not_override(self) -> None:
        challenger = Provider(
            name="challenger",
            version="9.9.9",
            capabilities=frozenset({SINGLE_IMAGE_VQA}),
            model_name="mock",
        )
        with self.assertRaises(ValueError):
            capabilities.register_provider(challenger)
        self.assertEqual(resolve_provider(SINGLE_IMAGE_VQA).provider_name, "qwen2.5vl-3b")

    def test_duplicate_same_name_registration_is_deterministic(self) -> None:
        original = resolve_provider(SINGLE_IMAGE_VQA)
        capabilities.register_default_providers()
        self.assertEqual(resolve_provider(SINGLE_IMAGE_VQA), original)

    def test_provider_cannot_advertise_unknown_capability(self) -> None:
        with self.assertRaises(ValueError):
            Provider(
                name="impostor",
                version="0.0.1",
                capabilities=frozenset({"time_travel"}),
                model_name="mock",
            )

    def test_provider_rejects_empty_identity(self) -> None:
        for kwargs in (
            {"name": "", "version": "0.0.1"},
            {"name": "x", "version": ""},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    Provider(
                        capabilities=frozenset({SINGLE_IMAGE_VQA}),
                        model_name="mock",
                        **kwargs,
                    )

    def test_status_snapshot_is_truthful_and_immutable_to_callers(self) -> None:
        status = capabilities.capabilities_status()
        self.assertEqual(
            status,
            [
                {"name": "single_image_vqa", "registered": True, "available": True, "state": "AVAILABLE", "provider": "qwen2.5vl-3b", "reason_code": None, "detail": None},
                {"name": "grounding", "registered": True, "available": True, "state": "AVAILABLE", "provider": "grounding-dino-swint", "reason_code": None, "detail": None},
                {"name": "change_vqa", "registered": True, "available": True, "state": "AVAILABLE", "provider": "change-deterministic", "reason_code": None, "detail": None},
                {"name": "optical_sar", "registered": True, "available": True, "state": "AVAILABLE", "provider": "optical-sar-deterministic", "reason_code": None, "detail": None},
            ],
        )
        status.clear()
        self.assertEqual(len(capabilities.capabilities_status()), 4)

    def test_registered_unready_provider_cannot_enter_inference(self) -> None:
        with (
            patch.object(QwenVLModel, "readiness", lambda _: ModelReadiness(False, "CUDA_UNAVAILABLE", "A CUDA GPU is required.")),
            patch.object(QwenVLModel, "infer", side_effect=AssertionError("inference must not run")) as infer,
        ):
            status = capabilities.capability_readiness(SINGLE_IMAGE_VQA)
            self.assertTrue(status["registered"])
            self.assertFalse(status["available"])
            self.assertEqual(status["reason_code"], "CUDA_UNAVAILABLE")
            self.assertEqual(resolve_provider(SINGLE_IMAGE_VQA).provider_name, "qwen2.5vl-3b")
            with self.assertRaises(ProviderNotReady):
                route(SINGLE_IMAGE_VQA, ["/missing.png"], "Is there water?", {"execution_mode": "live"})
            infer.assert_not_called()
            self.assertEqual(trace_store.records(), [])


if __name__ == "__main__":
    unittest.main()
