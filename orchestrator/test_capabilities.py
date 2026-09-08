import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import unittest

from orchestrator import capabilities
from orchestrator.capabilities import (
    CHANGE_VQA,
    GROUNDING,
    CapabilityUnavailable,
    KNOWN_CAPABILITIES,
    OPTICAL_SAR,
    Provider,
    SINGLE_IMAGE_VQA,
    UnknownCapability,
    resolve_provider,
)


class CapabilityRegistryTests(unittest.TestCase):
    def setUp(self) -> None:
        capabilities.reset_registry()
        capabilities.register_default_providers()

    def tearDown(self) -> None:
        capabilities.reset_registry()
        capabilities.register_default_providers()

    def test_known_vocabulary_matches_spec(self) -> None:
        self.assertEqual(
            KNOWN_CAPABILITIES,
            (SINGLE_IMAGE_VQA, GROUNDING, CHANGE_VQA, OPTICAL_SAR),
        )
        self.assertEqual(capabilities.IMPLEMENTED_CAPABILITIES, {SINGLE_IMAGE_VQA})

    def test_single_image_vqa_resolves_to_qwen_provider(self) -> None:
        resolved = resolve_provider(SINGLE_IMAGE_VQA)
        self.assertEqual(resolved.capability, SINGLE_IMAGE_VQA)
        self.assertEqual(resolved.provider_name, "qwen2.5vl-3b")
        self.assertEqual(resolved.model_name, "qwen2.5vl-3b")

    def test_provider_metadata_is_truthful(self) -> None:
        resolved = resolve_provider(SINGLE_IMAGE_VQA)
        self.assertEqual(resolved.model_version, "Qwen/Qwen2.5-VL-3B-Instruct")

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

    def test_provider_cannot_advertise_unimplemented_capability(self) -> None:
        with self.assertRaises(ValueError):
            Provider(
                name="impostor",
                version="0.0.1",
                capabilities=frozenset({"grounding"}),
                model_name="mock",
            )
        self.assertFalse(
            any(entry["available"] for entry in capabilities.capabilities_status()
                if entry["name"] == "grounding")
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
                {"name": "single_image_vqa", "available": True, "provider": "qwen2.5vl-3b"},
                {"name": "grounding", "available": False, "provider": None},
                {"name": "change_vqa", "available": False, "provider": None},
                {"name": "optical_sar", "available": False, "provider": None},
            ],
        )
        status.clear()
        self.assertEqual(len(capabilities.capabilities_status()), 4)


if __name__ == "__main__":
    unittest.main()
