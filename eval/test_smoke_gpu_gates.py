"""Unit tests for smoke.py's optional --gpu hardware gates.

The gates guard against configs that are invalid for the accelerator actually
present. They must therefore be testable without one, so torch.cuda is patched
here — a T4 is simulated as sm_75, an Ampere card as sm_80.
"""

import unittest
from contextlib import contextmanager
from unittest.mock import patch

from eval.smoke import check_gpu

GIGABYTE = 1e9


@contextmanager
def fake_gpu(name="Tesla T4", capability=(7, 5), total_gb=15.0,
             allocated_gb=2.0, peak_gb=3.0):
    """Patch torch.cuda to look like a specific accelerator."""

    class Properties:
        total_memory = total_gb * GIGABYTE

    with patch("eval.smoke.torch.cuda.is_available", return_value=True), \
         patch("eval.smoke.torch.cuda.get_device_name", return_value=name), \
         patch("eval.smoke.torch.cuda.get_device_capability", return_value=capability), \
         patch("eval.smoke.torch.cuda.get_device_properties", return_value=Properties()), \
         patch("eval.smoke.torch.cuda.memory_allocated",
               return_value=allocated_gb * GIGABYTE), \
         patch("eval.smoke.torch.cuda.max_memory_allocated",
               return_value=peak_gb * GIGABYTE):
        yield


class GpuGateTest(unittest.TestCase):
    def test_bf16_on_turing_is_rejected(self) -> None:
        # The gate that matters: T4 is sm_75 and has no bf16.
        with fake_gpu(capability=(7, 5)):
            with self.assertRaises(AssertionError) as caught:
                check_gpu({"precision": "bf16"})
        self.assertIn("bf16", str(caught.exception))
        self.assertIn("fp16", str(caught.exception))

    def test_bf16_on_ampere_is_allowed(self) -> None:
        with fake_gpu(name="A100", capability=(8, 0)):
            check_gpu({"precision": "bf16"})

    def test_fp16_on_turing_is_allowed(self) -> None:
        with fake_gpu(capability=(7, 5)):
            check_gpu({"precision": "fp16"})

    def test_flash_attention_2_on_turing_is_rejected(self) -> None:
        with fake_gpu(capability=(7, 5)):
            with self.assertRaises(AssertionError) as caught:
                check_gpu({"attn_implementation": "flash_attention_2"})
        self.assertIn("sdpa", str(caught.exception))

    def test_flash_attention_2_on_ampere_is_allowed(self) -> None:
        with fake_gpu(name="A100", capability=(8, 0)):
            check_gpu({"attn_implementation": "flash_attention_2"})

    def test_weights_filling_vram_are_rejected(self) -> None:
        with fake_gpu(total_gb=15.0, allocated_gb=14.5):
            with self.assertRaises(AssertionError) as caught:
                check_gpu({})
        self.assertIn("headroom", str(caught.exception))

    def test_weights_within_budget_are_allowed(self) -> None:
        with fake_gpu(total_gb=15.0, allocated_gb=6.0):
            check_gpu({})

    def test_empty_config_uses_safe_defaults(self) -> None:
        # A config that requests nothing must not trip the capability gates.
        with fake_gpu(capability=(7, 5)):
            check_gpu({})

    def test_missing_cuda_is_reported_clearly(self) -> None:
        with patch("eval.smoke.torch.cuda.is_available", return_value=False):
            with self.assertRaises(AssertionError) as caught:
                check_gpu({"precision": "fp16"})
        self.assertIn("CUDA is unavailable", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
