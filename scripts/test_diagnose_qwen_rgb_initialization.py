"""Harness checks use synthetic weights, never claim pretrained equivalence."""

import json

import pytest

torch = pytest.importorskip("torch")

from scripts import diagnose_qwen_rgb_initialization as diagnostic


def test_real_wrapper_controls_and_candidate():
    generator = torch.Generator().manual_seed(7)
    weight = torch.randn(4, 3, 2, 14, 14, generator=generator, dtype=torch.float64)
    before = weight.clone()
    result = diagnostic.diagnose(weight)
    assert torch.equal(before, weight)
    assert result["weight_shapes"] == {"original": [4, 3, 2, 14, 14], "converted": [4, 12, 2, 14, 14]}
    assert result["projection_output_shape"] == [64, 4]
    assert len(result["cases"]) == 9
    assert result["candidate_all_equivalent"]
    cases = result["cases"]
    assert cases["all_12_grayscale"]["current"]["equivalent"]
    assert cases["rgb_only_grayscale"]["current"]["relative_max_error"] == pytest.approx(.75)
    for name, row in cases.items():
        assert row["reference_wrapper_vs_full_conv3d"]["equivalent"]
        if name != "all_12_grayscale":
            assert not row["current"]["equivalent"]


def test_independent_packing_preserves_temporal_and_spatial_values():
    frames = torch.arange(2 * 3 * 112 * 112).reshape(2, 3, 112, 112)
    packed = diagnostic.pack_frames(frames).reshape(64, 3, 2, 14, 14)
    for index, (y, x) in enumerate((my + dy, mx + dx)
                                 for my in range(0, 8, 2) for mx in range(0, 8, 2)
                                 for dy in range(2) for dx in range(2)):
        assert torch.equal(packed[index], frames[:, :, y * 14:(y + 1) * 14,
                                                x * 14:(x + 1) * 14].permute(1, 0, 2, 3))


def test_missing_weights_report_blocked_without_synthetic_fallback(tmp_path, monkeypatch):
    def forbidden(*args):
        pytest.fail("Must not diagnose without pinned weights")
    monkeypatch.setattr(diagnostic, "diagnose", forbidden)
    output = tmp_path / "report.json"
    assert diagnostic.main(["--checkpoint", str(tmp_path / "missing"), "--output", str(output)]) == 2
    result = json.loads(output.read_text())
    assert result["status"] == "BLOCKED"
    assert "cases" not in result


def test_report_cannot_be_written_into_checkpoint_directory(tmp_path):
    with pytest.raises(SystemExit):
        diagnostic.main(["--checkpoint", str(tmp_path), "--output", str(tmp_path / "report.json")])


def test_zero_reference_has_no_relative_error_and_invalid_shape_fails():
    assert diagnostic.errors(torch.zeros(2), torch.zeros(2))["relative_max_error"] is None
    with pytest.raises(ValueError, match="Expected finite"):
        diagnostic.diagnose(torch.zeros(4, 3, 14, 14))
