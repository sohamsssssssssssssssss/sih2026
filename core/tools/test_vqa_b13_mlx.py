import os
from pathlib import Path
from unittest.mock import Mock

import pytest

from core.schema import Query
from core.tools.vqa_b13_mlx import B13MLXTool, UNAVAILABLE, VQAWorker


def test_tool_returns_worker_answer_and_surfaces_failure(tmp_path):
    image = tmp_path / "scene.png"
    worker = Mock(ready=True)
    worker.ask.return_value = "  yes"
    tool = B13MLXTool(worker)
    query = Query(text="Is there a road?", qtype="open")
    result = tool.run(None, query, image)
    assert result.value["answer"] == "  yes"
    worker.ask.assert_called_once_with(str(image.resolve()), query.text)
    worker.ask.side_effect = RuntimeError(UNAVAILABLE)
    assert tool.run(None, query, image).value["error"] == UNAVAILABLE


def test_missing_worker_closes_its_socket(monkeypatch):
    worker = VQAWorker()
    monkeypatch.setattr("core.tools.vqa_b13_mlx.subprocess.Popen", Mock(side_effect=FileNotFoundError))
    with pytest.raises(FileNotFoundError):
        worker.start()
    assert not worker.ready
    assert worker.channel is None


def test_real_mlx_worker():
    if os.environ.get("SATQUERY_TEST_VQA_MLX") != "1":
        pytest.skip("Set SATQUERY_TEST_VQA_MLX=1 to exercise the real MLX worker")
    models = Path.home() / "models"
    if not all((models / name).is_file() for name in (
        ".venv-satquery-mac-vqa/bin/python",
        "qwen2.5-vl-3b-mlx-8bit/MANIFEST",
        "qwen2.5-vl-3b-mlx-8bit/model.safetensors",
        "satquery-b13-adapter/adapter_model.safetensors",
    )):
        pytest.skip("The pinned MLX installation is unavailable")
    worker = VQAWorker()
    try:
        worker.start()
        assert worker.ready
        image = Path("data/demo/resolution/loveda_LoveDA_images_png_0_gsd10.0.png").resolve()
        assert worker.ask(str(image), "Is there a road?").strip().casefold() in {"yes", "no"}
    finally:
        worker.close()
    assert not worker.ready
