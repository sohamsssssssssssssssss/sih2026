from contextlib import nullcontext
from io import BytesIO
import json
from types import SimpleNamespace

from PIL import Image
import pytest
import torch

from kaggle import run_grounding_baselines as runner


def test_dino_preflight_runs_before_any_qwen_work(monkeypatch, tmp_path, capsys):
    image = BytesIO()
    Image.new("RGB", (2, 2)).save(image, format="PNG")
    rows = [
        dict(id=str(i), parent=str(i), dataset="dior-rsvg", split="eval", task="grounding",
             image="image.png", expression=f"object {i}")
        for i in range(26)
    ]
    args = SimpleNamespace(
        pack=tmp_path / "pack", pack_sha256="pack", manifest_sha256=runner.MANIFEST_SHA,
        output=tmp_path / "output", stub=False, adapter_eval=False, adapter_dir=None,
        adapter_receipt_sha256=None, qwen_dir=tmp_path, qwen_hashes=tmp_path / "qwen-hashes.json",
        reuse_dir=tmp_path, reuse_manifest=tmp_path / "reuse.json", reuse_manifest_sha256=None,
    )
    args.qwen_hashes.write_bytes((runner.REPO / "kaggle/grounding_qwen_hashes.json").read_bytes())
    monkeypatch.setattr(runner, "verify_qwen_directory", lambda *_: None)
    monkeypatch.setattr(runner, "open_pack", lambda *_: nullcontext(SimpleNamespace(read=lambda _: image.getvalue())))
    monkeypatch.setattr(runner, "pack_items", lambda _: rows)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _: "test")
    monkeypatch.setattr(torch, "manual_seed", lambda _: None)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.random, "fork_rng", lambda **_: nullcontext())
    monkeypatch.setattr(runner, "version", lambda _: "test")
    calls = []

    def dino_backend():
        calls.append("load DINO")

        def predict(_, expression):
            calls.append(expression)
            return [
                {"box_threshold": b, "text_threshold": t, "detections": []} for b, t in runner.GRID
            ]
        return predict

    def qwen_backend(*_):
        calls.append("load Qwen")

        def predict(_, prompt, tokens):
            calls.append("Qwen predict")
            return "[100,200,300,400]"
        return predict

    def broken_dino(*_):
        raise AttributeError("tokenizer")

    source = tmp_path / "qwen-eval.jsonl"
    raw = (json.dumps({"reused": True}) + "\n").encode()
    source.write_bytes(raw * len(rows))
    args.reuse_manifest.write_text(json.dumps({
        "commit": runner.REUSE_COMMIT, "files": {source.name: runner.digest(source)},
    }))
    args.reuse_manifest_sha256 = runner.digest(args.reuse_manifest)
    monkeypatch.setattr(runner, "qwen_backend", qwen_backend)
    monkeypatch.setattr(runner, "dino_backend", lambda: broken_dino)
    with pytest.raises(RuntimeError, match="HARD STOP: dino preflight failed: tokenizer"):
        runner.run(args)
    assert calls == []
    assert not args.output.exists()
    capsys.readouterr()

    monkeypatch.setattr(runner, "dino_backend", dino_backend)
    runner.run(args)
    assert calls[:5] == ["load DINO", "object 0", "object 1", "load Qwen", "Qwen predict"]
    progress = capsys.readouterr().out
    assert progress.index("PREFLIGHT dino: 2 eval items passed") < progress.index("PREFLIGHT qwen: 1 eval items passed")
    assert progress.index("PREFLIGHT qwen: 1 eval items passed") < progress.index("qwen-eval 26/26 reused")
    assert "dino-eval 25/26\ndino-eval 26/26" in progress
    assert (args.output / source.name).read_bytes() == source.read_bytes()
    receipt = json.loads((args.output / "run-info.json").read_text())
    assert receipt["reused"] == {
        source.name: {"commit": runner.REUSE_COMMIT, "sha256": runner.digest(source), "lines": len(rows)},
    }


@pytest.mark.parametrize("bad, error", [
    ("manifest", "manifest SHA-256"), ("commit", "producing commit"),
    ("hash", "reuse SHA-256"), ("lines", "line count"),
    ("filename", "unexpected reuse file"), ("missing", "missing or unsafe reuse"),
])
def test_reuse_rejects_bad_pins(tmp_path, bad, error):
    source = tmp_path / "qwen-eval.jsonl"
    source.write_bytes(b'{"id":"a"}\n')
    manifest = tmp_path / "reuse.json"
    manifest.write_text(json.dumps({
        "commit": "wrong" if bad == "commit" else runner.REUSE_COMMIT,
        "files": {"../escape" if bad == "filename" else source.name:
                  "f" * 64 if bad == "hash" else runner.digest(source)},
    }))
    args = SimpleNamespace(reuse_dir=tmp_path, reuse_manifest=manifest,
                           reuse_manifest_sha256="0" * 64 if bad == "manifest" else runner.digest(manifest))
    if bad == "missing":
        source.unlink()
    with pytest.raises(ValueError, match=f"HARD STOP: .*{error}"):
        runner.reusable_files(args, {source.name: 2 if bad == "lines" else 1})


def test_cached_forward_delegates_model_attributes_and_caches_one_call():
    class Model:
        tokenizer = object()
        max_text_len = 256
        specical_tokens = [101, 102, 1012, 1029]

        def eval(self):
            return self

        def __call__(self, image, *, captions):
            return image, captions

    model = Model()
    cache = runner.CachedForward(model)
    assert cache.to("cpu") is cache
    assert cache.tokenizer is model.tokenizer
    assert cache.max_text_len == model.max_text_len
    assert cache.specical_tokens is model.specical_tokens
    assert cache.eval() is model
    first = cache("image", captions=["object."])
    assert first == ("image", ["object."])
    assert cache("image", captions=["object."]) is first
    assert runner.CachedForward(model)("another", captions=["object."]) != first
