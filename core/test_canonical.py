import hashlib

from core.canonical import canonical_json
from core.schema import scene_id_for
from orchestrator import trace

RECORD = {
    "model_name": "qwen2.5vl-3b",
    "params": {"capability": "single_image_vqa", "n": 1},
    "input_summary": {"image_paths": ["a.png"], "question": "q", "n_images": 1},
}
RECORD_TEXT = (
    '{"input_summary": {"image_paths": ["a.png"], "n_images": 1, "question": "q"}, '
    '"model_name": "qwen2.5vl-3b", "params": {"capability": "single_image_vqa", "n": 1}}'
)


def test_canonical_json_matches_the_frozen_pre_t07_digest() -> None:
    text = canonical_json(RECORD)
    assert text == RECORD_TEXT
    assert hashlib.sha256(text.encode()).hexdigest() == (
        "d39adfafa52a84d95c0230f6a292655aa2387a1cadf41837b7a5e255b6ca5327"
    )


def test_known_scene_id_digests_are_unchanged() -> None:
    assert scene_id_for(
        b"pixels", {"gsd": 0.3, "bands": ["B02", "B03"], "nested": {"b": 1, "a": 2}}
    ) == "21414dbdbb6d3bcdf8a7a20107c299594e707336bb6ef03a7b31fb526a1b2989"
    assert scene_id_for(
        b"\x89PNG",
        {"sensor": "S2", "unicode": "caf\u00e9", "bounds": [1.0, 2.0, 3.0, 4.0]},
    ) == "d7efa01ebba2ae2ed236a84283a48a1b0be03399f912fdf762cf6188d2fe552c"


def test_trace_uses_the_shared_canonical_form() -> None:
    assert trace._canonical(RECORD) == canonical_json(RECORD)
