"""Synthetic CPU predictions and mocked calls; no pretrained/model loading."""
from copy import deepcopy
from pathlib import Path
import json
from types import SimpleNamespace

import pytest
from PIL import Image
import torch

from eval.rgb_pilot import (CONDITIONS, TYPES, adapter_identity, digest, paired_differences,
                           score, summarise, validate_pilot)
from scripts import evaluate_remote_sensing_adapter as evaluator
from training.remote_sensing import TrainingExample, sha256_file


@pytest.mark.parametrize("kind,pred,gold,correct,malformed", [
    ("presence", " YES ", "yes", True, False),
    ("presence", "1", "yes", False, True),
    ("presence", "yes or no", "yes", False, True),
    ("comp", "no", "yes", False, False),
    ("rural_urban", "URBAN", "urban", True, False),
    ("rural_urban", "suburban", "urban", False, True),
    ("count", " 12 ", "12", True, False),
    ("count", "13", "12", False, False),
    ("count", "012", "12", False, True),
    ("count", "12.0", "12", False, True),
    ("count", "twelve", "12", False, True),
    ("count", "there are 12", "12", False, True),
    ("count", "-1", "0", False, True),
    ("count", "", "0", False, True),
])
def test_synthetic_scoring(kind, pred, gold, correct, malformed):
    assert score(pred, gold, kind) == {"correct": correct, "malformed": malformed}


def test_invalid_gold_and_unknown_type_fail():
    with pytest.raises(ValueError, match="gold"):
        score("1", "one", "count")
    with pytest.raises(ValueError, match="type"):
        score("yes", "yes", "other")


@pytest.fixture
def frozen(tmp_path):
    rows = []
    for image_id in (1, 2):
        Image.new("RGB", (56, 28), (image_id, 20, 30)).save(tmp_path / f"{image_id}.tif")
        for t, answer in zip(TYPES, ("yes", "no", "12", "rural")):
            qid = len(rows)
            rows.append({"sample_id": f"validation-{qid}", "image_id": image_id,
                "question_id": qid, "question_type": t, "question": f"Synthetic {t}?", "answer": answer,
                "instruction": f"Synthetic {t}?", "response": answer, "image": f"{image_id}.tif",
                "dataset": "synthetic-rgb", "source": "synthetic fixture", "split": "validation"})
    manifest = tmp_path / "validation.jsonl"
    manifest.write_text("".join(json.dumps(r) + "\n" for r in rows))
    report = {"manifest_sha256": {"validation.jsonl": sha256_file(manifest)},
              "selected_image_sha256": {f"{i}.tif": sha256_file(tmp_path / f"{i}.tif") for i in (1, 2)}}
    identity = digest(report)
    report["manifest_identity"] = identity
    provenance = tmp_path / "provenance.json"
    provenance.write_text(json.dumps(report))
    return tmp_path, manifest, provenance, identity, rows


def test_integrity_success_and_tampered_identity_manifest_and_image(frozen):
    root, manifest, provenance, identity, rows = frozen
    loaded_rows, examples, p = validate_pilot(provenance, manifest, root, identity)
    assert loaded_rows == rows and len(examples) == 8
    with pytest.raises(ValueError, match="identity"):
        validate_pilot(provenance, manifest, root, "wrong")
    original = manifest.read_bytes()
    manifest.write_bytes(original + b"\n")
    with pytest.raises(ValueError, match="Manifest hash"):
        validate_pilot(provenance, manifest, root, identity)
    manifest.write_bytes(original)
    (root / "1.tif").write_bytes(b"synthetic corruption")
    with pytest.raises(ValueError, match="Image hash"):
        validate_pilot(provenance, manifest, root, identity)


def test_mocked_four_conditions_are_aligned_frozen_and_use_same_settings(frozen, monkeypatch):
    root, manifest, provenance, identity, rows = frozen
    _, examples, _ = validate_pilot(provenance, manifest, root, identity)
    calls = []
    class Model(torch.nn.Module):
        def __init__(self, name):
            super().__init__(); self.name = name; self.parameter = torch.nn.Parameter(torch.ones(1))
    def answer(model, processor, vision, example, image_size, *, blind=False):
        assert not model.training and not model.parameter.requires_grad
        calls.append((model.name, example.sample_id, image_size, blind, processor, vision))
        return example.response if model.name == "adapter" and not blind else "malformed synthetic"
    monkeypatch.setattr(evaluator, "_answer", answer)
    report = evaluator.evaluate_pilot_conditions(rows, examples, Model("base"), "processor", "vision",
                                                lambda _: Model("adapter"), repetitions=100)
    assert tuple(report["conditions"]) == CONDITIONS and len(calls) == 32
    assert {c[2] for c in calls} == {392}
    for condition in CONDITIONS:
        records = report["conditions"][condition]["records"]
        assert [r["question_id"] for r in records] == [r["question_id"] for r in rows]
        assert [r["image_id"] for r in records] == [r["image_id"] for r in rows]
        assert [r["question_type"] for r in records] == [r["question_type"] for r in rows]
    assert report["conditions"]["adapter_real"]["summary"]["macro_accuracy"] == 1
    assert report["conditions"]["adapter_constant"]["summary"]["malformed_n"] == 8


def test_actual_answer_path_constant_image_preserves_tokens_dimensions_and_decode(frozen):
    root, _, _, _, rows = frozen
    r = rows[0]
    example = TrainingExample(root / r['image'], r['question'], r['answer'], 'synthetic', 'synthetic', 'validation', r['sample_id'])
    prompts, images, decodes, generated = [], [], [], []
    class Inputs(dict):
        @property
        def input_ids(self): return self['input_ids']
        def to(self, device): return self
    class Processor:
        def apply_chat_template(self, conversation, **kwargs):
            assert conversation[0]['content'][0]['type'] == 'image'
            prompts.append((conversation[0]['content'][1]['text'], kwargs, conversation[0]['content'][0]['resized_width']))
            return '<image> identical synthetic prompt'
        def __call__(self, **kwargs):
            images.append(kwargs['images'][0].copy())
            return Inputs(input_ids=torch.tensor([[7, 8, 9]]), pixel_values=torch.zeros(1))
        def batch_decode(self, tokens, **kwargs):
            decodes.append((tokens.tolist(), kwargs)); return ['yes']
    def vision(conversation):
        value = conversation[0]['content'][0]['image']
        if isinstance(value, str):
            from urllib.parse import urlparse, unquote
            with Image.open(unquote(urlparse(value).path)) as im: value = im.convert('RGB').copy()
        return [value], None
    def generate(**kwargs):
        generated.append((kwargs['do_sample'], kwargs['max_new_tokens'], kwargs['input_ids'].tolist()))
        assert 'pixel_values' in kwargs
        return torch.tensor([[7, 8, 9, 10]])
    model = SimpleNamespace(device='cpu', generate=generate)
    processor = Processor()
    for blind in (False, True, True):
        assert evaluator._answer(model, processor, vision, example, 392, blind=blind) == 'yes'
    assert prompts[0] == prompts[1] == prompts[2]
    assert all(im.size == (56, 28) and im.mode == 'RGB' for im in images)
    assert images[1].tobytes() == images[2].tobytes()
    assert set(images[1].getdata()) == {(127, 127, 127)}
    assert generated[0] == generated[1] == generated[2] == (False, 32, [[7, 8, 9]])
    assert decodes[0] == decodes[1] == decodes[2]


def test_clustered_paired_bootstrap_reproducible_and_alignment_guard(frozen):
    rows = frozen[-1]
    conditions = {}
    for name in CONDITIONS:
        records = [{**r, 'expected': r['answer'], 'correct': name == 'adapter_real' and r['image_id'] == 1,
                    'malformed': False} for r in rows]
        conditions[name] = {'records': records, 'summary': summarise(records)}
    a = paired_differences(conditions, seed=17, repetitions=200)
    assert a == paired_differences(deepcopy(conditions), seed=17, repetitions=200)
    macro = a['comparisons']['adapter_real_minus_base_real']['macro']
    assert macro['difference'] == .5 and macro['ci95'] == [0., 1.]
    assert a['image_clusters'] == 2
    conditions['base_constant']['records'].reverse()
    with pytest.raises(ValueError, match='alignment'):
        paired_differences(conditions)


def test_adapter_identity_hashes_synthetic_files_without_loading(tmp_path):
    (tmp_path/'adapter_config.json').write_text('{}')
    p = tmp_path/'adapter_model.safetensors'
    p.write_bytes(b'synthetic bytes, not actual model tensors')
    before = adapter_identity(tmp_path)
    p.write_bytes(b'different synthetic bytes')
    assert adapter_identity(tmp_path) != before


def test_pilot_integrity_failure_precedes_model_loading(frozen, monkeypatch):
    root, manifest, provenance, identity, _ = frozen
    # The production identity pin deliberately rejects this synthetic manifest.
    args = SimpleNamespace(pilot_provenance=provenance, dataset_manifest=manifest, image_root=root)
    with pytest.raises(ValueError, match='identity'):
        evaluator.run_pilot(args)


def test_missing_category_and_single_image_have_no_unsupported_intervals():
    r = {'sample_id': 'synthetic', 'question_id': 1, 'image_id': 1,
         'question_type': 'presence', 'expected': 'yes', 'correct': True, 'malformed': False}
    conditions = {name: {'records': [dict(r)]} for name in CONDITIONS}
    report = paired_differences(conditions, repetitions=10)
    assert summarise([r])['macro_accuracy'] is None
    comparison = report['comparisons']['adapter_real_minus_base_real']
    assert comparison['macro']['difference'] is None and comparison['macro']['ci95'] is None
    assert comparison['per_type']['presence']['ci95'] is None


def test_adapter_decode_configuration_drift_rejected(frozen, monkeypatch):
    root, manifest, provenance, identity, rows = frozen
    _, examples, _ = validate_pilot(provenance, manifest, root, identity)
    class Model(torch.nn.Module):
        def __init__(self, token):
            super().__init__()
            self.generation_config = SimpleNamespace(to_dict=lambda: {'eos_token_id': token})
    monkeypatch.setattr(evaluator, '_answer', lambda *a, **kw: 'yes')
    with pytest.raises(ValueError, match='decoding configuration'):
        evaluator.evaluate_pilot_conditions(rows, examples, Model(1), None, None, lambda _: Model(2))
