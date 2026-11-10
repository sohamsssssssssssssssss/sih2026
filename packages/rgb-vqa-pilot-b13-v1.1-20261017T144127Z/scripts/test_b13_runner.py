"""Synthetic CPU checks for the B13 package; no model weights or GPU work."""

import json
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from scripts import rgb_pilot_package as package
from training.remote_sensing import sha256_file


def test_frozen_manifest_and_schedule_guards():
    root = package.ROOT
    plan = json.loads((root / 'pilot-config.json').read_text())
    provenance = json.loads((root / 'data/manifests/rsvqa/b13-6k.v1/provenance.json').read_text())
    manifest = root / 'data/manifests/rsvqa/b13-6k.v1/train.jsonl'
    assert sha256_file(manifest) == package.B13_SHA256
    package.check_b13_spec(plan, provenance, sha256_file(manifest))
    assert (plan['training']['max_samples'], plan['training']['max_steps']) == (6000, 750)
    assert package.config(root, root / 'synthetic-weights', root / 'synthetic-output').dataset_manifest == manifest
    with pytest.raises(ValueError, match='manifest'):
        package.check_b13_spec(plan, provenance, '0' * 64)
    changed = json.loads(json.dumps(plan))
    changed['training']['max_steps'] = 128
    with pytest.raises(ValueError, match='schedule'):
        package.check_b13_spec(changed, provenance, sha256_file(manifest))


def test_example_count_and_final_checkpoint_guard():
    examples = [SimpleNamespace(sample_id=f'synthetic-{i}') for i in range(6000)]
    assert package.check_b13_examples(examples) is examples
    with pytest.raises(ValueError, match='6000'):
        package.check_b13_examples(examples[:-1])
    examples[-1] = examples[0]
    with pytest.raises(ValueError, match='unique'):
        package.check_b13_examples(examples)
    assert package.force_final_checkpoint(750)
    assert not package.force_final_checkpoint(749)


def test_resume_identity_state_and_hashes(tmp_path):
    names = tuple(package.REQUIRED_CHECKPOINT_FILES)
    for name in names:
        (tmp_path / name).write_bytes(f'synthetic {name}'.encode())
    state = tmp_path / 'trainer_state.json'
    state.write_text(json.dumps({'max_steps': 750, 'global_step': 749}))

    def record():
        evidence = {'run_identity': 'synthetic-b13', 'step': json.loads(state.read_text())['global_step'],
                    'scaler_expected': False,
                    'files_sha256': {name: sha256_file(tmp_path / name) for name in names}}
        (tmp_path / 'pilot-checkpoint.json').write_text(json.dumps(evidence))
        return evidence

    evidence = record()
    assert package.verify_resume(tmp_path, 'synthetic-b13') == evidence
    with pytest.raises(ValueError, match='identity'):
        package.verify_resume(tmp_path, 'b12-identity')
    (tmp_path / 'optimizer.pt').write_bytes(b'tampered synthetic state')
    with pytest.raises(ValueError, match='hash'):
        package.verify_resume(tmp_path, 'synthetic-b13')
    record()
    (tmp_path / 'scheduler.pt').unlink()
    with pytest.raises(ValueError, match='inventory'):
        package.verify_resume(tmp_path, 'synthetic-b13')
    (tmp_path / 'scheduler.pt').write_bytes(b'synthetic scheduler.pt')
    state.write_text(json.dumps({'max_steps': 750, 'global_step': 750}))
    record()
    with pytest.raises(ValueError, match='750-step'):
        package.verify_resume(tmp_path, 'synthetic-b13')


def test_processor_preflight_checks_all_rows_and_reports_all_failures(monkeypatch, tmp_path):
    examples = [SimpleNamespace(sample_id=f'synthetic-{i}') for i in range(6000)]
    monkeypatch.setattr(package, 'check_payload', lambda root: {})
    monkeypatch.setattr(package, 'verify_base', lambda root, weights: {'measured_revision': 'synthetic'})
    monkeypatch.setattr(package, 'config', lambda *args: SimpleNamespace(
        dataset_manifest=tmp_path / 'synthetic-manifest', image_root=tmp_path,
        max_samples=6000, seed=17, image_size=392, max_sequence_length=32768))
    monkeypatch.setattr(package, 'load_manifest', lambda *args: examples)
    monkeypatch.setattr(package, 'select_examples', lambda rows, split, count, seed: rows[:count])
    class Processor:
        tokenizer = SimpleNamespace(pad_token_id=0)
        @classmethod
        def from_pretrained(cls, weights, *, local_files_only):
            assert local_files_only
            return cls()
    transformers = ModuleType('transformers')
    transformers.AutoProcessor = Processor
    monkeypatch.setitem(sys.modules, 'transformers', transformers)
    class Collator:
        def __init__(self, processor, image_size, *, max_sequence_length):
            assert image_size == 392 and max_sequence_length == 32768
        def __call__(self, batch):
            if batch[0].sample_id == 'synthetic-3':
                raise ValueError('synthetic unsupported image')
            labels = [-100, -100, 3, -100]
            if batch[0].sample_id == 'synthetic-5':
                labels[0] = 1
            return {'input_ids': torch.tensor([[1, 2, 3, 0]]),
                    'labels': torch.tensor([labels]),
                    'pixel_values': torch.ones(4, 1176),
                    'image_grid_thw': torch.ones(1, 3)}
        def _encode(self, batch, *, include_target):
            assert not include_target
            return {'input_ids': torch.tensor([[1, 2, 0]])}
    monkeypatch.setattr(package, 'QwenVQACollator', Collator)
    report = package.tokenization_preflight(tmp_path, tmp_path / 'synthetic-weights')
    assert report['examples_checked'] == 6000
    assert report['answer_masks_verified'] == 5998
    assert [item['sample_id'] for item in report['failures']] == ['synthetic-3', 'synthetic-5']
    assert report['pixel_value_shapes'] == [[4, 1176]]
    assert report['max_encoded_sequence_length'] == 3
    assert report['all_fit'] is False


def test_preflight_records_and_rechecks_trainer_loader_order(monkeypatch, tmp_path):
    examples = [SimpleNamespace(sample_id=f'synthetic-{i}') for i in range(6000)]
    monkeypatch.setattr(package, 'check_payload', lambda root: {})
    monkeypatch.setattr(package, 'config', lambda *args: SimpleNamespace(
        dataset_manifest=tmp_path / 'synthetic-manifest', image_root=tmp_path,
        max_samples=6000, seed=17, batch_size=1,
        gradient_accumulation_steps=8, max_steps=750))
    monkeypatch.setattr(package, 'load_manifest', lambda *args: examples)
    monkeypatch.setattr(package, 'select_examples', lambda rows, split, count, seed: rows[:count])
    monkeypatch.setattr(package, 'synthetic_checks', lambda: {})
    class Trainer:
        def __init__(self, *, train_dataset, **kwargs):
            self.rows = train_dataset
        def get_train_dataloader(self):
            return [{'sample_ids': [item.sample_id]} for item in reversed(self.rows)]
    transformers = ModuleType('transformers')
    transformers.Trainer = Trainer
    transformers.TrainingArguments = lambda **kwargs: SimpleNamespace(**kwargs)
    monkeypatch.setitem(sys.modules, 'transformers', transformers)
    report = package.preflight(package.ROOT)
    assert report['training_examples'] == report['cpu_dataloader_unique_examples'] == 6000
    assert report['trainer_sample_order_sha256'] == package.digest(
        [item.sample_id for item in reversed(examples)])
