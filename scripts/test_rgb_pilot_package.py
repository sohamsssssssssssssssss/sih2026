"""Synthetic CPU package contracts; no model loading, GPU or optimizer steps."""
import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts import rgb_pilot_package as package
from scripts import train_remote_sensing_adapter as trainer
from training.remote_sensing import TrainingConfig, sha256_file


def test_synthetic_answer_masking_and_intended_parameters():
    assert package.synthetic_checks()['answer_masking']=='PASS'


@pytest.mark.parametrize('name,shape',[
    ('model.visual.layers.0.q_proj.lora_A.default.weight',(16,32)),
    ('model.layers.0.q_proj.weight',(32,32)),
    ('model.layers.0.q_proj.lora_A.default.weight',(8,32)),
])
def test_trainable_policy_rejects_visual_base_or_wrong_rank(name,shape):
    model=SimpleNamespace(named_parameters=lambda:[(name,torch.nn.Parameter(torch.zeros(shape)))])
    with pytest.raises(ValueError,match='Unexpected'):
        package.trainable_guard(model)


def test_resume_checks_identity_and_all_preserved_state_hashes(tmp_path):
    for name in ('optimizer.pt','scheduler.pt','trainer_state.json','rng_state.pth','scaler.pt','adapter_config.json','adapter_model.safetensors'):
        (tmp_path/name).write_bytes(b'synthetic state only')
    (tmp_path/'trainer_state.json').write_text(json.dumps({'max_steps':128,'global_step':16}))
    evidence={'run_identity':'synthetic','files_sha256':{p.name:sha256_file(p) for p in tmp_path.iterdir()}}
    (tmp_path/'pilot-checkpoint.json').write_text(json.dumps(evidence))
    assert package.verify_resume(tmp_path,'synthetic')==evidence
    with pytest.raises(ValueError,match='identity'):
        package.verify_resume(tmp_path,'wrong')
    (tmp_path/'optimizer.pt').write_bytes(b'changed synthetic state')
    with pytest.raises(ValueError,match='hash'):
        package.verify_resume(tmp_path,'synthetic')


def test_training_factory_receives_all_1024_examples_without_training(tmp_path,monkeypatch):
    import sys
    from types import ModuleType

    examples=[SimpleNamespace(sample_id=f'synthetic-{i}',dataset='synthetic',source='synthetic') for i in range(1024)]
    monkeypatch.setattr(trainer,'load_manifest',lambda *a:examples)
    monkeypatch.setattr(trainer,'select_examples',lambda rows,split,n,seed: rows[:n])
    seen={}
    class FakeTrainer:
        def __init__(self,**kwargs):
            seen['examples']=len(kwargs['train_dataset'])
            seen['args']=kwargs['args']
        def train(self): return SimpleNamespace(metrics={'synthetic':True})
    transformers=ModuleType('transformers')
    transformers.Trainer=FakeTrainer
    transformers.TrainingArguments=lambda **kwargs:SimpleNamespace(**kwargs)
    monkeypatch.setitem(sys.modules,'transformers',transformers)
    class Model:
        def save_pretrained(self,p):p.mkdir(parents=True)
    class Processor:
        def save_pretrained(self,p):pass
    monkeypatch.setattr(trainer,'build_report',lambda *a:{'status':'synthetic'})
    cfg=TrainingConfig(model_path=tmp_path,dataset_manifest=tmp_path/'synthetic',image_root=tmp_path,
        output_dir=tmp_path/'out',max_samples=1024,max_steps=128,batch_size=1,gradient_accumulation_steps=8)
    trainer.run(cfg,component_loader=lambda c:(Model(),Processor()),collator_factory=lambda *a,**k:None,trainer_factory=FakeTrainer)
    assert seen['examples']==1024
    assert (seen['args'].max_steps,seen['args'].per_device_train_batch_size,seen['args'].gradient_accumulation_steps)==(128,1,8)
    assert {key:getattr(seen['args'],key) for key in (
        'optim','lr_scheduler_type','warmup_steps','warmup_ratio','weight_decay','adam_beta1',
        'adam_beta2','adam_epsilon','max_grad_norm','fp16','bf16','half_precision_backend',
        'fp16_full_eval','bf16_full_eval','tf32',
    )} == {
        'optim':'adamw_torch','lr_scheduler_type':'linear','warmup_steps':0,'warmup_ratio':0.0,
        'weight_decay':0.0,'adam_beta1':0.9,'adam_beta2':0.999,'adam_epsilon':1e-8,
        'max_grad_norm':1.0,'fp16':True,'bf16':False,'half_precision_backend':'auto',
        'fp16_full_eval':False,'bf16_full_eval':False,'tf32':False,
    }


def test_supervisor_timeout_preserves_log(tmp_path,monkeypatch):
    def timeout(*a,**kw):raise package.subprocess.TimeoutExpired('synthetic child',1)
    monkeypatch.setattr(package.subprocess,'run',timeout)
    result=package.supervise(SimpleNamespace(output=tmp_path,max_seconds=1))
    assert result['exit_code']==124
    assert (tmp_path/'worker.log').exists()


def test_incomplete_resume_requires_scaler_and_original_schedule(tmp_path):
    names=('optimizer.pt','scheduler.pt','trainer_state.json','rng_state.pth','scaler.pt','adapter_config.json','adapter_model.safetensors')
    for name in names:(tmp_path/name).write_bytes(b'synthetic')
    (tmp_path/'trainer_state.json').write_text(json.dumps({'max_steps':129,'global_step':16}))
    hashes={p.name:sha256_file(p) for p in tmp_path.iterdir()}
    (tmp_path/'pilot-checkpoint.json').write_text(json.dumps({'run_identity':'synthetic','files_sha256':hashes}))
    with pytest.raises(ValueError,match='128-step'):
        package.verify_resume(tmp_path,'synthetic')
    (tmp_path/'scaler.pt').unlink()
    del hashes['scaler.pt']
    (tmp_path/'pilot-checkpoint.json').write_text(json.dumps({'run_identity':'synthetic','files_sha256':hashes}))
    with pytest.raises(ValueError,match='scaler'):
        package.verify_resume(tmp_path,'synthetic')


def _training_checkpoint(path, *, step=16, identity='synthetic', scaler=True, record_scaler=False):
    names=tuple(package.REQUIRED_CHECKPOINT_FILES)+(('scaler.pt',) if scaler else ())
    for name in names:
        (path/name).write_bytes(f'synthetic {name}'.encode())
    (path/'trainer_state.json').write_text(json.dumps({'max_steps':128,'global_step':step}))
    hashes={name:sha256_file(path/name) for name in names}
    evidence={'run_identity':identity,'step':step,'files_sha256':hashes}
    if record_scaler:
        evidence['scaler_expected']=scaler
    (path/'pilot-checkpoint.json').write_text(json.dumps(evidence))
    return evidence


@pytest.mark.parametrize('scaler',[False,True])
def test_resume_accepts_runtime_recorded_scaler_state(tmp_path,scaler):
    evidence=_training_checkpoint(tmp_path,scaler=scaler,record_scaler=True)
    assert package.verify_resume(tmp_path,'synthetic')==evidence


def test_resume_rejects_unexpectedly_missing_or_unrecorded_scaler(tmp_path):
    evidence=_training_checkpoint(tmp_path,record_scaler=True)
    (tmp_path/'scaler.pt').unlink()
    with pytest.raises(ValueError,match='inventory'):
        package.verify_resume(tmp_path,'synthetic')
    evidence['files_sha256'].pop('scaler.pt')
    (tmp_path/'pilot-checkpoint.json').write_text(json.dumps(evidence))
    with pytest.raises(ValueError,match='scaler'):
        package.verify_resume(tmp_path,'synthetic')
    evidence.pop('scaler_expected')
    (tmp_path/'pilot-checkpoint.json').write_text(json.dumps(evidence))
    with pytest.raises(ValueError,match='scaler'):
        package.verify_resume(tmp_path,'synthetic')


def test_resume_rejects_scaler_file_when_runtime_record_says_absent(tmp_path):
    evidence=_training_checkpoint(tmp_path,scaler=True,record_scaler=True)
    evidence['scaler_expected']=False
    (tmp_path/'pilot-checkpoint.json').write_text(json.dumps(evidence))
    with pytest.raises(ValueError,match='scaler'):
        package.verify_resume(tmp_path,'synthetic')


def test_resume_rejects_corruption_and_frozen_identity_mismatch_without_scaler(tmp_path):
    _training_checkpoint(tmp_path,scaler=False,record_scaler=True)
    with pytest.raises(ValueError,match='identity'):
        package.verify_resume(tmp_path,'different')
    (tmp_path/'adapter_model.safetensors').write_bytes(b'corrupted synthetic adapter')
    with pytest.raises(ValueError,match='hash'):
        package.verify_resume(tmp_path,'synthetic')


@pytest.mark.parametrize('name',('optimizer.pt','scheduler.pt','rng_state.pth','adapter_model.safetensors'))
def test_resume_rejects_each_missing_required_training_state(tmp_path,name):
    evidence=_training_checkpoint(tmp_path)
    evidence['files_sha256'].pop(name)
    (tmp_path/name).unlink()
    (tmp_path/'pilot-checkpoint.json').write_text(json.dumps(evidence))
    with pytest.raises(ValueError,match=f'Resume state missing: {name}'):
        package.verify_resume(tmp_path,'synthetic')


@pytest.mark.parametrize('step',(128,129))
def test_resume_rejects_completed_or_overrun_training_checkpoint(tmp_path,step):
    _training_checkpoint(tmp_path,step=step)
    with pytest.raises(ValueError,match='incomplete checkpoint'):
        package.verify_resume(tmp_path,'synthetic')


def _install_fake_trainer(monkeypatch, seen, *, scaler=True):
    import sys
    from types import ModuleType

    class FakeTrainer:
        def __init__(self, **kwargs):
            self.args=kwargs['args']
            self.accelerator=SimpleNamespace(scaler=object() if scaler else None)

        def train(self, *args, **kwargs):
            seen['resume_from_checkpoint']=kwargs.get('resume_from_checkpoint')
            return SimpleNamespace(metrics={'synthetic':True})

    transformers=ModuleType('transformers')
    transformers.Trainer=FakeTrainer
    transformers.TrainerCallback=type('TrainerCallback',(),{})
    monkeypatch.setitem(sys.modules,'transformers',transformers)


@pytest.mark.parametrize('scaler',[False,True])
def test_training_resume_passes_only_validated_checkpoint_to_trainer(tmp_path,monkeypatch,scaler):
    checkpoint=tmp_path/'checkpoint-16'
    checkpoint.mkdir()
    _training_checkpoint(checkpoint,scaler=scaler,record_scaler=True)
    output=tmp_path/'output'
    output.mkdir()
    args=SimpleNamespace(resume_from=checkpoint,max_seconds=3600,output=output,weights=tmp_path/'weights')
    seen={}
    _install_fake_trainer(monkeypatch,seen,scaler=scaler)
    monkeypatch.setattr(package,'binding',lambda root:'synthetic')
    monkeypatch.setattr(package,'config',lambda root,weights,out:TrainingConfig(
        model_path=weights,dataset_manifest=tmp_path/'synthetic-manifest',image_root=tmp_path,
        output_dir=out,max_samples=1024,max_steps=128))
    (tmp_path/'pilot-config.json').write_text(json.dumps({'base_revision':'synthetic-revision'}))

    def mocked_run(config, *, component_loader, trainer_factory):
        trainer=trainer_factory(model=object(),args=SimpleNamespace(output_dir=str(output/'checkpoints')),
            train_dataset=[None]*1024,data_collator=None)
        trainer.train()
        saved=output/'checkpoints/checkpoint-16'
        saved.mkdir(parents=True)
        (saved/'trainer_state.json').write_text(json.dumps({'max_steps':128,'global_step':16}))
        return {'status':'training_complete'}

    monkeypatch.setattr(trainer,'run',mocked_run)
    package.train(tmp_path,args,time.monotonic())
    assert seen['resume_from_checkpoint']==str(checkpoint)


def test_training_resume_rejects_runtime_scaler_mismatch(tmp_path,monkeypatch):
    checkpoint=tmp_path/'checkpoint-16'
    checkpoint.mkdir()
    _training_checkpoint(checkpoint,scaler=False,record_scaler=True)
    output=tmp_path/'output'
    output.mkdir()
    args=SimpleNamespace(resume_from=checkpoint,max_seconds=3600,output=output,weights=tmp_path/'weights')
    seen={}
    _install_fake_trainer(monkeypatch,seen,scaler=True)
    monkeypatch.setattr(package,'binding',lambda root:'synthetic')
    monkeypatch.setattr(package,'config',lambda root,weights,out:TrainingConfig(
        model_path=weights,dataset_manifest=tmp_path/'synthetic-manifest',image_root=tmp_path,
        output_dir=out,max_samples=1024,max_steps=128))
    def mocked_run(config, *, component_loader, trainer_factory):
        instance=trainer_factory(model=object(),args=SimpleNamespace(output_dir=str(output/'checkpoints')),
            train_dataset=[None]*1024,data_collator=None)
        instance.train()
    monkeypatch.setattr(trainer,'run',mocked_run)
    with pytest.raises(ValueError,match='runtime scaler'):
        package.train(tmp_path,args,time.monotonic())
    assert 'resume_from_checkpoint' not in seen


def test_failed_training_resume_preserves_checkpoint_files(tmp_path,monkeypatch):
    checkpoint=tmp_path/'checkpoint-16'
    checkpoint.mkdir()
    _training_checkpoint(checkpoint)
    (checkpoint/'optimizer.pt').write_bytes(b'tampered synthetic optimizer')
    before={p.name:p.read_bytes() for p in checkpoint.iterdir()}
    output=tmp_path/'output'
    output.mkdir()
    args=SimpleNamespace(resume_from=checkpoint,max_seconds=3600,output=output,weights=tmp_path/'weights')
    _install_fake_trainer(monkeypatch,{})
    monkeypatch.setattr(package,'binding',lambda root:'synthetic')
    monkeypatch.setattr(trainer,'run',lambda *a,**k:pytest.fail('runner must not start for an invalid checkpoint'))

    with pytest.raises(ValueError,match='hash'):
        package.train(package.ROOT,args,time.monotonic())
    assert {p.name:p.read_bytes() for p in checkpoint.iterdir()}==before


def test_resume_identity_includes_effective_training_configuration(tmp_path,monkeypatch):
    monkeypatch.setattr(package,'check_payload',lambda root:{'source_fingerprint':'synthetic-source'})
    monkeypatch.setattr(package,'sha256_file',lambda path:'synthetic-pilot-config-hash')
    cfg=TrainingConfig(model_path=tmp_path/'weights',dataset_manifest=tmp_path/'manifest',image_root=tmp_path,
        output_dir=tmp_path/'out',max_steps=128,max_samples=1024)
    monkeypatch.setattr(package,'config',lambda *args:cfg)
    first=package.binding(tmp_path)
    checkpoint=tmp_path/'checkpoint-16'
    checkpoint.mkdir()
    _training_checkpoint(checkpoint,identity=first)

    from dataclasses import replace
    monkeypatch.setattr(package,'config',lambda *args:replace(
        cfg,model_path=tmp_path/'other-weights',dataset_manifest=tmp_path/'other-manifest',
        image_root=tmp_path/'other-images',output_dir=tmp_path/'other-output'))
    assert package.binding(tmp_path/'moved')==first
    monkeypatch.setattr(package,'config',lambda *args:replace(cfg,lr_scheduler_type='cosine'))
    second=package.binding(tmp_path)
    assert first!=second
    with pytest.raises(ValueError,match='identity'):
        package.verify_resume(checkpoint,second)
