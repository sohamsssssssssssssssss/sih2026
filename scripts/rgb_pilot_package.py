"""Offline B6 pilot package entry point. GPU work requires separate authorization."""
import argparse
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.rgb_pilot import PILOT_IDENTITY, digest, validate_pilot, versions
from training.remote_sensing import (TrainingConfig, QwenVQACollator, load_manifest,
    load_training_components, select_examples, sha256_file)

REQUIRED_CHECKPOINT_FILES=frozenset(('optimizer.pt','scheduler.pt','trainer_state.json',
    'rng_state.pth','adapter_config.json','adapter_model.safetensors'))


def paths(root):
    pilot = root / "data/manifests/rsvqa/pilot.v1"
    return pilot / "provenance.json", pilot / "validation.jsonl", root / "images"


def check_payload(root):
    manifest = json.loads((root / "package-manifest.json").read_text())
    for name, expected in manifest["payload_sha256"].items():
        if sha256_file(root / name) != expected:
            raise ValueError(f"Package payload mismatch: {name}")
    validate_pilot(*paths(root))
    return manifest


def config(root, weights, output):
    plan = json.loads((root / "pilot-config.json").read_text())
    return TrainingConfig(model_path=weights, dataset_manifest=root / "data/manifests/rsvqa/pilot.v1/pilot.jsonl",
        image_root=root / "images", output_dir=output, **plan["training"])


def trainable_guard(model):
    """Only rank-16 language attention LoRA A/B matrices may train."""
    expected = {"q_proj", "k_proj", "v_proj", "o_proj"}
    found, names = set(), []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        parts = name.split('.')
        target = next((part for part in parts if part in expected), None)
        if ('visual' in parts or 'layers' not in parts or target is None
                or not any(part in ('lora_A', 'lora_B') for part in parts)
                or parameter.ndim != 2
                or (parameter.shape[0] if 'lora_A' in parts else parameter.shape[1]) != 16):
            raise ValueError(f"Unexpected trainable parameter: {name}")
        found.add(target); names.append(name)
    if found != expected:
        raise ValueError("Missing intended language attention LoRA parameters")
    return names


def synthetic_checks():
    """Exercise production collator and parameter guard with synthetic CPU doubles."""
    import torch
    class Processor:
        tokenizer = type('Tokenizer', (), {'pad_token_id': 0, 'padding_side': 'right'})()
        def apply_chat_template(self, conversation, **kwargs):
            return 'full' if conversation[-1]['role'] == 'assistant' else 'prompt'
        def __call__(self, text, **kwargs):
            return {'input_ids': torch.tensor([[4, 5, 6, 7, 0]] if text[0] == 'full' else [[4, 5, 0]])}
    from training.remote_sensing import TrainingExample
    example = TrainingExample(Path('/synthetic/image.tif'), 'Synthetic question?', 'yes', 'synthetic', 'synthetic', 'train', 'synthetic-1')
    labels = QwenVQACollator(Processor(), 392, lambda _: ([object()], None))([example])['labels']
    if labels.tolist() != [[-100, -100, 6, 7, -100]]:
        raise ValueError('Answer masking self-check failed')
    class Parameters:
        def named_parameters(self):
            return [(f'model.layers.0.self_attn.{target}.lora_{kind}.default.weight',
                     torch.nn.Parameter(torch.zeros((16, 32) if kind == 'A' else (32, 16))))
                    for target in ('q_proj','k_proj','v_proj','o_proj') for kind in ('A','B')]
    if len(trainable_guard(Parameters())) != 8:
        raise ValueError('Trainable parameter self-check failed')
    return {'evidence': 'synthetic CPU doubles; not pretrained or target GPU verification',
            'answer_masking': 'PASS', 'trainable_parameter_policy': 'PASS'}


def preflight(root):
    check_payload(root)
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    for module in ('training.remote_sensing', 'scripts.train_remote_sensing_adapter',
                   'scripts.evaluate_remote_sensing_adapter', 'eval.rgb_pilot', 'scripts.stage1_runtime'):
        location = Path(importlib.import_module(module).__file__).resolve()
        if not location.is_relative_to(root.resolve()):
            raise ValueError(f'Import escaped package: {module}')
    examples = select_examples(load_manifest(config(root, Path('/unloaded'), Path('/unused')).dataset_manifest,
                                             root/'images'), 'train', 1024, 17)
    if len(examples) != 1024 or len({e.sample_id for e in examples}) != 1024:
        raise ValueError('Training selection must contain exactly 1024 unique examples')
    # Walk the actual Trainer dataloader with a synthetic CPU model; no forward,
    # optimizer, backward, or pretrained weights are involved.
    import torch
    from transformers import Trainer, TrainingArguments
    with tempfile.TemporaryDirectory(prefix='satquery-b6-cpu-') as temporary:
        arguments=TrainingArguments(output_dir=temporary,use_cpu=True,
            per_device_train_batch_size=1,gradient_accumulation_steps=8,max_steps=128,
            remove_unused_columns=False,report_to='none',seed=17,data_seed=17)
        trainer=Trainer(model=torch.nn.Linear(1,1),args=arguments,train_dataset=list(range(1024)),
                        data_collator=lambda batch:{'sample_ids':torch.tensor(batch)})
        consumed=[int(i) for batch in trainer.get_train_dataloader() for i in batch['sample_ids']]
    if len(consumed)!=1024 or set(consumed)!=set(range(1024)):
        raise ValueError('Trainer dataloader did not consume exactly one copy of each example')
    plan = config(root, Path('/unloaded'), Path('/unused'))
    if (plan.max_steps, plan.batch_size, plan.gradient_accumulation_steps) != (128, 1, 8):
        raise ValueError('Pilot optimizer-step contract differs')
    return {'hashes': 'PASS', 'offline_package_imports': 'PASS', 'training_examples': len(examples),
            'sample_sequence_sha256': digest([e.sample_id for e in examples]),
            'planned_sample_presentations': 128*1*8, 'cpu_dataloader_unique_examples':len(set(consumed)), **synthetic_checks()}


def verify_base(root, weights):
    from scripts.stage1_runtime import verify_checkpoint
    spec = json.loads((root/'configs/model_artifacts.json').read_text())['providers']['qwen2.5vl-3b']
    proof = verify_checkpoint(weights, spec['revision'], spec['weight_sha256'])
    if not proof['passed']:
        raise ValueError('Pinned base revision/weights missing or mismatched')
    return proof


def inventory(root, weights):
    import torch
    spec = json.loads((root/'configs/model_artifacts.json').read_text())['providers']['qwen2.5vl-3b']
    gpu = [{'name': torch.cuda.get_device_properties(i).name,
            'vram_bytes': torch.cuda.get_device_properties(i).total_memory} for i in range(torch.cuda.device_count())]
    required = list(spec['weight_sha256'])+spec['required_files']+['model.safetensors.index.json']
    missing = [n for n in required if weights is None or not (weights/n).is_file()]
    plan=json.loads((root/'pilot-config.json').read_text())
    dependency_status={n:{'expected':v,'installed':installed(n),'matches':installed(n) is not None and installed(n).split('+')[0]==v} for n,v in plan['dependency_pins'].items()}
    return {'dependency_status':dependency_status, 'gpu': gpu, 'cuda_runtime': torch.version.cuda, 'disk': dict(zip(('total','used','free'),shutil.disk_usage(root))),
            'versions': {**versions(), 'bitsandbytes': installed('bitsandbytes')},
            'expected_revision':spec['revision'], 'weights_path':str(weights) if weights else None,
            'missing_weight_files':missing,
            'base_verification':verify_base(root,weights) if not missing else 'NOT VERIFIED; provision externally, no downloads performed'}


def installed(name):
    try: return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError: return None


def binding(root):
    m=check_payload(root)
    effective_config=config(root,Path('/unloaded'),Path('/unused')).serializable()
    for name in ('model_path','dataset_manifest','image_root','output_dir'):
        effective_config.pop(name,None)
    return digest({'source':m['source_fingerprint'], 'config':sha256_file(root/'pilot-config.json'),
                   'effective_training':effective_config, 'manifest':PILOT_IDENTITY})


def verify_resume(checkpoint, identity):
    evidence=json.loads((checkpoint/'pilot-checkpoint.json').read_text())
    if evidence.get('run_identity') != identity:
        raise ValueError('Resume source/config/manifest identity mismatch')
    hashes=evidence.get('files_sha256')
    if not isinstance(hashes,dict):
        raise ValueError('Resume checkpoint hash inventory missing')
    files={}
    for path in checkpoint.rglob('*'):
        if path.is_symlink():
            raise ValueError('Resume checkpoint symlink rejected')
        if path.is_file() and path.name!='pilot-checkpoint.json':
            files[str(path.relative_to(checkpoint))]=sha256_file(path)
    if set(files)!=set(hashes):
        raise ValueError('Resume checkpoint file inventory differs')
    for name in REQUIRED_CHECKPOINT_FILES:
        if name not in hashes:
            raise ValueError(f'Resume state missing: {name}')
    scaler_expected=evidence.get('scaler_expected')
    if scaler_expected is None and 'scaler.pt' in hashes:
        scaler_expected=True  # Legacy checkpoints recorded every saved file, but not scaler presence.
    if type(scaler_expected) is not bool or ('scaler.pt' in hashes)!=scaler_expected:
        raise ValueError('Resume scaler state is missing or inconsistent')
    for name,h in hashes.items():
        if files[name]!=h:
            raise ValueError(f'Resume checkpoint hash mismatch: {name}')
    state=json.loads((checkpoint/'trainer_state.json').read_text())
    if state.get('max_steps')!=128 or not 0<int(state['global_step'])<128:
        raise ValueError('Resume must retain the 128-step schedule and an incomplete checkpoint')
    if evidence.get('step',state['global_step'])!=state['global_step']:
        raise ValueError('Resume checkpoint step differs from Trainer state')
    return evidence


def train(root, args, began):
    from scripts import train_remote_sensing_adapter as runner
    from transformers import Trainer, TrainerCallback
    identity=binding(root)
    resume_evidence=verify_resume(args.resume_from,identity) if args.resume_from else None
    trainer_ref={}
    class Budget(TrainerCallback):
        def on_step_end(self, args_, state, control, **kwargs):
            if time.monotonic()-began >= args.max_seconds-120:
                control.should_training_stop=True; control.should_save=True
            return control
        def on_save(self, args_, state, control, **kwargs):
            checkpoint=Path(args_.output_dir)/f'checkpoint-{state.global_step}'
            hashes={str(p.relative_to(checkpoint)):sha256_file(p) for p in checkpoint.rglob('*') if p.is_file() and p.name!='pilot-checkpoint.json'}
            scaler_expected=getattr(trainer_ref['trainer'].accelerator,'scaler',None) is not None
            if ('scaler.pt' in hashes)!=scaler_expected:
                raise ValueError('Trainer scaler save behavior differs from runtime')
            (checkpoint/'pilot-checkpoint.json').write_text(json.dumps({'run_identity':identity,'step':state.global_step,'scaler_expected':scaler_expected,'files_sha256':hashes},indent=2))
    class PilotTrainer(Trainer):
        def __init__(self, **kwargs):
            if len(kwargs['train_dataset'])!=1024:
                raise ValueError('Trainer dataset differs from frozen 1024 examples')
            kwargs['args'].save_steps=16
            kwargs['args'].save_total_limit=None
            super().__init__(**kwargs, callbacks=[Budget()])
            trainer_ref['trainer']=self
        def train(self, *a, **kw):
            if time.monotonic()-began >= args.max_seconds-120:
                raise TimeoutError('No remaining training budget after setup')
            if resume_evidence is not None:
                saved_scaler=resume_evidence.get('scaler_expected',True)
                runtime_scaler=getattr(self.accelerator,'scaler',None) is not None
                if saved_scaler!=runtime_scaler:
                    raise ValueError('Resume runtime scaler differs from saved checkpoint')
            return super().train(*a,resume_from_checkpoint=str(args.resume_from) if args.resume_from else None,**kw)
    def components(cfg):
        model,processor=load_training_components(cfg)
        from models.qwen_vl.stage1 import resolve_vision_module
        if resolve_vision_module(model).patch_embed.proj.in_channels!=3:
            raise ValueError('Original RGB projection required')
        names=trainable_guard(model)
        (args.output/'trainable-parameters.json').write_text(json.dumps(names,indent=2))
        return model,processor
    report=runner.run(config(root,args.weights,args.output), component_loader=components, trainer_factory=PilotTrainer)
    checkpoints=sorted((args.output/'checkpoints').glob('checkpoint-*'),key=lambda p:int(p.name.split('-')[-1]))
    step=json.loads((checkpoints[-1]/'trainer_state.json').read_text())['global_step'] if checkpoints else 0
    report.update(status='training_complete' if step==128 else 'budget_stopped_incomplete',global_step=step,
                  run_identity=identity,base_revision=json.loads((root/'pilot-config.json').read_text())['base_revision'])
    (args.output/'training-report.json').write_text(json.dumps(report,indent=2))
    return report


def baseline(root,args):
    import torch
    from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration
    from qwen_vl_utils import process_vision_info
    from scripts.evaluate_remote_sensing_adapter import _answer
    from eval.rgb_pilot import score,summarise
    rows,examples,p=validate_pilot(*paths(root))
    processor=AutoProcessor.from_pretrained(args.weights,local_files_only=True)
    model=Qwen2_5_VLForConditionalGeneration.from_pretrained(args.weights,local_files_only=True,torch_dtype=torch.float16,device_map='auto').eval()
    for parameter in model.parameters(): parameter.requires_grad_(False)
    conditions={}
    for mode in ('real','constant'):
        records=[]
        for row,example in zip(rows,examples):
            prediction=_answer(model,processor,process_vision_info,example,392,blind=mode=='constant')
            records.append({**row,'expected':row['answer'],'prediction':prediction,**score(prediction,row['answer'],row['question_type'])})
        conditions['base_'+mode]={'records':records,'summary':summarise(records)}
    return {'manifest_identity':p['manifest_identity'],'conditions':conditions,'versions':versions(),
            'preprocessing_image_size':392,'do_sample':False,'max_new_tokens':32}


def supervise(args):
    with (args.output/'worker.log').open('x') as log:
        try:
            result=subprocess.run([sys.executable,str(Path(__file__).resolve()),*sys.argv[1:],'--worker'],
                stdout=log,stderr=subprocess.STDOUT,timeout=args.max_seconds)
            return {'exit_code':result.returncode,'status':'completed' if result.returncode==0 else 'failed; outputs preserved'}
        except subprocess.TimeoutExpired:
            return {'exit_code':124,'status':'hard runtime limit; partial outputs preserved'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['preflight','inventory','baseline','train','evaluate'])
    p.add_argument('--weights',type=Path);p.add_argument('--adapter',type=Path)
    p.add_argument('--output',type=Path);p.add_argument('--resume-from',type=Path)
    p.add_argument('--budget-ledger',type=Path)
    p.add_argument('--max-seconds',type=int);p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    args=p.parse_args();os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1')
    if args.stage=='preflight': print(json.dumps(preflight(ROOT),indent=2));return
    if args.stage=='inventory':print(json.dumps(inventory(ROOT,args.weights),indent=2));return
    limits={'baseline':1200,'train':3600,'evaluate':2400}
    if not args.budget_ledger or not args.weights or not args.output or args.max_seconds is None or not 121<=args.max_seconds<=limits[args.stage]:
        p.error('GPU stages require weights, fresh output, and max-seconds within the approved proposal')
    if args.stage=='evaluate' and not args.adapter:p.error('evaluate requires adapter')
    if args.resume_from and args.stage!='train':p.error('resume-from is train-only')
    if not args.worker:
        preflight(ROOT)
        args.output.mkdir(parents=True,exist_ok=False)
        (args.output/'session.json').write_text(json.dumps({'stage':args.stage,'run_identity':binding(ROOT),'budget_seconds':args.max_seconds,'resume_from':str(args.resume_from) if args.resume_from else None},indent=2))
        ledger=args.budget_ledger
        ledger.parent.mkdir(parents=True,exist_ok=True)
        lock=ledger.with_suffix('.lock')
        lock_fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
        os.close(lock_fd)
        reservation_started=time.monotonic()
        try:
            entries=json.loads(ledger.read_text()) if ledger.exists() else []
            if sum(e['charged_seconds'] for e in entries)+args.max_seconds>7200:
                raise ValueError('Shared two-hour budget exhausted; new approval required')
            entries.append({'output':str(args.output.resolve()),'stage':args.stage,'charged_seconds':args.max_seconds})
            ledger.write_text(json.dumps(entries,indent=2))
            status=supervise(args)
            entries[-1]['charged_seconds']=min(args.max_seconds,int(time.monotonic()-reservation_started)+1)
            ledger.write_text(json.dumps(entries,indent=2))
        finally:
            lock.unlink()
        (args.output/'exit.json').write_text(json.dumps(status,indent=2))
        print(json.dumps(status));raise SystemExit(status['exit_code'])
    began=time.monotonic()

    plan=json.loads((ROOT/'pilot-config.json').read_text())
    mismatches=[n for n,v in plan['dependency_pins'].items() if installed(n) is None or installed(n).split('+')[0]!=v]
    if mismatches:raise ValueError(f'Target dependency pins not provisioned: {mismatches}')
    proof=verify_base(ROOT,args.weights)
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise ValueError('GPU stage requires exactly one visible CUDA GPU')
    if args.stage=='train':report=train(ROOT,args,began)
    elif args.stage=='baseline':report=baseline(ROOT,args)
    else:
        from scripts.evaluate_remote_sensing_adapter import run_pilot
        provenance,manifest,images=paths(ROOT)
        report=run_pilot(argparse.Namespace(pilot_provenance=provenance,dataset_manifest=manifest,image_root=images,
            model_path=args.weights,adapter_path=args.adapter,image_size=392,seed=17,bootstrap_samples=2000))
    report.update(base_verification=proof,run_identity=binding(ROOT))
    (args.output/'result.json').write_text(json.dumps(report,indent=2,allow_nan=False))


if __name__=='__main__':main()
