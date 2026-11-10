"""Offline B13 training package; GPU work requires separate authorization."""
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
B13_IDENTITY='16f97f1f332725d7ddf6cfbe643d1e0a61f20ac32f0ad83399bf5e367de01282'
B13_SHA256='3f727b271fc6c684bcefc6eef10d6a42ade1c25cfe3954851ec163d882f5e13f'
B13_COUNT=6000
B13_STEPS=750

def paths(root):
    pilot = root / "data/manifests/rsvqa/pilot.v1"
    return pilot / "provenance.json", pilot / "validation.jsonl", root / "images"


def check_payload(root):
    manifest = json.loads((root / "package-manifest.json").read_text())
    for name, expected in manifest["payload_sha256"].items():
        if sha256_file(root / name) != expected:
            raise ValueError(f"Package payload mismatch: {name}")
    validate_pilot(*paths(root))
    training_manifest=root/'data/manifests/rsvqa/b13-6k.v1/train.jsonl'
    provenance=json.loads((training_manifest.parent/'provenance.json').read_text())
    check_b13_spec(json.loads((root/'pilot-config.json').read_text()),provenance,
                   sha256_file(training_manifest))
    if any(sha256_file(root/'images'/name)!=expected
           for name,expected in provenance['selected_image_sha256'].items()):
        raise ValueError('Frozen B13 image hash differs')
    return manifest


def check_b13_spec(plan, provenance, measured_sha):
    training=plan['training']
    if (plan.get('training_manifest_identity')!=B13_IDENTITY
            or plan.get('training_manifest_sha256')!=B13_SHA256
            or provenance.get('manifest_identity')!=B13_IDENTITY
            or provenance.get('manifest_sha256')!=B13_SHA256
            or measured_sha!=B13_SHA256):
        raise ValueError('Frozen B13 manifest identity or SHA256 differs')
    if (training['max_samples'],training['epochs'],training['max_steps'],
            training['batch_size'],training['gradient_accumulation_steps'],training['seed']) != (
            B13_COUNT,1.0,B13_STEPS,1,8,17):
        raise ValueError('Frozen B13 training schedule differs')


def check_b13_examples(examples):
    if len(examples)!=B13_COUNT or len({item.sample_id for item in examples})!=B13_COUNT:
        raise ValueError('B13 training selection must contain 6000 unique examples')
    return examples


def force_final_checkpoint(step):
    return step==B13_STEPS


def config(root, weights, output):
    plan = json.loads((root / "pilot-config.json").read_text())
    return TrainingConfig(model_path=weights, dataset_manifest=root / "data/manifests/rsvqa/b13-6k.v1/train.jsonl",
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
    plan=config(root,Path('/unloaded'),Path('/unused'))
    examples=check_b13_examples(select_examples(load_manifest(plan.dataset_manifest,plan.image_root),
                                                 'train',plan.max_samples,plan.seed))
    # Walk the Trainer dataloader twice with the actual examples and synthetic CPU model.
    import torch
    from transformers import Trainer, TrainingArguments
    with tempfile.TemporaryDirectory(prefix='satquery-b6-cpu-') as temporary:
        def order():
            arguments=TrainingArguments(output_dir=temporary,use_cpu=True,
                per_device_train_batch_size=plan.batch_size,
                gradient_accumulation_steps=plan.gradient_accumulation_steps,
                max_steps=plan.max_steps,remove_unused_columns=False,report_to='none',
                seed=plan.seed,data_seed=plan.seed)
            trainer=Trainer(model=torch.nn.Linear(1,1),args=arguments,train_dataset=examples,
                            data_collator=lambda batch:{'sample_ids':[item.sample_id for item in batch]})
            return [sample_id for batch in trainer.get_train_dataloader()
                    for sample_id in batch['sample_ids']]
        consumed=order()
        if consumed!=order():
            raise ValueError('Seed-17 Trainer dataloader order is not reproducible')
    if len(consumed)!=B13_COUNT or set(consumed)!={e.sample_id for e in examples}:
        raise ValueError('Trainer dataloader did not consume each B13 example exactly once')
    return {'hashes':'PASS','offline_package_imports':'PASS','manifest_identity':B13_IDENTITY,
            'training_examples':len(examples),'sample_sequence_sha256':digest([e.sample_id for e in examples]),
            'trainer_sample_order_sha256':digest(consumed),'planned_sample_presentations':B13_STEPS*8,
            'cpu_dataloader_unique_examples':len(set(consumed)),**synthetic_checks()}


def tokenization_preflight(root, weights):
    """Check every frozen training example with the pinned processor on CPU."""
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', CUDA_VISIBLE_DEVICES='')
    check_payload(root)
    proof = verify_base(root, weights)
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(weights, local_files_only=True)
    plan = config(root, weights, Path('/unused'))
    examples=check_b13_examples(select_examples(load_manifest(plan.dataset_manifest,plan.image_root),
                                                'train',plan.max_samples,plan.seed))
    collator = QwenVQACollator(processor, plan.image_size,
                              max_sequence_length=plan.max_sequence_length)
    longest_id, maximum, failures = None, 0, []
    tensor_shapes=set()
    for example in examples:
        try:
            batch = collator([example])
            prompt = collator._encode([example], include_target=False)['input_ids'][0]
            if ('pixel_values' not in batch or 'image_grid_thw' not in batch
                    or not batch['pixel_values'].numel() or not batch['image_grid_thw'].numel()):
                raise ValueError('Image-token path missing or empty')
            tensor_shapes.add(tuple(batch['pixel_values'].shape))
            ids, labels = batch['input_ids'][0], batch['labels'][0]
            pad = processor.tokenizer.pad_token_id
            prompt_length = int((prompt != pad).sum())
            full_length = int((ids != pad).sum())
            if (prompt_length >= full_length or not ids[:prompt_length].equal(prompt[:prompt_length])
                    or not bool((labels[:prompt_length] == -100).all())
                    or not labels[prompt_length:full_length].equal(ids[prompt_length:full_length])
                    or not bool((labels[full_length:] == -100).all())):
                raise ValueError('Answer-token masking mismatch')
            if ids.numel()>plan.max_sequence_length:
                raise ValueError('Encoded example exceeds sequence limit')
            if full_length>maximum:
                longest_id,maximum=example.sample_id,full_length
        except Exception as exc:
            failures.append({'sample_id':example.sample_id,'error':f'{type(exc).__name__}: {exc}'})
    return {'manifest_identity':B13_IDENTITY,'base_revision':proof['measured_revision'],
            'examples_checked':len(examples),'answer_masks_verified':len(examples)-len(failures),
            'max_encoded_sequence_length':maximum,'longest_sample_id':longest_id,
            'max_sequence_length':plan.max_sequence_length,
            'pixel_value_shapes':[list(shape) for shape in sorted(tensor_shapes)],
            'all_fit':not failures,'failures':failures}


def verify_base(root, weights):
    from scripts.stage1_runtime import verify_checkpoint
    spec = json.loads((root/'configs/model_artifacts.json').read_text())['providers']['qwen2.5vl-3b']
    proof = verify_checkpoint(weights, spec['revision'], spec['weight_sha256'],
                              repository_id=spec['model_id'])
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
                   'effective_training':effective_config, 'training_manifest':B13_IDENTITY,
                   'validation_manifest':PILOT_IDENTITY})


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
    if state.get('max_steps')!=B13_STEPS or not 0<int(state['global_step'])<B13_STEPS:
        raise ValueError('Resume must retain the 750-step schedule and an incomplete checkpoint')
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
            if force_final_checkpoint(state.global_step):
                control.should_save=True
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
            check_b13_examples(kwargs['train_dataset'])
            if (kwargs['args'].max_steps,kwargs['args'].per_device_train_batch_size,
                    kwargs['args'].gradient_accumulation_steps)!=(B13_STEPS,1,8):
                raise ValueError('Trainer differs from frozen B13 schedule')
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
    report.update(status='training_complete' if step==B13_STEPS else 'budget_stopped_incomplete',
                  global_step=step,run_identity=identity,training_manifest_identity=B13_IDENTITY,
                  base_revision=json.loads((root/'pilot-config.json').read_text())['base_revision'])
    (args.output/'training-report.json').write_text(json.dumps(report,indent=2))
    return report


def baseline(root,args):
    import torch
    from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration
    from qwen_vl_utils import process_vision_info
    from scripts.evaluate_remote_sensing_adapter import _answer
    from eval.rgb_pilot import SCORING_VERSION, score,summarise
    rows,examples,p=validate_pilot(*paths(root))
    run_path=args.output/'baseline-run.json'
    progress_path=args.output/'baseline-predictions.jsonl'
    result_path=args.output/'result.json'
    if args.resume and result_path.exists():
        raise ValueError('Baseline is already complete; refusing to resume')
    processor=AutoProcessor.from_pretrained(args.weights,local_files_only=True)
    model=Qwen2_5_VLForConditionalGeneration.from_pretrained(args.weights,local_files_only=True,torch_dtype=torch.float16,device_map='auto').eval()
    for parameter in model.parameters(): parameter.requires_grad_(False)
    spec=json.loads((root/'configs/model_artifacts.json').read_text())['providers']['qwen2.5vl-3b']
    run_config={'manifest_identity':p['manifest_identity'],
        'validation_manifest_sha256':sha256_file(paths(root)[1]),
        'selected_image_hashes_identity':digest(p['selected_image_sha256']),
        'base_model_id':spec['model_id'],'base_revision':spec['revision'],
        'base_weight_sha256':spec['weight_sha256'],'weights_path':str(args.weights.resolve()),
        'preprocessing_image_size':392,'conditions':['base_real','base_constant'],
        'constant_control':{'mode':'RGB','value':[127,127,127],'source_dimensions_before_same_resize':True},
        'scoring_version':SCORING_VERSION,'do_sample':False,'max_new_tokens':32,
        'processor_image_config':processor.image_processor.to_dict(),
        'chat_template':processor.chat_template,
        'generation_config':model.generation_config.to_dict(),'versions':versions(),
        'package_run_identity':binding(root)}
    run_identity=digest(run_config)
    if args.resume:
        saved=json.loads(run_path.read_text())
        if saved.get('run_identity')!=run_identity or saved.get('configuration')!=run_config:
            raise ValueError('Baseline manifest or run configuration mismatch')
        if not progress_path.exists():
            progress_path.open('xb').close()
        progress=_read_baseline_progress(progress_path,rows)
    else:
        if run_path.exists() or progress_path.exists() or result_path.exists():
            raise FileExistsError('Baseline output already contains run state; use --resume only for an interrupted run')
        _write_durable_json(run_path,{'run_identity':run_identity,'configuration':run_config})
        progress_path.open('xb').close()
        progress=[]
    for row_index,(row,example) in enumerate(zip(rows,examples)):
        for condition in ('base_real','base_constant'):
            position=row_index*2+int(condition=='base_constant')
            if position<len(progress):
                continue
            prediction=_answer(model,processor,process_vision_info,example,392,blind=condition=='base_constant')
            record={key:row[key] for key in ('sample_id','question_id','image_id','image','question_type','question')}
            record.update(condition=condition,expected=row['answer'],prediction=prediction,
                          **score(prediction,row['answer'],row['question_type']))
            _append_baseline_record(progress_path,record)
            progress.append(record)
    if len(progress)!=len(rows)*2:
        raise RuntimeError('Baseline predictions are incomplete; no result was published')
    conditions={}
    for condition in ('base_real','base_constant'):
        records=[record for record in progress if record['condition']==condition]
        if len(records)!=len(rows):
            raise RuntimeError('Baseline condition is incomplete; no result was published')
        conditions[condition]={'records':records,'summary':summarise(records)}
    return {'manifest_identity':p['manifest_identity'],'baseline_run_identity':run_identity,
            'conditions':conditions,'versions':versions(),'preprocessing_image_size':392,
            'do_sample':False,'max_new_tokens':32,'completed_predictions':len(progress),
            'expected_predictions':len(rows)*2}


def _write_durable_json(path,payload):
    temporary=path.with_name(f'{path.name}.tmp-{os.getpid()}')
    encoded=(json.dumps(payload,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
    with temporary.open('xb') as handle:
        handle.write(encoded);handle.flush();os.fsync(handle.fileno())
    os.replace(temporary,path)
    directory=os.open(path.parent,os.O_RDONLY)
    try:os.fsync(directory)
    finally:os.close(directory)


def _append_baseline_record(path,record):
    encoded=(json.dumps(record,sort_keys=True,allow_nan=False)+'\n').encode()
    with path.open('ab') as handle:
        handle.write(encoded);handle.flush();os.fsync(handle.fileno())


def _read_baseline_progress(path,rows):
    raw=path.read_bytes()
    if raw and not raw.endswith(b'\n'):
        boundary=raw.rfind(b'\n')+1
        with path.open('r+b') as handle:
            handle.truncate(boundary);handle.flush();os.fsync(handle.fileno())
        raw=raw[:boundary]
    lines=raw.splitlines()
    if len(lines)>len(rows)*2:
        raise ValueError('Baseline progress contains extra records')
    records=[]
    for index,line in enumerate(lines):
        record=json.loads(line)
        row=rows[index//2]
        condition=('base_real','base_constant')[index%2]
        expected={'sample_id':row['sample_id'],'question_id':row['question_id'],
            'image_id':row['image_id'],'image':row['image'],'question_type':row['question_type'],
            'question':row['question'],'expected':row['answer'],'condition':condition}
        if any(record.get(key)!=value for key,value in expected.items()):
            raise ValueError('Baseline progress is not the expected manifest/condition prefix')
        if not isinstance(record.get('prediction'),str):
            raise ValueError('Baseline progress prediction is malformed')
        from eval.rgb_pilot import score
        if any(record.get(key)!=value for key,value in score(record['prediction'],row['answer'],row['question_type']).items()):
            raise ValueError('Baseline progress score differs from the pinned scoring policy')
        records.append(record)
    return records


def supervise(args):
    with (args.output/'worker.log').open('a' if args.resume else 'x') as log:
        try:
            result=subprocess.run([sys.executable,str(Path(__file__).resolve()),*sys.argv[1:],'--worker'],
                stdout=log,stderr=subprocess.STDOUT,timeout=args.max_seconds)
            return {'exit_code':result.returncode,'status':'completed' if result.returncode==0 else 'failed; outputs preserved'}
        except subprocess.TimeoutExpired:
            return {'exit_code':124,'status':'hard runtime limit; partial outputs preserved'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['preflight','token-preflight','inventory','baseline','train','evaluate'])
    p.add_argument('--weights',type=Path);p.add_argument('--adapter',type=Path)
    p.add_argument('--output',type=Path);p.add_argument('--resume-from',type=Path)
    p.add_argument('--resume',action='store_true',help='Resume an interrupted baseline in the same output directory')
    p.add_argument('--budget-ledger',type=Path)
    p.add_argument('--max-seconds',type=int);p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    args=p.parse_args();os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1')
    if args.stage=='preflight': print(json.dumps(preflight(ROOT),indent=2));return
    if args.stage=='token-preflight':
        if not args.weights:p.error('token-preflight requires --weights')
        report=tokenization_preflight(ROOT,args.weights)
        print(json.dumps(report,indent=2))
        if report['failures']:raise SystemExit(1)
        return
    if args.stage=='inventory':print(json.dumps(inventory(ROOT,args.weights),indent=2));return
    limits={'baseline':1200,'train':3600,'evaluate':2400}
    if not args.budget_ledger or not args.weights or not args.output or args.max_seconds is None or not 121<=args.max_seconds<=limits[args.stage]:
        p.error('GPU stages require weights, fresh output, and max-seconds within the approved proposal')
    if args.stage=='evaluate' and not args.adapter:p.error('evaluate requires adapter')
    if args.resume_from and args.stage!='train':p.error('resume-from is train-only')
    if args.resume and args.stage!='baseline':p.error('--resume is baseline-only')
    if args.resume and args.resume_from:p.error('--resume and --resume-from cannot be combined')
    if not args.worker:
        checks=preflight(ROOT)
        ledger=args.budget_ledger
        ledger.parent.mkdir(parents=True,exist_ok=True)
        if args.resume:
            session_path=args.output/'session.json'
            if not args.output.is_dir() or not session_path.is_file():
                p.error('Baseline resume requires the original output directory and session.json')
            session=json.loads(session_path.read_text())
            if (session.get('stage')!='baseline' or session.get('run_identity')!=binding(ROOT)
                    or session.get('budget_ledger')!=str(ledger.resolve())
                    or not (args.output/'baseline-run.json').is_file()):
                p.error('Baseline resume session, package identity, ledger, or run configuration is missing/mismatched')
            if (args.output/'result.json').exists():
                p.error('Baseline result is already complete; refusing resume')
            if (args.output/'exit.json').is_file() and json.loads((args.output/'exit.json').read_text()).get('exit_code')==0:
                p.error('Previous baseline completed; refusing resume')
        else:
            args.output.mkdir(parents=True,exist_ok=False)
            (args.output/'session.json').write_text(json.dumps({'stage':args.stage,'run_identity':binding(ROOT),
                **({'trainer_sample_order_sha256':checks['trainer_sample_order_sha256']}
                   if args.stage=='train' else {}),
                'budget_seconds':args.max_seconds,'budget_ledger':str(ledger.resolve()),
                'resume_from':str(args.resume_from) if args.resume_from else None},indent=2))
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
    if args.stage=='baseline':
        _write_durable_json(args.output/'result.json',report)
    else:
        (args.output/'result.json').write_text(json.dumps(report,indent=2,allow_nan=False))


if __name__=='__main__':main()
