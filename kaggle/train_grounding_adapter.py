"""Train or resume the grounding adapter from the checksum-pinned train pack."""
import argparse
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from time import monotonic

from PIL import Image

from kaggle.grounding_qwen_inputs import verify_qwen_directory
from kaggle.run_grounding_baselines import (
    CAPTION, PROMPT, REPO, digest, discover_pack, open_pack, pack_items, parse_box, require,
)

TRAIN_PACK_NAME = "satquery-grounding-train-pack"
RECEIPT = "ARTIFACT.json"


def tree_files(root):
    return {path.relative_to(root).as_posix(): path for path in root.rglob("*")
            if path.is_file() and path != root / RECEIPT}


def seal_tree(root, identity):
    files = {name: {"bytes": path.stat().st_size, "sha256": digest(path)}
             for name, path in sorted(tree_files(root).items())}
    receipt = root / RECEIPT
    receipt.write_text(json.dumps({"identity": identity, "files": files}, indent=2) + "\n")
    print(f"{receipt} SHA-256 {digest(receipt)}", flush=True)
    return receipt


def verify_tree(root, receipt_sha):
    receipt = root / RECEIPT
    require(receipt.is_file() and digest(receipt) == receipt_sha, f"artifact receipt mismatch: {root}")
    data = json.loads(receipt.read_bytes())
    files = tree_files(root)
    require(files.keys() == data["files"].keys(), f"artifact file set mismatch: {root}")
    for name, path in files.items():
        pin = data["files"][name]
        require(path.stat().st_size == pin["bytes"] and digest(path) == pin["sha256"], f"artifact file mismatch: {name}")
    return data["identity"]


def training_text(row):
    if row["task"] == "caption":
        return CAPTION, row["gold"]
    prompt = PROMPT.format(expression=row["expression"])
    answer = json.dumps([round(value * 1000) for value in row["gold"]])
    require(parse_box(answer) is not None, f"invalid native box target: {row['id']}")
    return prompt, answer


class GroundingCollator:
    def __init__(self, processor, pack, max_tokens):
        self.processor = processor
        self.pack = pack
        self.max_tokens = max_tokens

    def __call__(self, rows):
        require(len(rows) == 1, "grounding collator expects one image per microbatch")
        row = rows[0]
        prompt, answer = training_text(row)
        with Image.open(BytesIO(self.pack.read(row["image"]))) as source:
            image = source.convert("RGB")
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        prefix = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        full = self.processor.apply_chat_template(
            [*messages, {"role": "assistant", "content": [{"type": "text", "text": answer}]}],
            tokenize=False, add_generation_prompt=False,
        )
        before = self.processor(text=[prefix], images=[image], return_tensors="pt")["input_ids"]
        batch = self.processor(text=[full], images=[image], return_tensors="pt")
        ids = batch["input_ids"]
        length = before.shape[1]
        require(length < ids.shape[1] <= self.max_tokens and ids[:, :length].equal(before),
                f"assistant token boundary or context mismatch: {row['id']}")
        batch["labels"] = ids.clone()
        batch["labels"][:, :length] = -100
        return batch


def run(args, start):
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration,
        Trainer, TrainerCallback, TrainingArguments, set_seed,
    )

    require(torch.cuda.is_available() and "T4" in torch.cuda.get_device_name(0), "T4 required")
    config_path = REPO / "kaggle/grounding_train_config.json"
    config = json.loads(config_path.read_bytes())
    pins_path = REPO / "kaggle/grounding_qwen_hashes.json"
    pins = json.loads(pins_path.read_bytes())
    verify_qwen_directory(args.qwen_dir, pins)
    identity = {"config_sha256": digest(config_path), "train_pack_sha256": args.pack_sha256,
                "pack_manifest_sha256": args.manifest_sha256, "qwen_pins_sha256": digest(pins_path)}
    require(bool(args.resume) == bool(args.resume_receipt_sha256), "resume directory and pin are both required")
    if args.resume:
        require(verify_tree(args.resume, args.resume_receipt_sha256) == identity, "checkpoint identity mismatch")
    path = args.pack or discover_pack(args.input_root, TRAIN_PACK_NAME)
    require(not args.output.exists(), "training output already exists")
    set_seed(config["seed"])

    class Checkpoint(TrainerCallback):
        def on_step_end(self, _args, state, control, **kwargs):
            if monotonic() - start >= config["session_seconds"]:
                control.should_save = True
                control.should_training_stop = True
            return control

        def on_save(self, _args, state, control, **kwargs):
            seal_tree(args.output / f"checkpoint-{state.global_step}", identity)

    with open_pack(path, args.pack_sha256, args.manifest_sha256) as pack:
        rows = pack_items(pack, ("train",))
        require(len(rows) == 10000 and all(row["split"] == "train" for row in rows), "train inventory mismatch")
        processor = AutoProcessor.from_pretrained(args.qwen_dir, local_files_only=True)
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                  bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.qwen_dir, local_files_only=True, quantization_config=quant,
            torch_dtype=torch.float16, device_map={"": 0}, attn_implementation="sdpa",
        )
        model = prepare_model_for_kbit_training(model, gradient_checkpointing_kwargs={"use_reentrant": False})
        model.config.use_cache = False
        model = get_peft_model(model, LoraConfig(
            r=config["rank"], lora_alpha=config["lora_alpha"], lora_dropout=config["lora_dropout"],
            target_modules=r"model\.(?:language_model\.)?layers\.\d+\.self_attn\.[qkvo]_proj",
            task_type="CAUSAL_LM",
        ))
        training_args = TrainingArguments(
            output_dir=str(args.output), max_steps=config["max_steps"],
            per_device_train_batch_size=config["micro_batch_size"],
            gradient_accumulation_steps=config["gradient_accumulation_steps"],
            learning_rate=config["learning_rate"], warmup_steps=config["warmup_steps"],
            lr_scheduler_type="cosine", optim="paged_adamw_8bit", fp16=True,
            save_steps=config["save_steps"], save_total_limit=2, logging_steps=10,
            seed=config["seed"], data_seed=config["seed"], remove_unused_columns=False,
            report_to=[], dataloader_num_workers=0,
        )
        trainer = Trainer(model=model, args=training_args, train_dataset=rows,
                          data_collator=GroundingCollator(processor, pack, config["max_sequence_tokens"]),
                          callbacks=[Checkpoint()])
        trainer.train(resume_from_checkpoint=str(args.resume) if args.resume else None)
        if trainer.state.global_step == config["max_steps"]:
            final = args.output / "adapter"
            model.save_pretrained(final, safe_serialization=True)
            seal_tree(final, identity)
            weight = final / "adapter_model.safetensors"
            print(f"FINAL ADAPTER SHA-256 {digest(weight)}", flush=True)
        else:
            print(f"Resume from checkpoint-{trainer.state.global_step}", flush=True)


def main():
    start = monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--pack", type=Path)
    source.add_argument("--input-root", type=Path)
    parser.add_argument("--pack-sha256", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--qwen-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--resume-receipt-sha256")
    run(parser.parse_args(), start)


if __name__ == "__main__":
    main()
