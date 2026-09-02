"""Model construction from config. One place, so every script builds identically."""

from __future__ import annotations

import torch


def build_model(cfg: dict):
    """Return (model, processor) per cfg. Raises on unsupported backbone."""
    from transformers import AutoProcessor, BitsAndBytesConfig

    mc, hc = cfg["model"], cfg["hardware"]
    backbone = mc["backbone"]

    dtype = torch.float16 if hc["precision"] == "fp16" else torch.bfloat16
    kwargs = {
        "torch_dtype": dtype,
        "device_map": hc.get("device_map", "auto"),
        "attn_implementation": hc.get("attn_implementation", "sdpa"),
    }

    if mc.get("load_in_4bit"):
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=mc.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=mc.get("bnb_4bit_use_double_quant", True),
        )

    if "Qwen2.5-VL" in backbone or "Qwen2_5_VL" in backbone:
        from transformers import Qwen2_5_VLForConditionalGeneration as Cls
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")

    model = Cls.from_pretrained(backbone, **kwargs)
    processor = AutoProcessor.from_pretrained(backbone)

    if hc.get("gradient_checkpointing"):
        model.gradient_checkpointing_enable()

    lc = cfg.get("lora", {})
    if lc.get("enabled"):
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        if mc.get("load_in_4bit"):
            model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, LoraConfig(
            r=lc["r"],
            lora_alpha=lc["alpha"],
            lora_dropout=lc["dropout"],
            target_modules=lc["target_modules"],
            task_type="CAUSAL_LM",
        ))
        model.print_trainable_parameters()
    else:
        model.eval()

    return model, processor
