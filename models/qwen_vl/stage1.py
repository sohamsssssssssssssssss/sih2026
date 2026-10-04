"""Stage-1 12-band Qwen vision preparation; no model loading or training."""

import torch
from torch import nn


def convert_patch_embed(model: nn.Module) -> nn.Conv2d:
    old = model.visual.patch_embed.proj
    if not isinstance(old, nn.Conv2d) or old.in_channels != 3 or old.groups != 1:
        raise ValueError("Expected an unconverted 3-channel, groups=1 Conv2d patch embedding")
    new = nn.Conv2d(
        12, old.out_channels, old.kernel_size, old.stride,
        padding=old.padding, dilation=old.dilation, groups=old.groups,
        bias=old.bias is not None, padding_mode=old.padding_mode,
        device=old.weight.device, dtype=old.weight.dtype,
    )
    with torch.no_grad():
        mean_w = old.weight.mean(dim=1, keepdim=True)
        new.weight.copy_(mean_w.repeat(1, 12, 1, 1) * (3.0 / 12.0))
        if old.bias is not None:
            new.bias.copy_(old.bias)
    new.weight.requires_grad_(old.weight.requires_grad)
    if old.bias is not None:
        new.bias.requires_grad_(old.bias.requires_grad)
    model.visual.patch_embed.proj = new
    return new


def stage1_parameter_groups(model: nn.Module) -> list[dict]:
    """Group trainable vision, replacement patch embed, and LM LoRA parameters."""
    patch = model.visual.patch_embed.proj
    if not isinstance(patch, nn.Conv2d) or patch.in_channels != 12:
        raise ValueError("Convert the patch embedding before preparing parameter groups")
    patch_ids = {id(parameter) for parameter in patch.parameters()}
    groups = {"vision": [], "patch_embed": [], "lm_lora": []}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if id(parameter) in patch_ids:
            groups["patch_embed"].append(parameter)
        elif name.startswith("visual."):
            groups["vision"].append(parameter)
        elif "lora_" in name.lower():
            groups["lm_lora"].append(parameter)
        else:
            raise ValueError(f"Unexpected trainable parameter: {name}")
    return [
        {"name": name, "params": groups[name], "lr": lr}
        for name, lr in (("vision", 1e-5), ("patch_embed", 1e-4), ("lm_lora", 1e-4))
        if groups[name]
    ]
