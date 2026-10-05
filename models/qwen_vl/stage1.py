"""Stage-1 12-band Qwen vision preparation."""

import torch
from torch import nn
from torch.nn import functional as F


def pack_s2_pixel_values(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack one contract-normalized CHW patch for Qwen's 2-frame, 14px patches."""
    if tensor.shape != (12, 120, 120) or tensor.dtype != torch.float32 or not torch.isfinite(tensor).all():
        raise ValueError("Expected finite float32 [12, 120, 120] S2 tensor")
    image = F.interpolate(tensor[None], size=(112, 112), mode="bicubic", align_corners=False)[0]
    frames = image[None].repeat(2, 1, 1, 1)
    patches = frames.reshape(1, 2, 12, 4, 2, 14, 4, 2, 14)
    pixel_values = patches.permute(0, 3, 6, 4, 7, 2, 1, 5, 8).reshape(64, 4704)
    return pixel_values, torch.tensor([[1, 8, 8]], dtype=torch.long)


def convert_patch_embed(model: nn.Module) -> nn.Conv2d | nn.Conv3d:
    patch_embed = model.visual.patch_embed
    old = patch_embed.proj
    if not isinstance(old, (nn.Conv2d, nn.Conv3d)) or old.in_channels != 3 or old.groups != 1:
        raise ValueError("Expected an unconverted 3-channel, groups=1 Conv2d or Conv3d patch embedding")
    new = type(old)(
        12, old.out_channels, old.kernel_size, old.stride,
        padding=old.padding, dilation=old.dilation, groups=old.groups,
        bias=old.bias is not None, padding_mode=old.padding_mode,
        device=old.weight.device, dtype=old.weight.dtype,
    )
    with torch.no_grad():
        mean_w = old.weight.mean(dim=1, keepdim=True)
        new.weight.copy_(mean_w.repeat(1, 12, *([1] * (old.weight.ndim - 2))) * (3.0 / 12.0))
        if old.bias is not None:
            new.bias.copy_(old.bias)
    new.weight.requires_grad_(old.weight.requires_grad)
    if old.bias is not None:
        new.bias.requires_grad_(old.bias.requires_grad)
    patch_embed.proj = new
    if hasattr(patch_embed, "in_channels"):
        patch_embed.in_channels = 12
    if hasattr(model, "config") and hasattr(model.config, "vision_config"):
        model.config.vision_config.in_channels = 12
    return new


def stage1_parameter_groups(model: nn.Module) -> list[dict]:
    """Group trainable vision, replacement patch embed, and LM LoRA parameters."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    patch = base.visual.patch_embed.proj
    if not isinstance(patch, (nn.Conv2d, nn.Conv3d)) or patch.in_channels != 12:
        raise ValueError("Convert the patch embedding before preparing parameter groups")
    patch_ids = {id(parameter) for parameter in patch.parameters()}
    vision_ids = {id(parameter) for parameter in base.visual.parameters()}
    groups = {"vision": [], "patch_embed": [], "lm_lora": []}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if id(parameter) in patch_ids:
            groups["patch_embed"].append(parameter)
        elif id(parameter) in vision_ids:
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
