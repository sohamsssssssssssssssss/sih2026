"""Stage-1 12-band Qwen vision preparation."""

import torch
from torch import nn
from torch.nn import functional as F


def resolve_vision_module(model: nn.Module) -> nn.Module:
    """Resolve Qwen's vision tower across direct/nested and PEFT layouts.

    Transformers 4.49 owns ``visual`` directly; newer conditional-generation
    wrappers own ``model.visual``. Never guess by scanning arbitrary modules.
    """
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    nested = getattr(base, "model", None)
    candidates = [getattr(base, "visual", None), getattr(nested, "visual", None)]
    towers = []
    for candidate in candidates:
        if candidate is None:
            continue
        if not isinstance(candidate, nn.Module) or not isinstance(getattr(candidate, "patch_embed", None), nn.Module):
            raise ValueError("Qwen vision module at visual/model.visual is not a tower with patch_embed")
        if all(candidate is not tower for tower in towers):
            towers.append(candidate)
    if len(towers) != 1:
        raise ValueError("Cannot resolve a unique Qwen vision tower at visual or model.visual (after PEFT unwrap)")
    return towers[0]


def vision_output_tensor(output: object) -> torch.Tensor:
    """Return merged Qwen vision embeddings consumed by the language model.

    Older towers return a Tensor; Transformers 5.16.1 returns
    BaseModelOutputWithPooling and consumes pooler_output, not last_hidden_state.
    Tuple/dict guessing is intentionally unsupported.
    """
    if isinstance(output, torch.Tensor):
        tensor = output
    else:
        from transformers.modeling_outputs import BaseModelOutputWithPooling

        if not isinstance(output, BaseModelOutputWithPooling):
            raise TypeError("Unsupported Qwen vision output: expected Tensor or BaseModelOutputWithPooling")
        tensor = output.pooler_output
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("Qwen vision output must contain a Tensor in pooler_output")
    if tensor.ndim != 2 or tensor.numel() == 0 or not tensor.is_floating_point():
        raise ValueError("Qwen vision embeddings must be a nonempty floating Tensor [tokens, hidden_size]")
    return tensor


def pack_s2_pixel_values(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack one contract-normalized CHW patch for Qwen's 2-frame, 14px patches."""
    if tensor.shape != (12, 120, 120) or tensor.dtype != torch.float32 or not torch.isfinite(tensor).all():
        raise ValueError("Expected finite float32 [12, 120, 120] S2 tensor")
    image = F.interpolate(tensor[None], size=(112, 112), mode="bicubic", align_corners=False)[0]
    frames = image[None].repeat(2, 1, 1, 1)
    patches = frames.reshape(1, 2, 12, 4, 2, 14, 4, 2, 14)
    pixel_values = patches.permute(0, 3, 6, 4, 7, 2, 1, 5, 8).reshape(64, 4704)
    return pixel_values, torch.tensor([[1, 8, 8]], dtype=torch.long)


LEGACY_INITIALIZATION = "mean_repeat_v1"
RGB_INITIALIZATION = "rgb_preserving_zero9_v1"


def patch_initialization(config: dict) -> str:
    value = config.get("patch_embed_initialization", LEGACY_INITIALIZATION)
    if value not in (LEGACY_INITIALIZATION, RGB_INITIALIZATION):
        raise ValueError(f"Unsupported patch_embed_initialization: {value!r}")
    return value


def validate_initialization_metadata(saved: dict, config: dict) -> None:
    expected = patch_initialization(config)
    recorded = saved.get("patch_embed_initialization")
    if recorded != expected and not (recorded is None and "patch_embed_initialization" not in config):
        raise ValueError("Checkpoint patch embedding initialization differs or is missing")


def convert_patch_embed(model: nn.Module, initialization: str = LEGACY_INITIALIZATION) -> nn.Conv2d | nn.Conv3d:
    patch_initialization({"patch_embed_initialization": initialization})
    patch_embed = resolve_vision_module(model).patch_embed
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
        if initialization == RGB_INITIALIZATION:
            new.weight.zero_()
            for channel, band in enumerate((3, 2, 1)):  # R/G/B -> B04/B03/B02.
                new.weight[:, band].copy_(old.weight[:, channel])
        else:
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
    visual = resolve_vision_module(model)
    patch = visual.patch_embed.proj
    if not isinstance(patch, (nn.Conv2d, nn.Conv3d)) or patch.in_channels != 12:
        raise ValueError("Convert the patch embedding before preparing parameter groups")
    patch_ids = {id(parameter) for parameter in patch.parameters()}
    vision_ids = {id(parameter) for parameter in visual.parameters()}
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
