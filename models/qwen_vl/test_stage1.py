import pytest

# CI runs models/ without torch; skip rather than fail collection there.
torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from models.qwen_vl.stage1 import convert_patch_embed, pack_s2_pixel_values, stage1_parameter_groups  # noqa: E402


class TinyModel(nn.Module):
    def __init__(self, *, bias=True, dtype=torch.float64):
        super().__init__()
        self.visual = nn.Module()
        self.visual.patch_embed = nn.Module()
        self.visual.patch_embed.proj = nn.Conv2d(
            3, 5, 3, stride=2, padding=2, dilation=2, bias=bias,
            padding_mode="reflect", dtype=dtype,
        )
        self.visual.other = nn.Linear(2, 2, dtype=dtype)
        self.lm = nn.Module()
        self.lm.lora_A = nn.Linear(2, 2, dtype=dtype)


@pytest.mark.parametrize("bias", [True, False])
def test_patch_embed_conversion_and_equivalence(bias):
    model = TinyModel(bias=bias)
    old = model.visual.patch_embed.proj
    old_weight = old.weight.detach().clone()
    old_bias = old.bias.detach().clone() if bias else None
    signal = torch.randn(2, 1, 17, 19, dtype=old.weight.dtype)
    reference = old(signal.repeat(1, 3, 1, 1))

    new = convert_patch_embed(model)
    assert new.in_channels == 12 and new.out_channels == old.out_channels
    assert (new.kernel_size, new.stride, new.padding, new.dilation, new.groups, new.padding_mode) == (
        old.kernel_size, old.stride, old.padding, old.dilation, old.groups, old.padding_mode)
    assert new.weight.dtype == old.weight.dtype and new.weight.device == old.weight.device
    torch.testing.assert_close(new.weight, old_weight.mean(dim=1, keepdim=True).repeat(1, 12, 1, 1) * (3 / 12), rtol=0, atol=0)
    if bias:
        torch.testing.assert_close(new.bias, old_bias, rtol=0, atol=0)
    else:
        assert new.bias is None
    result = new(signal.repeat(1, 12, 1, 1))
    assert result.shape == reference.shape
    torch.testing.assert_close(result, reference, rtol=1e-12, atol=1e-12)
    with pytest.raises(RuntimeError):
        new(torch.ones(1, 3, 17, 19, dtype=new.weight.dtype))
    with pytest.raises(ValueError, match="unconverted 3-channel"):
        convert_patch_embed(model)


def test_parameter_groups_classify_trainable_parameters():
    model = TinyModel()
    convert_patch_embed(model)
    groups = stage1_parameter_groups(model)
    assert [(group["name"], group["lr"]) for group in groups] == [
        ("vision", 1e-5), ("patch_embed", 1e-4), ("lm_lora", 1e-4)]
    assert {id(p) for group in groups for p in group["params"]} == {
        id(p) for p in model.parameters() if p.requires_grad}
    model.lm.unexpected = nn.Linear(2, 2)
    with pytest.raises(ValueError, match="Unexpected trainable parameter"):
        stage1_parameter_groups(model)


def test_temporal_patch_embed_conversion():
    model = nn.Module()
    model.visual = nn.Module()
    model.visual.patch_embed = nn.Module()
    model.visual.patch_embed.in_channels = 3
    model.visual.patch_embed.proj = nn.Conv3d(3, 5, (2, 14, 14), stride=(2, 14, 14), bias=False)
    old = model.visual.patch_embed.proj
    weight = old.weight.detach().clone()
    gray = torch.randn(2, 1, 2, 28, 28)
    reference = old(gray.repeat(1, 3, 1, 1, 1))
    new = convert_patch_embed(model)
    assert model.visual.patch_embed.in_channels == 12
    assert new.weight.shape == (5, 12, 2, 14, 14)
    torch.testing.assert_close(new.weight, weight.mean(1, keepdim=True).repeat(1, 12, 1, 1, 1) * .25, rtol=0, atol=0)
    torch.testing.assert_close(new(gray.repeat(1, 12, 1, 1, 1)), reference, rtol=0, atol=1e-5)
    with pytest.raises(ValueError, match="unconverted 3-channel"):
        convert_patch_embed(model)


def test_pack_s2_pixel_values():
    source = torch.arange(12 * 120 * 120, dtype=torch.float32).reshape(12, 120, 120)
    values, grid = pack_s2_pixel_values(source)
    assert values.shape == (64, 4704)
    assert grid.tolist() == [[1, 8, 8]]
    recovered = values.reshape(1, 4, 4, 2, 2, 12, 2, 14, 14).permute(0, 6, 5, 1, 3, 7, 2, 4, 8)
    recovered = recovered.reshape(2, 12, 112, 112)
    expected = torch.nn.functional.interpolate(source[None], size=(112, 112), mode="bicubic", align_corners=False)[0]
    torch.testing.assert_close(recovered[0], expected, rtol=0, atol=0)
    torch.testing.assert_close(recovered[1], expected, rtol=0, atol=0)
    with pytest.raises(ValueError, match="Expected finite"):
        pack_s2_pixel_values(source.double())
