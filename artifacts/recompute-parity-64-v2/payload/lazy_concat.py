"""Generated-only CPU FP32 prototype for the final nnU-Net decoder concat.

The normal decoder materializes an entire channel concatenation before its
first Conv3d. This scoped prototype concatenates only the depth slab and
receptive-field halo needed by each output tile. It leaves complete Conv3d
outputs and subsequent InstanceNorm maps intact. No patient or clinical claim.
"""
import hashlib
import inspect
from pathlib import Path
import types


_PINNED_UNET_DECODER_SOURCE_SHA256 = (
    "5eec090b199daf154c8e0ab3d64c45a1b1621f5aac68c6dea6a7203b0ddcc999"
)
_PINNED_SIMPLE_CONV_BLOCKS_SOURCE_SHA256 = (
    "02a7c21bca3c0acc20923a10970f14e853510f32a1f7ea2ef068cf636559ba2e"
)


def _has_input_hooks(module):
    return bool(module._forward_pre_hooks or module._forward_hooks)


def _torch_and_adapter():
    import torch
    import torch.nn.functional as functional
    from resectionlab.low_memory_conv3d import (identity_snapshot, output_dim,
                                                 triple, _tiled_forward)
    return torch, functional, identity_snapshot, output_dim, triple, _tiled_forward


def concat_conv3d_tiled_depth(up, skip, weight, bias=None, stride=1, padding=0,
                              dilation=1, groups=1, output_depth_tile=1):
    """Conv3d of channel-concatenated inputs without the full-volume concat."""
    torch, functional, _, output_dim, triple, _ = _torch_and_adapter()
    if up.ndim != 5 or skip.ndim != 5 or weight.ndim != 5:
        raise ValueError("expected N,C,D,H,W tensors and O,I,kD,kH,kW weight")
    if up.shape[0] != skip.shape[0] or up.shape[2:] != skip.shape[2:]:
        raise ValueError("concatenated inputs must share batch and spatial shape")
    if (up.device.type != "cpu" or skip.device.type != "cpu" or
            weight.device.type != "cpu" or up.dtype != torch.float32 or
            skip.dtype != torch.float32 or weight.dtype != torch.float32):
        raise ValueError("prototype requires CPU FP32 inputs and weights")
    if bias is not None and (bias.device.type != "cpu" or bias.dtype != torch.float32):
        raise ValueError("prototype requires CPU FP32 bias")
    if not isinstance(output_depth_tile, int) or output_depth_tile <= 0:
        raise ValueError("output depth tile must be a positive integer")
    stride, padding, dilation = triple(stride), triple(padding), triple(dilation)
    if min(stride) <= 0 or min(dilation) <= 0 or min(padding) < 0:
        raise ValueError("invalid convolution geometry")
    batch, up_channels, depth, height, width = up.shape
    input_channels = up_channels + skip.shape[1]
    output_channels, channels_per_group, kd, kh, kw = weight.shape
    if groups <= 0 or input_channels != channels_per_group * groups or output_channels % groups:
        raise ValueError("input/output channels and groups do not match")
    if padding[0] > dilation[0] * (kd - 1):
        raise ValueError("depth padding exceeds kernel support")
    out_shape = tuple(output_dim(size, kernel, step, pad, dil) for size, kernel, step, pad, dil in zip(
        (depth, height, width), (kd, kh, kw), stride, padding, dilation))
    if min(out_shape) <= 0:
        raise ValueError("convolution has empty output")
    result = torch.empty((batch, output_channels, *out_shape), dtype=up.dtype, device=up.device)
    for out_begin in range(0, out_shape[0], output_depth_tile):
        out_end = min(out_begin + output_depth_tile, out_shape[0])
        needed_begin = out_begin * stride[0] - padding[0]
        needed_end = (out_end - 1) * stride[0] - padding[0] + dilation[0] * (kd - 1) + 1
        sl = slice(max(0, needed_begin), min(depth, needed_end))
        slab = torch.cat((up[:, :, sl], skip[:, :, sl]), dim=1)
        left, right = max(0, -needed_begin), max(0, needed_end - depth)
        if left or right:
            slab = functional.pad(slab, (0, 0, 0, 0, left, right))
        tile = functional.conv3d(slab, weight, bias=bias, stride=stride,
                                 padding=(0, padding[1], padding[2]),
                                 dilation=dilation, groups=groups)
        if tile.shape[2:] != (out_end - out_begin, *out_shape[1:]):
            raise RuntimeError("lazy concat convolution output geometry changed")
        result[:, :, out_begin:out_end] = tile
    return result


def _first_conv_with_lazy_concat(self, pair):
    if not isinstance(pair, tuple) or len(pair) != 2:
        raise ValueError("lazy final decoder first convolution needs (upsampled, skip)")
    return concat_conv3d_tiled_depth(pair[0], pair[1], self.weight, self.bias,
                                     self.stride, self.padding, self.dilation,
                                     self.groups, self._depth_tile)


def _decoder_without_final_full_concat(self, skips):
    torch, _, _, _, _, _ = _torch_and_adapter()
    if self.deep_supervision or len(skips) != len(self.stages) + 1:
        raise ValueError("unexpected decoder supervision or skip structure")
    lower = skips[-1]
    for index in range(len(self.stages)):
        up = self.transpconvs[index](lower)
        skip = skips[-(index + 2)]
        if index == len(self.stages) - 1:
            merged = (up, skip)
        else:
            merged = torch.cat((up, skip), dim=1)
        lower = self.stages[index](merged)
    return self.seg_layers[-1](lower)


def attach_lazy_final_decoder_concat(network, output_depth_tile=1,
                                     expected_stages=5):
    """Patch only the final decoder input path after ordinary Conv3d tiling.

    All checks precede mutation; module, parameter and state-dict identities
    are compared before and after. `expected_stages=1` is for tiny generated
    fixtures only; the pinned trained architecture has five decoder stages.
    """
    torch, _, identity_snapshot, _, _, tiled_forward = _torch_and_adapter()
    if network.training or not isinstance(output_depth_tile, int) or output_depth_tile <= 0:
        raise ValueError("expected eval network and positive output tile")
    decoder = getattr(network, "decoder", None)
    if decoder is None or "forward" in decoder.__dict__ or decoder.deep_supervision:
        raise ValueError("decoder is missing, already wrapped or uses deep supervision")
    if expected_stages == 5:
        from dynamic_network_architectures.building_blocks.unet_decoder import UNetDecoder
        source = inspect.getsourcefile(UNetDecoder)
        if (type(decoder) is not UNetDecoder or
                getattr(decoder.forward, "__func__", None) is not UNetDecoder.forward or
                getattr(decoder.forward, "__self__", None) is not decoder or
                source is None or
                hashlib.sha256(Path(source).read_bytes()).hexdigest() !=
                _PINNED_UNET_DECODER_SOURCE_SHA256):
            raise ValueError("five-stage decoder is not the pinned UNetDecoder forward contract")
    if (len(decoder.stages) != expected_stages or
            len(decoder.transpconvs) != expected_stages or
            len(decoder.seg_layers) != expected_stages):
        raise ValueError("unexpected decoder stage structure")
    stage = decoder.stages[-1]
    if not hasattr(stage, "convs") or not stage.convs:
        raise ValueError("final decoder stage has no convolution blocks")
    block = stage.convs[0]
    first = getattr(block, "conv", None)
    if (not isinstance(first, torch.nn.Conv3d) or type(first) is not torch.nn.Conv3d or
            not hasattr(block, "all_modules") or block.all_modules[0] is not first):
        raise ValueError("unexpected first decoder convolution block")
    if expected_stages == 5:
        from dynamic_network_architectures.building_blocks.simple_conv_blocks import (
            ConvDropoutNormReLU, StackedConvBlocks)
        source = inspect.getsourcefile(StackedConvBlocks)
        if (source is None or
                hashlib.sha256(Path(source).read_bytes()).hexdigest() !=
                _PINNED_SIMPLE_CONV_BLOCKS_SOURCE_SHA256 or
                type(stage) is not StackedConvBlocks or
                type(block) is not ConvDropoutNormReLU or
                type(stage.convs) is not torch.nn.Sequential or
                type(block.all_modules) is not torch.nn.Sequential or
                "forward" in stage.__dict__ or "forward" in block.__dict__ or
                "forward" in stage.convs.__dict__ or
                "forward" in block.all_modules.__dict__ or
                getattr(stage.forward, "__func__", None) is not StackedConvBlocks.forward or
                getattr(stage.forward, "__self__", None) is not stage or
                getattr(stage.convs.forward, "__func__", None) is not torch.nn.Sequential.forward or
                getattr(stage.convs.forward, "__self__", None) is not stage.convs or
                getattr(block.forward, "__func__", None) is not ConvDropoutNormReLU.forward or
                getattr(block.forward, "__self__", None) is not block or
                getattr(block.all_modules.forward, "__func__", None) is not torch.nn.Sequential.forward or
                getattr(block.all_modules.forward, "__self__", None) is not block.all_modules):
            raise ValueError("final decoder stage or block is not the pinned forward contract")
    if (any(_has_input_hooks(module) for module in
            (stage, stage.convs, block, block.all_modules, first)) or
            torch.nn.modules.module._global_forward_pre_hooks or
            torch.nn.modules.module._global_forward_hooks):
        raise ValueError("final decoder paired-input path has incompatible hooks")
    if (len(block.all_modules) != 3 or
            not isinstance(block.all_modules[1], torch.nn.InstanceNorm3d) or
            not isinstance(block.all_modules[2], torch.nn.LeakyReLU)):
        raise ValueError("unsupported final block operation order or dropout")
    if (first.padding_mode != "zeros" or first.weight.device.type != "cpu" or
            first.weight.dtype != torch.float32 or
            (first.bias is not None and (first.bias.device.type != "cpu" or
                                         first.bias.dtype != torch.float32)) or
            first.padding[0] > first.dilation[0] * (first.kernel_size[0] - 1)):
        raise ValueError("unsupported first convolution padding/device/dtype")
    if (first.groups != 1 or first.in_channels != 2 * decoder.transpconvs[-1].out_channels or
            getattr(first, "_depth_tile", None) != output_depth_tile or
            getattr(getattr(first, "forward", None), "__func__", None) is not tiled_forward):
        raise ValueError("first decoder convolution must have the pinned tiled input contract")
    before = identity_snapshot(network)
    old_first_forward = first.forward
    first.forward = types.MethodType(_first_conv_with_lazy_concat, first)
    decoder.forward = types.MethodType(_decoder_without_final_full_concat, decoder)
    after = identity_snapshot(network)
    if before != after:
        first.forward = old_first_forward
        del decoder.forward
        raise RuntimeError("lazy decoder attachment changed state/module identities")
    return {"final_decoder_stages": expected_stages,
            "full_concat_avoided_only_at_last_stage": True,
            "first_conv_depth_tile": output_depth_tile,
            "parameter_state_module_identities_preserved": True}
