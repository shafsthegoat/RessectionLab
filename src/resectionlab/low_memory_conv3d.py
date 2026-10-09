"""Optional, CPU/FP32 low-memory Conv3d inference adapter.

Output-depth tiling preserves the full receptive field and assembles a complete
feature map before later operations such as InstanceNorm. PyTorch is imported
only when an operation is used, so importing this module needs no ML runtime.
This is a numerical software adapter, not a patient or clinical validator.
"""

import types

__all__ = ("conv3d_tiled_depth", "attach_tiled_conv3d", "identity_snapshot")


def _require_torch():
    try:
        import torch
        import torch.nn.functional as F
    except ImportError as error:
        raise RuntimeError("PyTorch is required for low-memory Conv3d inference") from error
    return torch, F


def triple(value):
    if isinstance(value, int):
        return (value, value, value)
    if len(value) != 3:
        raise ValueError("expected 3 spatial values")
    return tuple(int(v) for v in value)


def output_dim(length, kernel, stride, padding, dilation):
    return (length + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1


def conv3d_tiled_depth(x, weight, bias=None, stride=1, padding=0,
                       dilation=1, groups=1, output_depth_tile=4):
    """Compute exact output coordinates in depth slabs with native Conv3d.

    Each slab includes the whole input receptive-field halo for its requested
    output coordinates. Only depth is tiled; height/width remain full size.
    Call with `torch.inference_mode()` for the intended bounded-memory use.
    """
    torch, F = _require_torch()
    if x.ndim != 5 or weight.ndim != 5:
        raise ValueError("expected N,C,D,H,W input and O,I,kD,kH,kW weight")
    if x.device.type != "cpu" or weight.device.type != "cpu":
        raise ValueError("this prototype is CPU-only")
    if x.dtype != torch.float32 or weight.dtype != torch.float32:
        raise ValueError("this prototype is FP32-only")
    if output_depth_tile <= 0:
        raise ValueError("output_depth_tile must be positive")
    stride, padding, dilation = triple(stride), triple(padding), triple(dilation)
    if min(stride) <= 0 or min(dilation) <= 0 or min(padding) < 0:
        raise ValueError("invalid convolution geometry")
    n, cin, depth, height, width = x.shape
    cout, cpergroup, kd, kh, kw = weight.shape
    if groups <= 0 or cin != cpergroup * groups or cout % groups:
        raise ValueError("input/output channels and groups do not match")
    if padding[0] > dilation[0] * (kd - 1):
        raise ValueError("depth padding exceeds kernel support")
    output_shape = tuple(output_dim(s, k, st, p, d) for s, k, st, p, d in zip(
        (depth, height, width), (kd, kh, kw), stride, padding, dilation))
    if min(output_shape) <= 0:
        raise ValueError("convolution has empty output")
    result = torch.empty((n, cout, *output_shape), dtype=x.dtype, device=x.device)
    for out_begin in range(0, output_shape[0], output_depth_tile):
        out_end = min(out_begin + output_depth_tile, output_shape[0])
        needed_begin = out_begin * stride[0] - padding[0]
        needed_end = (out_end - 1) * stride[0] - padding[0] + dilation[0] * (kd - 1) + 1
        slab = x[:, :, max(0, needed_begin):min(depth, needed_end), :, :]
        left = max(0, -needed_begin)
        right = max(0, needed_end - depth)
        if left or right:
            slab = F.pad(slab, (0, 0, 0, 0, left, right))
        tile = F.conv3d(slab, weight, bias=bias, stride=stride,
                        padding=(0, padding[1], padding[2]),
                        dilation=dilation, groups=groups)
        if tile.shape[2] != out_end - out_begin or tile.shape[3:] != output_shape[1:]:
            raise RuntimeError("tiled convolution produced unexpected output geometry")
        result[:, :, out_begin:out_end] = tile
    return result


def identity_snapshot(network):
    """Capture parameter/module identities and state-dict storage aliases."""
    parameters = {
        name: (id(value), value.untyped_storage().data_ptr(), value.storage_offset(),
               tuple(value.shape), tuple(value.stride()), str(value.dtype), value.device.type)
        for name, value in network.named_parameters(remove_duplicate=False)
    }
    states = {
        name: (value.untyped_storage().data_ptr(), value.storage_offset(),
               tuple(value.shape), tuple(value.stride()), str(value.dtype), value.device.type)
        for name, value in network.state_dict().items()
    }
    modules = {name: id(module) for name, module in network.named_modules(remove_duplicate=False)}
    return {"parameters": parameters, "state_dict": states, "modules": modules}


def _tiled_forward(self, values):
    return conv3d_tiled_depth(values, self.weight, self.bias, self.stride, self.padding,
                              self.dilation, self.groups, output_depth_tile=self._depth_tile)


def attach_tiled_conv3d(network, output_depth_tile=4):
    """Replace only unique Conv3d forward methods; preserve modules and weights."""
    torch, _ = _require_torch()
    if network.training:
        raise ValueError('network must already be in eval mode')
    if not isinstance(output_depth_tile, int) or output_depth_tile <= 0:
        raise ValueError("invalid output-depth tile")
    before = identity_snapshot(network)
    unique = []
    seen = set()
    for module in network.modules():
        if not isinstance(module, torch.nn.Conv3d) or id(module) in seen:
            continue
        seen.add(id(module))
        if type(module) is not torch.nn.Conv3d or "forward" in module.__dict__:
            raise ValueError("custom/previously wrapped Conv3d is unsupported")
        if module.padding_mode != "zeros" or not isinstance(module.padding, tuple) or len(module.padding) != 3:
            raise ValueError("nonzero/implicit Conv3d padding mode is unsupported")
        if module.padding[0] > module.dilation[0] * (module.kernel_size[0] - 1):
            raise ValueError("depth padding exceeds kernel support")
        if (module.weight.device.type != "cpu" or module.weight.dtype != torch.float32 or
                (module.bias is not None and (module.bias.device.type != "cpu" or module.bias.dtype != torch.float32))):
            raise ValueError("Conv3d parameters must be CPU FP32")
        unique.append(module)
    if not unique:
        raise ValueError("network has no Conv3d modules")
    for module in unique:
        module._depth_tile = output_depth_tile
        module.forward = types.MethodType(_tiled_forward, module)
    after = identity_snapshot(network)
    if before != after:
        raise RuntimeError("wrapping changed parameter, state-dict, or module identity")
    if sum(1 for module in network.modules() if isinstance(module, torch.nn.Conv3d)) != len(unique):
        raise RuntimeError("unique Conv3d count changed")
    return {"wrapped_unique_conv3d_modules": len(unique),
            "parameter_name_occurrences": len(before["parameters"]),
            "state_dict_entries": len(before["state_dict"]),
            "named_module_occurrences": len(before["modules"]),
            "parameter_state_module_identities_preserved": True}
