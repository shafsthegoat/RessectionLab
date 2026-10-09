"""Generated-only CPU FP32 native InstanceNorm on channel chunks.

This is an operator experiment, not an installed model adapter. It calls a
reviewed Conv3d internally so the output tensor is private until returned.
Spatial statistics are always computed over complete feature maps by PyTorch's
native InstanceNorm implementation. No patient or pretrained data is used.
"""

import torch
import torch.nn.functional as F


def _has_hooks(module):
    return bool(module._forward_pre_hooks or module._forward_hooks or
                module._backward_hooks or module._backward_pre_hooks)


def native_norm_after_private_conv_(conv, norm, values, *, channel_chunk):
    """Run exact native per-channel normalization, copy each chunk into fresh Conv output.

    The caller never supplies a supposedly owned Conv output: this function
    creates it internally through the exact native Conv3d method, with hooks
    and custom forward paths rejected. This bounds the output copyback to one
    channel chunk at a time. Only generated small-tensor evidence exists.
    """
    if not torch.is_inference_mode_enabled():
        raise ValueError("inference_mode is required")
    if type(conv) is not torch.nn.Conv3d or type(norm) is not torch.nn.InstanceNorm3d:
        raise ValueError("exact Conv3d and InstanceNorm3d modules required")
    if ("forward" in conv.__dict__ or
            getattr(conv.forward, "__func__", None) is not torch.nn.Conv3d.forward or
            "forward" in norm.__dict__ or
            getattr(norm.forward, "__func__", None) is not torch.nn.InstanceNorm3d.forward):
        raise ValueError("custom module forward is not an ownership proof")
    global_hooks = torch.nn.modules.module
    if (_has_hooks(conv) or _has_hooks(norm) or
            global_hooks._global_forward_pre_hooks or global_hooks._global_forward_hooks or
            global_hooks._global_backward_pre_hooks or global_hooks._global_backward_hooks):
        raise ValueError("hooks can retain a Conv output alias")
    if conv.training or norm.training or values.requires_grad:
        raise ValueError("eval-only without autograd")
    if (not norm.affine or norm.track_running_stats or
            norm.running_mean is not None or norm.running_var is not None or
            norm.weight is None or norm.bias is None or
            norm.num_features != conv.out_channels or
            not isinstance(norm.eps, float) or norm.eps <= 0):
        raise ValueError("unsupported InstanceNorm configuration")
    if (not isinstance(values, torch.Tensor) or values.device.type != "cpu" or
            values.dtype != torch.float32 or values.ndim != 5 or
            not values.is_contiguous(memory_format=torch.contiguous_format) or
            values._base is not None or values.storage_offset() != 0 or
            values.shape[1] != conv.in_channels or
            values.shape[0] < 1 or min(values.shape[2:]) < 2):
        raise ValueError("unsupported input tensor layout or shape")
    if (not isinstance(channel_chunk, int) or isinstance(channel_chunk, bool) or
            channel_chunk < 1 or channel_chunk > conv.out_channels):
        raise ValueError("invalid channel chunk")
    if (conv.weight.dtype != torch.float32 or conv.weight.device.type != "cpu" or
            norm.weight.dtype != torch.float32 or norm.bias.dtype != torch.float32 or
            norm.weight.device.type != "cpu" or norm.bias.device.type != "cpu"):
        raise ValueError("CPU FP32 parameters required")

    output = conv(values)
    if (output.device.type != "cpu" or output.dtype != torch.float32 or
            not output.is_contiguous() or output._base is not None or
            output.storage_offset() != 0 or
            output.untyped_storage().data_ptr() == values.untyped_storage().data_ptr()):
        raise RuntimeError("Conv output is not a fresh private tensor")
    for lo in range(0, output.shape[1], channel_chunk):
        hi = min(output.shape[1], lo + channel_chunk)
        normalized = F.instance_norm(
            output[:, lo:hi], None, None, norm.weight[lo:hi], norm.bias[lo:hi],
            True, norm.momentum, norm.eps)
        if normalized.shape != output[:, lo:hi].shape or not torch.isfinite(normalized).all():
            raise RuntimeError("native normalization returned invalid chunk")
        output[:, lo:hi].copy_(normalized)
        del normalized
    return output
