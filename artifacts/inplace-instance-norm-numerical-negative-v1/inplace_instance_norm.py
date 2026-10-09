"""Generated-only CPU FP32 InstanceNorm3d allocation experiment.

This operation mutates its input. A later model adapter must establish that the
Conv3d output is exclusively owned, and independently validate network parity.
It is not a drop-in replacement for arbitrary InstanceNorm3d call sites.
"""

import torch


def normalize_private_conv_output_(values, weight, bias, *, eps=1e-5):
    """Normalize one full NCDHW map in place using per-case/channel statistics.

    ``values`` must be a newly produced, unaliased Conv3d output. This function
    can reject obvious views but cannot discover all external tensor aliases;
    callers must prove ownership from their exact graph and hook contract.
    """
    if not torch.is_inference_mode_enabled():
        raise ValueError("inference_mode is required")
    if not isinstance(values, torch.Tensor) or values.ndim != 5:
        raise ValueError("expected NCDHW tensor")
    if values.device.type != "cpu" or values.dtype != torch.float32:
        raise ValueError("CPU FP32 only")
    if not values.is_contiguous() or values._base is not None or values.storage_offset() != 0:
        raise ValueError("view or noncontiguous tensor is not accepted")
    if values.requires_grad or values.shape[2] * values.shape[3] * values.shape[4] < 2:
        raise ValueError("autograd or degenerate spatial map is not accepted")
    channels = values.shape[1]
    for name, parameter in (("weight", weight), ("bias", bias)):
        if (not isinstance(parameter, torch.Tensor) or parameter.shape != (channels,) or
                parameter.dtype != torch.float32 or parameter.device.type != "cpu"):
            raise ValueError(f"invalid {name}")
    if not isinstance(eps, float) or eps <= 0.0:
        raise ValueError("invalid epsilon")
    if not torch.isfinite(values).all() or not torch.isfinite(weight).all() or not torch.isfinite(bias).all():
        raise ValueError("nonfinite input or affine parameter")

    # The reduction spans the entire spatial map, never a depth slab. PyTorch
    # InstanceNorm3d uses population (biased) variance when input stats apply.
    variance, mean = torch.var_mean(values, dim=(2, 3, 4), correction=0, keepdim=True)
    invstd = torch.rsqrt(variance + eps)
    values.sub_(mean)
    values.mul_(invstd)
    values.mul_(weight.reshape(1, channels, 1, 1, 1))
    values.add_(bias.reshape(1, channels, 1, 1, 1))
    if not torch.isfinite(values).all():
        raise ValueError("normalization produced nonfinite output")
    return values
