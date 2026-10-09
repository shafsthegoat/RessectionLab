"""Generated-only inference prototype that recomputes the first encoder skip.

The installed PlainConvUNet stores every encoder output until decoder use. For
an exact, stateless Conv/InstanceNorm/LeakyReLU stage, rerunning stage 0 from
the unchanged input can replace its early retained skip with a later result.
This trades compute for a possible memory reduction; no memory benefit or
trained-network parity is claimed by this prototype.
"""

import hashlib
import inspect
from pathlib import Path
import types


_UNET_SHA = "807d9081eda118ced76d8f96e7673975e563dcceb93e33f3c2543ab2dd42de08"
_ENCODER_SHA = "c1af990f1a613c8594234fbc185eb70da2e2b545ea7c933711e91618da2943c4"
_DECODER_SHA = "5eec090b199daf154c8e0ab3d64c45a1b1621f5aac68c6dea6a7203b0ddcc999"
_BLOCKS_SHA = "02a7c21bca3c0acc20923a10970f14e853510f32a1f7ea2ef068cf636559ba2e"
_LAZY_SHA = "ce4d67f816edc13b73f7bfda8bf9af2aa761021473fb4dac47bfac13d7d59f5b"
_TILED_ADAPTER_SHA = "e355495f60ee70e579da72522af93620da2bcf940c65dd748cbb54b8bee79f76"


def _source_is(cls, sha):
    source = inspect.getsourcefile(cls)
    return source is not None and hashlib.sha256(Path(source).read_bytes()).hexdigest() == sha


def _bound(module, expected):
    method = getattr(module, "forward", None)
    return (getattr(method, "__func__", None) is expected and
            getattr(method, "__self__", None) is module)


def _has_hooks(module):
    return bool(module._forward_pre_hooks or module._forward_hooks or
                module._backward_hooks or module._backward_pre_hooks)


class _OneTimeStageZero:
    def __init__(self, remaining, original_input, stage_zero):
        self._remaining = remaining
        self._input = original_input
        self._stage_zero = stage_zero
        self.recompute_calls = 0

    def __len__(self):
        return len(self._remaining)

    def __getitem__(self, index):
        if not isinstance(index, int):
            raise TypeError("decoder requested a noninteger skip")
        resolved = index if index >= 0 else len(self) + index
        if resolved < 0 or resolved >= len(self):
            raise IndexError(index)
        if resolved != 0:
            return self._remaining[resolved]
        if self.recompute_calls:
            raise RuntimeError("stage-zero skip was requested more than once")
        self.recompute_calls += 1
        return self._stage_zero(self._input)


def _forward_recompute_stage_zero(self, values):
    import torch

    if not torch.is_inference_mode_enabled() or self.training:
        raise ValueError("recomputation requires eval and inference_mode")
    if values.device.type != "cpu" or values.dtype != torch.float32:
        raise ValueError("recomputation prototype requires CPU FP32 input")
    stages = self.encoder.stages
    first = stages[0](values)
    if first.untyped_storage().data_ptr() == values.untyped_storage().data_ptr():
        raise RuntimeError("stage zero unexpectedly aliases the input")
    current = stages[1](first)
    if current.untyped_storage().data_ptr() == first.untyped_storage().data_ptr():
        raise RuntimeError("stage one unexpectedly aliases stage zero")
    del first
    skips = [None, current]
    for stage in stages[2:]:
        current = stage(current)
        skips.append(current)
    deferred = _OneTimeStageZero(skips, values, stages[0])
    output = self.decoder(deferred)
    if deferred.recompute_calls != 1:
        raise RuntimeError("decoder did not consume stage-zero skip exactly once")
    return output


def attach_recomputed_stage_zero(network, *, expected_encoder_stages,
                                 require_lazy_decoder=False):
    """Attach only to the pinned pure PlainConvUNet graph before observers.

    The 3-stage option is for tiny generated controls. The pinned trained graph
    would require six encoder stages and its already reviewed lazy decoder.
    All guards precede mutation; weights, modules and state aliases are kept.
    """
    import torch
    from dynamic_network_architectures.architectures.unet import PlainConvUNet
    from dynamic_network_architectures.building_blocks.plain_conv_encoder import PlainConvEncoder
    from dynamic_network_architectures.building_blocks.unet_decoder import UNetDecoder
    from dynamic_network_architectures.building_blocks.simple_conv_blocks import (
        ConvDropoutNormReLU, StackedConvBlocks)
    from resectionlab.low_memory_conv3d import identity_snapshot, _tiled_forward

    if (expected_encoder_stages not in (3, 6) or
            (expected_encoder_stages == 6) != require_lazy_decoder):
        raise ValueError("unexpected architecture/recompute scope")
    if (not _source_is(PlainConvUNet, _UNET_SHA) or
            not _source_is(PlainConvEncoder, _ENCODER_SHA) or
            not _source_is(UNetDecoder, _DECODER_SHA) or
            not _source_is(StackedConvBlocks, _BLOCKS_SHA) or
            not _source_is(identity_snapshot, _TILED_ADAPTER_SHA)):
        raise ValueError("installed source or tracked adapter changed")
    if (type(network) is not PlainConvUNet or "forward" in network.__dict__ or
            not _bound(network, PlainConvUNet.forward) or network.training):
        raise ValueError("network is not the pinned unwrapped eval PlainConvUNet")
    encoder, decoder = network.encoder, network.decoder
    if (type(encoder) is not PlainConvEncoder or type(decoder) is not UNetDecoder or
            "forward" in encoder.__dict__ or not _bound(encoder, PlainConvEncoder.forward) or
            not encoder.return_skips or decoder.deep_supervision or
            len(encoder.stages) != expected_encoder_stages or
            len(decoder.stages) != expected_encoder_stages - 1):
        raise ValueError("encoder/decoder structure changed")
    if require_lazy_decoder:
        from lazy_concat import _decoder_without_final_full_concat
        if (not _source_is(_decoder_without_final_full_concat, _LAZY_SHA) or
                not _bound(decoder, _decoder_without_final_full_concat)):
            raise ValueError("trained graph requires the pinned lazy decoder")
    elif "forward" in decoder.__dict__ or not _bound(decoder, UNetDecoder.forward):
        raise ValueError("tiny graph needs the native decoder")
    if (torch.nn.modules.module._global_forward_pre_hooks or
            torch.nn.modules.module._global_forward_hooks or
            torch.nn.modules.module._global_backward_hooks or
            torch.nn.modules.module._global_backward_pre_hooks or
            any(_has_hooks(module) or module.training for module in network.modules())):
        raise ValueError("hooks or training state make recomputation stateful")
    if (encoder.dropout_op is not None or encoder.norm_op is not torch.nn.InstanceNorm3d or
            encoder.nonlin is not torch.nn.LeakyReLU or
            encoder.norm_op_kwargs != {"eps": 1e-5, "affine": True} or
            encoder.nonlin_kwargs != {"inplace": True} or
            encoder.conv_op is not torch.nn.Conv3d):
        raise ValueError("encoder operator contract is not pinned/pure")
    stage_zero = encoder.stages[0]
    if (type(stage_zero) is not torch.nn.Sequential or len(stage_zero) != 1 or
            type(stage_zero[0]) is not StackedConvBlocks or
            "forward" in stage_zero.__dict__ or
            not _bound(stage_zero, torch.nn.Sequential.forward) or
            "forward" in stage_zero[0].__dict__ or
            not _bound(stage_zero[0], StackedConvBlocks.forward)):
        raise ValueError("stage-zero forward structure changed")
    if (type(stage_zero[0].convs) is not torch.nn.Sequential or
            "forward" in stage_zero[0].convs.__dict__ or
            not _bound(stage_zero[0].convs, torch.nn.Sequential.forward) or
            not stage_zero[0].convs):
        raise ValueError("empty stage zero")
    for block in stage_zero[0].convs:
        if (type(block) is not ConvDropoutNormReLU or
                "forward" in block.__dict__ or
                not _bound(block, ConvDropoutNormReLU.forward) or
                type(block.all_modules) is not torch.nn.Sequential or
                "forward" in block.all_modules.__dict__ or
                not _bound(block.all_modules, torch.nn.Sequential.forward) or
                type(block.conv) is not torch.nn.Conv3d or
                type(block.norm) is not torch.nn.InstanceNorm3d or
                type(block.nonlin) is not torch.nn.LeakyReLU or
                "forward" in block.norm.__dict__ or
                not _bound(block.norm, torch.nn.InstanceNorm3d.forward) or
                "forward" in block.nonlin.__dict__ or
                not _bound(block.nonlin, torch.nn.LeakyReLU.forward) or
                not block.nonlin.inplace or block.norm.track_running_stats or
                not block.norm.affine or block.norm.eps != 1e-5 or
                len(block.all_modules) != 3 or
                list(block.all_modules) != [block.conv, block.norm, block.nonlin]):
            raise ValueError("stage-zero block has stateful/custom operators")
        if ("forward" in block.conv.__dict__ and not _bound(block.conv, _tiled_forward)) or (
                "forward" not in block.conv.__dict__ and
                not _bound(block.conv, torch.nn.Conv3d.forward)):
            raise ValueError("stage-zero Conv3d has an unreviewed forward")
    before = identity_snapshot(network)
    network.forward = types.MethodType(_forward_recompute_stage_zero, network)
    after = identity_snapshot(network)
    if after != before:
        del network.forward
        raise RuntimeError("recompute wrapper changed parameter/state/module identities")
    return {"initial_stage_zero_released_after_stage_one": True,
            "stage_zero_recomputed_once_on_decoder_access": True,
            "state_parameter_module_identities_preserved": True}
