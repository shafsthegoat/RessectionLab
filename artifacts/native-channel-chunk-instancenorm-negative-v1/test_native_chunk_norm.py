"""Tiny generated native InstanceNorm parity and ownership controls."""

import unittest

import torch

from native_chunk_norm import native_norm_after_private_conv_


class NativeChunkNormTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def _modules(self, channels=4, *, identity=False):
        if identity:
            conv = torch.nn.Conv3d(channels, channels, 1, groups=channels, bias=False)
            with torch.no_grad():
                conv.weight.fill_(1)
        else:
            conv = torch.nn.Conv3d(channels, channels, 3, padding=1, bias=False)
        norm = torch.nn.InstanceNorm3d(channels, eps=1e-5, affine=True,
                                       track_running_stats=False)
        with torch.no_grad():
            norm.weight.copy_(torch.linspace(-1.2, 1.4, channels))
            norm.bias.copy_(torch.linspace(-0.06, 0.04, channels))
        return conv.eval(), norm.eval()

    def _compare(self, values, conv, norm, chunks):
        with torch.inference_mode():
            reference = norm(conv(values))
            for chunk in chunks:
                candidate = native_norm_after_private_conv_(
                    conv, norm, values, channel_chunk=chunk)
                self.assertEqual(candidate.shape, reference.shape)
                self.assertTrue(torch.isfinite(candidate).all())
                self.assertTrue(torch.equal(candidate, reference),
                                (chunk, float((candidate-reference).abs().max())))
                self.assertEqual(int(((candidate > 0) != (reference > 0)).sum()), 0)

    def test_random_affine_batch_and_chunk_boundaries(self):
        torch.manual_seed(17)
        values = torch.randn((2, 4, 5, 6, 7), dtype=torch.float32)
        self._compare(values, *self._modules(), (1, 2, 3, 4))

    def test_previous_large_offset_and_nearconstant_counterexamples(self):
        conv, norm = self._modules(identity=True)
        torch.manual_seed(1)
        large_offset = torch.randn((1, 4, 4, 5, 6)) * 0.1 + 1e6
        self._compare(large_offset, conv, norm, (1, 2, 3))
        nearconstant = torch.randn((1, 4, 4, 5, 6)) * 1e-4 + 3.0
        self._compare(nearconstant, conv, norm, (1, 2, 3))

    def test_retained_input_skip_is_unchanged_and_conv_output_is_new(self):
        torch.manual_seed(8)
        values = torch.randn((1, 4, 4, 5, 6))
        retained_skip = values.detach()
        original = values.clone()
        conv, norm = self._modules()
        with torch.inference_mode():
            result = native_norm_after_private_conv_(conv, norm, values, channel_chunk=1)
        self.assertTrue(torch.equal(values, original))
        self.assertTrue(torch.equal(retained_skip, original))
        self.assertNotEqual(result.untyped_storage().data_ptr(),
                            retained_skip.untyped_storage().data_ptr())

    def test_strided_channels_last_dtype_and_view_inputs_fail_closed(self):
        conv, norm = self._modules()
        values = torch.randn((1, 4, 4, 6, 8))
        with torch.inference_mode():
            for unsupported in (values[..., ::2],
                                values.contiguous(memory_format=torch.channels_last_3d),
                                values.double(), values.view_as(values)):
                with self.assertRaises(ValueError):
                    native_norm_after_private_conv_(conv, norm, unsupported,
                                                    channel_chunk=1)
            with self.assertRaises(ValueError):
                native_norm_after_private_conv_(conv, norm, values, channel_chunk=0)
            with self.assertRaises(ValueError):
                native_norm_after_private_conv_(conv, norm, values, channel_chunk=True)

    def test_hooks_custom_modules_and_training_fail_before_conv(self):
        conv, norm = self._modules()
        values = torch.randn((1, 4, 4, 5, 6))
        calls = []
        hook = conv.register_forward_hook(lambda _module, _args, output: calls.append(output))
        try:
            with torch.inference_mode(), self.assertRaisesRegex(ValueError, "hooks"):
                native_norm_after_private_conv_(conv, norm, values, channel_chunk=1)
            self.assertEqual(calls, [])
        finally:
            hook.remove()
        with torch.inference_mode():
            conv.train()
            with self.assertRaisesRegex(ValueError, "eval-only"):
                native_norm_after_private_conv_(conv, norm, values, channel_chunk=1)
            conv.eval()
            conv.forward = lambda x: x  # An aliased fake Conv output must not be accepted.
            with self.assertRaisesRegex(ValueError, "custom"):
                native_norm_after_private_conv_(conv, norm, values, channel_chunk=1)

    def test_autograd_and_unsupported_norm_configuration_fail(self):
        conv, norm = self._modules()
        values = torch.randn((1, 4, 4, 5, 6))
        with self.assertRaisesRegex(ValueError, "inference_mode"):
            native_norm_after_private_conv_(conv, norm, values, channel_chunk=1)
        with torch.inference_mode():
            norm.track_running_stats = True
            with self.assertRaisesRegex(ValueError, "configuration"):
                native_norm_after_private_conv_(conv, norm, values, channel_chunk=1)


if __name__ == "__main__":
    unittest.main()
