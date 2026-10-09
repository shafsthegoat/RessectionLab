"""Small generated PlainConvUNet controls; no checkpoint or patient input."""

import unittest
import weakref

import torch

from dynamic_network_architectures.architectures.unet import PlainConvUNet
from resectionlab.low_memory_conv3d import identity_snapshot
from recompute_skip import attach_recomputed_stage_zero


def tiny_network():
    torch.manual_seed(222)
    return PlainConvUNet(
        input_channels=2, n_stages=3, features_per_stage=[4, 8, 16],
        conv_op=torch.nn.Conv3d, kernel_sizes=3, strides=[1, 2, 2],
        n_conv_per_stage=[2, 2, 2], num_classes=3,
        n_conv_per_stage_decoder=[2, 2], conv_bias=True,
        norm_op=torch.nn.InstanceNorm3d,
        norm_op_kwargs={"eps": 1e-5, "affine": True}, dropout_op=None,
        nonlin=torch.nn.LeakyReLU, nonlin_kwargs={"inplace": True},
        deep_supervision=False).eval()


class GeneratedRecomputeTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_exact_generated_output_and_stage_ownership(self):
        network = tiny_network()
        torch.manual_seed(333)
        values = torch.randn((1, 2, 16, 16, 32))
        original_input = values.clone()
        initial_snapshot = identity_snapshot(network)
        original_state = {name: value.clone() for name, value in network.state_dict().items()}
        with torch.inference_mode():
            baseline = network(values)
        report = attach_recomputed_stage_zero(network, expected_encoder_stages=3)
        self.assertTrue(report["state_parameter_module_identities_preserved"])
        self.assertEqual(identity_snapshot(network), initial_snapshot)

        stage_zero_calls = []
        stage_one_calls = []
        initial_weak = []
        released_before_decoder = []

        def after_stage_zero(_module, _input, output):
            stage_zero_calls.append(1)
            if len(stage_zero_calls) == 1:
                initial_weak.append(weakref.ref(output))

        def after_stage_one(_module, _input, _output):
            stage_one_calls.append(1)

        def before_decoder(_module, _input):
            released_before_decoder.append(initial_weak[0]() is None)

        handles = [network.encoder.stages[0].register_forward_hook(after_stage_zero),
                   network.encoder.stages[1].register_forward_hook(after_stage_one),
                   network.decoder.register_forward_pre_hook(before_decoder)]
        try:
            with torch.inference_mode():
                candidate = network(values)
        finally:
            for handle in handles:
                handle.remove()
        self.assertEqual(len(stage_zero_calls), 2)
        self.assertEqual(len(stage_one_calls), 1)
        self.assertEqual(released_before_decoder, [True])
        self.assertTrue(torch.equal(values, original_input))
        self.assertEqual(identity_snapshot(network), initial_snapshot)
        for name, tensor in network.state_dict().items():
            self.assertTrue(torch.equal(tensor, original_state[name]))
        self.assertTrue(torch.equal(candidate, baseline))
        print({"generated_shape": list(candidate.shape), "stage_zero_calls": 2,
               "stage_one_calls": 1, "initial_stage_zero_object_gone_before_decoder": True,
               "exact_logits": True})

    def test_stateful_and_hooked_modules_rejected_before_mutation(self):
        for alteration in ("training", "track_running_stats", "hook", "custom_stage",
                           "custom_norm", "insert_dropout"):
            network = tiny_network()
            if alteration == "training":
                network.train()
            elif alteration == "track_running_stats":
                network.encoder.stages[0][0].convs[0].norm.track_running_stats = True
            elif alteration == "hook":
                network.encoder.stages[0].register_forward_hook(lambda *_: None)
            elif alteration == "custom_stage":
                network.encoder.stages[0].forward = lambda x: x
            elif alteration == "custom_norm":
                network.encoder.stages[0][0].convs[0].norm.forward = lambda x: x
            elif alteration == "insert_dropout":
                block = network.encoder.stages[0][0].convs[0]
                block.all_modules = torch.nn.Sequential(
                    block.conv, torch.nn.Dropout3d(0.5), block.norm, block.nonlin)
            with self.assertRaises(ValueError):
                attach_recomputed_stage_zero(network, expected_encoder_stages=3)
            self.assertNotIn("forward", network.__dict__)

    def test_global_hook_and_wrong_scope_rejected(self):
        network = tiny_network()
        with self.assertRaises(ValueError):
            attach_recomputed_stage_zero(network, expected_encoder_stages=3,
                                         require_lazy_decoder=True)
        handle = torch.nn.modules.module.register_module_forward_hook(lambda *_: None)
        try:
            with self.assertRaisesRegex(ValueError, "hooks"):
                attach_recomputed_stage_zero(network, expected_encoder_stages=3)
        finally:
            handle.remove()
        self.assertNotIn("forward", network.__dict__)

    def test_inference_mode_and_exception_propagation(self):
        network = tiny_network()
        attach_recomputed_stage_zero(network, expected_encoder_stages=3)
        values = torch.randn((1, 2, 16, 16, 32))
        with self.assertRaisesRegex(ValueError, "inference_mode"):
            network(values)
        calls = []

        def fail_on_recompute(_module, _inputs, _output):
            calls.append(1)
            if len(calls) == 2:
                raise RuntimeError("generated second-pass failure")

        handle = network.encoder.stages[0].register_forward_hook(fail_on_recompute)
        try:
            with torch.inference_mode():
                with self.assertRaisesRegex(RuntimeError, "second-pass failure"):
                    network(values)
        finally:
            handle.remove()
        self.assertEqual(len(calls), 2)


if __name__ == "__main__":
    unittest.main()
