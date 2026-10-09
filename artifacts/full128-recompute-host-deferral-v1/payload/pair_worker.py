"""One generated full128 nnU-Net stage-zero recomputation feasibility control.

No patient data, dense hooks, or nnU-Net folder initializer.
"""

import gc
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
from darwin_fast_sampler import FastDarwinSampler

MODEL = ROOT / "data/models/gliomoda-v1.0.2/t1c-t2f-pinned-v1"
CONTRACT = HERE / "pair-contract.json"
INPUT = HERE / "generated-input.npy"
INPUT_RECEIPT = HERE / "generated-input-receipt.json"
START = time.monotonic()
_PHASE_SAMPLER = None


def sha256_file(path):
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def phase(name):
    global _PHASE_SAMPLER
    if _PHASE_SAMPLER is None:
        _PHASE_SAMPLER = FastDarwinSampler()
    resident_bytes = _PHASE_SAMPLER.process_resident_bytes(os.getpid())
    record = {"phase": name, "elapsed_seconds": time.monotonic() - START,
              "self_rss_kib": resident_bytes // 1024,
              "phase_rss_source": "direct_libproc_self_pid_no_subprocess"}
    print(json.dumps(record, sort_keys=True), flush=True)


def alias_groups(network):
    groups = defaultdict(list)
    for name, parameter in network.named_parameters(remove_duplicate=False):
        groups[id(parameter)].append(name)
    return groups



def main():
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0":
        raise EnvironmentError("MPS CPU fallback must be disabled before torch import")
    phase("before_imports")
    import numpy as np
    import torch
    from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
    from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

    if np.__version__ != "2.0.1" or torch.__version__ != "2.14.1":
        raise RuntimeError("unpinned NumPy or PyTorch version")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    contract = json.loads(CONTRACT.read_text())
    arm = os.environ.get("RESECTIONLAB_PAIR_ARM")
    if (contract["gate"] != "CPU_FULL128_RECOMPUTE_FEASIBILITY_V1" or
            contract["model_patch_size"] != [128, 128, 128] or
            arm != "recompute" or contract["arm_order"] != ["recompute"]):
        raise ValueError("unexpected gate contract")
    if sha256_file(Path(__file__).resolve()) != contract["runner_source_sha256"]["pair_worker.py"]:
        raise ValueError("worker source changed after contract freeze")
    imported_sampler = Path(sys.modules[FastDarwinSampler.__module__].__file__).resolve()
    if (imported_sampler != (HERE / "darwin_fast_sampler.py").resolve() or
            sha256_file(imported_sampler) != contract["runner_source_sha256"]["darwin_fast_sampler.py"]):
        raise ValueError("early direct-RSS sampler import differs from frozen local source")
    contract_sha256 = sha256_file(CONTRACT)
    torch.manual_seed(0)
    receipt = json.loads(INPUT_RECEIPT.read_text())
    if receipt["npy_sha256"] != contract["input_npy_sha256"] or sha256_file(INPUT) != contract["input_npy_sha256"]:
        raise ValueError("generated input bytes differ from frozen contract")
    values = np.load(INPUT, allow_pickle=False)
    if values.shape != (1, 2, 128, 128, 128) or values.dtype != np.float32 or not values.flags.c_contiguous:
        raise ValueError("generated input shape/dtype/layout mismatch")
    if hashlib.sha256(values.tobytes(order="C")).hexdigest() != contract["input_raw_sha256"]:
        raise ValueError("generated input payload differs from frozen contract")
    if not np.isfinite(values).all() or not np.any(values != 0):
        raise ValueError("generated input must be finite and nonzero")
    adapter_path = ROOT / "src/resectionlab/low_memory_conv3d.py"
    if sha256_file(adapter_path) != contract["tracked_adapter_sha256"]:
        raise ValueError("promoted low-memory adapter changed")
    for name, expected in contract["model_file_sha256"].items():
        if sha256_file(MODEL / name) != expected:
            raise ValueError("pinned model file changed: " + name)
    phase("after_input_model_hashes")
    dataset = json.loads((MODEL / "dataset.json").read_text())
    plans = json.loads((MODEL / "plans.json").read_text())
    manager = PlansManager(plans)
    config = manager.get_configuration("3d_fullres")
    labels = manager.get_label_manager(dataset)
    channels = determine_num_input_channels(manager, config, dataset)
    heads = labels.num_segmentation_heads
    if channels != 2 or heads != 3 or list(config.patch_size) != [128, 128, 128]:
        raise ValueError("model channel/head/patch contract changed")
    architecture = plans["configurations"]["3d_fullres"]["architecture"]["arch_kwargs"]
    if (architecture["dropout_op"] is not None or architecture["dropout_op_kwargs"] is not None or
            architecture["norm_op"] != "torch.nn.modules.instancenorm.InstanceNorm3d" or
            architecture["conv_op"] != "torch.nn.modules.conv.Conv3d"):
        raise ValueError("dropout/norm/convolution assumptions changed")
    safe_before = list(torch.serialization.get_safe_globals())
    allow = [(np._core.multiarray.scalar, "numpy.core.multiarray.scalar"),
             np.dtype, type(np.dtype("f8")), type(np.dtype("f4"))]
    phase("before_weights_only_load")
    with torch.serialization.safe_globals(allow):
        checkpoint = torch.load(MODEL / "fold_all/checkpoint_final.pth",
                                weights_only=True, map_location="cpu")
    phase("after_weights_only_load")
    if list(torch.serialization.get_safe_globals()) != safe_before:
        raise ValueError("scoped safe globals not restored")
    if checkpoint["init_args"]["configuration"] != "3d_fullres" or checkpoint["trainer_name"] != "nnUNetTrainer":
        raise ValueError("checkpoint configuration/trainer mismatch")
    weights = checkpoint.pop("network_weights")
    del checkpoint
    gc.collect()
    phase("after_optimizer_cleanup")
    network = get_network_from_plans(
        config.network_arch_class_name, config.network_arch_init_kwargs,
        config.network_arch_init_kwargs_req_import, channels, heads,
        allow_init=False, deep_supervision=False)
    shapes = {name: list(value.shape) for name, value in network.state_dict().items()}
    if set(shapes) != set(weights) or any(shapes[name] != list(weights[name].shape) for name in shapes):
        raise ValueError("checkpoint/model key or shape mismatch")
    groups_before = alias_groups(network)
    duplicate_groups = [names for names in groups_before.values() if len(names) > 1]
    alias_pairs = 0
    for names in duplicate_groups:
        reference = weights[names[0]]
        for name in names[1:]:
            other = weights[name]
            if reference.dtype != other.dtype or reference.device.type != "cpu" or other.device.type != "cpu":
                raise ValueError("duplicate alias dtype/device mismatch")
            same_storage = (reference.untyped_storage().data_ptr() == other.untyped_storage().data_ptr()
                            and reference.storage_offset() == other.storage_offset()
                            and tuple(reference.stride()) == tuple(other.stride())
                            and tuple(reference.shape) == tuple(other.shape))
            if not same_storage and not torch.equal(reference, other):
                raise ValueError("duplicate alias values disagree")
            alias_pairs += 1
    phase("after_network_construction_alias_checks")
    network.load_state_dict(weights, strict=True)
    if not all(torch.equal(network.state_dict()[name], tensor) for name, tensor in weights.items()):
        raise ValueError("strict copied values differ")
    del weights
    gc.collect()
    after_names = sorted(tuple(sorted(names)) for names in alias_groups(network).values())
    before_names = sorted(tuple(sorted(names)) for names in groups_before.values())
    if after_names != before_names:
        raise ValueError("model alias structure changed")
    phase("after_strict_copy_weights_cleanup")
    network.eval()
    network = network.to("cpu")
    sys.path.insert(0, str(ROOT / "src"))  # tracked adapter bytes were hash-checked above
    from resectionlab.low_memory_conv3d import attach_tiled_conv3d
    wrapper_report = attach_tiled_conv3d(network, contract["output_depth_tile"])
    if wrapper_report["wrapped_unique_conv3d_modules"] != 27 or not wrapper_report["parameter_state_module_identities_preserved"]:
        raise ValueError("tiled Conv3d module/alias contract changed")
    phase("after_tiled_conv3d_attachment")
    lazy_report = None
    lazy_path = HERE / "lazy_concat.py"
    if sha256_file(lazy_path) != contract["runner_source_sha256"]["lazy_concat.py"]:
        raise ValueError("lazy-concat source changed after contract freeze")
    sys.path.insert(0, str(HERE))
    from lazy_concat import attach_lazy_final_decoder_concat
    lazy_report = attach_lazy_final_decoder_concat(
        network, output_depth_tile=contract["output_depth_tile"], expected_stages=5)
    if (not lazy_report["parameter_state_module_identities_preserved"] or
            lazy_report["final_decoder_stages"] != 5):
        raise ValueError("lazy final decoder attachment changed aliases or stage count")
    phase("after_lazy_final_decoder_attachment")
    recompute_path = HERE / "recompute_skip.py"
    if sha256_file(recompute_path) != contract["runner_source_sha256"]["recompute_skip.py"]:
        raise ValueError("recompute source changed after contract freeze")
    sys.path.insert(0, str(HERE))
    from recompute_skip import attach_recomputed_stage_zero
    recompute_report = attach_recomputed_stage_zero(
        network, expected_encoder_stages=6, require_lazy_decoder=True)
    if not recompute_report["state_parameter_module_identities_preserved"]:
        raise ValueError("recompute attachment changed aliases")
    phase("after_optional_recompute_attachment_no_module_hooks")
    generated = torch.from_numpy(values.copy())
    phase("before_generated_forward")
    begin = time.perf_counter()
    with torch.inference_mode():
        output = network(generated)
    forward_seconds = time.perf_counter() - begin
    phase("after_generated_forward")
    if not isinstance(output, torch.Tensor) or list(output.shape) != [1, 3, 128, 128, 128] or output.dtype != torch.float32:
        raise ValueError("unexpected output type/shape")
    logits = np.ascontiguousarray(output.detach().to("cpu").numpy(), dtype=np.float32)
    phase("after_output_transfer")
    if not np.isfinite(logits).all():
        raise ValueError("nonfinite logits")
    outdir = HERE / arm
    temp = outdir / "logits.npy.tmp"
    final = outdir / "logits.npy"
    with temp.open("xb") as stream:
        np.save(stream, logits, allow_pickle=False)
    temp.rename(final)
    checkpoint_after = sha256_file(MODEL / "fold_all/checkpoint_final.pth")
    if checkpoint_after != contract["model_file_sha256"]["fold_all/checkpoint_final.pth"]:
        raise ValueError("checkpoint bytes changed")
    result = {
        "scope": "Generated-only full128 tiled CPU stage-zero recomputation feasibility; no native128 comparator, patient or clinical claim",
        "arm": arm, "input_shape": list(values.shape), "output_shape": list(logits.shape),
        "input_npy_sha256": contract["input_npy_sha256"],
        "contract_sha256": contract_sha256,
        "logits_npy_sha256": sha256_file(final),
        "logits_raw_sha256": hashlib.sha256(logits.tobytes(order="C")).hexdigest(),
        "output_nonfinite": int((~np.isfinite(logits)).sum()),
        "forward_seconds": forward_seconds,
        "checkpoint_sha256_after": checkpoint_after,
        "safe_globals_restored": list(torch.serialization.get_safe_globals()) == safe_before,
        "tiled_adapter_sha256": sha256_file(adapter_path),
        "lazy_concat_source_sha256": sha256_file(lazy_path),
        "dense_module_hooks_attached": False,
        "stage_zero_recomputation_enabled": True,
        "recompute_source_sha256": sha256_file(recompute_path),
        "recompute_attachment_report": recompute_report,
        "full128_numerical_parity_reference_available": False,
        "phase_rss_source": "direct_libproc_self_pid_no_subprocess",
        "single_generated_volume_no_sliding_window_tta_autocast": True,
        "dropout_configured": False, "full_feature_map_instancenorm_unchanged": True,
        "alias_groups": len(duplicate_groups), "alias_compared_pairs": alias_pairs,
        "unique_parameter_elements": sum(p.numel() for p in network.parameters()),
        "wrapper_report": wrapper_report,
        "lazy_attachment_report": lazy_report,
        "torch_version": torch.__version__, "numpy_version": np.__version__,
    }
    (outdir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "completed", "arm": arm,
                      "logits_raw_sha256": result["logits_raw_sha256"],
                      "forward_seconds": forward_seconds}, sort_keys=True), flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 1:
        raise SystemExit("arm is supervisor-bound by environment, not CLI")
    try:
        main()
    except Exception as error:
        destination = HERE / os.environ.get("RESECTIONLAB_PAIR_ARM", "unknown") / "failure.json"
        destination.parent.mkdir(exist_ok=True)
        destination.write_text(json.dumps({"error_type": type(error).__name__,
                                           "error": str(error), "no_retry": True},
                                          indent=2, sort_keys=True) + "\n")
        raise
