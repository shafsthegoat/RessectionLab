"""One generated-only CPU or MPS nnU-Net forward; supervised externally.

The full model patch is 128^3. This fixed 32x64x64 input only checks software
and CPU/MPS numerical behavior. No patient data or nnU-Net folder initializer.
"""

import gc
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
MODEL = ROOT / "data/models/gliomoda-v1.0.2/t1c-t2f-pinned-v1"
CONTRACT = HERE / "gate-a-contract.json"
INPUT = HERE / "generated-input.npy"
INPUT_RECEIPT = HERE / "generated-input-receipt.json"
START = time.monotonic()


def sha256_file(path):
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def phase(name, torch_module=None, device=None):
    sample = subprocess.run(["ps", "-p", str(os.getpid()), "-o", "rss="],
                            capture_output=True, text=True, check=True)
    record = {"phase": name, "elapsed_seconds": time.monotonic() - START,
              "self_rss_kib": int(sample.stdout.strip())}
    if device == "mps" and torch_module is not None:
        record["mps_current_allocated_bytes"] = int(torch_module.mps.current_allocated_memory())
        record["mps_driver_allocated_bytes"] = int(torch_module.mps.driver_allocated_memory())
    print(json.dumps(record, sort_keys=True), flush=True)


def alias_groups(network):
    groups = defaultdict(list)
    for name, parameter in network.named_parameters(remove_duplicate=False):
        groups[id(parameter)].append(name)
    return groups


def main(device):
    if device not in ("cpu", "mps"):
        raise ValueError("device must be cpu or mps")
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
    if contract["gate"] != "A" or contract["model_patch_size"] != [128, 128, 128]:
        raise ValueError("unexpected gate contract")
    mps_limit = None
    if device == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS unavailable; no CPU fallback")
        recommended = int(torch.mps.recommended_max_memory())
        mps_limit = min(4 * 1024**3, int(0.32 * recommended))
        if recommended <= 0 or mps_limit <= 0:
            raise RuntimeError("invalid Metal recommended memory")
        torch.mps.set_per_process_memory_fraction(mps_limit / recommended)
        phase("after_mps_allocator_cap_before_tensors", torch, device)
    torch.manual_seed(0)
    receipt = json.loads(INPUT_RECEIPT.read_text())
    if receipt["npy_sha256"] != contract["input_npy_sha256"] or sha256_file(INPUT) != contract["input_npy_sha256"]:
        raise ValueError("generated input bytes differ from frozen contract")
    values = np.load(INPUT, allow_pickle=False)
    if values.shape != (1, 2, 32, 64, 64) or values.dtype != np.float32 or not values.flags.c_contiguous:
        raise ValueError("generated input shape/dtype/layout mismatch")
    if hashlib.sha256(values.tobytes(order="C")).hexdigest() != contract["input_raw_sha256"]:
        raise ValueError("generated input payload differs from frozen contract")
    if not np.isfinite(values).all() or not np.any(values != 0):
        raise ValueError("generated input must be finite and nonzero")
    for name, expected in contract["model_file_sha256"].items():
        if sha256_file(MODEL / name) != expected:
            raise ValueError("pinned model file changed: " + name)
    phase("after_input_model_hashes", torch, device)
    dataset = json.loads((MODEL / "dataset.json").read_text())
    plans = json.loads((MODEL / "plans.json").read_text())
    manager = PlansManager(plans)
    config = manager.get_configuration("3d_fullres")
    labels = manager.get_label_manager(dataset)
    channels = determine_num_input_channels(manager, config, dataset)
    heads = labels.num_segmentation_heads
    if channels != 2 or heads != 3 or list(config.patch_size) != [128, 128, 128]:
        raise ValueError("model channel/head/patch contract changed")
    safe_before = list(torch.serialization.get_safe_globals())
    allow = [(np._core.multiarray.scalar, "numpy.core.multiarray.scalar"),
             np.dtype, type(np.dtype("f8")), type(np.dtype("f4"))]
    phase("before_weights_only_load", torch, device)
    with torch.serialization.safe_globals(allow):
        checkpoint = torch.load(MODEL / "fold_all/checkpoint_final.pth",
                                weights_only=True, map_location="cpu")
    phase("after_weights_only_load", torch, device)
    if list(torch.serialization.get_safe_globals()) != safe_before:
        raise ValueError("scoped safe globals not restored")
    if checkpoint["init_args"]["configuration"] != "3d_fullres" or checkpoint["trainer_name"] != "nnUNetTrainer":
        raise ValueError("checkpoint configuration/trainer mismatch")
    weights = checkpoint.pop("network_weights")
    del checkpoint
    gc.collect()
    phase("after_optimizer_cleanup", torch, device)
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
    phase("after_network_construction_alias_checks", torch, device)
    network.load_state_dict(weights, strict=True)
    if not all(torch.equal(network.state_dict()[name], tensor) for name, tensor in weights.items()):
        raise ValueError("strict copied values differ")
    del weights
    gc.collect()
    after_names = sorted(tuple(sorted(names)) for names in alias_groups(network).values())
    before_names = sorted(tuple(sorted(names)) for names in groups_before.values())
    if after_names != before_names:
        raise ValueError("model alias structure changed")
    phase("after_strict_copy_weights_cleanup", torch, device)
    network.eval()
    network = network.to(device)
    generated = torch.from_numpy(values.copy()).to(device)
    if device == "mps":
        torch.mps.synchronize()
    phase("before_generated_forward", torch, device)
    begin = time.perf_counter()
    with torch.inference_mode():
        output = network(generated)
    if device == "mps":
        torch.mps.synchronize()
    forward_seconds = time.perf_counter() - begin
    phase("after_synchronized_forward", torch, device)
    if not isinstance(output, torch.Tensor) or list(output.shape) != [1, 3, 32, 64, 64]:
        raise ValueError("unexpected output type/shape")
    logits = np.ascontiguousarray(output.detach().to("cpu").numpy(), dtype=np.float32)
    phase("after_output_transfer", torch, device)
    if not np.isfinite(logits).all():
        raise ValueError("nonfinite logits")
    outdir = HERE / device
    temp = outdir / "logits.npy.tmp"
    final = outdir / "logits.npy"
    with temp.open("xb") as stream:
        np.save(stream, logits, allow_pickle=False)
    temp.rename(final)
    checkpoint_after = sha256_file(MODEL / "fold_all/checkpoint_final.pth")
    if checkpoint_after != contract["model_file_sha256"]["fold_all/checkpoint_final.pth"]:
        raise ValueError("checkpoint bytes changed")
    result = {
        "scope": "Generated-only 32x64x64 numerical control; not full 128^3 patch or patient inference",
        "device": device, "input_shape": list(values.shape), "output_shape": list(logits.shape),
        "input_npy_sha256": contract["input_npy_sha256"],
        "logits_npy_sha256": sha256_file(final),
        "logits_raw_sha256": hashlib.sha256(logits.tobytes(order="C")).hexdigest(),
        "output_nonfinite": int((~np.isfinite(logits)).sum()),
        "forward_seconds_synchronized": forward_seconds,
        "checkpoint_sha256_after": checkpoint_after,
        "safe_globals_restored": list(torch.serialization.get_safe_globals()) == safe_before,
        "alias_groups": len(duplicate_groups), "alias_compared_pairs": alias_pairs,
        "unique_parameter_elements": sum(p.numel() for p in network.parameters()),
        "mps_allocator_cap_bytes": mps_limit,
        "torch_version": torch.__version__, "numpy_version": np.__version__,
    }
    (outdir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "completed", "device": device,
                      "logits_raw_sha256": result["logits_raw_sha256"],
                      "forward_seconds": forward_seconds}, sort_keys=True), flush=True)


if __name__ == "__main__":
    selected = sys.argv[1] if len(sys.argv) == 2 else None
    try:
        main(selected)
    except Exception as error:
        if selected in ("cpu", "mps"):
            destination = HERE / selected / "failure.json"
            destination.parent.mkdir(exist_ok=True)
            destination.write_text(json.dumps({"error_type": type(error).__name__,
                                               "error": str(error), "no_retry": True},
                                              indent=2, sort_keys=True) + "\n")
        raise
