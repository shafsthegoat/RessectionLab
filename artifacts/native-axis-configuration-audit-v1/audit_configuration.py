"""Read-only configuration comparison; intentionally forbids engine construction."""
from dataclasses import asdict, fields
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    sys.path.insert(0, str(args.source.resolve() / "src"))
    import numpy as np
    from resectionlab.core import array_digest
    from resectionlab.geometry import AccessWindow
    from resectionlab.imaging import load_case
    from resectionlab import native_resection, native_simulation
    from resectionlab.worlds import WorldGeneratorConfig, content_hash

    started = time.perf_counter()
    target = json.loads(args.reference.read_text())["target"]
    bundle_hash = hashlib.sha256(args.bundle.read_bytes()).hexdigest()
    assert bundle_hash == target["bundle_sha256"]
    case = load_case(args.bundle)
    assert case.semantic_hash == target["semantic_hash"]
    assert case.planning_hash == target["planning_hash"]
    assert case.frame == target["source_frame"] == "RAS+"
    original_constructor = native_simulation.NativeSequentialSimulator
    constructor_defaults = inspect.signature(original_constructor.__init__).parameters
    captured = {}

    def forbid_engine(*args, **kwargs):
        raise AssertionError("Configuration audit must not instantiate a native engine")

    def capture_config(config, tips, **kwargs):
        captured.update(config=config, tips=tips, kwargs=kwargs)
        return config

    native_resection.NativeResectionEngine.__init__ = forbid_engine
    native_simulation.NativeSequentialSimulator = capture_config
    generator = WorldGeneratorConfig(**target["world_partitions"]["optimization"]["generator"])
    legacy = native_simulation.make_native_patient_simulator(case,
        access=AccessWindow(**target["access"]), world_generator=generator,
        candidate_count=len(target["candidate_tips_mm"]),
        max_steps=target["max_steps"], max_actions=target["max_actions"])
    helper = native_resection.native_config_from_case(case, access=AccessWindow(**target["access"]))

    def normalized(value):
        if isinstance(value, np.ndarray):
            return {"shape": list(value.shape), "dtype": str(value.dtype), "array_hash": array_digest(value),
                    **({"values": value.tolist()} if value.size <= 16 else {})}
        if hasattr(value, "__dataclass_fields__"):
            return {key: normalized(item) for key, item in asdict(value).items()}
        if isinstance(value, tuple):
            return [normalized(item) for item in value]
        return value

    comparison = {}
    for field in fields(legacy):
        if not field.init:
            continue
        before, after = getattr(legacy, field.name), getattr(helper, field.name)
        comparison[field.name] = {
            "equal": content_hash(normalized(before)) == content_hash(normalized(after)),
            "fixed_factory": normalized(before), "native_config_helper": normalized(after)}
        if isinstance(before, np.ndarray):
            comparison[field.name].update(exact_array_equal=bool(np.array_equal(before, after)),
                different_elements=int(np.count_nonzero(before != after)))
    differences = [key for key, row in comparison.items() if not row["equal"]]
    assert legacy.fingerprint == target["native_config_hash"]
    assert array_digest(helper.tissue_mask) == target["tissue_support_hash"]
    assert array_digest(helper.hard_exclusion) == target["hard_exclusion_hash"]
    assert [asdict(tool) for tool in helper.tools] == target["tools"]
    assert asdict(constructor_defaults["reward"].default) == target["reward"]
    assert constructor_defaults["partial_contact_weight"].default == target["partial_contact_weight"]
    assert np.array_equal(captured["tips"], target["candidate_tips_mm"])
    assert np.array_equal(captured["kwargs"]["candidate_entries_mm"], target["candidate_entries_mm"])
    assert all(captured["kwargs"][name] is None for name in ("nominal_motor", "nominal_language"))
    assert differences == ["tissue_support_provenance"], differences
    report = {
        "status": "configuration_only_comparison_passed",
        "source_root": str(args.source.resolve()),
        "source_files": {name: hashlib.sha256((args.source / name).read_bytes()).hexdigest()
            for name in ("src/resectionlab/native_resection.py", "src/resectionlab/native_simulation.py",
                         "src/resectionlab/structural_evidence.py", "scripts/preflight_native_axis.py")},
        "auditor_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "reference_sha256": hashlib.sha256(args.reference.read_bytes()).hexdigest(),
        "bundle_sha256": bundle_hash, "case_hash": case.semantic_hash, "planning_hash": case.planning_hash,
        "fixed_factory_native_fingerprint": legacy.fingerprint,
        "helper_native_fingerprint": helper.fingerprint,
        "configuration_fields": comparison, "different_fields": differences,
        "reference_checks": {
            "fixed_factory_fingerprint": True, "tissue_mask": True, "hard_exclusion": True,
            "tools": True, "fixed_candidate_tips_and_entries": True,
            "default_reward": True, "partial_contact_weight": True, "missing_functional_evidence": True,
            "world_generator": captured["kwargs"]["world_generator"].fingerprint == generator.fingerprint},
        "reward": asdict(constructor_defaults["reward"].default),
        "partial_contact_weight": constructor_defaults["partial_contact_weight"].default,
        "engine_constructions": 0, "axis_constructions": 0, "transitions": 0,
        "gradient_steps": 0, "final_worlds": 0, "stress_worlds": 0,
        "seconds": time.perf_counter() - started,
        "interpretation": "Every native config input field except descriptive tissue provenance is equal. Full fingerprints correctly differ because provenance remains a model input. No hash was weakened or replaced, and no simulation was constructed.",
        "repair_constraint": "A fresh declaration must pin the helper's actual fingerprint/provenance separately from the historical fixed-factory reference; arrays/tools/limits/reward/world assumptions must remain explicitly checked."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: report[key] for key in ("status", "different_fields", "fixed_factory_native_fingerprint", "helper_native_fingerprint", "seconds")}, indent=2))


if __name__ == "__main__":
    main()
