"""Independent v3 kernel and sealed generated native-preflight review.

Only pure injected-backend controls and existing tiny generated arrays; never
imports ANTs or invokes native code, a worker, or a patient image decoder.
"""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import resource
import sys

import numpy as np

from first_ants_mask_dtype_qa import independently_generate, frame_check, record
from first_ixi_positive_domain_qa import tiny_arrays, TransportImage, TransportOnlyBackend, UNKNOWN, NoModels

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT/"artifacts/poc-ants-typed-mask-native-preflight-v3"
BASE = ROOT/"outputs/poc-ants-typed-mask-native-preflight-v3"
KERNEL_SHA = "211d715a05b2dd269358b7bc5585488a053859ba65c04b9f31f60b69e41ec954"
PINS = {
    "scripts/poc_ixi265_atropos_typed_mask.py": KERNEL_SHA,
    "scripts/poc_ants_typed_mask_preflight.py": "db90c0a4f40a0c6be20438b8d1766c35a16d7ed7636852018b6babb6a852a107",
    "artifacts/poc-ants-typed-mask-native-preflight-v3/outcome.json": "b9b8841d81d1ac9f10303aecdbfa838bba59fe4764065cd7898f44026b129c78",
    "artifacts/ants-mask-dtype-independent-v1/independent-qa.json": "9c4b3bc648b9920ba7de4c5fd0c921b08382796e2ead3467caffc4213ee0c128",
    "artifacts/ixi265-positive-domain-independent-v2/result-qa.json": "9c90b165a5da5657394ee1d26f0aa35193bc072e627b3b198d5fea1018beae72",
}


def sha(path):
    with Path(path).open("rb") as stream: return hashlib.file_digest(stream, "sha256").hexdigest()


def stamp(path):
    p = Path(path); return {"sha256": sha(p), "bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}


def load_kernel():
    spec = importlib.util.spec_from_file_location("independent_typed_kernel", ROOT/"scripts/poc_ixi265_atropos_typed_mask.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


class TypedTransport(TransportOnlyBackend):
    def from_numpy(self, data, **geometry):
        image = TransportImage(data, **geometry)
        image.pixeltype = "float" if data.dtype == np.float32 else "unsigned char"
        return image


def kernel_controls(kernel):
    values, brain, affine = tiny_arrays()
    previous = kernel.previous()
    domain, unknown, _ = previous.positive_domain(values, brain, expected_unknown=UNKNOWN)
    arrays = (values, brain, domain, unknown, affine)
    before = [a.copy() for a in arrays]
    for a in arrays: a.flags.writeable = False
    backend = TypedTransport(); stages = []
    corrected, labels, probabilities, audit = kernel.fit_arrays(values, domain, affine, backend,
        stage_callback=lambda name, detail: stages.append((name, detail)))
    previous.require_unknown_partition(brain, domain, unknown, labels, probabilities)
    assert [s[0] for s in stages] == ["N4_start", "N4_complete", "Atropos_start", "Atropos_complete"]
    assert all(np.array_equal(a, b) for a, b in zip(arrays, before))
    assert np.array_equal(corrected, values)
    assert all(np.array_equal(a, b) for a, b in zip(probabilities, backend.raw_probabilities))
    assert audit["N4_mask_boundary"]["native_pixeltype"] == "float"
    assert audit["N4_mask_boundary"]["support_voxels"] == int(domain.sum())
    checks = ["readonly_intensity_brain_domain_unknown_affine_preserved", "float_mask_exact_support", "raw_posteriors_and_eight_unknowns_preserved", "ordered_stage_callbacks_after_checks"]
    for fault in ("native_type", "float64", "value", "origin", "shape", "input_intensity"):
        class Fault(TypedTransport):
            def from_numpy(self, data, **geometry):
                image = super().from_numpy(data, **geometry)
                mask = np.isin(data, [0, 1]).all()
                if mask:
                    if fault == "native_type": image.pixeltype = "unsigned char"
                    elif fault == "float64": image.data = image.data.astype(np.float64)
                    elif fault == "value": image.data[4, 2, 2] = 0
                    elif fault == "origin": image.origin += [.01, 0, 0]
                    elif fault == "shape": image.shape = (5, 6, 7)
                elif fault == "input_intensity": image.data[4, 2, 2] += 1
                return image
        bad = Fault(); events = []
        try:
            kernel.fit_arrays(values, domain, affine, bad, stage_callback=lambda name, _: events.append(name))
        except ValueError: pass
        else: raise AssertionError("Bad boundary was accepted: "+fault)
        assert bad.calls == [] and events == []
        checks.append("refuse_before_native_call_"+fault)
    class MutatesMask(TypedTransport):
        def n4_bias_field_correction(self, image, *, mask, **params):
            result = super().n4_bias_field_correction(image, mask=mask, **params)
            mask.data[4, 2, 2] = 0
            return result
    bad = MutatesMask(); events = []
    try: kernel.fit_arrays(values, domain, affine, bad, stage_callback=lambda name, _: events.append(name))
    except ValueError: pass
    else: raise AssertionError("Mask mutation passed onward to Atropos")
    assert [c[0] for c in bad.calls] == ["N4"] and events == ["N4_start", "N4_complete"]
    checks.append("recheck_same_mask_support_before_Atropos")
    class BadPosterior(TypedTransport):
        def atropos(self, **kwargs):
            result = super().atropos(**kwargs); result["probabilityimages"][0].data[4, 2, 2] = 1.1; return result
    events = []
    try: kernel.fit_arrays(values, domain, affine, BadPosterior(), stage_callback=lambda name, _: events.append(name))
    except ValueError: pass
    else: raise AssertionError("Malformed posterior accepted")
    assert events == ["N4_start", "N4_complete", "Atropos_start"]
    checks.append("bad_posterior_refused_before_completion_callback_without_repair")
    method, old = kernel.method(), previous.method()
    for key in ("parameters", "resources", "source_hashes", "expected_unknown_indices_native", "fitting_domain", "posterior_semantics"):
        assert method[key] == old[key]
    assert all(method[k] is False for k in ("mask_value_support_or_frame_change", "N4_or_Atropos_parameter_change", "patient_fitting_released", "anatomy_accepted", "host_accepted", "growth_use_accepted", "training_admitted"))
    checks.append("same_protocol_source_domain_parameters_resources_unreleased_no_acceptance")
    return checks


def main():
    destination = ROOT/"artifacts/ixi265-typed-mask-independent-v3/kernel-preflight-qa.json"
    assert not destination.exists()
    sys.meta_path.insert(0, NoModels())
    before = {name: stamp(ROOT/name) for name in PINS}
    assert {name: row["sha256"] for name, row in before.items()} == PINS
    kernel = load_kernel(); checks = kernel_controls(kernel)
    prior = json.loads((ROOT/"artifacts/ants-mask-dtype-independent-v1/independent-qa.json").read_text())
    outcome = json.loads((ART/"outcome.json").read_text())
    declaration = json.loads((ART/"declaration.json").read_text())
    result = json.loads((ART/"result.json").read_text())
    stages = json.loads((ART/"stages.json").read_text())
    assert sha(ART/"declaration.json") == outcome["declaration_sha256"]
    assert declaration["pins"]["scripts/poc_ixi265_atropos_typed_mask.py"] == result["kernel_sha256"] == KERNEL_SHA
    assert declaration["runner_sha256"] == PINS["scripts/poc_ants_typed_mask_preflight.py"]
    assert declaration["attempts"] == 1 and declaration["patient_fitting_released"] is False
    assert declaration["parameters"] == kernel.method()["parameters"]
    assert outcome["status"] == result["status"] == "exact_v3_kernel_native_preflight_passed"
    assert outcome["exit_code"] == 0 and outcome["runtime_unchanged"]
    assert not outcome["patient_inputs_read"] and not result["patient_sources_read"] and not result["clinical_validity_inferred"]
    limits = outcome["resources"]
    assert outcome["elapsed_seconds"] < limits["seconds_per_arm"]
    assert outcome["peak_supervisor_and_process_family_rss_bytes_sampled"] < limits["rss_limit_bytes"]
    assert outcome["minimum_available_memory_bytes_sampled"] > limits["available_memory_floor_bytes"]
    assert outcome["minimum_free_disk_bytes_sampled"] > limits["disk_floor_bytes"]
    assert outcome["output_and_temporary_bytes"] < limits["output_temporary_limit_bytes"]
    assert stages["worker_pid"] == outcome["worker_pid"]
    events = stages["stages"]
    assert [s["name"] for s in events] == ["N4_start", "N4_complete", "Atropos_start", "Atropos_complete"]
    audit = result["audit"]
    assert events[0]["details"]["mask_boundary"] == audit["N4_mask_boundary"]
    assert events[1]["details"]["frame"] == audit["frames"]["N4"]
    assert events[3]["details"]["classes"] == audit["classes"]
    assert events[0]["details"]["parameters"] == declaration["parameters"]["N4"]
    assert events[2]["details"]["parameters"] == declaration["parameters"]["Atropos"]
    assert audit["N4_mask_boundary"]["native_pixeltype"] == "float" and audit["N4_mask_boundary"]["numpy_dtype"] == "float32"
    assert audit["N4_mask_boundary"]["binary_values_exact"] and audit["N4_mask_boundary"]["support_voxels"] == 33552
    for frame in audit["frames"].values(): frame_check(frame)
    outputs_before = {}
    for row in outcome["manifest"]:
        path = BASE/row["path"]
        assert ".." not in Path(row["path"]).parts
        outputs_before[row["path"]] = stamp(path)
        assert outputs_before[row["path"]]["sha256"] == row["sha256"] and outputs_before[row["path"]]["bytes"] == row["bytes"]
        if (ART/row["path"]).is_file(): assert sha(ART/row["path"]) == row["sha256"]
    values, mask, truth = independently_generate()
    for key, a in (("values", values), ("binary_mask", mask), ("generated_class_truth", truth)):
        assert result["generated_input"][key] == declaration["generated_input"][key] == record(a)
    raw = {r["name"]: r["array"] for r in result["raw_outputs"]}
    corrected = np.load(BASE/"corrected_raw.npy", allow_pickle=False)
    labels = np.load(BASE/"classes_raw.npy", allow_pickle=False)
    assert record(corrected) == raw["corrected_raw"] == prior["arms"]["float32"]["N4_raw"]
    assert np.array_equal(corrected[~mask], values[~mask]) and np.isfinite(corrected).all() and np.all(corrected[mask] > 0)
    assert record(labels) == raw["classes_raw"] and labels.dtype == np.uint8 and np.array_equal(labels, truth)
    total = np.zeros(mask.shape, np.float64)
    for code in (1, 2, 3):
        p = np.load(BASE/f"posterior_{code}_raw.npy", allow_pickle=False)
        assert record(p) == raw[f"posterior_{code}_raw"] == prior["float_raw_posteriors"][code-1]
        assert np.isfinite(p).all() and p.min() >= 0 and p.max() <= 1 and not p[~mask].any()
        total += p
    error = float(np.max(abs(total[mask]-1)))
    assert error == prior["float_posterior_sum_max_abs_error"] == audit["classes"]["posterior_sum_max_abs_error"]
    assert error <= 2e-5 and audit["classes"]["probabilities_modified"] is False
    for row in audit["classes"]["classes"]:
        assert row["voxels"] == int((labels == row["class"]).sum()) and not row["anatomical_acceptance"]
    # Runtime comparisons reuse the exact records already independently checked.
    prior_declaration_path = ROOT/"artifacts/poc-ants-mask-dtype-generated-v1/declaration.json"
    assert sha(prior_declaration_path) == prior["generated_output_manifest_unchanged"]["declaration.json"]["sha256"]
    native_declaration = json.loads(prior_declaration_path.read_text())
    for actual in (declaration["runtime"], result["runtime"]):
        for key in ("packages", "runtime_plan_sha256", "verified_official_API_sources", "installed_native_binary_hashes", "python"):
            assert actual[key] == native_declaration["runtime"][key]
        assert Path(actual["prefix"]).resolve() == Path(native_declaration["runtime"]["prefix"]).resolve()
    assert outputs_before == {name: stamp(BASE/name) for name in outputs_before}
    assert before == {name: stamp(ROOT/name) for name in PINS}
    assert not any(k.split(".")[0] in {"ants", "nibabel", "torch", "monai", "totalsegmentator"} for k in sys.modules)
    report = {"schema": "ixi265-typed-mask-independent-kernel-preflight-v3", "status": "pass_kernel_and_generated_native_preflight_patient_unreleased",
        "created_utc": datetime.now(timezone.utc).isoformat(), "pinned_sources_evidence_unchanged": before,
        "pure_generated_controls": checks, "pure_generated_control_count": len(checks), "new_findings_requiring_fix": [],
        "native_saved_preflight": {"kernel_sha256": KERNEL_SHA, "output_manifest_unchanged": outputs_before,
            "mask_native_type": "float", "mask_dtype": "float32", "mask_support_voxels": 33552,
            "stage_names": [s["name"] for s in events], "all_eight_native_cell_corners_exact": True,
            "all_33552_generated_class_ids_exact": True, "corrected_and_posterior_bytes_match_previously_audited_float_arm": True,
            "class_array_dtype": "uint8", "posterior_sum_max_abs_error": error,
            "unchanged_parameter_source_and_runtime_lineage": True,
            "elapsed_seconds": outcome["elapsed_seconds"], "peak_supervisor_and_process_family_rss_bytes_sampled": outcome["peak_supervisor_and_process_family_rss_bytes_sampled"],
            "exit_code": 0},
        "reuse_of_prior_source_contract_and_two_arm_audit": PINS["artifacts/ants-mask-dtype-independent-v1/independent-qa.json"],
        "patient_data_decoded": False, "new_native_execution": False, "ANTs_or_model_import": False,
        "patient_fitting_released": False, "anatomical_host_growth_or_PVE_acceptance": False,
        "auditor_sha256": sha(__file__), "own_process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "limitations": ["Only the mask API representation changes numerically; added boundary and stage checks make that change observable. Source intensity, exact fitting support, parameters and output semantics stay fixed.",
            "This kernel is worker-only. Exact patient admission and a supervised release remain the caller's responsibility and require a separate review.",
            "The native preflight used a simple generated identity-grid phantom. Pure controls also check an oblique reflected frame, but do not execute native code.",
            "NPY files have no embedded affine; physical frames are independently recomputed from sealed native boundary/output metadata.",
            "One successful generated preflight does not prove real-patient compatibility, anatomical accuracy, or the precise cause of the earlier unsymbolicated crash.",
            "Historical real attempts remain two failures, with no accepted tissue results; no patient fit is released by this audit."]}
    text = json.dumps(report, indent=2, allow_nan=False)+"\n"; assert "/Users/" not in text
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x") as out: out.write(text)
    print(json.dumps({"status": report["status"], "artifact": str(destination.relative_to(ROOT)), "sha256": sha(destination)}))


if __name__ == "__main__": main()
