"""Case4 DEVELOPMENT diagnostic file handoff; no eager imaging or model imports.

This opt-in source module verifies saved source receipts and connects a scan-only
prepared patch to a separately supervised network output and display artifact.
It never launches a network or grants patient/planner admission on its own.
No Case4 patient-forward release is pinned in this source version.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any


RECEIPT_PINS = {
    "manifest": ("manifests/experiments/resect-case4-scan-diagnostic-v1.json",
                 "d37948a4dc7acd16bdfac0244b926cfda5c684c60da1f6e4dfccb09ddcf21b26"),
    "preparation": ("artifacts/resect-case4-scan-preparation-v1/preparation-result.json",
                    "92fdc93f1240c81ac0ecbd7910a8256a18693480bd4bed48b19e109fe308d72c"),
    "preparation_supervision": ("artifacts/resect-case4-scan-preparation-v1/supervision-result.json",
                                "bc105184dd05502e334ed9b758ee3faf37be97d0a45eb2a084c551bdd461d3fd"),
    "support": ("artifacts/case4-source-support-map-v2/support-map-result.json",
                "c3dbdd2b167bfe200575d78401ef4ca02b27bd34249ab180e0e11a5896faff8c"),
    "support_supervision": ("artifacts/case4-source-support-map-v2/support-map-supervision-result.json",
                            "af4cb2afb1bc70bebc1534f94f52aaf2b97464f65f16895dcfaecc641e72dd26"),
}
MODEL_META = "data/models/gliomoda-v1.0.2/t1c-t2f-pinned-v1"
CASE4_PREP = "outputs/scan-target/resect-case4-prep-v1/attempt-01"
MODEL_BACKEND = "accepted_full128_recompute_stage0_cpu_v3_requires_case4_adaptation"
# A future reviewed patient-specific worker/supervisor release must set this
# to its exact SHA in a new candidate. The generated-only v3 release is invalid.
CASE4_FORWARD_RELEASE_SHA256 = None


class DiagnosticHandoffError(ValueError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checked_json(root: Path, relative: str, expected: str) -> dict:
    path = root / relative
    if not path.is_file() or path.is_symlink():
        raise DiagnosticHandoffError("required exact JSON receipt is absent or changed: " + relative)
    before = sha256_file(path)
    if expected and before != expected:
        raise DiagnosticHandoffError("required exact JSON receipt is absent or changed: " + relative)
    value = json.loads(path.read_text())
    if sha256_file(path) != before:
        raise DiagnosticHandoffError("JSON receipt changed while reading: " + relative)
    if not isinstance(value, dict):
        raise DiagnosticHandoffError("expected JSON object: " + relative)
    return value


def _inventory(result: dict) -> dict[str, dict]:
    entries = result.get("generated_files", [])
    if not isinstance(entries, list):
        raise DiagnosticHandoffError("preparation file inventory is absent")
    out: dict[str, dict] = {}
    for entry in entries:
        name = entry.get("relative_path")
        if (not isinstance(name, str) or Path(name).is_absolute() or ".." in Path(name).parts
                or name in out):
            raise DiagnosticHandoffError("preparation inventory path is invalid")
        out[name] = entry
    return out


@dataclass(frozen=True)
class Case4Bindings:
    t1c_path: Path
    t1c_sha256: str
    flair_path: Path
    flair_sha256: str
    support_path: Path
    support_sha256: str
    plans_path: Path
    plans_sha256: str
    dataset_path: Path
    dataset_sha256: str
    checkpoint_sha256: str
    evidence_sha256: dict[str, str]


def verify_case4_metadata(root: Path, *, check_input_bytes: bool = False) -> Case4Bindings:
    """Bind trusted records; optional byte pass is for a separately released run.

    With ``check_input_bytes=False`` this reads only small JSON metadata. It
    never opens a patient volume or checkpoint during source-only review.
    """
    root = Path(root)
    records = {name: _checked_json(root, path, digest)
               for name, (path, digest) in RECEIPT_PINS.items()}
    manifest, prep, prep_guard = (records[key] for key in
                                  ("manifest", "preparation", "preparation_supervision"))
    support, support_guard = records["support"], records["support_supervision"]
    if (manifest.get("case_id") != "RESECT-Case4" or manifest.get("split_role") != "DEVELOPMENT"
            or manifest.get("model_inference_admitted") is not False
            or manifest.get("planning_admitted") is not False
            or prep.get("split_role") != "DEVELOPMENT"
            or prep.get("model_inference_performed") is not False
            or prep.get("planning_admitted") is not False
            or prep.get("manifest_sha256") != RECEIPT_PINS["manifest"][1]
            or prep_guard.get("status") != "completed_geometry_only_qc_pending"
            or prep_guard.get("stop_reason") is not None
            or support.get("schema_version") != "case4-support-map-v1"
            or support.get("source_preparation_result_sha256") != RECEIPT_PINS["preparation"][1]
            or support.get("source_manifest_sha256") != RECEIPT_PINS["manifest"][1]
            or support.get("diagnostic_input_domain_qualified") is not False
            or support.get("mask_anatomy_qualified") is not False
            or support.get("planning_admitted") is not False
            or support_guard.get("status") != "completed_source_support_only_qc_pending"
            or support_guard.get("cleanup_errors") != []
            or support_guard.get("final_owned_group_pids") != []):
        raise DiagnosticHandoffError("Case4 DEVELOPMENT/unknown-anatomy source chain changed")
    inventory = _inventory(prep)
    t1 = inventory["prepared/input_0000.nii.gz"]
    flair = inventory["prepared/input_0001.nii.gz"]
    for key, entry in (("prepared_t1c", t1), ("prepared_flair", flair)):
        if prep[key].get("sha256") != entry.get("sha256"):
            raise DiagnosticHandoffError("prepared channel does not match inventory")
    if (manifest["preoperative_t1c"].get("channel_index") != 0 or
            manifest["preoperative_flair"].get("channel_index") != 1 or
            manifest["model_extraction"].get("training_lineage_status") != "unknown" or
            manifest["model_extraction"].get("case_overlap_status") != "unknown" or
            manifest["model_extraction"].get("plans_json_sha256") !=
            "e85abbe6a41f4e5e0164d53dd3a297baf10e06cd5f05c1d46a22d2d869d64ad5" or
            manifest["model_extraction"].get("dataset_json_sha256") !=
            "3a7c7c1fd5eb420243a25496bae2c2fb9f210d7910363b9361aaebc69e615a3f" or
            manifest["model_extraction"].get("checkpoint_final_sha256") !=
            "0e29f882310fe8cb076d6cadb982067ef53c6a32231f40ae17d9c173aa4307b3"):
        raise DiagnosticHandoffError("channel identity or model metadata contract changed")
    source = root / CASE4_PREP
    model = root / MODEL_META
    bindings = Case4Bindings(
        source / t1["relative_path"], t1["sha256"],
        source / flair["relative_path"], flair["sha256"],
        root / support["support_bits_path"], support["support_bits_sha256"],
        model / "plans.json", manifest["model_extraction"]["plans_json_sha256"],
        model / "dataset.json", manifest["model_extraction"]["dataset_json_sha256"],
        manifest["model_extraction"]["checkpoint_final_sha256"],
        {name: digest for name, (_, digest) in RECEIPT_PINS.items() if digest})
    if check_input_bytes:
        checked = [(source / entry["relative_path"], entry["sha256"])
                   for entry in inventory.values()]
        checked += [
            (root / manifest["preoperative_t1c"]["path"],
             manifest["preoperative_t1c"]["sha256"]),
            (root / manifest["preoperative_flair"]["path"],
             manifest["preoperative_flair"]["sha256"]),
            (root / manifest["limited_scan_derived_t1_mask"]["path"],
             manifest["limited_scan_derived_t1_mask"]["sha256"]),
            (root / manifest["limited_scan_derived_t1_mask"]["generation_record_path"],
             manifest["limited_scan_derived_t1_mask"]["generation_record_sha256"]),
            (root / manifest["atlas"]["path"], manifest["atlas"]["sha256"]),
            (root / manifest["model_extraction"]["receipt_path"],
             manifest["model_extraction"]["receipt_sha256"]),
            (bindings.support_path, bindings.support_sha256),
            (bindings.plans_path, bindings.plans_sha256),
            (bindings.dataset_path, bindings.dataset_sha256),
            (model / "fold_all/checkpoint_final.pth", bindings.checkpoint_sha256),
        ]
        for path, digest in checked:
            if path.is_symlink() or not path.is_file() or sha256_file(path) != digest:
                raise DiagnosticHandoffError("exact input/model bytes changed: " + str(path))
    return bindings


def prepare_patch_from_bindings(bindings: Case4Bindings):
    """Explicit future DEVELOPMENT call; this is not invoked by metadata review."""
    from resectionlab.scan_preprocess_bridge import prepare_scan_only_patch

    return prepare_scan_only_patch(
        t1c_path=bindings.t1c_path, t1c_sha256=bindings.t1c_sha256,
        flair_path=bindings.flair_path, flair_sha256=bindings.flair_sha256,
        support_map_path=bindings.support_path,
        support_map_sha256=bindings.support_sha256,
        plans_path=bindings.plans_path, plans_sha256=bindings.plans_sha256,
        dataset_path=bindings.dataset_path, dataset_sha256=bindings.dataset_sha256)


def model_input_array(prepared) -> Any:
    """Single [1,2,128³] contiguous float32 patch for the accepted path."""
    import numpy as np

    value = np.asarray(prepared.tensor_czyx)
    if (value.shape != (2, 128, 128, 128) or value.dtype != np.float32
            or not np.isfinite(value).all()):
        raise DiagnosticHandoffError("preprocessed model input is not finite [2,128³] float32")
    return np.ascontiguousarray(value[None])


def _geometry_sha256(prepared) -> str:
    """Bind every inverse-map field, not just source hashes and atlas shape."""
    value = json.dumps(asdict(prepared.geometry), sort_keys=True,
                       separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def _require_prepared_input(prepared, receipt: dict, *, patient: bool) -> None:
    """Reject a different inverse map even when someone reuses valid tensor bytes."""
    expected_schema = ("case4_unreviewed_diagnostic_input_v1" if patient else
                       "unverified_scan_patch_input_control_v1")
    expected_origin = ("verified_case4_source_receipts" if patient else
                       "unverified_supplied_arrays")
    source = prepared.geometry
    values = model_input_array(prepared)
    if (receipt.get("schema") != expected_schema or
            receipt.get("source_origin") != expected_origin or
            receipt.get("input_shape") != [1, 2, 128, 128, 128] or
            receipt.get("input_raw_sha256") != hashlib.sha256(values.tobytes(order="C")).hexdigest() or
            receipt.get("source_sha256") != source.display_only_metadata()["source_sha256"] or
            receipt.get("patch_start_preprocessed_zyx") != list(source.patch_start_zyx) or
            receipt.get("atlas_shape_xyz") != list(source.atlas_shape_xyz) or
            receipt.get("atlas_affine_ras_mm") != [list(row) for row in source.atlas_affine_ras_mm] or
            receipt.get("geometry_sha256") != _geometry_sha256(prepared) or
            receipt.get("coordinate_frame") != "atlas_RAS_mm_coded_sform" or
            receipt.get("planning_eligible") is not False or
            receipt.get("evaluation_eligible") is not False or
            receipt.get("clinical_evidence") is not False):
        raise DiagnosticHandoffError("prepared source/frame/crop/patch differs from input receipt")
    if patient and receipt.get("evidence_sha256") != {
            name: digest for name, (_, digest) in RECEIPT_PINS.items()}:
        raise DiagnosticHandoffError("Case4 input lacks pinned source receipt chain")


def _write_model_input(prepared, destination: Path, *, trusted_case4: bool) -> dict:
    """Common no-clobber serialization; only the patient wrapper asserts origin."""
    import numpy as np

    source = prepared.geometry
    values = model_input_array(prepared)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    tensor_path = destination / "input.npy"
    with tensor_path.open("xb") as stream:
        np.save(stream, values, allow_pickle=False)
    result = {
        "schema": ("case4_unreviewed_diagnostic_input_v1" if trusted_case4
                   else "unverified_scan_patch_input_control_v1"),
        "source_origin": ("verified_case4_source_receipts" if trusted_case4
                          else "unverified_supplied_arrays"),
        "scope": "one scan-derived 128-cube patch; no annotation",
        "model_backend": MODEL_BACKEND,
        "input_shape": [1, 2, 128, 128, 128],
        "input_npy_sha256": sha256_file(tensor_path),
        "input_raw_sha256": hashlib.sha256(values.tobytes(order="C")).hexdigest(),
        "source_sha256": source.display_only_metadata()["source_sha256"],
        "patch_start_preprocessed_zyx": list(source.patch_start_zyx),
        "atlas_shape_xyz": list(source.atlas_shape_xyz),
        "atlas_affine_ras_mm": [list(row) for row in source.atlas_affine_ras_mm],
        "geometry_sha256": _geometry_sha256(prepared),
        "evidence_sha256": ({name: digest for name, (_, digest) in RECEIPT_PINS.items()}
                            if trusted_case4 else None),
        "coordinate_frame": "atlas_RAS_mm_coded_sform",
        "training_overlap_status": "unknown",
        "anatomical_qc": "unreviewed_inferior_mask_omission",
        "model_forward_performed": False,
        "planning_eligible": False,
        "evaluation_eligible": False,
        "clinical_evidence": False,
    }
    _require_prepared_input(prepared, result, patient=trusted_case4)
    (destination / "input-receipt.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def write_unverified_model_input_control(prepared, destination: Path) -> dict:
    """Generated tests may serialize supplied arrays, without Case4 provenance."""
    return _write_model_input(prepared, destination, trusted_case4=False)


def write_case4_model_input(root: Path, destination: Path) -> dict:
    """Prospective patient call verifies source bytes before opening scan arrays."""
    bindings = verify_case4_metadata(root, check_input_bytes=True)
    prepared = prepare_patch_from_bindings(bindings)
    source = prepared.geometry
    if (source.t1c_sha256 != bindings.t1c_sha256 or
            source.flair_sha256 != bindings.flair_sha256 or
            source.support_map_sha256 != bindings.support_sha256 or
            source.plans_sha256 != bindings.plans_sha256 or
            source.dataset_sha256 != bindings.dataset_sha256):
        raise DiagnosticHandoffError("prepared patch differs from pinned Case4 source chain")
    return _write_model_input(prepared, destination, trusted_case4=True)


def decode_model_output(prepared, logits, *, input_receipt: dict,
                        accepted_forward: dict):
    """Decode only after a separately pinned supervisor verifies a real forward.

    The returned object remains unreviewed display-only. The model worker
    source/output pins must be present in a *prospective* release; a caller's
    self-consistent hashes do not provide a patient trust anchor.
    """
    if CASE4_FORWARD_RELEASE_SHA256 is None:
        raise DiagnosticHandoffError("no reviewed Case4 patient-forward release exists")
    import numpy as np

    _require_prepared_input(prepared, input_receipt, patient=True)
    value = np.asarray(logits)
    if (value.shape != (1, 3, 128, 128, 128) or value.dtype != np.float32
            or not np.isfinite(value).all()):
        raise DiagnosticHandoffError("network output is not finite [1,3,128³] float32")
    actual_output_sha256 = hashlib.sha256(
        np.ascontiguousarray(value).tobytes(order="C")).hexdigest()
    if (accepted_forward.get("status") != "accepted_case4_development_single128" or
            accepted_forward.get("release_sha256") != CASE4_FORWARD_RELEASE_SHA256 or
            accepted_forward.get("input_raw_sha256") != input_receipt["input_raw_sha256"] or
            accepted_forward.get("geometry_sha256") != input_receipt["geometry_sha256"] or
            accepted_forward.get("logits_raw_sha256") != actual_output_sha256 or
            accepted_forward.get("planning_admitted") is not False or
            accepted_forward.get("clinical_evidence") is not False):
        raise DiagnosticHandoffError("forward receipt is absent, mismatched or overclaims")
    coverage = np.ones((128, 128, 128), np.uint8)
    return prepared.decode_supplied_logits_for_display(value[0], coverage)


def read_accepted_forward(model_backend_dir: Path, input_receipt: dict,
                          checkpoint_sha256: str) -> tuple[Any, dict]:
    """Verify one supervised result; fail closed until its release is pinned.

    This routine opens only the saved output, not patient source images or
    checkpoint. It cannot run before an independently reviewed release digest
    is fixed in a new source version.
    """
    if CASE4_FORWARD_RELEASE_SHA256 is None:
        raise DiagnosticHandoffError("no reviewed Case4 patient-forward release exists")
    import numpy as np

    if (input_receipt.get("schema") != "case4_unreviewed_diagnostic_input_v1" or
            input_receipt.get("source_origin") != "verified_case4_source_receipts" or
            input_receipt.get("evidence_sha256") != {
                name: digest for name, (_, digest) in RECEIPT_PINS.items()}):
        raise DiagnosticHandoffError("patient forward requires exact Case4 input provenance")
    here = Path(model_backend_dir)
    contract_path = here / "pair-contract.json"
    contract = _checked_json(here, "pair-contract.json", CASE4_FORWARD_RELEASE_SHA256)
    saved_input_path = here / "case4-input/input-receipt.json"
    saved_input = _checked_json(here, "case4-input/input-receipt.json", "")
    saved_input_sha256 = sha256_file(saved_input_path)
    saved_tensor = here / "case4-input/input.npy"
    result = _checked_json(here, "case4/result.json", "")
    supervision = _checked_json(here, "case4/supervision.json", "")
    finalization = supervision.get("finalization", {})
    if (contract.get("gate") != "CASE4_DEVELOPMENT_SINGLE128_RECOMPUTE_V1" or
            saved_input != input_receipt or
            contract.get("input_receipt_sha256") != saved_input_sha256 or
            not saved_tensor.is_file() or saved_tensor.is_symlink() or
            sha256_file(saved_tensor) != input_receipt.get("input_npy_sha256") or
            contract.get("geometry_sha256") != input_receipt.get("geometry_sha256") or
            contract.get("input_npy_sha256") != input_receipt.get("input_npy_sha256") or
            contract.get("input_raw_sha256") != input_receipt.get("input_raw_sha256") or
            contract.get("source_sha256") != input_receipt.get("source_sha256") or
            contract.get("model_file_sha256", {}).get("fold_all/checkpoint_final.pth") !=
            checkpoint_sha256 or
            supervision.get("arm") != "case4" or
            supervision.get("contract_sha256") != sha256_file(contract_path) or
            supervision.get("accepted_for_feasibility") is not True or
            supervision.get("exit_code") != 0 or
            supervision.get("watchdog_reason") is not None or
            supervision.get("post_guard_reason") is not None or
            finalization.get("cleanup_errors") != [] or
            finalization.get("remaining_detached_pids") != [] or
            finalization.get("remaining_group_pids") != [] or
            result.get("arm") != "case4" or
            result.get("contract_sha256") != sha256_file(contract_path) or
            result.get("input_npy_sha256") != input_receipt.get("input_npy_sha256") or
            result.get("input_raw_sha256") != input_receipt.get("input_raw_sha256") or
            result.get("source_sha256") != input_receipt.get("source_sha256") or
            result.get("checkpoint_sha256_after") != checkpoint_sha256 or
            result.get("model_inference_performed") is not True or
            result.get("output_nonfinite") != 0 or
            result.get("planning_admitted") is not False or
            result.get("clinical_evidence") is not False or
            result.get("output_shape") != [1, 3, 128, 128, 128]):
        raise DiagnosticHandoffError("network receipt/guard or Case4 lineage is not accepted")
    output_path = here / "case4/logits.npy"
    if not output_path.is_file() or output_path.is_symlink():
        raise DiagnosticHandoffError("exact logits output is absent")
    npy_sha256 = sha256_file(output_path)
    if npy_sha256 != result.get("logits_npy_sha256"):
        raise DiagnosticHandoffError("logits file changed after supervised result")
    logits = np.load(output_path, allow_pickle=False)
    if (logits.shape != (1, 3, 128, 128, 128) or logits.dtype != np.float32 or
            not np.isfinite(logits).all()):
        raise DiagnosticHandoffError("supervised logits shape/dtype/finite check failed")
    raw_sha256 = hashlib.sha256(np.ascontiguousarray(logits).tobytes(order="C")).hexdigest()
    if raw_sha256 != result.get("logits_raw_sha256") or sha256_file(output_path) != npy_sha256:
        raise DiagnosticHandoffError("supervised logits payload changed during decode")
    accepted = {
        "status": "accepted_case4_development_single128",
        "release_sha256": CASE4_FORWARD_RELEASE_SHA256,
        "input_raw_sha256": input_receipt["input_raw_sha256"],
        "input_receipt_sha256": saved_input_sha256,
        "geometry_sha256": input_receipt["geometry_sha256"],
        "logits_raw_sha256": raw_sha256,
        "supervision_sha256": sha256_file(here / "case4/supervision.json"),
        "result_sha256": sha256_file(here / "case4/result.json"),
        "planning_admitted": False,
        "clinical_evidence": False,
    }
    return logits, accepted


def _write_display_artifact(prepared, display, destination: Path, *,
                            input_receipt_sha256: str | None,
                            forward_receipt_sha256: str | None,
                            patient_output: bool) -> dict:
    """Serialize a verified or explicitly unverified display, no overlay claim."""
    import nibabel as nib
    import numpy as np

    for name, value in (("input", input_receipt_sha256),
                        ("forward", forward_receipt_sha256)):
        if value is None and not patient_output:
            continue
        if (not isinstance(value, str) or len(value) != 64 or
                any(c not in "0123456789abcdef" for c in value)):
            raise DiagnosticHandoffError(name + " receipt SHA-256 is invalid")

    affine = np.asarray(prepared.geometry.atlas_affine_ras_mm, dtype=np.float64)
    state = np.asarray(display.state_xyz)
    coverage = np.asarray(display.prediction_coverage_xyz)
    if (state.shape != prepared.geometry.atlas_shape_xyz or
            coverage.shape != state.shape or
            not np.isin(state, (-1, 0, 1)).all() or
            not np.isin(coverage, (0, 1)).all() or
            np.any((state != -1) != coverage.astype(bool)) or
            display.metadata.get("source_sha256") !=
            prepared.geometry.display_only_metadata()["source_sha256"] or
            display.metadata.get("shape_xyz") != list(prepared.geometry.atlas_shape_xyz) or
            display.metadata.get("affine_ras_mm") != affine.tolist() or
            display.metadata.get("planning_eligible") is not False or
            display.metadata.get("evaluation_eligible") is not False):
        raise DiagnosticHandoffError("display state/coverage or admission flags changed")
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    outputs = {}
    for name, value in (("state", state.astype(np.int8, copy=False)),
                        ("coverage", coverage.astype(np.uint8, copy=False))):
        path = destination / f"{name}.nii.gz"
        image = nib.Nifti1Image(value, affine)
        image.header.set_xyzt_units("mm")
        image.set_sform(affine, code=2)
        image.set_qform(None, code=0)
        nib.save(image, str(path))
        outputs[name] = {"path": path.name, "sha256": sha256_file(path),
                         "dtype": str(value.dtype)}
    artifact = {
        "schema": ("case4_unreviewed_diagnostic_display_layer_v1" if patient_output
                   else "unverified_scan_diagnostic_display_control_v1"),
        "scope": "display_only_atlas_native_grid",
        "output_origin": ("model_generated_from_observed_case4_preoperative_scans" if patient_output
                          else "unverified_supplied_arrays_may_include_patient_data"),
        "model_output_verified": patient_output,
        "state_semantics": {"-1": "unknown", "0": "candidate_negative_where_output_covered",
                            "1": "candidate_positive_where_output_covered"},
        "shape_xyz": list(prepared.geometry.atlas_shape_xyz),
        "affine_ras_mm": affine.tolist(),
        "source_sha256": display.metadata["source_sha256"],
        "geometry_sha256": _geometry_sha256(prepared),
        "input_receipt_sha256": input_receipt_sha256,
        "forward_receipt_sha256": forward_receipt_sha256,
        "predicted_voxels": int(coverage.sum()),
        "unknown_voxels": int((state == -1).sum()),
        "excluded_positive_voxels": display.positives_excluded_outside_coverage,
        "outputs": outputs,
        "anatomical_qc": "unreviewed_inferior_mask_omission",
        "training_overlap_status": "unknown",
        "same_grid_only": True,
        "planning_eligible": False,
        "evaluation_eligible": False,
        "clinical_evidence": False,
    }
    (destination / "display-layer.json").write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    return artifact


def write_unverified_display_control(prepared, display, destination: Path) -> dict:
    """For generated algorithm tests only; provenance is never asserted."""
    return _write_display_artifact(
        prepared, display, destination, input_receipt_sha256=None,
        forward_receipt_sha256=None, patient_output=False)


def write_case4_display_from_saved(root: Path, handoff_dir: Path,
                                   model_backend_dir: Path, destination: Path) -> dict:
    """Prospective patient display path: independently rebind every saved input.

    This opens patient scans only when called by a separately reviewed release.
    At present the release pin is intentionally unset and the call refuses
    before any payload read.
    """
    if CASE4_FORWARD_RELEASE_SHA256 is None:
        raise DiagnosticHandoffError("no reviewed Case4 patient-forward release exists")
    bindings = verify_case4_metadata(root, check_input_bytes=True)
    prepared = prepare_patch_from_bindings(bindings)
    handoff_dir = Path(handoff_dir)
    input_receipt = _checked_json(handoff_dir, "input-receipt.json", "")
    _require_prepared_input(prepared, input_receipt, patient=True)
    input_path = handoff_dir / "input.npy"
    if (not input_path.is_file() or input_path.is_symlink() or
            sha256_file(input_path) != input_receipt.get("input_npy_sha256")):
        raise DiagnosticHandoffError("saved Case4 model input bytes changed")
    logits, accepted = read_accepted_forward(model_backend_dir, input_receipt,
                                             bindings.checkpoint_sha256)
    if accepted.get("input_receipt_sha256") != sha256_file(handoff_dir / "input-receipt.json"):
        raise DiagnosticHandoffError("forward used a different Case4 handoff receipt")
    display = decode_model_output(prepared, logits, input_receipt=input_receipt,
                                  accepted_forward=accepted)
    return _write_display_artifact(
        prepared, display, destination,
        input_receipt_sha256=accepted["input_receipt_sha256"],
        forward_receipt_sha256=accepted["result_sha256"], patient_output=True)
