"""Tiny generated public boundaries; no acquired headers, arrays or model calls."""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from resectionlab import remind_planning_qc as qc
from resectionlab import public_patient_factory as factory


def case_fixture(subject="ReMIND-013", role="SELECT"):
    case = {"patient_id": subject, "patient_group": "ReMIND:" + subject[-3:], "role": role, "series": []}
    for i, kind in enumerate(("structural_t1ce", "whole_tumor", "cerebrum")):
        objects = [{"path": f"/generated/{kind}/{z}", "bytes": 10, "sha256": str(i) * 64}
                   for z in range(3 if i == 0 else 1)]
        case["series"].append({"kind": kind, "Modality": "MR" if i == 0 else "SEG",
            "SeriesInstanceUID": str(i), "StudyInstanceUID": "study", "objects": objects})
    return case


@pytest.mark.parametrize("subject", ("ReMIND-013", "ReMIND-037"))
def test_select_exact_map_and_original_family_role(subject):
    case = case_fixture(subject)
    cohort = {"members": [{"subject": subject, "patient_group": case["patient_group"], "role": "SELECT"}]}
    qc.validate_role(case, cohort, public_only=True)
    with pytest.raises(ValueError):
        qc.validate_role(case, cohort)
    for key, value in (("patient_group", "ReMIND:999"), ("role", "TRAIN"), ("patient_id", "ReMIND-067")):
        with pytest.raises(ValueError):
            qc.validate_role(dict(case, **{key: value}), cohort, public_only=True)


def install_generated_io(monkeypatch, case):
    reads = []
    data = np.arange(27, dtype=np.float32).reshape(3, 3, 3)
    def read(obj, *, pixels, accounting):
        reads.append((obj["path"], pixels))
        accounting["objects_verified"] += 1
        accounting["source_file_returned_bytes"] += obj["bytes"]
        series = next(s for s in case["series"] if obj in s["objects"])
        return SimpleNamespace(PatientID=case["patient_id"], Modality=series["Modality"],
            SeriesInstanceUID=series["SeriesInstanceUID"], StudyInstanceUID="study",
            SOPInstanceUID=obj["path"], index=int(obj["path"].split("/")[-1]),
            pixel_array_options=lambda **kwargs: None)
    def geometry(datasets, core):
        return {"shape_xyz": [3, 3, 3], "affine_xyz_to_ras_mm": np.eye(4).tolist(),
                "frame_of_reference_uid": "generated-frame", "sorted_source_indices": [0, 1, 2]}
    core = SimpleNamespace(convert_mr=lambda ds: (data, np.eye(4), {}),
                           convert_seg=lambda ds: (np.ones((3, 3, 3), np.uint8), np.eye(4), {}))
    monkeypatch.setattr(qc, "read_object", read)
    monkeypatch.setattr(qc, "geometry", geometry)
    monkeypatch.setattr(qc, "ancestry", lambda ds: {"referenced_sop_instance_uids": []})
    return reads, core, data


def test_public_headers_reject_extra_private_series_before_any_read(monkeypatch):
    case = case_fixture()
    reads, core, _ = install_generated_io(monkeypatch, case)
    accounting = {"source_file_returned_bytes": 0, "objects_verified": 0}
    rows = qc.project(case, core, accounting, public_only=True)["series"]
    assert {row["kind"] for row in rows} == qc.PUBLIC_SERIES_KINDS
    assert len(reads) == 5 and all(not pixels for _, pixels in reads)
    reads.clear()
    for extra in ("ventricles", "cerebrum"):
        changed = copy.deepcopy(case)
        changed["series"].append(dict(changed["series"][-1], kind=extra))
        with pytest.raises(ValueError, match="exact_public_series_only"):
            qc.project(changed, core, accounting, public_only=True)
    assert reads == []


def test_public_crop_reads_only_mri_and_two_public_labels(monkeypatch, tmp_path):
    case = case_fixture()
    reads, core, data = install_generated_io(monkeypatch, case)
    accounting = {"source_file_returned_bytes": 0, "objects_verified": 0}
    rows = qc.project(case, core, accounting, public_only=True)["series"]
    case_path = tmp_path / "case.json"; case_path.write_text(json.dumps(case))
    case_sha = qc.digest(case_path.read_bytes())
    headers = {"series": rows, "case_sha256": case_sha, "status": "header_geometry_and_ancestry_projected",
               "reused_converter_sha256": qc.CORE_SHA, "public_only": True}
    header_path = tmp_path / "headers.json"; header_path.write_text(json.dumps(headers))
    monkeypatch.setattr(qc, "validate_case", lambda c, root, **kwargs: {"role": c["role"]})
    monkeypatch.setattr(qc, "source_module", lambda *args: core)
    monkeypatch.setattr(qc, "raw_mr_samples", lambda ds: data[:, :, ds.index].T)
    monkeypatch.setattr(qc, "raw_seg_samples", lambda ds: np.ones((3, 3, 3), np.uint8))
    seen_masks = []
    def overlays(image, masks, output, point, *, public_only=False):
        assert public_only is True
        seen_masks.append(set(masks))
        return {"images": []}
    monkeypatch.setattr(qc, "overlays", overlays)
    reads.clear()
    output = tmp_path / "crop"
    args = SimpleNamespace(repository_root=tmp_path, public_only=True, phase="crop", output=output,
        case=case_path, case_sha256=case_sha, headers=header_path, headers_sha256=qc.digest(header_path.read_bytes()))
    assert qc.run_crop(args) == 0
    report = json.loads((output / "conversion-result.json").read_bytes())
    assert report["role"] == "SELECT" and report["optimizer_updates_performed"] == 0
    assert report["private_reference_loaded"] is False
    assert len(reads) == 5 and all(pixels for _, pixels in reads)
    assert seen_masks == [{"cerebrum", "whole_tumor"}]
    assert set(report["annotations"]) == {"cerebrum", "whole_tumor"}
    assert not any("ventricle" in p.name for p in output.iterdir())
    assert "public_support_private_ventricle_relation" not in report
    assert "evaluation_only_target_ventricle_relation" not in report
    np.testing.assert_array_equal(np.load(output / "MR_native_crop.npy"), data)


def test_public_header_snapshot_requires_same_public_scope(monkeypatch):
    case = case_fixture(); _, core, _ = install_generated_io(monkeypatch, case)
    rows = qc.project(case, core, {"source_file_returned_bytes": 0, "objects_verified": 0}, public_only=True)["series"]
    headers = {"series": rows, "case_sha256": "a" * 64, "status": "header_geometry_and_ancestry_projected",
               "reused_converter_sha256": qc.CORE_SHA, "public_only": True}
    assert len(qc.validate_header_binding(headers, case, "a" * 64, public_only=True)) == 3
    for mutation in ("scope", "frame", "source"):
        changed = copy.deepcopy(headers)
        if mutation == "scope": changed["public_only"] = False
        elif mutation == "frame": changed["series"][1]["geometry"]["frame_of_reference_uid"] = "other"
        else: changed["series"][1]["source_objects"] = []
        with pytest.raises(ValueError):
            qc.validate_header_binding(changed, case, "a" * 64, public_only=True)


def test_select_manifest_role_scope_and_extra_array_refuse(tmp_path, monkeypatch):
    subject = "ReMIND-013"
    cohort = json.dumps({"members": [{"subject": subject, "patient_group": "ReMIND:013", "role": "SELECT"}]}).encode()
    cohort_sha = hashlib.sha256(cohort).hexdigest()
    monkeypatch.setattr(factory, "COHORT_SHA256", cohort_sha)
    manifest = {"patient_id": subject, "patient_group": "ReMIND:013", "role": "SELECT", "public_only": True,
        "input_files": {k: {} for k in factory.ARRAY_KEYS}, "private_evaluation_files_included": False,
        "task_condition": "PARTIAL_TARGET_PROGRESS", "public_support_domain_fully_covered": True,
        "source_bindings": {"cohort_sha256": cohort_sha},
        "reindex_policy": {"source_MR_samples_preserved": True, "original_MR_affine_overwritten": False}}
    path = tmp_path / "public.json"
    def load(value, role="SELECT"):
        raw = json.dumps(value).encode(); path.write_bytes(raw)
        return factory.load_public_manifest(path, hashlib.sha256(raw).hexdigest(), cohort, expected_role=role)
    assert load(manifest)["role"] == "SELECT"
    with pytest.raises(ValueError): load(manifest, "TRAIN")
    for mutation in ("extra", "scope", "subject", "role"):
        altered = copy.deepcopy(manifest)
        if mutation == "extra": altered["input_files"]["private"] = {}
        elif mutation == "scope": altered["public_only"] = False
        elif mutation == "subject": altered["patient_id"] = "ReMIND-067"
        else: altered["role"] = "TRAIN"
        with pytest.raises(ValueError): load(altered)


def test_select_nonzero_updates_refused_before_array_load(monkeypatch, tmp_path):
    monkeypatch.setattr(factory, "load_public_manifest", lambda *args, **kwargs: pytest.fail("manifest loading reached"))
    limits = {"max_steps": 6, "max_optimizer_updates": 1, "max_native_previews": 10, "max_policy_forwards": 10,
              "worker_seconds": 10, "memory_bytes": 1000, "threads": 1,
              "search": {"max_calls": 1, "beam_width": 1, "seconds": 1}}
    with pytest.raises(ValueError, match="zero-update"):
        factory.prepare_public_source(tmp_path, {"limits": limits}, "b" * 64, None, lambda *a, **k: None,
            public_manifest_path=tmp_path / "absent", public_manifest_sha256="c" * 64, cohort_bytes=b"",
            learning_protocol_hash="sha256:" + "d" * 64, expected_role="SELECT", checkpoint_lineage={})


def test_select_factory_emits_frozen_checkpoint_protocol_with_same_public_construction(monkeypatch, tmp_path):
    from resectionlab import native_spatial_task
    learning_hash = "sha256:" + "d" * 64
    lineage = {"version": "frozen-TRAIN-checkpoint-lineage-v1", "checkpoint_sha256": "a" * 64,
        "method": "IL", "training_release_sha256": "b" * 64, "learning_protocol_hash": learning_hash,
        "initial_parameter_hash": "sha256:" + "e" * 64, "parameter_hash": "sha256:" + "f" * 64,
        "architecture_hash": "sha256:" + "c" * 64, "completed_updates": 1,
        "training_context_hashes": {"ReMIND:" + n: "sha256:" + "1" * 64 for n in ("008", "010", "020", "025")},
        "public_target_context_variant": None, "optimizer_updates_on_SELECT": 0}
    cohort = json.dumps({"members": [{"subject": "ReMIND-013", "patient_group": "ReMIND:013", "role": "SELECT"}]}).encode()
    cohort_sha = hashlib.sha256(cohort).hexdigest(); monkeypatch.setattr(factory, "COHORT_SHA256", cohort_sha)
    files = {}
    for key in factory.ARRAY_KEYS:
        data = np.arange(27, dtype=np.float32).reshape(3, 3, 3) if key == "image" else np.ones((3, 3, 3), np.uint8)
        path = tmp_path / (key + ".npy"); np.save(path, data, allow_pickle=False)
        files[key] = {"path": str(path), "bytes": path.stat().st_size, "sha256": factory.sha(path), "dtype": str(data.dtype)}
    manifest = {"patient_id": "ReMIND-013", "patient_group": "ReMIND:013", "role": "SELECT", "public_only": True,
        "input_files": files, "private_evaluation_files_included": False, "task_condition": "PARTIAL_TARGET_PROGRESS",
        "public_support_domain_fully_covered": True, "shape_xyz": [3, 3, 3], "affine_ras_mm": np.eye(4).tolist(),
        "source_MR_crop_affine_ras_mm": np.eye(4).tolist(), "public_label_resampling": {},
        "source_bindings": {"cohort_sha256": cohort_sha, "saved_array_review_sha256": "a" * 64},
        "reindex_policy": {"source_MR_samples_preserved": True, "original_MR_affine_overwritten": False},
        "public_target_support_consistency": {"whole_tumor_positive_voxels": 27, "whole_tumor_positive_outside_supplied_support": 0}}
    path = tmp_path / "manifest.json"; path.write_text(json.dumps(manifest))
    captured = {}
    def constructor(image, support, target, affine, access, tools, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(structural_intensity=image, observed_support=support, nominal_target=target,
            affine_ras_mm=affine, source_hash="generated-source", _supplied_goal_extent={}, _grid_record={}, _normalization_record={})
    monkeypatch.setattr(native_spatial_task, "NativeSpatialCase", constructor)
    limits = {"max_steps": 6, "max_optimizer_updates": 0, "max_native_previews": 10, "max_policy_forwards": 10,
              "worker_seconds": 10, "memory_bytes": 1000, "threads": 1,
              "search": {"max_calls": 1, "beam_width": 1, "seconds": 1}}
    output = tmp_path / "output"; output.mkdir()
    source, binding, review, protocol = factory.prepare_public_source(output, {"limits": limits}, "b" * 64, None,
        lambda *a, **k: None, public_manifest_path=path, public_manifest_sha256=factory.sha(path), cohort_bytes=cohort,
        learning_protocol_hash=learning_hash, expected_role="SELECT", checkpoint_lineage=lineage)
    assert binding["subject"] == "ReMIND-013" and review["subject"] == "ReMIND-013"
    assert protocol["role"] == "SELECT" and protocol["max_optimizer_updates"] == 0
    assert protocol["initialization"] == "frozen_TRAIN_checkpoint_reload" and protocol["checkpoint_lineage"] == lineage
    assert captured["crop_shape"] == (64, 64, 64) and captured["proposal_mode"] == "nominal_cavity_v1"
    assert int(source.nominal_target.sum()) == 27
