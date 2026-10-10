"""Generated controls only: no acquired images, private labels or models.

The small fabricated manifests exercise the acquired-source adapter contract;
they are not evidence of patient QC. Optional historical-file parity is invoked
by the owning source test command, without changing the import path in this test.
"""
from dataclasses import replace
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest

from resectionlab.core import array_digest, semantic_digest, thaw_json
from resectionlab import public_patient_factory as factory
from resectionlab import patient_planning_admission as admission
from resectionlab import native_spatial_task as native
from resectionlab.native_proposals import SUPPLIED_GOAL_REGION, NominalCavityProposalConfig
from resectionlab.public_target_context import VERSION as TARGET_CONTEXT

UNION = native.SUPPLIED_TUMOR_UNION_OCCUPANCY
RAW = native.RAW_CEREBRUM_OCCUPANCY


def fixture(tmp_path, subject="ReMIND-008", role="TRAIN"):
    repo = next(p for p in Path(__file__).resolve().parents if (p / "manifests").is_dir())
    cohort = (repo / "manifests/experiments/remind-component-cohort-v1.json").read_bytes()
    image = np.arange(125, dtype=np.float32).reshape(5, 5, 5)
    support = np.zeros(image.shape, np.uint8); support[1:4, 1:4, 1:4] = 1
    target = np.zeros_like(support); target[3:5, 2, 2] = 1
    domain = np.ones_like(support); domain[0, 0, 0] = 0
    files = {}
    for key, values in zip(factory.ARRAY_KEYS, (image, support, target, domain)):
        path = tmp_path / (key + ".npy"); np.save(path, values, allow_pickle=False)
        files[key] = {"path": str(path), "sha256": factory.sha(path),
            "bytes": path.stat().st_size, "dtype": str(values.dtype)}
    manifest = {"patient_id": subject, "patient_group": "ReMIND:"+subject.split("-")[1],
        "role": role, "input_files": files, "private_evaluation_files_included": False,
        "public_only": True, "task_condition": "PARTIAL_TARGET_PROGRESS",
        "public_support_domain_fully_covered": True, "shape_xyz": list(image.shape),
        "affine_ras_mm": np.eye(4).tolist(), "source_MR_crop_affine_ras_mm": np.eye(4).tolist(),
        "public_label_resampling": {}, "source_bindings": {"cohort_sha256": admission.COHORT_SHA256,
            "saved_array_review_sha256": "a"*64}, "reindex_policy": {
            "source_MR_samples_preserved": True, "original_MR_affine_overwritten": False},
        "public_target_support_consistency": {"whole_tumor_positive_voxels": 2,
            "whole_tumor_positive_outside_supplied_support": 1}}
    path = tmp_path / "manifest.json"; path.write_text(json.dumps(manifest))
    return dict(public_manifest_path=path, public_manifest_sha256=factory.sha(path),
        cohort_bytes=cohort), (image, support, target, domain)


def build(tmp_path, inputs, *, condition=RAW, module=factory, source_hash=None):
    tmp_path.mkdir()
    limits = {"max_steps": 2, "max_optimizer_updates": 0, "max_native_previews": 1024,
        "max_policy_forwards": 0 if condition == UNION else 1,
        "worker_seconds": 10, "memory_bytes": 512*1024**2, "threads": 1,
        "search": {"max_calls": 64, "beam_width": 2, "seconds": 5}}
    release = {"limits": limits}
    if source_hash is not None: release["baseline_public_source_hash"] = source_hash
    kwargs = {} if condition is None else {"occupancy_condition": condition}
    return module.prepare_public_source(tmp_path, release, "b"*64, None, lambda *a, **k: None,
        **inputs, learning_protocol_hash="sha256:"+"c"*64,
        public_target_context_variant=TARGET_CONTEXT, **kwargs)


def contract(built, inputs):
    case, binding, qc, protocol = built
    return case, dict(cohort_bytes=inputs["cohort_bytes"], source_binding=binding,
        qc_receipt=qc, protocol=protocol)


def rebind(args):
    digest = semantic_digest(args["source_binding"])
    args["qc_receipt"]["public_source_binding_hash"] = digest
    args["protocol"]["public_source_binding_hash"] = digest
    args["protocol"]["qc_receipt_hash"] = semantic_digest(args["qc_receipt"])


def test_raw_default_and_explicit_condition_reconstruct_exactly(tmp_path):
    inputs, _ = fixture(tmp_path)
    implicit = build(tmp_path / "implicit", inputs, condition=None)
    explicit = build(tmp_path / "explicit", inputs, source_hash=implicit[0].source_hash)
    assert implicit[1:] == explicit[1:]
    assert implicit[0].source_hash == explicit[0].source_hash
    for filename in ("public-task-derivation.json", "admitted-public-bindings.json"):
        assert (tmp_path / "implicit" / filename).read_bytes() == (tmp_path / "explicit" / filename).read_bytes()
    assert "occupancy_derivation" not in implicit[1]
    assert "occupancy_condition" not in implicit[3]


def test_historical_default_source_protocol_observation_and_inventory_parity(tmp_path, monkeypatch):
    baseline = os.environ.get("PAIRED_OCCUPANCY_BASELINE_DIR")
    if baseline is None: pytest.skip("historical source control is run with its pinned baseline directory")
    inputs, _ = fixture(tmp_path)
    modules = []
    for filename in ("native_spatial_task", "public_patient_factory"):
        name = "resectionlab._occupancy_baseline_" + filename
        spec = importlib.util.spec_from_file_location(name, Path(baseline) / (filename + ".py"))
        module = importlib.util.module_from_spec(spec); monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module); modules.append(module)
    old_native, old_factory = modules
    current = build(tmp_path / "current", inputs, condition=None)
    with monkeypatch.context() as m:
        m.setattr(native, "NativeSpatialCase", old_native.NativeSpatialCase)
        old = build(tmp_path / "old", inputs, condition=None, module=old_factory)
    assert current[0].source_hash == old[0].source_hash
    assert current[1:] == old[1:]
    a, b = native.NativeSpatialTask(current[0], max_steps=2), old_native.NativeSpatialTask(old[0], max_steps=2)
    assert a.decision_model_hash == b.decision_model_hash
    assert a.observation().fingerprint == b.observation().fingerprint
    assert a.candidate_inventory() == b.candidate_inventory()
    for filename in ("public-task-derivation.json", "admitted-public-bindings.json"):
        assert (tmp_path / "current" / filename).read_bytes() == (tmp_path / "old" / filename).read_bytes()


def test_union_preserves_raw_evidence_access_domains_and_full_denominator(tmp_path):
    inputs, (image, support, target, domain) = fixture(tmp_path)
    raw = build(tmp_path / "raw", inputs)
    union = build(tmp_path / "union", inputs, condition=UNION)
    a, b = raw[0], union[0]
    np.testing.assert_array_equal(b.occupancy_source_support, support)
    np.testing.assert_array_equal(b.observed_support, (support != 0) | (target != 0))
    np.testing.assert_array_equal(b.structural_intensity, image)
    np.testing.assert_array_equal(b.nominal_target, target)
    np.testing.assert_array_equal(b.reference_target, target)
    np.testing.assert_array_equal(b.public_target_domain, domain)
    np.testing.assert_array_equal(a.access.center_mm, b.access.center_mm)
    np.testing.assert_array_equal(a.access.normal_inward, b.access.normal_inward)
    assert a.access.radius_mm == b.access.radius_mm and a.tools == b.tools
    assert (tmp_path / "raw/public-task-derivation.json").read_bytes() == (tmp_path / "union/public-task-derivation.json").read_bytes()
    assert b._supplied_goal_extent["full_region_membership_mm3"] == a._supplied_goal_extent["full_region_membership_mm3"] == 2
    assert b._supplied_goal_extent["occupancy_modified"] is True
    assert b._supplied_goal_extent["target_modified"] is False
    assert b._supplied_goal_extent["unsupported_region_positive_voxels"] == 0
    assert b._occupancy_derivation["added_region_positive_voxels"] == 1
    assert b._occupancy_derivation["added_region_hash"] == array_digest((target != 0) & (support == 0))
    assert not b.observed_support[0, 0, 0] and not b.public_target_domain[0, 0, 0]
    assert a.source_hash != b.source_hash
    assert a._normalization_record != b._normalization_record
    assert a._normalization_record["upper"] != b._normalization_record["upper"]
    assert b._normalization_record["raw_source_preserved"] is True
    assert union[1]["support_source_sha256"] == raw[1]["support_source_sha256"]
    assert union[2]["derived_occupancy_anatomically_validated"] is False
    again = replace(b, observed_support=np.logical_or(b.observed_support, b.nominal_target > 0))
    assert again.source_hash == b.source_hash
    with pytest.raises(ValueError): b.occupancy_source_support[0, 0, 0] = True


def test_union_context_admits_native_search_but_no_training(tmp_path):
    inputs, _ = fixture(tmp_path)
    case, args = contract(build(tmp_path / "union", inputs, condition=UNION), inputs)
    task, context = admission.make_patient_planning_task(case, **args)
    assert context.record()["execution_kind"] == "search_only_no_policy"
    assert context.record()["budgets"]["max_policy_forwards"] == 0
    context.require_task(task.planning_clone()); context.require_task(task.fresh())
    context.require_observations([task.observation()])
    with pytest.raises(ValueError, match="TRAIN_gradient"): context.require_training()
    assert task.step("STOP").terminated


def test_explicit_union_with_no_added_cells_reports_no_bit_change(tmp_path):
    inputs, _ = fixture(tmp_path)
    case = build(tmp_path / "union", inputs, condition=UNION)[0]
    # A different generated source S already contains this T. The operation
    # remains disclosed even though it changes no cells in that source.
    same = replace(case, occupancy_source_support=case.observed_support)
    np.testing.assert_array_equal(same.observed_support, case.observed_support)
    assert same._occupancy_derivation["added_region_positive_voxels"] == 0
    assert same._supplied_goal_extent["occupancy_modified"] is False
    assert same._supplied_goal_extent["derived_occupancy"] is True


@pytest.mark.parametrize("field", ["raw_support_array_hash", "target_domain_binary_hash"])
def test_corresponding_metadata_copies_cannot_contradict_actual_raw_arrays(tmp_path, monkeypatch, field):
    inputs, _ = fixture(tmp_path)
    case, args = contract(build(tmp_path / "union", inputs, condition=UNION), inputs)
    provenance = thaw_json(case.support_provenance); provenance[field] = "sha256:"+"0"*64
    changed = replace(case, support_provenance=provenance)
    args["source_binding"]["source_hash"] = changed.source_hash
    if field == "target_domain_binary_hash":
        args["source_binding"][field] = provenance[field]
    rebind(args)
    def forbidden(*a, **k): raise AssertionError("must refuse before native inventory")
    monkeypatch.setattr(admission, "NativeSpatialTask", forbidden)
    with pytest.raises(ValueError): admission.make_patient_planning_task(changed, **args)


@pytest.mark.parametrize("mutation", ["SELECT", "EVAL", "forwards", "updates", "derived_hash", "raw_file", "domain", "QC_truth", "condition"])
def test_union_admission_refuses_role_budget_and_provenance_tampering(tmp_path, monkeypatch, mutation):
    inputs, _ = fixture(tmp_path)
    case, args = contract(build(tmp_path / "union", inputs, condition=UNION), inputs)
    args = copy.deepcopy(args)
    if mutation in ("SELECT", "EVAL"):
        subject = "ReMIND-013" if mutation == "SELECT" else "ReMIND-067"
        args["source_binding"]["subject"] = args["qc_receipt"]["subject"] = args["protocol"]["subject"] = subject
        args["source_binding"]["patient_group"] = "ReMIND:" + subject.split("-")[1]
        args["protocol"]["role"] = "SELECT" if mutation == "SELECT" else "MEASUREMENT_EVAL"
    elif mutation == "forwards": args["protocol"]["max_policy_forwards"] = 1
    elif mutation == "updates": args["protocol"]["max_optimizer_updates"] = 1
    elif mutation == "derived_hash": args["source_binding"]["occupancy_derivation"]["added_region_hash"] = "sha256:"+"0"*64
    elif mutation == "raw_file": args["source_binding"]["support_source_sha256"] = "0"*64
    elif mutation == "domain": args["source_binding"]["target_domain_binary_hash"] = "sha256:"+"0"*64
    elif mutation == "QC_truth": args["qc_receipt"]["derived_occupancy_anatomically_validated"] = True
    else: args["protocol"]["occupancy_condition"] = RAW
    rebind(args)
    def forbidden(*a, **k): raise AssertionError("must refuse before native inventory")
    monkeypatch.setattr(admission, "NativeSpatialTask", forbidden)
    with pytest.raises(ValueError): admission.make_patient_planning_task(case, **args)


@pytest.mark.parametrize("mutation", ["addition", "subtraction", "missing_raw", "source_kind", "target_kind"])
def test_native_union_cannot_hide_mask_or_provenance_changes(tmp_path, mutation):
    inputs, _ = fixture(tmp_path)
    case = build(tmp_path / "union", inputs, condition=UNION)[0]
    kwargs = {}
    if mutation in ("addition", "subtraction"):
        mask = case.observed_support.copy()
        mask[0, 0, 0] = mutation == "addition"
        if mutation == "subtraction": mask[2, 2, 2] = False
        kwargs["observed_support"] = mask
    elif mutation == "missing_raw": kwargs["occupancy_source_support"] = None
    elif mutation == "source_kind": kwargs["support_source_kind"] = "supplied_annotation"
    else: kwargs["target_source_kind"] = "derived_from_scan"
    with pytest.raises(ValueError): replace(case, **kwargs)


@pytest.mark.parametrize("role", ["SELECT", "MEASUREMENT_EVAL"])
def test_factory_refuses_derived_other_roles_before_loading(role, tmp_path, monkeypatch):
    def forbidden(*a, **k): raise AssertionError("must refuse before any public array load")
    monkeypatch.setattr(factory, "load_public_manifest", forbidden)
    with pytest.raises(ValueError, match="TRAIN search-only"):
        factory.prepare_public_source(tmp_path, {"limits": {"max_optimizer_updates": 0, "max_policy_forwards": 0}},
            "b"*64, None, lambda *a, **k: None, public_manifest_path="unused", public_manifest_sha256="0"*64,
            cohort_bytes=b"unused", learning_protocol_hash="sha256:"+"c"*64,
            expected_role=role, occupancy_condition=UNION)


def test_exact_union_native_removal_changes_only_declared_removable_domain():
    old = native.make_native_opening_task().case
    target = np.zeros(old.observed_support.shape, np.float32); target[4, 4, 4:7] = 1
    common = dict(track="annotation_assisted", target_source_kind="supplied_annotation",
        target_derivation="generated supplied target control", nominal_target=target, reference_target=target,
        target_semantics=SUPPLIED_GOAL_REGION, proposal_mode="nominal_cavity_v1",
        proposal_config=NominalCavityProposalConfig(tuple((a, b) for a in (-1, 0, 1) for b in (-1, 0, 1))))
    raw = replace(old, support_source_kind="supplied_annotation", **common)
    union = replace(raw, observed_support=raw.observed_support | (target > 0),
        occupancy_source_support=raw.observed_support, support_source_kind=native.DERIVED_OCCUPANCY_SOURCE_KIND,
        support_derivation="generated explicit S union T material assumption")
    tasks = [native.NativeSpatialTask(case, max_steps=2) for case in (raw, union)]
    for task, last in zip(tasks, (5, 6)):
        def pick(tool, voxel):
            return next(r["action_id"] for r in task.candidate_inventory()["emitted"]
                if r["feasible"] and r["tool_id"] == tool and r["voxel"] == list(voxel))
        task.step(pick("short-wide-opener", (4, 4, 1)))
        task.step(pick("long-narrow-cutter", (4, 4, last)))
        assert not np.any(task._engine.removed_mask & ~task.case.observed_support)
        assert task.independent_geometry_check().feasible
        assert task.metrics()["supplied_goal_region"]["full_region_membership_mm3"] == 3
    assert not tasks[0]._engine.removed_mask[4, 4, 6]
    assert tasks[1]._engine.removed_mask[4, 4, 6]
    assert tasks[0].metrics()["target_removed_mm3"] == 2
    assert tasks[1].metrics()["target_removed_mm3"] == 3
