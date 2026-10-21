"""Independent prospective v2 audit: metadata and tiny generated arrays only.

Never opens real images/bundles or imports ANTs/model frameworks. Never starts
a worker. The frozen output is separate from the failed v1 and source QA.
"""
from __future__ import annotations

import ast
import builtins
from datetime import datetime, timezone
import hashlib
import importlib.abc
import importlib.util
import io
import itertools
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PINS = {
    "scripts/poc_ixi265_atropos_positive_domain.py": "0652eca1ba81304e053f24c9a9fc2f8c89174bfe7447accfd21105263592d81d",
    "tests/test_poc_ixi265_atropos_positive_domain.py": "8568e4455190c5735263b56541c8893e060111401cc527dd1a9a1948cf40a318",
    "artifacts/poc-ixi265-atropos-positive-domain-preparation-v2/preparation.json": "2c1d39e4a318598385dfe331de08e1f6de3e2dba5850a8b0acd8978045376c3d",
    "scripts/poc_ixi265_atropos.py": "4a5c5b4b2f85153212ac27e6356d5f1dca5c8b42d77257fb55a2fee53253624b",
    "artifacts/ixi265-quality-independent-v1/independent-qa.json": "7a51ca88a898b11eb7fc132180d5f77da4a0bffaf49ceb6e5b8eb229d6d162a6",
    "artifacts/poc-ixi265-atropos-run-v1/outcome.json": "55f4dd46f7cc692d4a66799cefca21bfa37430d3f5cf2ebe3eda675b6e4c989e",
}
UNKNOWN = ((1, 1, 1), (1, 2, 3), (2, 2, 2), (2, 3, 1), (3, 2, 1), (3, 4, 2), (4, 4, 3), (5, 3, 2))
CHECKS = []


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def mark(name):
    CHECKS.append(name)


def refuses(callback, name):
    try:
        callback()
    except (ValueError, TypeError, KeyError):
        mark(name)
    else:
        raise AssertionError("Expected refusal: " + name)


def forbidden(*args, **kwargs):
    raise AssertionError("Forbidden patient/model/worker access")


class NoModels(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"ants", "torch", "nibabel", "totalsegmentator", "monai"}:
            forbidden()


def open_guard(original):
    def guarded(file, *args, **kwargs):
        if isinstance(file, (str, bytes, Path)) and str(file).lower().endswith((".nii", ".nii.gz", ".ressectionlab")):
            forbidden()
        return original(file, *args, **kwargs)
    return guarded


def load_v2():
    spec = importlib.util.spec_from_file_location("independent_positive_domain_review", ROOT / "scripts/poc_ixi265_atropos_positive_domain.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tiny_arrays():
    shape = (7, 6, 5)
    brain = np.zeros(shape, bool); brain[1:-1, 1:-1, 1:-1] = True
    values = np.indices(shape)[0].astype(np.float32)*20 + 7
    values[~brain] = 0
    for p in UNKNOWN:
        values[p] = 0
    affine = np.eye(4)
    angle = 0.31
    affine[:3, :3] = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]) @ np.diag([-0.9, 1.4, 2.7])
    affine[:3, 3] = [-7, 4, 8]
    return values, brain, affine


class TransportImage:
    def __init__(self, data, **geometry):
        self.data = np.array(data, copy=True)
        self.shape = self.data.shape
        for key, value in geometry.items():
            setattr(self, key, np.array(value, copy=True))
    def numpy(self):
        return self.data.copy()
    def like(self, data):
        return TransportImage(data, origin=self.origin, spacing=self.spacing, direction=self.direction)


class TransportOnlyBackend:
    """Explicit fake transport fixture; no numerical fitting claim."""
    def __init__(self):
        self.calls = []
        self.raw_probabilities = None
    def from_numpy(self, data, **geometry):
        return TransportImage(data, **geometry)
    def n4_bias_field_correction(self, image, *, mask, **parameters):
        self.calls.append(("N4", parameters, mask.data.copy()))
        return image.like(image.data)
    def atropos(self, *, a, x, **parameters):
        self.calls.append(("Atropos", parameters, x.data.copy()))
        labels = np.select((a.data < 45, a.data < 85), (1, 2), default=3).astype(np.uint8)
        labels[x.data == 0] = 0
        probabilities = [np.where(x.data, np.where(labels == code, .7, .15), 0).astype(np.float32) for code in (1, 2, 3)]
        self.raw_probabilities = [p.copy() for p in probabilities]
        return {"segmentation": a.like(labels), "probabilityimages": [a.like(p) for p in probabilities]}


def numerical_controls(v2, base):
    values, brain, affine = tiny_arrays()
    originals = [x.copy() for x in (values, brain, affine)]
    values.flags.writeable = brain.flags.writeable = affine.flags.writeable = False
    domain, unknown, receipt = v2.positive_domain(values, brain, expected_unknown=UNKNOWN)
    assert np.array_equal(domain | unknown, brain) and not (domain & unknown).any()
    assert np.array_equal(domain, brain & (values > 0)) and set(map(tuple, np.argwhere(unknown))) == set(UNKNOWN)
    assert receipt["excluded_unknown_voxels"] == 8 and receipt["fit_domain_voxels"] + 8 == int(brain.sum())
    assert not any(np.shares_memory(a, b) for a in (domain, unknown) for b in (values, brain))
    mark("eight_unknowns_exact_disjoint_complete_partition_with_readonly_source")
    backend = TransportOnlyBackend()
    corrected, labels, posteriors, audit = base.fit_arrays(values, domain, affine, backend)
    v2.require_unknown_partition(brain, domain, unknown, labels, posteriors)
    assert np.array_equal(corrected, values)
    assert all(np.array_equal(a, b) for a, b in zip(posteriors, backend.raw_probabilities))
    assert all(not p[unknown].any() for p in posteriors)
    assert np.array_equal((labels == 0) & brain, unknown)
    assert all(np.array_equal(a, b) for a, b in zip((values, brain, affine), originals))
    assert [c[0] for c in backend.calls] == ["N4", "Atropos"]
    assert backend.calls[0][1] == base.N4_PARAMETERS and backend.calls[1][1] == base.ATROPOS_PARAMETERS
    assert all(np.array_equal(c[2], domain) for c in backend.calls)
    assert audit["classes"]["probabilities_modified"] is False
    mark("fake_transport_retains_native_source_raw_classes_posteriors_and_unknown_abstention")
    geometry = base.ras_to_lps_geometry(affine)
    for p in itertools.product(*[(-.5, n-.5) for n in values.shape]):
        actual = geometry["origin"] + geometry["direction"] @ (geometry["spacing"] * np.asarray(p))
        expected = (affine @ np.r_[p, 1])[:3] * [-1, -1, 1]
        assert np.allclose(actual, expected, rtol=0, atol=1e-13)
    mark("independent_reflected_oblique_eight_cell_corners_transport")
    for fault in ("extra_zero", "missing_zero", "negative", "nan", "brain_excludes_unknown", "unsigned_brain"):
        x, b = values.copy(), brain.copy()
        if fault == "extra_zero": x[4, 2, 2] = 0
        elif fault == "missing_zero": x[UNKNOWN[0]] = 1
        elif fault == "negative": x[0, 0, 0] = -1
        elif fault == "nan": x[0, 0, 0] = np.nan
        elif fault == "brain_excludes_unknown": b[UNKNOWN[0]] = False
        else: b = b.astype(np.uint8)
        refuses(lambda: v2.positive_domain(x, b, expected_unknown=UNKNOWN), "refuse_" + fault)
    for fault in ("CSF_at_unknown", "posterior_at_unknown", "extra_abstention", "overlapping_partition", "missing_partition_cell"):
        l, pp, d, u = labels.copy(), [p.copy() for p in posteriors], domain.copy(), unknown.copy()
        if fault == "CSF_at_unknown": l[UNKNOWN[0]] = 1
        elif fault == "posterior_at_unknown": pp[0][UNKNOWN[0]] = .01
        elif fault == "extra_abstention": l[4, 2, 2] = 0
        elif fault == "overlapping_partition": d[UNKNOWN[0]] = True
        else: u[UNKNOWN[0]] = False
        refuses(lambda: v2.require_unknown_partition(brain, d, u, l, pp), "refuse_" + fault)
    shifted = {**geometry, "origin": np.asarray(geometry["origin"]) + [.001, 0, 0]}
    refuses(lambda: base.require_native_frame(values.shape, affine, values.shape, **shifted), "refuse_changed_physical_origin")
    refuses(lambda: base.require_native_frame(values.shape, affine, (5, 6, 7), **geometry), "refuse_axis_permutation")


def guard_controls(v2, base):
    admitted = {"patient_group": "IXI:265", "role": "TRAIN", "training_admitted": False,
        "T1_sha256": base.T1_SHA, "brain_mask_sha256": base.MASK_SHA, "host_sha256": base.HOST_SHA}
    with patch.object(v2, "reference", forbidden):
        refuses(lambda: v2.run({}, {}, "/never/runtime", "/never/output"), "unreleased_v2_stops_before_code_source_runtime")
    with tempfile.TemporaryDirectory(prefix="ixi-v2-independent-generated-") as tmp:
        tmp = Path(tmp)
        with patch.object(v2, "reference", lambda: base), patch.object(v2, "preparation_intact", forbidden), patch.object(base, "digest", forbidden):
            alterations = ({"patient_group": "UPENN:999"}, {"role": "SELECT"}, {"role": "EVAL"},
                {"nested": {"role": "validation"}}, {"roles": {"SELECT": True}}, {"training_admitted": True},
                {"training_admitted": 0}, {"T1_sha256": "0"*64})
            for index, change in enumerate(alterations):
                metadata = {**admitted, **change}
                spec = tmp/"worker-generated.json"
                spec.write_text(json.dumps({"released_single_attempt": True, "admission": metadata, "paths": {}}))
                refuses(lambda: v2.worker(spec), "worker_bad_identity_before_preparation_or_payload_" + str(index))
                refuses(lambda: v2.run({}, metadata, tmp/"absent-runtime", tmp/"absent-output", released_single_attempt=True),
                        "supervisor_bad_identity_before_preparation_or_payload_" + str(index))
        canonical = {"T1": tmp/"outputs/poc-ixi265/originals/IXI265-T1.nii.gz",
            "brain_mask": tmp/"outputs/poc-ixi265/synthstrip/brain_mask.nii.gz",
            "host_bundle": tmp/"outputs/poc-ixi265/IXI265-Guys-0845.ressectionlab"}
        with patch.object(base, "ROOT", tmp), patch.object(base, "digest", forbidden):
            for key in canonical:
                paths = {**canonical, key: tmp/"foreign-protected-never-open.nii.gz"}
                refuses(lambda: base.bound_inputs(paths, admitted), "all_path_validation_precedes_first_hash_" + key)
            canonical["brain_mask"].parent.mkdir(parents=True)
            canonical["brain_mask"].symlink_to(tmp/"foreign-never-open.nii.gz")
            refuses(lambda: base.bound_inputs(canonical, admitted), "symlink_source_refused_before_hash")
    # Resource accounting is generated arithmetic, with no spawned process.
    class Process:
        def __init__(self, pid): self.pid = pid
        def memory_info(self): return SimpleNamespace(rss={11: 101, 22: 203, 33: 307}[self.pid])
        def is_running(self): return True
        def children(self, recursive=False):
            assert recursive is True
            return [Process(33)]
    with patch.object(base.os, "getpid", lambda: 11):
        assert base.sampled_rss(SimpleNamespace(Process=Process), 22) == 101+203+307
    mark("resource_sum_includes_supervisor_worker_and_recursive_descendant")
    standard = {"rss": base.RSS_LIMIT, "available": base.MEMORY_FLOOR, "elapsed": base.TIME_LIMIT,
        "output_bytes": base.OUTPUT_LIMIT, "free_disk": base.DISK_FLOOR}
    assert base.resource_violation(**standard) is None
    for name, delta in (("rss", 1), ("available", -1), ("elapsed", .01), ("output_bytes", 1), ("free_disk", -1)):
        assert base.resource_violation(**{**standard, name: standard[name]+delta}) is not None
        mark("resource_limit_refuses_" + name)


def main():
    destination = ROOT/"artifacts/ixi265-positive-domain-independent-v2/prospective-qa.json"
    assert not destination.exists(), "Existing audit is immutable"
    finder = NoModels(); sys.meta_path.insert(0, finder)
    with patch("builtins.open", open_guard(builtins.open)), patch("io.open", open_guard(io.open)):
        before = {name: {"sha256": sha(ROOT/name), "mtime_ns": (ROOT/name).stat().st_mtime_ns} for name in PINS}
        assert {k: v["sha256"] for k, v in before.items()} == PINS
        preparation = json.loads((ROOT/"artifacts/poc-ixi265-atropos-positive-domain-preparation-v2/preparation.json").read_text())
        v2 = load_v2(); base = v2.reference()
        intact = v2.preparation_intact()
        assert intact == preparation
        for row in preparation["frozen_files"]:
            assert not row["path"].endswith((".nii", ".nii.gz", ".ressectionlab"))
            assert sha(ROOT/row["path"]) == row["sha256"]
        method = v2.method()
        assert json.loads(json.dumps(method)) == preparation["method"]
        assert method["expected_fit_domain_voxels"] == 1451501
        assert not any(method[k] for k in ("brain_mask_replaced", "intensity_offset_or_clipping", "source_crop", "anatomy_accepted", "host_accepted", "growth_use_accepted", "training_admitted"))
        assert method["parameters"] == base.plan()["parameters"] and method["resources"] == base.plan()["resources"]
        assert method["resources"]["rss_limit_bytes"] == 3*1024**3 and method["resources"]["threads"] == 2
        assert method["resources"]["attempts"] == 1
        assert "not_physical_partial_volume" in " ".join(method["limitations"])
        assert "does_not_identify_ventricles" in " ".join(method["limitations"])
        spec = json.loads((v2.PREPARATION/"prospective-run-spec.json").read_text())
        assert spec["released_single_attempt"] is False
        assert v2.MARKER != "ixi265-n4-atropos-single-attempt.json"
        old_marker = v2.RUNTIME/"ixi265-n4-atropos-single-attempt.json"
        old_marker_sha = sha(old_marker)
        assert old_marker_sha == preparation["historical_v1"]["marker_sha256"]
        assert not (v2.RUNTIME/v2.MARKER).exists()
        assert not Path(spec["output"]).exists()
        mark("pinned_lineage_parameters_roles_no_acceptance_separate_unreleased_attempt")
        numerical_controls(v2, base)
        guard_controls(v2, base)
        tree = ast.parse((ROOT/"scripts/poc_ixi265_atropos_positive_domain.py").read_text())
        run = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run")
        source = ast.unparse(run)
        assert "base.sampled_rss(psutil, process.pid)" in source and "base.resource_violation" in source
        assert "os.killpg(process.pid, signal.SIGTERM)" in source and "os.killpg(process.pid, signal.SIGKILL)" in source
        assert "start_new_session=True" in source and "process.wait" in source
        assert "(runtime / MARKER).open('x')" in source and "output.exists()" in source
        mark("supervisor_samples_family_caps_terminates_group_and_uses_exclusive_new_marker_output")
        assert before == {name: {"sha256": sha(ROOT/name), "mtime_ns": (ROOT/name).stat().st_mtime_ns} for name in PINS}
        assert sha(old_marker) == old_marker_sha
        assert not any(n.split(".")[0] in {"ants", "torch", "nibabel", "totalsegmentator", "monai"} for n in sys.modules)
        report = {"schema": "ixi265-positive-domain-independent-prospective-audit-v2", "status": "pass_preparation_only_fit_unreleased",
            "created_utc": datetime.now(timezone.utc).isoformat(), "pinned_inputs_unchanged": before,
            "checks": CHECKS, "check_count": len(CHECKS), "findings_requiring_fix": [],
            "declared_real_domain": {"original_estimated_brain_voxels": 1451509, "fitting_voxels": 1451501,
                "unknown_abstained_voxels": 8, "unknown_indices_native": [list(p) for p in v2.EXPECTED_UNKNOWN],
                "diagnostic_lineage_sha256": v2.DIAGNOSTIC_SHA},
            "original_brain_and_acquired_intensity_preservation": True,
            "raw_class_probabilities_are_not_PVE": True, "CSF_like_class_is_not_ventricles": True,
            "anatomical_host_growth_or_training_acceptance": False,
            "resource_policy": method["resources"], "sampled_RSS_includes_supervisor": True,
            "historical_v1": {"marker_sha256_unchanged": old_marker_sha, "started": 1, "N4_completed": 0, "Atropos_completed": 0, "accepted": 0},
            "v2_real_attempts_started": 0, "real_patient_image_or_bundle_open": False,
            "ANTs_or_model_import": False, "new_assets_or_downloads": False,
            "test_backend": "Independent generated transport fixture only; no N4/Atropos numerical accuracy claim",
            "auditor_sha256": sha(__file__),
            "limitations": ["This prospective review does not release fitting or establish a successful real candidate.",
                "The mask remains an unaccepted model estimate. Removing eight missing-intensity cells from fitting support does not correct or validate anatomy.",
                "Real output class ordering, posterior normalization, abstention maps, native frames, source fixity and resource use require a separate post-fit audit.",
                "The3GiB family RSS cap is sampled every0.2seconds and can miss between-sample peaks.",
                "A future v2 attempt must retain the historical failed v1 denominator separately; there is no automatic retry."]}
        content = json.dumps(report, indent=2, allow_nan=False)+"\n"
        assert "/Users/" not in content
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("x") as out: out.write(content)
        print(json.dumps({"status": report["status"], "checks": len(CHECKS), "artifact": str(destination.relative_to(ROOT)), "sha256": sha(destination)}))


if __name__ == "__main__":
    main()
