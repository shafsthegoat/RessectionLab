"""Generated v2 support/abstention tests; no real-image or ANTs fitting."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module

v2 = load("ixi_atropos_positive_domain", ROOT/"scripts/poc_ixi265_atropos_positive_domain.py")
controls = load("immutable_v1_generated_controls", ROOT/"tests/test_poc_ixi265_atropos.py")
UNKNOWN = ((1,1,1), (1,3,2), (2,4,3), (3,5,4), (4,2,5), (5,6,2), (6,4,1), (7,5,3))


def generated():
    values, brain, labels, affine = controls.generated()
    for index in UNKNOWN: values[index] = 0
    return values, brain, labels, affine


def test_support_exact_partition_and_unknowns_preserve_originals():
    values, brain, _, affine = generated()
    originals = [array.copy() for array in (values, brain, affine)]
    domain, unknown, receipt = v2.positive_domain(values, brain, expected_unknown=UNKNOWN)
    expected = brain.copy()
    for index in UNKNOWN: expected[index] = False
    assert np.array_equal(domain, expected)
    assert np.array_equal(domain | unknown, brain) and not np.any(domain & unknown)
    assert receipt["excluded_unknown_voxels"] == 8
    assert receipt["fit_domain_voxels"] + 8 == receipt["original_brain_voxels"]
    assert receipt["excluded_semantics"] == "UNKNOWN_ABSTAINED"
    for array, original in zip((values, brain, affine), originals): assert np.array_equal(array, original)
    assert not np.shares_memory(domain, brain) and not np.shares_memory(unknown, brain)


def test_raw_classes_and_posteriors_abstain_only_at_declared_cells_inside_original_brain():
    values, brain, _, affine = generated()
    domain, unknown, _ = v2.positive_domain(values, brain, expected_unknown=UNKNOWN)
    _, labels, posteriors, _ = v2.reference().fit_arrays(values, domain, affine, controls.GeneratedBackend())
    v2.require_unknown_partition(brain, domain, unknown, labels, posteriors)
    assert set(map(tuple, np.argwhere(brain & (labels == 0)))) == set(UNKNOWN)
    assert np.all(labels[domain] > 0) and all(np.all(p[unknown] == 0) for p in posteriors)
    for code, p in enumerate(posteriors, 1):
        assert np.array_equal(p, np.where(domain, np.where(labels == code, .8, .1), 0).astype(np.float32))


@pytest.mark.parametrize("change", ["extra_zero", "missing_zero", "moved_zero", "negative", "nonfinite", "brain_excludes_declared_cell"])
def test_changed_missing_intensity_set_or_bad_source_is_refused(change):
    values, brain, _, _ = generated()
    if change == "extra_zero": values[8,6,5] = 0
    elif change == "missing_zero": values[UNKNOWN[0]] = 1
    elif change == "moved_zero": values[UNKNOWN[0]] = 1; values[8,6,5] = 0
    elif change == "negative": values[UNKNOWN[0]] = -1
    elif change == "nonfinite": values[UNKNOWN[0]] = float("nan")
    else: brain[UNKNOWN[0]] = False
    with pytest.raises(ValueError): v2.positive_domain(values, brain, expected_unknown=UNKNOWN)


@pytest.mark.parametrize("fault", ["unknown_assigned_CSF", "extra_abstention", "unknown_posterior", "partition_overlap"])
def test_unknown_cells_cannot_be_relabelled_or_concealed(fault):
    values, brain, _, affine = generated()
    domain, unknown, _ = v2.positive_domain(values, brain, expected_unknown=UNKNOWN)
    _, labels, posteriors, _ = v2.reference().fit_arrays(values, domain, affine, controls.GeneratedBackend())
    if fault == "unknown_assigned_CSF": labels[UNKNOWN[0]] = 1
    elif fault == "extra_abstention": labels[8,6,5] = 0
    elif fault == "unknown_posterior": posteriors[0][UNKNOWN[0]] = 1
    else: domain[UNKNOWN[0]] = True
    with pytest.raises(ValueError): v2.require_unknown_partition(brain, domain, unknown, labels, posteriors)


def test_v2_release_is_required_before_any_files_or_source_access(monkeypatch):
    monkeypatch.setattr(v2, "reference", lambda: pytest.fail("Unreleased attempt accessed shared code or data"))
    with pytest.raises(ValueError, match="not released"):
        v2.run({}, {}, "/unused/runtime", "/unused/output")


def test_v1_pin_separate_marker_parameters_and_no_acceptance():
    method = v2.method(); base = v2.reference()
    assert v2.sha(v2.V1_PATH) == "4a5c5b4b2f85153212ac27e6356d5f1dca5c8b42d77257fb55a2fee53253624b"
    assert v2.MARKER != "ixi265-n4-atropos-single-attempt.json"
    assert method["parameters"] == base.plan()["parameters"]
    assert method["resources"] == base.plan()["resources"]
    assert not method["anatomy_accepted"] and not method["host_accepted"] and not method["growth_use_accepted"]
    assert not method["brain_mask_replaced"] and not method["intensity_offset_or_clipping"]


def test_independent_diagnostic_lineage_matches_frozen_eight_cells_without_image_decode():
    lineage = v2.diagnostic_lineage()
    assert lineage["independent_diagnostic"]["sha256"] == v2.DIAGNOSTIC_SHA
    assert lineage["unknown_indices_native"] == [list(index) for index in v2.EXPECTED_UNKNOWN]


def test_foreign_diagnostic_identity_refused_before_referenced_payload(tmp_path, monkeypatch):
    path = tmp_path/"diagnostic-lineage.json"
    document = {"unknown_indices_native": [list(p) for p in v2.EXPECTED_UNKNOWN],
        "independent_diagnostic": {"path": "protected/never-open", "sha256": "0"*64}}
    path.write_text(json.dumps(document)); monkeypatch.setattr(v2, "PREPARATION", tmp_path)
    monkeypatch.setattr(v2, "sha", lambda _: pytest.fail("Foreign diagnostic reached payload hashing"))
    with pytest.raises(ValueError, match="diagnostic identity"): v2.diagnostic_lineage()


def test_foreign_extra_lineage_path_refused_before_any_payload_hash(tmp_path, monkeypatch):
    document = {"unknown_indices_native": [list(p) for p in v2.EXPECTED_UNKNOWN],
        "independent_diagnostic": {"path": v2.DIAGNOSTIC_PATH, "sha256": v2.DIAGNOSTIC_SHA},
        "records": [{"path": "protected/never-open", "sha256": "0"*64}]}
    (tmp_path/"diagnostic-lineage.json").write_text(json.dumps(document))
    monkeypatch.setattr(v2, "PREPARATION", tmp_path)
    monkeypatch.setattr(v2, "sha", lambda _: pytest.fail("Foreign lineage reached payload hashing"))
    with pytest.raises(ValueError, match="metadata paths"): v2.diagnostic_lineage()
