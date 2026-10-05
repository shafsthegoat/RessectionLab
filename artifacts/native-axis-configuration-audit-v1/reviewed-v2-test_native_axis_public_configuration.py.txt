"""Actual source-bundle configuration gate; no simulator, engine or patient steps.

The public bundle is deliberately not checked into Git. Archive validation may
set RESECTIONLAB_TEST_CASE_BUNDLE to the already verified local derivative.
"""
import copy
from dataclasses import fields, replace
import hashlib
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import preflight_native_axis as runner
from resectionlab.geometry import AccessWindow
from resectionlab.imaging import load_case
from resectionlab.native_resection import NativeResectionConfig, NativeResectionEngine, native_config_from_case
import resectionlab.native_simulation as fixed

BUNDLE = Path(os.environ.get("RESECTIONLAB_TEST_CASE_BUNDLE", ROOT / "outputs/cases/UCSF-PDGM-0004.ressectionlab"))


@pytest.fixture(scope="module")
def actual_configurations():
    if not BUNDLE.is_file():
        pytest.skip("Verified local UCSF bundle absent; run the required public configuration gate with RESECTIONLAB_TEST_CASE_BUNDLE")
    reference = runner._checked_manifest(ROOT / runner.REFERENCE_PATH, runner.REFERENCE_HASH)
    declaration = runner._checked_manifest(ROOT / runner.DECLARATION_PATH, runner.DECLARATION_HASH)
    target = reference["target"]
    assert hashlib.sha256(BUNDLE.read_bytes()).hexdigest() == target["bundle_sha256"]
    case = load_case(BUNDLE)
    assert (case.semantic_hash, case.planning_hash) == (target["semantic_hash"], target["planning_hash"])
    with patch.object(NativeResectionEngine, "__init__", side_effect=AssertionError("No patient engine in config-only gate")):
        direct = native_config_from_case(case, access=AccessWindow(**target["access"]))
        with patch.object(fixed, "NativeSequentialSimulator", side_effect=lambda cfg, *a, **k: cfg) as capture:
            prototype = fixed.make_native_patient_simulator(case, candidate_count=4, max_steps=3, max_actions=7)
    assert capture.call_count == 1 and isinstance(prototype, NativeResectionConfig)
    return direct, prototype, declaration, reference


def test_real_bundle_has_one_exact_provenance_difference_and_no_physical_difference(actual_configurations):
    direct, prototype, declaration, reference = actual_configurations
    assert prototype.fingerprint == reference["target"]["native_config_hash"]
    assert direct.fingerprint == declaration["native_configuration"]["fingerprint"] != prototype.fingerprint
    old = runner.native_configuration_record(prototype)
    new = runner.native_configuration_record(direct)
    assert runner.content_hash(old) == runner.content_hash(declaration["historical_native_configuration"])
    differing = [name for name in new["components"]
                 if runner.content_hash(new["components"][name]) != runner.content_hash(old["components"][name])]
    assert differing == ["tissue_support_provenance"]
    for name in ("tissue_mask", "target_labels", "affine", "hard_exclusion"):
        before, after = getattr(prototype, name), getattr(direct, name)
        assert before.dtype == after.dtype and before.shape == after.shape
        np.testing.assert_array_equal(before, after)
    receipt = runner.assert_declared_native_configuration(direct, declaration, reference)
    assert receipt["physical_component_equality"] and receipt["different_components"] == differing


@pytest.mark.parametrize("component", [field.name for field in fields(NativeResectionConfig) if not field.name.startswith("_")])
def test_every_declared_config_component_is_enforced(actual_configurations, component):
    direct, _, declaration, reference = actual_configurations
    altered = copy.deepcopy(declaration)
    altered["native_configuration"]["components"][component] = {"changed": True}
    with pytest.raises(ValueError, match="complete V2 declaration"):
        runner.assert_declared_native_configuration(direct, altered, reference)


def test_same_claimed_fingerprint_cannot_hide_changed_source_cells(actual_configurations):
    direct, _, declaration, reference = actual_configurations
    labels = direct.target_labels.copy()
    source_cell = tuple(np.argwhere(labels > 0)[0])
    labels[source_cell] = 0
    altered = replace(direct, target_labels=labels)
    object.__setattr__(altered, "_fingerprint", direct.fingerprint)  # Deliberate tampering attack only.
    with pytest.raises(ValueError, match="complete V2 declaration"):
        runner.assert_declared_native_configuration(altered, declaration, reference)


def test_v2_snapshot_contains_actual_runner_and_declaration_and_refuses_edits(tmp_path):
    for name in ("scripts/preflight_native_axis.py", str(runner.DECLARATION_PATH), str(runner.REFERENCE_PATH)):
        destination = tmp_path / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / name).read_bytes())
    snapshot = runner.source_snapshot(tmp_path)
    assert "manifests/experiments/native-axis-preflight-v2.json" in snapshot["file_sha256"]
    assert "scripts/preflight_native_axis.py" in snapshot["file_sha256"]
    (tmp_path / runner.DECLARATION_PATH).write_text("{}\n")
    with pytest.raises(RuntimeError, match="SOURCE_CHANGED"):
        runner.assert_source(snapshot, tmp_path)


def test_original_v1_declaration_remains_exact():
    value = json.loads((ROOT / "manifests/experiments/native-axis-preflight-v1.json").read_text())
    declared = value.pop("declaration_content_hash")
    assert declared == "sha256:ae12468ad89898ba997cea2d4576fd65ecff9ea49a8fcc35f6d3e7850255d341"
    assert runner.content_hash(value) == declared
