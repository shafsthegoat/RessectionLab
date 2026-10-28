"""Frozen assessment roster and pre-pixel admission tests."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "evaluate_frozen_sector_assessment_test",
    ROOT / "scripts/evaluate_frozen_sector_assessment.py")
assessment = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(assessment)


def test_fixed_fourteen_method_roster_does_not_select_by_assessment_outcome():
    # Build the exact declared six-file roster without opening model bytes.
    from prepare_frozen_ucsf_assessment import ROSTER
    checkpoints = {name: {"path": "/synthetic/" + name,
                          "fileSha256": "h", "actorSha256": "a", "stage": "bc"}
                   for name in ROSTER}
    specs = list(assessment._specs({"checkpoints": checkpoints}))
    names = [name for name, _, _ in specs]
    assert len(names) == 14
    assert names[:5] == ["script-near-left", "fixed-left-near-first",
                         "fixed-left-far-first", "fixed-right-near-first",
                         "fixed-right-far-first"]
    assert names[5:8] == ["untrained-seed17", "untrained-seed29", "untrained-seed43"]
    assert names[8:] == [name.removesuffix(".pt") for name in ROSTER]
    assert names[-5:].count("ppo-seed17") == 1


def test_executor_refuses_before_any_pixel_access_without_go(tmp_path):
    with pytest.raises(ValueError, match="PIXEL GO"):
        assessment.run(tmp_path, tmp_path / "access-receipt.json",
                       tmp_path / "output", pixel_go=False)
    assert not (tmp_path / "output").exists()


def test_source_patient_and_revision_are_bound_to_protocol():
    from prepare_frozen_ucsf_assessment import REVISION

    package_root = ROOT / "testcases"
    source = assessment.output_root(package_root) / assessment.CASE / "frozen-access-comparison.ressectionlab"
    case = SimpleNamespace(metadata={"surgical_adapter": {
        "sourcePatient": "UCSF-PDGM-0009", "sourceManifest": {
            "patient_id": "UCSF-PDGM-0009",
            "series": [{"source_mirror_revision": "wrong"}]}}})
    artifacts = {"surgical_toolbox": {"planning": {"task": "anatomy-access"}}}
    with pytest.raises(ValueError, match="patient, revision"):
        assessment._source_identity(case, artifacts, {"mirrorRevision": REVISION},
                                    source, package_root)


def _fake_preparation(tmp_path, monkeypatch, calls, seconds):
    monkeypatch.setattr(assessment, "ROOT", tmp_path)
    package_root = tmp_path / "testcases"
    package_root.mkdir()
    from prepare_frozen_ucsf_assessment import freeze_path, protocol_path
    protocol_file = protocol_path(package_root)
    protocol_file.parent.mkdir(parents=True)
    protocol_file.write_text("synthetic protocol")
    freeze_file = freeze_path(package_root)
    freeze_file.write_text("synthetic freeze")
    source = package_root / "synthetic-access.ressectionlab"
    source.write_bytes(b"synthetic metadata only, no patient arrays")
    preparation_path = package_root / "access-receipt.json"
    selection = package_root / "source-only-selection.json"
    selection.write_text("synthetic selection")
    preparation = {"status": "saved", "sourceRole": "FROZEN_ASSESSMENT",
        "patientId": assessment.CASE,
        "protocolSha256": assessment.digest(protocol_file),
        "freezeSha256": assessment.digest(freeze_file),
        "sourceOnlySelectionFileSha256": assessment.digest(selection),
        "output": str(source), "outputSha256": assessment.digest(source),
        "assessmentRouteSupported": True,
        "nativeCalls": {"physicalCalls": calls, "wallSeconds": seconds}}
    preparation_path.write_text(json.dumps(preparation))
    protocol = {"nativeBudget": {
        "assessmentAggregateMaxPhysicalCalls": 10000,
        "assessmentAggregateMaxWallSeconds": 900}}
    monkeypatch.setattr(assessment, "validate_freeze", lambda *_: (protocol, {}))
    monkeypatch.setattr(assessment, "validate_acquired_role", lambda *_: ({}, {}))
    return package_root, preparation_path, package_root / "output"


@pytest.mark.parametrize("calls,seconds", [(-1, 1.), (1, -1.), (1, float("nan")),
                                            (1, float("inf")), (True, 1.)])
def test_preparation_cost_must_be_nonnegative_finite_and_bounded(tmp_path,
                                                                   monkeypatch,
                                                                   calls, seconds):
    package_root, preparation, output = _fake_preparation(
        tmp_path, monkeypatch, calls, seconds)
    with pytest.raises(ValueError, match="preparation is missing"):
        assessment.run(package_root, preparation, output, pixel_go=True)
    assert not output.exists()


def test_export_wall_overrun_cannot_be_marked_evaluated(tmp_path, monkeypatch):
    package_root, preparation, output = _fake_preparation(
        tmp_path, monkeypatch, 0, 0.)
    case = SimpleNamespace(case_id="synthetic-assessment")
    monkeypatch.setattr(assessment, "load_workspace", lambda *_: (case, {}, None))
    monkeypatch.setattr(assessment, "_source_identity", lambda *_: None)
    monkeypatch.setattr(assessment, "prepare_case", lambda *_ , **__: SimpleNamespace(
        case_id=case.case_id))
    monkeypatch.setattr(assessment, "_specs", lambda *_: [
        (f"method-{i}", lambda: lambda _: None, None) for i in range(14)])
    monkeypatch.setattr(assessment, "rollout", lambda *_ , **__: {
        "report": {}, "engine": object()})
    monkeypatch.setattr(assessment, "_export", lambda *_: {"reopenedExact": True})
    monkeypatch.setattr(assessment.NativeCallCounter, "check_time",
        lambda self: (_ for _ in ()).throw(TimeoutError("export exceeded wall")))
    result = assessment.run(package_root, preparation, output, pixel_go=True)
    assert result["status"] == "completed_with_failures"
    assert len(result["executions"]) == 14
    assert all(row["status"] == "failed" and
               "export exceeded wall" in row["failure"]
               for row in result["executions"])

