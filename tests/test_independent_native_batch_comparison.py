"""Static/mocked prospective runner gates; no native scene or history executed."""
from copy import deepcopy
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("native_batch_comparison", ROOT / "scripts/compare_independent_native_batch.py")
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


def test_declaration_pins_one_previously_certified_history_and_sources():
    spec = probe.declaration()
    probe.verify_declared_files(spec)
    row = probe.reference_row(spec)
    assert row["row_id"] == "tilted_20_degrees-0.125mm"
    assert spec["expected_microsteps"] == 97
    assert spec["phase_order"] == ["scalar_before", "batch", "scalar_after"]
    assert spec["max_native_strokes"] == spec["max_histories"] == 1
    assert spec["human_patients"] == spec["training_updates"] == 0
    assert spec["caps"]["cooperative_seconds"] == 140
    assert spec["caps"]["parent_wall_seconds"] == 150
    assert spec["caps"]["worker_rss_bytes"] == 3 * 1024**3
    assert spec["cpu_workers"] == spec["numerical_threads"] == 1
    receipt = probe.source_receipt(spec)
    assert "src/resectionlab/evaluation.py" in receipt
    assert "src/resectionlab/independent_geometry_batch.py" in receipt


def test_v2_changes_only_certificate_representation_and_harness_provenance():
    old = json.loads((ROOT / "manifests/experiments/independent-native-batch-history-v1.json").read_text())
    current = probe.declaration()
    assert current["version"] == "independent-native-batch-history-v2"
    assert current["harness_revision"]["previous_execution_status"] == "failed_or_incomplete"
    assert "no numerical tolerance" in current["equality"]
    for key, value in old.items():
        if key not in {"version", "equality", "fixed_files_sha256"}:
            assert current[key] == value
    for name, value in old["fixed_files_sha256"].items():
        assert current["fixed_files_sha256"][name] == value
    added = current["fixed_files_sha256"].keys() - old["fixed_files_sha256"].keys()
    assert added == {
        "manifests/experiments/independent-native-batch-history-v1.json",
        "artifacts/independent-native-batch-history-v1/attempt-01/report.json",
        "artifacts/independent-native-batch-history-v1/attempt-01/launcher.json",
        "artifacts/independent-native-batch-history-v1/failure-diagnostic.json",
        "artifacts/independent-native-batch-history-v1/artifact-index.json",
    }


def test_changed_declared_source_refused_before_runtime():
    spec = probe.declaration()
    spec["fixed_files_sha256"]["src/resectionlab/evaluation.py"] = "0" * 64
    with pytest.raises(ValueError, match="source changed"):
        probe.verify_declared_files(spec)


@pytest.mark.parametrize("key", ["expected_source_hash", "expected_config_hash", "expected_history_hash", "expected_microsteps"])
def test_historical_identity_mismatch_refused(key):
    spec = probe.declaration()
    spec[key] = "wrong"
    with pytest.raises(ValueError, match="Historical row identity"):
        probe.reference_row(spec)


@pytest.mark.parametrize("mismatch", ["source", "config", "history", "microsteps", "two_strokes"])
def test_constructed_identity_must_match_before_timing(mismatch):
    spec = probe.declaration()
    case = SimpleNamespace(semantic_hash=spec["expected_source_hash"])
    config = SimpleNamespace(fingerprint=spec["expected_config_hash"])
    fixture = SimpleNamespace(digest=lambda value: spec["expected_history_hash"])
    history = [{"microsteps": [None] * spec["expected_microsteps"]}]
    probe.verify_constructed_binding(spec, case, config, history, fixture)
    if mismatch == "source": case.semantic_hash = "wrong"
    elif mismatch == "config": config.fingerprint = "wrong"
    elif mismatch == "history": fixture.digest = lambda value: "wrong"
    elif mismatch == "microsteps": history[0]["microsteps"].pop()
    else: history.append({"microsteps": []})
    with pytest.raises(ValueError, match="Reconstructed"):
        probe.verify_constructed_binding(spec, case, config, history, fixture)


def fake_evaluation():
    return SimpleNamespace(_native_active_contacts=lambda *a, **k: set(),
        _cell_collision=lambda *a, **k: None, _extend_independent_free_space=lambda *a, **k: None)


@pytest.mark.parametrize("failure", ["body", "reentry", "conflict"])
def test_observational_hooks_restore_on_every_failure(failure):
    evaluation = fake_evaluation()
    before = dict(vars(evaluation))
    with pytest.raises(RuntimeError):
        with probe.trace_prefixes(evaluation) as record:
            if failure == "body": raise RuntimeError("body failed")
            if failure == "reentry":
                with probe.trace_prefixes(evaluation): pass
            else: evaluation._cell_collision = lambda *a, **k: (1, 2, 3)
    assert record["hooks_restored"]
    assert vars(evaluation) == before
    assert not probe._TRACING


def test_scientific_trace_excludes_only_explicit_instrumentation_metadata():
    trace = {"contacts": [{"indices": [[1, 2, 3]]}], "cell_queries": [], "prefixes": [],
             "trace_seconds": 2., "batch_cell_queries": 4, "hooks_restored": True}
    original = probe.digest(probe.scientific_trace(trace))
    trace.update(trace_seconds=7., batch_cell_queries=8)
    assert probe.digest(probe.scientific_trace(trace)) == original
    trace["contacts"][0]["indices"][0][0] = 5
    assert probe.digest(probe.scientific_trace(trace)) != original


def _saved_payloads(output):
    spec = deepcopy(probe.declaration())
    history = [{"microsteps": ["mock"]}]
    spec.update(expected_history_hash=probe.digest(history), expected_microsteps=1)
    trace = {"contacts": [{"indices": [[1, 2, 3]]}], "cell_queries": [], "prefixes": [{"remaining": "mock"}]}
    report = {"phases": [{"phase": phase, "trace_digest": probe.digest(trace),
                          "scalar_cell_queries": 0, "batch_cell_queries": 0}
                         for phase in spec["phase_order"]]}
    probe.write_gzip(output / "native-history.json.gz", history)
    for row in report["phases"]:
        probe.write_gzip(output / f"{row['phase']}-trace.json.gz", trace)
    return spec, report


def test_saved_payload_hashes_and_exact_trace_parity(tmp_path):
    spec, report = _saved_payloads(tmp_path)
    assert probe.saved_artifacts_match(tmp_path, report, spec)
    report["phases"][1]["trace_digest"] = "wrong"
    assert not probe.saved_artifacts_match(tmp_path, report, spec)


@pytest.mark.parametrize("failure", ["missing", "corrupt", "too_large", "history"])
def test_saved_payload_failure_never_counts_as_completed(tmp_path, failure):
    spec, report = _saved_payloads(tmp_path)
    target = tmp_path / "batch-trace.json.gz"
    if failure == "missing": target.unlink()
    elif failure == "corrupt": target.write_bytes(b"not gzip")
    elif failure == "too_large": spec["max_artifact_uncompressed_bytes"] = 1
    else: spec["expected_history_hash"] = "wrong"
    assert not probe.saved_artifacts_match(tmp_path, report, spec)


def test_existing_output_refused_without_launch(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs): raise AssertionError("Must not launch")
    monkeypatch.setattr(probe.subprocess, "Popen", forbidden)
    with pytest.raises(FileExistsError, match="fresh"):
        probe.launch(tmp_path)


def _actual_audit_phase(phase="scalar_before"):
    # Serialize the real audit dataclass without executing an audit or a scene.
    from resectionlab.evaluation import NativeRemovalAudit

    spec = probe.declaration()
    historical = probe.reference_row(spec)["independent_audit"]["certificate"]
    audit = NativeRemovalAudit(**{**historical, "failures": tuple(historical["failures"])})
    certificate = audit.to_dict()
    count = spec["expected_microsteps"]
    row = dict(phase=phase, backend="batch" if phase == "batch" else "scalar",
        error=None, hooks_restored=True, certificate=certificate,
        active_contact_scans=count, committed_prefixes=count,
        scalar_cell_queries=(6 if phase == "batch" else 7) * count,
        batch_cell_queries=count if phase == "batch" else 0)
    return row, historical, count


@pytest.mark.parametrize("phase", ["scalar_before", "batch", "scalar_after"])
def test_actual_audit_to_dict_json_roundtrip_has_exact_certificate_parity(phase):
    row, historical, count = _actual_audit_phase(phase)
    serialized = json.loads(json.dumps(row["certificate"], allow_nan=False))
    assert isinstance(row["certificate"]["failures"], tuple)
    assert isinstance(serialized["failures"], list)
    assert row["certificate"] != serialized
    assert serialized == historical
    assert probe.canonical(row["certificate"]) == probe.canonical(serialized)
    probe.validate_phase(row, historical, count)


@pytest.mark.parametrize("fault", ["one_ulp_volume", "source_identity", "rejected", "integer_representation"])
def test_actual_audit_normalization_does_not_relax_scientific_equality(fault):
    row, historical, count = _actual_audit_phase()
    certificate = row["certificate"]
    if fault == "one_ulp_volume":
        key = "contained_source_tissue_volume_mm3"
        certificate[key] = math.nextafter(certificate[key], math.inf)
    elif fault == "source_identity":
        certificate["source_case_hash"] = "changed"
    elif fault == "integer_representation":
        # JSON canonical comparison retains the distinction between 1 and 1.0.
        certificate["action_count"] = float(certificate["action_count"])
    else:
        certificate["feasible"] = False
    assert probe.canonical(certificate) != probe.canonical(historical)
    with pytest.raises(ValueError, match="historical certificate"):
        probe.validate_phase(row, historical, count)
