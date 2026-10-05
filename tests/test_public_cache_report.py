"""Small receipt fixtures only; no simulation, cache or patient data loaded."""
import gzip
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import report_public_native_capsule_cache as report


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


@pytest.fixture
def receipts(tmp_path):
    root, source = tmp_path / "run", tmp_path / "archive-source"
    source.mkdir()
    (source / "one.py").write_bytes(b"# explicit report fixture\n")
    (source / "case.bundle").write_bytes(b"not a patient")
    archive, manifest, runtime = (tmp_path / name for name in ("archive.tar", "source-files.json", "runtime.json"))
    archive.write_bytes(b"synthetic archive receipt")
    write(manifest, {"revision": "fixture", "tracked_file_sha256": {"one.py": report.file_digest(source / "one.py")}})
    snapshot = {"file_sha256": {}, "runtime_versions": {}, "python": "fixture"}
    snapshot["runtime_content_hash"] = report.content_hash({"files": {}, "versions": {}, "python": "fixture"})
    write(runtime, snapshot)
    declaration = {"v2_model": {"fixed_actions": ["A", "B", "C"], "greedy_history_hash": "history", "decision_model_hash": "model"},
        "fixed_work": {"preview_attempts_per_episode": 66}, "cache": {"max_entries": 16, "max_payload_bytes": 1000}}
    declaration["declaration_content_hash"] = report.content_hash(declaration)
    baseline = {"source_commit": "fixture", "source_archive_path": str(archive), "source_archive_sha256": report.file_digest(archive),
        "source_manifest_path": str(manifest), "source_manifest_sha256": report.file_digest(manifest), "source_file_count": 1,
        "immutable_source_root": str(source), "case_bundle_path": str(source / "case.bundle"), "case_bundle_relative_path": "case.bundle",
        "case_bundle_sha256": report.file_digest(source / "case.bundle"), "runtime_manifest_path": str(runtime),
        "runtime_manifest_sha256": report.file_digest(runtime), "runtime_content_hash": snapshot["runtime_content_hash"],
        "declaration_content_hash": declaration["declaration_content_hash"]}
    baseline_path = tmp_path / "baseline.json"
    write(baseline_path, baseline)
    write(root / "launch-source.json", snapshot)
    write(root / "declaration.json", declaration)
    for name, value in {"preparation": {"seconds": .1}, "template": {"factory_seconds": 2.},
        "cache-construction": {"seconds": .1}, "worker-resource": {"elapsed_seconds": 30., "observed_peak_rss_bytes": 100000}}.items():
        write(root / (name + ".json"), value)
    raw = b'{"receipt_fixture":true}'
    zipped = gzip.compress(raw, mtime=0)
    empty = {key: 0 for key in report.COUNTERS} | {"entries": 0, "retained_payload_bytes": 0, "max_entries": 16, "max_payload_bytes": 1000}
    cold = empty | {"calls": 10, "misses": 4, "hits": 6, "entries": 4, "retained_payload_bytes": 200}
    warm = cold | {"calls": 20, "hits": 16}
    phases = []
    for index, mode in enumerate(report.MODES):
        timing = {"reset_seconds": 1., "transitions": [{"transition_and_next_inventory_seconds": 1.}] * 3,
            "seconds_before_export": 4.5, "adapter_proposal_accounting": {"preview_calls": 66, "preview_seconds": 2.},
            "full_proposal_verification": {"calls": 9, "seconds": 1.5}}
        row = {"mode": mode, "actions": ["A", "B", "C"], "exact_scientific_equality": True, "timing": timing,
            "scientific_trace_export": {"compressed_sha256": report.digest(zipped), "canonical_sha256": report.digest(raw)},
            "scientific_trace_hash": "sha256:" + report.digest(raw), "history_hash": "history", "total_reward": 3.,
            "cache_before": (None, empty, cold, warm)[index], "cache_after": (None, cold, warm, warm)[index],
            "scientific_trace_export_seconds": .1, "seconds_including_scientific_comparison_and_export": 5.,
            "process_cumulative_peak_rss_bytes": 100000}
        write(root / mode / "receipt.json", row)
        write(root / mode / "timing.json", timing)
        (root / mode / "scientific-trace.json.gz").write_bytes(zipped)
        phases.append(row)
    audits = {mode: {"receipt": {"feasible": True, "complete_tool_checked": True, "frontier_checked": True}, "seconds": 1.}
              for mode in report.MODES}
    for mode, audit in audits.items():
        write(root / (mode + "-independent-audit.json"), audit)
    result = {"status": "completed", "runtime_content_hash": snapshot["runtime_content_hash"], "scientific_model_hash": "model",
        "gradient_updates": 0, "final_worlds_used": False, "stress_worlds_used": False, "phases": phases, "independent_audits": audits}
    write(root / "result.json", result)
    write(root / "worker-status.json", {"status": "completed", "runtime_content_hash": snapshot["runtime_content_hash"],
        "result_hash": report.content_hash(result)})
    write(root / "launcher-status.json", {"status": "completed", "worker_returncode": 0,
        "parent_timeout_requested": False, "hard_killed": False, "full_launcher_seconds": 31.})
    return root, baseline_path, baseline


def test_completed_report_has_exact_scopes_and_no_fabricated_capture_timing(receipts):
    root, baseline, _ = receipts
    result, index = report.extract(root, baseline)
    assert result["archived_source_verification"]["tracked_files_verified"] == 1
    assert result["phases"][0]["capture_accounting_and_phase_overhead_seconds"] == .5
    assert result["phases"][0]["certificate_capture_seconds"] is None
    assert result["phases"][2]["cache_delta"]["hits"] == 10
    assert result["new_simulator_calls"] == result["new_audit_replays"] == result["new_gradient_updates"] == 0
    assert "reference_before/scientific-trace.json.gz" in index
    assert "not a confidence interval" in result["interpretation"]
    assert "Cumulative RSS" in report.markdown(result)


@pytest.mark.parametrize("change", ["growing", "authority", "archive", "source", "case", "trace", "receipt", "added_source"])
def test_changed_or_growing_evidence_is_rejected(receipts, change):
    root, baseline_path, baseline = receipts
    if change == "growing":
        write(root / "launcher-status.json", {"status": "running"})
    elif change == "authority":
        write(root / "worker-status.json", {"status": "completed", "result_hash": "different"})
    elif change in ("archive", "case"):
        Path(baseline["source_archive_path" if change == "archive" else "case_bundle_path"]).write_bytes(b"changed")
    elif change in ("source", "added_source"):
        (Path(baseline["immutable_source_root"]) / ("one.py" if change == "source" else "extra.py")).write_bytes(b"changed")
    elif change == "trace":
        (root / "cached_cold/scientific-trace.json.gz").write_bytes(gzip.compress(b"different"))
    else:
        write(root / "cached_cold/receipt.json", {"different": True})
    with pytest.raises(ValueError):
        report.extract(root, baseline_path)


def test_raw_gzip_duplicate_must_agree_and_prior_outputs_remain_untouched(receipts, monkeypatch, tmp_path):
    root, baseline, _ = receipts
    path = root / "preparation.json"
    path.with_suffix(".json.gz").write_bytes(gzip.compress(b"{}"))
    with pytest.raises(ValueError, match="Raw/gzip"):
        report.extract(root, baseline)
    output = tmp_path / "prior-report"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_bytes(b"preserve")
    monkeypatch.setattr(sys, "argv", ["report", "--input", str(root), "--baseline", str(baseline), "--output", str(output)])
    with pytest.raises(ValueError, match="already exists"):
        report.main()
    assert marker.read_bytes() == b"preserve"
