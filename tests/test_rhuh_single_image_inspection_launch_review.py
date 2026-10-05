"""Independent offline launcher controls; no actual image or inspector call."""
import json
from types import SimpleNamespace

import pytest

from test_rhuh_single_image_inspection_launch import mod, fixture, process_control, run

ORIGINAL_SAMPLE_RSS = mod.sample_rss


@pytest.mark.parametrize("kind", ["empty", "unrelated_only", "child_only", "duplicate_launcher"])
def test_rss_requires_one_live_launcher_measurement(monkeypatch, kind):
    """Missing RSS must remain unavailable rather than becoming zero."""
    launcher = mod.os.getpid()
    streams = {
        "empty": b"",
        "unrelated_only": b"900001 900001 33\n",
        "child_only": b"900002 812345 17\n",
        "duplicate_launcher": f"{launcher} 1 13\n{launcher} 1 13\n".encode(),
    }
    monkeypatch.setattr(mod.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout=streams[kind]))
    with pytest.raises(mod.Rejected):
        mod.sample_rss(812345)


def test_exited_child_group_can_leave_only_launcher_snapshot(monkeypatch):
    raw = f"{mod.os.getpid()} 1 23\n900001 900001 9999\n".encode()
    monkeypatch.setattr(mod.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout=raw))
    assert mod.sample_rss(812345) == 23 * 1024


def test_missing_rss_rejects_completed_child_and_consumes_attempt(fixture, process_control, monkeypatch, capsys):
    monkeypatch.setattr(mod, "sample_rss", ORIGINAL_SAMPLE_RSS)
    monkeypatch.setattr(mod.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout=b""))
    assert run(fixture) == 1
    result = json.loads(capsys.readouterr().out)
    assert result["accepted"] is False and result["rss_samples"] == 0
    assert len(process_control["kills"]) == 1
    assert run(fixture) == 1
    assert process_control["launches"] == 1


@pytest.mark.parametrize("length,accepted", [(65536, True), (65537, False)])
def test_stderr_has_its_own_exact_cap(fixture, process_control, monkeypatch, capsys, length, accepted):
    original = mod.subprocess.Popen
    def with_stderr(argv, **kwargs):
        kwargs["stderr"].write(b"x" * length)
        return original(argv, **kwargs)
    monkeypatch.setattr(mod.subprocess, "Popen", with_stderr)
    code = run(fixture)
    result = json.loads(capsys.readouterr().out)
    assert result["accepted"] is accepted and (code == 0) is accepted
    assert len(process_control["kills"]) == 1


@pytest.mark.parametrize("change", [{"status": "rejected_no_inspection_result"}, {"clinical_validation": True}])
def test_exit_zero_cannot_promote_rejected_or_clinical_result(fixture, process_control, capsys, change):
    fixture[4].update(change)
    assert run(fixture) == 1
    result = json.loads(capsys.readouterr().out)
    assert result["accepted"] is False and result["status"] == "inspection_result_not_accepted"
    assert "original" not in result


@pytest.mark.parametrize("key", ["inspector", "request"])
def test_final_readback_metadata_drift_is_not_labeled_unchanged(fixture, process_control, monkeypatch, capsys, key):
    root, plan, _, _, _ = fixture
    original = mod.reconcile
    def changing(*args):
        result = original(*args)  # Hashes inert fixture bytes only.
        (root / plan[key]["path"]).write_bytes(b"changed during final readback")
        return result
    monkeypatch.setattr(mod, "reconcile", changing)
    assert run(fixture) == 1
    result = json.loads(capsys.readouterr().out)
    assert result["accepted"] is False
    assert result.get("source_closure_unchanged") is not True
