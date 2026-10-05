"""Independent offline/mock PAT25 ingress runner checks; no native/patient runs."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import diagnose_pat25_ingress_access as d


def test_observer_exception_cannot_resume_a_later_unguarded_phase():
    namespace = {"__name__": "resectionlab.native_spatial_task", "body_ran": False}
    exec("def step():\n    global body_ran\n    body_ran = True\n", namespace)
    trace = d.InitialOnlyTrace()
    with pytest.raises(RuntimeError):
        with trace:
            trace.begin("original")
            with pytest.raises(RuntimeError, match="prohibited"):
                namespace["step"]()
            assert namespace["body_ran"] is False
            assert sys.getprofile() is None  # CPython removed the failing callback.
            trace.end()
            trace.begin("screened_selected")
    assert sys.getprofile() is None


def parent_fixture(monkeypatch, tmp_path):
    state = {"seconds": 0., "unchanged_calls": 0, "commands": [], "caps": []}
    monkeypatch.setattr(d, "time", SimpleNamespace(perf_counter=lambda: state["seconds"]))
    manifest, release, output = tmp_path / "manifest.json", tmp_path / "release.json", tmp_path / "attempt"
    manifest.write_text("manifest-bound-to-parent")
    release.write_text("release-bound-to-parent")
    digests = {str(manifest): "1" * 64, str(release): "2" * 64}
    monkeypatch.setattr(d, "sha256", lambda path: digests.get(str(Path(path)), "3" * 64))
    def preflight(*args, **kwargs):
        state["seconds"] += 30.
        if "check" in kwargs:
            kwargs["check"]()
        return {}, {}, dict(digests), {"source_commit": "a" * 40}
    monkeypatch.setattr(d, "preflight", preflight)
    def supervisor(command, destination, settings, digest):
        state["commands"].append(command)
        state["caps"].append(settings)
        state["seconds"] += 140.
        d.write_json(destination / "receipt.json", {"version": d.VERSION, "subject": d.SUBJECT,
            "status": "complete", "inputs_unchanged": True})
        return {"status": "complete"}
    monkeypatch.setattr(d, "supervise_worker", supervisor)
    monkeypatch.setattr(d, "unchanged", lambda bound: True)
    return manifest, release, output, state, digests


def test_full_attempt_budget_includes_last_input_hash_and_acceptance_publication(monkeypatch, tmp_path):
    manifest, release, output, state, _ = parent_fixture(monkeypatch, tmp_path)
    real_write = d.write_json
    def unchanged(bound):
        state["unchanged_calls"] += 1
        if state["unchanged_calls"] == 2:
            state["seconds"] += 20.  # Final encoded-input/source hashing costs real time.
        return True
    def write(path, payload):
        if Path(path).name == "acceptance.json":
            state["seconds"] += 10.  # Final serialization/publication also belongs to lifecycle.
        return real_write(path, payload)
    monkeypatch.setattr(d, "unchanged", unchanged)
    monkeypatch.setattr(d, "write_json", write)
    result = d.run(manifest, release, output)
    assert state["seconds"] >= 190.
    assert result["status"] != "complete", "Late hashing/publication exceeded180s but was accepted"


def test_parent_binds_manifest_and_release_bytes_into_worker_invocation(monkeypatch, tmp_path):
    manifest, release, output, state, digests = parent_fixture(monkeypatch, tmp_path)
    d.run(manifest, release, output)
    assert len(state["commands"]) == 1
    command = state["commands"][0]
    assert digests[str(manifest)] in command, "Worker must receive the parent's exact declaration byte identity"
    assert digests[str(release)] in command, "Worker must receive the parent's exact release byte identity"


@pytest.mark.parametrize("subject,role", [("sub-PAT05", "TRAIN"), ("sub-PAT22", "TRAIN"),
    ("sub-PAT26", "TRAIN"), ("sub-PAT27", "TRAIN"), ("sub-PAT29", "TRAIN"),
    ("sub-PAT31", "TRAIN"), ("sub-PAT25", "SELECT")])
def test_fixed_patient_role_rejected_before_registry_or_decode(monkeypatch, subject, role):
    monkeypatch.setattr(d, "declaration", lambda: pytest.fail("Role check must precede input registry"))
    with pytest.raises(ValueError, match="PAT25"):
        d.validate({"version": d.VERSION, "subject": subject, "role": role, "settings": d.SETTINGS})


def derivative_fixture(monkeypatch):
    """Seven-cell-axis analytic metadata fixture; no CaseData or native engine."""
    import numpy as np
    support = np.ones((7, 7, 7), bool)
    nominal = np.zeros_like(support)
    nominal[3, 3, 3] = True
    affine = np.diag([1., 2., 3., 1.])
    access, derivation = d.prep.derive_access(nominal, support, affine, d.SUBJECT)
    member = {"case_bundle": "not-a-patient.fixture", "case_bundle_sha256": "bytes",
        "case_semantic_hash": "case", "planning_hash": "plan", "evidence_id": "support", "evidence_hash": "evidence"}
    evidence = SimpleNamespace(mask=support, evidence_hash="evidence", review_status="review_required",
        provenance="estimated", review=None, model_sha256="model", assert_matches=lambda case: None)
    source = {"accession": "ds001226", "release": "5.0.1", "git_commit": "revision"}
    case = SimpleNamespace(case_id="BTC-ds001226-sub-PAT25-preop", semantic_hash="case", planning_hash="plan",
        metadata={"source_collection": source}, source_refs=[SimpleNamespace(source_id="structural", provenance="observed")],
        structural_evidence={"support": evidence}, compartments={"permitted_annotation": nominal}, affine=affine, frame="RAS+")
    saved = {"binding": {"member": {"access": access, "access_derivation": derivation}},
             "coverage": {"full_target_source_cells": 1}}
    monkeypatch.setattr(d.legacy, "validate", lambda record: (saved, {"source": source}))
    monkeypatch.setattr(d.legacy, "sha256", lambda path: "bytes")
    monkeypatch.setattr(d.prep, "_decode_case", lambda path: case)  # Object return only, never image decoding.
    monkeypatch.setattr(d.prep, "_construct_task", lambda *a, **k: pytest.fail("No native construction in descriptor review"))
    return {"member": member}, saved, case, derivation


@pytest.mark.parametrize("index", range(6))
@pytest.mark.parametrize("field", ["distance_mm", "axis", "outward_sign"])
def test_every_exit_exact_distance_axis_sign_recomputed_before_screen(monkeypatch, index, field):
    import numpy as np
    record, saved, case, derivation = derivative_fixture(monkeypatch)
    exit_record = derivation["six_axis_exit_distances_mm"][index]
    if field == "distance_mm":
        exit_record[field] = float(np.nextafter(exit_record[field], np.inf))
    elif field == "axis":
        exit_record[field] = (exit_record[field] + 1) % 3
    else:
        exit_record[field] = -exit_record[field]
    with pytest.raises(ValueError, match="axis-walk binding changed"):
        d.legacy.load_bound_case(record, lambda: None)


def test_exact_six_exit_distance_tie_order_is_source_physical_rule(monkeypatch):
    record, saved, case, derivation = derivative_fixture(monkeypatch)
    _, _, exits = d.legacy.load_bound_case(record, lambda: None)
    assert [(row["axis"], row["outward_sign"]) for row in exits] == [(a, s) for a in range(3) for s in (-1, 1)]
    assert [row["distance_mm"] for row in exits] == [3.5, 3.5, 7., 7., 10.5, 10.5]
    assert [row["selected_original"] for row in exits] == [True, False, False, False, False, False]
    assert exits[0]["access"] == saved["binding"]["member"]["access"]


@pytest.mark.parametrize("field", ["case_semantic_hash", "planning_hash", "evidence_hash"])
def test_original_source_and_support_identity_mismatch_rejected(monkeypatch, field):
    record, _, _, _ = derivative_fixture(monkeypatch)
    record["member"][field] += "changed"
    with pytest.raises(ValueError):
        d.legacy.load_bound_case(record, lambda: None)


def mock_worker(monkeypatch, tmp_path, *, selected=(0, 1), complete=True, screens=156,
                selected_accepted=0, selected_error=False):
    import resectionlab.native_ingress as ingress
    import resectionlab.native_spatial_task as native
    manifest, release, output = tmp_path / "manifest.json", tmp_path / "release.json", tmp_path / "attempt"
    manifest.write_text("{}"); release.write_text("{}"); output.mkdir()
    bound = {str(manifest): d.sha256(manifest), str(release): d.sha256(release)}
    exits = [{"axis": a, "outward_sign": sign, "selected_original": a == 0 and sign == -1,
              "distance_mm": float(a + 1), "access": {"identity": (a, sign)}} for a in range(3) for sign in (-1, 1)]
    inventory = {"identity": "exact-original", "accepted_count": 0}
    common = {"objective": {"fixed": 1.}, "max_steps": 3}
    saved = {"binding": {"member": {}, "decision_model_hash": "original-model"}, "initial_inventory": inventory}
    record = {"legacy_input": {"common_task": common}}
    task = SimpleNamespace(case=object(), reward_spec=object())
    calls = {"phases": [], "constructors": [], "static_candidates": None, "screen_calls": 0}
    monkeypatch.setattr(d, "preflight", lambda *a, **k: (record, saved, bound, {"source_commit": "a" * 40}))
    monkeypatch.setattr(d.legacy, "load_bound_case", lambda *a: (object(), saved, exits))
    monkeypatch.setattr(d.prep, "_construct_task", lambda *a: calls["constructors"].append("original") or task)
    monkeypatch.setattr(native, "NativeSpatialTask", lambda *a, **k: calls["constructors"].append("selected") or task)
    class Trace:
        total = 0
        check = None
        violation = None
        phases = ["original"]
        def __enter__(self): return self
        def __exit__(self, *args): pass
    monkeypatch.setattr(d, "InitialOnlyTrace", Trace)
    candidates = tuple(object() for _ in range(6))
    sources = {(row["axis"], row["outward_sign"]): object() for row in exits}
    def prepare(source, given_exits, check):
        check()
        assert source is task.case and given_exits == exits
        calls["static_candidates"] = candidates
        return candidates, sources
    monkeypatch.setattr(d, "prepare_candidates", prepare)
    def screen(given, **kwargs):
        calls["screen_calls"] += 1
        assert given is candidates
        return SimpleNamespace(complete=complete, selected_exit=selected, screen_count=screens,
            to_dict=lambda: {"complete": complete, "selected_exit": selected, "screen_count": screens,
                "exits": [{"axis": x["axis"], "outward_sign": x["outward_sign"]} for x in exits]})
    monkeypatch.setattr(ingress, "screen_axis_accesses", screen)
    def inspect(factory, exit_record, phase, trace, check):
        calls["phases"].append((phase, (exit_record["axis"], exit_record["outward_sign"])))
        factory()  # Both factories are pure mocked object returns above.
        if phase == "screened_selected" and selected_error:
            trace.total += 7
            failure = RuntimeError("selected full-motion diagnostic interrupted")
            failure.diagnostic = {"status": "unassessed", "partial_preview_trace": ["preserved"], "started_preview_calls": 7}
            raise failure
        trace.total += 78
        return task, {"status": "complete", "initial_inventory": inventory if phase == "original" else {"accepted_count": selected_accepted},
            "decision_model_hash": "original-model", "reward": common["objective"], "horizon": 3,
            "preview_calls": 78, "actor_coverage": {"nominal_fraction": 0. if phase == "screened_selected" else 1.}}
    monkeypatch.setattr(d, "inspect_initial", inspect)
    return manifest, release, output, calls


def run_mock_worker(args):
    return d.worker(*args[:3], expected_manifest_sha256=d.sha256(args[0]),
                    expected_release_sha256=d.sha256(args[1]))


@pytest.mark.parametrize("accepted", [0, 78])
def test_selection_is_not_reranked_by_full_motion_outcomes_or_crop_visibility(monkeypatch, tmp_path, accepted):
    args = mock_worker(monkeypatch, tmp_path, selected_accepted=accepted)
    result = run_mock_worker(args)
    assert result["status"] == "complete"
    assert args[3]["phases"] == [("original", (0, -1)), ("screened_selected", (0, 1))]
    assert args[3]["constructors"] == ["original", "selected"]
    assert result["screened_selected"]["initial_inventory"]["accepted_count"] == accepted
    assert result["screened_selected"]["actor_coverage"]["nominal_fraction"] == 0.
    assert result["total_preview_calls"] == 156 and args[3]["screen_calls"] == 1
    assert not any(result[name] for name in ("executed_transitions", "commits", "searches", "policy_forwards", "checkpoints", "optimizer_updates"))


def test_incomplete_static_screen_never_constructs_selected_inventory(monkeypatch, tmp_path):
    args = mock_worker(monkeypatch, tmp_path, complete=False)
    result = run_mock_worker(args)
    assert result["status"] == "incomplete"
    assert args[3]["constructors"] == ["original"]
    assert result["total_preview_calls"] == 78
    assert not result["screening"]["complete"]


def test_excess_static_count_is_rejected_before_selected_inventory(monkeypatch, tmp_path):
    args = mock_worker(monkeypatch, tmp_path, screens=469)
    result = run_mock_worker(args)
    assert result["status"] == "incomplete" and args[3]["constructors"] == ["original"]


def test_selected_partial_failure_is_durable_without_fallback(monkeypatch, tmp_path):
    args = mock_worker(monkeypatch, tmp_path, selected_error=True)
    result = run_mock_worker(args)
    durable = json.loads((args[2] / "receipt.json").read_text())
    assert result["status"] == durable["status"] == "incomplete"
    assert result["screened_selected"]["partial_preview_trace"] == ["preserved"]
    assert result["total_preview_calls"] == 85
    assert args[3]["constructors"] == ["original", "selected"]
    assert not result["automatic_retry"]


@pytest.mark.parametrize("phase", ["original", "screened_selected"])
@pytest.mark.parametrize("field", ["reward", "horizon"])
def test_frozen_objective_and_horizon_drift_remain_failed_without_fallback(monkeypatch, tmp_path, phase, field):
    args = mock_worker(monkeypatch, tmp_path)
    inspect = d.inspect_initial
    def changed(*positional, **kwargs):
        task, result = inspect(*positional, **kwargs)
        if positional[2] == phase:
            result[field] = {"changed": 1.} if field == "reward" else 4
        return task, result
    monkeypatch.setattr(d, "inspect_initial", changed)
    result = run_mock_worker(args)
    assert result["status"] == "incomplete"
    assert args[3]["constructors"] == (["original"] if phase == "original" else ["original", "selected"])
    assert not result["automatic_retry"]


def test_no_admissible_access_preserves_absence_without_motion_attempt(monkeypatch, tmp_path):
    args = mock_worker(monkeypatch, tmp_path, selected=None)
    result = run_mock_worker(args)
    assert result["status"] == "complete"
    assert result["screened_selected"]["status"] == "not_executed_no_ingress_admissible_access"
    assert args[3]["constructors"] == ["original"] and result["total_preview_calls"] == 78


@pytest.mark.parametrize("changed", ["manifest", "release", "missing"])
def test_worker_byte_pin_failure_is_durable_before_preflight_or_decode(monkeypatch, tmp_path, changed):
    manifest, release, output, calls = mock_worker(monkeypatch, tmp_path)
    pins = {"expected_manifest_sha256": d.sha256(manifest), "expected_release_sha256": d.sha256(release)}
    if changed == "missing":
        pins = {}
    else:
        pins["expected_" + changed + "_sha256"] = "a" * 64
    monkeypatch.setattr(d, "preflight", lambda *a, **k: pytest.fail("Changed parent bytes reached preflight"))
    result = d.worker(manifest, release, output, **pins)
    assert result["status"] == "incomplete" and result["total_preview_calls"] == 0
    assert calls["constructors"] == []
    assert "exact manifest/release bytes" in result["error"]
    assert json.loads((output / "receipt.json").read_text())["error"] == result["error"]


@pytest.mark.parametrize("changed", ["manifest", "release"])
def test_orchestrator_rejects_changed_outer_pins_before_preflight(monkeypatch, tmp_path, changed):
    manifest, release, output, _, digests = parent_fixture(monkeypatch, tmp_path)
    pins = {"expected_manifest_sha256": digests[str(manifest)], "expected_release_sha256": digests[str(release)]}
    pins["expected_" + changed + "_sha256"] = "a" * 64
    monkeypatch.setattr(d, "preflight", lambda *a, **k: pytest.fail("Changed outer pins reached preflight"))
    with pytest.raises(ValueError, match="outer launcher"):
        d.run(manifest, release, output, **pins)
    assert not output.exists()


def outer_fixture(monkeypatch, tmp_path, *, first_time=0., group_rss=1024, child_exits=False,
                  surviving_descendant=False):
    """All processes/signals/RSS are mocks; only tiny receipt files are written."""
    state = {"seconds": first_time, "group_alive": True, "signals": [], "commands": [], "rss_calls": 0}
    manifest, release, output = tmp_path / "m.json", tmp_path / "r.json", tmp_path / "out"
    manifest.write_text("{}"); release.write_text("{}")
    monkeypatch.setattr(d, "_BOOT_STARTED", 0.)
    monkeypatch.setattr(d, "time", SimpleNamespace(perf_counter=lambda: state["seconds"],
        sleep=lambda seconds: state.__setitem__("seconds", state["seconds"] + seconds)))
    class Process:
        pid = 54321
        code = 0 if child_exits else None
        def poll(self): return self.code
        def wait(self, timeout):
            if not state["group_alive"]: self.code = self.code if self.code is not None else -9
            return self.code
    process = Process()
    def popen(command, **kwargs):
        state["commands"].append((command, kwargs))
        output.mkdir()
        (output / "acceptance.json").write_text('{"status":"complete"}')
        if child_exits and not surviving_descendant:
            state["group_alive"] = False
        return process
    def signal_group(pgid, signum):
        assert pgid == process.pid
        state["signals"].append(signum)
        if signum == d.signal.SIGKILL:
            state["group_alive"] = False
            process.code = process.code if process.code is not None else -9
    def rss(pgid, timeout):
        assert pgid == process.pid and 0 < timeout <= .5
        state["rss_calls"] += 1
        return group_rss
    monkeypatch.setattr(d.subprocess, "Popen", popen)
    monkeypatch.setattr(d, "_group_alive", lambda pgid: state["group_alive"])
    monkeypatch.setattr(d, "_signal_group", signal_group)
    monkeypatch.setattr(d, "_group_rss", rss)
    argv = ["--manifest", str(manifest), "--release", str(release), "--output", str(output)]
    return argv, output, state, process


def test_outer_budget_includes_already_elapsed_imports_and_preflight(monkeypatch, tmp_path):
    argv, output, state, _ = outer_fixture(monkeypatch, tmp_path, first_time=174.01)
    assert d._outer_main(argv) == 1
    record = json.loads(output.with_name("out.outer.json").read_text())
    assert record["reason"] == "outer_whole_attempt_wall_budget"
    assert record["group_gone"] and state["rss_calls"] == 0
    assert d.signal.SIGTERM in state["signals"] and d.signal.SIGKILL in state["signals"]
    assert record["elapsed_seconds"] < 180


def test_outer_aggregated_group_rss_is_bounded_and_one_cpu_environment(monkeypatch, tmp_path):
    argv, output, state, _ = outer_fixture(monkeypatch, tmp_path, group_rss=6 * 1024**3 + 1)
    assert d._outer_main(argv) == 1
    record = json.loads(output.with_name("out.outer.json").read_text())
    assert record["reason"] == "outer_combined_group_rss_budget" and record["group_gone"]
    command, options = state["commands"][0]
    assert options["start_new_session"] is True
    assert all(options["env"][key] == "1" for key in d.THREAD_ENV)
    assert "--expected-manifest-sha256" in command and "--expected-release-sha256" in command
    assert record["sampled_peak_group_rss_bytes"] > 6 * 1024**3


def test_outer_kills_known_group_after_leader_exits_with_descendants(monkeypatch, tmp_path):
    argv, output, state, process = outer_fixture(monkeypatch, tmp_path, child_exits=True, surviving_descendant=True)
    assert d._outer_main(argv) == 1
    record = json.loads(output.with_name("out.outer.json").read_text())
    assert record["reason"] == "orchestrator_exited_with_surviving_descendants"
    assert record["owned_process_group"] == process.pid and record["group_gone"]
    assert state["signals"] == [d.signal.SIGKILL]


def test_outer_rejects_late_final_receipt_publication(monkeypatch, tmp_path):
    argv, output, state, _ = outer_fixture(monkeypatch, tmp_path, first_time=179., child_exits=True)
    original = Path.write_text
    def write(path, text, *args, **kwargs):
        if path.name == "out.outer.json":
            state["seconds"] += 2.
        return original(path, text, *args, **kwargs)
    monkeypatch.setattr(Path, "write_text", write)
    assert d._outer_main(argv) == 1
    record = json.loads(output.with_name("out.outer.json").read_text())
    assert record["status"] == "failed" and record["reason"] == "outer_receipt_publication_overrun"


def test_group_rss_sums_only_owned_pgid_members(monkeypatch):
    seen = []
    def ps(command, **kwargs):
        seen.append((command, kwargs))
        return SimpleNamespace(stdout="54321 100\n999 9000\n54321 200\n")
    monkeypatch.setattr(d.subprocess, "run", ps)
    assert d._group_rss(54321, .25) == 300 * 1024
    assert seen[0][1]["timeout"] == .25


def test_inner_native_worker_inherits_outer_owned_group(monkeypatch, tmp_path):
    seen = []
    process = SimpleNamespace(pid=123, poll=lambda: 0)
    monkeypatch.setattr(d.subprocess, "Popen", lambda command, **kw: seen.append(kw) or process)
    result = d.supervise_worker(["mock-only"], tmp_path, d.SETTINGS, "a" * 64)
    assert result["status"] == "complete" and result["inherits_outer_process_group"]
    assert seen[0]["start_new_session"] is False


def test_outer_dispatch_precedes_scientific_project_imports():
    """AST ordering check supplements process mocks; importing this test is not a launch."""
    import ast
    tree = ast.parse(Path(d.__file__).read_text())
    dispatch = next(node for node in tree.body if isinstance(node, ast.If)
        and any(isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
                and call.func.id == "_outer_main" for call in ast.walk(node)))
    project_imports = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))
        and any(token in ast.unparse(node) for token in ("diagnose_pat25_access", "diagnose_training_observation", "preflight_real_spatial"))]
    assert project_imports and all(dispatch.lineno < node.lineno for node in project_imports)


@pytest.mark.parametrize("module,name", [
    ("resectionlab.native_spatial_task", "step"),
    ("resectionlab.native_resection", "commit_preview"),
    ("resectionlab.native_resection", "execute_stroke"),
    ("resectionlab.spatial_policy", "__init__"),
    ("resectionlab.spatial_policy", "forward"),
    ("resectionlab.spatial_policy", "gradient_step"),
    ("resectionlab.spatial_policy", "reinforce_loss"),
    ("resectionlab.spatial_policy", "behavior_cloning_loss"),
    ("resectionlab.observed_search", "observed_search"),
    ("torch.optim.optimizer", "__init__"),
    ("torch.optim.adam", "step"),
])
def test_prohibited_operation_guard_blocks_mock_function_body(module, name):
    namespace = {"__name__": module, "body_ran": False}
    exec(f"def {name}():\n    global body_ran\n    body_ran = True\n", namespace)
    with pytest.raises(RuntimeError, match="prohibited"):
        with d.InitialOnlyTrace() as trace:
            trace.begin("original")
            namespace[name]()
    assert namespace["body_ran"] is False and sys.getprofile() is None


def test_guard_allows_optimizer_module_definition_without_optimizer_calls():
    namespace = {"__name__": "torch.optim.optimizer"}
    with d.InitialOnlyTrace() as trace:
        trace.begin("original")
        exec("module_definition_only = True", namespace)
        trace.end()
    assert namespace["module_definition_only"] and sys.getprofile() is None


def test_git_object_mismatch_rejected_before_bundle_or_decoder(monkeypatch, tmp_path):
    manifest, release, output = tmp_path / "m.json", tmp_path / "r.json", tmp_path / "out"
    name = "scripts/diagnose_pat25_ingress_access.py"
    digest = d.sha256(d.ROOT / name)
    record = {"subject": d.SUBJECT, "role": "TRAIN", "source_sha256": {name: digest}, "metadata_sha256": {}}
    manifest.write_text(json.dumps(record))
    release.write_text(json.dumps({"version": d.VERSION + "-release", "authorized": True,
        "subject": d.SUBJECT, "role": "TRAIN", "phase": "initial_inventory_diagnostic",
        "declaration_sha256": d.sha256(manifest), "manifest_path": str(manifest),
        "archive_root": str(d.ROOT), "output": str(output), "repository_root": str(tmp_path),
        "source_commit": "a" * 40}))
    monkeypatch.setattr(d, "validate", lambda _: {})
    seen = []
    def git(command, **kwargs):
        seen.append(command)
        return SimpleNamespace(stdout=b"different committed source")
    monkeypatch.setattr(d.subprocess, "run", git)
    monkeypatch.setattr(d.prep, "_decode_case", lambda *a: pytest.fail("No decoder in source preflight"))
    with pytest.raises(ValueError, match="exact committed Git closure"):
        d.preflight(manifest, release, output)
    assert seen == [["git", "-C", str(tmp_path), "show", f"{'a' * 40}:{name}"]]
    assert not output.exists()


def test_bound_relative_source_cannot_escape_archive_by_symlink(tmp_path):
    archive = tmp_path / "archive"
    archive.mkdir()
    outside = tmp_path / "outside"
    outside.write_text("not an archive source")
    (archive / "linked.py").symlink_to(outside)
    with pytest.raises(ValueError, match="escape"):
        d._local(archive, "linked.py")
