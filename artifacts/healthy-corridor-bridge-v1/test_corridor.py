"""Generated executable contracts; no learned models or real image payloads."""
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests"))
import test_ixi_vascular_admission as metadata
from resectionlab import ixi_vascular_admission as admission
from resectionlab.core import array_digest, semantic_digest, thaw_json
from resectionlab.geometry import ToolGeometry

SPEC = importlib.util.spec_from_file_location("healthy_corridor_candidate", Path(__file__).with_name("healthy_corridor.py"))
c = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = c
SPEC.loader.exec_module(c)


@pytest.fixture
def tmp_path():
    with tempfile.TemporaryDirectory(prefix="healthy-corridor-", dir=ROOT / "build") as name:
        yield Path(name)


def fixture(*, positive=True, unknown_plane=False, definition_change=None):
    value = metadata.fixture()
    shape = (16, 16, 16)
    affine = np.eye(4)
    grid = {"shape": list(shape), "affine_ras_mm": affine.tolist(),
        "frame_sha256": semantic_digest({"shape": list(shape), "affine_ras_mm": affine.tolist(), "frame": "RAS+"})}
    t1 = np.indices(shape).sum(axis=0).astype(np.float32)
    support = np.zeros(shape, bool)
    support[6:11, 6:11, 10:15] = True
    coverage = np.ones(shape, bool)
    if unknown_plane:
        coverage[:, :, 5] = False
    mra = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    mask = np.zeros(shape, bool)
    if positive:
        mask[8, 8, 10] = True
        mask[8, 10, 5] = True  # proximal shaft contact outside the brain support
    domains = {name: np.ones(shape, bool) for name in ("annotation_domain", "mra_domain", "registration_domain")}
    tool = ToolGeometry("generated_corridor_probe", .2, .3, 8., max_access_angle_deg=60., tip_length_mm=1.,
                        parameter_source="generated interface geometry only")
    definition = {"version": c.VERSION, "waypoint_rule": "six_neighbor_interior_lexicographic_quantiles_v1",
        "access_axis": 2, "access_margin_voxels": 1, "access_radius_mm": 3., "tools": [asdict(tool)],
        "reach_credit": 1., "effort_per_mm": .01, "horizon": 1}
    if definition_change:
        definition_change(definition)
    for name, array in (("T1", t1), ("MRA", mra), ("vessel_annotation", mask)):
        value["sources"][name]["grid"] = grid
        value["sources"][name]["array_sha256"] = array_digest(array)
    value["support"].update(output_sha256=array_digest(support), coverage_sha256=array_digest(coverage), frame_sha256=grid["frame_sha256"])
    value["task"].update(support_output_sha256=array_digest(support), frame_sha256=grid["frame_sha256"], definition_sha256=semantic_digest(definition))
    value["registration"].update(from_frame_sha256=grid["frame_sha256"], to_frame_sha256=grid["frame_sha256"],
        valid_domain_sha256=array_digest(domains["registration_domain"]))
    value["coverage"].update(mask_sha256=array_digest(mask), coverage_sha256=array_digest(domains["annotation_domain"]),
        frame_sha256=grid["frame_sha256"], annotation_domain_sha256=array_digest(domains["annotation_domain"]),
        mra_valid_domain_sha256=array_digest(domains["mra_domain"]), registration_valid_domain_sha256=array_digest(domains["registration_domain"]))
    prepared = admission.prepare_ixi_vascular_case(cohort_bytes=metadata.COHORT, evidence=value,
        reviewed_evidence_sha256=semantic_digest(value), purpose="frozen_scoring")
    task = c.CorridorTask(prepared.actor_contract, t1, support, coverage, definition)
    return task, prepared, {"MRA": mra, "mask": mask, **domains}


def controls():
    return {"IL": lambda observation: next(row["action_id"] for row in observation["candidates"] if row["legal"]),
            "RL": lambda observation: "STOP",
            "HYBRID": lambda observation: list(reversed([row["action_id"] for row in observation["candidates"] if row["legal"]] + ["STOP"]))}


def seal(task, path, selectors=None):
    return c.write_corridor_batch(task, selectors=controls() if selectors is None else selectors, output_directory=path)


def score(task, prepared, arrays, sealed, path, loader=None):
    return c.evaluate_corridor_batch(task, **sealed, private_contract=prepared.evaluator_contract,
        load_reference=(lambda: arrays) if loader is None else loader, output_directory=path)


def read_strategy(sealed, method):
    return json.loads((Path(sealed["manifest_path"]).parent / (method + "-corridor-strategy.json")).read_text())


def test_generated_data_contract_to_complete_histories_to_independent_positive_contacts(tmp_path):
    task, prepared, arrays = fixture()
    sealed = seal(task, tmp_path / "seals")
    called = []
    def loader():
        assert all((tmp_path / "seals" / (m + "-corridor-strategy.json")).exists() for m in c.METHODS)
        attempt = json.loads((tmp_path / "score/attempt.json").read_text())
        assert attempt["all_complete_methods_preflighted"]
        assert attempt["pre_private_independent_motion_checks"] == 6
        called.append(True)
        return arrays
    result = score(task, prepared, arrays, sealed, tmp_path / "score", loader)
    assert called == [True] and result["status"] == "evaluated_generated_corridor_batch"
    assert result["patient_admission"] is False and result["planning_after_private_load"] is False
    search = result["methods"]["SEARCH"]["evaluation"]
    assert search["public_metrics"]["waypoint_reached"] is True
    assert search["contacts"]["whole_tool"]["positive_reference_cells"] == 2
    assert search["contacts"]["shaft"]["positive_reference_cells"] == 2
    assert search["contacts"]["tip"]["positive_reference_cells"] == 1
    assert search["contacts"]["whole_tool"]["biological_vessel_free"] is None
    history = read_strategy(sealed, "SEARCH")["physical_history"][0]
    first, last = history["motion_legs"]
    assert first["phase"] == "insertion" and last["phase"] == "withdrawal"
    assert first["tip_start_mm"] == last["tip_end_mm"] and first["tip_end_mm"] == last["tip_start_mm"]
    assert "removed_indices_native" not in history and result["tissue_removal_assessed"] is False
    assert np.array_equal(task.support, fixture()[0].support)


def test_new_objective_and_complete_search_are_shared_and_STOP_is_not_route_success(tmp_path):
    task, prepared, arrays = fixture()
    for row in task.candidates:
        assert row["nominal_score"] == pytest.approx(1. - .01 * row["round_trip_mm"])
    sealed = seal(task, tmp_path / "seals")
    search, hybrid, stopped = (read_strategy(sealed, m) for m in ("SEARCH", "HYBRID", "RL"))
    assert search["action_ids"] == hybrid["action_ids"]
    assert search["schema"] == c.STRATEGY_VERSION
    assert search["method_accounting"]["nominal_candidate_comparisons"] == len(task.candidates) + 1
    assert hybrid["method_accounting"]["nominal_candidate_comparisons"] == len(task.candidates) + 1
    assert stopped["terminal_reason"] == "STOP" and stopped["public_metrics"]["waypoint_reached"] is False
    assert stopped["public_metrics"]["nominal_score"] == 0.
    result = score(task, prepared, arrays, sealed, tmp_path / "score")
    assert result["methods"]["RL"]["evaluation"]["corridor_selected"] is False
    assert result["methods"]["RL"]["evaluation"]["contacts"]["whole_tool"]["touched_reference_cells"] == 0


def test_private_world_mutation_changes_scores_not_actor_candidates_or_any_strategy(tmp_path):
    task_a, public_a, arrays_a = fixture(positive=True)
    task_b, public_b, arrays_b = fixture(positive=False)
    assert task_a.fingerprint == task_b.fingerprint and task_a.candidates == task_b.candidates
    assert public_a.actor_contract == public_b.actor_contract
    a = seal(task_a, tmp_path / "a")
    b = seal(task_b, tmp_path / "b")
    for method in c.METHODS:
        assert read_strategy(a, method) == read_strategy(b, method)
    hit = score(task_a, public_a, arrays_a, a, tmp_path / "hit")
    no_hit = score(task_a, public_b, arrays_b, a, tmp_path / "no-hit")
    assert hit["methods"]["SEARCH"]["evaluation"]["contacts"]["whole_tool"]["positive_reference_cells"] == 2
    assert no_hit["methods"]["SEARCH"]["evaluation"]["contacts"]["whole_tool"]["positive_reference_cells"] == 0


def test_no_nominal_generation_or_geometry_preflight_after_private_load(tmp_path, monkeypatch):
    task, prepared, arrays = fixture()
    sealed = seal(task, tmp_path / "seals")
    loaded = []
    for name in ("_generate_candidates", "_preflight", "_scene"):
        old = getattr(c, name)
        def guarded(*args, _old=old, **kwargs):
            assert not loaded
            return _old(*args, **kwargs)
        monkeypatch.setattr(c, name, guarded)
    def loader():
        loaded.append(True)
        return arrays
    assert score(task, prepared, arrays, sealed, tmp_path / "score", loader)["status"] == "evaluated_generated_corridor_batch"


def test_partial_public_coverage_abstains_every_slot_and_never_loads_private(tmp_path):
    task, prepared, arrays = fixture(unknown_plane=True)
    assert all(not row["legal"] for row in task.candidates)
    sealed = seal(task, tmp_path / "seals")
    def forbidden():
        pytest.fail("No complete strategy needs private anatomy")
    result = score(task, prepared, arrays, sealed, tmp_path / "score", forbidden)
    assert result["private_loader_calls"] == 0
    assert all(row == {"planning_status": "abstained", "evaluation": None} for row in result["methods"].values())


@pytest.mark.parametrize("change", [lambda d: d.update(access_radius_mm=.1),
    lambda d: d["tools"][0].update(working_length_mm=50.),
    lambda d: d["tools"][0].update(tip_radius_mm=1.1)])
def test_bad_aperture_missing_proximal_ROI_or_starting_tissue_contact_is_not_legal(change):
    task, _, _ = fixture(definition_change=change)
    assert all(not row["legal"] for row in task.candidates)


def test_failed_and_abstained_control_slots_are_not_silently_STOP(tmp_path):
    task, prepared, arrays = fixture()
    selectors = controls()
    selectors["IL"] = lambda observation: None
    selectors["RL"] = lambda observation: "private-only-invalid-action"
    sealed = seal(task, tmp_path / "seals", selectors)
    report = score(task, prepared, arrays, sealed, tmp_path / "score")
    assert report["methods"]["IL"] == {"planning_status": "abstained", "evaluation": None}
    assert report["methods"]["RL"] == {"planning_status": "failed", "evaluation": None}


@pytest.mark.parametrize("tamper", ["withdrawal", "tool", "score"])
def test_resealed_invalid_history_or_objective_refuses_before_any_private_load(tmp_path, tamper):
    task, prepared, arrays = fixture()
    sealed = seal(task, tmp_path / "seals")
    path = tmp_path / "seals/HYBRID-corridor-strategy.json"
    record = json.loads(path.read_text())
    if tamper == "withdrawal":
        record["physical_history"][0]["motion_legs"].pop()
    elif tamper == "tool":
        record["physical_history"][0]["tool"]["shaft_radius_mm"] = .01
    else:
        record["public_metrics"]["nominal_score"] += .2
    record.pop("seal_hash")
    record["seal_hash"] = semantic_digest(record)
    path.write_text(json.dumps(record))
    manifest = json.loads(Path(sealed["manifest_path"]).read_text())
    import hashlib
    manifest["methods"]["HYBRID"]["strategy_file_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    Path(sealed["manifest_path"]).write_text(json.dumps(manifest))
    sealed["manifest_sha256"] = hashlib.sha256(Path(sealed["manifest_path"]).read_bytes()).hexdigest()
    calls = []
    with pytest.raises(ValueError, match="strategy_record_changed"):
        score(task, prepared, arrays, sealed, tmp_path / "score", lambda: calls.append(True))
    assert calls == []


def test_loader_mutation_cannot_change_later_histories_or_expose_partial_scores(tmp_path):
    task, prepared, arrays = fixture()
    sealed = seal(task, tmp_path / "seals")
    def loader():
        (tmp_path / "seals/HYBRID-corridor-strategy.json").write_text("{}")
        return arrays
    report = score(task, prepared, arrays, sealed, tmp_path / "score", loader)
    assert report["status"] == "evaluation_failed"
    assert all(row["evaluation"] is None for row in report["methods"].values())


def test_caller_private_mapping_replacement_during_callback_does_not_switch_world(tmp_path):
    task, prepared, arrays = fixture()
    private = thaw_json(prepared.evaluator_contract)
    sealed = seal(task, tmp_path / "seals")
    def loader():
        private["registration"]["matrix_ras_mm"][0][3] = 500.
        private["coverage"]["coverage_sha256"] = metadata.digest("changed")
        return arrays
    report = c.evaluate_corridor_batch(task, **sealed, private_contract=private, load_reference=loader, output_directory=tmp_path / "score")
    assert report["status"] == "evaluated_generated_corridor_batch"
    assert report["methods"]["SEARCH"]["evaluation"]["contacts"]["whole_tool"]["positive_reference_cells"] == 2


def test_loaded_positive_or_domain_hash_mismatch_has_no_successful_scores(tmp_path):
    task, prepared, arrays = fixture()
    sealed = seal(task, tmp_path / "seals")
    arrays["registration_domain"][8, 8, 10] = False
    result = score(task, prepared, arrays, sealed, tmp_path / "score")
    assert result["status"] == "evaluation_failed"
    assert all(row["evaluation"] is None for row in result["methods"].values())


def test_private_fields_and_real_domains_cannot_enter_public_task():
    task, _, _ = fixture()
    public = thaw_json(task.actor_contract)
    public["MRA"] = "private-marker"
    with pytest.raises(ValueError, match="exact_public_actor"):
        c.CorridorTask(public, task.t1, task.support, task.coverage, task.definition)
    public = thaw_json(task.actor_contract)
    public["evidence_domain"] = "real_source_receipts"
    with pytest.raises(ValueError, match="real_person_array_admission_not_implemented"):
        c.CorridorTask(public, task.t1, task.support, task.coverage, task.definition)


def test_public_array_binding_and_definition_changes_require_new_contract():
    task, _, _ = fixture()
    image = task.t1.copy()
    image[0, 0, 0] += 1
    with pytest.raises(ValueError, match="array_receipt_mismatch"):
        c.CorridorTask(task.actor_contract, image, task.support, task.coverage, task.definition)
    changed = thaw_json(task.definition)
    changed["effort_per_mm"] = .5
    with pytest.raises(ValueError, match="task_definition_changed"):
        c.CorridorTask(task.actor_contract, task.t1, task.support, task.coverage, changed)


def test_directed_registration_mismatch_refuses_before_private_load(tmp_path):
    task, prepared, arrays = fixture()
    private = thaw_json(prepared.evaluator_contract)
    private["registration"]["direction"] = "MRA_RAS_mm_to_T1_RAS_mm"
    sealed = seal(task, tmp_path / "seals")
    calls = []
    with pytest.raises(ValueError, match="private_registration_binding"):
        c.evaluate_corridor_batch(task, **sealed, private_contract=private, load_reference=lambda: calls.append(True), output_directory=tmp_path / "score")
    assert calls == []


def test_exactly_two_interior_cells_produce_two_distinct_waypoints():
    task, _, _ = fixture()
    support = np.zeros_like(task.support)
    support[6:9, 6:9, 10:14] = True
    actor = thaw_json(task.actor_contract)
    actor["support"]["output_sha256"] = actor["task"]["support_output_sha256"] = array_digest(support)
    two = c.CorridorTask(actor, task.t1, support, task.coverage, task.definition)
    assert two.waypoints == ((7., 7., 11.), (7., 7., 12.))


@pytest.mark.parametrize("method", ["SEARCH", "HYBRID"])
def test_resealed_exhaustive_method_cannot_choose_STOP_over_positive_public_optimum(method):
    task, _, _ = fixture()
    accounting = {"policy_callback_calls": int(method != "SEARCH"), "nominal_candidate_comparisons": len(task.candidates) + 1,
                  "actor_forward_calls": 0, "optimizer_updates": 0}
    record = c._strategy(task, method, "STOP", accounting)
    with pytest.raises(ValueError, match="exhaustive_method_must_choose_public_optimum"):
        c._preflight(task, record, method)


def test_type_only_record_tamper_cannot_retain_original_semantic_seal():
    task, _, _ = fixture()
    record = c._strategy(task, "RL", "STOP", {"policy_callback_calls": 1, "nominal_candidate_comparisons": 0,
        "actor_forward_calls": 0, "optimizer_updates": 0})
    record["horizon"] = True
    with pytest.raises(ValueError, match="strategy_semantic_seal_changed"):
        c._preflight(task, record, "RL")


@pytest.mark.parametrize("part,field,bad", [("support", "kind", "whole_tumor_candidate"),
    ("support", "project_fit_roles", ["SELECT"]), ("support", "pretrained_exposure", "assumed_unseen"),
    ("support", "available_at", "2026-10-08T00:00:00Z"), ("task", "available_at", "2026-10-09T00:00:00Z"),
    ("task", "availability_record_sha256", None)])
def test_prepared_actor_assertions_cannot_be_changed_before_array_build(part, field, bad):
    task, _, _ = fixture()
    actor = thaw_json(task.actor_contract)
    actor[part][field] = bad
    with pytest.raises(ValueError):
        c.CorridorTask(actor, task.t1, task.support, task.coverage, task.definition)


@pytest.mark.parametrize("status,field,bad", [("complete", "nominal_candidate_comparisons", 99),
    ("failed", "actor_forward_calls", 1), ("abstained", "optimizer_updates", True),
    ("failed", "nominal_candidate_comparisons", 99)])
def test_method_accounting_is_strict_for_complete_failed_and_abstained(status, field, bad):
    task, _, _ = fixture()
    accounting = {"policy_callback_calls": 1, "nominal_candidate_comparisons": 0, "actor_forward_calls": 0, "optimizer_updates": 0}
    accounting[field] = bad
    with pytest.raises(ValueError, match="accounting"):
        c._accounting(task, "RL", status, accounting)


def test_type_only_budget_tamper_refuses_before_any_loader(tmp_path):
    import hashlib
    task, prepared, arrays = fixture()
    sealed = seal(task, tmp_path / "seals")
    path = Path(sealed["manifest_path"])
    manifest = json.loads(path.read_text())
    manifest["budgets"]["horizon"] = True
    path.write_text(json.dumps(manifest))
    sealed["manifest_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    calls = []
    with pytest.raises(ValueError, match="batch_manifest_mismatch"):
        score(task, prepared, arrays, sealed, tmp_path / "score", lambda: calls.append(True))
    assert calls == []
