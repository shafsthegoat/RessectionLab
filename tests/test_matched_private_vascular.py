"""Scripted generated interface controls, never learned-method performance."""
from dataclasses import replace
import json
import sys
from pathlib import Path
import tempfile

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
from resectionlab import private_vascular_evaluation as v
from resectionlab import matched_private_vascular as adapter
import test_private_vascular_evaluation as generated
from resectionlab import research_estimate_planning as planning
from resectionlab.core import semantic_digest


@pytest.fixture
def tmp_path():
    with tempfile.TemporaryDirectory(prefix="matched-private-", dir=ROOT / "build") as path:
        yield Path(path)


@pytest.fixture(scope="module")
def fixture():
    spec = generated.helpers.fixture_spec()
    public = generated.identity(spec)
    plans, rows = {}, {}
    for method in adapter.METHODS:
        rule = generated.helpers.stop_planner if method == "RL" else generated.fixed_moves
        plan = planning.research_planning_from_estimates(spec, method=method,
            configuration_hash=semantic_digest({"scripted_interface_control": method}), planner=rule)
        plans[method] = plan
        rows[method] = {"status": "complete", "strategy": planning.research_strategy_to_record(plan, spec),
            "outcomes": None, "details": {"actor_forward_calls": 0, "scripted_actions_only": True}}
    suite = {"version": adapter.MATCHED_VERSION, "input_hash": spec.fingerprint,
        "method_order": list(adapter.METHODS), "methods": rows,
        "training_accounting": {"IL": {"lineage": "scripted_interface_control_not_learned"},
            "RL": {"lineage": "scripted_interface_control_not_learned"}},
        "real_patient_count": 0, "optimizer_updates": 0}
    suite["seal_hash"] = semantic_digest(suite)
    return spec, public, plans, suite


def stage(fixture, path, suite=None):
    spec, public, plans, original = fixture
    sealed = adapter.write_matched_vascular_seals(original if suite is None else suite,
        spec=spec, planning_identity=public, output_directory=path)
    manifest = json.loads(Path(sealed["manifest_path"]).read_text())
    return sealed, manifest


def references(fixture, sealed, manifest, *, positive=True):
    spec, public, plans, _ = fixture
    folder = Path(sealed["manifest_path"]).parent
    return {method: generated.reference((spec, plans[method], public,
                folder / row["strategy_filename"], row["strategy_file_sha256"]), positive=positive)
            for method, row in manifest["methods"].items() if row["planning_status"] == "complete"}


def evaluate(fixture, sealed, refs, out, loaders=None):
    spec, public, _, _ = fixture
    return adapter.evaluate_matched_private_vessels(**sealed, spec=spec, planning_identity=public,
        reference_bindings={m: r.binding for m, r in refs.items()},
        load_references=loaders or {m: lambda r=r: r for m, r in refs.items()}, output_directory=out)


def resigned(suite):
    copied = json.loads(json.dumps(suite))
    copied.pop("seal_hash", None)
    copied["seal_hash"] = semantic_digest(copied)
    return copied


def test_all_strategies_and_nominal_preflights_precede_first_private_loader(fixture,tmp_path,monkeypatch):
    sealed, manifest = stage(fixture, tmp_path / "sealed")
    refs = references(fixture, sealed, manifest)
    loaded = []
    for module, name in ((planning, "_replay_strategy_nominal"),
                         (v, "_replay_strategy_nominal"), (v, "independent_check_native_history")):
        old = getattr(module, name)
        def guarded(*args, _old=old, **kwargs):
            assert loaded == [], "nominal work occurred after private access"
            return _old(*args, **kwargs)
        monkeypatch.setattr(module, name, guarded)
    def loader(method):
        assert all((tmp_path / "sealed" / (m + "-vascular-seal.json")).is_file() for m in adapter.METHODS)
        attempt = json.loads((tmp_path / "evaluation/attempt.json").read_text())
        assert set(attempt["pre_private_accounting"]) == set(adapter.METHODS)
        loaded.append(method)
        return refs[method]
    report = evaluate(fixture, sealed, refs, tmp_path / "evaluation",
                      {m: lambda m=m: loader(m) for m in refs})
    assert loaded == list(adapter.METHODS)
    assert report["all_complete_methods_preflighted_before_private_load"]
    for method in ("SEARCH", "IL", "HYBRID"):
        result = report["methods"][method]["evaluation"]
        assert result["whole_tool"]["positive_reference_cells"] == 2
        assert result["removed_overlap"]["positive_overlap_cells"] == 1
        assert result["microstep_count"] > 2 and result["nonstop_sweeps"] == 2
    stopped = report["methods"]["RL"]["evaluation"]
    assert stopped["nonstop_sweeps"] == 0 and stopped["whole_tool"]["touched_reference_cells"] == 0
    assert stopped["clinical_injury_probability"] is None
    assert all(row["evaluation"]["planning_after_private_load"] is False for row in report["methods"].values())


def test_reference_swap_changes_only_private_scores_for_exact_same_method_seals(fixture,tmp_path):
    sealed, manifest = stage(fixture, tmp_path / "sealed")
    before = {p.name: p.read_bytes() for p in (tmp_path / "sealed").iterdir()}
    hit = evaluate(fixture, sealed, references(fixture, sealed, manifest), tmp_path / "hit")
    empty = evaluate(fixture, sealed, references(fixture, sealed, manifest, positive=False), tmp_path / "empty")
    for method in adapter.METHODS:
        a, b = (r["methods"][method]["evaluation"] for r in (hit, empty))
        assert a["strategy_file_sha256"] == b["strategy_file_sha256"]
        assert a["full_history_hash"] == b["full_history_hash"]
    assert hit["methods"]["SEARCH"]["evaluation"]["whole_tool"]["annotated_positive_encounter"] is True
    assert empty["methods"]["SEARCH"]["evaluation"]["whole_tool"]["annotated_positive_encounter"] is False
    assert before == {p.name: p.read_bytes() for p in (tmp_path / "sealed").iterdir()}


def test_failed_and_abstained_methods_remain_null_slots(fixture,tmp_path):
    suite = json.loads(json.dumps(fixture[3]))
    for method, status in (("RL", "failed"), ("HYBRID", "abstained")):
        suite["methods"][method].update(status=status, strategy=None, reason="scripted_control")
    sealed, manifest = stage(fixture, tmp_path / "sealed", resigned(suite))
    report = evaluate(fixture, sealed, references(fixture, sealed, manifest), tmp_path / "evaluation")
    assert set(report["methods"]) == set(adapter.METHODS)
    for method, status in (("RL", "failed"), ("HYBRID", "abstained")):
        assert report["methods"][method] == {"planning_status": status, "evaluation_status": "not_evaluated", "evaluation": None}


def test_last_corrupted_seal_refuses_before_any_private_loader(fixture,tmp_path):
    sealed, manifest = stage(fixture, tmp_path / "sealed")
    refs = references(fixture, sealed, manifest)
    path = tmp_path / "sealed/HYBRID-vascular-seal.json"
    value = json.loads(path.read_text())
    value["strategy"]["physical_history"][0]["microsteps"][0]["tip_end_mm"][0] += .1
    path.write_text(json.dumps(value))
    calls = []
    with pytest.raises(ValueError, match="seal_file_changed"):
        evaluate(fixture, sealed, refs, tmp_path / "evaluation", {m: lambda: calls.append(True) for m in refs})
    assert calls == [] and not (tmp_path / "evaluation").exists()


def test_methods_cannot_receive_different_private_reference_worlds(fixture,tmp_path):
    sealed, manifest = stage(fixture, tmp_path / "sealed")
    refs = references(fixture, sealed, manifest)
    refs["RL"] = references(fixture, sealed, manifest, positive=False)["RL"]
    calls = []
    with pytest.raises(ValueError, match="same_private_reference"):
        evaluate(fixture, sealed, refs, tmp_path / "evaluation", {m: lambda: calls.append(True) for m in refs})
    assert calls == []


def test_wrong_method_binding_refuses_before_any_private_loader(fixture,tmp_path):
    sealed, manifest = stage(fixture, tmp_path / "sealed")
    refs = references(fixture, sealed, manifest)
    refs["IL"] = refs["SEARCH"]
    calls = []
    with pytest.raises(ValueError, match="reference_person_or_planning_source_mismatch"):
        evaluate(fixture, sealed, refs, tmp_path / "evaluation", {m: lambda: calls.append(True) for m in refs})
    assert calls == []


def test_private_loader_failure_preserves_denominator_without_private_error_text(fixture,tmp_path):
    sealed, manifest = stage(fixture, tmp_path / "sealed")
    refs = references(fixture, sealed, manifest)
    calls = []
    def fail():
        calls.append("IL")
        raise OSError("SECRET_PRIVATE_DATA")
    loaders = {m: lambda r=r: r for m, r in refs.items()}
    loaders["IL"] = fail
    report = evaluate(fixture, sealed, refs, tmp_path / "evaluation", loaders)
    assert calls == ["IL"] and len(report["methods"]) == 4
    assert report["methods"]["IL"]["evaluation_status"] == "evaluation_failed"
    assert report["methods"]["IL"]["evaluation"]["outcomes"] is None
    assert "SECRET_PRIVATE_DATA" not in json.dumps(report)


def test_method_substitution_or_unfinished_slot_cannot_be_resealed(fixture,tmp_path):
    for change in ("substitute", "unfinished"):
        suite = json.loads(json.dumps(fixture[3]))
        if change == "substitute":
            suite["methods"]["IL"]["strategy"] = suite["methods"]["SEARCH"]["strategy"]
        else:
            suite["methods"]["IL"]["status"] = "started"
        with pytest.raises(ValueError):
            stage(fixture, tmp_path / change, resigned(suite))
        assert not (tmp_path / change).exists()


def test_generated_t1_alias_is_not_relabelled_to_t1(fixture,tmp_path):
    spec, public, _, suite = fixture
    source = replace(spec.sources[0], modality="GENERATED_T1")
    source_hash = planning.source_set_hash((source,))
    support, target = (replace(e, source_set_sha256=source_hash) for e in (spec.support, spec.target))
    declaration = replace(spec.declaration, source_set_sha256=source_hash,
        support_identity_sha256=semantic_digest(support.identity()), target_identity_sha256=semantic_digest(target.identity()))
    changed = replace(spec, sources=(source,), actor_modality="GENERATED_T1", support=support, target=target, declaration=declaration)
    with pytest.raises(ValueError, match="bounded_T1_spec_required"):
        adapter.write_matched_vascular_seals(suite, spec=changed, planning_identity=public, output_directory=tmp_path / "alias")


def test_real_person_role_is_not_admitted(fixture,tmp_path):
    spec, public, _, suite = fixture
    public = {**public, "person_id": "IXI001", "role": "HELD_OUT_TEST", "source_domain": "real_person"}
    with pytest.raises(ValueError, match="real_person_admission_not_implemented"):
        adapter.write_matched_vascular_seals(suite, spec=spec, planning_identity=public, output_directory=tmp_path / "patient")


def test_callback_mapping_replacement_cannot_change_later_reference(fixture,tmp_path):
    sealed, manifest = stage(fixture, tmp_path / "sealed")
    refs = references(fixture, sealed, manifest)
    foreign = references(fixture, sealed, manifest, positive=False)["IL"]
    bindings = {m: r.binding for m, r in refs.items()}
    calls = []
    loaders = {m: lambda m=m, r=r: (calls.append(m), r)[1] for m, r in refs.items()}
    def first():
        bindings["IL"] = foreign.binding
        loaders["IL"] = lambda: (calls.append("replacement"), foreign)[1]
        calls.append("SEARCH")
        return refs["SEARCH"]
    loaders["SEARCH"] = first
    report = adapter.evaluate_matched_private_vessels(**sealed, spec=fixture[0], planning_identity=fixture[1],
        reference_bindings=bindings, load_references=loaders, output_directory=tmp_path / "evaluation")
    assert calls == list(adapter.METHODS)
    assert report["methods"]["IL"]["evaluation"]["whole_tool"]["positive_reference_cells"] == 2
    assert report["status"] == "completed_fixed_generated_method_denominator"


@pytest.mark.parametrize("mutator,victim", [("SEARCH", "IL"), ("HYBRID", "SEARCH")])
def test_consistently_resigned_binding_mutation_is_rejected_against_captured_identity(fixture,tmp_path,mutator,victim):
    sealed, manifest = stage(fixture, tmp_path / "sealed")
    refs = references(fixture, sealed, manifest)
    foreign = references(fixture, sealed, manifest, positive=False)[victim]
    bindings = {m: r.binding for m, r in refs.items()}
    calls = []
    loaders = {m: lambda m=m, r=r: (calls.append(m), r)[1] for m, r in refs.items()}
    def mutate():
        binding = bindings[victim]
        object.__setattr__(binding, "mask_hash", foreign.binding.mask_hash)
        object.__setattr__(binding, "fingerprint", semantic_digest(binding.record()))
        binding.assert_intact()  # Self-consistency alone must not authorize the swap.
        calls.append(mutator)
        return refs[mutator]
    loaders[mutator] = mutate
    report = adapter.evaluate_matched_private_vessels(**sealed, spec=fixture[0], planning_identity=fixture[1],
        reference_bindings=bindings, load_references=loaders, output_directory=tmp_path / "evaluation")
    assert report["status"] == "batch_integrity_failed"
    if victim == "IL":
        assert victim not in calls
        assert report["methods"][victim]["evaluation_status"] == "evaluation_failed"
        assert report["methods"][victim]["evaluation"] is None
    else:
        assert calls == list(adapter.METHODS)


@pytest.mark.parametrize("field,value", [("real_patient_count", 1), ("optimizer_updates", 1),
    ("real_patient_count", False), ("optimizer_updates", 0.)])
def test_generated_frozen_session_headers_are_exact_integers(fixture,tmp_path,field,value):
    suite = json.loads(json.dumps(fixture[3]))
    suite[field] = value
    with pytest.raises(ValueError, match="generated_frozen_planning_session"):
        stage(fixture, tmp_path / "bad", resigned(suite))


def test_prescored_outcomes_cannot_enter_planning_seal_stage(fixture,tmp_path):
    suite = json.loads(json.dumps(fixture[3]))
    suite["methods"]["RL"]["outcomes"] = {"private_vascular_score": 0}
    with pytest.raises(ValueError, match="unfinished_method_slot"):
        stage(fixture, tmp_path / "prescored", resigned(suite))


def test_no_policy_library_imported():
    assert "torch" not in sys.modules
