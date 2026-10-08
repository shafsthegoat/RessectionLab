"""Focused longer-fit controls; no patient data or full capacity fit."""
import copy
import json
from pathlib import Path

import pytest
import torch

from resectionlab.data_policy import GeneratedDevelopmentContext
from resectionlab.native_spatial_task import make_native_opening_task
from resectionlab.spatial_policy import SpatialPolicy, SpatialPolicyConfig, parameter_hash, imitation_loss, gradient_step


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def runner(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    import run_native_opening_bc_capacity
    return run_native_opening_bc_capacity


@pytest.fixture
def prepared(runner):
    bundle = runner.load_frozen_teacher(runner.declaration())
    base = make_native_opening_task()
    context = GeneratedDevelopmentContext("1" * 64, (base.case.source_hash,), base.decision_model_hash)
    return base, bundle, context


def initial(runner):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(11)
        model = SpatialPolicy(SpatialPolicyConfig(**runner.specification()["policy"]))
    assert parameter_hash(model) == runner.INITIAL_HASH
    return model


def test_original_audits_and_exact_five_prefixes_reconstruct_without_teacher_search(runner, prepared, monkeypatch):
    base, bundle, context = prepared
    monkeypatch.setattr(runner.prior, "exact_teacher", lambda *_a, **_k: pytest.fail("No new teacher search"))
    before = runner.canonical(base.metrics())
    samples, receipt = runner.reconstruct_teacher(base, bundle, context, lambda: None)
    assert receipt["prefix_transitions"] == 4 and receipt["new_teacher_search_calls"] == 0
    assert len(samples) == 5 and len({o.fingerprint for o, _ in samples}) == 5
    assert sum(label == "STOP" for _, label in samples) == 2
    assert [o.fingerprint for o, _ in samples] == [p["observation_hash"] for p in bundle["teacher-demonstration-checks"]]
    assert runner.canonical(base.metrics()) == before
    assert sum(row["started"] for row in receipt["preview_profile"]["phases"].values()) > 0


@pytest.mark.parametrize("mutation", ["prefix", "label", "inventory", "model"])
def test_prefix_mutations_refused_before_any_learning(runner, prepared, monkeypatch, mutation):
    base, bundle, context = prepared
    bundle = copy.deepcopy(bundle)
    proof = bundle["teacher-demonstration-checks"][0]
    if mutation == "prefix": proof["prefix"] = ["STOP"]
    elif mutation == "label": proof["selected_action_id"] = "STOP"
    elif mutation == "inventory": bundle["rows"][proof["observation_hash"]]["action_ids"].reverse()
    else: bundle["preparation"]["decision_model_hash"] = "sha256:" + "0" * 64
    monkeypatch.setattr(torch.optim, "Adam", lambda *_a, **_k: pytest.fail("No optimizer for invalid inherited states"))
    with pytest.raises(ValueError, match="prefix|source/model|observation/inventory"):
        runner.reconstruct_teacher(base, bundle, context, lambda: None)


@pytest.mark.parametrize("mutation", ["audit", "trajectory", "declaration_proof", "label_observation"])
def test_inherited_audit_or_teacher_lineage_refused_even_before_task(runner, tmp_path, monkeypatch, mutation):
    names = ("teacher", "teacher-demonstration-checks", "result", "preparation", "BC-updates",
             *(f"teacher-state-{i:02}" for i in range(5)))
    for name in names:
        (tmp_path / (name + ".json")).write_bytes((runner.SAVED / (name + ".json")).read_bytes())
    name = "teacher" if mutation == "declaration_proof" else "teacher-state-00"
    path = tmp_path / (name + ".json")
    data = json.loads(path.read_text())
    if mutation == "audit": data["independent_evaluation"]["accepted"] = False
    elif mutation == "trajectory": data["decisions"][0]["action_id"] = "STOP"
    elif mutation == "declaration_proof": data["supervised_state_demonstrations"][0]["prefix"] = []
    else: data["decisions"][1]["observation_hash"] = "sha256:" + "0" * 64
    path.write_text(json.dumps(data))
    monkeypatch.setattr(runner, "SAVED", tmp_path)
    monkeypatch.setattr(runner, "validate", lambda _: None)
    monkeypatch.setattr(torch.optim, "Adam", lambda *_a, **_k: pytest.fail("No optimizer before lineage gate"))
    with pytest.raises(ValueError, match="Teacher prefix|Inherited independent"):
        runner.load_frozen_teacher({})


def test_readout_keeps_weights_rng_and_gradients_and_has_exact_per_state_values(runner, prepared):
    base, bundle, context = prepared
    samples, _ = runner.reconstruct_teacher(base, bundle, context, lambda: None)
    model = initial(runner)
    rng = torch.random.get_rng_state().clone()
    row = runner.readout(model, samples, lambda: None)
    assert row["forward_calls"] == 5 and row["parameter_hash"] == runner.INITIAL_HASH
    assert torch.equal(rng, torch.random.get_rng_state())
    assert all(p.grad is None for p in model.parameters())
    assert row["mean_cross_entropy"] == pytest.approx(1.4025094509124756, abs=1e-7)
    assert all(r["teacher_rank"] >= 1 and len(r["logits"]) == len(r["action_ids"]) for r in row["states"])


def test_one_actual_gradient_matches_original_first_update_and_gate_rejects_early_model(runner, prepared):
    base, bundle, context = prepared
    samples, _ = runner.reconstruct_teacher(base, bundle, context, lambda: None)
    model = initial(runner)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    loss, stats = imitation_loss(model, samples, learning_context=context)
    report = gradient_step(model, optimizer, loss, max_norm=5., learning_context=context)
    assert report == bundle["BC-updates"][0]["gradient"]
    assert stats["loss_forward_calls"] == 5 and report["actor_gradient_norm_before_clip"] > 0
    assert report["critic_gradient_norm_before_clip"] == 0
    with pytest.raises(ValueError, match="Update16 did not reproduce"):
        runner.verify_update16(model, [{}] * 16)


@pytest.mark.parametrize("mutation", ["source", "inputs", "updates", "readouts", "weight_hash"])
def test_prospective_contract_is_closed(runner, mutation):
    record = runner.declaration()
    if mutation == "source": record["source_sha256"].pop("src/resectionlab/spatial_policy.py")
    elif mutation == "inputs": record["input_sha256"].pop("teacher-state-00.json")
    elif mutation == "updates": record = copy.deepcopy(record); record["settings"]["updates"] = 128
    elif mutation == "readouts": record = copy.deepcopy(record); record["settings"]["readouts"] = [256]
    else: record["initial_parameter_hash"] = "sha256:" + "0" * 64
    with pytest.raises(ValueError): runner.validate(record)


def test_preparation_failure_preserves_both_null_evaluations(runner, tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "validate", lambda _: None)
    def fail(_): raise ValueError("fixture withheld")
    monkeypatch.setattr(runner, "load_frozen_teacher", fail)
    with pytest.raises(ValueError, match="fixture withheld"):
        runner.worker(runner.specification(), tmp_path, declaration_sha256="1" * 64)
    result = json.loads((tmp_path / "result.json").read_text())
    assert result["status"] == "failed" and result["updates"] == 0
    assert set(result["methods"]) == {"initial", "BC256"}
    assert all(r["outcomes"] is None for r in result["methods"].values())
