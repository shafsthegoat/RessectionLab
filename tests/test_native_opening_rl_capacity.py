"""Small synthetic controls; no long fit or patient access."""
import copy
import json
from pathlib import Path

import pytest
import torch

from resectionlab.data_policy import GeneratedDevelopmentContext
from resectionlab.native_spatial_task import make_native_opening_task
from resectionlab.spatial_policy import SpatialPolicy, SpatialPolicyConfig, parameter_hash, reinforce_loss, gradient_step


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def runner(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    import run_native_opening_rl_capacity
    return run_native_opening_rl_capacity


@pytest.fixture
def saved(runner):
    return runner.load_evaluation_tree(runner.declaration())


def initial(runner):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(11)
        model = SpatialPolicy(SpatialPolicyConfig(**runner.specification()["policy"]))
    assert parameter_hash(model) == runner.INITIAL_HASH
    return model


def test_all_sixteen_assessment_routes_come_from_audited_original_episodes(runner, saved):
    bundle, tree = saved
    assert len(tree["routes"]) == 16 and len(tree["original_episodes"]) == 64
    assert len(tree["state_actions"]) == 5
    assert min(r["nominal_return"] for r in tree["routes"]) == pytest.approx(-.896)
    assert max(r["nominal_return"] for r in tree["routes"]) == pytest.approx(1.1)
    assert sum(r["full_target"] for r in tree["routes"]) == 2


def test_five_forward_assessment_matches_independent_saved_initial_expectation(runner, saved):
    bundle, tree = saved
    base, model = make_native_opening_task(), initial(runner)
    context = GeneratedDevelopmentContext("1" * 64, (base.case.source_hash,), base.decision_model_hash)
    samples, _ = runner.evaluation.reconstruct_teacher(base, bundle, context, lambda: None)
    rng = torch.random.get_rng_state().clone()
    readout = runner.evaluation.readout(model, samples, lambda: None)
    metrics = runner.expected_metrics(readout, tree)
    assert metrics["nominal_expected_return"] == pytest.approx(-.2754633654, abs=1e-10)
    assert metrics["generated_any_target_probability"] == pytest.approx(.235496, abs=1e-6)
    assert metrics["terminal_probability_sum"] == pytest.approx(1., abs=1e-12)
    assert parameter_hash(model) == runner.INITIAL_HASH and torch.equal(rng, torch.random.get_rng_state())
    assert all(p.grad is None for p in model.parameters())


@pytest.mark.parametrize("mutation", ["missing_state", "duplicate_state", "action_order", "negative", "nan", "wrong_mass"])
def test_incomplete_or_corrupted_readout_cannot_become_an_expectation(runner, saved, mutation):
    _, tree = saved
    rows = [{"observation_hash": key, "action_ids": list(ids), "probabilities": [1/len(ids)] * len(ids)}
            for key, ids in tree["state_actions"].items()]
    if mutation == "missing_state": rows.pop()
    elif mutation == "duplicate_state": rows[-1] = copy.deepcopy(rows[0])
    elif mutation == "action_order": rows[0]["action_ids"].reverse()
    else: rows[0]["probabilities"][0] = {"negative": -.1, "nan": float("nan"), "wrong_mass": .8}[mutation]
    with pytest.raises(ValueError): runner.expected_metrics({"states": rows}, tree)


def test_one_real_on_policy_batch_reproduces_original_actions_and_gradient(runner, saved, tmp_path):
    _, tree = saved
    base, model = make_native_opening_task(), initial(runner)
    permit = GeneratedDevelopmentContext("1" * 64, (base.case.source_hash,), base.decision_model_hash)
    rng, batch = torch.Generator().manual_seed(11), []
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    for index in range(4):
        episode, report, _ = runner.prior.guarded_episode(base, model, tmp_path, f"unit-{index}", lambda: None,
            mode="sample", generator=rng)
        runner.verify_original_episode(1, index, report, tree)
        batch.append(episode)
    loss, stats = reinforce_loss(model, batch, gamma=1., entropy_weight=.01, value_weight=.5, learning_context=permit)
    gradient = gradient_step(model, optimizer, loss, max_norm=5., learning_context=permit)
    runner.verify_original_update(1, {"loss": stats, "gradient": gradient}, tree)
    assert gradient["parameters_changed"] and stats["completed_episodes"] == 4
    assert stats["loss_forward_calls"] == sum(map(len, batch)) == 7


@pytest.mark.parametrize("mutation", ["reward", "action", "gradient"])
def test_original_prefix_mismatch_refuses_continuation(runner, saved, mutation):
    _, tree = saved
    if mutation == "gradient":
        row = copy.deepcopy(tree["original_updates"][0]); row["gradient"]["updated_parameter_hash"] = "bad"
        with pytest.raises(ValueError, match="gradient/hash"):
            runner.verify_original_update(1, row, tree)
    else:
        row = copy.deepcopy(tree["original_episodes"]["rl-00-0.json"])
        row["decisions"][0]["reward" if mutation == "reward" else "action_id"] = .1 if mutation == "reward" else "bad"
        with pytest.raises(ValueError, match="sampled observation/action/reward"):
            runner.verify_original_episode(1, 0, row, tree)


@pytest.mark.parametrize("mutation", ["settings", "sources", "input"])
def test_closed_declaration(runner, mutation):
    record = copy.deepcopy(runner.declaration())
    if mutation == "settings": record["settings"]["episodes_per_update"] = 8
    elif mutation == "sources": record["source_sha256"].pop("src/resectionlab/spatial_policy.py")
    else: record["input_sha256"].pop("rl-00-0.json")
    with pytest.raises(ValueError): runner.validate(record)


def test_early_failure_retains_null_final_and_no_updates(runner, monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "validate", lambda _: None)
    def refuse(_): raise ValueError("missing complete saved tree")
    monkeypatch.setattr(runner, "load_evaluation_tree", refuse)
    with pytest.raises(ValueError, match="missing complete saved tree"):
        runner.worker(runner.specification(), tmp_path, declaration_sha256="1" * 64)
    result = json.loads((tmp_path / "result.json").read_text())
    assert result["status"] == "failed" and result["updates"] == 0
    assert result["completed_training_episodes"] == 0
    assert result["methods"]["RL256"]["outcomes"] is None
