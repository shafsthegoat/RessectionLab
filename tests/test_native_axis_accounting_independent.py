"""Independent accounting adversaries; tiny native arrays, no patient runs."""
import hashlib
import json
import gc
import weakref
from dataclasses import replace

import numpy as np
import pytest

import resectionlab.native_axis_accounting as accounting_module
from resectionlab.geometry import AccessWindow
from resectionlab.native_axis_accounting import AxisTrainingAccounting, AccountingReceiptError
from resectionlab.native_axis_simulation import AxisColumnNativeSimulator, CommittedTransitionInterrupted
from resectionlab.native_proposals import AxisColumnProposalConfig
from resectionlab.native_resection import NATIVE_GENERIC_TOOLS, NativeResectionConfig
from resectionlab.worlds import WorldGeneratorConfig, generate_partitions


def session(tmp_path):
    tissue = np.ones((7, 7, 8), bool)
    labels = np.zeros(tissue.shape, np.int16)
    labels[1:6, 1:6, 2:7] = 1
    cfg = NativeResectionConfig(tissue, labels, np.eye(4),
        AccessWindow((3, 3, -.5), (0, 0, 1), 4), NATIVE_GENERIC_TOOLS,
        "synthetic-axis-accounting-independent-v1", "explicit synthetic support",
        case_id="synthetic_axis_accounting_independent")
    cancelled = {"value": False}
    def construct():
        return AxisColumnNativeSimulator(cfg,
            proposal_config=AxisColumnProposalConfig(((0, 0), (1, 0)), 4), max_steps=2,
            cancelled=lambda: cancelled["value"])
    template = construct()
    worlds = generate_partitions(template.case_hash, WorldGeneratorConfig(), 20261004,
        optimization=2, selection=1, final_evaluation=1, stress=1)
    raws = []
    def factory():
        raw = construct()
        raws.append(raw)
        return raw
    owner = AxisTrainingAccounting(factory, worlds.optimization, worlds.selection,
        receipt_path=tmp_path / "accounting.json", expected_model_hash=template.decision_model_hash)
    return owner, worlds, raws, cancelled


def assert_durable_hash(path):
    record = json.loads(path.read_text())
    digest = record.pop("receipt_hash")
    assert digest == "sha256:" + hashlib.sha256(json.dumps(record, sort_keys=True, allow_nan=False).encode()).hexdigest()


def test_constructor_has_no_role_and_stop_is_returned_without_native_commit(tmp_path):
    owner, worlds, raws, _cancel = session(tmp_path)
    sim = owner.recorded_factory()
    initial = owner.snapshot()
    assert initial["events"][0]["role"] is None
    assert all(row["executed_transitions"] == 0 for row in initial["totals"].values())
    sim.reset(worlds.selection.seeds[0])
    result = sim.step(0)
    assert result.terminated
    record = owner.snapshot()
    assert record["totals"]["selection"] == {
        **record["totals"]["selection"], "executed_transitions": 1, "returned_transitions": 1,
        "native_commits": 0, "committed_but_unreturned": 0, "executed_reward": 0.}
    assert record["totals"]["optimization"]["executed_transitions"] == 0
    assert record["candidate_eligible"] is False and record["resume_supported"] is False
    assert not raws[-1].removed_mask.any()
    assert_durable_hash(owner.receipt_path)


def test_real_postcommit_interrupt_is_durable_and_classified_by_declared_role(tmp_path, monkeypatch):
    owner, worlds, raws, cancel = session(tmp_path)
    sim = owner.recorded_factory()
    sim.reset(worlds.selection.seeds[0])
    raw = raws[-1]
    original = raw.engine.commit_preview
    def commit_then_cancel(preview):
        result = original(preview)
        cancel["value"] = True
        return result
    monkeypatch.setattr(raw.engine, "commit_preview", commit_then_cancel)
    with pytest.raises(CommittedTransitionInterrupted) as caught:
        sim.step(1)
    record = json.loads(owner.receipt_path.read_text())
    totals = record["totals"]["selection"]
    assert totals["executed_transitions"] == totals["native_commits"] == totals["committed_but_unreturned"] == 1
    assert totals["returned_transitions"] == 0
    assert totals["executed_reward"] == pytest.approx(caught.value.reward)
    transition = next(event for event in record["events"] if event.get("native_commit"))
    # Durable JSON canonicalizes immutable coordinate tuples to arrays.
    assert transition["info"] == json.loads(json.dumps(raw._history[-1]))
    assert transition["next_observation_complete"] is False
    assert caught.value.axis_accounting_receipt["receipt_hash"] == record["receipt_hash"]
    assert record["status"] == "failed" and record["candidate_eligible"] is False
    with pytest.raises(RuntimeError, match="closed"):
        owner.recorded_factory()
    assert_durable_hash(owner.receipt_path)


def test_forged_commit_exception_does_not_invent_a_cut(tmp_path, monkeypatch):
    owner, worlds, raws, _cancel = session(tmp_path)
    sim = owner.recorded_factory()
    sim.reset(worlds.optimization.seeds[0])
    forged = CommittedTransitionInterrupted({"action_id": "forged"}, 100.)
    def forged_step(_action):
        raise forged
    monkeypatch.setattr(raws[-1], "step", forged_step)
    with pytest.raises(CommittedTransitionInterrupted) as caught:
        sim.step(1)
    assert caught.value is forged
    record = owner.snapshot()
    assert record["totals"]["optimization"]["native_commits"] == 0
    assert record["totals"]["optimization"]["executed_transitions"] == 0
    assert any(event.get("counts_complete") is False for event in record["events"])
    assert record["counts_complete"] is False
    assert not raws[-1].removed_mask.any()


def test_precommit_cancellation_is_known_zero_work_not_an_unverified_commit(tmp_path, monkeypatch):
    owner, worlds, raws, _cancel = session(tmp_path)
    sim = owner.recorded_factory()
    sim.reset(worlds.optimization.seeds[0])
    def before_step(_action):
        raise InterruptedError("before native commit")
    monkeypatch.setattr(raws[-1], "step", before_step)
    with pytest.raises(InterruptedError, match="before native commit"):
        sim.step(1)
    record = owner.snapshot()
    assert record["totals"]["optimization"]["executed_transitions"] == 0
    assert record["counts_complete"] is True
    assert not raws[-1].removed_mask.any()


def test_receipt_export_failure_retains_committed_work_and_old_durable_file(tmp_path, monkeypatch):
    owner, worlds, raws, _cancel = session(tmp_path)
    sim = owner.recorded_factory()
    sim.reset(worlds.optimization.seeds[0])
    previous = owner.receipt_path.read_bytes()
    def fail_write(*args, **kwargs):
        raise OSError("simulated full disk")
    monkeypatch.setattr(accounting_module, "_atomic_receipt", fail_write)
    with pytest.raises(AccountingReceiptError) as caught:
        sim.step(1)
    receipt = caught.value.receipt
    totals = receipt["totals"]["optimization"]
    assert totals["native_commits"] == totals["executed_transitions"] == totals["committed_but_unreturned"] == 1
    assert totals["returned_transitions"] == 0
    assert len(raws[-1]._history) == 1 and raws[-1].removed_mask.any()
    assert receipt["receipt_export_failure"]["exception"] == "OSError"
    assert owner.receipt_path.read_bytes() == previous
    assert receipt["status"] == "failed" and receipt["resume_supported"] is False


def test_undeclared_reset_does_not_relabel_an_already_completed_episode(tmp_path):
    owner, worlds, _raws, _cancel = session(tmp_path)
    sim = owner.recorded_factory()
    sim.reset(worlds.selection.seeds[0])
    sim.step("STOP")
    with pytest.raises(ValueError, match="declared"):
        sim.reset(worlds.final_evaluation.seeds[0])
    record = owner.snapshot()
    assert record["episodes"][0]["status"] == "complete"
    assert record["totals"]["selection"]["returned_transitions"] == 1
    assert record["status"] == "failed"


def test_out_of_band_valid_native_commit_cannot_disappear_from_accounting(tmp_path):
    owner, worlds, raws, _cancel = session(tmp_path)
    sim = owner.recorded_factory()
    sim.reset(worlds.optimization.seeds[0])
    # The factory retained a raw reference. This is a valid native transition,
    # so engine seals alone do not reveal that the accounting layer missed it.
    raws[-1].step(1)
    assert raws[-1].removed_mask.any()
    with pytest.raises(RuntimeError, match="state|outside|unrecorded|unaccounted|changed"):
        sim.step("STOP")
    record = owner.snapshot()
    assert record["status"] == "failed" and record["counts_complete"] is False
    assert record["candidate_eligible"] is False


def test_replayed_old_postcommit_exception_is_not_double_counted(tmp_path, monkeypatch):
    owner, worlds, raws, _cancel = session(tmp_path)
    sim = owner.recorded_factory()
    sim.reset(worlds.optimization.seeds[0])
    prior = sim.step(1)
    stale = CommittedTransitionInterrupted(prior.info, prior.reward)
    def replayed(_action):
        raise stale
    monkeypatch.setattr(raws[-1], "step", replayed)
    with pytest.raises(CommittedTransitionInterrupted) as caught:
        sim.step("STOP")
    assert caught.value is stale
    totals = owner.snapshot()["totals"]["optimization"]
    assert totals["native_commits"] == totals["returned_transitions"] == 1
    assert totals["committed_but_unreturned"] == 0


def test_receipt_does_not_keep_every_episode_native_volume_alive(tmp_path):
    owner, worlds, raws, _cancel = session(tmp_path)
    first = owner.recorded_factory()
    first.reset(worlds.selection.seeds[0])
    first.step("STOP")
    reference = weakref.ref(raws[-1])
    raws.clear()  # Release the test factory's separate observation hook.
    del first
    gc.collect()
    assert reference() is None, "Accounting owner retained a completed full simulator"
    second = owner.recorded_factory()
    second.reset(worlds.selection.seeds[0])
    second.step("STOP")
    factories = [event for event in owner.snapshot()["events"] if event["kind"] == "factory"]
    assert [event["instance"] for event in factories] == [0, 1]
    assert owner.snapshot()["totals"]["selection"]["returned_transitions"] == 2


def test_original_postcommit_exception_survives_unwritable_receipt(tmp_path, monkeypatch):
    owner, worlds, raws, cancel = session(tmp_path)
    sim = owner.recorded_factory()
    sim.reset(worlds.optimization.seeds[0])
    previous = owner.receipt_path.read_bytes()
    original = raws[-1].engine.commit_preview
    def commit_then_cancel(preview):
        result = original(preview)
        cancel["value"] = True
        return result
    monkeypatch.setattr(raws[-1].engine, "commit_preview", commit_then_cancel)
    def fail_write(*args, **kwargs):
        raise OSError("simulated full disk during cancellation")
    monkeypatch.setattr(accounting_module, "_atomic_receipt", fail_write)
    with pytest.raises(CommittedTransitionInterrupted) as caught:
        sim.step(1)
    receipt = caught.value.axis_accounting_receipt
    assert receipt["totals"]["optimization"]["native_commits"] == 1
    assert receipt["totals"]["optimization"]["returned_transitions"] == 0
    assert receipt["receipt_export_failure"]["exception"] == "OSError"
    assert receipt["status"] == "failed" and owner.receipt_path.read_bytes() == previous


@pytest.mark.parametrize("reported_as_exception", [False, True])
def test_verified_cut_survives_forged_reported_reward_or_exception_info(tmp_path, monkeypatch, reported_as_exception):
    owner, worlds, raws, _cancel = session(tmp_path)
    sim = owner.recorded_factory()
    sim.reset(worlds.optimization.seeds[0])
    raw = raws[-1]
    original = raw.step
    actual = {}
    def bad_report(action):
        result = original(action)
        actual["reward"] = result.reward
        if reported_as_exception:
            error = CommittedTransitionInterrupted({**result.info, "forged_field": True}, result.reward + 1000.)
            actual["error"] = error
            raise error
        return replace(result, reward=result.reward + 1000.)
    monkeypatch.setattr(raw, "step", bad_report)
    with pytest.raises(CommittedTransitionInterrupted if reported_as_exception else RuntimeError) as caught:
        sim.step(1)
    if reported_as_exception:
        assert caught.value is actual["error"]
    record = owner.snapshot()
    totals = record["totals"]["optimization"]
    assert totals["native_commits"] == totals["executed_transitions"] == totals["committed_but_unreturned"] == 1
    assert totals["returned_transitions"] == 0
    assert totals["executed_reward"] == pytest.approx(actual["reward"])
    assert record["counts_complete"] is True
    assert record["status"] == "failed" and record["candidate_eligible"] is False
