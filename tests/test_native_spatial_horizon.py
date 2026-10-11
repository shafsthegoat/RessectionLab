"""Declared horizon controls over generated geometry; no patient/model work."""
import inspect

import numpy as np
import pytest

from resectionlab.core import semantic_digest
from resectionlab.geometry import AccessWindow, ToolGeometry
from resectionlab.native_spatial_task import (
    MAX_NATIVE_SPATIAL_STEPS, NativeSpatialCase, NativeSpatialTask, make_native_opening_task)
from resectionlab.patient_planning_admission import make_patient_planning_task
from resectionlab.simulation import InvalidActionError
from test_patient_planning_admission import fixture_contract


def independent_cells_case():
    # 24 separated exposed cells permit 24 genuine, one-cell native removals.
    # This is a horizon mechanism control, not a corridor reachability claim.
    support = np.zeros((20, 14, 3), bool)
    for x in (2, 5, 8, 11, 14, 17):
        for y in (2, 5, 8, 11):
            support[x, y, 1] = True
    return NativeSpatialCase(support.astype(np.float32), support, support, np.eye(4),
        AccessWindow((9.5, 6.5, .5), (0., 0., 1.), 20., "generated-horizon-control"),
        (ToolGeometry("generated-cell-remover", .9, .2, 5., 30., 2.),),
        track="synthetic_scan", support_source_kind="derived_from_scan",
        support_derivation="generated isolated occupied cells",
        nominal_target=support, target_source_kind="derived_from_scan",
        target_derivation="generated occupied-cell objective", crop_shape=(20, 14, 3))


def remove_first_remaining_cell(task):
    row = next(row for row in task.candidate_inventory()["ledger"]
               if row["feasible"] and task._engine.remaining_mask[tuple(row["voxel"])])
    result = task.step(row["action_id"])
    assert result.info["removed_indices_native"] == [row["voxel"]]
    return result


@pytest.mark.parametrize("horizon", [1, 6, 24, 48])
def test_declared_horizon_bound_and_context_propagate(horizon):
    case, args = fixture_contract()
    args["protocol"]["max_steps"] = horizon
    task, context = make_patient_planning_task(case, **args)
    context.require_training()
    assert task.max_steps == context.max_steps == horizon
    assert context.record()["max_optimizer_updates"] == 2
    assert context.record()["budgets"] == {key: args["protocol"][key] for key in
        ("search", "max_native_previews", "max_policy_forwards", "worker_seconds", "memory_bytes", "threads")}
    assert context.record()["protocol_sha256"] == semantic_digest(args["protocol"])
    for copy in (task.fresh(), task.clone(), task.planning_clone(), task.planning_clone().fresh()):
        assert copy.max_steps == horizon and copy.decision_model_hash == task.decision_model_hash
        assert copy.observation().state_features[1] == horizon
        context.require_task(copy)
        context.require_observations([copy.observation()])


@pytest.mark.parametrize("horizon", [0, 49, True, False, 24., 48.])
def test_invalid_or_nontyped_horizon_refuses_in_native_and_admission(horizon):
    case, args = fixture_contract()
    with pytest.raises(ValueError, match="horizon"):
        NativeSpatialTask(case, max_steps=horizon)
    args["protocol"]["max_steps"] = horizon
    with pytest.raises(ValueError):
        make_patient_planning_task(case, **args)


def test_horizon_changes_existing_identity_and_cross_context_is_refused():
    case, args = fixture_contract()
    args["protocol"]["max_steps"] = 6
    six, context_six = make_patient_planning_task(case, **args)
    args["protocol"]["max_steps"] = 24
    longer, context_longer = make_patient_planning_task(case, **args)
    assert six.case.source_hash == longer.case.source_hash
    assert six.decision_model_hash != longer.decision_model_hash
    assert six.observation().fingerprint != longer.observation().fingerprint
    assert context_six.fingerprint != context_longer.fingerprint
    with pytest.raises(ValueError):
        context_six.require_task(longer)
    with pytest.raises(ValueError):
        context_six.require_observations([longer.observation()])


def test_numpy_integer_keeps_existing_boundary_specific_behavior():
    case, args = fixture_contract()
    with pytest.raises(ValueError, match="horizon"):
        NativeSpatialTask(case, max_steps=np.int64(24))
    # Admission already normalizes JSON-compatible NumPy scalars into Python
    # scalars before checking the protocol. The horizon extension preserves it.
    args["protocol"]["max_steps"] = np.int64(24)
    task, context = make_patient_planning_task(case, **args)
    assert type(task.max_steps) is int and task.max_steps == context.max_steps == 24


def test_twenty_four_actual_commits_terminate_at_declared_boundary():
    task = NativeSpatialTask(independent_cells_case(), max_steps=24)
    for count in range(1, 25):
        result = remove_first_remaining_cell(task)
        assert result.terminated == (count == 24)
        assert task.metrics()["steps"] == count
        assert task.metrics()["simulated_removed_volume_mm3"] == count
        assert result.observation.state_features[0] == count
        assert result.observation.state_features[1] == 24
        if count == 12:
            for branch in (task.clone(), task.planning_clone()):
                assert branch.metrics()["steps"] == 12 and branch.max_steps == 24
                clean = lambda rows: [{key: value for key, value in row.items() if key != "outcome_scope"}
                                      for row in rows]
                assert clean(branch.metrics()["history"]) == clean(task.metrics()["history"])
            assert task.fresh().metrics()["steps"] == 0
    assert task.candidate_inventory()["remaining_steps"] == 0
    assert task.observation().action_ids == ("STOP",)
    assert task.independent_geometry_check().feasible
    with pytest.raises(InvalidActionError, match="terminated"):
        task.step("STOP")


def test_stop_still_terminates_early_at_longer_horizon():
    task = NativeSpatialTask(independent_cells_case(), max_steps=24)
    result = task.step("STOP")
    assert result.terminated and result.reward == 0
    assert task.metrics()["steps"] == 1 and task.metrics()["simulated_removed_volume_mm3"] == 0


def test_six_real_commits_keep_exact_pre_extension_outputs():
    task = NativeSpatialTask(independent_cells_case(), max_steps=6)
    assert task.observation().fingerprint == "sha256:b7c3e78d771b57825aa5d1f85c3566c7a53645a3f6769c39b8f7049b9533a42c"
    for count in range(1, 7):
        assert remove_first_remaining_cell(task).terminated == (count == 6)
    assert semantic_digest(task.metrics()) == "sha256:9836199f69e82ce17a5c1b827657d66f7648acdaa2c36f03de1c3b95abdd644f"
    assert task.observation().fingerprint == "sha256:e69197c30b7e6b9dcdc3cd7680e57c5f87d8f5d6482a19413692fb0c99a7379e"
    assert semantic_digest(task.metrics()["history"]) == "sha256:03616401b64bf77bcb0c831937692177e61fc9712d532bf163c0a2d5a8cc88ea"
    assert semantic_digest(task.candidate_inventory()) == "sha256:f22113b86705fd1f7926a071326815a71b015a03882ae572698f14f2e8ac407d"


@pytest.mark.parametrize("horizon,model,observation,inventory,stop_metrics,stop_observation", [
    (3, "874aa851f2325566f2b9cb7eaaf332cd85c05fe25f71b488ef9d669edcf5f4c8",
     "d84c0d00ed23f74f0bfe43f5b94f9c418e7c2fae631cc0c997c0a04c40e76853",
     "f77d5596378b86537a7bb1ad392b796085c735fd1f1b46ac4814e30641638c69",
     "05c8c92f77a4dba5fad8848ba035b9be4f247e5d20b22bfe28d0dc5ce067714d",
     "2081f8a91fc13024c1021f2acc53b8ae1c7b5acd241d352f647033f96bb6072d"),
    (6, "5b29562d84aa86557895cd20813496223d5d1953de75e1f96c2cc4b20f027c57",
     "09d7cd6284c093ea84ba17fd598b7fe1e3da8242520dd5a77630549491ef4ea2",
     "14c6ea89249c6d357f56327f1d5d3263726a9ff5113f8d3ebf34dd0b1c1f7d67",
     "da7b5199b219e5f420bc7ec984a063810c4f7ada168c436b910d4defec383a73",
     "1f36c6e456cf335b79925a24eb4e9ec1473672e28b941a0c246abc8e904e1d17"),
])
def test_unchanged_default_and_six_step_contracts(horizon, model, observation, inventory,
                                                stop_metrics, stop_observation):
    # Exact baseline fingerprints captured from canonical code before this patch.
    assert MAX_NATIVE_SPATIAL_STEPS == 48
    assert inspect.signature(NativeSpatialTask).parameters["max_steps"].default == 3
    case = make_native_opening_task().case
    task = NativeSpatialTask(case) if horizon == 3 else NativeSpatialTask(case, max_steps=horizon)
    assert task.case.source_hash == "sha256:9b39c31e1e012b0822d538451a0452fa9c564acc863d22dabb311ec4cf8c5d4e"
    assert task.decision_model_hash == "sha256:" + model
    assert task.observation().fingerprint == "sha256:" + observation
    assert semantic_digest(task.candidate_inventory()) == "sha256:" + inventory
    task.step("STOP")
    assert semantic_digest(task.metrics()) == "sha256:" + stop_metrics
    assert task.observation().fingerprint == "sha256:" + stop_observation


def test_twenty_four_does_not_relax_select_gradient_or_runtime_limits():
    case, args = fixture_contract("ReMIND-013", "SELECT")
    args["protocol"]["max_steps"] = 24
    _, context = make_patient_planning_task(case, **args)
    with pytest.raises(ValueError, match="TRAIN_gradient"):
        context.require_training()
    args["protocol"]["max_optimizer_updates"] = 1
    with pytest.raises(ValueError, match="role_and_budget"):
        make_patient_planning_task(case, **args)
    args["protocol"]["max_optimizer_updates"] = 0
    args["protocol"]["max_native_previews"] = 0
    with pytest.raises(ValueError, match="runtime_budgets"):
        make_patient_planning_task(case, **args)


def test_forty_eight_actual_commits_and_history_stop_boundary():
    support = np.zeros((20, 26, 3), bool)
    for x in (2, 5, 8, 11, 14, 17):
        for y in (2, 5, 8, 11, 14, 17, 20, 23): support[x, y, 1] = True
    case = NativeSpatialCase(support.astype(np.float32), support, support, np.eye(4),
        AccessWindow((9.5,12.5,.5),(0.,0.,1.),20.,"generated-h48"),
        (ToolGeometry("generated-cell-remover",.9,.2,5.,30.,2.),),track="synthetic_scan",
        support_source_kind="derived_from_scan",support_derivation="generated48 isolated cells",
        nominal_target=support,target_source_kind="derived_from_scan",target_derivation="generated goal",
        crop_shape=(20,26,3))
    task=NativeSpatialTask(case,max_steps=48)
    for count in range(1,49):
        outcome=remove_first_remaining_cell(task)
        assert outcome.terminated == (count==48)
        if count==24:
            assert not task.terminated and task.clone().max_steps==task.planning_clone().max_steps==48
    assert task.metrics()["simulated_removed_volume_mm3"]==48 and task.independent_geometry_check().feasible
    with pytest.raises(InvalidActionError):task.step("STOP")


def test_h24_exact_pre_extension_identity_and_first23_real_layers():
    from dataclasses import fields,replace
    import hashlib,importlib.util,os,sys
    from pathlib import Path
    from resectionlab.observed_search import observed_beam_search
    root=next(p for p in Path(__file__).resolve().parents if (p/'src/resectionlab').is_dir())
    case=independent_cells_case()
    a=NativeSpatialTask(case,max_steps=24);b=NativeSpatialTask(case,max_steps=48)
    # Native physical transitions, rewards and available actions agree before
    # the old terminal boundary; only declared model/state seals differ.
    for _ in range(23):
        ia,ib=a.candidate_inventory(),b.candidate_inventory()
        assert ia['ledger']==ib['ledger']
        x,y=remove_first_remaining_cell(a),remove_first_remaining_cell(b)
        physical=lambda row:{k:v for k,v in row.items() if k not in ('decision_model_hash','source_state_hash')}
        assert physical(x.info)==physical(y.info)
    assert not a.terminated and not b.terminated
    assert a.case.source_hash==b.case.source_hash and a.decision_model_hash!=b.decision_model_hash
    # Explicit generated chain isolates exact beam horizon semantics cheaply;
    # it is not represented as a native reachability or patient-world test.
    from test_observed_search_opening_lane import Tree
    tree={('move',)*i:{'move':(-.1,((i//7,i%7,1),),False)} for i in range(48)}
    options=dict(max_calls=1000,beam_width=2,seconds=20,transition_mode='lazy_planning',
                 retention_mode='return_plus_opening_depth_volume_v1',retained_prefix_diagnostics=True)
    _,ra=observed_beam_search(Tree(tree,max_steps=24),**options)
    _,rb=observed_beam_search(Tree(tree,max_steps=48),**options)
    assert len(ra['layers'])==24 and len(rb['layers'])==48
    assert ra['layers'][:23]==rb['layers'][:23]
    assert ra['layers'][23]['retained_prefix_diagnostics']['retained_nonterminal_prefixes']==[]
    assert rb['layers'][23]['retained_prefix_diagnostics']['retained_nonterminal_prefixes']
    a=a.fresh()
    path=Path(os.environ.get('H48_BASELINE_NATIVE',str(root/'build/select013-paged-route-h48-v1/before/native_spatial_task.py')))
    if not path.is_file():pytest.skip('Optional exact pre-h48 module unavailable; real first23 parity above completed')
    assert hashlib.sha256(path.read_bytes()).hexdigest()=='f30a536aa2960f854765eca54504dbd3d43a9698ccfd9f9c8b12712d32b5afbf'
    spec=importlib.util.spec_from_file_location('resectionlab._pre_h48_native',path)
    old=importlib.util.module_from_spec(spec);sys.modules[spec.name]=old;spec.loader.exec_module(old)
    oldcase=old.NativeSpatialCase(**{f.name:getattr(case,f.name) for f in fields(case) if f.init})
    previous=old.NativeSpatialTask(oldcase,max_steps=24)
    assert previous.decision_model_hash==a.decision_model_hash
    assert previous.observation().fingerprint==a.observation().fingerprint
    assert previous.candidate_inventory()==a.candidate_inventory()
    previous.step('STOP');a.step('STOP');assert previous.metrics()==a.metrics()
