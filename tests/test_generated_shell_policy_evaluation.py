"""Known-shell development controls, separate from the frozen held-out cohort."""
from pathlib import Path

import numpy as np
import pytest

from resectionlab.core import array_digest
from resectionlab.surgical_cranial_planning_roots import FIELD, validate_registry
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_sequential_benchmark import FEATURE_NAMES


def original_cube():
    mask = np.zeros((64, 64, 96), bool)
    mask[31:34, 31:34, 67:70] = True
    return dict(routeOrder=['central', 'lateral', 'STOP'], neuralLimitMm3=1.,
                targetMaskHash=array_digest(mask), targetSourceCells=np.flatnonzero(mask).tolist())


def test_closed_shell_observation_preserves_true_target_and_source_route_order(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'scripts'))
    from generated_cranial_demo_case import build_generated_cranial_case
    from generated_shell_policy_evaluation import shell_inputs

    case = original_cube()
    _, base, saved = build_generated_cranial_case(1., target_source_cells=case['targetSourceCells'])
    root, = saved[FIELD]['roots']
    scene = validate_registry(base, saved[FIELD])[root['id']]
    engine = SurgicalEngine.restore(scene, root['sourceSnapshot'], use_cache=False)
    features, legal, routes = shell_inputs(engine, root, case)
    assert set(routes) == {'central', 'lateral'} and legal.all()
    assert np.all(features[:, FEATURE_NAMES.index('global_target_remaining')] == 27 / 16)
    assert np.all(features[:, FEATURE_NAMES.index('remaining_skills')] == 1 / 8)
    swapped, _, _ = shell_inputs(engine, root, {**case, 'routeOrder': ['STOP', 'lateral', 'central']})
    assert np.array_equal(swapped, features[[2, 1, 0]])
    with pytest.raises(ValueError, match='original target differs'):
        shell_inputs(engine, root, {**case, 'targetMaskHash': 'unbound'})
    # Native label 1 is bone, whereas the actor's feature view uses 1 for target.
    assert np.count_nonzero(scene.labels == 1) > 27 and not engine.actions


def test_closed_shell_immediate_stop_preserves_intact_bone_and_membrane(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'scripts'))
    from generated_shell_policy_evaluation import closed_shell

    class Stop:
        def select(self, features, mask, stochastic):
            index, = np.flatnonzero(features[:, FEATURE_NAMES.index('kind_stop')])
            return int(index), {}

    profile = dict(maxPublicCalls=600, maxAcceptedActions=300, maxOnlineSeconds=180., maxReplaySeconds=180.)
    row = closed_shell(original_cube(), 'learned', Stop(), profile)
    assert row['initialStateId'] == row['finalStateId']
    assert row['trainingAdmissionRefusal'] == 'GENERATED_FINITE_SKILL_TRAINING_ADMISSION_REFUSED'
    assert row['computation']['physicalCalls'] == row['acceptedActions'] == 0
    assert row['ledger']['boneRemovedMm3'] == row['ledger']['preparationNonboneRemovedMm3'] == 0
    assert row['endpoint']['classification'] == 'valid-stop-with-residual'
    assert row['replay']['verified'] and row['replay']['checkedStop']['status'] == 'accepted'
    assert row['endpoint']['progress']['residualOriginalTargetCells'] == 27
