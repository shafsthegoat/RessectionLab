"""Source-bound interactive geometry and exact public materialization contracts."""
import copy
from pathlib import Path

import numpy as np
import pytest

from resectionlab.core import semantic_digest
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_scene import SurgicalScene
from resectionlab.surgical_interactive_planning import (
    generate_interactive_approaches, interactive_planning_availability,
    materialize_interactive_approach, normalize_constraints,
    validate_interactive_report, validate_saved_interactive_report,
)
from resectionlab.surgical_joint_geometry import (
    _axial_cover, _screen_cover, _post_exposure_rejection, post_exposure_candidate_pairs,
)


@pytest.fixture(scope='module')
def acquired():
    from resectionlab.imaging import read_case_artifacts
    from resectionlab.surgical_environment import load_environment
    path = Path(__file__).resolve().parents[1]/'testcases/cases/UCSF-PDGM-0007-operative-demo.ressectionlab'
    if not path.exists():
        pytest.skip('Local ignored acquired UCSF-PDGM-0007 fixture unavailable')
    scene = load_environment(path)
    record = read_case_artifacts(path)['surgical_toolbox']
    engine = SurgicalEngine(scene)
    for action in record['branches']['original-exposure']['actions']:
        assert engine.step(action)['status'] == 'accepted'
    return engine


@pytest.fixture(scope='module')
def generated(acquired):
    return generate_interactive_approaches(acquired, interactive_planning_availability(acquired)['defaults'])


def _reseal(report):
    report.pop('reportSeal', None)
    report['reportSeal'] = semantic_digest(report)
    return report


@pytest.mark.parametrize('source', [{'kind': 'generated control'}, {'kind': 'generated control', 'adapter': None},
    {'kind': 'generated control', 'adapter': 'unsupported'}, {'kind': 'generated control', 'adapter': ['unsupported']}])
def test_source_gate_does_not_promote_generated_anatomy(source):
    labels = np.zeros((12, 12, 12), np.uint16); labels[5:7, 5:7, 5:7] = 1
    engine = SurgicalEngine(SurgicalScene(labels, np.eye(4),
        [{'id': 1, 'name': 'tumor', 'material': 'tumor', 'target': True}],
        source, {}))
    availability = interactive_planning_availability(engine)
    assert not availability['available'] and availability['defaults'] is None
    with pytest.raises(ValueError, match='documented UCSF'):
        generate_interactive_approaches(engine, {})


def test_defaults_bind_actual_exposure_and_retained_target(acquired):
    availability = interactive_planning_availability(acquired)
    assert availability['available']
    assert availability['defaults']['entry'] == {'centerMm': [-84., 118., 42.], 'radiusMm': 7.}
    assert availability['defaults']['target']['centerMm'] == [-84., 118., 66.]
    assert availability['evidence']['historicalExposure'] == {'targetRemovedMm3': 560., 'nonTargetRemovedMm3': 4940.}
    assert set(availability['evidence']['unknownEvidence']) == {'vessels', 'functional', 'shift', 'skull'}


@pytest.mark.parametrize('field,value', [('candidateCount', True), ('candidateCount', 2.5),
    ('maxCandidates', 97), ('maxWallSeconds', float('nan')), ('minEntrySeparationMm', '2')])
def test_constraints_reject_coercion_and_out_of_bounds(acquired, field, value):
    constraints = interactive_planning_availability(acquired)['defaults']; constraints[field] = value
    with pytest.raises(ValueError):
        normalize_constraints(acquired, constraints)


def test_target_and_entry_must_use_actual_source_bounds(acquired):
    constraints = interactive_planning_availability(acquired)['defaults']
    constraints['target']['centerMm'] = [0., 0., 0.]
    with pytest.raises(ValueError, match='retained owned source'):
        normalize_constraints(acquired, constraints)
    constraints = interactive_planning_availability(acquired)['defaults']
    constraints['entry']['radiusMm'] = 8.
    with pytest.raises(ValueError, match='existing authored'):
        normalize_constraints(acquired, constraints)


def test_required_unknown_evidence_returns_sealed_refusal_without_routes(acquired):
    constraints = interactive_planning_availability(acquired)['defaults']
    constraints['requiredEvidence'] = ['vessels', 'shift', 'vessels']
    report = generate_interactive_approaches(acquired, constraints)
    assert report['status'] == 'required_evidence_unavailable'
    assert report['missingRequiredEvidence'] == ['shift', 'vessels'] and report['routes'] == []
    assert validate_interactive_report(acquired, report)


def test_resealed_unknown_refusal_cannot_gain_executable_routes(acquired, generated):
    changed = copy.deepcopy(generated)
    changed['constraints']['requiredEvidence'] = ['vessels']
    changed['status'] = 'required_evidence_unavailable'
    changed['missingRequiredEvidence'] = ['vessels']
    with pytest.raises(ValueError, match='unavailable evidence'):
        materialize_interactive_approach(acquired, _reseal(changed), changed['routes'][0]['id'])


@pytest.mark.parametrize('metric', ['clinicalRisk', 'minimumClearanceMm', 'deformationInjuryMm3'])
def test_resealed_unknown_metric_cannot_be_promoted_on_save(acquired, generated, metric):
    changed = copy.deepcopy(generated); changed['routes'][0]['metrics'][metric] = 0.
    with pytest.raises(ValueError, match='geometry metrics'):
        validate_saved_interactive_report(acquired.scene, acquired.snapshot(), _reseal(changed))


def test_generated_routes_are_separated_read_only_and_source_bound(acquired, generated):
    before = acquired.snapshot()
    assert generated['status'] == 'geometric_candidates' and len(generated['routes']) == 3
    assert generated['screening']['continuumEvaluated'] is False
    assert generated['sourceSceneId'] == acquired.scene.identity
    assert generated['startStateId'] == acquired.state_id
    assert validate_interactive_report(acquired, generated)
    assert validate_saved_interactive_report(acquired.scene, before, generated)
    for row in generated['routes']:
        assert row['metrics']['nonTargetTraversalMm3'] == 0.
        assert row['metrics']['clinicalRisk'] is None
        assert row['metrics']['localManeuverability']['attemptedCount'] == 4
        assert all(a['operation'] in ('exchange', 'place', 'move') for a in row['actionProgram'])
        assert all(x['entrySeparationMm'] >= 2. for x in row['separationComparisons'])
    assert acquired.snapshot() == before


def test_exact_public_approach_and_withdrawal_preserve_source_tissue(acquired, generated):
    before = acquired.snapshot()
    for row in generated['routes']:
        approach, receipt = materialize_interactive_approach(acquired, generated, row['id'])
        assert approach.state['slots']['left']['inserted'] and not approach.state['slots']['left']['active']
        assert approach.state['slots']['left']['position'] == row['targetPose']['position']
        assert approach.state['removed'] == acquired.state['removed']
        assert receipt['publicAccessAccepted'] and receipt['publicWithdrawalAccepted']
        assert receipt['exactPhysicsPreserved'] and approach.scene is acquired.scene
        assert SurgicalEngine.restore(acquired.scene, approach.snapshot()).snapshot() == approach.snapshot()
    assert acquired.snapshot() == before


def test_edited_tool_target_and_entry_change_source_proposals(acquired, generated):
    defaults = interactive_planning_availability(acquired)['defaults']
    small = copy.deepcopy(defaults); small['tool']['radiusMm'] = .5
    smaller = generate_interactive_approaches(acquired, small)
    assert smaller['routes'][0]['targetPose'] != generated['routes'][0]['targetPose']
    changed = copy.deepcopy(defaults); changed['entry']['centerMm'][0] += 2.
    assert generate_interactive_approaches(acquired, changed)['routes'][0]['entryPose'] != generated['routes'][0]['entryPose']
    narrow = copy.deepcopy(defaults); narrow['target']['radiusMm'] = 1.
    report = generate_interactive_approaches(acquired, narrow)
    assert report['targetSelection']['retainedSourceCells'] < generated['targetSelection']['retainedSourceCells']


def test_known_source_non_target_margin_refuses_and_never_relaxes_geometry(acquired, generated):
    defaults = interactive_planning_availability(acquired)['defaults']
    defaults['nonTargetClearanceMm'] = 2.
    report = generate_interactive_approaches(acquired, defaults)
    assert report['status'] == 'no_geometric_route' and not report['routes']
    assert any(row['reason'] == 'NON_TARGET_CLEARANCE' for row in report['rejections'])
    limit = report['limitingStructures'][0]
    assert limit['name'] == limit['structureName'] == 'brain'
    assert limit['limitingSourceCell'] not in acquired.state['removed']
    assert limit['limitingSourceCell'] in acquired.occupancy().material_ids([limit['occupancyCell']])
    assert limit['positionMm'] == acquired.scene.points([limit['occupancyCell']])[0].tolist()
    assert limit['positionMm'] != defaults['target']['centerMm']
    assert limit['part'] in ('shaft', 'working')
    forged = copy.deepcopy(report)
    forged['limitingStructures'][0]['positionMm'] = defaults['target']['centerMm']
    with pytest.raises(ValueError, match='limiting structure projection'):
        validate_saved_interactive_report(acquired.scene, acquired.snapshot(), _reseal(forged))
    assert validate_saved_interactive_report(acquired.scene, acquired.snapshot(), report)
    row = generated['routes'][0]
    params = acquired.parameters(acquired.state['slots']['left']); params['radiusMm'] = 1.5
    blocked = acquired.clone(); blocked.state['slots']['right'].update(row['targetPose'])
    cover = _axial_cover(blocked, row['outsidePose'], row['targetPose'], params)
    assert _screen_cover(blocked, cover, 0.)[0] == _screen_cover(blocked, cover, 2.)[0] == 'TOOL_COLLISION:right'


def test_seal_baseline_snapshot_and_resealed_route_tampering_refuse(acquired, generated):
    changed = copy.deepcopy(generated); changed['routes'][0]['targetPose']['position'][2] += 1.
    with pytest.raises(ValueError, match='seal'):
        validate_interactive_report(acquired, changed)
    _reseal(changed)
    with pytest.raises(ValueError, match='program or geometry'):
        materialize_interactive_approach(acquired, changed, changed['routes'][0]['id'])
    changed = copy.deepcopy(generated); changed['routes'][0]['entryPose']['position'][0] += 100.
    with pytest.raises(ValueError, match='escapes requested'):
        materialize_interactive_approach(acquired, _reseal(changed), changed['routes'][0]['id'])
    snapshot = acquired.snapshot(); snapshot['checkpoint']['state']['elapsedSeconds'] += .1
    with pytest.raises(ValueError, match='native baseline'):
        validate_saved_interactive_report(acquired.scene, snapshot, generated)
    future = acquired.clone(); future.step({'slot': 'left', 'operation': 'inspect', 'duration': .1})
    assert not validate_interactive_report(future, generated, require_current=False)
    with pytest.raises(ValueError, match='baseline differs'):
        validate_interactive_report(future, generated)


def test_full_shaft_and_second_tool_are_limiting_controls(acquired, generated):
    row = generated['routes'][0]
    params = acquired.parameters(acquired.state['slots']['left']); params['radiusMm'] = 1.5
    cover = _axial_cover(acquired, row['outsidePose'], row['targetPose'], params)
    assert _screen_cover(acquired, cover)[0] is None
    blocked = acquired.clone()
    blocked.state['slots']['right'].update(row['targetPose'])
    assert _screen_cover(blocked, cover)[0] == 'TOOL_COLLISION:right'
    # A long shaft crosses retained brain even though its distal tip is free.
    pose = {'position': [-100., 118., 20.], 'quaternion': [1., 0., 0., 0.]}
    assert _screen_cover(acquired, _axial_cover(acquired, pose, pose, params))[0] == 'SHAFT_COLLISION'
    located = _post_exposure_rejection(acquired, _axial_cover(acquired, pose, pose, params), 'SHAFT_COLLISION')
    assert located['part'] == 'shaft' and located['positionMm'][2] > pose['position'][2]
    assert located['structureName'] == 'brain'
    assert 'positionMm' not in _post_exposure_rejection(blocked, cover, 'TOOL_COLLISION:right')


def test_existing_joint_sampler_keeps_all_entries_in_requested_disk():
    pairs = post_exposure_candidate_pairs([1., 2., 3.], [1., 2., 20.], [0., 0., 1.], 7.)
    assert len(pairs) == 25
    assert all(np.linalg.norm(entry-[1., 2., 3.]) <= 7.+1e-9 and entry[2] == 3. for entry, _ in pairs)
