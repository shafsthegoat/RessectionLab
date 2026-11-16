"""Patient planning refuses; historical geometry parsing/inspection stays available."""
import copy
from pathlib import Path

import numpy as np
import pytest

from resectionlab.core import semantic_digest
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_scene import SurgicalScene
from resectionlab.surgical_interactive_planning import (
    generate_interactive_approaches, interactive_planning_availability,
    normalize_constraints,
    validate_saved_interactive_report,
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


@pytest.mark.parametrize('source', [{'kind': 'generated control'}, {'kind': 'generated control', 'adapter': None},
    {'kind': 'generated control', 'adapter': 'unsupported'}, {'kind': 'generated control', 'adapter': ['unsupported']}])
def test_source_gate_does_not_promote_generated_anatomy(source):
    labels = np.zeros((12, 12, 12), np.uint16); labels[5:7, 5:7, 5:7] = 1
    engine = SurgicalEngine(SurgicalScene(labels, np.eye(4),
        [{'id': 1, 'name': 'tumor', 'material': 'tumor', 'target': True}],
        source, {}))
    availability = interactive_planning_availability(engine)
    assert not availability['available'] and availability['defaults'] is None
    with pytest.raises(ValueError, match='PROCEDURE_POLICY_REQUIRED_FOR_PATIENT_PLANNING'):
        generate_interactive_approaches(engine, {})


def _historical_constraints():
    # Archived source coordinates are parser test inputs, never new task admission.
    return {'entry': {'centerMm': [-84., 118., 42.], 'radiusMm': 7.},
        'target': {'centerMm': [-84., 118., 66.], 'radiusMm': 12.},
        'tool': {'instrument': 'cusa', 'radiusMm': 1.5}, 'requiredEvidence': [],
        'candidateCount': 3, 'maxCandidates': 32, 'minEntrySeparationMm': 2.,
        'minAngleDeg': 0., 'maxWallSeconds': 15., 'nonTargetClearanceMm': 0.}


def test_explicit_engineering_source_still_requires_supported_interactive_source():
    labels = np.zeros((12, 12, 12), np.uint16); labels[5:7, 5:7, 5:7] = 1
    engine = SurgicalEngine(SurgicalScene(labels, np.eye(4),
        [{'id': 1, 'name': 'generated tumor', 'material': 'tumor', 'target': True}],
        {'kind': 'generated-control'}, {}))
    availability = interactive_planning_availability(engine)
    assert not availability['available'] and availability['defaults'] is None
    assert 'documented UCSF' in availability['reason']
    with pytest.raises(ValueError, match='documented UCSF'):
        generate_interactive_approaches(engine, {})


def test_historical_patient_exposure_cannot_authorize_new_planning(acquired):
    before = acquired.snapshot()
    availability = interactive_planning_availability(acquired)
    assert not availability['available']
    assert availability['defaults'] is availability['evidence'] is None
    assert 'PROCEDURE_POLICY_REQUIRED_FOR_PATIENT_PLANNING' in availability['reason']
    with pytest.raises(ValueError, match='PROCEDURE_POLICY_REQUIRED_FOR_PATIENT_PLANNING'):
        generate_interactive_approaches(acquired, _historical_constraints())
    assert acquired.snapshot() == before


@pytest.mark.parametrize('field,value', [('candidateCount', True), ('candidateCount', 2.5),
    ('maxCandidates', 97), ('maxWallSeconds', float('nan')), ('minEntrySeparationMm', '2')])
def test_constraints_reject_coercion_and_out_of_bounds(acquired, field, value):
    constraints = _historical_constraints(); constraints[field] = value
    with pytest.raises(ValueError):
        normalize_constraints(acquired, constraints)


def test_target_and_entry_must_use_actual_source_bounds(acquired):
    constraints = _historical_constraints()
    constraints['target']['centerMm'] = [0., 0., 0.]
    with pytest.raises(ValueError, match='retained owned source'):
        normalize_constraints(acquired, constraints)
    constraints = _historical_constraints()
    constraints['entry']['radiusMm'] = 8.
    with pytest.raises(ValueError, match='existing authored'):
        normalize_constraints(acquired, constraints)


@pytest.mark.parametrize('required', [[], ['vessels'], ['functional', 'shift']])
def test_required_evidence_requests_cannot_bypass_patient_admission(acquired, required):
    constraints = _historical_constraints()
    constraints['requiredEvidence'] = required
    before = acquired.snapshot()
    with pytest.raises(ValueError, match='PROCEDURE_POLICY_REQUIRED_FOR_PATIENT_PLANNING'):
        generate_interactive_approaches(acquired, constraints)
    assert acquired.snapshot() == before


def test_historical_source_shaft_geometry_remains_an_inspection_control(acquired):
    before = acquired.snapshot()
    params = acquired.parameters(acquired.state['slots']['left']); params['radiusMm'] = 1.5
    pose = {'position': [-100., 118., 20.], 'quaternion': [1., 0., 0., 0.]}
    cover = _axial_cover(acquired, pose, pose, params)
    assert _screen_cover(acquired, cover)[0] == 'SHAFT_COLLISION'
    located = _post_exposure_rejection(acquired, cover, 'SHAFT_COLLISION')
    assert located['part'] == 'shaft' and located['positionMm'][2] > pose['position'][2]
    assert located['structureName'] == 'brain'
    assert acquired.snapshot() == before


@pytest.fixture(scope='module')
def archived_interactive():
    """Read an existing sealed report; never generate a replacement patient plan."""
    from resectionlab.imaging import read_case_artifacts
    from resectionlab.surgical_environment import load_environment
    path = Path(__file__).resolve().parents[1]/(
        'testcases/outputs/interactive-planning-20261031/bridge-v4-head-physics/'
        'interactive-planning.ressectionlab')
    if not path.is_file():
        pytest.skip('Local ignored historical interactive report unavailable')
    record = read_case_artifacts(path)['surgical_toolbox']
    scene = load_environment(path)
    report = copy.deepcopy(record['interactiveApproaches'][-1]['report'])
    source = next(snapshot for snapshot in record['branches'].values()
        if snapshot['stateId'] == report['startStateId'])
    return scene, source, report


def _reseal(report):
    report.pop('reportSeal', None)
    report['reportSeal'] = semantic_digest(report)
    return report


def test_archived_patient_report_geometry_and_literal_baseline_remain_readable(archived_interactive):
    scene, source, report = archived_interactive
    before = copy.deepcopy(source)
    assert validate_saved_interactive_report(scene, source, report)
    baseline = SurgicalEngine.restore(scene, source)
    assert baseline.snapshot() == source
    assert len(report['routes']) == 3
    for row in report['routes']:
        assert row['metrics']['clinicalRisk'] is None
        assert row['metrics']['nonTargetTraversalMm3'] == 0.
        assert all(x['entrySeparationMm'] >= 2. for x in row['separationComparisons'])
    assert source == before
    assert not interactive_planning_availability(baseline)['available']


@pytest.mark.parametrize('metric', ['clinicalRisk', 'minimumClearanceMm', 'deformationInjuryMm3'])
def test_archived_unknown_metric_cannot_be_promoted_on_save(archived_interactive, metric):
    scene, source, report = archived_interactive
    changed = copy.deepcopy(report); changed['routes'][0]['metrics'][metric] = 0.
    with pytest.raises(ValueError, match='geometry metrics'):
        validate_saved_interactive_report(scene, source, _reseal(changed))


def test_archived_refusal_and_geometry_tampering_remain_rejected(archived_interactive):
    scene, source, report = archived_interactive
    changed = copy.deepcopy(report)
    changed['constraints']['requiredEvidence'] = ['vessels']
    changed['status'] = 'required_evidence_unavailable'
    changed['missingRequiredEvidence'] = ['vessels']
    with pytest.raises(ValueError, match='unavailable evidence'):
        validate_saved_interactive_report(scene, source, _reseal(changed))
    changed = copy.deepcopy(report); changed['routes'][0]['targetPose']['position'][2] += 1.
    with pytest.raises(ValueError, match='seal'):
        validate_saved_interactive_report(scene, source, changed)
    with pytest.raises(ValueError, match='program or geometry'):
        validate_saved_interactive_report(scene, source, _reseal(changed))
    changed = copy.deepcopy(report); changed['routes'][0]['entryPose']['position'][0] += 100.
    with pytest.raises(ValueError, match='escapes requested'):
        validate_saved_interactive_report(scene, source, _reseal(changed))
    snapshot = copy.deepcopy(source); snapshot['checkpoint']['state']['elapsedSeconds'] += .1
    with pytest.raises(ValueError, match='native baseline'):
        validate_saved_interactive_report(scene, snapshot, report)


def test_archived_complete_capsules_still_detect_second_tool_collision(archived_interactive):
    scene, source, report = archived_interactive
    baseline = SurgicalEngine.restore(scene, source)
    before = baseline.snapshot()
    row = report['routes'][0]
    params = baseline.parameters(baseline.state['slots']['left']); params['radiusMm'] = 1.5
    cover = _axial_cover(baseline, row['outsidePose'], row['targetPose'], params)
    assert _screen_cover(baseline, cover)[0] is None
    blocked = baseline.clone(); blocked.state['slots']['right'].update(row['targetPose'])
    assert _screen_cover(blocked, cover)[0] == 'TOOL_COLLISION:right'
    assert baseline.snapshot() == before


def test_existing_joint_sampler_keeps_all_entries_in_requested_disk():
    pairs = post_exposure_candidate_pairs([1., 2., 3.], [1., 2., 20.], [0., 0., 1.], 7.)
    assert len(pairs) == 25
    assert all(np.linalg.norm(entry-[1., 2., 3.]) <= 7.+1e-9 and entry[2] == 3. for entry, _ in pairs)
