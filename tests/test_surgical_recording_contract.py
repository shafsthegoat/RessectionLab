"""Native recording contracts verify physical state independently of producer labels and report filenames."""
import copy
import shutil

import pytest

from resectionlab.anatomy_approaches import dispatch_plan
from resectionlab.core import semantic_digest
from resectionlab.patient_anatomy import PatientImagingCollection
from resectionlab.surgical_recording_contract import bind_recording, _treatment_indices, _generator
from resectionlab.surgical_study_replay import verify_recording, validate_saved_binding
from test_surgical_task_admission import prepared


def test_new_native_derivative_contract_replays_and_refuses_resealed_false_observations(prepared, tmp_path):
    case, endpoint, original, native = prepared[0]
    study = PatientImagingCollection(shutil.copytree(native.root, tmp_path/'native'))
    dispatch_plan(study, {'action': 'upsert', 'approach': {'id': 'example-route', 'entry': [18., 18., 0.],
        'target': [18., 18., 20.], 'radiusMm': 5., 'targetLayerId': 'tumor'}})
    record = copy.deepcopy(original)
    producer = {'kind': 'scripted', 'name': 'Source-bound fixture commands', 'sourceSha256': 'a'*64}
    exact = bind_recording(study, case, record, approach_id='example-route', target_id='tumor', generator=producer)
    record['exactGridReplay'] = exact
    assert len(exact['boundaries']) == len(endpoint.actions)
    assert exact['observations'][-1]['removedHash'] == semantic_digest(sorted(endpoint.state['removed']))
    cold = []
    receipt, _ = verify_recording(study, 'tumor', case, record, approach_id='example-route', on_verified_engine=cold.append)
    assert cold[0].state == endpoint.state
    assert receipt['generatorProvenance'] == 'producer-declared' and receipt['actionGenerator'] == producer
    validate_saved_binding(case, {'surgical_toolbox': record, 'nativeStudyReplay': receipt})
    with pytest.raises(ValueError, match='Select the recorded approach'):
        verify_recording(study, 'tumor', case, record, approach_id='different-route')
    # Even a self-consistent altered digest cannot substitute a numerical observation.
    exact['observations'][2]['metricsHash'] = 'sha256:'+'0'*64
    exact['contractHash'] = semantic_digest({k: v for k, v in exact.items() if k != 'contractHash'})
    with pytest.raises(ValueError, match='Cold engine pose, metrics, time or removal'):
        verify_recording(study, 'tumor', case, record, approach_id='example-route', on_verified_engine=cold.append)
    assert len(cold) == 1


def test_treatment_motion_checks_effective_radius_and_origin_from_each_observed_slot():
    route = {'entry': [0., 0., 0.], 'target': [0., 0., 5.], 'radiusMm': 1.5}
    def pose(position):
        return {'position': position, 'quaternion': [0., 0., 0., 1.]}
    def observed(position, radius=1.5, inserted=True):
        return {'tools': {'right': {**pose(position), 'instrument': 'cusa', 'parameters': {'radiusMm': radius},
                                     'inserted': inserted, 'active': False}}}
    place = {'slot': 'right', 'operation': 'place', 'path': [pose(route['entry'])]}
    apply = {'slot': 'right', 'operation': 'apply', 'path': [pose(route['target'])]}
    start, entry, goal = observed([0., 0., -10.], inserted=False), observed(route['entry']), observed(route['target'])
    assert _treatment_indices([place, apply], route, [start, entry, goal]) == [1]
    with pytest.raises(ValueError, match='exceeds'):
        _treatment_indices([place, {**apply, 'parameters': {'radiusMm': 4.}}], route,
                           [start, entry, observed(route['target'], radius=4.)])
    off = [1., 0., 2.]
    with pytest.raises(ValueError, match='leaves'):
        _treatment_indices([place, {'slot': 'right', 'operation': 'move', 'pose': pose(off)}, apply],
                           route, [start, entry, observed(off), goal])
    with pytest.raises(ValueError, match='leaves'):
        _treatment_indices([place, {**apply, 'path': [pose(off), pose(route['target'])]}], route, [start, entry, goal])


def test_rl_producer_requires_checkpoint_provenance():
    missing = {'kind': 'rl', 'name': 'Policy', 'sourceSha256': 'a'*64}
    with pytest.raises(ValueError, match='trained checkpoint'):
        _generator(missing)
    assert _generator({**missing, 'checkpointSha256': 'b'*64})['kind'] == 'rl'
