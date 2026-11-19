"""Verify a recorded treatment against the native plan, source arrays and causal engine history."""
import copy
import re

import nibabel as nib
import numpy as np

from .anatomy_approaches import read_approaches
from .anatomy_planning import manifest_target_binding
from .core import semantic_digest
from .patient_anatomy import digest, file_hash
from .surgical_engine import SurgicalEngine
from .surgical_scene import original_target_memberships
from .surgical_tools import TOOLS


SCHEMA = 'resectionlab.native-route-engine-recording.v2'
ROUTE_FIELDS = ('id', 'name', 'entry', 'target', 'radiusMm', 'measurementHash', 'parameterHash')


def _generator(value):
    if (not isinstance(value, dict) or set(value) - {'kind', 'name', 'sourceSha256', 'checkpointSha256', 'description'}
            or value.get('kind') not in ('scripted', 'search', 'rl', 'manual')
            or not isinstance(value.get('name'), str) or not 0 < len(value['name'].strip()) <= 160
            or not re.fullmatch(r'[0-9a-f]{64}', str(value.get('sourceSha256', '')))
            or ('description' in value and (not isinstance(value['description'], str) or len(value['description']) > 1000))
            or (value.get('kind') == 'rl') != ('checkpointSha256' in value)
            or ('checkpointSha256' in value and not re.fullmatch(r'[0-9a-f]{64}', str(value['checkpointSha256'])))):
        raise ValueError('Declare the recorded generator, source digest and trained checkpoint for RL')
    return value


def observed_boundary(engine):
    """Capture physical outputs from the independently re-executed engine, including partial actions."""
    from .surgical_sequence_outcomes import original_target_residual_mm3
    slots = engine.state['slots']
    return {'stateId': engine.state_id, 'slotsHash': semantic_digest(slots),
            'tools': {hand: {k: copy.deepcopy(slot[k]) for k in
                             ('instrument', 'position', 'quaternion', 'active', 'inserted', 'parameters')}
                      for hand, slot in slots.items()},
            'removedHash': semantic_digest(sorted(engine.state['removed'])), 'metricsHash': semantic_digest(engine.metrics()),
            'residualTargetMm3': original_target_residual_mm3(engine), 'elapsedSeconds': engine.state['elapsedSeconds']}


def _treatment_indices(actions, route, observations):
    """Bind active removal to the selected straight approach, including partial motion and effective radius."""
    from scipy.spatial.transform import Rotation
    entry, target = np.asarray(route['entry']), np.asarray(route['target'])
    axis = target - entry
    length = float(np.linalg.norm(axis))
    if not np.isfinite(length) or length <= 0 or len(observations) != len(actions) + 1:
        raise ValueError('Invalid recorded approach geometry or observation count')
    unit = axis / length

    def on_line(pose):
        offset = np.asarray(pose['position']) - entry
        depth = float(offset @ unit)
        attitude = Rotation.from_quat(pose['quaternion']).apply([0., 0., 1.])
        return (-1.e-5 <= depth <= length + 1.e-5 and np.linalg.norm(offset - depth * unit) <= 1.e-5
                and np.linalg.norm(attitude - unit) <= 1.e-5)

    inserted_on_route = {hand: tool['inserted'] and on_line(tool) and
                         np.allclose(tool['position'], entry, atol=1.e-5, rtol=0)
                         for hand, tool in observations[0]['tools'].items()}
    indices, goal_requested = [], False
    for index, action in enumerate(actions):
        hand, op = action['slot'], action['operation']
        before, after = (observations[i]['tools'][hand] for i in (index, index + 1))
        poses = action.get('path', [action['pose']] if 'pose' in action else [])
        tool = TOOLS[after['instrument']]
        if op in ('exchange', 'withdraw'):
            inserted_on_route[hand] = False
            continue
        if tool['effect'] != 'remove' or 'tumor' not in tool['materials']:
            continue
        if op == 'place':
            inserted_on_route[hand] = (len(poses) == 1 and np.allclose(after['position'], entry, atol=1.e-5, rtol=0)
                                       and on_line(after) and on_line(poses[0]))
            continue
        if op == 'activate' or before['active'] or after['active']:
            raise ValueError('Persistent treatment activation requires a separate swept-route contract')
        if op in ('hold', 'grasp', 'extract'):
            raise ValueError('Unsupported treatment operation in straight-approach recording')
        if op not in ('apply', 'move'):
            continue
        inserted_on_route[hand] = inserted_on_route[hand] and all(on_line(p) for p in [before, *poses, after])
        if op == 'apply':
            if not inserted_on_route[hand]:
                raise ValueError('Treatment origin, motion or orientation leaves the selected approach line')
            if after['parameters'].get('radiusMm', tool['radiusMm']) > route['radiusMm'] + 1.e-5:
                raise ValueError('The treatment tool exceeds the selected approach radius')
            indices.append(index)
            goal_requested |= any(np.allclose(p['position'], target, atol=1.e-5, rtol=0) for p in poses)
    if not indices or not goal_requested:
        raise ValueError('The recorded treatment sequence did not request the selected approach target')
    return indices


def _transition_proof(scene, snapshot, timeline, initial_state_id, initial_removed):
    if not isinstance(timeline, list) or len(timeline) != len(snapshot['actions']):
        raise ValueError('Recording requires its entire accepted engine history')
    memberships = original_target_memberships(scene)
    if len(memberships) != 1:
        raise ValueError('Exactly one original target is required')
    target = set(map(int, next(iter(memberships.values()))))
    removed = set(initial_removed)
    previous = initial_state_id
    chain = []
    for index, (action, outcome) in enumerate(zip(snapshot['actions'], timeline)):
        if (outcome['action'] != action or outcome['stateBefore'] != previous
                or outcome['status'] not in ('accepted', 'partial')):
            raise ValueError('Recorded actions and state transitions are discontinuous')
        delta = outcome['deltas']['removedCells']
        if removed.intersection(delta) or len(set(delta)) != len(delta):
            raise ValueError('Engine history removes an original cell more than once')
        removed.update(delta)
        previous = outcome['stateAfter']
        chain.append({'position': index + 1, 'stateId': previous, 'actionHash': semantic_digest(action),
                      'outcomeHash': semantic_digest(outcome), 'removedHash': semantic_digest(sorted(removed)),
                      'residualTargetCells': len(target - removed)})
    if set(snapshot['checkpoint']['state']['removed']) != removed or previous != snapshot['stateId']:
        raise ValueError('Recorded tissue or final state differs from the accepted engine history')
    return chain


def identity(case, scene, snapshot, record, exact):
    manifest = exact['sourceManifest']
    target_id = exact['targetLayerId']
    row = next(row for row in manifest['layers'] if row['id'] == target_id)
    adapter = scene.source.get('adapter') or {}
    derivative = adapter.get('type') == 'native-patient-research-derivative'
    target_name, = [r['name'] for r in scene.regions if r['target']]
    if derivative and adapter['sourceManifest'] != manifest:
        raise ValueError('Native derivative source manifest differs from the recorded patient study')
    expected_target = adapter['sourceLayers'][target_name]['layerId'] if derivative else target_name
    if (exact['schema'] != SCHEMA or expected_target != target_id
            or exact['sourceMode'] != ('research-derivative' if derivative else 'exact-grid')
            or manifest['patient_id'] != (adapter['sourcePatient'] if derivative else case.metadata.get('patient_id'))
            or manifest['snapshot_id'] != digest({k: v for k, v in manifest.items() if k != 'snapshot_id'})
            or exact['caseHash'] != case.semantic_hash or exact['sceneId'] != scene.identity
            or exact['branchId'] != record['branchId'] or exact['endpointStateId'] != snapshot['stateId']):
        raise ValueError('Recorded source, scene, target or branch identity differs')
    chain = _transition_proof(scene, snapshot, record['timeline'][record['branchId']], exact['initialStateId'], exact['initialRemoved'])
    if chain != exact['boundaries'] or semantic_digest({k: v for k, v in exact.items() if k != 'contractHash'}) != exact['contractHash']:
        raise ValueError('Recording contract or per-action state/removal receipt differs')
    _generator(exact['actionGenerator'])
    observations = exact['observations']
    if (not observations or observations[0]['stateId'] != exact['initialStateId']
            or len(observations) != len(chain) + 1
            or any(o['stateId'] != b['stateId'] or o['removedHash'] != b['removedHash']
                   or not np.isclose(o['residualTargetMm3'], b['residualTargetCells'] * scene.voxel_mm3, rtol=0, atol=1.e-7)
                   for o, b in zip(observations[1:], chain))):
        raise ValueError('Recorded physical observations differ from the transition chain')
    if _treatment_indices(snapshot['actions'], exact['recordedApproach'], observations) != exact['treatmentActionIndices']:
        raise ValueError('Recorded treatment actions differ from the selected route')
    return dict(patientId=manifest['patient_id'], sourceSnapshotId=manifest['snapshot_id'],
                targetBinding=manifest_target_binding(manifest, row), recordedApproach=exact['recordedApproach'],
                actionGenerator=exact['actionGenerator'], generatorProvenance='producer-declared',
                contractHash=exact['contractHash'], initialStateId=exact['initialStateId'], transitionHash=semantic_digest(chain))


def verify_source(collection, case, scene, exact, approach_id, *, check):
    from .surgical_exact_grid_replay import verify_selected_route
    if collection.document != exact['sourceManifest']:
        raise ValueError('Recorded patient or native study revision differs from the open study')
    verify_selected_route(collection, exact, approach_id)
    if exact['sourceMode'] == 'research-derivative':
        from .surgical_study_replay import _verify_resampled_source
        _verify_resampled_source(collection, case, scene, check)
        return
    reference = collection.series(collection.document['reference_series'])
    structural = next((r for r in case.source_refs if r.source_id == 'structural' and r.provenance == 'observed'), None)
    if structural is None or reference['original']['sha256'] != structural.sha256 or not np.array_equal(reference['affine'], case.affine):
        raise ValueError('Exact grid requires the original acquired reference image and native affine')
    reg = case.metadata.get('workspace_registration', {})
    for source in collection.document['series']:
        check()
        sid = source['id']
        sha = source['original']['sha256']
        path = collection.path(source['original']['path'])
        matrix = collection.registration_matrix(sid, require_geometry=True)
        if sid == reference['id']:
            expected = np.eye(4)
        elif sha == reg.get('original_moving_image_sha256'):
            expected = reg['original_moving_to_reference_ras_mm']
        elif sha in reg.get('source_sha256_by_modality', {}).values() and sha != structural.sha256:
            expected = np.eye(4)
        else:
            raise ValueError('Additional scan lacks a recorded source and registration binding')
        if sid != reference['id']:
            transform = collection.document['transforms'][sid]
            quality = transform.get('quality', {})
            evidence = reg.get('evidence_sha256')
            decision = reg.get('review_decision')
            person = reg.get('source_person_id')
            if (transform['status'] != 'unreviewed' or not isinstance(evidence, str)
                    or re.fullmatch('[a-f0-9]{64}', evidence) is None or not isinstance(decision, str)
                    or not decision.strip() or quality.get('reviewEvidenceSha256') != evidence
                    or quality.get('reviewDecision') != decision
                    or structural.sha256 not in reg.get('source_sha256_by_modality', {}).values()
                    or not isinstance(person, str) or not person
                    or person not in (case.metadata.get('patient_id'), case.metadata.get('patient_group'))
                    or reg.get('patient_id') != case.case_id):
                raise ValueError('Registered display source exceeds its recorded overlay scope')
        scan = collection.array(source['data'])
        if (file_hash(path) != sha or not np.array_equal(expected, matrix)
                or not np.array_equal(scan, nib.load(path).get_fdata(dtype=np.float32))
                or sid == reference['id'] and not np.array_equal(scan, case.mri)):
            raise ValueError('Acquired source pixels or registered physical geometry differ')
    if {region['name'] for region in scene.regions} != set(case.compartments):
        raise ValueError('Exact-grid recording omits or substitutes a modeled source compartment')
    if scene.shape != case.mri.shape or not np.array_equal(scene.affine, case.affine):
        raise ValueError('Recorded scene differs from the measured native lattice')
    for region in scene.regions:
        check()
        name = region['name']
        row = collection.layer(name)
        collection.require_native_geometry(row)
        affine = collection.registration_matrix(row['source_series'], require_geometry=True) @ row['affine']
        if not np.array_equal(affine, case.affine) or not np.array_equal(collection.mask(row), case.compartments[name]):
            raise ValueError('Recorded anatomical layer geometry or membership differs: ' + name)
        source_layer = case.metadata.get('anatomy_layers', {}).get(name)
        if not source_layer or not source_layer.get('kind') or not source_layer.get('provenance'):
            raise ValueError('The recording requires named source layer kind and provenance for ' + name)
        origin, source_kind = source_layer['provenance'], source_layer['kind']
        if row['kind'] != source_kind and not (origin == 'simulated' and source_kind == 'tumor' and row['kind'] == 'synthetic-target'):
            raise ValueError('Native layer kind differs from the recorded anatomical compartment')
        if (origin == 'simulated') != (row['provenance'].get('kind') == 'simulated'):
            raise ValueError('Synthetic layer provenance differs from the recorded layer')
        if row['source_series'] != reference['id'] and row['provenance'].get('evidence_use') != 'context_only':
            raise ValueError('Registered annotation lacks its overlay-only evidence scope')


def verify_history(engine, record, exact):
    initial = engine.fresh()
    if (engine.history != record['timeline'][record['branchId']] or initial.state_id != exact['initialStateId']
            or initial.state['slots'] != exact['initialSlots'] or initial.state['removed'] != exact['initialRemoved']):
        raise ValueError('Independent engine replay differs from the recorded initial state or transitions')


def bind_recording(collection, case, record, *, approach_id, target_id, generator, check=lambda: None):
    """Create a read-only binding after reconstructing the source and replaying every accepted action."""
    from .surgical_study_replay import _recorded_branch, study_target
    study_target(collection, target_id)
    route = next((r for r in read_approaches(collection) if r['id'] == approach_id), None)
    if route is None:
        raise ValueError('Select an assessed approach before binding a recording')
    scene, snapshot, _ = _recorded_branch(case, record, exact_target=target_id)
    exact = {'schema': SCHEMA, 'sourceManifest': copy.deepcopy(collection.document), 'targetLayerId': target_id,
             'recordedApproach': {k: route[k] for k in ROUTE_FIELDS}, 'caseHash': case.semantic_hash,
             'sceneId': scene.identity, 'endpointStateId': snapshot['stateId'], 'branchId': record['branchId'],
             'sourceMode': 'research-derivative' if (scene.source.get('adapter') or {}).get('type') == 'native-patient-research-derivative' else 'exact-grid',
             'actionGenerator': copy.deepcopy(_generator(generator))}
    verify_source(collection, case, scene, exact, approach_id, check=check)
    observations = []
    engine = SurgicalEngine.restore(scene, snapshot, check=check, use_cache=False,
        on_boundary=lambda current: observations.append(observed_boundary(current)))
    exact['observations'] = observations
    initial = engine.fresh()
    exact.update(initialStateId=initial.state_id, initialSlots=initial.state['slots'], initialRemoved=initial.state['removed'])
    verify_history(engine, record, exact)
    exact['treatmentActionIndices'] = _treatment_indices(engine.actions, exact['recordedApproach'], observations)
    exact['boundaries'] = _transition_proof(scene, snapshot, engine.history, initial.state_id, initial.state['removed'])
    exact['contractHash'] = semantic_digest(exact)
    identity(case, scene, snapshot, record, exact)
    return exact
