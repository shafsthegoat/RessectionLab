"""Join one recorded engine branch to the currently open native study and target."""
from __future__ import annotations

import numpy as np

from .core import semantic_digest
from .patient_anatomy import PatientImagingCollection
from .surgical_engine import SurgicalEngine
from .surgical_environment import SCHEMA, resolve_branch_scene, standardize
from .surgical_scene import from_case
from .surgical_task_admission import verify_source
from .surgical_tools import catalog


def study_target(collection, target_id):
    from .anatomy_approaches import _read_plan
    from .anatomy_planning import native_target_binding

    _, _, active, selection, _ = _read_plan(collection)
    declared = [row['id'] for row in collection.document['layers'] if row['role'] == 'target']
    selected = [active] if active else declared
    if not target_id or selected != [target_id]:
        raise ValueError('Select the recorded target in the native study before opening this episode')
    row = collection.layer(target_id)
    binding = native_target_binding(collection, row)
    if selection is not None and selection['candidateBinding'] != binding:
        raise ValueError('The selected native target evidence changed')
    return binding


def _recorded_branch(case, record):
    if not isinstance(record, dict) or record.get('environmentSchema') != SCHEMA:
        raise ValueError('Select a saved standard SurgicalEngine environment with recorded actions')
    branch_id = record.get('branchId')
    snapshot = record.get('branches', {}).get(branch_id)
    if not isinstance(snapshot, dict) or not snapshot.get('actions'):
        raise ValueError('The saved active branch has no recorded engine actions')
    scene = standardize(from_case(case, record['mapping']), physics_profile=record['physicsProfile'])
    if record.get('cranialPlanningRoots') is not None:
        scene = resolve_branch_scene(scene, record, branch_id)
    adapter = scene.source.get('adapter') or {}
    targets = [r for r in scene.regions if r['target']]
    if len(targets) != 1 or targets[0]['material'] != 'tumor':
        raise ValueError('The recording requires one mapped tumor treatment target')
    target_id = adapter.get('sourceLayers', {}).get(targets[0]['name'], {}).get('layerId')
    if 'exactGridReplay' in record:
        target_id = record['exactGridReplay']['targetLayerId']
    if not target_id:
        raise ValueError('Recorded treatment target lacks a native layer binding')
    if type(record.get('position')) is not int or not 0 <= record['position'] <= len(snapshot['actions']):
        raise ValueError('Recorded playback cursor is outside the recorded branch')
    return scene, snapshot, target_id


def _receipt(case, scene, snapshot, target_id, record):
    from .anatomy_planning import manifest_target_binding
    if 'exactGridReplay' in record:
        from .surgical_exact_grid_replay import fixed_route_identity
        source = fixed_route_identity(case, scene, snapshot, record['exactGridReplay'])
    else:
        adapter = scene.source['adapter']
        manifest = adapter['sourceManifest']
        layer = next(row for row in manifest['layers'] if row['id'] == target_id)
        source = dict(patientId=adapter['sourcePatient'], sourceSnapshotId=adapter['sourceCollectionHash'],
                      targetBinding=manifest_target_binding(manifest, layer))
    state = snapshot['checkpoint']['state']
    return {'schema': 'resectionlab.native-study-engine-replay.v1', **source, 'targetLayerId': target_id,
        'caseHash': case.semantic_hash, 'sceneId': scene.identity, 'branchId': record['branchId'],
        'actions': len(snapshot['actions']), 'actionsHash': semantic_digest(snapshot['actions']),
        'endpointStateId': snapshot['stateId'], 'snapshotHash': semantic_digest(snapshot),
        'physicsHash': semantic_digest(scene.properties), 'instrumentCatalogHash': semantic_digest(catalog()),
        'endpointSlotsHash': semantic_digest(state['slots']), 'endpointRemovalHash': semantic_digest(state['removed'])}


def validate_saved_binding(case, artifacts):
    """Offline archive consistency only. Never mint a live-source verification receipt."""
    record = artifacts.get('surgical_toolbox')
    try:
        scene, snapshot, target_id = _recorded_branch(case, record)
        expected = _receipt(case, scene, snapshot, target_id, record)
        matches = (snapshot['sceneId'] == scene.identity and snapshot['checkpoint']['stateId'] == snapshot['stateId']
                   and artifacts['nativeStudyReplay'] == expected)
    except (KeyError, TypeError, StopIteration, ValueError) as error:
        raise ValueError('STUDY_REPLAY_BINDING_CHANGED: saved recording identity is incomplete') from error
    if not matches:
        raise ValueError('STUDY_REPLAY_BINDING_CHANGED: saved study, target, tools or recorded branch differs')


def _verify_resampled_source(collection, case, scene, check):
    from .surgical_patient import _research_grid, _resample
    adapter = scene.source['adapter']
    if (adapter.get('sourcePatient') != collection.document['patient_id']
            or adapter.get('sourceCollectionHash') != collection.identity()):
        raise ValueError('Recorded patient or native study revision differs from the open study')
    verify_source(scene, native_collection=collection.root, check=check)
    # The source verifier rebuilds the material field. Also reproduce the MRI
    # overlay so the viewer cannot display substituted pixels around those tools.
    reference = collection.series(collection.document['reference_series'])
    source_image = collection.array(reference['data'])
    resolution = adapter.get('resolutionMm')
    if type(resolution) not in (int, float) or not .5 <= resolution <= 2.:
        raise ValueError('Recorded engine grid has an unsupported physical sampling interval')
    affine, expected_affine, expected_shape = _research_grid(collection, reference, resolution)
    if tuple(expected_shape) != scene.shape or not np.array_equal(expected_affine, scene.affine):
        raise ValueError('Recorded engine grid differs from the registered reference image frame')
    # Use the same interpolation as export, including its oblique boundary convention.
    reference_pixels = _resample(source_image, affine, scene.affine, scene.shape)
    if not np.array_equal(reference_pixels, case.mri):
        raise ValueError('Recorded reference image differs from the open study on the engine grid')


def verify_recording(collection, target_id, case, record, *, approach_id=None, check=lambda: None):
    check()
    binding = study_target(collection, target_id)
    scene, snapshot, mapped_target = _recorded_branch(case, record)
    if mapped_target != target_id:
        raise ValueError('Recorded treatment target differs from the selected native target')
    receipt = _receipt(case, scene, snapshot, target_id, record)
    exact = record.get('exactGridReplay')
    if exact is not None:
        from .surgical_exact_grid_replay import verify_exact_source
        verify_exact_source(collection, case, scene, exact, approach_id, check=check)
    else:
        _verify_resampled_source(collection, case, scene, check)
    SurgicalEngine.restore(scene, snapshot, check=check, use_cache=False)
    native = collection.layer(target_id)
    raw = collection.array(native['data'])
    count = sum(int(np.count_nonzero(raw[:, :, k] == native['label_value'])) for k in range(raw.shape[2]))
    basis = np.asarray(native['affine'])[:3, :3]
    from .surgical_scene import original_target_memberships
    cells = next(iter(original_target_memberships(scene).values()))
    sampling = {'nativeAnnotationMm3': count * abs(float(np.linalg.det(basis))),
        'nativeGridSpacingMm': np.linalg.norm(basis, axis=0).tolist(),
        'engineAnnotationMm3': len(cells) * scene.voxel_mm3, 'engineGridSpacingMm': scene.spacing.tolist()}
    current = PatientImagingCollection(collection.root)
    if exact is not None:
        from .surgical_exact_grid_replay import verify_selected_route
        verify_selected_route(current, exact, approach_id)
    if (current.identity() != collection.identity() or study_target(current, target_id) != binding
            or receipt['targetBinding'] != binding):
        raise ValueError('Native study or target selection changed while checking the episode')
    return receipt, sampling
