"""A separate manual episode at a causally reconstructed engine boundary."""
from __future__ import annotations

import copy
from uuid import UUID, uuid4

import numpy as np

from .core import semantic_digest
from .surgical_scene import original_target_memberships

SCHEMA = 'resectionlab.surgical-practice.v1'
# Frozen, reproducible controls. The legacy thermal coupon predates generated
# role metadata. A human-readable case label or a generated flag grants no access.
# The model digest covers the versioned snapshot fields (including guard/stages)
# other than the actions and their resulting checkpoint.
CRANIAL = 'sha256:9818506d2f779b8fece10d048e0f21e2357f86851936ad0f5b86ee6f6cfaa1e0'
THERMAL = 'sha256:c298887cb67b1034d782d682b01ee03cdaf86ee13237602eb3239ec114d5a710'
BLOCKED_OUTLET = 'sha256:33adb0e7a4405f6dc5231512a637fc519f3339bc3967b01dbfb2f8627962ed9e'
CONTROLS = {
    BLOCKED_OUTLET: ('sha256:8cb4482db8d770aca0983389f1d7bcd828a4d7c777bde83f3a69785d252e7d0b',
                     'sha256:f7b9fcc8a6c749812a6c6827907e8b0a92447debfefdbbf6ce91199fdf200631'),
    CRANIAL: ('sha256:8364dc4a5e12d30d3111362488b775160f0cc7b836e434c1bf4dc5a2cf179478',
              'sha256:c9d8d4d94acfa20f9c8a57c01dfb36b58562722ad32ce18e71550d9b1e9167a9'),
    THERMAL: ('sha256:3d860b9d648ab5fa606f0f18f01d4749e27d1e5a3405d3c0a0ca732e5cb6369a',
              'sha256:acd49ee35cda61286138c68d33e7ddc6c67ddafed20fad0444da144ae92f9a55'),
}
# Exact synthetic case manifests bind the raster, assignments and provenance.
# Keep the incision scout and biopsy controls readable alongside the wider scalp exposure.
CONVEXITY_CASES = {
    'sha256:a79b3d81bb593da6ef04a0925838053bb6b912878772ca066d22d405cb5ca86c':
        'sha256:4ace1749d19b920dd14fda0ed47cd802353d56e177b50be9c5a2a872d4112656',
    'sha256:80781bb098c0d874614472a3ccbc7c9c47963a81751d556685cc7e2f2604da41':
        'sha256:44fc58ffcc4ba2c22ba5f4fef3ff76ea0a7d3b72625554e6f4626c18d83ad023',
    'sha256:ba6fe43ff7b1e6f50f2f0051690a5e758354120f2c846a53c9d7cebefb3a0945':
        'sha256:368b168965cf135602da3c93269da448e1de7e2b9b916b2736cdad1c3321cbf0',
}
MANUAL_COMMANDS = {'inspect', 'inspectArchive', 'step', 'preview', 'seek', 'seekFrame', 'undo', 'redo', 'selectBranch'}


def refusal(reason):
    return {'available': False, 'reason': reason, 'checkpointSeal': None}


def model_hash(snapshot):
    return semantic_digest({k: v for k, v in snapshot.items() if k not in ('actions', 'checkpoint', 'stateId')})


def _convexity_target(scene, memberships, targets):
    # The digital phantom has two disjoint source masks for one lesion. Validate
    # its independently reproduced physical grid before sealing their joint task.
    from .core import array_digest
    from .surgical_planning_admission import source_classification
    names = ('Glioma margin', 'Glioma core')
    grid_digest = CONVEXITY_CASES.get(scene.source.get('caseHash'))
    if (grid_digest is None
            or source_classification(scene) != 'generated-engineering'
            or scene.source.get('caseId') != 'generated-convexity-glioma-v1'
            or array_digest(scene.labels) != grid_digest
            or array_digest(scene.affine) != 'sha256:b3dc49c8f1845a33a08c25a2b84532bd67f6b33a124ec36bad0103a47df549ca'
            or memberships is None or set(memberships) != set(names)
            or [(r['id'], r['name']) for r in targets] != list(zip((9, 10), names))):
        raise ValueError('The generated glioma grid and its source target masks must match the qualified anatomy.')
    parts = []
    for region in targets:
        name = region['name']
        if not np.array_equal(memberships[name], np.flatnonzero(scene.labels.ravel() == region['id'])):
            raise ValueError('The glioma source mask differs from its qualified anatomical membership.')
        parts.append({'regionId': region['id'], 'name': name, 'annotationHash': scene.source['annotationHashes'][name]})
    return {'name': 'Glioma core and margin', 'regions': parts, 'annotationHash': semantic_digest(parts)}


def _host_target(scene, memberships, targets):
    # A declared host rehearsal names its generated lesion layers. Each must
    # still own exactly the cells of its source mask.
    from .surgical_host_rehearsal import declaration
    names = [row['name'] for row in declaration(scene)['layers'] if row['role'] in ('tumour_rim', 'tumour_core')]
    if not names or memberships is None or set(memberships) != set(names) or [r['name'] for r in targets] != names:
        raise ValueError('The host rehearsal target differs from its declared generated lesion.')
    parts = []
    for region in targets:
        name = region['name']
        if not np.array_equal(memberships[name], np.flatnonzero(scene.labels.ravel() == region['id'])):
            raise ValueError('The generated lesion mask differs from its engine membership.')
        parts.append({'regionId': region['id'], 'name': name, 'annotationHash': scene.source['annotationHashes'][name]})
    return {'name': 'Generated tumour rim and core', 'regions': parts, 'annotationHash': semantic_digest(parts)}


def source_binding(engine, case_hash, snapshot, record):
    scene = engine.scene
    from .surgical_procedure_policy import PROPERTY, procedure_admission
    if scene.source.get('caseHash') != case_hash or snapshot.get('sceneId') != scene.identity:
        raise ValueError('The accepted practice checkpoint belongs to a different source.')
    if any(record.get(name) for name in ('taskAdmission', 'preparationApplications', 'exactGridReplay', 'sequentialEpisodes')):
        raise ValueError('A task-specific or imported recording grants no independent manual practice permission.')
    if case_hash in CONTROLS:
        model = model_hash(snapshot)
        if CONTROLS[case_hash] != (scene.identity, model):
            raise ValueError('The scene, physics or permission contract differs from the qualified practice control.')
    elif engine._operative:
        model = model_hash(snapshot)
    elif PROPERTY in scene.properties:
        # The native procedure gate verifies generated provenance, observed
        # domain, tissue permissions, guard and versioned physical models.
        policy = procedure_admission(scene, engine.protection)
        if not policy['available']:
            raise ValueError(policy['blockers'][0]['message'])
        if (snapshot.get('procedurePolicy') or {}).get('policyHash') != policy['policyHash']:
            raise ValueError('Practice snapshot does not carry the admitted procedure policy.')
        model = model_hash(snapshot)
    else:
        raise ValueError('No qualified manual practice domain for this source and target.')
    memberships = original_target_memberships(scene)
    targets = [r for r in scene.regions if r.get('target') and r['material'] == 'tumor']
    if engine._operative:
        from .surgical_host_rehearsal import declaration as host_rehearsal
        target_binding = (_host_target if host_rehearsal(scene) is not None else _convexity_target)(
            scene, memberships, targets)
    else:
        if memberships is None or len(targets) != 1 or len(memberships) != 1:
            raise ValueError('Practice requires one bound treatment target.')
        name = targets[0]['name']
        if name not in memberships or not len(memberships[name]):
            raise ValueError('Practice target has no original source membership.')
        target_binding = {'regionId': targets[0]['id'], 'name': name, 'annotationHash': scene.source['annotationHashes'][name]}
    from .surgical_tools import catalog
    return {'caseHash': case_hash, 'sceneId': scene.identity, 'modelHash': model,
            'targetBinding': target_binding,
            'instrumentCatalogHash': semantic_digest(catalog(operative=engine._operative_tool_revision))}


def checkpoint(binding, branch_id, position, state_id, actions):
    parent = {**binding, 'branchId': branch_id, 'position': position, 'stateId': state_id,
              'actionsHash': semantic_digest(actions)}
    return {**parent, 'checkpointSeal': semantic_digest(parent)}


def admission(engine, record, case_hash):
    if record.get('practiceEpisodes'):
        return refusal('This practice episode is already open.')
    branch = record['branchId']
    try:
        binding = source_binding(engine, case_hash, record['branches'][branch], record)
    except ValueError as error:
        return refusal(str(error))
    parent = checkpoint(binding, branch, len(engine.actions), engine.state_id, engine.actions)
    return {'available': True, 'reason': None, 'checkpointSeal': parent['checkpointSeal']}


def start(engine, record, case_hash, args, base_scene):
    snapshot = engine.snapshot()
    branch = record['branchId']
    position = len(engine.actions)
    if (args.get('branchId') != branch or type(args.get('position')) is not int or
            args['position'] != record['position'] or args['position'] != position):
        raise ValueError('Practice requires the current accepted branch and action boundary.')
    binding = source_binding(engine, case_hash, snapshot, record)
    parent = checkpoint(binding, branch, position, engine.state_id, engine.actions)
    if parent['checkpointSeal'] != args.get('checkpointSeal'):
        raise ValueError('Practice checkpoint changed. Return to its verified boundary.')
    episode_id = 'practice-' + str(uuid4())
    episode = {'schema': SCHEMA, 'episodeId': episode_id, 'parent': parent}
    child = {k: copy.deepcopy(record[k]) for k in ('environmentSchema', 'physicsProfile', 'mapping') if k in record}
    child.update(branchId=episode_id, position=position, branches={episode_id: snapshot},
                 timeline={episode_id: copy.deepcopy(engine.history)}, parents={episode_id: {'branchId': branch, 'position': position}},
                 practiceEpisodes={episode_id: episode})
    if branch in record.get('cranialPlanningRoots', {}).get('branchRoots', {}):
        from .surgical_cranial_planning_roots import create_registry
        registry = record['cranialPlanningRoots']
        root_id = registry['branchRoots'][branch]
        root = next(row for row in registry['roots'] if row['id'] == root_id)
        declaration = {key: root[key] for key in ('id', 'label', 'openingProperty', 'openingContract', 'sourceSnapshot')}
        child['cranialPlanningRoots'] = create_registry(base_scene, [declaration], {episode_id: root_id})
    # Legacy standard environments can omit their implicit profile. Persist it
    # explicitly so load_environment() and the desktop agree on a cold reopen.
    if 'environmentSchema' in child and 'physicsProfile' not in child:
        child['physicsProfile'] = copy.deepcopy(engine.scene.properties['tissuePhysicsV2'])
    return child


def verify_episode(engine, record, case_hash):
    """Called against the full native replay, before seeking its saved cursor."""
    branch = record['branchId']
    episodes = record['practiceEpisodes']
    invalid = 'Practice episode or parent checkpoint differs from its accepted action prefix.'
    if type(episodes) is not dict or set(episodes) != {branch} or set(record['branches']) != {branch}:
        raise ValueError(invalid)
    episode = episodes[branch]
    try:
        valid_id = branch == 'practice-' + str(UUID(branch.removeprefix('practice-')))
        parent = episode['parent']
        position = parent['position']
        if (not valid_id or set(episode) != {'schema', 'episodeId', 'parent'} or episode['schema'] != SCHEMA or
                episode['episodeId'] != branch or type(parent['branchId']) is not str or not parent['branchId'] or
                parent['branchId'] == branch or type(position) is not int or not 0 <= position <= len(engine.actions) or
                record['parents'] != {branch: {'branchId': parent['branchId'], 'position': position}}):
            raise ValueError(invalid)
        state = engine.history[position-1]['stateAfter'] if position else (
            engine.history[0]['stateBefore'] if engine.history else engine.state_id)
        binding = source_binding(engine, case_hash, record['branches'][branch], record)
        if parent != checkpoint(binding, parent['branchId'], position, state, engine.actions[:position]):
            raise ValueError(invalid)
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError(invalid) from error


def metrics(engine, case):
    """Report original mask volume and exclusively source-declared brain loss."""
    from .surgical_sequence_outcomes import _original_target_cells
    cells = _original_target_cells(engine.scene)
    voxel = engine.scene.voxel_mm3
    brain_removed = None
    brain = case.brain_mask
    if brain is None:
        layers = case.metadata.get('anatomy_layers', {})
        names = [r['name'] for r in engine.scene.regions if layers.get(r['name'], {}).get('kind') == 'brain']
        if engine._operative:
            brain = np.isin(engine.scene.labels, [r['id'] for r in engine.scene.regions
                if layers.get(r['name'], {}).get('kind') in ('grayMatter', 'whiteMatter', 'tract')])
        elif len(names) == 1:
            brain = case.compartments.get(names[0])
        elif case.semantic_hash == THERMAL and engine.scene.identity == CONTROLS[THERMAL][0]:
            # The frozen coupon labels its complete modeled brain compartment
            # as soft tissue. Other soft-tissue or neural regions have no such binding.
            brain = case.compartments['Generated brain']
    if brain is not None and cells is not None:
        removed = np.setdiff1d(engine.state['removed'], cells).astype(np.intp)
        brain_removed = float(np.count_nonzero(brain.flat[removed]) * voxel)
    return {'originalTargetVolumeMm3': float(len(cells)*voxel) if cells is not None else None,
            'nonTargetBrainRemovedMm3': brain_removed}
