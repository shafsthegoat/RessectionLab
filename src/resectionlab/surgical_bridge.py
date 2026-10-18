"""Desktop transaction/persistence boundary for the general instrument engine."""
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import numpy as np
from .core import CaseData, SourceRef, freeze_json, thaw_json, semantic_digest
from .surgical_scene import from_case, engineering_scene
from .surgical_engine import SurgicalEngine

OPERATIONS = frozenset({'surgicalRehearsal', 'createSurgicalFixture', 'importSurgicalScene'})
ARTIFACT = 'surgical_toolbox'


def dispatch(session, operation, args, request, progress):
    from .desktop_bridge import BridgeError, _keys, _require_json_budget
    if operation == 'importSurgicalScene':
        _keys(args, {'path', 'frame', 'units', 'material', 'resolution', 'topology', 'thicknessMm'})
        from .surgical_adapters import from_nifti, from_mesh
        from .workspace_bundle import OriginalFile, view_state
        path = Path(args['path'])
        if path.stat().st_size > 256 * 1024**2: raise ValueError('Source file exceeds the 256 MiB import budget')
        if path.name.endswith(('.nii', '.nii.gz')):
            import nibabel as nib
            im = nib.load(path)
            if np.prod(im.shape) > 64_000_000: raise ValueError('Labelmap exceeds the interactive grid budget')
            labels = np.unique(np.asanyarray(im.dataobj))
            s = from_nifti(path, {int(i): args.get('material', 'unknown') for i in labels if i},
                           frame=args['frame'], units=args.get('units'))
        else:
            s = from_mesh(path, args.get('material', 'unknown'), frame=args['frame'], units=args['units'],
                          resolution=args.get('resolution', 1.), topology=args.get('topology', 'solid'),
                          thickness_mm=args.get('thicknessMm'))
        case = CaseData(case_id=path.name, mri=np.zeros(s.shape, np.float32),
            compartments={r['name']: s.labels == r['id'] for r in s.regions}, affine=s.affine,
            source_refs=(SourceRef('imported-segmentation', path.resolve().as_uri(), s.source['sha256'], native_frame=args['frame']),),
            metadata={'generated_signal': True, 'surgical_adapter': s.source, 'surgical_properties': s.properties,
                'anatomy_layers': {r['name']: {'kind': r['material'], 'color': r['color'], 'provenance': 'supplied segmented geometry'} for r in s.regions},
                'description': 'Imported segmentation; no acquired intensity image supplied'})
        return session._install_case(case, {}, request, workspace={'sources': [], 'imagingState': view_state(case, []),
            'episode': None, 'selection': None, 'sessionHash': None,
            'originals': (OriginalFile.from_path(path, role='annotation', expected_sha256=s.source['sha256']),)})
    if operation == 'createSurgicalFixture':
        _keys(args, set())
        s = engineering_scene()
        case = CaseData(case_id=s.source['caseId'], mri=s.labels.astype('float32'),
            compartments={r['name']: s.labels == r['id'] for r in s.regions}, affine=s.affine, source_refs=(SourceRef("surgical-engineering-v1", "synthetic://resectionlab/surgical-engineering-v1", native_frame="RAS+", provenance="simulated"),) ,
            metadata={'is_synthetic': True, 'generated': True, 'anatomy_layers': {r['name']: {'kind': r['material'], 'color': r['color'],
                       'provenance': 'generated engineering fixture'} for r in s.regions}, 'surgical_properties': s.properties})
        return session._install_case(case, {}, request)
    _keys(args, {'caseHash', 'command', 'expectedStateId', 'action', 'mapping', 'position', 'branchId', 'source'})
    _require_json_budget(args, 256 * 1024, 'SURGICAL_REQUEST_LIMIT', 'Instrument request exceeds budget')
    entry = None
    source = args.get('source')
    destination = None
    if source:
        from .surgical_adapters import from_native_collection
        from .surgical_scene import KINDS
        if set(source) != {'collectionId', 'snapshotId', 'seriesId'}: raise ValueError('Invalid native source binding')
        collection = getattr(session, 'anatomy_collections', {}).get(source['collectionId'])
        if collection is None or collection.identity() != source['snapshotId']:
            raise ValueError('Native anatomy changed; reopen before rehearsing')
        if json.loads((collection.root / 'patient-anatomy.json').read_text())['snapshot_id'] != source['snapshotId']:
            raise ValueError('Native source changed on disk')
        if source['seriesId'] != collection.document['reference_series']:
            raise ValueError('Select the registered reference series for surgical interaction')
        if source['collectionId'] in getattr(session, 'anatomy_drafts', {}):
            raise ValueError('Accept or discard the segmentation draft before rehearsing')
        identity = 'sha256:' + hashlib.sha256((collection.identity() + ':' + source['seriesId']).encode()).hexdigest()
        if args['caseHash'] != identity: raise ValueError('Displayed native geometry does not match the simulation source')
        def scene_builder(mapping=None):
            region_map = {r['name']: r for r in mapping} if mapping else {r['id']: {'name': r['id'],
                'material': KINDS.get(r.get('kind'), 'unknown'), 'color': r.get('color', '#92a7b8'),
                'target': r.get('kind') == 'tumor'} for r in collection.document['layers']}
            return from_native_collection(collection, region_map, frame='RAS+')
        destination = collection.root / 'surgical-toolbox.json'
        artifacts = json.loads(destination.read_text()) if destination.exists() else {}
        if artifacts.get('sourceIdentity', identity) != identity:
            raise ValueError('Saved tools belong to an older anatomy revision; preserve that file and prepare a new source session')
    else:
        entry = session._get_case(args['caseHash'])
        identity = entry.case.semantic_hash
        scene_builder = lambda mapping=None: from_case(entry.case, mapping)
        artifacts = thaw_json(entry.artifacts)
    command = args.get('command', 'inspect')
    if command not in {'create', 'inspect', 'preview', 'step', 'undo', 'redo', 'seek', 'branch', 'selectBranch', 'reset'}:
        raise ValueError('Unsupported surgical command')
    raw = artifacts.get(ARTIFACT)
    cache = getattr(session, 'surgical_sessions', {})
    saved_hash = semantic_digest(raw)
    cached = cache.get(identity)
    if command == 'create' or raw is None and command != 'inspect':
        scene = scene_builder(args.get('mapping'))
        engine = SurgicalEngine(scene)
        record = {'mapping': scene.regions, 'branches': {'main': engine.snapshot()}, 'branchId': 'main', 'position': 0, 'parents': {'main': None}}
    elif raw is None:
        return None
    elif cached and cached[0] == saved_hash:
        engine, record = cached[1], copy.deepcopy(cached[2])
    else:
        record = copy.deepcopy(raw)
        scene = scene_builder(record['mapping'])
        full = SurgicalEngine.restore(scene, record['branches'][record['branchId']])
        engine = full.branch(record['position'])
    request.check()
    engine = engine.clone()
    if command not in ('create', 'inspect') and args.get('expectedStateId') != engine.state_id:
        raise BridgeError('SURGICAL_STATE_STALE', 'The scene changed. Refresh before applying this action.')
    outcome = None
    if command == 'preview':
        return {**engine.preview(args['action']), 'caseHash': identity}
    if command == 'step':
        outcome = engine.step(args['action'])
        if outcome['status'] != 'rejected':
            full = record['branches'][record['branchId']]
            if record['position'] != len(full['actions']):
                parent = {'branchId': record['branchId'], 'position': record['position']}
                record['branchId'] = 'branch-' + str(len(record['branches']))
                record.setdefault('parents', {})[record['branchId']] = parent
            record['branches'][record['branchId']] = engine.snapshot()
            record['position'] = len(engine.actions)
    elif command in ('undo', 'redo', 'seek', 'selectBranch'):
        if command == 'selectBranch':
            if args['branchId'] not in record['branches']: raise ValueError('Unknown branch')
            record['branchId'] = args['branchId']
            record['position'] = len(record['branches'][record['branchId']]['actions'])
        full = SurgicalEngine.restore(engine.scene, record['branches'][record['branchId']])
        n = {'undo': record['position'] - 1, 'redo': record['position'] + 1}.get(command, args.get('position', record['position']))
        if type(n) is not int or not 0 <= n <= len(full.actions): raise ValueError('History position is unavailable')
        engine = full.branch(n); record['position'] = n
    elif command in ('branch', 'reset'):
        parent = {'branchId': record['branchId'], 'position': record['position'], 'reset': command == 'reset'}
        if command == 'reset': engine = SurgicalEngine(engine.scene)
        record['branchId'] = 'branch-' + str(len(record['branches']))
        record.setdefault('parents', {})[record['branchId']] = parent
        record['branches'][record['branchId']] = engine.snapshot()
        record['position'] = len(engine.actions)
    result = {'caseHash': identity, 'view': engine.view(), 'outcome': outcome,
              'branchId': record['branchId'], 'position': record['position'],
              'branches': [{'id': k, 'steps': len(v['actions']), 'parent': record.get('parents', {}).get(k)} for k, v in record['branches'].items()]}
    _require_json_budget(result, 32 * 1024 * 1024, 'SURGICAL_RESULT_LIMIT', 'Save this sequence before continuing')
    request.begin_commit()
    if command == 'create' and raw is not None:
        artifacts.setdefault('surgical_prior_scenes', []).append(raw)
    artifacts[ARTIFACT] = record
    if destination:
        artifacts['sourceIdentity'] = identity
        data = json.dumps(artifacts, allow_nan=False, separators=(',', ':'))
        if len(data) > 32 * 1024**2: raise ValueError('Native surgical history exceeds archive budget')
        fd, name = tempfile.mkstemp(dir=destination.parent, prefix='.surgical-', suffix='.json')
        try:
            with os.fdopen(fd, 'w') as stream: stream.write(data)
            os.replace(name, destination)
        finally:
            Path(name).unlink(missing_ok=True)
    else:
        entry.artifacts = freeze_json(artifacts)
    cache[identity] = (semantic_digest(record), engine, record)
    session.surgical_sessions = dict(list(cache.items())[-4:])
    return result
