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
from .surgical_replay_cache import SurgicalReplayCache

OPERATIONS = frozenset({'surgicalRehearsal', 'createSurgicalFixture', 'importSurgicalScene', 'prepareSurgicalEnvironment', 'compareSurgicalAnatomy'})
ARTIFACT = 'surgical_toolbox'
RESULT_BYTES = 32 * 1024**2 - 4096  # Reserve the bounded JSONL envelope.


def display_timeline(history):
    """Bound redundant UI event samples; canonical replay history stays exact."""
    result = []
    numeric = {'drill_work': ('inputEnergyJ', 'cuttingWorkJ', 'removedVolumeMm3'),
               'removal': ('cells', 'volumeMm3'), 'aspiration': ('fluid', 'debris'),
               'irrigation': ('volumeMm3',)}
    for entry in history:
        row = {k: copy.deepcopy(v) for k,v in entry.items() if k != 'events'}
        path = row.get('action', {}).pop('path', None)
        if path is not None:
            # Playback obtains actual poses from the canonical action archive.
            # The UI timeline needs operation/outcome, not every dense waypoint.
            row['recordedPathPoseCount'] = len(path)
            if path: row['recordedPathEndpoints'] = [path[0], path[-1]]
        events, groups = [], {}
        for event in entry.get('events', []):
            kind = event.get('kind')
            if kind not in numeric:
                events.append(copy.deepcopy(event)); continue
            if kind not in groups:
                summary = {'kind': kind, 'summarySamples': 0}
                groups[kind] = summary; events.append(summary)
            summary = groups[kind]; summary['summarySamples'] += 1
            for key in numeric[kind]: summary[key] = summary.get(key, 0) + event.get(key, 0)
            if kind == 'drill_work':
                summary.setdefault('removedCells', []).extend(event.get('removedCells', []))
                summary['pendingCells'] = event.get('pendingCells', 0)
        row['events'] = events; result.append(row)
    return result


def _frame_admission(session, frames, engine, snapshot, history, replay_cache,
                     identity, branch_id, check):
    """Private immutable playback input for one admitted branch generation."""
    admissions = getattr(session, 'surgical_frame_admissions', {})
    key = (identity, branch_id)
    item = admissions.get(key)
    # Both references belong to the canonical private session record, never
    # request JSON. Workspace replacement and cursor reconstruction replace
    # those objects and require a new admission.
    if item is None or item[0] is not snapshot or item[1] is not history or item[2] is not engine.scene:
        token = frames.admit(engine, snapshot, replay_cache, verified_history=history, check=check)
        shown = display_timeline(thaw_json(token.history))
        item = (snapshot, history, engine.scene, token, shown)
        admissions.pop(key, None)
        admissions[key] = item
        session.surgical_frame_admissions = dict(list(admissions.items())[-4:])
    return item[3], item[4]


def dispatch(session, operation, args, request, progress):
    from .desktop_bridge import BridgeError, _keys, _require_json_budget
    if operation == 'compareSurgicalAnatomy':
        _keys(args, {'beforeCaseHash', 'afterCaseHash', 'changedLayers', 'prefix', 'decisions'})
        _require_json_budget(args, 256 * 1024, 'SURGICAL_REQUEST_LIMIT', 'Comparison request exceeds budget')
        from .surgical_anatomy_comparison import compare_anatomy
        result = compare_anatomy(session._get_case(args['beforeCaseHash']), session._get_case(args['afterCaseHash']),
            changed_layers=args.get('changedLayers', ['vessels']), prefix=args.get('prefix', []),
            decisions=args.get('decisions', 1), check=request.check, progress=progress)
        _require_json_budget(result, 2 * 1024 * 1024, 'SURGICAL_RESULT_LIMIT', 'Comparison result exceeds budget')
        return result
    if operation == 'prepareSurgicalEnvironment':
        _keys(args, {'collectionId', 'snapshotId', 'layers', 'allowUnreviewedRegistration'})
        from .surgical_patient import prepare_surgical_environment
        from .imaging import load_case, read_case_artifacts
        collection = getattr(session, 'anatomy_collections', {}).get(args['collectionId'])
        if collection is None or collection.identity() != args['snapshotId']:
            raise ValueError('Native anatomy changed; reopen before preparing an environment')
        if args['collectionId'] in getattr(session, 'anatomy_drafts', {}):
            raise ValueError('Accept or discard the segmentation draft before preparing an environment')
        if type(args.get('allowUnreviewedRegistration', False)) is not bool:
            raise ValueError('Research registration choice must be explicit')
        progress(0.1, 'Preparing standard 1 mm research environment from selected source masks')
        with tempfile.TemporaryDirectory(prefix='surgical-environment-') as directory:
            report = prepare_surgical_environment(collection.root, directory, layer_ids=args['layers'],
                resolution_mm=1., allow_unreviewed_registration=args.get('allowUnreviewedRegistration', False))
            if report['sourceCollectionHash'] != args['snapshotId']:
                raise ValueError('Source collection changed during environment preparation')
            request.check()
            case = load_case(report['casePath'])
            artifacts = read_case_artifacts(report['casePath'])
        return session._install_case(case, artifacts, request)
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
    _keys(args, {'caseHash', 'command', 'expectedStateId', 'action', 'mapping', 'position', 'branchId', 'source', 'sceneIndex', 'acknowledgeResearchUnknowns', 'opponent', 'scenario', 'maxActions', 'frameIndex'})
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
    if command not in {'create', 'inspect', 'preview', 'step', 'undo', 'redo', 'seek', 'branch', 'selectBranch', 'reset', 'selectScene', 'plan', 'fullCase', 'seekFrame'}:
        raise ValueError('Unsupported surgical command')
    raw = artifacts.get(ARTIFACT)
    from .surgical_environment import standardize, default_target_names, SCHEMA as ENVIRONMENT_SCHEMA
    original_builder = scene_builder
    scene_builder = lambda mapping=None: standardize(original_builder(mapping), default_target_names(entry.case) if entry is not None and mapping is None else None,
        raw.get('physicsProfile') if raw and command != 'create' else None) if (
        command == 'create' or raw and raw.get('environmentSchema') == ENVIRONMENT_SCHEMA) else original_builder(mapping)
    if command == 'selectScene':
        index = args.get('sceneIndex')
        prior = artifacts.get('surgical_prior_scenes', [])
        if type(index) is not int or not 0 <= index < len(prior):
            raise ValueError('Previous environment is unavailable')
        # Exchange complete archives, including their mapping, physics and branches.
        selected = prior[index]
        prior[index] = raw
        raw = selected
        artifacts[ARTIFACT] = raw
        scene_builder = lambda mapping=None: standardize(original_builder(mapping), physics_profile=raw.get('physicsProfile')) if raw.get('environmentSchema') == ENVIRONMENT_SCHEMA else original_builder(mapping)
    cache = getattr(session, 'surgical_sessions', {})
    replay_cache = getattr(session, 'surgical_replay_cache', None)
    if replay_cache is None:
        replay_cache = SurgicalReplayCache(verified_cache=bool(raw and raw.get('fullCase')))
        session.surgical_replay_cache = replay_cache
    elif raw and raw.get('fullCase') and replay_cache.verified_cache is None:
        from .surgical_verified_cache import VerifiedSurgicalCache
        replay_cache.verified_cache = VerifiedSurgicalCache()
    cached = cache.get(identity)
    # Bundle artifacts are recursively immutable and privately constructed by
    # freeze_json at this boundary. Only an exact object we installed may reuse
    # hot admission. Replacement (even equal bytes) and scene exchange require
    # fresh replay. Native JSON remains externally writable and is always hashed.
    saved_hash = semantic_digest(raw) if destination else None
    admitted_bundle = (entry is not None and cached and len(cached) > 3
                       and cached[3] is entry.artifacts and command != 'selectScene')
    admitted_native = (destination is not None and cached and cached[0] == saved_hash
                       and command != 'selectScene')
    if command == 'create' and raw is not None and args.get('mapping') == raw.get('mapping') and raw.get('environmentSchema') == ENVIRONMENT_SCHEMA:
        command = 'inspect'  # An unchanged mapping must not erase a timeline.
    if command == 'create' or raw is None and command != 'inspect':
        scene = scene_builder(args.get('mapping'))
        from .surgical_rl_protection import GuardConfig
        engine = SurgicalEngine(scene, guard_config=GuardConfig(require_critical_coverage=False))
        record = {'environmentSchema': ENVIRONMENT_SCHEMA, 'physicsProfile': copy.deepcopy(scene.properties['tissuePhysicsV2']), 'mapping': scene.regions, 'branches': {'main': engine.snapshot()}, 'branchId': 'main', 'position': 0, 'parents': {'main': None}, 'timeline': {'main': []}}
    elif raw is None:
        return None
    elif admitted_bundle or admitted_native:
        # Microstep reads return before persistence and never mutate record or
        # the editable engine. Avoid copying the entire long archive twice.
        engine = cached[1]
        record = cached[2] if command == 'seekFrame' else copy.deepcopy(cached[2])
    else:
        record = copy.deepcopy(raw)
        scene = scene_builder(record['mapping'])
        full = SurgicalEngine.restore(scene, record['branches'][record['branchId']], check=request.check, use_cache=bool(record.get('fullCase')))
        engine, timeline = replay_cache.seek(full, record['branches'][record['branchId']], record['position'], check=request.check, include_history=True)
        record.setdefault('timeline', {})[record['branchId']] = timeline
    request.check()
    cursor_commands = {'undo', 'redo', 'seek', 'selectBranch'}
    # Cursor reconstruction returns a detached engine and never writes its
    # current-engine input. Avoid copying a complete unchanged effect state
    # before immediately replacing it with that reconstructed result.
    if command not in cursor_commands and command != 'seekFrame':
        engine = engine.clone()
    if command not in ('create', 'inspect', 'selectScene') and args.get('expectedStateId') != engine.state_id:
        raise BridgeError('SURGICAL_STATE_STALE', 'The scene changed. Refresh before applying this action.')
    outcome = None
    verified_display_state_id = None
    unchanged_tool = False
    if command == 'preview':
        result = {**engine.preview(args['action'], check=request.check, include_history=not bool(record.get('fullCase'))), 'caseHash': identity}
        _require_json_budget(result, RESULT_BYTES, 'SURGICAL_RESULT_LIMIT', 'Preview exceeds the display budget')
        return result
    if command == 'seekFrame':
        from .surgical_frame_replay import SurgicalFrameCache
        frames = getattr(session, 'surgical_frame_cache', None)
        if frames is None:
            frames = session.surgical_frame_cache = SurgicalFrameCache()
        branch_id = args.get('branchId', record['branchId'])
        if branch_id not in record['branches']: raise ValueError('Unknown playback branch')
        token, shown_timeline = _frame_admission(session, frames, engine,
            record['branches'][branch_id], record.get('timeline', {}).get(branch_id), replay_cache,
            identity, branch_id, request.check)
        displayed, _, playback = frames.seek(engine, token.snapshot, args.get('position'),
            args.get('frameIndex'), replay_cache, check=request.check,
            verified_history=token.history, admission=token, display_only=True)
        playback['branchId'] = branch_id
        # The frame cache just verified this complete state identity. Reuse it
        # for display instead of hashing the large effect state a second time.
        shown_view = displayed.view(_state_id=playback['frameStateId'], include_history=False)
        shown_view['history'] = shown_timeline[:len(displayed.actions)]
        result = {'caseHash': identity, 'view': shown_view, 'outcome': None, 'changed': False,
            'timeline': shown_timeline, 'playback': playback, 'branchId': branch_id, 'position': record['position'],
            'fullCase': copy.deepcopy(record.get('fullCase', {}).get(branch_id)),
            'branches': [{'id': k, 'steps': len(v['actions']), 'parent': record.get('parents', {}).get(k)} for k,v in record['branches'].items()]}
        _require_json_budget(result, RESULT_BYTES, 'SURGICAL_RESULT_LIMIT', 'Playback frame exceeds the display budget')
        return result
    if command == 'fullCase':
        from .full_case_rehearsal import bind_case_protection, new_director, run_slice, resume_director
        scenario = args.get('scenario', 'primary')
        director = record.get('fullCase', {}).get(record['branchId'])
        if not director or director.get('scenario') != scenario or director.get('lastStateId', director.get('startStateId')) != engine.state_id:
            parent = {'branchId': record['branchId'], 'position': record['position']}
            engine = bind_case_protection(engine)
            annotation = None
            if entry is not None:
                names = {r['name'] for r in engine.scene.regions if r.get('target')}
                annotation = sum(int(np.count_nonzero(entry.case.compartments[name])) for name in names if name in entry.case.compartments) * engine.scene.voxel_mm3
            director = new_director(engine, scenario, full_annotation_target_mm3=annotation)
            record['branchId'] = 'full-' + scenario + '-' + str(len(record['branches']))
            record.setdefault('parents', {})[record['branchId']] = parent
        director = resume_director(director)
        engine, director = run_slice(engine, director, max_actions=args.get('maxActions', 2), check=request.check, progress=progress)
        record.setdefault('fullCase', {})[record['branchId']] = director
        record['branches'][record['branchId']] = engine.snapshot()
        record['position'] = len(engine.actions)
        record.setdefault('timeline', {})[record['branchId']] = copy.deepcopy(engine.history)
        outcome = {'status': 'accepted', 'reason': director.get('stopReason') or director.get('currentPhaseId'), 'completedFraction': 1., 'events': []}
    elif command == 'plan':
        from .surgical_planning import compare_continuations
        branches, comparison = compare_continuations(engine,
            acknowledge_unknowns=args.get('acknowledgeResearchUnknowns', False),
            opponent=args.get('opponent', 'stop'), check=request.check, progress=progress)
        # Install only after BOTH complete deterministic replay checks pass.
        # The original live branch and cursor remain unchanged.
        for candidate, report in branches:
            branch_id = 'plan-' + str(len(record['branches'])) + '-' + report['method']
            report['branchId'] = branch_id
            record['branches'][branch_id] = candidate.snapshot()
            record.setdefault('timeline', {})[branch_id] = copy.deepcopy(candidate.history)
            record.setdefault('parents', {})[branch_id] = {'branchId': record['branchId'], 'position': record['position']}
        comparison['comparisonSeal'] = semantic_digest(comparison)
        record['planning'] = comparison
    elif command == 'step':
        action = args['action']
        unchanged_tool = action.get('operation') == 'exchange' and engine.state['slots'].get(action.get('slot'), {}).get('instrument') == action.get('instrument')
        outcome = {'status': 'accepted', 'reason': 'Instrument already selected; state unchanged.', 'completedFraction': 1., 'events': []} if unchanged_tool else engine.step(action, check=request.check)
        if outcome['status'] != 'rejected' and not unchanged_tool:
            full = record['branches'][record['branchId']]
            planned_ids = {row['branchId'] for row in (record.get('planning') or {}).get('alternatives', [])}
            if record['position'] != len(full['actions']) or record['branchId'] in planned_ids:
                parent = {'branchId': record['branchId'], 'position': record['position']}
                record['branchId'] = 'branch-' + str(len(record['branches']))
                record.setdefault('parents', {})[record['branchId']] = parent
            record['branches'][record['branchId']] = engine.snapshot()
            record['position'] = len(engine.actions)
            record.setdefault('timeline', {})[record['branchId']] = copy.deepcopy(engine.history)
    elif command in ('undo', 'redo', 'seek', 'selectBranch'):
        if command == 'selectBranch':
            if args['branchId'] not in record['branches']: raise ValueError('Unknown branch')
            record['branchId'] = args['branchId']
            record['position'] = len(record['branches'][record['branchId']]['actions'])
        snapshot = record['branches'][record['branchId']]
        n = {'undo': record['position'] - 1, 'redo': record['position'] + 1}.get(command, args.get('position', record['position']))
        if type(n) is not int or not 0 <= n <= len(snapshot['actions']): raise ValueError('History position is unavailable')
        frame_cache = getattr(session, 'surgical_frame_cache', None)
        observed_boundary = frame_cache.boundary(snapshot, n, verified_history=record.get('timeline', {}).get(record['branchId'])) if frame_cache is not None else None
        if observed_boundary is not None:
            engine, timeline = observed_boundary
        else:
            engine, timeline = replay_cache.seek(engine, snapshot, n, check=request.check, include_history=True)
        record['position'] = n
        # Rebuild optional legacy/malformed timelines from admitted history.
        record.setdefault('timeline', {})[record['branchId']] = timeline
        # seek checked the complete reconstructed state against this admitted
        # ancestry hash. No state mutation occurs between that check and view.
        # This identity never comes from the request's expectedStateId field.
        verified_display_state_id = timeline[n - 1]['stateAfter'] if n else (
            timeline[0]['stateBefore'] if timeline else snapshot['stateId'])
    elif command in ('branch', 'reset'):
        parent = {'branchId': record['branchId'], 'position': record['position'], 'reset': command == 'reset'}
        if command == 'reset': engine = engine.fresh()
        record['branchId'] = 'branch-' + str(len(record['branches']))
        record.setdefault('parents', {})[record['branchId']] = parent
        record['branches'][record['branchId']] = engine.snapshot()
        record['position'] = len(engine.actions)
        record.setdefault('timeline', {})[record['branchId']] = copy.deepcopy(engine.history)
    full_case = bool(record.get('fullCase'))
    shown_view = engine.view(_state_id=verified_display_state_id, include_history=not full_case)
    shown_timeline = record.get('timeline', {}).get(record['branchId'], engine.history)
    if full_case:
        shown_timeline = display_timeline(shown_timeline)
        shown_view['history'] = shown_timeline[:len(engine.actions)]
    result = {'caseHash': identity, 'view': shown_view, 'outcome': outcome,
              'changed': command != 'inspect' and not unchanged_tool and (outcome is None or outcome['status'] != 'rejected'),
              'timeline': shown_timeline,
              'previousScenes': [{'index': i, 'label': f"Environment {i + 1}", 'branches': len(s.get('branches', {}))} for i, s in enumerate(artifacts.get('surgical_prior_scenes', []))],
              'branchId': record['branchId'], 'position': record['position'],
              'fullCase': copy.deepcopy(record.get('fullCase', {}).get(record['branchId'])),
              'branches': [{'id': k, 'steps': len(v['actions']), 'parent': record.get('parents', {}).get(k)} for k, v in record['branches'].items()]}
    from .surgical_planning import availability
    result['planningAvailability'] = availability(engine)
    comparison = copy.deepcopy(record.get('planning'))
    if comparison is not None:
        from .surgical_planning import validate_saved_comparison
        validate_saved_comparison(engine.scene, record, comparison)
    result['planning'] = comparison
    _require_json_budget(result, RESULT_BYTES, 'SURGICAL_RESULT_LIMIT', 'Save this sequence before continuing')
    request.begin_commit()
    if command == 'create' and raw is not None:
        artifacts.setdefault('surgical_prior_scenes', []).append(raw)
        result['previousScenes'] = [{'index': i, 'label': f"Environment {i + 1}", 'branches': len(s.get('branches', {}))} for i, s in enumerate(artifacts['surgical_prior_scenes'])]
    persisted_record = copy.deepcopy(record)
    if persisted_record.get('fullCase'):
        # The timeline is derived from the action log by verified replay. Long
        # episodes must not serialize its redundant metrics/events in every
        # native save; keep the complete live timeline in the session/result.
        persisted_record.pop('timeline', None)
    artifacts[ARTIFACT] = persisted_record
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
    cache[identity] = (semantic_digest(persisted_record) if destination else None, engine, record,
                       entry.artifacts if entry is not None else None)
    session.surgical_sessions = dict(list(cache.items())[-4:])
    return result
