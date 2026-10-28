"""Desktop transaction/persistence boundary for the general instrument engine."""
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
from types import MappingProxyType
import numpy as np
from .core import CaseData, SourceRef, freeze_json, thaw_json, semantic_digest
from .surgical_scene import from_case, engineering_scene
from .surgical_engine import SurgicalEngine
from .surgical_replay_cache import SurgicalReplayCache

OPERATIONS = frozenset({'surgicalRehearsal', 'createSurgicalFixture', 'importSurgicalScene', 'prepareSurgicalEnvironment', 'compareSurgicalAnatomy'})
ARTIFACT = 'surgical_toolbox'
RESULT_BYTES = 32 * 1024**2 - 4096  # Reserve the bounded JSONL envelope.
PREFIX_COMMANDS = frozenset({'seekPrefix', 'seekPrefixFrame'})


def display_timeline(history):
    """Bound redundant UI event samples; canonical replay history stays exact."""
    def detach(value):
        # Private frozen prefix receipts are already recursively immutable.
        return thaw_json(value) if isinstance(value, (MappingProxyType, tuple)) else copy.deepcopy(value)
    result = []
    numeric = {'drill_work': ('inputEnergyJ', 'cuttingWorkJ', 'removedVolumeMm3'),
               'removal': ('cells', 'volumeMm3'), 'aspiration': ('fluid', 'debris'),
               'irrigation': ('volumeMm3',)}
    for entry in history:
        row = {k: ({name: detach(value) for name, value in v.items() if name != 'path'}
                   if k == 'action' else detach(v)) for k,v in entry.items() if k != 'events'}
        path = entry.get('action', {}).get('path')
        if path is not None:
            # Playback obtains actual poses from the canonical action archive.
            # The UI timeline needs operation/outcome, not every dense waypoint.
            row['recordedPathPoseCount'] = len(path)
            if path: row['recordedPathEndpoints'] = [detach(path[0]), detach(path[-1])]
        events, groups = [], {}
        for event in entry.get('events', []):
            kind = event.get('kind')
            if kind not in numeric:
                events.append(detach(event)); continue
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


def display_receipt(entry):
    """Row-local prefix display projection; canonical receipt stays complete."""
    return display_timeline((entry,))[0]


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
        serial = getattr(session, 'surgical_frame_serial', 0) + 1
        session.surgical_frame_serial = serial
        stream_token = token.snapshot_hash + '/' + token.history_hash + '/' + str(serial)
        item = (snapshot, history, engine.scene, token, shown, stream_token)
        admissions.pop(key, None)
        admissions[key] = item
        session.surgical_frame_admissions = dict(list(admissions.items())[-4:])
    return item[3], item[4], item[5]


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
    argument_keys = {'caseHash', 'command', 'expectedStateId', 'action', 'mapping', 'position', 'branchId', 'source', 'sceneIndex', 'acknowledgeResearchUnknowns', 'allowModeledBrainCorridor', 'maxHealthyRemovalMm3', 'maxWallSeconds', 'opponent', 'scenario', 'maxActions', 'frameIndex', 'timelineToken', 'viewFieldTokens','initialAccessRadiusMm'}
    if args.get('command') == 'seekPrefixFrame':
        argument_keys.update({'prefixToken', 'prefixFieldTokens'})
    if args.get('command') == 'seekPrefix' and type(args.get('position')) is int and args['position'] == 0:
        argument_keys.add('cacheOnly')
    _keys(args, argument_keys)
    if 'initialAccessRadiusMm' in args and (args.get('command')!='plan' or
            args.get('opponent')!='anatomy-staged-strategy'):
        raise ValueError('Initial access configuration applies only to staged source planning')
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
        prior_admission = getattr(session, 'surgical_sessions', {}).get(identity)
        hot_frame = (args.get('command') == 'seekFrame' and prior_admission and len(prior_admission) > 3
                     and prior_admission[3] is entry.artifacts)
        # A frame is a read-only request over the exact immutable artifact we
        # already admitted. It returns before persistence. Borrow this object
        # rather than thawing every saved branch/path/checkpoint on each frame.
        # Replacement archives and all editable commands retain full admission.
        artifacts = entry.artifacts if hot_frame or args.get('command') in PREFIX_COMMANDS else thaw_json(entry.artifacts)
    command = args.get('command', 'inspect')
    if command not in {'create', 'inspect', 'inspectArchive', 'preview', 'step', 'undo', 'redo', 'seek', 'branch', 'selectBranch', 'reset', 'selectScene', 'plan', 'fullCase', 'seekFrame', *PREFIX_COMMANDS}:
        raise ValueError('Unsupported surgical command')
    raw = artifacts.get(ARTIFACT)
    from .surgical_environment import standardize, default_target_names, SCHEMA as ENVIRONMENT_SCHEMA
    original_builder = scene_builder
    scene_builder = lambda mapping=None: standardize(original_builder(mapping), default_target_names(entry.case) if entry is not None and mapping is None else None,
        raw.get('physicsProfile') if raw and command != 'create' else None) if (
        command == 'create' or raw and raw.get('environmentSchema') == ENVIRONMENT_SCHEMA) else original_builder(mapping)
    if command in PREFIX_COMMANDS:
        # This independent read-only owner never populates the editable session,
        # canonical admission, or either authenticated whole-archive cache.
        prefix_keys = {'caseHash', 'command', 'branchId', 'position', 'frameIndex'}
        if command == 'seekPrefix' and type(args.get('position')) is int and args['position'] == 0:
            prefix_keys.add('cacheOnly')
        if command == 'seekPrefixFrame':
            prefix_keys.update({'prefixToken', 'prefixFieldTokens'})
        _keys(args, prefix_keys)
        if 'cacheOnly' in args and type(args['cacheOnly']) is not bool:
            raise ValueError('Prepared-only replay requires a boolean cacheOnly flag')
        if 'prefixToken' in args and (not isinstance(args['prefixToken'], str)
                or not 1 <= len(args['prefixToken']) <= 1024):
            raise ValueError('Invalid read-only prefix display generation')
        prefix_fields = args.get('prefixFieldTokens', {})
        if (not isinstance(prefix_fields, dict) or set(prefix_fields) - {'flapComponents'}
                or any(not isinstance(value, str) or len(value) != 64
                       or any(c not in '0123456789abcdef' for c in value)
                       for value in prefix_fields.values())):
            raise ValueError('Invalid read-only prefix field acknowledgement')
        if entry is None or raw is None:
            raise ValueError('Read-only prefix replay requires a saved workspace branch')
        selected = args.get('branchId', raw.get('branchId'))
        if not isinstance(selected, str) or selected not in raw.get('branches', {}):
            raise ValueError('Unknown prefix replay branch')
        if command == 'seekPrefix' and 'frameIndex' in args:
            raise ValueError('Boundary prefix requests do not select observed frames')
        owned = getattr(session, 'surgical_prefix_session', None)
        restarting = command == 'seekPrefix' and type(args.get('position')) is int and args['position'] == 0
        if restarting:
            # An explicit rewind starts a new bounded reconstruction, including
            # after an old boundary/frame was evicted. It grants no authority.
            owned = None
        matches_owner = bool(owned and owned[0] is entry and owned[1] is entry.artifacts and owned[2] == selected)
        if not restarting and not matches_owner:
            raise BridgeError('SURGICAL_PREFIX_OWNER_REQUIRED',
                'Read-only replay source, branch or session changed; explicitly restart from the beginning before continuing.')
        if not matches_owner:
            from .surgical_prefix_replay import SurgicalPrefixReplay
            from .surgical_prefix_cache import PreparedPrefixStore
            # This branch belongs to the bundle's privately frozen artifact.
            # Future reads retain that exact origin, independently of edits or
            # canonical session state.
            frozen_snapshot = raw['branches'][selected]
            # Only the small mapping/profile need ordinary JSON. Keep the
            # original private frozen action log/checkpoint shared and bounded.
            raw_profile, raw_mapping = thaw_json(raw.get('physicsProfile')), thaw_json(raw['mapping'])
            prefix_scene = original_builder(raw_mapping)
            if raw.get('environmentSchema') == ENVIRONMENT_SCHEMA:
                prefix_scene = standardize(prefix_scene, physics_profile=raw_profile)
            owner = SurgicalPrefixReplay._borrow_frozen_snapshot(prefix_scene, frozen_snapshot,
                max_input_bytes=128 * 1024**2, max_frame_bytes=64 * 1024**2,
                _display_row_projection=display_receipt, _prepared_store=PreparedPrefixStore(),
                _cache_only=args.get('cacheOnly', False))
            request.check()
            owned = (entry, entry.artifacts, selected, owner)
            session.surgical_prefix_session = owned
        owner = owned[3]
        progress(.1, 'Loading prepared observations in a separate read-only prefix' if
                 owner._cache_only else
                 'Reconstructing saved actions in a separate read-only prefix')
        result = (owner.seek(args.get('position'), check=request.check) if command == 'seekPrefix'
                  else owner.seek_frame(args.get('position'), args.get('frameIndex'), check=request.check,
                                        prefix_token=args.get('prefixToken'), prefix_field_tokens=prefix_fields))
        if 'playback' in result:
            result['playback']['branchId'] = selected
        result.update(caseHash=identity, branchId=selected, changed=False, outcome=None,
            fullCase=None, branches=[{'id': name, 'steps': len(snapshot['actions'])}
                                     for name, snapshot in raw['branches'].items()])
        _require_json_budget(result, RESULT_BYTES, 'SURGICAL_RESULT_LIMIT', 'Read-only prefix exceeds the display budget', _builtin_fast=True)
        request.check()
        return result
    prefix_owner = getattr(session, 'surgical_prefix_session', None)
    prior_canonical = getattr(session, 'surgical_sessions', {}).get(identity)
    if (entry is not None and prefix_owner and prefix_owner[0] is entry
            and prefix_owner[1] is entry.artifacts and command not in {'inspect', 'inspectArchive'}
            and not (prior_canonical and len(prior_canonical) > 3 and prior_canonical[3] is entry.artifacts)):
        raise BridgeError('SURGICAL_PREFIX_READ_ONLY',
            'Reconstructed prefixes are read only. Explicitly verify the full archive before planning or editing.')
    if command == 'inspectArchive':
        if raw is None:
            return None
        if raw.get('fullCase'):
            from .surgical_archive_view import inspect_archive_checkpoint
            result = inspect_archive_checkpoint(scene_builder(raw['mapping']), raw,
                branch_id=args.get('branchId'), check=request.check)
            result.update(caseHash=identity, changed=False)
            _require_json_budget(result, RESULT_BYTES, 'SURGICAL_RESULT_LIMIT', 'Imported checkpoint exceeds the display budget')
            # Display data never enters surgical_sessions, artifacts, or either
            # replay cache. Ordinary inspect still requires canonical replay.
            return result
        command = 'inspect'
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
    verify_branch = command == 'inspect' and args.get('branchId') is not None
    if verify_branch:
        if raw is None or args['branchId'] not in raw['branches']:
            raise ValueError('Unknown verification branch')
        raw = copy.deepcopy(raw)
        raw['branchId'] = args['branchId']
        raw['position'] = len(raw['branches'][raw['branchId']]['actions'])
        artifacts[ARTIFACT] = raw
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
                       and cached[3] is entry.artifacts and command != 'selectScene' and not verify_branch)
    admitted_native = (destination is not None and cached and cached[0] == saved_hash
                       and command != 'selectScene' and not verify_branch)
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
        token, shown_timeline, timeline_token = _frame_admission(session, frames, engine,
            record['branches'][branch_id], record.get('timeline', {}).get(branch_id), replay_cache,
            identity, branch_id, request.check)
        displayed, _, playback = frames.seek(engine, token.snapshot, args.get('position'),
            args.get('frameIndex'), replay_cache, check=request.check,
            verified_history=token.history, admission=token, display_only=True)
        playback['branchId'] = branch_id
        # The frame cache just verified this complete state identity. Reuse it
        # for display instead of hashing the large effect state a second time.
        compact = args.get('timelineToken') == timeline_token
        known_fields = args.get('viewFieldTokens', {})
        if not isinstance(known_fields, dict) or set(known_fields) - {'flapComponents'} or any(
                not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value)
                for value in known_fields.values()):
            raise ValueError('Invalid recorded view field acknowledgement')
        field_tokens = playback.get('fieldTokens')
        if field_tokens is None:
            # Older observed-frame adapters can expose the same authenticated
            # descriptor without changing the frozen numerical implementation.
            item = frames.store.load(token, args.get('position'))
            frame = item['frames'][args['frameIndex']] if item is not None else {}
            field_tokens = {name: blob['sha256'] for name, blob in
                frame.get('fields', {}).get('fields', {}).items() if name == 'flapComponents'} if (
                frame.get('stateId') == playback['frameStateId']) else {}
        omitted = [name for name, digest in field_tokens.items()
                   if compact and known_fields.get(name) == digest and name in displayed.state]
        # Current engines avoid detaching an acknowledged heavy value at all.
        # Older replay adapters still produce a full view and discard it below.
        view_options = {'_omit_fields': omitted} if getattr(displayed, '_view_omit_fields', None) else {}
        shown_view = displayed.view(_state_id=playback['frameStateId'], include_history=False, **view_options)
        if not compact:
            shown_view['history'] = shown_timeline[:len(displayed.actions)]
        else:
            # Resident source/catalog/guard descriptors share this immutable
            # admission generation. Effects and numerical ledgers stay fresh.
            for name in ('scene', 'catalog', 'protection'):
                shown_view.pop(name, None)
            for name in omitted:
                shown_view.pop(name, None)
        result = {'caseHash': identity, 'view': shown_view, 'outcome': None, 'changed': False,
            **({} if compact else {'timeline': shown_timeline}),
            'timelineToken': timeline_token, 'timelineOmitted': compact,
            'viewFieldTokens': field_tokens, 'viewFieldsOmitted': omitted,
            'playback': playback, 'branchId': branch_id, 'position': record['position'],
            **({} if compact else {'fullCase': copy.deepcopy(record.get('fullCase', {}).get(branch_id))}),
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
    elif command == 'plan' and args.get('opponent') == 'anatomy-staged-strategy':
        from .surgical_staged_planning import (
            compare_staged_strategies, install_staged_comparison,
            require_staged_development)
        from .surgical_planning import validate_saved_comparison
        from .surgical_sector_planning import validate_saved_sector_comparison
        if entry is None:
            raise ValueError('Save and prepare a declared DEVELOPMENT source before staged execution')
        role = require_staged_development(entry.case)
        access, sectors = record.get('planning'), record.get('sectorPlanning')
        if (not isinstance(access, dict) or not isinstance(sectors, dict) or
                sectors.get('sourceAccessComparisonSeal') != access.get('comparisonSeal')):
            raise ValueError('Staged work requires a paired source access/sector comparison')
        if (record['position'] != 0 or engine.actions or
                record['branches'][record['branchId']]['actions'] or
                engine.state_id != access.get('startStateId') or 'stagedPlanning' in record):
            raise ValueError('Open an original common intact source for staged work')
        validate_saved_comparison(engine.scene, record, access)
        validate_saved_sector_comparison(engine.scene, record, sectors)
        staged = compare_staged_strategies(engine, record,
            acknowledge_unknowns=args.get('acknowledgeResearchUnknowns', False),
            initial_access_radius_mm=args.get('initialAccessRadiusMm'),
            check=request.check, progress=progress)
        if staged['report'].get('interruption'):
            raise ValueError('Staged comparison interrupted: '+staged['report']['interruption'])
        install_staged_comparison(record, staged, engine.scene)
        for candidate, row in staged['branches']:
            replay_cache.seek(candidate, record['branches'][row['branchId']],
                              len(candidate.actions), check=request.check)
        outcome = {'status': 'accepted', 'reason': 'staged_comparison_recorded',
                   'completedFraction': 1., 'events': []}
    elif command == 'plan' and args.get('opponent') == 'anatomy-joint-strategy':
        from .surgical_joint_planning import (
            compare_joint_strategies, install_joint_comparison, validate_joint_report)
        from .surgical_planning import validate_saved_comparison
        from .surgical_sector_planning import validate_saved_sector_comparison
        if 'jointPlanning' in record:
            raise ValueError('Open an original classic source before generating another joint comparison')
        access, sectors = record.get('planning'), record.get('sectorPlanning')
        if (not isinstance(access, dict) or not isinstance(sectors, dict) or
                access.get('task') != 'anatomy-access' or
                sectors.get('task') != 'anatomy-strategy' or
                sectors.get('sourceAccessComparisonSeal') != access.get('comparisonSeal')):
            raise ValueError('Generate paired classic access and sector comparisons before joint planning')
        if (record['position'] != 0 or record['branches'][record['branchId']]['actions'] or
                engine.actions or engine.state_id != access['startStateId']):
            raise ValueError('Return to the common intact source branch before joint planning')
        if args.get('allowModeledBrainCorridor') is not True:
            raise ValueError('Explicit mapped-brain corridor opt-in is required')
        validate_saved_comparison(engine.scene, record, access)
        validate_saved_sector_comparison(engine.scene, record, sectors)
        joint = compare_joint_strategies(engine,
            acknowledge_unknowns=args.get('acknowledgeResearchUnknowns', False),
            max_native_actions=args.get('maxActions', 48),
            check=request.check, progress=progress)
        created = install_joint_comparison(record, joint,
            joint['report']['accessComparison'], joint['report']['sectorComparison'])
        record['jointPlanning'] = copy.deepcopy(joint['report'])
        validate_joint_report(engine.scene, record, record['jointPlanning'])
        for branch_id, item in created.items():
            candidate = item['candidate']
            # Live endpoint admission is read-only cache priming. It performs
            # no additional physics for these just-verified native endpoints.
            replay_cache.seek(candidate, record['branches'][branch_id], len(candidate.actions),
                              check=request.check)
            record.setdefault('timeline', {})[branch_id] = copy.deepcopy(candidate.history)
        outcome = {'status': 'accepted', 'reason': 'joint_comparison_recorded',
                   'completedFraction': 1., 'events': []}
    elif command == 'plan':
        sector_planning = args.get('opponent') == 'anatomy-strategy'
        if sector_planning:
            from .surgical_sector_planning import compare_sector_strategies
            source_access = record.get('planning')
            if not isinstance(source_access, dict) or source_access.get('task') != 'anatomy-access':
                raise ValueError('Generate source-bound access approaches before comparing sector strategies')
            branches, comparison = compare_sector_strategies(engine,
                source_access, record['branches'],
                acknowledge_unknowns=args.get('acknowledgeResearchUnknowns', False),
                max_actions_per_strategy=args.get('maxActions', 48),
                max_total_non_target_mm3=args.get('maxHealthyRemovalMm3', 200.),
                max_wall_seconds=args.get('maxWallSeconds', 120.),
                check=request.check, progress=progress)
        elif args.get('opponent') == 'anatomy-access':
            from .surgical_access_planning import compare_access_approaches
            branches, comparison = compare_access_approaches(engine,
                acknowledge_unknowns=args.get('acknowledgeResearchUnknowns', False),
                allow_modeled_brain_corridor=args.get('allowModeledBrainCorridor', False),
                max_actions_per_approach=args.get('maxActions', 48),
                max_healthy_removed_mm3=args.get('maxHealthyRemovalMm3', 200.),
                max_wall_seconds=args.get('maxWallSeconds', 120.),
                check=request.check, progress=progress)
        elif args.get('opponent') == 'vascular-rescue':
            from .surgical_strategy_planning import compare_rescue
            branches, comparison = compare_rescue(engine, record,
                acknowledge_unknowns=args.get('acknowledgeResearchUnknowns', False),
                check=request.check, progress=progress)
        else:
            from .surgical_planning import compare_continuations
            branches, comparison = compare_continuations(engine,
                acknowledge_unknowns=args.get('acknowledgeResearchUnknowns', False),
                opponent=args.get('opponent', 'stop'), check=request.check, progress=progress)
        # Install only after every complete continuation replay check passes.
        # The original live branch and cursor remain unchanged.
        for candidate, report in branches:
            branch_id = 'plan-' + str(len(record['branches'])) + '-' + report['method']
            report['branchId'] = branch_id
            record['branches'][branch_id] = candidate.snapshot()
            # These live endpoints and their independent continuation replays
            # just passed above. Keep them in the existing bounded session
            # cache so selecting an alternative can immediately show it.
            replay_cache.seek(candidate, record['branches'][branch_id], len(candidate.actions), check=request.check)
            record.setdefault('timeline', {})[branch_id] = copy.deepcopy(candidate.history)
            record.setdefault('parents', {})[branch_id] = {'branchId': record['branchId'], 'position': record['position']}
        comparison['comparisonSeal'] = semantic_digest(comparison)
        if record.get('sectorPlanning') is not None:
            if sector_planning:
                history = record.setdefault('sectorPlanningHistory', [])
                if len(history) >= 8:
                    raise ValueError('Save this workspace and open a new source before adding more sector comparisons')
                history.append({'comparison': copy.deepcopy(record['sectorPlanning']),
                    'sourceAccessComparison': copy.deepcopy(record.get('sectorSourceAccessComparison', record['planning']))})
            elif 'sectorSourceAccessComparison' not in record:
                record['sectorSourceAccessComparison'] = copy.deepcopy(record['planning'])
        if sector_planning:
            record['sectorSourceAccessComparison'] = copy.deepcopy(source_access)
        record['sectorPlanning' if sector_planning else 'planning'] = comparison
    elif command == 'step':
        action = args['action']
        unchanged_tool = action.get('operation') == 'exchange' and engine.state['slots'].get(action.get('slot'), {}).get('instrument') == action.get('instrument')
        outcome = {'status': 'accepted', 'reason': 'Instrument already selected; state unchanged.', 'completedFraction': 1., 'events': []} if unchanged_tool else engine.step(action, check=request.check)
        if outcome['status'] != 'rejected' and not unchanged_tool:
            full = record['branches'][record['branchId']]
            planned_ids = {row['branchId'] for key in ('planning', 'sectorPlanning')
                           for row in (record.get(key) or {}).get('alternatives', [])}
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
    from .surgical_strategy_planning import rescue_availability
    result['planningAvailability'].update(rescue_availability(engine, record))
    from .surgical_access_planning import access_availability
    result['planningAvailability'].update(access_availability(engine))
    comparison = copy.deepcopy(record.get('planning'))
    if comparison is not None:
        from .surgical_planning import validate_saved_comparison
        validate_saved_comparison(engine.scene, record, comparison)
    result['planning'] = comparison
    sectors = copy.deepcopy(record.get('sectorPlanning'))
    if sectors is not None:
        from .surgical_sector_planning import validate_saved_sector_comparison
        sector_record = {**record, 'planning': record.get('sectorSourceAccessComparison', record.get('planning'))}
        validate_saved_sector_comparison(engine.scene, sector_record, sectors)
    result['sectorPlanning'] = sectors
    retraction_study = record.get('retractionStudy')
    if isinstance(retraction_study, dict) and retraction_study.get('schema') == 'resectionlab.retractor-native-study.v1':
        from .surgical_retraction_study import validate_retraction_study
        validate_retraction_study(engine.scene, record, retraction_study)
        result['retractionStudy'] = copy.deepcopy(retraction_study)
    staged_report = record.get('stagedPlanning')
    if staged_report is not None:
        from .surgical_staged_planning import validate_staged_report, staged_summary
        validate_staged_report(engine.scene, record, staged_report)
        result['stagedPlanning'] = staged_summary(staged_report)
    joint_report = record.get('jointPlanning')
    if joint_report is not None:
        from .surgical_joint_planning import validate_joint_report
        validate_joint_report(engine.scene, record, joint_report)
        result['jointPlanning'] = copy.deepcopy(joint_report)
        result['jointBaselineHistory'] = copy.deepcopy(record.get('jointBaselineHistory',
            record.get('sectorPlanningHistory', [])[-1:]))
    _require_json_budget(result, RESULT_BYTES, 'SURGICAL_RESULT_LIMIT', 'Save this sequence before continuing')
    request.begin_commit()
    if command == 'create' and raw is not None:
        artifacts.setdefault('surgical_prior_scenes', []).append(raw)
        result['previousScenes'] = [{'index': i, 'label': f"Environment {i + 1}", 'branches': len(s.get('branches', {}))} for i, s in enumerate(artifacts['surgical_prior_scenes'])]
    if record.get('fullCase'):
        # The timeline is derived from the action log by verified replay. Long
        # episodes must not serialize its redundant metrics/events in every
        # native save. Exclude it BEFORE detaching: copying thousands of unused
        # outcomes only to discard them repeats the original ancestry cost.
        persisted_record = copy.deepcopy({name: value for name, value in record.items() if name != 'timeline'})
    else:
        persisted_record = copy.deepcopy(record)
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
