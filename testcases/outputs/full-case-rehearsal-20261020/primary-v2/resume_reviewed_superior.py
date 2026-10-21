"""Resume the exact deferred superior pass at a real withdrawn left endpoint.

Execute once in the existing saved console, after root review. This script does
not reload modules, restore a patient, repeat a trial, or delete pause requests.
An interrupted metadata save must be inspected at the prompt, not retried here.
"""
def _resume_reviewed_superior():
    global director
    import copy
    import os
    import subprocess
    import types
    from collections import Counter

    commit = '7cbecb11a11a925bdb25065b414546532cb61da8'
    pass_hash = 'sha256:24b100f8c29b22e98a2c4dae1283dfbc8ba8d822f9fcc6bdf219a9330e8f1e29'
    archive_hash = 'sha256:8f32afabfc98cef09b0af6bd368e9e8080bbede57d48f85347c7d8f99e09e0af'
    # These archive pins come from the preserved pre-left-trial 1644 native.
    pause = destination.parent / 'pause-after-batch'
    proof_path = destination.parent / 'superior-pass-resume-execution-proof.json'
    assert not pause.exists(), 'Resolve the existing pause request explicitly before configuration'
    assert not proof_path.exists(), 'Resume proof already exists; inspect instead of retrying'
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip() == commit
    subprocess.run(['git', 'diff', '--quiet', commit, '--', 'src/resectionlab'], cwd=ROOT, check=True)
    assert not subprocess.check_output(['git', 'ls-files', '--others', '--exclude-standard',
        '--', 'src/resectionlab'], cwd=ROOT, text=True).strip(), 'Unpinned production source exists'

    def normalized(code):
        return (code.co_code, tuple(normalized(v) if isinstance(v, types.CodeType) else v for v in code.co_consts),
            code.co_names, code.co_varnames, code.co_freevars, code.co_cellvars, code.co_argcount,
            code.co_posonlyargcount, code.co_kwonlyargcount, code.co_nlocals, code.co_stacksize,
            code.co_flags, code.co_exceptiontable, code.co_name, code.co_qualname)

    for short_name in ('full_case_rehearsal', 'full_case_geometry'):
        module = importlib.import_module('resectionlab.' + short_name)
        relative = 'src/resectionlab/' + short_name + '.py'
        source = subprocess.check_output(['git', 'show', commit + ':' + relative], cwd=ROOT)
        assert (ROOT / relative).read_bytes() == source
        compiled = compile(source, str(ROOT / relative), 'exec')
        expected = {c.co_name: normalized(c) for c in compiled.co_consts if isinstance(c, types.CodeType)}
        actual = {name: normalized(fn.__code__) for name, fn in vars(module).items()
            if isinstance(fn, types.FunctionType) and fn.__module__ == module.__name__}
        assert actual == expected, 'Loaded proposal functions differ from pinned source: ' + short_name
    assert rehearsal is importlib.import_module('resectionlab.full_case_rehearsal')

    def encoded(value):
        return json.dumps(value, separators=(',', ':'), allow_nan=False)

    def source_check():
        assert engine.scene is scene
        assert scene.identity == 'sha256:7a5d368382fe2a66192bdef88ff6af3a55cd533c9ba3787c68a1d1e3cb48f856'
        assert hashlib.sha256(original.read_bytes()).hexdigest() == source_sha == 'ff5a49b8706a8ccda08d8c3e82dad3e079f85dea98cb6bd40783d2262eec9838'
        assert array_digest(scene.labels) == label_hash and array_digest(case.mri) == image_hash

    assert director['scenario'] == 'primary' and director['status'] == 'stopped' and director['stage'] == 'endpoint'
    assert director['accessAxis'] == 0 and director['lastStateId'] == engine.state_id
    assert not engine.state['holds']
    assert all(not slot['inserted'] and not slot['active'] and not slot['grip'] for slot in engine.state['slots'].values())
    assert engine.guard_report()['mode'] == 'strict-research'
    pending = director['deferredAccessPass']; frozen = pending['pass']
    assert pending['status'] == 'pending' and pending['passHash'] == pass_hash and pending['archiveHash'] == archive_hash
    assert frozen['accessAxis'] == 2 and frozen['accessIndex'] == 22 and len(frozen['accesses']) == 99
    assert all(access.get('nativeAxis', 2) == 2 and access['id'].startswith('superior-') for access in frozen['accesses'])
    assert len({access['id'] for access in frozen['accesses']}) == 99
    assert pending['trials'] and pending['trials'][-1]['axis'] == 0
    metrics_before = engine.metrics()
    assert metrics_before['activeVesselWounds'] == 0 and metrics_before['vesselPhysicsV2']['woundCount'] == 0
    baseline_wounds = metrics_before['vesselPhysicsV2']['woundCount']
    source_check()
    before, previous = engine.snapshot(), copy.deepcopy(director)
    history_before = encoded(engine.history); count = len(engine.actions)
    saved = load_workspace(destination)[1]['surgical_toolbox']
    assert saved['branches'][branch_id] == before and saved['fullCase'][branch_id] == previous
    checkpoint = destination.parent / 'checkpoints' / ('primary-before-superior-resume-' + str(count) + '.ressectionlab')
    assert not checkpoint.exists(), 'Checkpoint exists; inspect the prior attempt instead of retrying'
    native_before_sha = hashlib.sha256(destination.read_bytes()).hexdigest()
    methods = {name: getattr(type(engine), name).__code__ for name in
        ('step', 'preview', '_execute', '_substep', '_physical_substep', 'cells', 'occupancy')}
    thermal = importlib.import_module('resectionlab.surgical_drilling_v2')
    thermal_codes = {name: fn.__code__ for name, fn in vars(thermal).items()
        if isinstance(fn, types.FunctionType) and fn.__module__ == thermal.__name__}
    director_code = rehearsal.run_slice.__code__

    def code_check():
        assert all(getattr(type(engine), name).__code__ is code for name, code in methods.items())
        assert all(getattr(thermal, name).__code__ is code for name, code in thermal_codes.items())
        assert rehearsal.run_slice.__code__ is director_code

    # This reviewed API validates the sealed archive and real request/defer/
    # stop/trial prefix boundaries, then returns detached scheduling metadata.
    proposed = rehearsal.resume_deferred_access_pass(engine, director)
    assert engine.snapshot() == before and director == previous and encoded(engine.history) == history_before
    assert proposed['status'] == 'ready' and proposed['stage'] == 'directional_transition'
    assert proposed['accessAxis'] == 2 and proposed['accessIndex'] == 22 and len(proposed['accesses']) == 99
    assert encoded(proposed['accesses'][:22]) == encoded(frozen['accesses'][:22])
    old_tail = {row['id']: row for row in frozen['accesses'][22:]}
    tail = proposed['accesses'][22:]
    assert {row['id'] for row in tail} == set(old_tail) and len(tail) == len(old_tail)
    rank_fields = {'remainingGeometricCapacityCells', 'capacityStateId'}
    for row in tail:
        original_row = old_tail[row['id']]
        assert {k: v for k, v in row.items() if k not in rank_fields} == {k: v for k, v in original_row.items() if k not in rank_fields}
        assert row['capacityStateId'] == engine.state_id
        assert type(row['remainingGeometricCapacityCells']) is int and row['remainingGeometricCapacityCells'] >= 0
    order = {row['id']: i for i, row in enumerate(frozen['accesses'][22:])}
    assert [row['id'] for row in tail] == [row['id'] for row in sorted(tail,
        key=lambda row: (-row['remainingGeometricCapacityCells'], order[row['id']]))]
    for key in rehearsal._ACCESS_PASS_FIELDS:
        if key not in ('accesses', 'passStartTargetMm3'):
            assert (key in proposed) == (key in frozen)
            if key in frozen: assert encoded(proposed[key]) == encoded(frozen[key])
    target_now = metrics_before['targetRemovedMm3']
    expected_baseline = frozen['passStartTargetMm3'] + (target_now - pending['targetAtDeferMm3'])
    assert target_now >= pending['targetAtDeferMm3'] and proposed['passStartTargetMm3'] == expected_baseline
    assert target_now - expected_baseline == pending['targetAtDeferMm3'] - frozen['passStartTargetMm3']
    resumed_pending = proposed['deferredAccessPass']
    assert resumed_pending['status'] == proposed['accessPassDeferral']['status'] == 'resumed'
    assert encoded(resumed_pending['pass']) == encoded(frozen)
    for key in pending:
        if key != 'status': assert encoded(resumed_pending[key]) == encoded(pending[key])
    assert resumed_pending['resumedBoundary']['position'] == count and resumed_pending['resumedBoundary']['stateId'] == engine.state_id
    assert resumed_pending['restoredPassStartTargetMm3'] == expected_baseline
    assert proposed['interveningAccessTrials'][:-1] == previous.get('interveningAccessTrials', [])
    trial = proposed['interveningAccessTrials'][-1]
    for key in rehearsal._ACCESS_PASS_FIELDS:
        if key in previous: assert encoded(trial[key]) == encoded(previous[key])
    assert trial['stoppedStage'] == 'endpoint' and trial['stopReason'] == previous.get('stopReason')
    assert trial['directionalTransition'] == previous.get('directionalTransition')
    assert trial['endBoundary']['position'] == count and trial['endBoundary']['stateId'] == engine.state_id
    allowed = set(rehearsal._ACCESS_PASS_FIELDS) | {'status', 'stopReason', 'stage', 'priorStops',
        'interveningAccessTrials', 'fieldCleanup', 'priorFieldCleanups', 'fieldCleanupAttempted',
        'directionalTransition', 'deferredAccessPass', 'accessPassDeferral'}
    for key in previous:
        if key not in allowed: assert encoded(proposed[key]) == encoded(previous[key])
    if 'fieldCleanup' in previous:
        assert proposed['priorFieldCleanups'][-1] == previous['fieldCleanup']
        assert proposed['priorFieldCleanups'][:-1] == previous.get('priorFieldCleanups', [])
    assert proposed['directionalTransition']['poses'] and proposed['directionalTransition']['index'] == 0
    code_check(); source_check()
    os.link(destination, checkpoint)
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == native_before_sha
    director = proposed
    persist(case, artifacts, workspace, engine, director, destination, branch_id, parent)
    saved = load_workspace(destination)[1]['surgical_toolbox']
    assert saved['branches'][branch_id] == before and saved['fullCase'][branch_id] == director
    assert len(engine.actions) == count and engine.snapshot() == before and encoded(engine.history) == history_before
    code_check(); source_check()
    proof = {'utc': datetime.now(timezone.utc).isoformat(), 'codeCommit': commit,
        'startPosition': count, 'startStateId': engine.state_id, 'sourceUnchanged': True,
        'savedBoundaryEqualsLive': True, 'metadataOnlyConfiguration': True,
        'patientHistoryAndCodeUnchangedDuringConfiguration': True, 'passHash': pass_hash,
        'archiveHash': archive_hash, 'superiorAccessCount': 99, 'restoredCursor': 22,
        'finishedPrefixUnchanged': True, 'freshStableRankingOnlyRemainingTail': True,
        'leftTrialActionsAndPassMetadataRetained': True, 'targetAtDeferMm3': pending['targetAtDeferMm3'],
        'checkedHistoricalBoundaryPositions': {key: pending[key]['position'] for key in
            ('requestBoundary', 'deferBoundary', 'stoppedBoundary')},
        'targetAtResumeMm3': target_now, 'originalPassBaselineMm3': frozen['passStartTargetMm3'],
        'restoredPassBaselineMm3': expected_baseline, 'baselineCorrectionAppliedOnce': True,
        'checkpoint': str(checkpoint), 'checkpointSha256': native_before_sha,
        'scope': 'Existing reviewed resume API only; no reload, patient replay, source/guard change, or inferred tool arrival.'}
    with proof_path.open('x') as stream: stream.write(json.dumps(proof, indent=2) + '\n')
    receipt()
    print(json.dumps({'superiorPassResumePrepared': proof}), flush=True)
    if pause.exists():
        proof['firstAdvance'] = {'called': False, 'newActionCount': 0, 'reason': 'Pause requested after metadata save; actual ready state retained.'}
        proof_path.write_text(json.dumps(proof, indent=2) + '\n')
        return False, baseline_wounds

    advance(1, slice_actions=12)
    saved = load_workspace(destination)[1]['surgical_toolbox']
    assert saved['branches'][branch_id] == engine.snapshot() and saved['fullCase'][branch_id] == director
    assert engine.actions[:count] == before['actions'] and encoded(engine.history[:count]) == history_before
    rows = engine.history[count:]
    assert len(rows) == len(engine.actions) - count and director['lastStateId'] == engine.state_id
    prior_state = before['stateId']
    for row in rows:
        assert row['stateBefore'] == prior_state and row['status'] in ('accepted', 'partial')
        prior_state = row['stateAfter']
    assert prior_state == engine.state_id
    code_check(); source_check()
    metrics = engine.metrics()
    proof['firstAdvance'] = {'called': True, 'endPosition': len(engine.actions),
        'newActionCount': len(rows), 'endStateId': engine.state_id,
        'statuses': dict(Counter(row['status'] for row in rows)), 'directorStatus': director['status'],
        'stage': director['stage'], 'stopReason': director.get('stopReason'),
        'activeWounds': metrics['activeVesselWounds'], 'woundCount': metrics['vesselPhysicsV2']['woundCount'],
        'targetRemovedMm3': metrics['targetRemovedMm3'], 'savedSnapshotEqualsLive': True,
        'continuousRecordedStateChain': True, 'sourceUnchanged': True,
        'qualification': 'One new ordinary public slice, or an explicit zero-action stop; no earlier interval repeated.'}
    proof_path.write_text(json.dumps(proof, indent=2) + '\n')
    receipt()
    print(json.dumps({'firstSuperiorResumeAdvanceVerified': proof['firstAdvance']}), flush=True)
    return director['status'] == 'running', baseline_wounds


_superior_resume_ready, _superior_resume_baseline_wounds = _resume_reviewed_superior()
if _superior_resume_ready:
    for _superior_resume_batch in range(4000):
        _superior_resume_metrics = engine.metrics()
        if director['status'] != 'running' or (destination.parent / 'pause-after-batch').exists():
            break
        if (_superior_resume_metrics['activeVesselWounds'] != 0
                or _superior_resume_metrics['vesselPhysicsV2']['woundCount'] != _superior_resume_baseline_wounds):
            receipt()
            print('Actual new wound retained at its saved boundary for source-control review.', flush=True)
            break
        advance(1, slice_actions=12)
else:
    print('Superior resume remains at its actual saved boundary for review.', flush=True)
