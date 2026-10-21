"""Configure one existing bounded left trial at the real deferred endpoint."""
def _start_reviewed_left_trial():
    global director
    import copy
    import os
    import subprocess
    from collections import Counter

    commit = 'de837ad687423bbe7ef852c7ca926bc93c2c50df'
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip() == commit
    for relative in ('src/resectionlab/full_case_rehearsal.py', 'src/resectionlab/full_case_directional.py'):
        assert (ROOT / relative).read_bytes() == subprocess.check_output(['git', 'show', commit + ':' + relative], cwd=ROOT)
    assert director['status'] == 'stopped' and director['stage'] == 'endpoint'
    assert director['accessPassDeferral']['status'] == 'deferred'
    assert director['deferredAccessPass']['status'] == 'pending'
    assert director['lastStateId'] == engine.state_id
    assert not engine.state['holds']
    assert all(not slot['inserted'] and not slot['active'] and not slot['grip'] for slot in engine.state['slots'].values())
    assert engine.metrics()['activeVesselWounds'] == 0
    baseline_wounds = engine.metrics().get('vesselPhysicsV2', {}).get('woundCount', 0)
    count = len(engine.actions)
    before, previous = engine.snapshot(), copy.deepcopy(director)
    history = json.dumps(engine.history, separators=(',', ':'), allow_nan=False)
    saved = load_workspace(destination)[1]['surgical_toolbox']
    assert saved['branches'][branch_id] == before and saved['fullCase'][branch_id] == previous
    assert hashlib.sha256(original.read_bytes()).hexdigest() == source_sha
    assert array_digest(scene.labels) == label_hash and array_digest(case.mri) == image_hash
    methods = {name: getattr(type(engine), name).__code__ for name in
        ('step', 'preview', '_execute', '_substep', '_physical_substep', 'cells', 'occupancy')}
    proposed = rehearsal.configure_directional_fallback(engine, director, axis=0, limit=8)
    assert engine.snapshot() == before and director == previous
    assert json.dumps(engine.history, separators=(',', ':'), allow_nan=False) == history
    assert all(getattr(type(engine), name).__code__ is code for name, code in methods.items())
    assert proposed['accessAxis'] == 0 and proposed['stage'] == 'directional_transition'
    assert 0 < len(proposed['accesses']) <= 8
    assert all(access['nativeAxis'] == 0 and 0 < access['drillRadiusMm'] <= 3. for access in proposed['accesses'])
    assert proposed['deferredAccessPass']['pass'] == previous['deferredAccessPass']['pass']
    assert proposed.get('completedAccessPasses', []) == previous.get('completedAccessPasses', [])
    checkpoint = destination.parent / 'checkpoints' / ('primary-before-bounded-left-trial-' + str(count) + '.ressectionlab')
    assert not checkpoint.exists()
    os.link(destination, checkpoint)
    proof_path = destination.parent / 'bounded-left-trial-execution-proof.json'
    assert not proof_path.exists()
    director = proposed
    persist(case, artifacts, workspace, engine, director, destination, branch_id, parent)
    saved = load_workspace(destination)[1]['surgical_toolbox']
    assert saved['branches'][branch_id] == before and saved['fullCase'][branch_id] == director
    assert len(engine.actions) == count and engine.snapshot() == before
    assert json.dumps(engine.history, separators=(',', ':'), allow_nan=False) == history
    assert all(getattr(type(engine), name).__code__ is code for name, code in methods.items())
    assert hashlib.sha256(original.read_bytes()).hexdigest() == source_sha
    assert array_digest(scene.labels) == label_hash and array_digest(case.mri) == image_hash
    proof = {'utc': datetime.now(timezone.utc).isoformat(), 'codeCommit': commit,
        'startPosition': count, 'startStateId': engine.state_id, 'nativeAxis': 0, 'candidateLimit': 8,
        'sourceUnchanged': True, 'savedBoundaryEqualsLive': True, 'engineAndHistoryUnchangedDuringConfiguration': True,
        'engineMethodCodesUnchanged': list(methods), 'deferredSuperiorPassRetainedExactly': True,
        'deferredSuperiorCursor': previous['deferredAccessPass']['pass']['accessIndex'],
        'candidateIds': [row['id'] for row in director['accesses']],
        'checkpoint': str(checkpoint), 'checkpointSha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        'scope': 'Metadata-only setup of existing source-column proposals after real public cleanup. No left reachability or target-removal claim before actual preview/step.'}
    proof_path.write_text(json.dumps(proof, indent=2) + '\n')
    receipt()
    print(json.dumps({'boundedLeftTrialPrepared': proof}), flush=True)

    advance(1, slice_actions=12)
    after = engine.snapshot()
    saved = load_workspace(destination)[1]['surgical_toolbox']
    assert saved['branches'][branch_id] == after and saved['fullCase'][branch_id] == director
    assert engine.actions[:count] == before['actions'] and director['lastStateId'] == engine.state_id
    assert len(engine.history[count:]) == len(engine.actions) - count
    prior_state = before['stateId']
    for row in engine.history[count:]:
        assert row['stateBefore'] == prior_state and row['status'] in ('accepted', 'partial')
        prior_state = row['stateAfter']
    assert prior_state == engine.state_id
    assert all(getattr(type(engine), name).__code__ is code for name, code in methods.items())
    assert hashlib.sha256(original.read_bytes()).hexdigest() == source_sha
    assert array_digest(scene.labels) == label_hash and array_digest(case.mri) == image_hash
    metrics = engine.metrics()
    proof['firstAdvance'] = {'utc': datetime.now(timezone.utc).isoformat(), 'endPosition': len(engine.actions),
        'newActionCount': len(engine.actions) - count, 'endStateId': engine.state_id,
        'statuses': dict(Counter(row['status'] for row in engine.history[count:])),
        'directorStatus': director['status'], 'stopReason': director.get('stopReason'),
        'targetRemovedMm3': metrics['targetRemovedMm3'], 'activeVesselWounds': metrics['activeVesselWounds'],
        'woundCount': metrics.get('vesselPhysicsV2', {}).get('woundCount', 0),
        'savedSnapshotEqualsLive': True, 'continuousRecordedStateChain': True, 'sourceUnchanged': True,
        'qualification': 'New actual public interval, or an honestly recorded zero-action stop. No duplicated patient replay.'}
    proof_path.write_text(json.dumps(proof, indent=2) + '\n')
    receipt()
    print(json.dumps({'firstBoundedLeftAdvanceVerified': proof['firstAdvance']}), flush=True)
    return (director['status'] == 'running' and metrics['activeVesselWounds'] == 0
        and metrics.get('vesselPhysicsV2', {}).get('woundCount', 0) == baseline_wounds), baseline_wounds


_left_trial_ready, _left_trial_baseline_wounds = _start_reviewed_left_trial()
if _left_trial_ready:
    for _left_trial_batch in range(4000):
        _left_trial_metrics = engine.metrics()
        if director['status'] != 'running':
            break
        if (destination.parent / 'pause-after-batch').exists():
            print('Saved boundary pause retained; left trial engine remains live.', flush=True)
            break
        if (_left_trial_metrics['activeVesselWounds'] != 0
                or _left_trial_metrics.get('vesselPhysicsV2', {}).get('woundCount', 0) != _left_trial_baseline_wounds):
            receipt()
            print('New modeled wound retained at the saved boundary for source-control review.', flush=True)
            break
        advance(1, slice_actions=12)
else:
    print('Left trial remains at its actual saved boundary for review.', flush=True)
