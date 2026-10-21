"""Live-console runner adjustment only; engine and director code stay loaded."""
_runner_before = engine.snapshot()
_runner_director_before = json.loads(json.dumps(director))
_runner_saved = load_workspace(destination)[1]['surgical_toolbox']
assert director['lastStateId'] == engine.state_id
_saved_actions = _runner_saved['branches'][branch_id]['actions']
assert _saved_actions == engine.actions[:len(_saved_actions)]
if _runner_saved['branches'][branch_id] != _runner_before:
    assert len(_saved_actions) < len(engine.actions)
    persist(case, artifacts, workspace, engine, director, destination, branch_id, parent)
    _runner_saved = load_workspace(destination)[1]['surgical_toolbox']
assert _runner_saved['branches'][branch_id] == _runner_before
assert _runner_saved['fullCase'][branch_id] == director
_runner_methods = {name: getattr(type(engine), name).__code__ for name in
                   ('step', 'preview', '_execute', '_substep', '_physical_substep', 'cells', 'occupancy')}
_runner_director_code = rehearsal.run_slice.__code__


def advance(slices=4000, *, slice_actions=12):
    """Commit bounded ordinary director slices and persist every complete slice."""
    global engine, director, live_slice
    if type(slice_actions) is not int or not 1 <= slice_actions <= 64:
        raise ValueError('Headless slice requires 1..64 public actions')
    for _ in range(slices):
        before = (len(engine.actions), director['stage'], director['accessIndex'])
        engine, director = rehearsal.run_slice(engine, director, max_actions=slice_actions)
        row = {'slice': live_slice, 'executionMode': 'live-recovery',
            'runnerSliceActionLimit': slice_actions, 'position': len(engine.actions),
            'stateId': engine.state_id, 'stage': director['stage'],
            'phase': director['currentPhaseId'], 'status': director['status'],
            'metrics': engine.metrics(), 'wallSeconds': time.perf_counter() - started,
            'lastOutcome': director.get('lastOutcome')}
        with (destination.parent / 'execution.jsonl').open('a') as stream:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
        persist(case, artifacts, workspace, engine, director, destination, branch_id, parent)
        live_slice += 1
        print(json.dumps({key: row[key] for key in ('position', 'stage', 'phase', 'status')} |
            {'targetRemovedMm3': row['metrics']['targetRemovedMm3']}), flush=True)
        if director['status'] in ('complete', 'stopped'):
            receipt()
            print('Saved stop:', director.get('stopReason'), flush=True)
            return
        if before == (len(engine.actions), director['stage'], director['accessIndex']):
            raise ValueError('Director made no physical or scheduling progress')
        if (destination.parent / 'pause-after-batch').exists():
            receipt()
            print('Saved boundary pause requested; live engine retained.', flush=True)
            return
    receipt()


assert engine.snapshot() == _runner_before and director == _runner_director_before
assert all(getattr(type(engine), name).__code__ is code for name, code in _runner_methods.items())
assert rehearsal.run_slice.__code__ is _runner_director_code
assert hashlib.sha256(original.read_bytes()).hexdigest() == source_sha
assert array_digest(scene.labels) == label_hash and array_digest(case.mri) == image_hash
_runner_proof = {'utc': datetime.now(timezone.utc).isoformat(),
    'position': len(engine.actions), 'stateId': engine.state_id,
    'savedBoundaryExactlyEqualsLive': True, 'engineAndDirectorCodeUnchanged': True,
    'sourceUnchanged': True, 'previousSaveIntervalActionLimit': 2,
    'newSaveIntervalActionLimit': 12,
    'scope': 'Ignored headless runner save interval only; each action still uses unchanged public preview/step checks. No model, action, or source modification.'}
(destination.parent / 'bounded-headless-batches-proof.json').write_text(json.dumps(_runner_proof, indent=2) + '\n')
print(json.dumps(_runner_proof), flush=True)
