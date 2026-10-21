"""Profile one real saved interval, then continue the same patient engine.

Execute only at the existing live console's actual prompt after the owned
pause-after-batch request. No engine, director or physics source is reloaded.
"""
def _profile_next_live_slice():
    import cProfile
    import copy
    import subprocess

    pause = destination.parent / 'pause-after-batch'
    request = json.loads(pause.read_text())
    assert request['purpose'] == 'profile-one-live-slice-20261020'
    assert director['status'] == 'running' and director['lastStateId'] == engine.state_id
    before = engine.snapshot()
    saved = load_workspace(destination)[1]['surgical_toolbox']
    assert saved['branches'][branch_id] == before
    assert saved['fullCase'][branch_id] == director
    assert hashlib.sha256(original.read_bytes()).hexdigest() == source_sha
    assert array_digest(scene.labels) == label_hash and array_digest(case.mri) == image_hash
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    assert commit == '3e0bd07c7a5ad7b84a9cf921c20f1c2564cd2c69'
    method_codes = {name: getattr(type(engine), name).__code__ for name in
        ('step', 'preview', '_execute', '_substep', '_physical_substep', 'cells', 'occupancy')}
    director_code = rehearsal.run_slice.__code__
    before_director = copy.deepcopy(director)
    count = len(engine.actions)
    output = destination.parent / ('profile-live-slice-from-' + str(count) + '.pstats')
    proof_path = output.with_suffix('.json')
    assert not output.exists() and not proof_path.exists()
    pause.unlink()
    profiler = cProfile.Profile()
    began = time.perf_counter()
    try:
        profiler.enable()
        advance(1, slice_actions=2)
    finally:
        profiler.disable()
        profiler.dump_stats(output)
    duration = time.perf_counter() - began
    after = engine.snapshot()
    assert engine.actions[:count] == before['actions']
    assert director['lastStateId'] == engine.state_id
    saved = load_workspace(destination)[1]['surgical_toolbox']
    assert saved['branches'][branch_id] == after and saved['fullCase'][branch_id] == director
    assert all(getattr(type(engine), name).__code__ is code for name, code in method_codes.items())
    assert rehearsal.run_slice.__code__ is director_code
    assert hashlib.sha256(original.read_bytes()).hexdigest() == source_sha
    assert array_digest(scene.labels) == label_hash and array_digest(case.mri) == image_hash
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip() == commit
    value = {'schema': 'resectionlab.live-interval-profile.v1',
        'utc': datetime.now(timezone.utc).isoformat(), 'checkedOutCommit': commit,
        'startPosition': count, 'endPosition': len(engine.actions),
        'startStateId': before['stateId'], 'endStateId': engine.state_id,
        'startStage': before_director['stage'], 'endStage': director['stage'],
        'status': director['status'], 'pstats': str(output),
        'instrumentedWallSeconds': duration, 'patientSourceUnchanged': True,
        'profileActionLimit': 2,
        'savedBoundaryEqualsLive': True, 'engineAndDirectorCodeUnchanged': True,
        'actionsPreserveExactPriorPrefix': True,
        'scope': 'At most two new actual public actions, not a replay or variant. Only profiling and save grouping change; existing engine/director code remains loaded. Profiling overhead is included; this is not a speed benchmark.'}
    proof_path.write_text(json.dumps(value, indent=2) + '\n')
    print(json.dumps({'profiledActualSavedInterval': value}), flush=True)


_profile_next_live_slice()
if director['status'] == 'running':
    print('Uninstrumented live continuation resumed.', flush=True)
    advance(4000)
