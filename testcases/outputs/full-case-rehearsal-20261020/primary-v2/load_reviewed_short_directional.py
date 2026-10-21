"""Load reviewed proposal-only code at an actual saved console prompt."""
def _load_reviewed_short_directional():
    import copy
    import os
    import subprocess
    import types
    from collections import Counter

    expected_commit = '7cbecb11a11a925bdb25065b414546532cb61da8'
    previous_commit = '520e3cda06f870de839ac063c4eda803190a5bf2'
    assert len(expected_commit) == 40
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip() == expected_commit
    relative = 'src/resectionlab/full_case_directional.py'
    source = subprocess.check_output(['git', 'show', expected_commit + ':' + relative], cwd=ROOT)
    assert (ROOT / relative).read_bytes() == source
    pause = destination.parent / 'pause-after-batch'
    pause_text = pause.read_text()
    assert pause_text.startswith('Root-owned request ') and 'short checked-directional proposal integration only.' in pause_text
    module = importlib.import_module('resectionlab.full_case_directional')

    def normalized(code):
        return (code.co_code, tuple(normalized(v) if isinstance(v, types.CodeType) else v for v in code.co_consts),
            code.co_names, code.co_varnames, code.co_freevars, code.co_cellvars, code.co_argcount,
            code.co_posonlyargcount, code.co_kwonlyargcount, code.co_nlocals, code.co_stacksize,
            code.co_flags, code.co_exceptiontable, code.co_name, code.co_qualname)

    def functions(compiled):
        return {v.co_name: normalized(v) for v in compiled.co_consts if isinstance(v, types.CodeType)}

    old_source = subprocess.check_output(['git', 'show', previous_commit + ':' + relative], cwd=ROOT)
    old_codes = functions(compile(old_source, str(ROOT / relative), 'exec'))
    for name, value in vars(module).items():
        if isinstance(value, types.FunctionType) and value.__module__ == module.__name__:
            assert normalized(value.__code__) == old_codes[name], name
    before, before_director = engine.snapshot(), copy.deepcopy(director)
    before_history = json.dumps(engine.history, separators=(',', ':'), allow_nan=False)
    count = len(engine.actions)
    saved = load_workspace(destination)[1]['surgical_toolbox']
    assert saved['branches'][branch_id] == before and saved['fullCase'][branch_id] == director
    assert director['status'] == 'running' and director['lastStateId'] == engine.state_id
    assert director.get('accessAxis') == 0 and director['deferredAccessPass']['status'] == 'pending'
    assert engine.guard_report()['mode'] == 'strict-research'
    metrics_before = engine.metrics()
    assert metrics_before['activeVesselWounds'] == 0 and metrics_before['vesselPhysicsV2']['woundCount'] == 0
    assert hashlib.sha256(original.read_bytes()).hexdigest() == source_sha
    assert array_digest(scene.labels) == label_hash and array_digest(case.mri) == image_hash
    method_codes = {name: getattr(type(engine), name).__code__ for name in
        ('step', 'preview', '_execute', '_substep', '_physical_substep', 'cells', 'occupancy')}
    thermal = importlib.import_module('resectionlab.surgical_drilling_v2')
    thermal_codes = {name: fn.__code__ for name, fn in vars(thermal).items()
        if isinstance(fn, types.FunctionType) and fn.__module__ == thermal.__name__}
    director_code = rehearsal.run_slice.__code__
    checkpoint = destination.parent / 'checkpoints' / ('primary-before-short-directional-proposals-' + str(count) + '.ressectionlab')
    proof_path = destination.parent / 'short-directional-proposal-reload-proof.json'
    assert not checkpoint.exists() and not proof_path.exists()
    os.link(destination, checkpoint)
    importlib.reload(module)
    new_codes = functions(compile(source, str(ROOT / relative), 'exec'))
    for name, value in vars(module).items():
        if isinstance(value, types.FunctionType) and value.__module__ == module.__name__:
            assert normalized(value.__code__) == new_codes[name], name
    assert engine.snapshot() == before and director == before_director
    assert json.dumps(engine.history, separators=(',', ':'), allow_nan=False) == before_history
    assert all(getattr(type(engine), name).__code__ is code for name, code in method_codes.items())
    assert all(getattr(thermal, name).__code__ is code for name, code in thermal_codes.items())
    assert rehearsal.run_slice.__code__ is director_code
    proof = {'utc': datetime.now(timezone.utc).isoformat(), 'codeCommit': expected_commit,
        'priorProposalCommit': previous_commit, 'loadedOnly': ['full_case_directional'],
        'startPosition': count, 'startStateId': engine.state_id, 'sourceUnchanged': True,
        'engineSnapshotHistoryDirectorUnchanged': True, 'engineMethodCodesUnchanged': list(method_codes),
        'thermalFunctionCodesUnchanged': list(thermal_codes), 'directorCodeUnchanged': True,
        'physicalActionsDuringReload': 0, 'checkpoint': str(checkpoint),
        'checkpointSha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        'qualification': 'Short geometric proposals still require public preview/step; outside fallback retained. No prior patient interval repeated.'}
    proof_path.write_text(json.dumps(proof, indent=2) + '\n')
    assert pause.read_text() == pause_text
    pause.unlink()
    receipt()
    print(json.dumps({'shortDirectionalProposalReloaded': proof}), flush=True)
    interval_started = time.perf_counter()
    advance(1, slice_actions=12)
    saved = load_workspace(destination)[1]['surgical_toolbox']
    assert saved['branches'][branch_id] == engine.snapshot() and saved['fullCase'][branch_id] == director
    assert engine.actions[:count] == before['actions'] and director['lastStateId'] == engine.state_id
    previous_state = before['stateId']
    for row in engine.history[count:]:
        assert row['stateBefore'] == previous_state and row['status'] in ('accepted', 'partial')
        previous_state = row['stateAfter']
    assert previous_state == engine.state_id
    assert hashlib.sha256(original.read_bytes()).hexdigest() == source_sha
    assert array_digest(scene.labels) == label_hash and array_digest(case.mri) == image_hash
    assert all(getattr(type(engine), name).__code__ is code for name, code in method_codes.items())
    after_metrics = engine.metrics()
    proof['firstAdvance'] = {'utc': datetime.now(timezone.utc).isoformat(), 'endPosition': len(engine.actions),
        'newActionCount': len(engine.actions)-count, 'endStateId': engine.state_id,
        'statuses': dict(Counter(row['status'] for row in engine.history[count:])),
        'targetRemovedMm3': after_metrics['targetRemovedMm3'],
        'newTargetRemovedMm3': after_metrics['targetRemovedMm3']-metrics_before['targetRemovedMm3'],
        'newTravelMm': after_metrics['travelMm']-metrics_before['travelMm'],
        'newSimulationSeconds': after_metrics['elapsedSeconds']-metrics_before['elapsedSeconds'],
        'observedWallSeconds': time.perf_counter()-interval_started,
        'activeWounds': after_metrics['activeVesselWounds'], 'woundCount': after_metrics['vesselPhysicsV2']['woundCount'],
        'directorStatus': director['status'], 'savedSnapshotEqualsLive': True,
        'continuousRecordedStateChain': True, 'sourceUnchanged': True}
    proof_path.write_text(json.dumps(proof, indent=2) + '\n')
    receipt()
    print(json.dumps({'firstShortDirectionalAdvanceVerified': proof['firstAdvance']}), flush=True)
    return metrics_before['vesselPhysicsV2']['woundCount']


_short_directional_baseline_wounds = _load_reviewed_short_directional()
for _short_directional_batch in range(4000):
    _short_directional_metrics = engine.metrics()
    if director['status'] != 'running' or (destination.parent / 'pause-after-batch').exists():
        break
    if (_short_directional_metrics['activeVesselWounds'] != 0
            or _short_directional_metrics['vesselPhysicsV2']['woundCount'] != _short_directional_baseline_wounds):
        receipt()
        print('Actual new wound retained at the saved boundary for source-control review.', flush=True)
        break
    advance(1, slice_actions=12)
