"""Load reviewed scheduling only; request a trial after natural opening finish.

Execute only at the live console's actual fully saved prompt. Commit must be
pinned after independent review. No patient restore or physics module reload.
"""
def _load_reviewed_access_deferral():
    global rehearsal, director
    import copy
    import os
    import subprocess
    import types

    expected_commit = 'de837ad687423bbe7ef852c7ca926bc93c2c50df'
    previous_director_commit = '1ae5f004'
    assert len(expected_commit) == 40
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip() == expected_commit
    relatives = ('src/resectionlab/full_case_cleanup.py', 'src/resectionlab/full_case_rehearsal.py')
    source = {}
    for relative in relatives:
        source[relative] = subprocess.check_output(['git', 'show', expected_commit + ':' + relative], cwd=ROOT)
        assert (ROOT / relative).read_bytes() == source[relative]
    pause = destination.parent / 'pause-after-batch'
    assert json.loads(pause.read_text())['purpose'] == 'load-reviewed-access-deferral-20261020'

    def child(code, name):
        values = [value for value in code.co_consts if isinstance(value, types.CodeType) and value.co_name == name]
        assert len(values) == 1, name
        return values[0]

    def normalized(code):
        return (code.co_code, tuple(normalized(v) if isinstance(v, types.CodeType) else v for v in code.co_consts),
            code.co_names, code.co_varnames, code.co_freevars, code.co_cellvars, code.co_argcount,
            code.co_posonlyargcount, code.co_kwonlyargcount, code.co_nlocals, code.co_stacksize,
            code.co_flags, code.co_exceptiontable, code.co_name, code.co_qualname)

    relative = relatives[1]
    prior_source = subprocess.check_output(['git', 'show', previous_director_commit + ':' + relative], cwd=ROOT)
    prior_compiled = compile(prior_source, str(ROOT / relative), 'exec')
    assert normalized(rehearsal.run_slice.__code__) == normalized(child(prior_compiled, 'run_slice'))
    before, before_director = engine.snapshot(), copy.deepcopy(director)
    history_before = json.dumps(engine.history, separators=(',', ':'), allow_nan=False)
    count = len(engine.actions)
    saved = load_workspace(destination)[1]['surgical_toolbox']
    assert saved['branches'][branch_id] == before and saved['fullCase'][branch_id] == director
    assert director['status'] == 'running' and director['lastStateId'] == engine.state_id
    assert director.get('accessAxis', 2) == 2
    assert hashlib.sha256(original.read_bytes()).hexdigest() == source_sha
    assert array_digest(scene.labels) == label_hash and array_digest(case.mri) == image_hash
    methods = {name: getattr(type(engine), name).__code__ for name in
        ('step', 'preview', '_execute', '_substep', '_physical_substep', 'cells', 'occupancy')}
    thermal_module = importlib.import_module('resectionlab.surgical_drilling_v2')
    thermal_codes = {name: value.__code__ for name, value in vars(thermal_module).items()
        if isinstance(value, types.FunctionType) and value.__module__ == thermal_module.__name__}
    checkpoint = destination.parent / 'checkpoints' / ('primary-before-access-deferral-' + str(count) + '.ressectionlab')
    assert not checkpoint.exists()
    os.link(destination, checkpoint)
    proof_path = destination.parent / 'access-deferral-scheduling-reload-proof.json'
    assert not proof_path.exists()

    for name in ('resectionlab.full_case_cleanup', 'resectionlab.full_case_rehearsal'):
        importlib.reload(importlib.import_module(name))
    rehearsal = importlib.import_module('resectionlab.full_case_rehearsal')
    current_compiled = compile(source[relative], str(ROOT / relative), 'exec')
    for name in ('run_slice', 'request_access_pass_deferral', 'resume_deferred_access_pass'):
        assert normalized(getattr(rehearsal, name).__code__) == normalized(child(current_compiled, name))
    assert engine.snapshot() == before and director == before_director
    assert json.dumps(engine.history, separators=(',', ':'), allow_nan=False) == history_before
    assert all(getattr(type(engine), name).__code__ is code for name, code in methods.items())
    assert all(getattr(thermal_module, name).__code__ is code for name, code in thermal_codes.items())
    proposed = rehearsal.request_access_pass_deferral(engine, director,
        reason='Bounded left-access trial after this opening finishes; preserve and later resume the unfinished superior pass.')
    assert director == before_director and engine.snapshot() == before
    assert proposed['lastStateId'] == engine.state_id and proposed['sourceSceneId'] == scene.identity
    assert proposed['stage'] == director['stage'] and proposed['status'] == director['status']
    assert proposed['accesses'] == director['accesses'] and proposed['accessIndex'] == director['accessIndex']
    director = proposed
    persist(case, artifacts, workspace, engine, director, destination, branch_id, parent)
    saved = load_workspace(destination)[1]['surgical_toolbox']
    assert saved['branches'][branch_id] == before and saved['fullCase'][branch_id] == director
    assert len(engine.actions) == count
    assert json.dumps(engine.history, separators=(',', ':'), allow_nan=False) == history_before
    assert all(getattr(type(engine), name).__code__ is code for name, code in methods.items())
    assert all(getattr(thermal_module, name).__code__ is code for name, code in thermal_codes.items())
    assert hashlib.sha256(original.read_bytes()).hexdigest() == source_sha
    assert array_digest(scene.labels) == label_hash and array_digest(case.mri) == image_hash
    proof = {'utc': datetime.now(timezone.utc).isoformat(), 'codeCommit': expected_commit,
        'loadedOnly': ['full_case_cleanup', 'full_case_rehearsal'], 'priorDirectorCommit': previous_director_commit,
        'actions': count, 'stateIdBefore': before['stateId'], 'stateIdAfter': engine.state_id,
        'savedBoundaryEqualsLiveSnapshot': True, 'engineAndHistoryUnchanged': True,
        'engineMethodCodeIdentitiesUnchanged': list(methods), 'thermalFunctionCodesUnchanged': list(thermal_codes),
        'sourceUnchanged': True, 'requestSaved': True, 'requestedAccessIndex': director['accessIndex'],
        'requestedAccessId': director['accesses'][director['accessIndex']]['id'],
        'physicalActionsDuringReloadAndRequest': 0, 'originalDirectorUnchangedDuringRequestCall': True,
        'checkpoint': str(checkpoint), 'checkpointSha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        'qualification': 'Scheduling request only. Current core is retained until its existing natural opening-finish path; ordinary guarded cleanup, a real withdrawn boundary and an admitted left trial remain separate work. No exhaustion or left reachability claim.'}
    proof_path.write_text(json.dumps(proof, indent=2) + '\n')
    pause.unlink()
    receipt()
    print(json.dumps({'accessDeferralRequestSaved': proof}), flush=True)


_load_reviewed_access_deferral()
if director['status'] == 'running':
    advance(4000)
