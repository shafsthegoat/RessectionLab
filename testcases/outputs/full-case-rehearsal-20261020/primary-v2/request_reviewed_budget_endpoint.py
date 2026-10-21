"""Run once at the actual saved prompt; request cleanup without advancing physics."""
def _request_reviewed_budget_endpoint():
    import copy
    import hashlib
    import importlib
    import json
    import re
    import subprocess
    import types
    from datetime import datetime, timezone

    expected = '4087a8fa48f68712cb6acf8883a9ea38d4dbda50'
    previous = '28962e66647760043d123d3e3326e0a29b7b6a08'
    relative = 'src/resectionlab/full_case_rehearsal.py'
    changed = {'run_slice', 'resume_director', 'request_budget_limited_endpoint'}
    assert re.fullmatch(r'[0-9a-f]{40}', expected), 'Reviewed commit must be pinned first'
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip() == expected
    subprocess.run(['git', 'diff', '--quiet', expected, '--', 'src/resectionlab'], cwd=ROOT, check=True)
    pause = destination.parent / 'pause-after-batch'
    assert not pause.is_symlink() and pause.read_text().startswith('User steering:')
    assert len(engine.actions) == 2126 and engine.state_id == 'sha256:85471c3dfb72b0af1a8f2e9d0ae1e41ba7ab82f620b3147c66876c67a1426979'
    checkpoint = destination.parent / 'checkpoints/primary-before-budget-endpoint-2126.ressectionlab'
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == 'fce43e61bae4557ac9e0c3bdf7e26f4c343dba5369e21c11fbab9040b06b8a6b'
    proof_path = destination.parent / 'budget-endpoint-live-transition-proof.json'
    assert not proof_path.exists(), 'Inspect prior transition; never retry automatically'

    def normalized(code):
        def const(value):
            if isinstance(value, types.CodeType): return ('code', normalized(value))
            if type(value) is tuple: return ('tuple', tuple(const(v) for v in value))
            if type(value) is frozenset: return ('frozenset', tuple(sorted((const(v) for v in value), key=repr)))
            return (type(value).__name__, repr(value))
        return (code.co_code, tuple(const(v) for v in code.co_consts), code.co_names,
            code.co_varnames, code.co_freevars, code.co_cellvars, code.co_argcount,
            code.co_posonlyargcount, code.co_kwonlyargcount, code.co_nlocals,
            code.co_stacksize, code.co_flags, code.co_exceptiontable, code.co_name, code.co_qualname)

    def compiled(commit):
        raw = subprocess.check_output(['git', 'show', commit + ':' + relative], cwd=ROOT)
        if commit == expected: assert raw == (ROOT / relative).read_bytes()
        return {v.co_name: v for v in compile(raw, str(ROOT / relative), 'exec').co_consts
                if isinstance(v, types.CodeType)}

    old_codes, new_codes = compiled(previous), compiled(expected)
    assert set(new_codes) == set(old_codes) | {'request_budget_limited_endpoint'}
    assert rehearsal is importlib.import_module('resectionlab.full_case_rehearsal')
    members = dict(vars(rehearsal))
    loaded = {k: v for k, v in members.items() if isinstance(v, types.FunctionType) and v.__module__ == rehearsal.__name__}
    assert set(loaded) == set(old_codes)
    for name, function in loaded.items():
        assert normalized(function.__code__) == normalized(old_codes[name]), name
        if name not in changed: assert normalized(new_codes[name]) == normalized(old_codes[name]), name
    replacements = {}
    for name in changed:
        old = loaded.get(name)
        function = types.FunctionType(new_codes[name], vars(rehearsal), name, old.__defaults__ if old else None)
        if old:
            function.__kwdefaults__ = copy.deepcopy(old.__kwdefaults__)
            function.__annotations__ = copy.deepcopy(old.__annotations__)
        replacements[name] = function
    engine_module = importlib.import_module('resectionlab.surgical_engine')
    assert type(engine) is engine_module.SurgicalEngine
    engine_members = dict(vars(engine_module))
    physical = [(v, dict(vars(v))) for v in engine_members.values()
                if isinstance(v, type) and v.__module__ == engine_module.__name__]
    functions = [(v, v.__code__) for v in engine_members.values() if isinstance(v, types.FunctionType)]
    for cls, values in physical:
        for value in values.values():
            if isinstance(value, (staticmethod, classmethod)): value = value.__func__
            elif isinstance(value, property): value = value.fget
            if isinstance(value, types.FunctionType): functions.append((value, value.__code__))
    core_digest = engine_module.semantic_digest
    core_digest_code = core_digest.__code__
    before = engine.snapshot(); prior_history = copy.deepcopy(engine.history); prior_director = copy.deepcopy(director)

    def checks():
        assert engine.snapshot() == before and engine.history == prior_history
        assert set(vars(engine_module)) == set(engine_members) and all(vars(engine_module)[k] is v for k, v in engine_members.items())
        for cls, values in physical:
            assert set(vars(cls)) == set(values) and all(vars(cls)[k] is v for k, v in values.items())
        assert all(fn.__code__ is code for fn, code in functions)
        assert engine_module.semantic_digest is core_digest and core_digest.__code__ is core_digest_code
        assert engine.scene is scene and array_digest(scene.labels) == label_hash and array_digest(case.mri) == image_hash
        assert hashlib.sha256(original.read_bytes()).hexdigest() == source_sha == 'ff5a49b8706a8ccda08d8c3e82dad3e079f85dea98cb6bd40783d2262eec9838'
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == 'fce43e61bae4557ac9e0c3bdf7e26f4c343dba5369e21c11fbab9040b06b8a6b'

    saved = load_workspace(destination)[1]['surgical_toolbox']
    assert saved['branches'][branch_id] == before and saved['fullCase'][branch_id] == prior_director
    checks()
    for name, value in replacements.items(): setattr(rehearsal, name, value)
    assert set(vars(rehearsal)) == set(members) | {'request_budget_limited_endpoint'}
    assert all(vars(rehearsal)[k] is (replacements[k] if k in replacements else v) for k, v in members.items())
    checks()
    requested = rehearsal.request_budget_limited_endpoint(engine, director)
    checks()
    assert requested['stage'] == 'cleanup' and requested['status'] == 'ready'
    globals()['director'] = requested
    persist(case, artifacts, workspace, engine, director, destination, branch_id, parent)
    saved = load_workspace(destination)[1]['surgical_toolbox']
    assert saved['branches'][branch_id] == before and saved['fullCase'][branch_id] == requested
    checks()
    proof = {'status': 'cleanup-requested-no-physical-actions-yet', 'utc': datetime.now(timezone.utc).isoformat(),
        'reviewedCommit': expected, 'priorDirectorCommit': previous, 'position': len(engine.actions),
        'stateId': engine.state_id, 'loadedOnly': sorted(changed), 'physicalActionsDuringBinding': 0,
        'engineAndLiveHashCodeUnchanged': True, 'sourceAndEntireActionHistoryUnchanged': True,
        'checkpoint': str(checkpoint), 'pausedPass': requested['budgetLimitedEndpoint'],
        'resectionSchedulingPermitted': False, 'pauseFlagRetained': pause.exists()}
    with proof_path.open('x') as stream: stream.write(json.dumps(proof, indent=2)+'\n')
    receipt()
    print(json.dumps({key: proof[key] for key in ('status', 'position', 'stateId',
        'loadedOnly', 'physicalActionsDuringBinding', 'resectionSchedulingPermitted', 'pauseFlagRetained')}), flush=True)


_request_reviewed_budget_endpoint()
