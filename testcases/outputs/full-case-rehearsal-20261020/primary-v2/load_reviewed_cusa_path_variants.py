"""Run once at a real saved prompt after root pins the reviewed commit.

Only the new proposal helper and run_slice are installed. No reload, patient
restore, old action rewrite, physical-law update or automatic retry.
"""
def _load_reviewed_cusa_path_variants():
    import copy
    import hashlib
    import importlib
    import json
    import os
    import re
    import subprocess
    import sys
    import tempfile
    import types
    from pathlib import Path
    from collections import Counter
    from datetime import datetime, timezone

    expected = '28962e66647760043d123d3e3326e0a29b7b6a08'
    previous = '252afcebfc66e5e0a075ee698bea3d4b92d08890'
    assert re.fullmatch(r'[0-9a-f]{40}', expected), 'Root must pin the reviewed commit before execution'
    relative = 'src/resectionlab/full_case_rehearsal.py'
    helper_relative = 'src/resectionlab/full_case_path_variants.py'
    helper_name = 'resectionlab.full_case_path_variants'
    pause = destination.parent / 'pause-after-batch'
    assert not pause.is_symlink()
    pause_text = pause.read_text(); pause_stat = pause.stat()
    assert pause_text.startswith('Root-owned request ') and 'checked CUSA path-variant proposal integration only.' in pause_text
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip() == expected
    subprocess.run(['git', 'diff', '--quiet', expected, '--', 'src/resectionlab'], cwd=ROOT, check=True)
    assert not subprocess.check_output(['git', 'ls-files', '--others', '--exclude-standard', '--', 'src/resectionlab'], cwd=ROOT, text=True).strip()

    def pinned(path, commit=expected):
        value = subprocess.check_output(['git', 'show', commit + ':' + path], cwd=ROOT)
        if commit == expected: assert (ROOT / path).read_bytes() == value
        return compile(value, str(ROOT / path), 'exec')

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

    def top(code): return {v.co_name: v for v in code.co_consts if isinstance(v, types.CodeType)}
    old_codes = top(pinned(relative, previous)); new_codes = top(pinned(relative))
    assert set(new_codes) == set(old_codes)
    assert rehearsal is importlib.import_module('resectionlab.full_case_rehearsal')
    loaded = {k: v for k, v in vars(rehearsal).items() if isinstance(v, types.FunctionType) and v.__module__ == rehearsal.__name__}
    assert set(loaded) == set(old_codes)
    for name, fn in loaded.items():
        assert normalized(fn.__code__) == normalized(old_codes[name]), name
        if name != 'run_slice': assert normalized(new_codes[name]) == normalized(old_codes[name]), name
    old_run = rehearsal.run_slice
    assert old_run.__globals__ is vars(rehearsal) and not new_codes['run_slice'].co_freevars
    replacement = types.FunctionType(new_codes['run_slice'], vars(rehearsal), old_run.__name__, old_run.__defaults__)
    replacement.__kwdefaults__ = copy.deepcopy(old_run.__kwdefaults__)
    replacement.__annotations__ = copy.deepcopy(old_run.__annotations__)
    replacement.__qualname__ = old_run.__qualname__
    # The reviewed integration imports this helper locally inside run_slice.
    assert ('cusa_path_variants' in new_codes['run_slice'].co_varnames or 'cusa_path_variants' in new_codes['run_slice'].co_cellvars)
    old_members = dict(vars(rehearsal))
    assert 'cusa_path_variants' not in old_members
    engine_module = importlib.import_module('resectionlab.surgical_engine')
    assert type(engine) is engine_module.SurgicalEngine
    engine_members = dict(vars(engine_module))
    engine_classes = [(v, dict(vars(v))) for v in engine_members.values()
        if isinstance(v, type) and v.__module__ == engine_module.__name__]
    engine_functions = [(v, v.__code__) for v in engine_members.values()
        if isinstance(v, types.FunctionType) and v.__module__ == engine_module.__name__]
    for cls, members in engine_classes:
        for value in members.values():
            if isinstance(value, (staticmethod, classmethod)): value = value.__func__
            elif isinstance(value, property): value = value.fget
            if isinstance(value, types.FunctionType): engine_functions.append((value, value.__code__))
    thermal = importlib.import_module('resectionlab.surgical_drilling_v2')
    thermal_codes = {k: v.__code__ for k, v in vars(thermal).items()
        if isinstance(v, types.FunctionType) and v.__module__ == thermal.__name__}
    director_codes = {k: v.__code__ for k, v in loaded.items()}
    encoded = lambda v: json.dumps(v, separators=(',', ':'), allow_nan=False)

    def code_check(installed):
        assert set(vars(rehearsal)) == set(old_members)
        assert all(vars(rehearsal)[k] is (replacement if installed and k == 'run_slice' else v) for k, v in old_members.items())
        assert all(fn.__code__ is director_codes[name] for name, fn in loaded.items())
        if installed: assert replacement.__code__ is new_codes['run_slice']
        assert set(vars(engine_module)) == set(engine_members) and all(vars(engine_module)[k] is v for k, v in engine_members.items())
        for cls, members in engine_classes:
            assert set(vars(cls)) == set(members) and all(vars(cls)[k] is v for k, v in members.items())
        assert all(fn.__code__ is code for fn, code in engine_functions)
        assert all(getattr(thermal, k).__code__ is code for k, code in thermal_codes.items())

    def source_check():
        assert engine.scene is scene and scene.identity == 'sha256:7a5d368382fe2a66192bdef88ff6af3a55cd533c9ba3787c68a1d1e3cb48f856'
        scene.require_cache_source()
        assert hashlib.sha256(original.read_bytes()).hexdigest() == source_sha == 'ff5a49b8706a8ccda08d8c3e82dad3e079f85dea98cb6bd40783d2262eec9838'
        assert array_digest(scene.labels) == label_hash and array_digest(case.mri) == image_hash

    def saved_check():
        source_check()
        assert len(engine.history) == len(engine.actions)
        last = None
        for action, row in zip(engine.actions, engine.history):
            assert row['action'] == action and row['status'] in ('accepted', 'partial')
            if last is not None: assert row['stateBefore'] == last
            last = row['stateAfter']
        assert last == engine.state_id
        saved = load_workspace(destination)[1]['surgical_toolbox']
        assert saved['branchId'] == branch_id == 'full-primary' and saved['position'] == len(engine.actions)
        assert saved['branches'][branch_id] == engine.snapshot() and saved['fullCase'][branch_id] == director
        assert director['lastStateId'] == engine.state_id

    assert director['scenario'] == 'primary' and director['status'] == 'running'
    assert engine.guard_report()['mode'] == 'strict-research'
    baseline = engine.metrics()
    assert baseline['activeVesselWounds'] == baseline['vesselPhysicsV2']['woundCount'] == 0
    saved_check(); code_check(False)
    before = engine.snapshot(); prior_director = copy.deepcopy(director); prior_history = encoded(engine.history)
    count = len(engine.actions)
    checkpoint = destination.parent / 'checkpoints' / ('primary-before-cusa-path-variants-' + str(count) + '.ressectionlab')
    proof_path = destination.parent / 'cusa-path-variants-live-load-proof.json'
    assert not checkpoint.exists() and not proof_path.exists(), 'Prior integration exists: inspect, do not retry'
    # The new pure helper is imported once, never reloaded. Its loaded code is
    # compared to the reviewed source before any director binding.
    helper_compiled = pinned(helper_relative); helper_codes = top(helper_compiled)
    assert helper_name not in sys.modules, 'Helper already imported; inspect before attempting integration'
    helper = importlib.import_module(helper_name)
    assert Path(helper.__file__).resolve() == (ROOT / helper_relative).resolve()
    helper_functions = {k: v for k, v in vars(helper).items()
        if isinstance(v, types.FunctionType) and v.__module__ == helper.__name__}
    assert set(helper_functions) == set(helper_codes)
    assert 'cusa_path_variants' in helper_functions
    for name, fn in helper_functions.items(): assert normalized(fn.__code__) == normalized(helper_codes[name]), name
    assert engine.snapshot() == before and director == prior_director and encoded(engine.history) == prior_history
    code_check(False); saved_check()
    native_sha = hashlib.sha256(destination.read_bytes()).hexdigest()
    os.link(destination, checkpoint)
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == native_sha
    checkpoint_saved = load_workspace(checkpoint)[1]['surgical_toolbox']
    assert checkpoint_saved['branches'][branch_id] == before and checkpoint_saved['fullCase'][branch_id] == prior_director
    proof = {'utc': datetime.now(timezone.utc).isoformat(), 'status': 'prepared-not-bound',
        'codeCommit': expected, 'priorDirectorCommit': previous, 'startPosition': count, 'startStateId': before['stateId'],
        'loadedOnly': ['full_case_rehearsal.run_slice', helper_name],
        'checkpoint': str(checkpoint), 'checkpointSha256': native_sha,
        'engineFunctionsAndClassesIncludingCheckpointUnchanged': True, 'thermalCodeUnchanged': True,
        'allOtherDirectorFunctionsUnchanged': True, 'patientSnapshotHistoryDirectorUnchangedDuringBinding': True,
        'physicalActionsDuringBinding': 0,
        'scope': 'New proposal variants only; every future physical transition still uses public preview/step. No old action, physics, source or protection changes.'}
    with proof_path.open('x') as stream: stream.write(json.dumps(proof, indent=2) + '\n')

    def update_proof():
        fd, path = tempfile.mkstemp(prefix='.cusa-variant-proof-', dir=proof_path.parent)
        with os.fdopen(fd, 'w') as stream: stream.write(json.dumps(proof, indent=2) + '\n')
        os.replace(path, proof_path)

    rehearsal.run_slice = replacement
    code_check(True)
    assert engine.snapshot() == before and director == prior_director and encoded(engine.history) == prior_history
    saved_check()
    proof.update(status='bound-no-new-interval', stateIdAfterBinding=engine.state_id, sourceUnchanged=True)
    update_proof()
    assert not pause.is_symlink() and pause.read_text() == pause_text
    current = pause.stat()
    assert (current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns) == (pause_stat.st_dev, pause_stat.st_ino, pause_stat.st_size, pause_stat.st_mtime_ns)
    pause.unlink()
    print(json.dumps({'cusaPathVariantsInstalled': proof}), flush=True)

    def guarded():
        metrics = engine.metrics()
        if director['status'] != 'running' or pause.exists(): return False
        if metrics['activeVesselWounds'] or metrics['vesselPhysicsV2']['woundCount'] != 0:
            receipt(); print('Actual new wound retained at saved boundary for review.', flush=True); return False
        return True

    assert guarded(), 'No new public interval permitted'
    advance(1, slice_actions=2)
    saved_check(); code_check(True)
    assert engine.actions[:count] == before['actions'] and encoded(engine.history[:count]) == prior_history
    last = before['stateId']; rows = engine.history[count:]
    for row in rows:
        assert row['stateBefore'] == last
        last = row['stateAfter']
    assert last == engine.state_id
    metrics = engine.metrics()
    proof['firstNewOrdinaryPublicInterval'] = {'startPosition': count, 'endPosition': len(engine.actions),
        'newActionCount': len(rows), 'endStateId': engine.state_id,
        'statuses': dict(Counter(row['status'] for row in rows)), 'savedNativeEqualsLive': True,
        'continuousRecordedStateChain': True, 'fullPriorActionHistoryPreserved': True,
        'sourceAndEngineCodeUnchanged': True, 'targetRemovedMm3': metrics['targetRemovedMm3'],
        'activeWounds': metrics['activeVesselWounds'], 'woundCount': metrics['vesselPhysicsV2']['woundCount'],
        'directorStatus': director['status'], 'stage': director['stage']}
    proof['status'] = 'first-new-interval-verified' if rows else 'zero-action-stop-no-interval-claimed'
    update_proof(); receipt()
    print(json.dumps({'firstCusaPathVariantsInterval': proof['firstNewOrdinaryPublicInterval']}), flush=True)
    assert rows, 'Zero-action first boundary retained; inspect instead of retrying'
    for _ in range(4000):
        if not guarded(): break
        advance(1, slice_actions=12)


_load_reviewed_cusa_path_variants()
