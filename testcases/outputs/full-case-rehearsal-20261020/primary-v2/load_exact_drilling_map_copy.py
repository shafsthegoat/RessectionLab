"""Run once at an actual saved prompt ONLY after root pins the reviewed commit.

Only _microstep_checkpoint is rebound. No module/class reload, patient restore,
prior interval replay, pause-flag retry, physics policy change or speed claim.
A partial setup leaves its checkpoint/proof and requires manual inspection.
"""
def _load_exact_drilling_map_copy():
    import copy
    import hashlib
    import importlib
    import json
    import os
    import re
    import struct
    import subprocess
    import types
    from collections import Counter
    from datetime import datetime, timezone

    expected = '252afcebfc66e5e0a075ee698bea3d4b92d08890'
    previous = '4dbdee81c5fcd35a6504b572d3e8c46d544becc0'
    assert re.fullmatch(r'[0-9a-f]{40}', expected), 'Root must pin the reviewed commit; do not execute this placeholder'
    relative = 'src/resectionlab/surgical_engine.py'
    pause = destination.parent / 'pause-after-batch'
    assert not pause.is_symlink()
    pause_text = pause.read_text()
    pause_stat = pause.stat()
    assert pause_text.startswith('Root-owned request ') and 'exact flat drilling-map rollback-copy integration only.' in pause_text
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip() == expected
    subprocess.run(['git', 'diff', '--quiet', expected, '--', 'src/resectionlab'], cwd=ROOT, check=True)
    assert not subprocess.check_output(['git', 'ls-files', '--others', '--exclude-standard', '--', 'src/resectionlab'], cwd=ROOT, text=True).strip()
    source = subprocess.check_output(['git', 'show', expected + ':' + relative], cwd=ROOT)
    old_source = subprocess.check_output(['git', 'show', previous + ':' + relative], cwd=ROOT)
    assert (ROOT / relative).read_bytes() == source
    module = importlib.import_module('resectionlab.surgical_engine')
    assert type(engine) is module.SurgicalEngine and engine.scene is scene

    def normalized(code):
        # Exclude filenames/line tables/line numbers only, including nested code.
        def const(value):
            if isinstance(value, types.CodeType): return ('code', normalized(value))
            if type(value) is tuple: return ('tuple', tuple(const(v) for v in value))
            if type(value) is frozenset: return ('frozenset', tuple(sorted((const(v) for v in value), key=repr)))
            return (type(value).__name__, repr(value))
        return (code.co_code, tuple(const(v) for v in code.co_consts), code.co_names,
            code.co_varnames, code.co_freevars, code.co_cellvars, code.co_argcount,
            code.co_posonlyargcount, code.co_kwonlyargcount, code.co_nlocals,
            code.co_stacksize, code.co_flags, code.co_exceptiontable, code.co_name, code.co_qualname)

    def child(code, name):
        found = [v for v in code.co_consts if isinstance(v, types.CodeType) and v.co_name == name]
        assert len(found) == 1, name
        return found[0]

    old_compiled = compile(old_source, str(ROOT / relative), 'exec')
    compiled = compile(source, str(ROOT / relative), 'exec')
    old_top = {v.co_name: v for v in old_compiled.co_consts if isinstance(v, types.CodeType)}
    new_top = {v.co_name: v for v in compiled.co_consts if isinstance(v, types.CodeType)}
    assert set(old_top) == set(new_top)
    for name in old_top:
        if name != '_microstep_checkpoint': assert normalized(old_top[name]) == normalized(new_top[name]), name
    old_helper = module._microstep_checkpoint
    assert old_helper.__globals__ is vars(module) and old_helper.__defaults__ is None and old_helper.__kwdefaults__ is None
    assert normalized(old_helper.__code__) == normalized(old_top['_microstep_checkpoint'])
    # Pin every loaded engine function, including the unchanged _execute.
    for name, value in vars(module).items():
        if isinstance(value, types.FunctionType) and value.__module__ == module.__name__:
            assert normalized(value.__code__) == normalized(old_top[name]), name
    for class_name, cls in vars(module).items():
        if isinstance(cls, type) and cls.__module__ == module.__name__:
            class_code = old_top[class_name]
            for name, value in vars(cls).items():
                if isinstance(value, (staticmethod, classmethod)): value = value.__func__
                elif isinstance(value, property): value = value.fget
                if isinstance(value, types.FunctionType):
                    assert normalized(value.__code__) == normalized(child(class_code, name)), class_name + '.' + name
    helper_code = new_top['_microstep_checkpoint']
    assert not helper_code.co_freevars
    candidate = types.FunctionType(helper_code, vars(module), '_microstep_checkpoint')
    candidate.__qualname__ = old_helper.__qualname__
    candidate.__annotations__ = copy.deepcopy(old_helper.__annotations__)
    frozen = types.FunctionType(old_top['_microstep_checkpoint'], vars(module), '_microstep_checkpoint')
    module_members = dict(vars(module))
    class_members = {name: (cls, dict(vars(cls))) for name, cls in vars(module).items()
        if isinstance(cls, type) and cls.__module__ == module.__name__}
    function_codes = [(v, v.__code__) for v in module_members.values()
        if isinstance(v, types.FunctionType) and v.__module__ == module.__name__]
    for cls, members in class_members.values():
        for value in members.values():
            if isinstance(value, (staticmethod, classmethod)): value = value.__func__
            elif isinstance(value, property): value = value.fget
            if isinstance(value, types.FunctionType): function_codes.append((value, value.__code__))
    thermal = importlib.import_module('resectionlab.surgical_drilling_v2')
    thermal_codes = {name: fn.__code__ for name, fn in vars(thermal).items()
        if isinstance(fn, types.FunctionType) and fn.__module__ == thermal.__name__}
    director_code = rehearsal.run_slice.__code__
    encoded = lambda v: json.dumps(v, separators=(',', ':'), allow_nan=False)

    def code_check(installed):
        assert set(vars(module)) == set(module_members)
        assert all(vars(module)[k] is (candidate if k == '_microstep_checkpoint' and installed else v)
                   for k, v in module_members.items())
        for cls, members in class_members.values():
            assert set(vars(cls)) == set(members) and all(vars(cls)[k] is v for k, v in members.items())
        assert all(fn.__code__ is code for fn, code in function_codes)
        if installed: assert candidate.__code__ is helper_code
        assert all(getattr(thermal, k).__code__ is v for k, v in thermal_codes.items())
        assert rehearsal.run_slice.__code__ is director_code

    def source_check():
        assert engine.scene is scene and scene.identity == 'sha256:7a5d368382fe2a66192bdef88ff6af3a55cd533c9ba3787c68a1d1e3cb48f856'
        scene.require_cache_source()
        assert hashlib.sha256(original.read_bytes()).hexdigest() == source_sha == 'ff5a49b8706a8ccda08d8c3e82dad3e079f85dea98cb6bd40783d2262eec9838'
        assert array_digest(scene.labels) == label_hash and array_digest(case.mri) == image_hash

    def history_check():
        assert len(engine.history) == len(engine.actions)
        previous_state = None
        for action, row in zip(engine.actions, engine.history):
            assert row['action'] == action and row['status'] in ('accepted', 'partial')
            if previous_state is not None: assert row['stateBefore'] == previous_state
            previous_state = row['stateAfter']
        assert previous_state == engine.state_id

    def saved_check():
        source_check(); history_check()
        saved = load_workspace(destination)[1]['surgical_toolbox']
        assert saved['branchId'] == branch_id == 'full-primary' and saved['position'] == len(engine.actions)
        assert saved['branches'][branch_id] == engine.snapshot() and saved['fullCase'][branch_id] == director
        assert director['lastStateId'] == engine.state_id

    assert director['scenario'] == 'primary' and director['status'] == 'running'
    assert engine.guard_report()['mode'] == 'strict-research'
    metrics_before = engine.metrics()
    assert metrics_before['activeVesselWounds'] == metrics_before['vesselPhysicsV2']['woundCount'] == 0
    saved_check(); code_check(False)
    count = len(engine.actions); before = engine.snapshot(); previous_director = copy.deepcopy(director)
    old_history = encoded(engine.history); original_state_json = encoded(engine.state)

    def graph(value):
        # Deterministic container reference numbering proves types and alias topology.
        seen = {}; mutable_ids = set()
        def visit(v):
            t = type(v)
            if t in (dict, list, tuple):
                if t in (dict, list): mutable_ids.add(id(v))
                if id(v) in seen: return ['ref', seen[id(v)]]
                n = len(seen); seen[id(v)] = n
                children = [[visit(k), visit(x)] for k, x in v.items()] if t is dict else [visit(x) for x in v]
                return [t.__name__, n, children]
            if t is float: return ['float', struct.pack('!d', v).hex()]
            assert t in (str, int, bool, type(None)), t
            return [t.__name__, v]
        return visit(value), mutable_ids

    original_graph, original_ids = graph(engine.state)
    legacy = frozen(engine.state); proposed = candidate(engine.state)
    legacy_graph, _ = graph(legacy); proposed_graph, proposed_ids = graph(proposed)
    assert encoded(legacy) == encoded(proposed) and legacy_graph == proposed_graph
    components = engine.state.get('flapComponents')
    assert legacy.get('flapComponents') is components and proposed.get('flapComponents') is components
    _, exempt_ids = graph(components)
    assert original_ids.intersection(proposed_ids) <= exempt_ids
    assert proposed['removed'] is not engine.state['removed'] and proposed['debris'] is not engine.state['debris']
    assert proposed['drillingV2'] is not engine.state['drillingV2']
    map_names = ('workJ', 'temperatureC', 'thermalDoseCSeconds')
    for name in map_names:
        original_map = engine.state['drillingV2'][name]
        assert type(original_map) is dict and all(type(k) is str and type(v) in (int, float) for k, v in original_map.items())
        assert proposed['drillingV2'][name] is not original_map
    assert encoded(engine.state) == original_state_json and graph(engine.state)[0] == original_graph
    assert engine.snapshot() == before and director == previous_director and encoded(engine.history) == old_history
    code_check(False); saved_check()

    checkpoint = destination.parent / 'checkpoints' / ('primary-before-exact-drilling-map-copy-' + str(count) + '.ressectionlab')
    proof_path = destination.parent / 'exact-drilling-map-copy-live-reload-proof.json'
    assert not checkpoint.exists() and not proof_path.exists(), 'Prior setup exists; inspect it, never automatically retry'
    proof = {'utc': datetime.now(timezone.utc).isoformat(), 'status': 'prepared-not-bound',
        'codeCommit': expected, 'oldHelperCommit': previous, 'loadedOnly': ['surgical_engine._microstep_checkpoint'],
        'startPosition': count, 'startStateId': before['stateId'], 'checkpoint': str(checkpoint),
        'allEngineClassMembersIncludingExecutePreserved': True, 'thermalAndDirectorCodePreserved': True,
        'exactDetachedJsonOrderTypesFloatBitsAndAliasGraph': True, 'threeMapsIndependentlyOwned': list(map_names),
        'existingComponentExemptionPreserved': True, 'physicalActionsDuringBinding': 0,
        'detachedCopyJsonSha256': hashlib.sha256(encoded(proposed).encode()).hexdigest(),
        'detachedCopyGraphSha256': hashlib.sha256(encoded(proposed_graph).encode()).hexdigest(),
        'scope': 'Helper-only equivalent rollback data copying; no module reload, patient replay or action-speed claim.'}
    native_sha = hashlib.sha256(destination.read_bytes()).hexdigest()
    os.link(destination, checkpoint)
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == native_sha
    checkpoint_record = load_workspace(checkpoint)[1]['surgical_toolbox']
    assert checkpoint_record['branches'][branch_id] == before and checkpoint_record['fullCase'][branch_id] == previous_director
    proof['checkpointSha256'] = native_sha
    with proof_path.open('x') as stream: stream.write(json.dumps(proof, indent=2) + '\n')

    def update_proof():
        import tempfile
        fd, path = tempfile.mkstemp(prefix='.drilling-map-proof-', dir=proof_path.parent)
        with os.fdopen(fd, 'w') as stream: stream.write(json.dumps(proof, indent=2) + '\n')
        os.replace(path, proof_path)

    module._microstep_checkpoint = candidate
    code_check(True)
    assert engine.snapshot() == before and len(engine.actions) == count and director == previous_director
    assert encoded(engine.history) == old_history and encoded(engine.state) == original_state_json
    assert graph(engine.state)[0] == original_graph
    saved_check()
    proof.update(status='bound-no-new-interval', stateIdAfterBinding=engine.state_id,
        sourceUnchanged=True, savedBoundaryUnchanged=True, patientHistoryAndDirectorUnchanged=True)
    update_proof()
    assert not pause.is_symlink() and pause.read_text() == pause_text
    current_pause = pause.stat()
    assert (current_pause.st_dev, current_pause.st_ino, current_pause.st_size, current_pause.st_mtime_ns) == (pause_stat.st_dev, pause_stat.st_ino, pause_stat.st_size, pause_stat.st_mtime_ns)
    pause.unlink()
    print(json.dumps({'exactDrillingMapCopyBound': proof}), flush=True)

    def guarded():
        metrics = engine.metrics()
        if director['status'] != 'running' or pause.exists(): return False
        if metrics['activeVesselWounds'] or metrics['vesselPhysicsV2']['woundCount'] != 0:
            receipt(); print('Actual new wound retained at saved boundary for review.', flush=True)
            return False
        return True

    def interval(size):
        assert guarded(), 'No new interval permitted at this actual boundary'
        start = len(engine.actions); prior_snapshot = engine.snapshot(); prior_history = encoded(engine.history)
        advance(1, slice_actions=size)
        saved_check(); code_check(True)
        assert engine.actions[:start] == prior_snapshot['actions'] and encoded(engine.history[:start]) == prior_history
        rows = engine.history[start:]; last = prior_snapshot['stateId']
        for row in rows:
            assert row['stateBefore'] == last
            last = row['stateAfter']
        assert last == engine.state_id
        metrics = engine.metrics()
        return {'startPosition': start, 'endPosition': len(engine.actions), 'newActionCount': len(rows),
            'endStateId': engine.state_id, 'statuses': dict(Counter(row['status'] for row in rows)),
            'savedNativeEqualsLive': True, 'fullPriorActionsAndHistoryPreserved': True,
            'continuousRecordedStateChain': True, 'sourceAndCodeUnchanged': True,
            'activeWounds': metrics['activeVesselWounds'], 'woundCount': metrics['vesselPhysicsV2']['woundCount'],
            'directorStatus': director['status'], 'stage': director['stage']}

    proof['firstNewOrdinaryPublicInterval'] = interval(2)
    proof['status'] = 'first-new-interval-verified' if proof['firstNewOrdinaryPublicInterval']['newActionCount'] else 'zero-action-stop-no-interval-claimed'
    update_proof(); receipt()
    print(json.dumps({'firstDrillingMapCopyInterval': proof['firstNewOrdinaryPublicInterval']}), flush=True)
    assert proof['firstNewOrdinaryPublicInterval']['newActionCount'] > 0, 'Zero-action boundary retained; inspect proof and do not retry'
    for _ in range(4000):
        if not guarded(): break
        advance(1, slice_actions=12)


_load_exact_drilling_map_copy()
