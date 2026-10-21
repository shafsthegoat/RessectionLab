"""At a real saved prompt, replace only the reviewed rollback call/helper.

No module reload, patient restore, source change or extra patient replay.
The first post-change interval is new ordinary public preview/step work.
"""
def _load_exact_rollback_copy():
    import copy
    import marshal
    import os
    import struct
    import subprocess
    import types
    from collections import Counter

    expected = '0c4045cfe59c4f6fcad11e041e982baf22ba9193'
    previous = '3e0bd07c7a5ad7b84a9cf921c20f1c2564cd2c69'
    relative = 'src/resectionlab/surgical_engine.py'
    pause = destination.parent / 'pause-after-batch'
    assert json.loads(pause.read_text())['purpose'] == 'load-exact-rollback-copy-20261020'
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip() == expected
    current_source = subprocess.check_output(['git', 'show', expected + ':' + relative], cwd=ROOT)
    assert (ROOT / relative).read_bytes() == current_source
    prior_source = subprocess.check_output(['git', 'show', previous + ':' + relative], cwd=ROOT)
    module = importlib.import_module('resectionlab.surgical_engine')
    owner = next(cls for cls in type(engine).__mro__ if '_execute' in vars(cls))
    assert owner is module.SurgicalEngine
    old_execute = owner._execute
    assert old_execute.__globals__ is vars(module)
    assert '_microstep_checkpoint' not in vars(module)

    def child(code, name):
        matches = [value for value in code.co_consts if isinstance(value, types.CodeType) and value.co_name == name]
        assert len(matches) == 1, name
        return matches[0]

    def normalized(code):
        # Ignore file/line positions only; compare bytecode, constants, names,
        # signature, flags and exception handling, including nested code.
        return (code.co_code, tuple(normalized(v) if isinstance(v, types.CodeType) else v for v in code.co_consts),
            code.co_names, code.co_varnames, code.co_freevars, code.co_cellvars, code.co_argcount,
            code.co_posonlyargcount, code.co_kwonlyargcount, code.co_nlocals, code.co_stacksize,
            code.co_flags, code.co_exceptiontable, code.co_name, code.co_qualname)

    prior_compiled = compile(prior_source, str(ROOT / relative), 'exec')
    current_compiled = compile(current_source, str(ROOT / relative), 'exec')
    old_code = child(child(prior_compiled, 'SurgicalEngine'), '_execute')
    new_code = child(child(current_compiled, 'SurgicalEngine'), '_execute')
    assert normalized(old_execute.__code__) == normalized(old_code)
    helper_code = child(current_compiled, '_microstep_checkpoint')
    assert not helper_code.co_freevars and not new_code.co_freevars
    helper = types.FunctionType(helper_code, vars(module), '_microstep_checkpoint')
    replacement = types.FunctionType(new_code, vars(module), old_execute.__name__, old_execute.__defaults__)
    replacement.__kwdefaults__ = copy.deepcopy(old_execute.__kwdefaults__)
    replacement.__annotations__ = copy.deepcopy(old_execute.__annotations__)
    replacement.__qualname__ = old_execute.__qualname__

    count = len(engine.actions)
    before = engine.snapshot()
    before_director = copy.deepcopy(director)
    before_history = json.dumps(engine.history, separators=(',', ':'), allow_nan=False)
    saved = load_workspace(destination)[1]['surgical_toolbox']
    assert saved['branches'][branch_id] == before and saved['fullCase'][branch_id] == director
    assert director['status'] == 'running' and director['lastStateId'] == engine.state_id
    assert hashlib.sha256(original.read_bytes()).hexdigest() == source_sha
    assert array_digest(scene.labels) == label_hash and array_digest(case.mri) == image_hash

    components = engine.state.get('flapComponents')
    legacy_copy = copy.deepcopy(engine.state, {id(components): components} if components is not None else None)
    candidate_copy = helper(engine.state)
    encoded = lambda value: json.dumps(value, separators=(',', ':'), allow_nan=False)
    exact_json = encoded(legacy_copy)
    assert encoded(candidate_copy) == exact_json

    def floats(value):
        if type(value) is float:
            return [struct.pack('!d', value).hex()]
        if type(value) is dict:
            return [item for key, child_value in value.items() for item in floats(child_value)]
        if type(value) in (list, tuple):
            return [item for child_value in value for item in floats(child_value)]
        return []

    assert floats(legacy_copy) == floats(candidate_copy)
    assert candidate_copy['removed'] is not engine.state['removed']
    assert candidate_copy['debris'] is not engine.state['debris']
    assert candidate_copy.get('flapComponents') is components
    methods = {name: getattr(type(engine), name).__code__ for name in
        ('step', 'preview', '_execute', '_substep', '_physical_substep', 'cells', 'occupancy')}
    director_code = rehearsal.run_slice.__code__
    checkpoint = destination.parent / 'checkpoints' / ('primary-before-exact-rollback-copy-' + str(count) + '.ressectionlab')
    assert not checkpoint.exists()
    os.link(destination, checkpoint)
    proof_path = destination.parent / 'exact-rollback-copy-live-reload-proof.json'
    assert not proof_path.exists()

    # Bind only this function and its single new helper. All existing class,
    # module, source, model, action, observer and director objects stay live.
    module._microstep_checkpoint = helper
    owner._execute = replacement
    assert engine.snapshot() == before and director == before_director
    assert json.dumps(engine.history, separators=(',', ':'), allow_nan=False) == before_history
    assert all(getattr(type(engine), name).__code__ is code for name, code in methods.items() if name != '_execute')
    assert rehearsal.run_slice.__code__ is director_code
    assert normalized(owner._execute.__code__) == normalized(new_code)
    assert array_digest(scene.labels) == label_hash and array_digest(case.mri) == image_hash
    assert hashlib.sha256(original.read_bytes()).hexdigest() == source_sha
    proof = {'utc': datetime.now(timezone.utc).isoformat(), 'codeCommit': expected,
        'loadedOnly': ['SurgicalEngine._execute', '_microstep_checkpoint'], 'actions': count,
        'stateIdBefore': before['stateId'], 'stateIdAfter': engine.state_id,
        'oldExecuteMatchesFrozenCommit': previous,
        'oldExecuteSemanticCodeSha256': hashlib.sha256(marshal.dumps(normalized(old_code))).hexdigest(),
        'newExecuteSemanticCodeSha256': hashlib.sha256(marshal.dumps(normalized(new_code))).hexdigest(),
        'savedBoundaryEqualsLiveSnapshot': True, 'directorMetadataAndCodeUnchanged': True,
        'otherEngineMethodCodeIdentitiesUnchanged': [name for name in methods if name != '_execute'],
        'sourceUnchanged': True, 'exactDetachedCurrentStateJsonOrderTypes': True, 'exactFloatBits': True,
        'copiedStateJsonSha256': hashlib.sha256(exact_json.encode()).hexdigest(),
        'floatBitDigest': hashlib.sha256(''.join(floats(candidate_copy)).encode()).hexdigest(),
        'removedAndDebrisIndependentlyOwned': True, 'existingComponentExemptionPreserved': True,
        'patientActionsDuringReload': 0, 'checkpoint': str(checkpoint),
        'checkpointSha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        'scope': 'Reviewed data-graph-equivalent rollback call/helper replacement only. No module reload, patient replay, timestep/model/protection change or whole-episode speed claim.'}
    proof_path.write_text(json.dumps(proof, indent=2) + '\n')
    pause.unlink()
    print(json.dumps({'exactRollbackCopyLoaded': proof}), flush=True)

    advance(1, slice_actions=2)
    after = engine.snapshot()
    new_action_count = len(engine.actions) - count
    assert new_action_count > 0
    assert len(engine.history[count:]) == new_action_count
    assert all(row['status'] in ('accepted', 'partial') for row in engine.history[count:])
    saved = load_workspace(destination)[1]['surgical_toolbox']
    assert saved['branches'][branch_id] == after and saved['fullCase'][branch_id] == director
    assert engine.actions[:count] == before['actions'] and director['lastStateId'] == engine.state_id
    previous_state = before['stateId']
    for row in engine.history[count:]:
        assert row['stateBefore'] == previous_state
        previous_state = row['stateAfter']
    assert previous_state == engine.state_id
    assert array_digest(scene.labels) == label_hash and array_digest(case.mri) == image_hash
    assert hashlib.sha256(original.read_bytes()).hexdigest() == source_sha
    proof['firstActualPostReloadBatch'] = {'utc': datetime.now(timezone.utc).isoformat(),
        'startPosition': count, 'endPosition': len(engine.actions), 'endStateId': engine.state_id,
        'newActionCount': new_action_count,
        'continuousRecordedStateChain': True, 'savedSnapshotEqualsLive': True,
        'statuses': dict(Counter(row['status'] for row in engine.history[count:])),
        'targetRemovedMm3': engine.metrics()['targetRemovedMm3'],
        'sourceUnchanged': True, 'scope': 'New ordinary public preview/step interval; no legacy patient replay.'}
    proof_path.write_text(json.dumps(proof, indent=2) + '\n')
    receipt()
    print(json.dumps({'firstRollbackCopyIntervalVerified': proof['firstActualPostReloadBatch']}), flush=True)


_load_exact_rollback_copy()
if director['status'] == 'running':
    advance(4000)
