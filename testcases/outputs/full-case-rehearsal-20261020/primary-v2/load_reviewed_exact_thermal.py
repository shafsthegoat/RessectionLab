"""Load a reviewed numerically equivalent thermal kernel at a saved boundary."""
def _load_reviewed_exact_thermal():
    import ast
    import copy
    import os
    import subprocess
    import types

    expected = subprocess.check_output(['git', 'rev-parse', '3e0bd07c'], cwd=ROOT, text=True).strip()
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip() == expected
    relative = 'src/resectionlab/surgical_drilling_v2.py'
    committed = subprocess.check_output(['git', 'show', expected + ':' + relative], cwd=ROOT)
    assert (ROOT / relative).read_bytes() == committed
    module = importlib.import_module('resectionlab.surgical_drilling_v2')
    old_thermal = module._thermal
    old_dose_code = module._dose_integral.__code__
    legacy = subprocess.check_output(['git', 'show', '5d05ddb3:' + relative], cwd=ROOT)
    compiled = compile(legacy, module.__file__, 'exec', dont_inherit=True)
    reference = next(value for value in compiled.co_consts if isinstance(value, types.CodeType) and value.co_name == '_thermal')
    assert old_thermal.__code__ == reference, 'Live kernel must be the audited pre-change implementation'
    before = engine.snapshot()
    before_director = copy.deepcopy(director)
    assert director['lastStateId'] == engine.state_id
    saved = load_workspace(destination)[1]['surgical_toolbox']
    saved_count = len(saved['branches'][branch_id]['actions'])
    assert saved['branches'][branch_id]['actions'] == engine.actions[:saved_count]
    if saved['branches'][branch_id] != before:
        assert saved_count < len(engine.actions)
        persist(case, artifacts, workspace, engine, director, destination, branch_id, parent)
        saved = load_workspace(destination)[1]['surgical_toolbox']
    assert saved['branches'][branch_id] == before and saved['fullCase'][branch_id] == director
    assert hashlib.sha256(original.read_bytes()).hexdigest() == source_sha
    assert array_digest(scene.labels) == label_hash and array_digest(case.mri) == image_hash
    methods = {name: getattr(type(engine), name).__code__ for name in
        ('step', 'preview', '_execute', '_substep', '_physical_substep', 'cells', 'occupancy')}
    director_code = rehearsal.run_slice.__code__
    checkpoint = destination.parent / 'checkpoints' / ('primary-before-exact-thermal-' + str(len(engine.actions)) + '.ressectionlab')
    assert not checkpoint.exists()
    os.link(destination, checkpoint)
    importlib.reload(module)
    assert module._dose_integral.__code__ == old_dose_code
    probes = []
    existing = list(engine.state['drillingV2']['temperatureC'])
    powers = {key: float(i + 1) / 1000. for i, key in enumerate(existing[-3:])}
    for dt, supplied in ((0., {}), (.1, {}), (.02, powers)):
        old = types.SimpleNamespace(scene=scene, state=copy.deepcopy(before['checkpoint']['state']))
        new = types.SimpleNamespace(scene=scene, state=copy.deepcopy(before['checkpoint']['state']))
        old_thermal(old, dt, supplied)
        module._thermal(new, dt, supplied)
        first = json.dumps(old.state, separators=(',', ':'), allow_nan=False).encode()
        second = json.dumps(new.state, separators=(',', ':'), allow_nan=False).encode()
        assert first == second
        probes.append({'dt': dt, 'powerCells': len(supplied), 'exactStateJsonSha256': hashlib.sha256(first).hexdigest()})
    assert engine.snapshot() == before and director == before_director
    assert all(getattr(type(engine), name).__code__ is code for name, code in methods.items())
    assert rehearsal.run_slice.__code__ is director_code
    assert hashlib.sha256(original.read_bytes()).hexdigest() == source_sha
    assert array_digest(scene.labels) == label_hash and array_digest(case.mri) == image_hash
    # Reload only the ignored runner function definition, preserving its first
    # 908-action save-interval proof and all patient/director metadata.
    runner = (destination.parent / 'use_bounded_headless_batches.py').read_text()
    definition = next(node for node in ast.parse(runner).body if isinstance(node, ast.FunctionDef) and node.name == 'advance')
    exec(compile(ast.get_source_segment(runner, definition), '<bounded-live-runner>', 'exec'), globals())
    proof = {'utc': datetime.now(timezone.utc).isoformat(), 'codeCommit': expected,
        'loadedOnly': ['surgical_drilling_v2', 'ignored headless advance function definition'],
        'actions': len(engine.actions), 'stateIdBefore': before['stateId'], 'stateIdAfter': engine.state_id,
        'savedBoundaryEqualsLiveSnapshot': True, 'directorMetadataAndCodeUnchanged': True,
        'engineMethodCodeIdentitiesUnchanged': list(methods), 'sourceUnchanged': True,
        'exactDetachedCurrentStateKernelProbes': probes, 'patientActionsDuringReload': 0,
        'flushedCompletedUnsavedActions': len(engine.actions) - saved_count,
        'checkpoint': str(checkpoint), 'checkpointSha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        'scope': 'Numerically equivalent thermal-module reload only; full patient replay and GUI acceptance remain pending.'}
    (destination.parent / 'exact-thermal-live-reload-proof.json').write_text(json.dumps(proof, indent=2) + '\n')
    receipt()
    print(json.dumps({'exactThermalReloadPassed': True, 'actions': len(engine.actions),
        'stateId': engine.state_id, 'targetRemovedMm3': engine.metrics()['targetRemovedMm3']}), flush=True)


_load_reviewed_exact_thermal()
