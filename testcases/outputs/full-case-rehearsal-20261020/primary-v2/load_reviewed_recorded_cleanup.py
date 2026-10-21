"""Load reviewed proposal code only at a fully withdrawn patient endpoint.

Execute in the existing live console after its actual >>> prompt. This loader
does not configure a new cleanup pass or perform any patient action.
"""
def _load_reviewed_recorded_cleanup():
    global rehearsal
    import copy
    import os
    import subprocess

    expected_commit = '0c4045cfe59c4f6fcad11e041e982baf22ba9193'
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip() == expected_commit
    for relative in ('src/resectionlab/full_case_cleanup.py', 'src/resectionlab/full_case_rehearsal.py'):
        committed = subprocess.check_output(['git', 'show', expected_commit + ':' + relative], cwd=ROOT)
        assert (ROOT / relative).read_bytes() == committed
    assert director['status'] in ('stopped', 'complete') and director['stage'] == 'endpoint'
    assert director['scenario'] == 'primary' and director['lastStateId'] == engine.state_id
    assert not engine.state['holds'] and all(not state['active'] and not state['inserted'] and not state['grip']
        for state in engine.state['slots'].values())
    before, before_director = engine.snapshot(), copy.deepcopy(director)
    saved = load_workspace(destination)[1]['surgical_toolbox']
    assert saved['branches'][branch_id] == before
    assert saved['fullCase'][branch_id] == before_director
    assert hashlib.sha256(original.read_bytes()).hexdigest() == source_sha
    assert array_digest(scene.labels) == label_hash and array_digest(case.mri) == image_hash
    methods = {name: getattr(type(engine), name).__code__ for name in
        ('step', 'preview', '_execute', '_substep', '_physical_substep', 'cells', 'occupancy')}
    checkpoint = destination.parent / 'checkpoints' / ('primary-before-recorded-ray-cleanup-' + str(len(engine.actions)) + '.ressectionlab')
    if checkpoint.exists():
        assert checkpoint.read_bytes() == destination.read_bytes()
    else:
        os.link(destination, checkpoint)
    for name in ('resectionlab.full_case_cleanup', 'resectionlab.full_case_rehearsal'):
        importlib.reload(importlib.import_module(name))
    rehearsal = importlib.import_module('resectionlab.full_case_rehearsal')
    assert callable(rehearsal.configure_recorded_field_cleanup)
    assert engine.snapshot() == before and director == before_director
    assert all(getattr(type(engine), name).__code__ is code for name, code in methods.items())
    assert hashlib.sha256(original.read_bytes()).hexdigest() == source_sha
    assert array_digest(scene.labels) == label_hash and array_digest(case.mri) == image_hash
    proof = {'utc': datetime.now(timezone.utc).isoformat(), 'codeCommit': expected_commit,
        'loadedOnly': ['full_case_cleanup', 'full_case_rehearsal'], 'actions': len(engine.actions),
        'stateIdBefore': before['stateId'], 'stateIdAfter': engine.state_id,
        'savedBoundaryEqualsLiveSnapshot': True, 'directorMetadataUnchanged': True,
        'engineMethodCodeIdentitiesUnchanged': list(methods), 'sourceUnchanged': True,
        'checkpoint': str(checkpoint), 'checkpointSha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        'configuredRecordedCleanup': False, 'performedPatientActions': 0,
        'qualification': 'Reviewed proposal reload only. Candidate measurement, public admission and actual cleanup remain separate pending work.'}
    (destination.parent / 'recorded-ray-cleanup-reload-proof.json').write_text(json.dumps(proof, indent=2) + '\n')
    receipt()
    print(json.dumps({'recordedCleanupCapabilityLoaded': True, 'actions': len(engine.actions),
        'stateId': engine.state_id, 'patientActionsPerformed': 0}), flush=True)


_load_reviewed_recorded_cleanup()
