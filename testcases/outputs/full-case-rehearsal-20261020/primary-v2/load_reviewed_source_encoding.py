"""Load reviewed save-only code at an exact genuine patient action boundary."""
def _load_reviewed_source_encoding():
    from datetime import datetime, timezone
    import os
    from zipfile import ZipFile
    saved = load_workspace(destination)[1]['surgical_toolbox']
    before = engine.snapshot()
    assert saved['branches'][branch_id] == before
    assert saved['fullCase'][branch_id] == director
    assert director['lastStateId'] == engine.state_id
    assert hashlib.sha256(original.read_bytes()).hexdigest() == source_sha
    assert array_digest(scene.labels) == label_hash and array_digest(case.mri) == image_hash
    methods = ('step','preview','_execute','_substep','_physical_substep','cells','occupancy')
    method_codes = {name:getattr(type(engine),name).__code__ for name in methods}
    director_code = rehearsal.run_slice.__code__
    checkpoint = destination.parent/'checkpoints'/('primary-before-source-array-cache-'+str(len(engine.actions))+'.ressectionlab')
    assert not checkpoint.exists()
    os.link(destination,checkpoint)
    with ZipFile(destination) as archive:
        array_payload_sha = hashlib.sha256(archive.read('arrays.npz')).hexdigest()
    for name in ('resectionlab.case_array_payload','resectionlab.imaging'):
        if name in sys.modules:
            importlib.reload(sys.modules[name])
        else:
            importlib.import_module(name)
    timings = []
    for _ in range(2):
        saved_started = time.perf_counter()
        persist(case,artifacts,workspace,engine,director,destination,branch_id,parent)
        timings.append(time.perf_counter()-saved_started)
        with ZipFile(destination) as archive:
            assert hashlib.sha256(archive.read('arrays.npz')).hexdigest() == array_payload_sha
        reopened,decoded,saved_workspace = load_workspace(destination)
        assert decoded == artifacts
        assert decoded['surgical_toolbox']['branches'][branch_id] == before
        assert decoded['surgical_toolbox']['fullCase'][branch_id] == director
        assert reopened.semantic_hash == case.semantic_hash and array_digest(reopened.mri) == image_hash
        assert saved_workspace['originals'] == workspace['originals']
    assert before == engine.snapshot()
    assert all(getattr(type(engine),name).__code__ is code for name,code in method_codes.items())
    assert rehearsal.run_slice.__code__ is director_code
    assert array_digest(scene.labels) == label_hash and array_digest(case.mri) == image_hash
    assert hashlib.sha256(original.read_bytes()).hexdigest() == source_sha
    proof = {'utc':datetime.now(timezone.utc).isoformat(),'loadedOnly':['case_array_payload','imaging'],
        'codeCommit':'9b1ce926','actions':len(engine.actions),'stateIdBefore':before['stateId'],
        'stateIdAfter':engine.state_id,'nativeBoundaryEqualsLiveSnapshot':True,
        'flushedUnpersistedCompletedActionsBeforeReload':len(engine.actions)-len(saved_pause_record['branches'][branch_id]['actions']),
        'engineMethodCodeIdentitiesUnchanged':list(methods),'activeDirectorCodeIdentityUnchanged':True,
        'saveOnlyWallSecondsColdWarm':timings,'npzPayloadSha256Unchanged':array_payload_sha,
        'allArtifactsSourceAndOriginalFilesUnchanged':True,'noPhysicsReplay':True,
        'checkpoint':str(checkpoint),'checkpointSha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        'qualification':'Save-only module reload and two unchanged-state native roundtrips; no whole-action speed claim or final canonical/GUI acceptance.'}
    (destination.parent/'source-array-cache-live-reload-proof.json').write_text(json.dumps(proof,indent=2)+'\n')
    receipt()
    print(json.dumps({'saveOnlyReloadPassed':True,'actions':len(engine.actions),'stateId':engine.state_id,
        'saveColdWarmSeconds':timings,'targetRemovedMm3':engine.metrics()['targetRemovedMm3']}),flush=True)
_load_reviewed_source_encoding()
