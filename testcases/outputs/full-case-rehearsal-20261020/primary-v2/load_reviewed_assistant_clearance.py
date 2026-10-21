"""Proposal-only reload; the next normal slice must execute assistant retirement."""
def _load_reviewed_assistant_clearance():
    import os
    from datetime import datetime,timezone
    saved=load_workspace(destination)[1]['surgical_toolbox']
    assert director['lastStateId']==engine.state_id
    assert saved['branches'][branch_id]==engine.snapshot()
    assert saved['fullCase'][branch_id]==director
    before=engine.snapshot()
    methods=('step','preview','_execute','_substep','_physical_substep','cells','occupancy')
    codes={name:getattr(type(engine),name).__code__ for name in methods}
    assert array_digest(scene.labels)==label_hash and array_digest(case.mri)==image_hash
    assert hashlib.sha256(original.read_bytes()).hexdigest()==source_sha
    checkpoint=destination.parent/'checkpoints'/('primary-before-assistant-clearance-'+str(len(engine.actions))+'.ressectionlab')
    assert not checkpoint.exists()
    os.link(destination,checkpoint)
    for name in ('full_case_assist','full_case_rehearsal'):
        importlib.reload(importlib.import_module('resectionlab.'+name))
    assert before==engine.snapshot() and saved['fullCase'][branch_id]==director
    assert all(getattr(type(engine),name).__code__ is code for name,code in codes.items())
    observed=rehearsal._assistant_clearance(engine,director)
    assert observed is not None
    proof={'utc':datetime.now(timezone.utc).isoformat(),'loadedOnly':['full_case_assist','full_case_rehearsal'],
        'codeCommit':'1ae5f004','actions':len(engine.actions),'stateIdBefore':before['stateId'],'stateIdAfter':engine.state_id,
        'nativeBoundaryEqualsLiveSnapshot':True,'directorMetadataUnchanged':True,
        'engineMethodCodeIdentitiesUnchanged':list(methods),'noPhysicsReplay':True,'sourceUnchanged':True,
        'observedCoordinationTrigger':observed,'checkpoint':str(checkpoint),
        'checkpointSha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        'qualification':'Proposal-only reload at genuine saved boundary; actual retirement must occur in following public actions.'}
    (destination.parent/'assistant-clearance-reload-proof.json').write_text(json.dumps(proof,indent=2)+'\n')
    receipt()
    print(json.dumps({'assistantClearanceReloadPassed':True,'actions':len(engine.actions),
        'stateId':engine.state_id,'observedTrigger':observed}),flush=True)
_load_reviewed_assistant_clearance()
