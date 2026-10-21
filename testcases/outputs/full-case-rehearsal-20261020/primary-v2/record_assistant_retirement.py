"""Record actual public retirement outcomes, then continue the same patient engine."""
def _record_assistant_retirement():
    from datetime import datetime,timezone
    start=json.loads((destination.parent/'assistant-clearance-reload-proof.json').read_text())['actions']
    rows=engine.history[start:start+3]
    assert len(rows)==3 and [row['action']['operation'] for row in rows]==['deactivate','withdraw','withdraw']
    assert all(row['action']['slot']=='right' and row['status']=='accepted' and not row['deltas']['removedCells'] for row in rows)
    assert all(first['stateAfter']==second['stateBefore'] for first,second in zip(rows,rows[1:]))
    right=engine.state['slots']['right']
    assert not right['active'] and not right['inserted'] and not right['grip'] and 'right' not in engine.state['holds']
    assert director['assistantParked'] and director['status']=='running'
    assert load_workspace(destination)[1]['surgical_toolbox']['branches'][branch_id]==engine.snapshot()
    assert array_digest(scene.labels)==label_hash and array_digest(case.mri)==image_hash
    assert hashlib.sha256(original.read_bytes()).hexdigest()==source_sha
    value={'utc':datetime.now(timezone.utc).isoformat(),'actionsBefore':start,'actionsAfter':len(engine.actions),
        'codeCommit':'1ae5f004','sourceSceneId':scene.identity,'sourceUnchanged':True,
        'coordinationEvent':director['coordinationEvents'][-1],
        'retirementActions':[{'position':start+i+1,'operation':row['action']['operation'],
            'status':row['status'],'reason':row.get('reason'),'completedFraction':row['completedFraction'],
            'stateBefore':row['stateBefore'],'stateAfter':row['stateAfter'],
            'removedSourceCells':len(row['deltas']['removedCells'])} for i,row in enumerate(rows)],
        'rightStateAfter':right,'resumedStage':director['stage'],'targetRemovedMm3':engine.metrics()['targetRemovedMm3'],
        'scope':'Three actual preview/step-matched public right-hand actions; no source tissue removed. Final canonical/GUI verification remains pending.'}
    (destination.parent/'assistant-retirement-execution-proof.json').write_text(json.dumps(value,indent=2)+'\n')
    receipt()
    print(json.dumps({'actualAssistantRetirementPassed':True,'actions':len(engine.actions),
        'rightInactiveWithdrawnAndParked':True,'stage':director['stage'],'targetRemovedMm3':engine.metrics()['targetRemovedMm3']}),flush=True)
_record_assistant_retirement()
advance(4000)
