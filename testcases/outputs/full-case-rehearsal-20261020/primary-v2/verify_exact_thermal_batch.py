"""Verify the first actual post-reload batch, release its boundary pause."""
def _verify_exact_thermal_batch():
    from collections import Counter
    proof_path = destination.parent / 'exact-thermal-live-reload-proof.json'
    proof = json.loads(proof_path.read_text())
    start = proof['actions']
    rows = engine.history[start:]
    assert 0 < len(rows) == len(engine.actions) - start <= 12
    assert director['status'] in ('stopped', 'complete') or len(rows) == 12
    assert rows[0]['stateBefore'] == proof['stateIdAfter']
    assert rows[-1]['stateAfter'] == engine.state_id
    assert all(a['stateAfter'] == b['stateBefore'] for a, b in zip(rows, rows[1:]))
    assert [row['action'] for row in rows] == engine.actions[start:]
    assert all(row['status'] in ('accepted', 'partial') for row in rows)
    saved = load_workspace(destination)[1]['surgical_toolbox']
    assert saved['branches'][branch_id] == engine.snapshot()
    assert saved['fullCase'][branch_id] == director
    assert director['lastStateId'] == engine.state_id
    assert hashlib.sha256(original.read_bytes()).hexdigest() == source_sha
    assert array_digest(scene.labels) == label_hash and array_digest(case.mri) == image_hash
    counts = rehearsal.accounting(engine)
    assert counts['healthyRemovedMm3'] == counts['unknownRemovedMm3'] == 0.
    pause = destination.parent / 'pause-after-batch'
    assert pause.is_file()
    proof['firstActualPostReloadBatch'] = {'verifiedUtc': datetime.now(timezone.utc).isoformat(),
        'startPosition': start, 'endPosition': len(engine.actions), 'endStateId': engine.state_id,
        'statuses': dict(Counter(row['status'] for row in rows)),
        'continuousRecordedStateChain': True, 'savedSnapshotEqualsLive': True,
        'targetRemovedMm3': engine.metrics()['targetRemovedMm3'],
        'ownedHealthyRemovedMm3': counts['healthyRemovedMm3'],
        'ownedUnknownRemovedMm3': counts['unknownRemovedMm3'],
        'gracefulSavedBoundaryPausePassed': True,
        'qualification': 'Actual public action interval and saved equality, not a second legacy patient replay or a whole-episode speed claim.'}
    proof_path.write_text(json.dumps(proof, indent=2) + '\n')
    index_path = destination.parent.parent / 'episode-index.json'
    index = json.loads(index_path.read_text())
    index['updatedUtc'] = datetime.now(timezone.utc).isoformat()
    index['codeCommit'] = proof['codeCommit']
    index['episodes']['primary'].update(recordedStateId=engine.state_id,
        recordedActionCount=len(engine.actions), recordedTargetRemovedMm3=engine.metrics()['targetRemovedMm3'],
        directorStatus=director['status'])
    index['evidence'].update(exactThermalReload='primary-v2/exact-thermal-live-reload-proof.json',
        exactThermalKernelAudit='review/thermal-kernel-audit-20261020T153736Z.json')
    index['runtime'].update(liveThermalCommit='3e0bd07c', exactThermalReloadBoundary=start,
        pendingDirectorReload='3e0bd07c', boundaryPauseFile=str(pause), activeCommand='advance(4000)')
    index_path.write_text(json.dumps(index, indent=2) + '\n')
    pause.unlink()
    receipt()
    print(json.dumps({'actualPostThermalBatchVerified': True, 'actions': len(engine.actions),
        'targetRemovedMm3': engine.metrics()['targetRemovedMm3'], 'status': director['status']}), flush=True)


_verify_exact_thermal_batch()
if director['status'] not in ('complete', 'stopped'):
    advance(4000)
