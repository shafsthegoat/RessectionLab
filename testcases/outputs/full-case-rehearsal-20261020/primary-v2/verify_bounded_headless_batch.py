"""Verify the first real bounded batch, then continue the same live engine."""
def _verify_bounded_headless_batch():
    from collections import Counter
    start = len(_runner_before['actions'])
    snapshot_now = engine.snapshot()
    saved_now = load_workspace(destination)[1]['surgical_toolbox']
    rows = engine.history[start:]
    assert len(rows) == len(engine.actions) - start == 12
    assert snapshot_now['actions'][:start] == _runner_before['actions']
    assert [row['action'] for row in rows] == engine.actions[start:]
    assert rows[0]['stateBefore'] == _runner_before['stateId']
    assert rows[-1]['stateAfter'] == engine.state_id
    assert all(a['stateAfter'] == b['stateBefore'] for a, b in zip(rows, rows[1:]))
    assert all(row['status'] != 'rejected' for row in rows)
    assert saved_now['branches'][branch_id] == snapshot_now
    assert saved_now['fullCase'][branch_id] == director
    assert all(getattr(type(engine), name).__code__ is code for name, code in _runner_methods.items())
    assert rehearsal.run_slice.__code__ is _runner_director_code
    assert hashlib.sha256(original.read_bytes()).hexdigest() == source_sha
    assert array_digest(scene.labels) == label_hash and array_digest(case.mri) == image_hash
    counts = rehearsal.accounting(engine)
    assert counts['healthyRemovedMm3'] == counts['unknownRemovedMm3'] == 0.
    _runner_proof.update(firstActualBatch={
        'startPosition': start, 'endPosition': len(engine.actions), 'endStateId': engine.state_id,
        'continuousRecordedStateChain': True, 'savedSnapshotEqualsLive': True,
        'statuses': dict(Counter(row['status'] for row in rows)),
        'targetRemovedMm3': engine.metrics()['targetRemovedMm3'],
        'ownedHealthyRemovedMm3': counts['healthyRemovedMm3'],
        'ownedUnknownRemovedMm3': counts['unknownRemovedMm3'],
        'verifiedUtc': datetime.now(timezone.utc).isoformat(),
        'qualification': 'Actual public actions; no second patient physics replay or episode-speed claim. Final canonical and GUI acceptance remain pending.'})
    (destination.parent / 'bounded-headless-batches-proof.json').write_text(json.dumps(_runner_proof, indent=2) + '\n')
    index_path = destination.parent.parent / 'episode-index.json'
    index = json.loads(index_path.read_text())
    index['updatedUtc'] = datetime.now(timezone.utc).isoformat()
    primary = index['episodes']['primary']
    primary.update(recordedStateId=engine.state_id, recordedActionCount=len(engine.actions),
        recordedTargetRemovedMm3=engine.metrics()['targetRemovedMm3'],
        overTwoThirdsCheckpoint=str(destination.parent / 'checkpoints/primary-over-two-thirds-target.ressectionlab'))
    index['notes'][0] = 'Primary file is atomically replaced after each bounded 12-action headless director slice; all public action records remain in the native snapshot.'
    index['runtime'].update(activeCommand='advance(4000)', headlessSliceActionLimit=12,
        headlessBatchProof='primary-v2/bounded-headless-batches-proof.json')
    index['evidence'].update(currentResidualGeometry='geometry/current-residual-902.json',
        twoThirdsMilestone='primary-v2/checkpoints/primary-over-two-thirds-target.provenance.json')
    index_path.write_text(json.dumps(index, indent=2) + '\n')
    print(json.dumps({'firstBoundedBatchVerified': True,
        'actions': len(engine.actions), 'targetRemovedMm3': engine.metrics()['targetRemovedMm3'],
        'statuses': _runner_proof['firstActualBatch']['statuses']}), flush=True)


_verify_bounded_headless_batch()
advance(4000)
