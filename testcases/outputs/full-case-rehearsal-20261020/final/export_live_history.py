"""Read-only export of the preserved original live public-action receipts."""
def _export_original_live_history():
    import hashlib
    import json
    from datetime import datetime, timezone
    path = destination.parent.parent / 'final/live-primary-history.jsonl'
    before, count = engine.state_id, len(engine.actions)
    assert count == len(engine.history) == 2148
    assert before == 'sha256:c97de5651109a57695fc3dd1bd49e504f57e6c0e8b45052d915232df6ba86021'
    last = None
    digest = hashlib.sha256()
    with path.open('xb') as stream:
        for index, (action, outcome) in enumerate(zip(engine.actions, engine.history), 1):
            assert outcome['action'] == action and outcome['status'] in ('accepted', 'partial')
            assert last is None or outcome['stateBefore'] == last
            last = outcome['stateAfter']
            raw = (json.dumps({'position': index, 'outcome': outcome}, separators=(',', ':'), allow_nan=False)+'\n').encode()
            stream.write(raw); digest.update(raw)
    assert last == engine.state_id == before and len(engine.actions) == count
    receipt = {'utc': datetime.now(timezone.utc).isoformat(), 'source': 'Preserved original live engine history; not reconstructed by final canonical replay',
        'actions': count, 'stateId': before, 'history': str(path), 'historySha256': digest.hexdigest(),
        'engineStateAndActionCountUnchanged': True, 'physicalActionsDuringExport': 0}
    with path.with_suffix('.receipt.json').open('x') as stream: json.dump(receipt, stream, indent=2); stream.write('\n')
    print(json.dumps(receipt), flush=True)


_export_original_live_history()
