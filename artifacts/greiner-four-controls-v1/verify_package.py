"""Verify this compact archive without native execution or source writes."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parent
sha = lambda raw: hashlib.sha256(raw).hexdigest()
index = json.loads((ROOT / 'artifact-index.json').read_bytes())
actual_names = {str(path.relative_to(ROOT)) for path in ROOT.rglob('*') if path.is_file()}
assert actual_names == set(index) | {'artifact-index.json'}
for name, expected in index.items():
    file = ROOT / name
    assert not file.is_symlink()
    raw = file.read_bytes()
    assert expected == {'sha256': sha(raw), 'bytes': len(raw)}, name
snapshots = json.loads((ROOT / 'snapshot-map.json').read_bytes())
for name, row in snapshots.items():
    raw = (ROOT / name).read_bytes()
    if row['storage'] == 'canonical_json_compact':
        raw = (json.dumps(json.loads(raw), sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    else:
        assert row['storage'] == 'exact_bytes'
    assert sha(raw) == row['original']['sha256'], name
    assert len(raw) == row['original']['bytes'], name
outputs = json.loads((ROOT / 'retained-output-inventory.json').read_bytes())
assert len(outputs['files']) == outputs['total_files'] == 35
assert sum(row['bytes'] for row in outputs['files'].values()) == outputs['total_bytes'] == 1966363
print(json.dumps({'status': 'verified', 'artifact_files': len(index) + 1,
                  'exact_original_snapshots': len(snapshots), 'retained_output_hashes': 35,
                  'native_calls': 0}, sort_keys=True))
