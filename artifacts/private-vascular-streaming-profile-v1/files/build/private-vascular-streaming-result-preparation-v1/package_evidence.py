"""Copy an explicit small source/JSON/text inventory; no arrays or reruns."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT/'build/private-vascular-streaming-result-integration-v1'
sources = []
def add(folder, names):
    sources.extend(folder+'/'+name for name in names)
add('build/private-vascular-streaming-preparation-v1', [
    'streaming_contact.py', 'profile_generated.py', 'launch_profile.py',
    'test_generated.py', 'test_profile_boundary.py', 'test_launch_profile.py',
    'repository-source-pins.json', 'prior-repository-source-pins-v1.json', 'source-rebind-v2.json',
    'supervisor-disabled-release.json', 'supervisor-author-controls.json', 'root-supervisor-release.json',
    'root-preflight.json', 'RESULT.txt', 'SUPERVISION.txt', 'REVIEWED-HANDOFF.txt'])
add('build/private-vascular-streaming-profile-v1', [
    'receipt.json', 'readout-console.txt', 'worker-release.json', 'worker/attempt.json', 'worker/profile.json'])
add('build/private-vascular-streaming-independent-v1', [
    'REPORT.md', 'run-receipt.json', 'boundary-run-receipt.json', 'test_independent.py', 'test_profile_independent.py'])
add('build/private-vascular-streaming-launch-independent-v1', [
    'REPORT.md', 'run-receipt.json', 'test_independent.py'])
add('build/private-vascular-streaming-result-independent-v1', [
    'REPORT.md', 'audit-result.json', 'audit_saved.py', 'source-and-evidence-snapshot.json'])
add('build/private-vascular-streaming-result-preparation-v1', [
    'RESULT.txt', 'summary.json', 'summarize_saved.py', 'package_evidence.py'])
add('src/resectionlab', ['independent_geometry_batch.py', 'evaluation.py'])
add('scripts', ['mechanics_hbe_v5_remaining_one_shot.py', 'mechanics_hbe_v5_n8_one_shot.py', 'febio_runtime.py'])
add('build/menichetti-structure-launch-preparation-v1', ['launch.py'])
add('build/menichetti-structure-launch-independent-v1', ['REPORT.md'])
assert len(sources) == len(set(sources))
assert not DEST.exists()
DEST.mkdir()
records = []
for name in sources:
    source = ROOT/name
    assert source.is_file() and not source.is_symlink()
    assert source.suffix in {'.py', '.txt', '.json', '.md'} and source.stat().st_size <= 1024**2
    raw = source.read_bytes()
    target = DEST/'files'/name
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('xb') as stream: stream.write(raw)
    assert target.read_bytes() == raw == source.read_bytes()
    records.append({'original_path': name, 'package_path': str(target.relative_to(DEST)),
                    'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()})
index = {'schema': 'generated-streaming-profile-compact-evidence-v1',
         'scope': 'Exact reviewed source, root release, saved generated profile and audits; no evaluation run by packaging.',
         'source_commit': 'abbc3e31ef2dd9123e470a31b3701eb375f85424',
         'file_count': len(records), 'file_bytes': sum(r['bytes'] for r in records),
         'volumetric_data_or_model_arrays_included': False,
         'small_generated_capsule_and_affine_metadata_included': True,
         'scientific_runtime_binaries_included': False,
         'standalone_runtime_bundle': False,
         'copy_verification': 'Every copy byte-compared to its explicit original before and after writing.',
         'files': records}
path = DEST/'source-index.json'
with path.open('x') as stream: json.dump(index, stream, indent=2, sort_keys=True); stream.write('\n')
print(json.dumps({'package': str(DEST.relative_to(ROOT)), 'files': len(records), 'file_bytes': index['file_bytes'],
                  'index_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}, sort_keys=True))
