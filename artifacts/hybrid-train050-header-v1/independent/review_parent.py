"""Static parent/index review; no production imports or child execution."""
from pathlib import Path
import ast
import hashlib
import json

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PREP = ROOT/'build/hybrid-train050-preparation-v1'

def pin(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024**2), b''): h.update(b)
    return {'path':str(path.relative_to(ROOT)), 'bytes':path.stat().st_size, 'sha256':h.hexdigest()}

source = PREP/'run_headers_owned.py'
assert pin(source)['sha256'] == 'ecce58401e9a2ba73f4720a5b388c594c87cbb32cfaacde445fd8531b43838bd'
index_path = PREP/'source-index.json'
assert pin(index_path)['sha256'] == '46523b9bc41d5610b52492e9951c2a3b5bbe4226ebb3352c84fa7a3f29123cc7'
index = json.loads(index_path.read_text())
for record in index['files']: assert pin(ROOT/record['path']) == record
release = json.loads((PREP/'prepared-release.json').read_text())
assert release['execution_released'] is False
assert release['source_index'] == pin(index_path)
assert release['public_index'] == pin(PREP/'public-train-index.json')
tree = ast.parse(source.read_text())
functions = {n.name:n for n in tree.body if isinstance(n, ast.FunctionDef)}
baseline_path = ROOT/'build/remind-train-public-qc-preparation-v1/run_public_qc_owned.py'
baseline = {n.name:n for n in ast.parse(baseline_path.read_text()).body if isinstance(n, ast.FunctionDef)}
unchanged = []
for name in ('sha','need','cleanup_phase','final_log_size'):
    assert ast.dump(functions[name]) == ast.dump(baseline[name])
    unchanged.append(name)
phase_loops = [n for n in ast.walk(functions['main']) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id == 'phase']
assert len(phase_loops) == 1 and ast.literal_eval(phase_loops[0].iter) == ('headers',)
assert not (PREP/'header-attempt-01').exists()
result = {
    'status':'GO_EXACT_TRAIN050_HEADER_ONLY_SOURCE_REVIEW',
    'parent':pin(source), 'source_index':pin(index_path), 'prepared_release':pin(PREP/'prepared-release.json'),
    'case_audit':pin(HERE/'exact-case-audit.json'), 'reader_review':pin(HERE/'reader-source-review.json'),
    'all_24_index_bindings_current': len(index['files']) == 24,
    'unchanged_owned_helper_functions':unchanged,
    'single_phase':'headers', 'expected_objects':194, 'expected_source_bytes':27142222,
    'caps':release['caps'],
    'source_review_findings':[
        'Preflight verifies immutable source/index/case/flat bindings, fixed TRAIN050 and exactly three public series before child dispatch.',
        'Only execution_released may change from the frozen false release. No crop phase or automatic retry is reachable in the parent loop.',
        'Terminal acceptance checks expected header status, exact case/reader/patient/role, zero optimizer and pixel-decode calls, no native conversion and no training admission.',
        'Existing retained-Popen cleanup fallback and final output/log/timeout checks remain; receipts retain failure and denominator.',
        'Reader source only extends the public-header allowlist; geometry/decoder and domain-crop gate unchanged.',
        'Owner seven generated controls and metadata check-only logs read as saved evidence; reviewer did not rerun or import them.'
    ],
    'parent_launch_recipe_condition':'Use -I -B and an absolute -X pycache_prefix whose absence and non-symlink status are checked immediately before launch; worker already uses fresh output/fresh-pycache. This prevents ambient helper .pyc from substituting for verified source.',
    'limits':['RSS and output limits are sampled; wall guard excludes preparation and includes bounded cleanup overhead, not a kernel resource limit.', 'Interpreter/source pins and recorded versions do not claim a fully hash-closed third-party environment.', 'Header success is not alignment/coverage/anatomy or training admission; saved metadata findings require review.'],
    'patient_headers_arrays_or_geometry_executed_by_reviewer':False
}
output = HERE/'parent-source-review.json'
with output.open('x') as f:
    json.dump(result,f,indent=2)
    f.write('\n')
print(json.dumps(pin(output)))
