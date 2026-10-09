"""Independent small-evidence checks only; never opens predecessor outputs."""
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).parent
BASE = ROOT / 'build/hbe-v5-n12-preflight-memory-diagnostic-v1'
PACKAGE = BASE / 'integration/artifacts/hbe-v5-n12-preflight-memory-v1'
inputs = {}
small_file_bytes_read = 0
historical_git_source_bytes_read = 0

def bounded(path, cap=128 * 1024):
    global small_file_bytes_read
    assert not path.is_symlink() and path.is_file()
    before = path.stat()
    assert before.st_size <= cap
    raw = path.read_bytes()
    small_file_bytes_read += len(raw)
    after = path.stat()
    assert (before.st_ino, before.st_size, before.st_mtime_ns) == (after.st_ino, after.st_size, after.st_mtime_ns)
    inputs[str(path.relative_to(ROOT))] = {'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}
    return raw

def doc(path):
    return json.loads(bounded(path))

summary = doc(PACKAGE / 'summary.json')
compact = doc(PACKAGE / 'snapshots.json')
bounded(PACKAGE / 'README.md')
bounded(BASE / 'RESULT_ANALYSIS.md')
receipt = doc(BASE / 'attempt-01/receipt.json')
worker = doc(BASE / 'attempt-01/worker-result.json')
raw_samples = doc(BASE / 'attempt-01/worker-samples.json')
assert len(raw_samples) == worker['samples'] == summary['read_only_diagnostic']['source_samples'] == 24
assert compact['source_snapshot_sha256'] == inputs[str((BASE / 'attempt-01/worker-samples.json').relative_to(ROOT))]['sha256']
for binding in summary['raw_evidence_bindings'].values():
    assert inputs[binding['path']]['sha256'] == binding['sha256']
for hold in summary['historical_preflights']:
    content = doc(ROOT / hold['source_path'])
    assert inputs[hold['source_path']]['sha256'] == hold['source_sha256']

counter_keys = {'pageins':'Pageins', 'file_backed_pages':'File-backed pages',
                'anonymous_pages':'Anonymous pages', 'compressor_occupied_pages':'Pages occupied by compressor',
                'compressor_stored_pages':'Pages stored in compressor', 'pageouts':'Pageouts'}
def parse_vm(sample):
    text = sample['vm_stat']['stdout']
    assert sample['vm_stat']['exit_code'] == 0
    assert int(re.search(r'page size of (\d+) bytes', text)[1]) == compact['page_size_bytes'] == 16384
    return {key:int(re.search(r'^'+re.escape(label)+r':\s+(\d+)\.',text,re.M)[1]) for key,label in counter_keys.items()}

for sample in compact['selected_snapshots']:
    original = raw_samples[sample['index']]
    for key in ['label','elapsed_seconds','direct_kernel_host','process_rss_bytes']:
        assert sample[key] == original[key], (sample['index'], key)
    assert sample['vm_pages'] == parse_vm(original)
assert [x['index'] for x in compact['selected_snapshots']] == [0,2,7,16,21,23]
before,after = raw_samples[7],raw_samples[16]
assert before['label'] == 'before_predecessor_chain_hashing'
assert after['label'] == 'after_predecessor_chain_hashing'
delta = {key:parse_vm(after)[key]-parse_vm(before)[key] for key in counter_keys}
expected_delta = {'pageins':173521, 'file_backed_pages':139470, 'anonymous_pages':-283375,
                  'compressor_occupied_pages':153949, 'compressor_stored_pages':273403, 'pageouts':221}
assert delta == expected_delta
assert [before['direct_kernel_host']['available_percent'], after['direct_kernel_host']['available_percent']] == [55,40]
assert [before['direct_kernel_host']['kernel_pressure_mask'],after['direct_kernel_host']['kernel_pressure_mask']] == [1,2]
assert after['direct_kernel_host']['swap_used_bytes'] == before['direct_kernel_host']['swap_used_bytes'] == 7356481536
assert raw_samples[21]['direct_kernel_host']['available_percent'] == raw_samples[23]['direct_kernel_host']['available_percent'] == 42
assert raw_samples[21]['direct_kernel_host']['kernel_pressure_mask'] == raw_samples[23]['direct_kernel_host']['kernel_pressure_mask'] == 2
assert summary['source_commit'] == worker['source_commit'] == receipt['head_required']
assert summary['adapted_deck_sha256'] == worker['adapted_deck_sha256'] == worker['expected_adapted_deck_sha256']
assert summary['template_sha256'] == receipt['template_sha256']
assert receipt['status'] == 'completed_read_only_diagnostic'
assert receipt['readout_stage']['status'] == 'completed_within_caps'
assert receipt['owned_worker_cleanup'] == {'contained': True,'direct_child_reaped': True,'errors': [],'exit_code':0,'fallback_used':False,'remaining_members':[]}
assert receipt['native_calls'] == receipt['hbe_readout_calls'] == worker['native_calls'] == 0
assert receipt['release_written'] == receipt['hbe_output_reserved'] == worker['release_written'] == worker['target_reserved'] == False
assert receipt['caps'] == {'aggregate_output_bytes':4194304,'attempts':1,'numerical_threads':1,'outer_wall_seconds':150,'sampled_process_group_rss_bytes':3221225472,'worker_wall_seconds':147}
assert receipt['outer_elapsed_seconds'] < 150 and receipt['readout_stage']['elapsed_seconds'] < 147
assert receipt['readout_stage']['peak_sampled_process_group_rss_bytes'] == 143982592
assert receipt['readout_stage']['peak_sampled_process_group_rss_bytes'] < 3221225472
files = {p.name:p.stat().st_size for p in (BASE/'attempt-01').iterdir()}
assert set(files) == {'receipt.json','worker-result.json','worker-samples.json','readout-console.txt'}
closed = sum(files.values())
assert closed == 63869 and closed < 4194304
manifest = doc(PACKAGE / 'MANIFEST.json')
assert len(manifest['files']) == 9
for entry in manifest['files']:
    path = PACKAGE/entry['path']
    value = bounded(path)
    assert len(value) == entry['bytes']
    assert hashlib.sha256(value).hexdigest() == entry['sha256']
for name in files:
    assert bounded(PACKAGE/'attempt-01'/name) == bounded(BASE/'attempt-01'/name)
assert bounded(PACKAGE/'diagnose.py') == bounded(BASE/'diagnose.py')
assert bounded(PACKAGE/'test_generated_cleanup.py') == bounded(BASE/'test_generated_cleanup.py')

template_path = ROOT/'build/hbe-v5-tension-n12-s60-release-prep/RELEASE_TEMPLATE.json'
template = doc(template_path)
assert inputs[str(template_path.relative_to(ROOT))]['sha256'] == summary['template_sha256']
assert template['status'] == 'TEMPLATE_NOT_RELEASED'
assert not (ROOT/template['output_directory']).exists()
source_matches = {}
assert len(template['source_bindings']) == 20
for path,binding in template['source_bindings'].items():
    current = bounded(ROOT/path)
    historical = subprocess.run(['/usr/bin/git','show',summary['source_commit']+':'+path],cwd=ROOT,capture_output=True,check=True,timeout=5).stdout
    historical_git_source_bytes_read += len(historical)
    assert hashlib.sha256(current).hexdigest() == hashlib.sha256(historical).hexdigest() == binding['sha256']
    source_matches[path] = binding['sha256']
diagnostic = bounded(BASE/'diagnose.py')
assert hashlib.sha256(diagnostic).hexdigest() == '8684aa673e8e0189bd6c5fdfab144e7ff6d8a33af5e16047caa5a88b196dce16'
sampler = bounded(ROOT/'build/limited-input-guard-design/gliomoda-tile1-parity-64-v2/darwin_fast_sampler.py')
assert hashlib.sha256(sampler).hexdigest() == '2ff8f3e4d30ed0a8976fb01da99403ff5b2b2ffd6e750791fe7a954946d0a00f'
head = subprocess.run(['/usr/bin/git','rev-parse','HEAD'],cwd=ROOT,capture_output=True,text=True,check=True,timeout=5).stdout.strip()
result = {'status':'passed_small_saved_evidence_audit','review_head':head,'historical_source_commit':summary['source_commit'],
          'raw_to_compact_snapshots_exact':6, 'raw_samples':24,'outer_interval_seconds':after['elapsed_seconds']-before['elapsed_seconds'],
          'outer_vm_delta_pages':delta,'outer_rss_delta_bytes':after['process_rss_bytes']-before['process_rss_bytes'],
          'receipt_recorded_output_snapshot_bytes':receipt['final_retained_output_bytes'],
          'independent_closed_output_bytes':closed,'closed_files':files,
          'compact_manifest_entries_verified':9,'copied_raw_and_source_files_byte_exact':6,
          'twenty_sources_match_current_and_historical':source_matches,
          'full_predecessor_files_read':0,'native_calls':0,'hbe_replays':0,'patient_or_measured_data_reads':0,
          'input_bindings':inputs,'unique_bound_input_bytes':sum(x['bytes'] for x in inputs.values()),
          'small_file_bytes_read_including_repeats':small_file_bytes_read,
          'historical_git_source_bytes_read':historical_git_source_bytes_read,
          'decision':'GO_compact_evidence_integration_HOLD_native_release'}
(OUT/'saved-evidence-audit.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('input_bindings','twenty_sources_match_current_and_historical')},sort_keys=True))
