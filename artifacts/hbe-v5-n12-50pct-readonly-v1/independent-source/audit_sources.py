"""Small source/metadata audit only; never validate or open predecessor outputs."""
import hashlib
import json
from pathlib import Path
import subprocess
from types import ModuleType

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
CANDIDATE = ROOT / 'build/hbe-v5-n12-nocache-50pct-probe-v1'
EXPECTED = {
    'probe.py': 'e5c1734bb70ac20b1dd5f4a09c4c98df85e5525c237978bf8b2426b1dcaf9181',
    'test_generated_probe.py': '5961f652653b0cffeb43c7ad08c8a932151bc7406cc93c334ab98c471b84b9cb',
    'PROTOCOL.md': 'fe6bb7182f951169ee601e9fe9e3a41409c2e7f1bc6b0e145ed128858ee79807',
}
def digest(raw): return hashlib.sha256(raw).hexdigest()
def binding(path, expected=None):
    assert not path.is_symlink() and path.is_file(), str(path)
    assert path.stat().st_size <= 1024**2, str(path)
    raw = path.read_bytes()
    observed = digest(raw)
    if expected is not None: assert observed == expected, str(path)
    return {'path': str(path.relative_to(ROOT)), 'sha256': observed, 'bytes': len(raw)}

head = subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT,text=True).strip()
files = {name:binding(CANDIDATE/name,expected) for name,expected in EXPECTED.items()}
raw = (CANDIDATE/'probe.py').read_bytes()
probe = ModuleType('independent_source_only_probe50')
probe.__file__ = str(CANDIDATE/'probe.py')
exec(compile(raw,probe.__file__,'exec'),probe.__dict__)
closure = probe.selected_sources(head)
compare, diagnostic, hint = probe.load_dependencies()
deps = [binding(probe.COMPARE,probe.COMPARE_SHA),binding(probe.DIAGNOSTIC,probe.DIAGNOSTIC_SHA),
        binding(probe.HINT,probe.HINT_SHA),binding(probe.TEMPLATE,probe.TEMPLATE_SHA),
        binding(diagnostic.HOST_SAMPLER,diagnostic.HOST_SAMPLER_SHA)]
template = json.loads(probe.TEMPLATE.read_bytes())
sources = []
for relative, expected in template['source_bindings'].items():
    item = binding(ROOT/relative,expected['sha256'])
    blob = subprocess.check_output(['git','show',head+':'+relative],cwd=ROOT)
    assert digest(blob) == item['sha256']
    item['matches_selected_committed_blob'] = True
    sources.append(item)
new = []
for name,expected in {
    'hbe_v5_nocache_v1.py':'efeda27abc3c1a6851e1066789e51a2f05b94728def62e71061d03da7f4629de',
    'hbe_v5_nocache_hash_v1.py':'96c71d0d9b8d896b713aef7305e3e86d5b12b9d2dca5fcc55515f6b2eb9cf3fc',
    'hbe_v5_nocache_host_v1.py':'2ff8f3e4d30ed0a8976fb01da99403ff5b2b2ffd6e750791fe7a954946d0a00f',
}.items():
    item=binding(ROOT/'launchers'/name,expected)
    assert digest(subprocess.check_output(['git','show',head+':'+item['path']],cwd=ROOT))==expected
    new.append(item)
sidecar='outputs/mechanics/hbe-v5-nocache-policy-v1/08-tension-N12-S60-reference/attempt-01'
absent={}
for path in [probe.TARGET,ROOT/template['output_directory'],ROOT/sidecar]:
    assert not path.exists() and not path.is_symlink(), str(path)
    absent[str(path.relative_to(ROOT))]=True
post=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
assert post==head
assert probe.self_sha()==EXPECTED['probe.py']
record={'status':'source_generated_review_passed_no_actual_probe', 'review_head':head,
 'candidate_files':files, 'pinned_dependencies':deps, 'old_source_closure':closure,
 'old20_working_and_git_bindings':sources,'unchanged_committed_v1_launcher_sources':new,
 'absent_targets':absent, 'source_guard_bytes_read_per_full_binding_pass':sum(x['bytes'] for x in sources),
 'tested':{'candidate_generated_controls':10,'candidate_seconds':0.012,
           'independent_real_supervisor_fake_child_controls':4,'independent_seconds':0.218},
 'actual_probe_calls':0,'native_calls':0,'hbe_readout_calls':0,'predecessor_payload_files_opened':0,
 'limits':{'outer_seconds':diagnostic.WALL_CAP,'worker_seconds':diagnostic.WORKER_WALL_CAP,
   'sampled_worker_group_rss_bytes':diagnostic.RSS_CAP,'output_bytes':diagnostic.OUTPUT_CAP,
   'initial_min_available_percent':probe.INITIAL_FLOOR,'observed_ongoing_min_available_percent':probe.UNCHANGED_NATIVE_FLOOR},
 'review_files':{name:binding(HERE/name) for name in ['audit_sources.py','test_incomplete_identity.py']}}
(HERE/'audit.json').write_text(json.dumps(record,indent=2,sort_keys=True)+'\n')
print(json.dumps({'status':record['status'],'head':head,'source_count':len(sources),
 'source_binding_bytes':record['source_guard_bytes_read_per_full_binding_pass'],
 'targets_absent':absent,'audit_sha256':digest((HERE/'audit.json').read_bytes())},sort_keys=True))
