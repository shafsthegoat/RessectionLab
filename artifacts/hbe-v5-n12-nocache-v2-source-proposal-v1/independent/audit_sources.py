"""Prospective v2 exact-source audit; no release or predecessor validation."""
import ast
import difflib
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
CANDIDATE=ROOT/'build/hbe-v5-n12-nocache-v2-candidate'
EXPECTED={
 'hbe_v5_nocache_v2.py':'4271cb49de70a63de5bf8369628208d8b90907434deb6324a16a7b36c314790f',
 'test_generated_v2.py':'cc2b272052d75297642b68c3bd07c169192db11df7d3af02b3087e359b9a06ef',
 'test_v2_boundary.py':'d353e4200d7f1371b31957c1681e23996bfb281d9ce759391e02ba66036fa285',
 'V2_PROPOSAL.md':'c06ce9d0e63c5dfda58d06bfb6caa179bf794aa12ebae2aa2f5042c405e3c59e',
 'hbe_v5_nocache_hash_v1.py':'96c71d0d9b8d896b713aef7305e3e86d5b12b9d2dca5fcc55515f6b2eb9cf3fc',
 'hbe_v5_nocache_host_v1.py':'2ff8f3e4d30ed0a8976fb01da99403ff5b2b2ffd6e750791fe7a954946d0a00f'}
BINDINGS={}
def digest(raw):return hashlib.sha256(raw).hexdigest()
def read(path,expected=None):
 assert path.is_file() and not path.is_symlink() and path.stat().st_size<=1024**2
 raw=path.read_bytes()
 observed=digest(raw)
 if expected is not None:assert observed==expected,str(path)
 BINDINGS[str(path.relative_to(ROOT))]={'sha256':observed,'bytes':len(raw)}
 return raw
head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
for name,expected in EXPECTED.items():read(CANDIDATE/name,expected)
v1=read(ROOT/'launchers/hbe_v5_nocache_v1.py','efeda27abc3c1a6851e1066789e51a2f05b94728def62e71061d03da7f4629de').decode()
v2=read(CANDIDATE/'hbe_v5_nocache_v2.py',EXPECTED['hbe_v5_nocache_v2.py']).decode()
expected=v1
replacements=[
 ('Candidate separate HBE v5 N12 I/O-policy launcher','Prospective separate HBE v5 N12 v2 I/O-policy launcher',1),
 ('launchers/hbe_v5_nocache_v1.py','launchers/hbe_v5_nocache_v2.py',2),
 ('SIDECAR = ("outputs/mechanics/hbe-v5-nocache-policy-v1/"','SIDECAR = ("outputs/mechanics/hbe-v5-nocache-policy-v2/"',1),
 ('darwin_F_NOCACHE_48_read_only_predecessor_hash_v1','darwin_F_NOCACHE_48_read_only_predecessor_hash_v2_initial54',1),
 ('"initial_available_percent_floor": 55','"initial_available_percent_floor": 54',1),
 ('hbe-v5-nocache-launch-envelope-v1','hbe-v5-nocache-launch-envelope-v2',1),
 ('"root_released_one_native_call_with_declared_io_policy"','"root_released_one_native_call_with_declared_io_policy_v2"',1),
 ('hbe-v5-nocache-policy-sidecar-v1','hbe-v5-nocache-policy-sidecar-v2',1),
 ('or host_before["available_percent"] < 55','or host_before["available_percent"] < POLICY["initial_available_percent_floor"]',1),
 ('Initial host below declared 55%/normal preflight floor','Initial host below declared v2 54%/normal preflight floor',1),
 ('"passed_numerical_software_only_with_declared_io_policy"','"passed_numerical_software_only_with_declared_io_policy_v2"',2),
]
for old,new,count in replacements:
 assert expected.count(old)==count,(old,expected.count(old))
 expected=expected.replace(old,new)
anchor='           "08-tension-N12-S60-reference/attempt-01")\nNATIVE_OUTPUT'
assert expected.count(anchor)==1
expected=expected.replace(anchor,'           "08-tension-N12-S60-reference/attempt-01")\nPRIOR_POLICY_SIDECAR = ("outputs/mechanics/hbe-v5-nocache-policy-v1/"\n                        "08-tension-N12-S60-reference/attempt-01")\nNATIVE_OUTPUT')
anchor='    native_output = _local(root, NATIVE_OUTPUT)'
assert expected.count(anchor)==1
expected=expected.replace(anchor,'    prior_sidecar = _local(root, PRIOR_POLICY_SIDECAR)\n    if prior_sidecar.exists() or prior_sidecar.is_symlink():\n        raise ValueError("Prior v1 no-cache policy attempt already exists")\n'+anchor)
assert expected==v2,'Unexpected additional source delta'
def definitions(source):
 return {x.name:ast.dump(x,include_attributes=False) for x in ast.parse(source).body if isinstance(x,(ast.FunctionDef,ast.ClassDef))}
a,b=definitions(v1),definitions(v2)
assert set(a)==set(b)
changed=[name for name in a if a[name]!=b[name]]
assert changed==['_read_exact_release','execute_envelope','main'],changed
(HERE/'v1-to-v2.diff').write_text(''.join(difflib.unified_diff(v1.splitlines(True),v2.splitlines(True),fromfile='committed-v1',tofile='proposed-v2')))
template=json.loads(read(ROOT/'build/hbe-v5-tension-n12-s60-release-prep/RELEASE_TEMPLATE.json','32d7566d10d1a671fe1d7fcd2cb3bfc73dc70c3632d286480fd860e2e15d2feb'))
old20=[]
for relative,binding in template['source_bindings'].items():
 raw=read(ROOT/relative,binding['sha256'])
 assert subprocess.check_output(['git','show',head+':'+relative],cwd=ROOT)==raw
 old20.append(relative)
assert len(old20)==20
for name,expected_sha in [('hbe_v5_nocache_v1.py','efeda27abc3c1a6851e1066789e51a2f05b94728def62e71061d03da7f4629de'),
                          ('hbe_v5_nocache_hash_v1.py',EXPECTED['hbe_v5_nocache_hash_v1.py']),
                          ('hbe_v5_nocache_host_v1.py',EXPECTED['hbe_v5_nocache_host_v1.py'])]:
 raw=read(ROOT/'launchers'/name,expected_sha)
 assert subprocess.check_output(['git','show',head+':launchers/'+name],cwd=ROOT)==raw
 if name!='hbe_v5_nocache_v1.py':assert raw==read(CANDIDATE/name)
absent={}
for relative in ['launchers/hbe_v5_nocache_v2.py',template['output_directory'],
                 'outputs/mechanics/hbe-v5-nocache-policy-v1/08-tension-N12-S60-reference/attempt-01',
                 'outputs/mechanics/hbe-v5-nocache-policy-v2/08-tension-N12-S60-reference/attempt-01']:
 path=ROOT/relative
 assert not path.exists() and not path.is_symlink(),relative
 absent[relative]=True
post=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
assert post==head
for name,expected_sha in EXPECTED.items():read(CANDIDATE/name,expected_sha)
record={'status':'prospective_v2_source_delta_verified','head':head,'bindings':BINDINGS,
 'old20_matches_original_template_and_current_committed':old20,
 'exact_delta_only':['version_identity','initial_floor_55_to_54','prior_v1_sidecar_refusal'],
 'changed_definition_names':changed,'unchanged_definition_names':[name for name in a if a[name]==b[name]],
 'shared_helpers_unchanged':True,'tracked_v1_unchanged':True,'absent_targets':absent,
 'release_calls':0,'predecessor_payload_bytes_read':0,'native_calls':0,'hbe_replays':0,
 'patient_or_measured_data_reads':0}
(HERE/'audit.json').write_text(json.dumps(record,indent=2,sort_keys=True)+'\n')
print(json.dumps({'status':record['status'],'head':head,'old_source_count':len(old20),
                 'audit_sha256':digest((HERE/'audit.json').read_bytes())},sort_keys=True))
