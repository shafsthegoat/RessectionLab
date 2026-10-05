"""Saved-receipt and static-byte audit only: never loads or executes the runtime."""
from pathlib import Path
import hashlib,json,os,struct,time
R=Path(__file__).resolve().parents[2]
O=R/'artifacts/febio-accelerate-csc-runtime-v3'
P=R/'data/optional-runtimes/febio-4.13-accelerate-csc-v1'
OLD=R/'data/optional-runtimes/febio-4.13'

def sha(p):
    with p.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def read(p):return json.loads(p.read_bytes())
def pin(b):
    p=R/b['path'];assert sha(p)==b['sha256']
    if 'bytes' in b:assert p.stat().st_size==b['bytes']
    return p
def inventory(base):
    out={}
    for p in sorted(base.rglob('*')):
        if p.is_symlink():out[str(p.relative_to(base))]={'symlink':os.readlink(p)}
        elif p.is_file():out[str(p.relative_to(base))]={'bytes':p.stat().st_size,'sha256':sha(p)}
    return out

def macho(p):
    with p.open('rb') as stream:
        h=stream.read(32);magic,cpu,sub,kind,n,size,flags,reserved=struct.unpack('<8I',h)
        assert magic==0xfeedfacf and cpu==0x100000c and size<1024**2
        data=stream.read(size);assert len(data)==size
    offset=0;dependencies=[];rpaths=[]
    for _ in range(n):
        command,length=struct.unpack_from('<II',data,offset)
        assert length>=8 and offset+length<=size
        if command in (0xc,0xd,0x80000018,0x8000001f,0x80000023,0x20,0x8000001c):
            start=struct.unpack_from('<I',data,offset+8)[0]
            assert 12<=start<length
            value=data[offset+start:offset+length].split(b'\0',1)[0].decode('utf8')
            (rpaths if command==0x8000001c else dependencies).append(value)
        offset+=length
    assert offset==size
    return dependencies,rpaths

started=time.monotonic()
identity=read(O/'runtime-identity.json')
assert sha(O/'runtime-identity.json')=='13c4f60cbae8ad89232996e4f7d773bafbd2489f23a669c355f5ac09e417cc57'
v3=read(R/'manifests/experiments/febio-accelerate-csc-runtime-v3.json');v1=read(pin(v3['basis']))
assert sha(R/v3['driver_path'])==v3['driver_sha256']==identity['driver_sha256']
assert sha(R/'manifests/experiments/febio-accelerate-csc-runtime-v3.json')==identity['declaration_sha256']
for b in v3['saved_evidence'].values():pin(b)
for b in identity['adapter_controls'].values():pin(b)
assert identity['adapter_controls']==v3['adapter_control_validation']
for key in ('source_inventory','installed_inventory','linkage','build_acceptance','private_openmp','patch'):pin(identity[key])
original=read(pin(v1['upstream']['original_input_inventory']))
original_actual={folder+'/'+k:v for folder in sorted({k.split('/')[0] for k in original}) for k,v in inventory(OLD/folder).items()}
assert original_actual==original and len(original)==6164
old_identity=read(pin(v1['upstream']['original_runtime_identity']))
old_install=read(R/'artifacts/febio-runtime-build-v1/build-01/installed-inventory.json')
assert inventory(OLD/'install')==old_install and len(old_install)==1037
for name,digest in old_identity['receipts_sha256'].items():assert sha(R/'artifacts/febio-runtime-build-v1'/name)==digest
source=read(pin(identity['source_inventory']));assert inventory(P/'source')==source and len(source)==2322
source_old={k[7:]:v for k,v in original.items() if k.startswith('source/')}
assert set(source_old)==set(source)
assert [k for k in source if source[k]!=source_old[k]]==['NumCore/AccelerateSparseSolver.cpp']
assert source['NumCore/AccelerateSparseSolver.cpp']['sha256']==identity['patched_source_sha256']==v3['patch_override']['patched_sha256']
assert identity['source_patch_sha256']==v3['patch_override']['file']['sha256']
installed=read(pin(identity['installed_inventory']));assert inventory(P/'install')==installed
stages={};stage_files={'configure':{'result.json','supervision.json','source-inventory.json','verified-cache.json','source-preparation.json','commands.json'},'build':{'result.json','supervision.json','installed-inventory.json','linkage.json','runtime-identity-candidate.json','commands.json'}}
for stage,cap in (('configure',120),('build',900)):
    base=O/(stage+'-01');a=read(base/'acceptance.json');result=read(base/'result.json');s=read(base/'supervision.json')
    assert set(a['artifacts'])==stage_files[stage]
    for name,digest in a['artifacts'].items():assert sha(base/name)==digest
    assert a['status']==result['status']==s['status']=='completed' and a['stage']==result['stage']==stage
    assert a['driver_sha256']==result['driver_sha256']==identity['driver_sha256']
    assert a['declaration_sha256']==result['declaration_sha256']==identity['declaration_sha256']
    assert s['exit_code']==0 and s['kill_reason'] is None and s['cleanup_error'] is None
    assert s['wall_cap_seconds']==cap and s['rss_cap_bytes']==3*1024**3
    assert s['elapsed_seconds']<cap and s['sampled_peak_process_group_rss_bytes']<3*1024**3
    assert result['original_before']==result['original_after']==a['parent_original_after']
    assert result['solver_executed'] is False
    stages[stage]={'acceptance_sha256':sha(base/'acceptance.json'),'seconds':s['elapsed_seconds'],'sampled_rss_bytes':s['sampled_peak_process_group_rss_bytes']}
assert read(O/'configure-01/verified-cache.json')==v1['configure']['cache_options']
cmds=read(O/'build-01/commands.json')['commands'];cmake=str(R/v1['tools']['cmake']['path'])
assert cmds==[[cmake,'--build',str(P/'build'),'--target','febio4','--parallel','2'],[cmake,'--install',str(P/'build'),'--prefix',str(P/'install')]]
a=read(O/'build-01/acceptance.json');candidate=read(O/'build-01/runtime-identity-candidate.json')
expected=dict(candidate,status='isolated_patched_runtime_built_pending_numerical_controls',build_acceptance_sha256=sha(O/'build-01/acceptance.json'),build_acceptance_path=str(O/'build-01/acceptance.json'),build_acceptance={'path':str(O/'build-01/acceptance.json'),'sha256':sha(O/'build-01/acceptance.json')},receipts_sha256={'build-01/'+k:v for k,v in a['artifacts'].items()})
assert identity==expected
assert identity['configure_acceptance_sha256']==sha(O/'configure-01/acceptance.json')
assert identity['solver_executed'] is False and identity['models_executed']==0 and identity['numerically_validated'] is False and identity['version_executable_launched'] is False
known=set(old_identity['libraries']);assert len(known)==13 and set(identity['libraries'])==known
magic={bytes.fromhex(x) for x in ('feedface','cefaedfe','feedfacf','cffaedfe','cafebabe','bebafeca','cafebabf','bfbafeca')};found=set()
for name in installed:
    with (P/'install'/name).open('rb') as f:
        if f.read(4) in magic:found.add('install/'+name)
assert found==known
linkage=read(pin(identity['linkage']));assert linkage['status']=='static_linkage_passed' and len(linkage['files'])==13
omp=pin(identity['private_openmp']);assert omp==R/v1['tools']['openmp_library']['path']
rows=[]
for row in linkage['files']:
    name=row['path'];p=P/name;assert sha(p)==row['sha256']==identity['libraries'][name]
    assert all(c['exit_code']==0 for c in row['commands'])
    dependencies,rpaths=macho(p);assert dependencies==row['dependencies'] and rpaths==row['rpaths']
    assert set(dependencies)<=set(old_identity['dependency_install_names'])
    assert set(rpaths)==set(v1['configure']['cache_options']['CMAKE_INSTALL_RPATH'].split(';'))
    for dep in dependencies:
        if dep.startswith(('/usr/lib/','/System/Library/')):assert row['resolved_dependencies'][dep]=='system_install_name_not_loaded';continue
        assert dep.startswith('@rpath/');basename=dep[7:]
        paths=[Path(q.replace('@loader_path',str(p.parent)).replace('@executable_path',str(P/'install/bin')))/basename for q in rpaths]
        actual=next(q.resolve() for q in paths if q.exists());expected=omp if basename=='libomp.dylib' else P/'install/lib'/basename
        assert actual==expected.resolve()
        assert row['resolved_dependencies'][dep]=={'path':str(actual),'sha256':sha(actual)}
    rows.append({'path':name,'sha256':sha(p),'architecture':'arm64','static_load_commands_match_saved':True})
assert sha(O/'runtime-identity.json')=='13c4f60cbae8ad89232996e4f7d773bafbd2489f23a669c355f5ac09e417cc57'
record={'status':'saved_build_and_static_runtime_closure_review_passed','runtime_identity_sha256':sha(O/'runtime-identity.json'),'audit_source_sha256':sha(Path(__file__)),'stages':stages,'original_acquisition_entries_rehashed':len(original),'original_install_entries_rehashed':len(old_install),'new_source_entries_rehashed':len(source),'new_install_entries_rehashed':len(installed),'installed_macho':rows,'complete_identity_equals_accepted_candidate_publication':True,'static_load_command_parsing':'Independent direct thin64 Mach-O header/load-command byte parsing; no lipo/otool/runtime invoked during audit. Includes LC_ID_DYLIB to match otool-L saved convention.','private_openmp':identity['private_openmp'],'initial_review_assertion_correction':'First audit incorrectly required one shared RPATH order. Actual per-file order matches saved commands; allowed-set contract plus actual ordered resolution is checked. Initial script/failure retained.','elapsed_seconds':time.monotonic()-started,'executable_or_solver_or_model_launched':False,'limits':['System install names and static path closure checked; dynamic symbol resolution and executable startup not tested.','Runtime version4.13.0 is source-declared only; real solver/physics controls remain pending.','Resource peaks are sampled process-group RSS, not a guarantee about between-sample peaks.']}
with Path(__file__).with_name('review.json').open('x') as f:json.dump(record,f,indent=2,sort_keys=True);f.write('\n')
print(json.dumps({'status':record['status'],'seconds':record['elapsed_seconds'],'receipt_sha256':sha(Path(__file__).with_name('review.json'))}))
