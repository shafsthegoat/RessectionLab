"""Exactly one authorized private Gmsh version probe; no meshing or data input."""
import datetime,hashlib,json,os,signal,subprocess,time
from pathlib import Path
OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]
def digest(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2);f.write('\n')
def process_family_rss(root_pid):
    """Sample descendants plus the dedicated process group; RSS is summed."""
    output = subprocess.check_output(["ps", "-axo", "pid=,ppid=,pgid=,rss="], text=True)
    rows = [tuple(map(int, row.split())) for row in output.splitlines() if row.strip()]
    selected = {root_pid} | {pid for pid, _, group, _ in rows if group == root_pid}
    while True:
        expanded = selected | {pid for pid, parent, _, _ in rows if parent in selected}
        if expanded == selected:
            break
        selected = expanded
    return sum(rss * 1024 for pid, _, _, rss in rows if pid in selected)

def capped_run(command, *, cwd, env, handle, wall_seconds, memory_bytes, interval=0.5):
    started = time.monotonic()
    process = subprocess.Popen(command, cwd=cwd, env=env, stdout=handle,
                               stderr=subprocess.STDOUT, start_new_session=True)
    peak = 0
    samples = 0
    reason = None
    try:
        while process.poll() is None:
            rss = process_family_rss(process.pid)
            samples += 1
            peak = max(peak, rss)
            if rss > memory_bytes:
                reason = "sampled_process_family_memory_cap"
            elif time.monotonic() - started > wall_seconds:
                reason = "wall_time_cap"
            if reason is not None:
                break
            time.sleep(interval)
    finally:
        if reason is not None or process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        process.wait()
    return {"exit_code": process.returncode, "cap_failure": reason,
            "sampled_peak_process_family_rss_bytes": peak, "resource_samples": samples,
            "sample_interval_seconds": interval, "wall_cap_seconds": wall_seconds,
            "memory_cap_bytes": memory_bytes}

SUPERVISION_SOURCE_SHA256='271667258961d604c7523986fc14fe4f51b09417ea78e0c3adcd9bd04648bd15'
CHILD = r'''
import hashlib,importlib.util,json,pathlib,resource,sys,time
started=time.monotonic();resource.setrlimit(resource.RLIMIT_CPU,(12,12))
info=json.loads(pathlib.Path(sys.argv[1]).read_text());release=json.loads(pathlib.Path(sys.argv[2]).read_text())
module_path=pathlib.Path(info['module']['path']);library_path=pathlib.Path(info['library']['path'])
for key in ('module','library'):
    with pathlib.Path(info[key]['path']).open('rb') as handle:assert hashlib.file_digest(handle,'sha256').hexdigest()==info[key]['sha256']
assert not (module_path.parent/library_path.name).exists()
assert library_path==module_path.parent/'lib'/library_path.name
interpreter=pathlib.Path(sys.executable).resolve()
with interpreter.open('rb') as handle:assert hashlib.file_digest(handle,'sha256').hexdigest()==release['interpreter_sha256']
assert 'gmsh' not in sys.modules
spec=importlib.util.spec_from_file_location('ressectionlab_private_gmsh_4_15_2',module_path)
gmsh=importlib.util.module_from_spec(spec);spec.loader.exec_module(gmsh)
assert pathlib.Path(gmsh.__file__).resolve()==module_path.resolve()
assert pathlib.Path(gmsh.lib._name).resolve()==library_path.resolve()
assert not gmsh.use_numpy
assert gmsh.__version__=='4.15.2'
assert gmsh.isInitialized()==0
gmsh.initialize(['gmsh-private-version-probe','-nt','1'],readConfigFiles=False,run=False,interruptible=False)
try:
    gmsh.option.setNumber('General.NumThreads',1)
    actual_version=gmsh.option.getString('General.Version')
    actual_threads=gmsh.option.getNumber('General.NumThreads')
    assert actual_version=='4.15.2' and actual_threads==1
finally:gmsh.finalize()
assert gmsh.isInitialized()==0
print(json.dumps({'status':'passed','api_version':gmsh.__version__,'runtime_version':actual_version,'module_path':gmsh.__file__,'library_path':gmsh.lib._name,'numpy_used':gmsh.use_numpy,'threads_option':actual_threads,'read_config_files_argument':False,'run_argument':False,'mesh_generation':False,'geometry_input':False,'patient_or_curve_input':False,'interpreter':str(interpreter),'interpreter_version':sys.version,'sys_path':sys.path,'elapsed_seconds':time.monotonic()-started,'self_peak_rss_bytes_macos':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'finalized':True}))
'''
def main():
    release=json.loads((OUT/'root-release.json').read_text())
    extraction=json.loads((OUT/'extraction-receipt.json').read_text())
    inventory=json.loads((OUT/'extracted-files.json').read_text())
    prefix=Path(inventory['prefix'])
    before={str(p.relative_to(prefix)):digest(p) for p in prefix.rglob('*') if p.is_file()}
    expected={v['relative_path']:v['sha256'] for v in inventory['files']}
    assert before==expected
    assert digest(Path(release['interpreter_resolved']))==release['interpreter_sha256']
    environment={k:v for k,v in os.environ.items() if not k.startswith(('DYLD_','LD_','PYTHON'))}
    environment.update(PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1')
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
        environment[key]='1'
    command=[release['interpreter'],'-I','-S','-B','-c',CHILD,str(OUT/'extraction-receipt.json'),str(OUT/'root-release.json')]
    attempt={'schema_version':1,'status':'running','probe_number':1,'argv':command,'wall_seconds':15,'rss_cap_bytes':1024**3,'numerical_threads':1,'source_supervision_sha256':SUPERVISION_SOURCE_SHA256,'probe_driver_sha256':digest(Path(__file__)),'created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'extraction_receipt_sha256':digest(OUT/'extraction-receipt.json')}
    save(OUT/'probe-attempt.json',attempt)
    started=time.monotonic()
    with (OUT/'probe.log').open('x') as handle:
        outcome=capped_run(command,cwd=prefix,env=environment,handle=handle,wall_seconds=15,memory_bytes=1024**3,interval=.05)
    elapsed=time.monotonic()-started
    after={str(p.relative_to(prefix)):digest(p) for p in prefix.rglob('*') if p.is_file()}
    unchanged=before==after
    log=(OUT/'probe.log').read_text()
    child=json.loads(log.strip()) if outcome['exit_code']==0 else None
    passed=outcome['exit_code']==0 and outcome['cap_failure'] is None and unchanged and child['status']=='passed'
    receipt={'schema_version':1,'status':'verified_ready_for_separate_meshing_release' if passed else 'failed_no_retry','name':'Gmsh','version':'4.15.2','private_prefix':extraction['private_prefix'],'runtime_prefix':extraction['runtime_prefix'],'module':extraction['module'],'library':extraction['library'],'interpreter':{'path':release['interpreter'],'resolved_path':release['interpreter_resolved'],'sha256':release['interpreter_sha256'],'version':release['interpreter_version']},'wheel':{'url':release['mesher']['url'],'bytes':release['mesher']['bytes'],'sha256':release['mesher']['sha256'],'acquisition_receipt_path':str(OUT/'acquisition-receipt.json'),'acquisition_receipt_sha256':digest(OUT/'acquisition-receipt.json')},'source_declaration':{'git_commit':release['git_commit'],'manifest_path':release['manifest_path'],'manifest_sha256':release['manifest_sha256']},'license':{**extraction['license'],'identifier':release['mesher']['license'],'scope':'separate local research utility, no app bundling or distribution approval'},'receipt_bindings':{name:digest(OUT/name) for name in ('root-release.json','extraction-format-failure.json','extraction-continuation-release.json','record-format-controls.json','extraction-receipt.json','extracted-files.json','linkage.json','probe-attempt.json','probe.log')},'probe':{**outcome,'elapsed_seconds':elapsed,'child':child,'probe_count':1,'all_extracted_files_unchanged':unchanged,'file_count':len(before)},'mesh_generation':False,'patient_or_curve_access':False,'global_or_app_environment_changes':False,'historical_failure':'Strict base64-only RECORD verifier stopped; explicit root-approved continuation accepted actual creator hex only after controls, no overwrite or transfer/probe retry.'}
    save(OUT/'runtime-receipt.json',receipt)
    print(json.dumps({'status':receipt['status'],'probe':receipt['probe'],'runtime_receipt_sha256':digest(OUT/'runtime-receipt.json')}))
    if not passed:raise SystemExit(1)
if __name__=='__main__':main()
