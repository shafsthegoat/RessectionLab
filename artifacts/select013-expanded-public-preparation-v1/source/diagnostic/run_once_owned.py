"""One root-authorized two-public-SEG diagnostic; existing owned cleanup reused."""
import ast, hashlib, json, os, signal, stat, subprocess, sys, time
from pathlib import Path

HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[1]
HEAD='290bdccfe3350b89343876f899b4c3798bc40b38'
PINS={
 'build/select013-source-domain-diagnostic-v1/diagnose_source_domains.py':'8ed28ed8038031c0b3f6b70ec7b40c38bcf9d871134a151efb2ef9f2c7d2ddad',
 'build/goal-conditioned-policy-v1/darwin_fast_sampler.py':'906a70502e4f0a48fff925cca9fc1fe45882818558cc7ac95e9034e3d6910d2c',
 'build/goal-conditioned-policy-v1/run_contact_owned.py':'de8c1c5a97b4d6adcd7746f080c2b54bd3b1c7f4cb0be3cd447039c5a951b0c7',
 'build/post-exposure-IL045-evaluation-recovery-v1/run_owned.py':'3e48eb654eaee6a048840a5ef4592e3a9938f6cd864dd9af3dbc7e49150196ff',
 'src/resectionlab/remind_planning_qc.py':'f35a28f2cdedd777e15031b5cba370c4106c0436eefe9590ca17466d8944b3ca',
 'scripts/convert_remind_development.py':'34e6ceb9560344a88226489b2dd8ca6bfb90cb87c52d2ff3184d393293df4ef6',
}
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def guard():
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True,timeout=5).strip()==HEAD
    assert all(sha(ROOT/p)==s for p,s in PINS.items())
def save(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
def load_functions(path,names):
    tree=ast.parse((ROOT/path).read_bytes(),filename=str(ROOT/path))
    body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert {n.name for n in body}==set(names)
    exec(compile(ast.Module(body=body,type_ignores=[]),str(ROOT/path),'exec'),globals())

guard()
# Exact bodies from the existing reviewed owner, including retained-Popen
# fallback if sampler cleanup raises; no new cleanup policy.
load_functions('build/goal-conditioned-policy-v1/run_contact_owned.py',{'cleanup_owned'})
load_functions('build/post-exposure-IL045-evaluation-recovery-v1/run_owned.py',{'cleanup_phase','tree_bytes'})
ns={'__file__':str(ROOT/'build/goal-conditioned-policy-v1/darwin_fast_sampler.py')}
exec(compile(Path(ns['__file__']).read_bytes(),ns['__file__'],'exec'),ns)
FastDarwinSampler=ns['FastDarwinSampler']
attempt=HERE/'attempt-01';attempt.mkdir(exist_ok=False)
command=[str(ROOT/'.venv/bin/python'),'-I','-B','-X','pycache_prefix='+str(attempt/'fresh-pycache'),
 str(HERE/'diagnose_source_domains.py'),'--execute-two-public-segs','--output',str(attempt/'result.json')]
caps={'wall_seconds':60,'RSS_bytes':1536*1024**2,'output_bytes':2*1024**2,'threads':1}
save(attempt/'declaration.json',{'command':command,'head':HEAD,'source_pins':PINS,'caps':caps,
 'root_authorized':True,'original_public_SEG_objects':2,'MR_pixel_reads':0,'private_reads':0,'automatic_retry':False,
 'parent_sha256':sha(__file__)})
env=dict(os.environ)
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):env[k]='1'
for k in ('PYTHONPATH','PYTHONHOME','PYTHONUSERBASE','LD_PRELOAD','DYLD_LIBRARY_PATH'):env.pop(k,None)
started=time.monotonic();sampler=FastDarwinSampler();process=None;reason=None;peak=samples=0
detached={};actions=[];errors=[];remaining=[];result={}
def stop(*_):raise InterruptedError('owned_parent_signal')
prior=signal.signal(signal.SIGTERM,stop)
try:
    with (attempt/'worker.log').open('x') as log:
        process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        while process.poll() is None:
            if time.monotonic()-started>=caps['wall_seconds']:reason='wall_cap';break
            measured=sampler.group(process.pid,process.pid);samples+=1
            detached.update(measured['detached_descendant_start_identities'])
            peak=max(peak,measured['process_group_resident_bytes'])
            if peak>caps['RSS_bytes']:reason='RSS_cap';break
            if tree_bytes(attempt)>caps['output_bytes']-65536:reason='output_cap';break
            time.sleep(.05)
except BaseException as error:reason=reason or type(error).__name__+':'+str(error)
finally:
    if process is not None:remaining=cleanup_phase(process,sampler,detached,actions,errors,cleanup_owned)
    code=None if process is None else process.poll()
    try:
        guard()
        assert tree_bytes(attempt)<=caps['output_bytes']-65536
        result=json.loads((attempt/'result.json').read_bytes())
        assert result['status']=='source_domain_diagnostic_complete'
        assert result['source_read_accounting']=={'objects_verified':2,'source_file_returned_bytes':7464858}
        assert result['MR_pixels_read']==result['private_objects_read']==result['native_previews']==result['model_forwards']==0
    except BaseException as error:reason=reason or 'terminal_guard:'+type(error).__name__+':'+str(error)
    elapsed=time.monotonic()-started
    if elapsed>=caps['wall_seconds']:reason=reason or 'wall_cap'
    good=reason is None and code==0 and samples>0 and not errors and not remaining
    receipt={'status':'complete' if good else 'failed','stop_reason':reason,'exit_code':code,
      'worker_pid':None if process is None else process.pid,'child_reaped':code is not None,
      'elapsed_seconds':elapsed,'sampled_peak_RSS_bytes':peak,'samples':samples,'remaining_owned_pids':remaining,
      'cleanup_actions':actions,'cleanup_errors':errors,'caps':caps,'head':HEAD,'source_pins':PINS,
      'result_sha256':sha(attempt/'result.json') if (attempt/'result.json').is_file() else None,
      'worker_log_sha256':sha(attempt/'worker.log'),'output_bytes_before_receipt':tree_bytes(attempt),
      'source_and_HEAD_unchanged':reason is None or not reason.startswith('terminal_guard'),
      'automatic_retry':False,'sampling_limit':'transient RSS peaks between samples may be missed'}
    save(attempt/'receipt.json',receipt);signal.signal(signal.SIGTERM,prior)
print(json.dumps(receipt,indent=2))
raise SystemExit(0 if good else 1)
