"""Exact TRAIN endpoint diagnostic; metadata preflight never opens payloads."""
import hashlib,json,subprocess,types
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUTPUT=HERE/'attempt-01'
SUBJECTS=['ReMIND-002','ReMIND-015','ReMIND-018','ReMIND-045']
STEPS=[1,14,1,13]
CAPS={'worker_seconds':180,'parent_seconds':210,'sampled_rss_bytes':3*1024**3,
      'output_bytes':16*1024**2,'supervision_bytes':4*1024**2,'log_bytes':2*1024**2,
      'native_previews':35040,'threads':1,'policy_forwards':29,'autograd_grad_calls':79,
      'collector_replay_steps':58,'source_visits':4,'checkpoint_loads':1}
ORIGINAL_CONTRACT={'path':'build/public-motion-ranking-il64-v1/pilot_contract.py','sha256':'130be4f52979eb40a0919492a806eb4b2a71e6f76b2ba1173065677492a1fa58'}
HELPER={'path':'build/public-motion-ranking-v1/frozen-gradient-diagnostic-v1/gradient_diagnostic.py','sha256':'cc183fcd3e76667ab81a2b117ee3d058269a6831f0fd33318504c81e4d7140dd'}
PIN_SHA='dbf171f3a3876abd153c91ba36c0efbd83639339fec3ed16c3dc02ede17f9d2b'
CHECKPOINT={'path':'build/public-motion-ranking-il64-v1/IL64/attempt-01/IL-final.psckpt',
    'sha256':'2825ad738591f786f1fa5b3383cd74c4126e1ab57f0494b04b2c8c52e44bfa0e',
    'bytes':137636,'parameter_hash':'sha256:ec843adeb3ce854c2d516848b5f26a4f855a25d3f4369b28543e1ae6be7c1ed6'}
def need(ok,why):
    if not ok:raise ValueError(why)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)
def small(path,digest=None):
    path=ROOT/path
    need(path.is_file() and not path.is_symlink() and 0<path.stat().st_size<=4*1024**2,'bounded_metadata')
    raw=path.read_bytes()
    if digest is not None:need(hashlib.sha256(raw).hexdigest()==digest,'metadata_hash:'+str(path))
    return json.loads(raw)
def source_module(ref,name):
    path=ROOT/ref['path'];raw=path.read_bytes()
    need(not path.is_symlink() and hashlib.sha256(raw).hexdigest()==ref['sha256'],'exact_source_loader')
    module=types.ModuleType(name);module.__file__=str(path)
    exec(compile(raw,str(path),'exec'),module.__dict__)
    return module
def metadata():
    pins=small(HERE/'input-pins.json',PIN_SHA)
    rows={name:small(ref['path'],ref['sha256']) for name,ref in pins.items()}
    owner=source_module(ORIGINAL_CONTRACT,'frozen_ranking_owner_contract')
    result,parent,release=rows['result'],rows['parent'],rows['release']
    need(owner.complete_result(result) and parent['status']=='complete' and parent['exit_code']==0
         and parent['worker_termination_confirmed'] is True and not parent['cleanup_errors']
         and not parent['final_owned_pids'] and parent['result_sha256']==pins['result']['sha256']
         and parent['release_sha256']==pins['release']['sha256']
         and parent['source_index']==release['source_index'],'completed_ranking_endpoint')
    expected,limits=owner.expected_configuration('IL')
    need(canonical(expected)==canonical(release['learning_protocol']) and
         canonical(limits)==canonical(release['cohort_limits']),'exact_historical_context')
    endpoint=result['checkpoints']['IL']
    need(all(endpoint[k]==CHECKPOINT[k] for k in ('sha256','bytes','parameter_hash'))
         and endpoint['path']==Path(CHECKPOINT['path']).name,'exact_checkpoint_metadata_only')
    need(pins['helper_index']['sha256']=='0124829d72bc43c3fb0370706012f0e4e40a2e570ebc1b54191d54e1da7b6f91'
         and rows['helper_index']['files']['gradient_diagnostic.py']['sha256']==HELPER['sha256'],'reviewed_helper')
    for subject,n in zip(SUBJECTS,STEPS):
        teacher=rows['teacher_pins'][subject]
        need(teacher['steps']==n and teacher['stop_steps']==1,'exact_teacher_steps')
        for step in range(n):
            row=rows[f'{subject}:{step}']
            need(row['subject']==subject and row['step']==step and row['endpoint']=='IL64'
                 and row['teacher_action']==teacher['actions'][step],'exact_saved_endpoint_rows')
    return owner,rows,pins
def release_template(head,source_index):
    owner,rows,pins=metadata();original=rows['release']
    return {'version':'frozen-ranking-gradient-diagnostic-v1','execution_released':False,
        'expected_head':head,'caps':CAPS,'source_index':source_index,'input_pins_sha256':PIN_SHA,
        'helper':HELPER,'checkpoint':CHECKPOINT,'subjects':SUBJECTS,
        'learning_protocol':original['learning_protocol'],'learning_protocol_hash':original['learning_protocol_hash'],
        'cohort_limits':original['cohort_limits'],'limits':original['limits'],
        'public_index':original['public_manifest_index'],
        'occupancy_condition':owner.OCCUPANCY,'training_updates_authorized':0,
        'scope':'fixed TRAIN raw-gradient diagnostic; no new policy rollouts or SELECT/EVAL access'}
def guard(path,digest,*,execute):
    release=small(path,digest)
    if execute:need(release.get('execution_released') is True,'root_release_required')
    need(type(release.get('execution_released')) is bool,'typed_release')
    expected=release_template(release['expected_head'],release['source_index'])
    need({**release,'execution_released':False}==expected,'exact_diagnostic_release')
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    need(head==release['expected_head'],'HEAD_changed')
    index=small(release['source_index']['path'],release['source_index']['sha256'])
    need(index['head']==head,'index_HEAD')
    required={HELPER['path'],ORIGINAL_CONTRACT['path'],'src/resectionlab/public_motion_ranking.py',
        'src/resectionlab/patient_teacher_trace_cache.py','src/resectionlab/patient_planning_cohort_io.py'}
    required.update(str((HERE/name).relative_to(ROOT)) for name in ('batch_contract.py','initial_inventory_worker.py','run_owned.py','freeze-runtime.py'))
    need(required<=set(index['files']),'runtime_closure')
    for name,pin in index['files'].items():need(sha(ROOT/name)==pin,'source_or_metadata_changed:'+name)
    return release,index,True
def complete_result(result,release,release_sha):
    need(result['status']=='complete_frozen_gradient_diagnostic' and result['release_sha256']==release_sha,'complete_diagnostic')
    need(result['completed_source_visits']==4 and result['collector_replay_steps']==58
         and result['policy_forwards']==29 and result['autograd_grad_calls']==79
         and result['checkpoint_loads']==1 and result['native_previews']<=35040,'exact_diagnostic_work')
    need(all(result[name]==0 for name in ('optimizer_calls','backward_calls','forbidden_io_calls','search_calls')),'zero_training_or_foreign_work')
    need(result['parameters_unchanged'] and result['gradient_buffers_unchanged']
         and result['requires_grad_flags_restored'],'frozen_endpoint_after_diagnostic')
    need(sha(OUTPUT/'gradient-alignment.json')==result['gradient_report_sha256'],'gradient_report_binding')
    report=small(OUTPUT/'gradient-alignment.json')
    need(report['status']=='complete_frozen_gradient_readout' and len(report['rows'])==29
         and report['counts']=={'policy_forwards':29,'autograd_grad_calls':79,'optimizer_updates':0}
         and report['parameter_hash']==CHECKPOINT['parameter_hash'],'complete_reviewed_helper')
