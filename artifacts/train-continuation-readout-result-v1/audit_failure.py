"""Single saved-only origin audit. ZIP is checksummed, never decoded."""
import ast,hashlib,json,math,pathlib,signal,sys,time
HERE=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import audit_saved as common
base=common.base;need=common.need;read=common.read_json;ROOT=common.ROOT
REPORT='64bd8f8d46efdd0f85baac977de3e34b2105301ec48a4b1ac9ab9735dce81a32'
LAUNCHER='613f388fda57d8963db7bd6b063988bfeab723223173fa575c66b99b52792b15'
INDEX='37c03f09c3754429ecb2b8fa24e5093a7efd11424ef33c1c093a43ce1b15a3b5'

def audit():
    stage=HERE.parent;out=stage/'run-01'
    report=read(out/'report.json',REPORT);launcher=read(out/'launcher.json',LAUNCHER)
    index=read(stage/'runner/source-index.json',INDEX)
    need(report['source_index_sha256']==launcher['source_index_sha256']==INDEX
         and launcher['report_sha256']==REPORT,'Exact terminal pins')
    life=launcher['owned_lifecycle']
    need(launcher['status']=='incomplete_or_failed' and report['status']=='failed_or_incomplete'
         and launcher['worker_exit_code']==life['exit_code']==1,'Original failure retained')
    need(life['verified'] and life['child_reaped'] and not life['remaining_owned_pids']
         and not life['remaining_state_unknown'] and not life['errors'] and launcher['stop_reason'] is None,
         'Clean supervised failure; no unknown/surviving worker')
    failure=report['failure']
    need(failure['type']=='FileExistsError' and failure['phase']=='authenticated_cache_save_and_exact_reload'
         and 'logged-targets.json' in failure['message'],'Exact saved publication failure')
    need(report['head_optimizer_updates']==report['calibration_calls_attempted']==report['calibration_calls_completed']==0
         and report['cases']=={} and 'head_fit' not in report and 'calibration' not in report,
         'No fit/calibration/routes preceded failure')
    worker='build/continuation-next-iteration-v1/runner/worker.py'
    raw=(ROOT/worker).read_bytes();need(hashlib.sha256(raw).hexdigest()==index['files'][worker],'Exact failed worker source')
    tree=ast.parse(raw);writes=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='write' and node.args:
            argument=ast.get_source_segment(raw.decode(),node.args[0])
            if argument=="output/'logged-targets.json'":writes.append(node.lineno)
    need(len(writes)==2,'Two exclusive writes target same path in frozen source')
    manifest=read(out/'trace-manifest.json');need(common.semantic(manifest)==report['trace_manifest_hash'],'Manifest hash')
    declared={v['key']:v for v in manifest};rows=[]
    for item in report['reconstructed_traces']:
        key,subject,family=item['key'],item['subject'],item['kind']
        update=None if family=='teacher' else int(key.rsplit('RL',1)[1])
        folder=base.RL+'/teachers/'+subject if family=='teacher' else base.RL+'/RL/update-%02d/'%update+subject
        values=[read(ROOT/(folder+'/'+name),index['input_metadata'][folder+'/'+name])
                for name in ('complete-trace.json','plan.json','native-replay.json')]
        row=base.validate_saved_episode(*values,subject=subject,family=family,update=update);rows.append(row)
        need(item['states']==row['steps'] and item['historical_trace_seal']==row['trace_seal']
             and item['historical_plan_seal']==row['plan_seal'],'Original complete replay joins')
        need(item['descriptor_hash']==common.semantic(declared[key])
             and item['history_hash']==declared[key]['history_hash'],'Reconstructed descriptor/history join')
        base.valid_geometry(item['audit'],key)
        need(item['audit']['committed_history_hash']==values[2]['independent_geometry']['committed_history_hash'],
             'Reconstruction certificate equals exact historical native history')
    summary=base.validate_corpus(rows)
    logged=read(out/'logged-targets.json');oldpath=common.ORIGINAL+'/logged-targets.json'
    need(logged==read(ROOT/oldpath,index['input_metadata'][oldpath]),'First scalar-label write exactly preserved')
    p=0
    for row in rows:
        for label in row['labels']:
            base.check_target_rounding(logged['targets'][p],label['mandatory_logged_action_return'])
            need(logged['weights'][p]==base.f32(1/(4*9*row['steps'])),'Original weight mass')
            need(logged['STOP_rows'][p] is (label['action_id']=='STOP'),'Original STOP denominator');p+=1
    need(p==497==report['total_frozen_policy_forwards'],'Complete497 reconstruction/extraction')
    cache=common.check_cache(report,index,out,manifest)
    archive=out/'authenticated-features.zip';need(not archive.is_symlink() and archive.stat().st_size==cache['bytes'],'Exact regular bounded cache')
    sha=hashlib.sha256()
    with archive.open('rb') as stream:
        for chunk in iter(lambda:stream.read(65536),b''):sha.update(chunk)
    need(sha.hexdigest()==cache['cache_sha256'],'Whole cache byte identity, no ZIP decode')
    need(report['source_unchanged'] and not report['closed_roles_opened']
         and report['base_optimizer_updates']==report['new_training_rollouts']==0,'Source/split/frozen base intact')
    native=report['native_budget'];need(native['counting_reliable'] and native['native_preview_entries']==45560,'Prior native cost retained')
    paid=cache['original_costs_retained'];need(paid['native_preview_entries']==45560
         and paid['native_transition_calls']==497 and paid['frozen_policy_forwards']==497,'Prior work independently joined')
    return {'original_experiment_status':'failed_or_incomplete','clean_failure_cleanup':True,
        'recovery_point':'PASS_SAVED_CACHE_ORIGIN_AND_BYTES','failure':failure,'duplicate_write_lines':sorted(writes),
        'corpus':summary,'cache':cache,'cache_zip_decoded':False,'head_updates':0,'calibrations':0,'routes':0,
        'prior_native_previews':45560,'prior_forwards':497,'prior_parent_seconds':launcher['elapsed_seconds'],
        'prior_worker_seconds':report['worker_seconds'],'prior_peak_rss_bytes':launcher['sampled_peak_group_rss_bytes'],
        'report_sha256':REPORT,'launcher_sha256':LAUNCHER,'index_sha256':INDEX,
        'limitation':'No cached arrays/tensors decoded; original successful cache reload and independent native certificates are provenance. Fresh bounded recovery must authenticate the same external binding and bytes. This is not a completed comparison.',
        'input_files':base.READS}

if __name__=='__main__':
    started=time.monotonic()
    def timeout(*_):raise TimeoutError('30s saved audit cap')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(30)
    try:r=audit();r['audit_status']='PASS_SAVED_EVIDENCE';code=0
    except BaseException as e:r={'audit_status':'FAILED_REVIEWER_CHECK','error':type(e).__name__+': '+str(e),'input_files':base.READS};code=1
    signal.alarm(0);r.update(checks=base.CHECKS,seconds=time.monotonic()-started,native_calls=0,model_calls=0,
        auditor_sha256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest())
    raw=json.dumps(r,sort_keys=True,indent=2,allow_nan=False).encode()+b'\n';assert len(raw)<=65536
    with (HERE/'failure-audit-01.json').open('xb') as stream:stream.write(raw)
    print(json.dumps({k:r[k] for k in ('audit_status','checks','seconds')}));sys.exit(code)
