"""Single terminal JSON/scalar audit of fixed045 frontier preparation.

No scientific imports or payload opens. Native geometry is checked by the saved
independent full-episode certificates. This reader verifies those bindings and
recalculates plan selection, complete costs and net suffix gain independently.
"""
import argparse
import hashlib
import gzip
import io
import json
import math
from pathlib import Path
import signal
import time

HERE=Path(__file__).resolve().parent
ROOT=next(p for p in HERE.parents if (p/'src/resectionlab').is_dir())
STAGE=HERE.parent
SUBJECT='ReMIND-045'
ARMS=('ORIGINAL_STOP','FOCUSED_PREPARATION')
PRIOR='build/continuation-endgame-pair-v1/recovery-01/report.json'
PRIOR_SHA='e0ab3670571e64ddf5818c190a7d599f55f96dd5e59be24ca29b1038e7b29751'
PRIOR_AUDIT='build/continuation-endgame-pair-v1/independent/recovery-audit-01.json'
PRIOR_AUDIT_SHA='44ee7677f3e3a7028f0de34ec4f7ecdda4754047026eb2d55e634afdcbe52b80'
OLD_PLANS='build/continuation-endgame-pair-v1/run-01/plans.json'
OLD_PLANS_SHA='6798288cbb05cce8692f61f64438b68fcad8aef15ebc5ec008b2f3e3a151521b'
PINS={'worker.py':'c279b57151202079f8c72e077618b053157c0bd4504fae6e0cc5505c0e73b6cc',
      'frontier_protocol.py':'8c878476fa62e63ff721b98085fe78672162ffcbe75f76c8c615958ec2239df0',
      'frontier_selector.py':'2bd6633b953e7e208040e48a36ed6842873462df32b2eaeca868ba2cd90f8ac2',
      'inputs.py':'1a4ad0375e831033201d11260a9bc74fbe2ee889ae1770be20ff5e5997de902b',
      'config.json':'cca5b7ffd849af4966cebfea94ae8b6a300eb50014ec624369b14a5067f1a12a'}
RECOVERY_PINS={**PINS,
      'worker.py':'dce14f33dd5b60ec2206b3f727fe324c6d43576d5d2ad5c3734d4599d7e2c0a0',
      'inputs.py':'042461ad4ae4fd12eec19756395c9429223c80fb99d057044eb367c4baaecb15',
      'config.json':'df3861fc2ae376303c86ff3c4262ff23eafcea850a305bac3cca1efc12ad3785'}
COMPACT_PINS={**RECOVERY_PINS,
      'worker.py':'528b8868188415fbc2417c53e8d955abf72772e32845865ebf1689b559abe07d',
      'inputs.py':'13bd7b21183ea90d6a59e1bfd2fb7175533bb4e5b047349d2113d7ed28ba8ed0',
      'config.json':'a9e74fdc255b5a1e5513270f20e89bffee4fbe70c558240c91d9e633615dab46',
      'compact_output.py':'5005b61f346695e9c053e375cb84d4539f03b13b930bf7e3421f3bba4624ee70'}
OUTPUT_FAILED_REPORT_SHA='61ab2279199f1548eb1b67558535ffcc1cd9793b012d0a7a70ce459f349f580b'
OUTPUT_FAILED_PARENT_SHA='a2a12688bc6a2aebdce696e1f01145299c2d398884cd7a9ff3a95a1fe246aeaa'
OUTPUT_FAILED_INDEX_SHA='748f9725b3be8523dae7768594bc529e4b590ba90f17eedb63af0e53e0ef0972'
FAILED_REPORT_SHA='0a2027b9a7cf8ca48cd3dcd8692dfee84b883e4bbe66e2a4d0d557ec3e215870'
FAILED_PARENT_SHA='938ba71ecc04d60bf426dd91f7daf31ac02442323691021adc68b32ec08a71c4'
FAILED_INDEX_SHA='07e36b83b13c9d30567e32913b47c6767efa78b5556472c356b37b91f929001f'
CHECKS=0;READS={}


def need(ok,label):
    global CHECKS
    CHECKS+=1
    if not ok:raise ValueError(label)


def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def semantic(value):return 'sha256:'+hashlib.sha256(canonical(value)).hexdigest()
def close(a,b,label):
    need(type(a) in (int,float) and type(b) in (int,float)
         and math.isfinite(a) and math.isfinite(b) and abs(a-b)<=1e-9,label)


def read_json(path,digest=None):
    path=Path(path)
    need(path.suffix=='.json' and not path.is_symlink(),'Only regular JSON evidence')
    need(0<path.stat().st_size<=8*1024**2,'Bounded JSON')
    raw=path.read_bytes();actual=hashlib.sha256(raw).hexdigest()
    if digest is not None:need(actual==digest,'Exact pinned JSON '+str(path))
    READS[str(path.relative_to(ROOT))]={'sha256':actual,'bytes':len(raw)}
    return json.loads(raw)


def read_artifact(directory,record):
    path=directory/record['path']
    if path.suffix!='.gz':return read_json(path,record['sha256'])
    need(path.name.endswith('.json.gz') and not path.is_symlink(),'Only regular compressed JSON diagnostics')
    need(record['encoding']=='gzip-json-canonical-v1' and record['decompressed_limit']==32*1024**2,
         'Declared lossless gzip diagnostics with32MiB decode limit')
    need(0<path.stat().st_size==record['bytes']<=3*1024**2
         and 0<record['decompressed_bytes']<=32*1024**2,'Separate compressed and decoded byte caps')
    blob=path.read_bytes();need(hashlib.sha256(blob).hexdigest()==record['sha256'],'Exact compressed bytes')
    with gzip.GzipFile(fileobj=io.BytesIO(blob),mode='rb') as stream:raw=stream.read(32*1024**2+1)
    need(len(raw)==record['decompressed_bytes'] and len(raw)<=32*1024**2
         and hashlib.sha256(raw).hexdigest()==record['decompressed_sha256'],'Exact complete bounded decompressed JSON')
    value=json.loads(raw)
    need(raw==canonical(value)+b'\n','Lossless original full JSON schema and canonical encoding')
    READS[str(path.relative_to(ROOT))]={k:record[k] for k in ('sha256','bytes','decompressed_sha256','decompressed_bytes')}
    return value


def geometry(endpoint,label):
    audit=endpoint['audit'];g=audit['geometry']
    need(endpoint['complete'] and audit['accepted'] and audit['complete_episode']
         and g['feasible'] and not g['failures'] and g['complete_tool_checked'] and g['frontier_checked'],
         label+' complete independent full-tool/exposure acceptance')
    need(audit['committed_history_hash']==endpoint['history_hash']
         and audit['source_hash']==endpoint['source_hash']
         and audit['decision_model_hash']==endpoint['decision_model_hash'],label+' exact independent certificate binding')
    need(endpoint['method']==label,label+' endpoint method')
    history=endpoint['history'];actions=endpoint['actions'];n=len(history)
    need(13<=n<=24 and actions==[h['action_id'] for h in history],label+' full absolute24 action denominator')
    need(('STOP' not in actions[:-1]) and (actions[-1]=='STOP' or (n==24 and 'STOP' not in actions)),
         label+' genuine STOP or horizon completion')
    outcomes=audit['outcomes']
    for key in ('reward','target_removed_mm3','normal_removed_mm3','complete_tool_path_length_mm'):
        field='total_reward' if key=='reward' else key
        close(math.fsum(h[key] for h in history),outcomes[field],label+' scalar increment sum '+key)
    tool=None;switches=0
    for row in history:
        if row['action_id']=='STOP':need(row['reward']==0 and row['microsteps']==0,label+' STOP0')
        else:
            need(row['outcome_scope']=='separate_evaluator_reference',label+' authoritative endpoint scope')
            switches+=int(tool is not None and row['tool_id']!=tool);tool=row['tool_id']
    need(outcomes['nonstop_actions']==sum(a!='STOP' for a in actions)
         and outcomes['tool_changes']==switches,label+' exact motion/switch count')
    need(outcomes['clinical_deficit_probability'] is None and outcomes['motor_surrogate'] is None
         and outcomes['language_surrogate'] is None,label+' function remains unassessed')
    return outcomes


def compact(row):
    fields=('action_id','reward','target_removed_mm3','normal_removed_mm3','tool_id','entry_mm','tip_mm',
            'physical_start_mm','complete_tool_path_length_mm','interaction_mode','outcome_scope')
    return {**{k:row[k] for k in fields if k in row},'history_row_hash':semantic(row),
        'microsteps':len(row.get('microsteps',())),
        'removed_indices_hash':semantic(row.get('removed_indices_native',[])),
        'contact_indices_hash':semantic(row.get('contact_indices_native',[]))}


def physical_compact(row):
    # Identity/scope hashes are checked separately. No coordinate/reward/cell
    # digest is omitted from this explicit physical comparison.
    return {k:v for k,v in row.items() if k not in ('action_id','history_row_hash','outcome_scope')}


def packet(record,history,step,model,current_tool):
    need(record['seal']==semantic({k:v for k,v in record.items() if k!='seal'})
         and record['schema']=='focused-public-goal-action-v1','Exact query/action packet seal')
    need(record['steps_taken']==step and record['current_tool']==current_tool
         and record['source_hash']==model['new_source_hash']
         and record['decision_model_hash']==model['new_task_model']
         and record['action_id']==history['action_id'],'Packet current source/tool/absolute step/action')
    if record['action_id']=='STOP':need(record['physical_action'] is None,'STOP has no physical ray')
    else:
        need(record['physical_action']=={k:history[k] for k in ('tool_id','entry_mm','tip_mm')},'Packet exact physical ray')
    need(isinstance(record['query_catalog_digest'],str) and record['query_catalog_digest'].startswith('sha256:'),
         'Current query catalog bound')
    focus=record['focus']
    if focus is not None:
        need(focus['schema']=='public-base-goal-focus-v1' and focus['source_hash']==model['new_source_hash']
             and focus['native_decision_model_hash']==model['new_native_config']
             and focus['source_state_hash']==record['source_state_hash'],'Focused public query uses current native state')


def preserved_failure(index):
    r=read_json(STAGE/'run-01/report.json',FAILED_REPORT_SHA)
    l=read_json(STAGE/'run-01/launcher.json',FAILED_PARENT_SHA)
    read_json(STAGE/'source-index.json',FAILED_INDEX_SHA)
    for path,digest in [('run-01/report.json',FAILED_REPORT_SHA),('run-01/launcher.json',FAILED_PARENT_SHA),
                        ('source-index.json',FAILED_INDEX_SHA)]:
        need(index['input_metadata'][str((STAGE/path).relative_to(ROOT))]==digest,'Frozen original instrumentation failure')
    b=r['native_budget'];c=l['owned_lifecycle']
    need(r['status']=='failed_or_incomplete' and r['source_unchanged']
         and r['failure']=={'phase':'saved_metadata_admission','type':'PlanningBudgetViolation',
            'message':'preview_wrapper_replaced','committed':False}
         and r['branches']=={} and r['costs']=={} and not r['plans_sealed_before_replay']
         and b['native_preview_entries'] is None and b['counting_reliable'] is False
         and b['observed_admissions_before_accounting_failure']==0 and b['native_preview_profile']['phases']=={}
         and r['native_transition_counts']=={'attempted':0,'completed':0,'raised':0}
         and r['total_frozen_policy_forwards']==0,'Original pre-source failure, unreliable count retained')
    need(l['status']=='incomplete_or_failed' and l['worker_exit_code']==1 and l['report_sha256']==FAILED_REPORT_SHA
         and r['source_index_sha256']==l['source_index_sha256']==FAILED_INDEX_SHA
         and c['verified'] and c['child_reaped'] and c['exit_code']==1 and not c['errors']
         and not c['remaining_owned_pids'] and not c['remaining_state_unknown'],'Original failed lifecycle cleanly reaped')
    return {'report_path':str((STAGE/'run-01/report.json').relative_to(ROOT)),
        'report_sha256':FAILED_REPORT_SHA,'launcher_sha256':FAILED_PARENT_SHA,'source_index_sha256':FAILED_INDEX_SHA,
        'failure':r['failure'],'native_preview_entries':None,'counting_reliable':False,
        'observed_admissions_before_accounting_failure':0,'native_transition_attempts':0,
        'worker_seconds':r['worker_seconds'],'parent_seconds':l['elapsed_seconds'],
        'sampled_peak_group_rss_bytes':l['sampled_peak_group_rss_bytes'],
        'cost_scope':'Paid failed attempt preserved; unreliable preview counter is null, never a certified zero.'}


def preserved_output_failure(index,original):
    r=read_json(STAGE/'recovery-01/report.json',OUTPUT_FAILED_REPORT_SHA)
    l=read_json(STAGE/'recovery-01/launcher.json',OUTPUT_FAILED_PARENT_SHA)
    read_json(STAGE/'recovery/source-index.json',OUTPUT_FAILED_INDEX_SHA)
    for path,digest in [('recovery-01/report.json',OUTPUT_FAILED_REPORT_SHA),
                        ('recovery-01/launcher.json',OUTPUT_FAILED_PARENT_SHA),
                        ('recovery/source-index.json',OUTPUT_FAILED_INDEX_SHA)]:
        need(index['input_metadata'][str((STAGE/path).relative_to(ROOT))]==digest,'Frozen output-cap failure')
    b=r['native_budget'];c=l['owned_lifecycle']
    need(r['status']=='failed_or_incomplete' and r['source_unchanged'] and r['branches']=={}
         and not r['plans_sealed_before_replay'] and r['total_frozen_policy_forwards']==0
         and r['failure']=={'phase':'focused_suffix_planning','type':'InterruptedError',
             'message':'Cohort output write would exceed bound','committed':False}
         and b['counting_reliable'] and b['native_preview_entries']==4926
         and r['native_transition_counts']=={'attempted':36,'completed':36,'raised':0}
         and r['preserved_failed_attempt']==original,'Unsealed output-cap failure and all original paid work retained')
    need(l['status']=='incomplete_or_failed' and l['worker_exit_code']==1 and l['report_sha256']==OUTPUT_FAILED_REPORT_SHA
         and r['source_index_sha256']==l['source_index_sha256']==OUTPUT_FAILED_INDEX_SHA
         and c['verified'] and c['child_reaped'] and c['exit_code']==1 and not c['errors']
         and not c['remaining_owned_pids'] and not c['remaining_state_unknown'],'Output-cap failure cleanly reaped')
    return {'report_path':str((STAGE/'recovery-01/report.json').relative_to(ROOT)),
        'report_sha256':OUTPUT_FAILED_REPORT_SHA,'launcher_sha256':OUTPUT_FAILED_PARENT_SHA,
        'source_index_sha256':OUTPUT_FAILED_INDEX_SHA,'failure':r['failure'],
        'native_preview_entries':4926,'counting_reliable':True,'native_transition_counts':r['native_transition_counts'],
        'worker_seconds':r['worker_seconds'],'parent_seconds':l['elapsed_seconds'],
        'sampled_peak_group_rss_bytes':l['sampled_peak_group_rss_bytes'],
        'original_instrumentation_failure':original,
        'cost_scope':'All failed source/prefix/suffix work retained; no sealed plans or independent endpoints, no accepted route.'}


def failed_result(report,launcher,index,args,out):
    partial=None
    if 'incomplete_selection' in report:
        p=report['incomplete_selection'];partial=read_artifact(out,p)
        need(not partial['complete'],'Capped/failed selection never relabeled complete')
    return {'report_sha256':args.report_sha256,'launcher_sha256':args.launcher_sha256,
        'source_index_sha256':args.index_sha256,'scope':'Failed/capped fixed frontier attempt retained',
        'comparison_admissible':False,'useful_gain':False,'report_status':report['status'],
        'supervision_status':launcher['status'],'owned_lifecycle':launcher.get('owned_lifecycle'),
        'failure':report.get('failure'),'partial_selection':None if partial is None else {k:partial.get(k) for k in (
            'complete','fixed_prefix_length','fixed_prefix_return','selected_prefix_length','selected_total_return',
            'selected_net_suffix_return','explored_total_return','failure','time_cap_reached','call_cap_reached')},
        'native_budget':report['native_budget'],'transition_counts':report['native_transition_counts'],
        'costs':report['costs'],'worker_seconds':report['worker_seconds'],'parent_seconds':launcher['elapsed_seconds'],
        'preserved_failed_attempt':report.get('preserved_failed_attempt'),
        'preserved_output_failed_attempt':report.get('preserved_output_failed_attempt'),'input_files':READS}


def run(args):
    runtime=STAGE/'recovery-compact';pins=COMPACT_PINS
    out=STAGE/'recovery-compact-01'
    report=read_json(out/'report.json',args.report_sha256)
    launcher=read_json(out/'launcher.json',args.launcher_sha256)
    index=read_json(runtime/'source-index.json',args.index_sha256)
    need(index['execution_released'] and report['source_index_sha256']==launcher['source_index_sha256']==args.index_sha256
         and launcher['report_sha256']==args.report_sha256,'Exact root-released terminal evidence')
    for name,digest in pins.items():
        path=runtime/name
        need(index['files'][str(path.relative_to(ROOT))]==digest
             and hashlib.sha256(path.read_bytes()).hexdigest()==digest,'Reviewed runtime bytes '+name)
    config=read_json(runtime/'config.json',pins['config.json']);caps=config['caps']
    need(ROOT/config['output_directory']==out,'Exact declared output directory')
    need(report['config']==config and config['subject']==SUBJECT and config['methods']==list(ARMS)
         and config['max_steps']==24 and config['prefix_nonstop_actions']==12
         and caps['native_transitions']==75 and caps['native_previews']==12000
         and caps['frozen_policy_forwards']==0 and caps['output_bytes']==3*1024**2
         and caps['whole_build_bytes']==8*1024**2,'Exact fixed frontier/horizon/budget declaration')
    need(index['checkpoints']=={} and len(index['public_arrays'])==5
         and all(not p.endswith(('.zip','.psckpt','.qckpt','.npy')) for p in index['input_metadata']),
         'Only original public anatomy payloads admitted by runtime; no model/cache')
    need(report['source_unchanged'] and not report['closed_roles_opened']
         and all(report[k]==0 for k in ('checkpoint_loads','cache_loads','head_optimizer_updates','total_frozen_policy_forwards')),
         'No source/split/learning/model/cache expansion')
    failed_origin=preserved_failure(index)
    output_failure=preserved_output_failure(index,failed_origin)
    if report['status']=='completed' or 'preserved_failed_attempt' in report:
        need(report['preserved_failed_attempt']==failed_origin,'Original failed attempt and costs retained exactly')
    if report['status']=='completed' or 'preserved_output_failed_attempt' in report:
        need(report['preserved_output_failed_attempt']==output_failure,'Unsealed output-cap failure and costs retained exactly')
    if report['status']!='completed':return failed_result(report,launcher,index,args,out)
    prior=read_json(ROOT/PRIOR,PRIOR_SHA)
    prior_audit=read_json(ROOT/PRIOR_AUDIT,PRIOR_AUDIT_SHA)
    oldplans=read_json(ROOT/OLD_PLANS,OLD_PLANS_SHA)
    old=prior['branches']['RETAINED_TAIL']
    need(prior_audit['audit_status']=='PASS_SAVED_EVIDENCE' and prior_audit['clean_supervision']
         and prior_audit['report_sha256']==PRIOR_SHA and prior_audit['original_full_route_parity'],
         'Accepted exact retained baseline with original failure preserved')
    need(index['input_metadata'][PRIOR]==PRIOR_SHA and index['input_metadata'][OLD_PLANS]==OLD_PLANS_SHA,
         'Original baseline/plan authority frozen')
    need(old['complete'] and len(old['actions'])==13 and old['actions'][-1]=='STOP'
         and old['audit']['outcomes']['nonstop_actions']==12,'Retained exact12-motion frontier')
    need(report['retained_result']=={'path':PRIOR,'sha256':PRIOR_SHA,'plans_sha256':OLD_PLANS_SHA,
        'plan_seal':oldplans['RETAINED_TAIL']['plan_seal'],'prior_costs_not_erased':prior['pair_plus_recovery_costs']},
         'Prior failed/recovered costs and baseline seal retained')
    need(set(report['branches'])==set(ARMS) and report['baseline_reproduced'],'Two complete retained endpoints')
    selected=report['branches']['FOCUSED_PREPARATION'];baseline=report['branches']['ORIGINAL_STOP'];explored=report['exploration']
    for name,path in [('ORIGINAL_STOP','original.json'),('FOCUSED_PREPARATION','selected.json')]:
        need(read_json(out/'evaluation'/path)==report['branches'][name],'Exact independently audited saved endpoint '+name)
    need(read_json(out/'evaluation/exploration.json')==explored,'Exact saved full fresh exploration endpoint')
    outcomes={m:geometry(v,m) for m,v in (*report['branches'].items(),('FOCUSED_EXPLORATION',explored))}
    need(baseline['actions']==old['actions'] and baseline['history']==old['history']
         and baseline['history_hash']==old['history_hash'] and outcomes['ORIGINAL_STOP']==old['audit']['outcomes'],
         'Exact retained original-catalog whole outcome/history baseline')
    for endpoint in (baseline,selected,explored):
        need(endpoint['original_admitted_world']==old['world'],'Original admitted public world retained')
        close(endpoint['audit']['outcomes']['total_reference_target_mm3'],old['audit']['outcomes']['total_reference_target_mm3'],
              'Entire unchanged target denominator')
    model=report['physical_model_join']
    need(model['native_config_except_declared_source_hash_equal']
         and model['post_exposure_authority']=={'same_prepared_object':True,'same_canonical_capability':True,'capability_serialized':False},
         'Actual same prepared authority and physical configuration source comparison')
    need(model['old_source_hash']==old['world']['source_hash']==baseline['source_hash']
         and model['old_task_model']==old['world']['decision_model_hash']==baseline['decision_model_hash']
         and model['old_native_config']==baseline['native_model_hash'],'Original model/source identities exact')
    need(model['new_source_hash']==selected['source_hash']==explored['source_hash']
         and model['new_task_model']==selected['decision_model_hash']==explored['decision_model_hash']
         and model['new_native_config']==selected['native_model_hash']==explored['native_model_hash']
         and model['old_source_hash']!=model['new_source_hash'],'Explicit changed catalog identities, never claimed identical DTOs')
    before=model['old_proposals'];after=model['new_proposals']
    intended={**before,'obstruction_opening':True,'tool_footprint_opening':True,
              'obstruction_page_size':8,'obstruction_selection_rule':'target_ray_coverage_v3'}
    need(after==intended and before['max_candidates']==after['max_candidates']==120,
         'Only explicitly declared focused-v3 proposal fields changed')
    prefix=read_json(out/'fixed-prefix.json');frontier=report['fixed_frontier']
    need(frontier==prefix['focused_frontier'] and frontier['seal']==semantic({k:v for k,v in frontier.items() if k!='seal'})
         and frontier['steps']==12 and frontier['current_tool']==old['history'][11]['tool_id'],
         'Actual12-motion sealed frontier/tool')
    close(frontier['total_reward'],outcomes['ORIGINAL_STOP']['total_reward'],'Prefix return equals original STOP incumbent')
    selection=read_artifact(out,report['selection'])
    need(selection['complete'] and not selection['call_cap_reached'] and not selection['time_cap_reached']
         and not selection.get('failure') and not selection['private_reference_used']
         and selection['all_preparation_costs_charged'],'Complete public-only suffix selection')
    need(selection['fixed_prefix_length']==12 and selection['fixed_frontier_seal']==frontier['seal'],
         'Authenticated frontier identity initializes incumbent')
    # planning_clone recomputes the nominal total with sum(); the original task
    # accumulated each reward. Keep exact identities, use the existing arithmetic
    # tolerance only for their one-ULP accumulation-order difference.
    close(selection['fixed_prefix_return'],frontier['total_reward'],'Authenticated frontier prefix return')
    nominal=selection['explored_history'];epackets=selection['explored_action_records'];chosen=selection['selected_action_records']
    need(len(nominal)==len(epackets)==len(explored['history'])==len(selection['explored_actions'])
         and selection['explored_actions']==explored['actions']==[p['action_id'] for p in epackets],
         'Every full explored action is genuinely freshly replayed')
    need(13<=len(epackets)<=24 and len(epackets)-12==selection['model_transition_calls']<=12,
         'At most twelve remaining transitions of absolute24, prefix never reset')
    need(prefix['focused_packets']==epackets[:12] and len(prefix['joins'])==12
         and frontier['action_packets_hash']==semantic(epackets[:12])
         and frontier['nominal_history_hash']==semantic(nominal[:12]),'Full prefix packet and nominal-history seals')
    running=0.;tool=None
    for i,(packet_record,n,h) in enumerate(zip(epackets,nominal,explored['history'])):
        packet(packet_record,n,i,model,tool)
        need(physical_compact(compact(n))==physical_compact(h),'Complete nominal/fresh physical history at every explored step')
        running+=n['reward']
        if n['action_id']!='STOP':tool=n['tool_id']
        if i<12:
            row=prefix['joins'][i]
            need(packet_record['focus'] is None and physical_compact(h)==physical_compact(old['history'][i]),
                 'Fixed prefix exact tool/path/target/outside/removed/contact/microsteps')
            need(row['step']==i+1 and row['old_action_id']==old['actions'][i]
                 and row['new_action_id']==packet_record['action_id'] and row['reward']==n['reward']
                 and row['physical_geometry']==packet_record['physical_action']
                 and row['original_full_row_hash']==old['history'][i]['history_row_hash']
                 and row['focused_nominal_full_row_hash']==semantic(n)
                 and row['physical_history_equal'] and row['base_physical_catalog_equal'],
                 'Original live/history and newly certified catalog physical prefix join')
            close(row['cumulative_return'],running,'Actual committed prefix return')
            if i:need(packet_record['source_state_hash']==prefix['joins'][i-1]['focused_native_state_hash'],
                      'Current prefix packet source state')
    need(prefix['joins'][-1]['mask_hashes']==frontier['committed_masks']
         and prefix['joins'][-1]['focused_native_state_hash']==frontier['native_state_hash']
         and prefix['joins'][-1]['original_native_state_hash']==prefix['original_frontier_state'],
         'Authenticated physically equal committed frontier masks and both identity hashes')
    need(epackets[12]['source_state_hash']==frontier['native_state_hash'],'Suffix starts at actual frontier state')
    close(running,selection['explored_total_return'],'All preparation/exploration rewards paid')
    close(selection['explored_net_suffix_return'],running-frontier['total_reward'],'Net explored return excludes initial positive prefix')
    nbest=selection['selected_prefix_length']
    need(12<=nbest<=24 and len(chosen)==nbest+int(nbest<24)
         and chosen[:nbest]==epackets[:nbest] and chosen[:12]==prefix['focused_packets'],
         'Complete best prefix preserves all12 original motions')
    if nbest<24:need(chosen[-1]['action_id']=='STOP' and chosen[-1]['steps_taken']==nbest,'Real selected-prefix STOP packet')
    else:need(chosen==epackets and chosen[-1]['action_id']!='STOP','Absolute horizon complete endpoint')
    close(selection['selected_total_return'],math.fsum(n['reward'] for n in nominal[:nbest]),'Chosen full reward direct sum')
    close(selection['selected_net_suffix_return'],selection['selected_total_return']-frontier['total_reward'],
          'Chosen net suffix subtracts fixed-prefix incumbent')
    close(selection['estimated_incremental_return'],selection['selected_total_return'],'Legacy field explicitly full total')
    need('prefix plus suffix total' in selection['legacy_estimated_incremental_return_scope'],'Misleading inherited name qualified')
    # Independently enumerate every actual explored prefix from the mandatory
    # frontier. No future cut can repay a prefix that was never actually explored.
    prefix_sums=[math.fsum(n['reward'] for n in nominal[:length]) for length in range(12,len(nominal)+1)]
    close(max(prefix_sums),selection['selected_total_return'],'Highest actual complete-prefix objective')
    need(all(v<=selection['selected_total_return']+1e-9 for v in prefix_sums),'No omitted better explored incumbent')
    need(selected['actions']==[p['action_id'] for p in chosen]
         and selected['history'][:nbest]==explored['history'][:nbest],
         'Selected authoritative history shares exact genuinely replayed prefix')
    tool=None
    for i,(p,h) in enumerate(zip(chosen,selected['history'])):
        packet(p,h,i,model,tool)
        if h['action_id']!='STOP':tool=h['tool_id']
    close(outcomes['FOCUSED_PREPARATION']['total_reward'],selection['selected_total_return'],'Selected endpoint total')
    close(outcomes['FOCUSED_EXPLORATION']['total_reward'],selection['explored_total_return'],'Independent full exploration total')
    for k,v in report['selection'].items():
        if k not in ('path','sha256','encoding','bytes','decompressed_bytes','decompressed_sha256','decompressed_limit'):need(v==selection[k],'Compact selection projection '+k)
    plans=read_json(out/'plans.json',report['plans_sealed_before_replay']['sha256'])
    need(set(plans)==set(ARMS),'Both full plans sealed before endpoint evaluation')
    for m,p in plans.items():
        need(p['seal']==semantic({k:v for k,v in p.items() if k!='seal'})
             ==report['plans_sealed_before_replay']['seals'][m],'All plan bytes bound')
    need(plans['ORIGINAL_STOP']['actions']==old['actions']
         and plans['ORIGINAL_STOP']['original_plan_seal']==oldplans['RETAINED_TAIL']['plan_seal'],
         'Exact original baseline plan')
    need(plans['FOCUSED_PREPARATION']['selected_action_records']==chosen
         and plans['FOCUSED_PREPARATION']['explored_action_records']==epackets
         and plans['FOCUSED_PREPARATION']['selected_total_return']==selection['selected_total_return']
         and plans['FOCUSED_PREPARATION']['frontier_seal']==frontier['seal'],
         'Full selected+discarded exploration plans sealed together')
    sidecars=read_artifact(out,report['sidecars']);records=sidecars['records']
    need(len(records)==report['sidecars']['unique_count']<=caps['unique_sidecars'],'Bounded deduplicated sidecar record set')
    for key,s in records.items():
        need(key==s['fingerprint']==semantic({k:v for k,v in s.items() if k!='fingerprint'}),
             'Independent first-failure sidecar hash')
        need(s['reason']=='SHAFT_BLOCKED_BY_REMAINING_NATIVE_TISSUE' and s['cell_limit']==16
             and s['retained_cell_count']==len(s['blocked_indices_native'])==min(s['blocked_cell_count'],16)
             and s['complete_first_failure_set']==(s['blocked_cell_count']<=16)
             and s['truncated']==(s['blocked_cell_count']>16) and s['prior_temporary_removals_committed'] is False,
             'Retained/truncated first rejected interval never treated as complete corridor or committed cut')
        need(s['source_hash'] in (model['old_source_hash'],model['new_source_hash'])
             and s['decision_model_hash'] in (model['old_native_config'],model['new_native_config']),
             'Sidecar belongs to declared source/native models only')
    for phase,counts in sidecars['phase_counts'].items():
        need(phase in ('original_frontier_witness','focused_suffix_planning')
             and all(k in records and type(n) is int and n>0 for k,n in counts.items()),'Deduplicated sidecar use counts')
    witness=read_json(out/'original-frontier.json')
    need(witness['original_native_state_hash']==prefix['original_frontier_state']
         and len(witness['outcomes'])==91,'Exact original post-prefix91-action frontier')
    for row in witness['outcomes']:
        need(not row['feasible'] and row['reason']=='SHAFT_BLOCKED_BY_REMAINING_NATIVE_TISSUE',
             'Original emitted frontier is still all shaft-blocked')
        s=records[row['sidecar_fingerprint']]
        need(s['source_state_hash']==prefix['original_frontier_state'] and s['source_hash']==model['old_source_hash']
             and s['tool']['tool_id']==row['tool_id'] and s['entry_mm']==row['entry_mm']
             and s['requested_tip_mm']==row['tip_mm'],'Exact current original-ray sidecar')
    states=selection['states'];need(len(states)==selection['model_transition_calls'],'Every suffix exploration state recorded')
    goal_calls=0;dep_calls=0;cleared=[];omitted=0;prerequisite_states=0
    for j,row in enumerate(states,12):
        need(row['depth']==j+1 and row['source_state_hash']==epackets[j]['source_state_hash']
             and row['goal_focus']==epackets[j]['focus'] and row['complete_page_enumeration'],
             'Absolute suffix state/current focus/page completeness')
        need(len(row['goals'])<=120 and len(row['pages'])<=5,'Fixed goal/page counts')
        goal_calls+=len(row['goals']);omitted+=row['goals_omitted_by_diagnostic_cap']
        if row['focus_explicitly_cleared'] is not None:cleared.append(row['focus_explicitly_cleared'])
        need([p['page_index'] for p in row['pages']]==list(range(len(row['pages']))),'All pages enumerated in order')
        action=row['selected'];need(action['action_id']==epackets[j]['action_id'],'Chosen current action matches actual exploration')
        if action['action_id']!='STOP':
            need(action['action_geometry']==epackets[j]['physical_action']
                 and action['actual_removed_cells']==nominal[j]['removed_indices_native']
                 and action['nominal_reward']==nominal[j]['reward'],'Exact chosen action geometry/removed cells/reward')
            close(action['cumulative_return'],math.fsum(v['reward'] for v in nominal[:j+1]),'Suffix full cumulative reward')
            if j+1<len(epackets):need(action['result_state_hash']==epackets[j+1]['source_state_hash'],'Successive real committed state chain')
        for goal in row['goals']:
            key=goal['sidecar_fingerprint']
            if key is None:continue
            s=records[key]
            need(s['source_state_hash']==row['source_state_hash'] and s['source_hash']==model['new_source_hash']
                 and s['decision_model_hash']==model['new_native_config']
                 and s['tool']['tool_id']==goal['tool_id'] and s['entry_mm']==goal['entry_mm']
                 and s['requested_tip_mm']==goal['tip_mm']
                 and sorted(s['blocked_indices_native'])==goal['retained_blocker_cells']
                 and s['complete_first_failure_set']==goal['first_failure_set_complete'],
                 'Every active/existing goal uses fresh exact physical first-failure evidence')
        if 'dependency_diagnostics' in row:
            d=row['dependency_diagnostics'];dep_calls+=len(d['probes'])
            need(d['dependency_depth']==2 and d['source_state_hash']==row['source_state_hash']
                 and d['pair_limit']==32 and not d['pair_cap_reached'] and len(d['probes'])<=32,
                 'Only one bounded current dependency edge')
            for probe in d['probes']:
                if probe['sidecar'] is not None:need(probe['sidecar']==records[probe['sidecar']['fingerprint']],
                    'Exact dependency sidecar held once in authoritative sidecar set')
            effect=d['selected_effect']
            if effect is not None:
                prerequisite_states+=1
                removed={tuple(v) for v in nominal[j]['removed_indices_native']}
                union={tuple(v) for v in d['retained_prerequisite_union']}
                need(effect['action_id']==action['action_id']
                     and {tuple(v) for v in effect['sampled_prerequisite_cells_removed']}==removed&union,
                     'Prerequisite effect is actual removed-cell intersection, no inferred unlock')
    need(goal_calls==selection['goal_diagnostic_preview_calls'] and dep_calls==selection['dependency_preview_calls']
         and goal_calls+dep_calls==selection['diagnostic_preview_calls'],'All exact-ray/dependency queries counted')
    shared=report['shared_replay_cost_scope'];extra=shared['selected_additional_STOP_transitions']
    need(shared['shared_fresh_prefix_transitions']==len(chosen)-int(chosen[-1]['action_id']=='STOP')
         and shared['exploration_transitions']==len(epackets) and extra in (0,1)
         and shared['distinct_complete_endpoints']==1+extra,'Genuine shared full replay and optional one STOP clone')
    counts=report['native_transition_counts'];expected=24+selection['model_transition_calls']+13+len(epackets)+extra
    need(counts['attempted']==counts['completed']==report['native_transition_attempts']==expected<=75
         and counts['raised']==0,'Both12 prefixes + suffix + original13 replay + full focused replay + optionalSTOP charged')
    costs=report['costs'];budget=report['native_budget']
    disjoint=('paired_fixed_prefix','original_frontier_witness','focused_suffix_planning',
              'original_fresh_full_replay_and_audit','focused_shared_fresh_full_replay',
              'focused_exploration_independent_audit','focused_selected_independent_audit')
    inner=sum(costs[k]['native_preview_entries'] for k in disjoint)
    need(costs['same045_source']['native_preview_entries']==inner
         and costs['same045_source.public_reconstruction']['native_preview_entries']+inner==budget['native_preview_entries']
         and costs['saved_metadata_admission']['native_preview_entries']==0,
         'Nonoverlapping source+phase costs join total; outer callback not counted twice')
    need(costs['focused_suffix_planning']['native_preview_entries']>=goal_calls+dep_calls
         and costs['original_frontier_witness']['native_preview_entries']>=91,'No free diagnostic previews')
    for row in costs.values():need(row.get('policy_forward_calls',0)==row.get('policy_forward_returned',0)==0,'Zero model calls all phases')
    need(budget['counting_reliable'] and budget['blocked_preview_attempts']==0
         and budget['status']=='complete_history_awaiting_independent_audit'
         and budget['native_preview_entries']<=12000,'Complete reliable preview budget')
    profiles=budget['native_preview_profile']['phases']
    need(sum(p['started'] for p in profiles.values())==sum(p['returned'] for p in profiles.values())==budget['native_preview_entries']
         and all(p['raised']==0 for p in profiles.values()),'Native started/returned/raised profile exact')
    da=outcomes['FOCUSED_PREPARATION'];db=outcomes['ORIGINAL_STOP']
    delta={k:da[k]-db[k] for k in ('positive_target_source_cells_removed','target_removed_mm3','total_reward',
             'normal_removed_mm3','complete_tool_path_length_mm','tool_changes','nonstop_actions',
             'cumulative_contacted_tissue_upper_bound_mm3')}
    useful=delta['positive_target_source_cells_removed']>0 and delta['target_removed_mm3']>1e-8 and delta['total_reward']>1e-8
    d=report['diagnosis'];need(d['comparison_admissible'] and d['tie_is_negative'] and d['tolerance']==1e-8
         and d['useful_gain'] is useful,'Independent prospective net suffix/target gate; ties negative')
    for key,field in (('additional_target_source_cells','positive_target_source_cells_removed'),
                      ('additional_target_mm3','target_removed_mm3'),('net_incremental_return','total_reward'),
                      ('additional_outside_mm3','normal_removed_mm3')):
        close(d[key],delta[field],'Independent useful-gate arithmetic '+key)
    close(selection['selected_net_suffix_return'],delta['total_reward'],'No positive-prefix leakage into success')
    visits=report['source_visits'];need(len(visits)==1 and visits[0]['subject']==SUBJECT and visits[0]['source_released'],
         'One released TRAIN source, no other patient role')
    cleanup=launcher.get('owned_lifecycle') or {}
    clean=bool(launcher['status']=='completed' and launcher.get('stop_reason') is None and launcher['worker_exit_code']==0
         and cleanup.get('verified') and cleanup.get('child_reaped') and cleanup.get('exit_code')==0
         and not cleanup.get('remaining_owned_pids') and not cleanup.get('remaining_state_unknown') and not cleanup.get('errors'))
    if launcher['status']=='completed':need(clean,'Verified clean supervision')
    need(launcher['elapsed_seconds']<caps['parent_wall_seconds'] and report['worker_seconds']<caps['work_seconds']
         and launcher['sampled_peak_group_rss_bytes']<=caps['worker_peak_rss_bytes']
         and report['self_peak_rss_bytes']<=caps['worker_peak_rss_bytes'] and launcher['rss_samples']>0,
         'Measured time/RSS caps')
    return {'scope':'One fixed TRAIN045 simulator frontier; new action-family capability, no learning/clinical claim',
        'report_sha256':args.report_sha256,'launcher_sha256':args.launcher_sha256,'source_index_sha256':args.index_sha256,
        'comparison_admissible':True,'clean_supervision':clean,'useful_gain':useful,'candidate_minus_original_STOP':delta,
        'outcomes':outcomes,'full_target_mm3':db['total_reference_target_mm3'],
        'fixed_prefix_length':12,'fixed_prefix_return':frontier['total_reward'],
        'selected_absolute_prefix_length':nbest,'selected_full_return':selection['selected_total_return'],
        'selected_net_suffix_return':selection['selected_net_suffix_return'],'explored_full_return':selection['explored_total_return'],
        'explored_net_suffix_return':selection['explored_net_suffix_return'],'explored_transitions':len(epackets),
        'goal_diagnostic_previews':goal_calls,'dependency_previews':dep_calls,'prerequisite_preparations':prerequisite_states,
        'focus_explicitly_cleared':cleared,'diagnostic_goals_omitted':omitted,'sidecars_unique':len(records),
        'original_frontier':{'emitted':91,'shaft_blocked':91,'scope':'current emitted proposals only'},
        'plans_sealed_before_replay':report['plans_sealed_before_replay'],'shared_fresh_replay':shared,
        'total_previews':budget['native_preview_entries'],'total_transition_counts':counts,'model_calls':0,
        'parent_seconds':launcher['elapsed_seconds'],'worker_seconds':report['worker_seconds'],
        'sampled_peak_rss_bytes':launcher['sampled_peak_group_rss_bytes'],'costs_inclusive':costs,
        'prior_costs_not_erased':prior['pair_plus_recovery_costs'],
        'preserved_failed_attempt':failed_origin,'preserved_output_failed_attempt':output_failure,
        'frontier_iteration_paid_worker_seconds':report['worker_seconds']+failed_origin['worker_seconds']+output_failure['worker_seconds'],
        'frontier_iteration_paid_parent_seconds':launcher['elapsed_seconds']+failed_origin['parent_seconds']+output_failure['parent_seconds'],
        'frontier_iteration_known_preview_entries':budget['native_preview_entries']+output_failure['native_preview_entries'],
        'frontier_iteration_total_transition_attempts':counts['attempted']+output_failure['native_transition_counts']['attempted'],
        'frontier_iteration_preview_count_scope':'Successful attempt count exact; original attempt remains null/unreliable with zero observed admissions, not an independently certified total',
        'limitations':['Independent certificates and exact saved joins checked; no native/array/model rerun',
          'New catalog/source/task identities are declared; this is not matched-observation model performance',
          'Reference labels are evaluation only; selector consumes public nominal model and current first-failure evidence',
          'Unknown coverage and full target denominator remain unchanged; no functional/clinical evidence',
          'First-failure samples and bounded catalogs do not prove complete obstruction or global reachability',
          'Exploration and original prefix costs remain paid even if selected endpoint is original STOP',
          'Inclusive phases nest; shared replay prefix is charged once and both endpoints independently checked'],
        'input_files':READS}


def main():
    parser=argparse.ArgumentParser()
    for k in ('report-sha256','launcher-sha256','index-sha256','output'):parser.add_argument('--'+k,required=True)
    args=parser.parse_args();target=Path(args.output).resolve()
    need(target.parent==HERE and not target.exists(),'Exclusive receipt inside owned independent scope')
    start=time.monotonic()
    def timeout(*_):raise TimeoutError('Saved audit30s cap')
    signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,30.)
    try:result=run(args);result['audit_status']='PASS_SAVED_EVIDENCE';code=0
    except BaseException as error:
        result={'audit_status':'FAILED_REVIEWER_CHECK','failure':{'type':type(error).__name__,'message':str(error)},'input_files':READS};code=1
    signal.setitimer(signal.ITIMER_REAL,0.)
    result.update(checks=CHECKS,seconds=time.monotonic()-start,native_calls=0,model_calls=0,
        auditor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    raw=json.dumps(result,sort_keys=True,indent=2,allow_nan=False).encode()+b'\n'
    need(len(raw)<=64*1024,'64KiB receipt cap')
    with target.open('xb') as f:f.write(raw)
    print(json.dumps({k:result[k] for k in ('audit_status','checks','seconds','native_calls','model_calls')}))
    return code

if __name__=='__main__':raise SystemExit(main())
