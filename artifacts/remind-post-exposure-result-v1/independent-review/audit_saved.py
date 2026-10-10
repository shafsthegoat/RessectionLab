"""Saved JSON/source-only arithmetic audit; no project imports or array reads."""
import hashlib,json,math,os,resource,signal,stat,time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'build/remind-post-exposure-feasibility-v1'
RUN=BASE/'attempt-01'
OUT=Path(__file__).resolve().parent
started=time.perf_counter();checks=0;pins={}
signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('60s saved audit cap')))
signal.alarm(60)
def need(v,reason):
    global checks
    checks+=1
    if not v:raise AssertionError(reason)
def raw(path,expected=None):
    path=Path(path);path=path if path.is_absolute() else ROOT/path
    need(path.is_relative_to(ROOT) and '..' not in path.parts and path.suffix in {'.py','.json','.txt','.log'},'saved text only')
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    try:
        s=os.fstat(fd);need(stat.S_ISREG(s.st_mode) and s.st_size<=4*1024**2,'bounded regular text')
        with os.fdopen(fd,'rb',closefd=False) as stream:data=stream.read(4*1024**2+1)
        need(len(data)<=4*1024**2,'bounded read')
    finally:os.close(fd)
    digest=hashlib.sha256(data).hexdigest()
    if expected:need(digest==expected,'hash '+str(path))
    rel=str(path.relative_to(ROOT));need(rel not in pins or pins[rel]['sha256']==digest,'changed input')
    pins[rel]={'sha256':digest,'bytes':len(data)}
    return data
def read(path,expected=None):return json.loads(raw(path,expected))
def sd(v):return 'sha256:'+hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def near(a,b,label):need(math.isfinite(a) and math.isfinite(b) and math.isclose(a,b,rel_tol=1e-7,abs_tol=1e-7),label)
def cells(rows):
    s={tuple(r) for r in rows};need(len(s)==len(rows) and all(len(r)==3 and all(type(v)is int for v in r) for r in s),'unique integer cells');return s
def det(m):return m[0][0]*(m[1][1]*m[2][2]-m[1][2]*m[2][1])-m[0][1]*(m[1][0]*m[2][2]-m[1][2]*m[2][0])+m[0][2]*(m[1][0]*m[2][1]-m[1][1]*m[2][0])

release=read(BASE/'root-release.json','8315099d95b16a9dacac54a65c4d84d6385e458534b6dab26b73a37555d4af1a')
prepared=read(BASE/'prepared-release.json','038d66126aa55a595aadb316871dbb12f0e4d6a85b3be078e2b1afb893710f5b')
need(release['execution_released']is True and prepared['execution_released']is False,'separate release')
need({**release,'execution_released':False,'expected_head':None}==prepared,'only declared release delta')
index=read(BASE/'source-index.json','4ac60917a28368056b1b0d07481a88f447be4116ec677fed1bb31dd2766c7b50')
need(index['canonical_head']==release['expected_head']=='d583008cb1ebc36b4ed34b5f722f3b4f8cc959ec','source commit')
for name,digest in index['files'].items():raw(name,digest)
result=read(RUN/'result.json','7a57fd74f58b15d6975be4459fe4f46e6396dfa9e2f3c0fec0ef68cf5e550358')
receipt=read(BASE/'attempt-01.supervision/receipt.json')
ledger=read(RUN/'construction-ledger.json')
need(ledger==result['cases'],'complete ledger')
subjects=['ReMIND-002','ReMIND-015','ReMIND-018','ReMIND-045']
need([x['patient_id'] for x in ledger]==subjects==release['subjects']==receipt['planned_subjects'],'four fixed denominator')
need(result['status']==receipt['result_status']=='post_exposure_batch_complete' and receipt['status']=='complete','terminal success')
need(receipt['exit_code']==0 and receipt['worker_termination_confirmed']is True and not receipt['remaining_owned_pids'] and not receipt['cleanup_errors'] and receipt['stop_reason']is None,'clean owned terminal')
need(receipt['release_sha256']==result['release_sha256']==hashlib.sha256(raw(BASE/'root-release.json')).hexdigest(),'release joins')
need(receipt['result_sha256']==hashlib.sha256(raw(RUN/'result.json')).hexdigest(),'receipt result seal')
for field,limit in [('elapsed_seconds','parent_seconds'),('sampled_peak_rss_bytes','sampled_rss_bytes'),('output_bytes','output_bytes')]:need(receipt[field]<release['caps'][limit],'parent cap '+field)
need(result['elapsed_seconds']<300 and receipt['samples']>0,'worker cap/sampling')
need(all(result[k]==0 for k in ('model_calls','optimizer_calls','checkpoint_loads','forbidden_search_calls','blocked_file_or_external_calls','source_arrays_written')),'zero forbidden work')
need(result['training_admitted']is False and result['clinical_claim']is False and result['failed_cases_replaced']is False,'scope')
budget=result['native_budget'];need(budget['status']=='complete_history_awaiting_independent_audit' and budget['counting_reliable']is True and budget['history_complete_caller_attestation']is True and budget['failure']is None and budget['blocked_preview_attempts']==0,'budget')
need(budget['native_preview_entries']==4823<=40000,'preview cap')
public=read(release['public_index']['path'],release['public_index']['sha256'])
expected_loads=[];summaries=[];total_steps=0;total_previews=0
for row,public_row in zip(ledger,public['cases']):
    subject=row['patient_id'];folder=RUN/subject
    manifest=read(public_row['path'],public_row['sha256'])
    expected_loads.extend(str(Path(manifest['input_files'][key]['path']).resolve()) for key in ('image','supplied_support','supplied_whole_tumor','whole_tumor_domain','supplied_support_domain'))
    need(read(folder/'construction-result.json')==row,'case result exact')
    m=read(folder/'episode-metrics.json');e=read(folder/'independent-episode.json');plan=read(folder/'greedy-plan.json')
    inv=read(folder/'initial-inventory.json');bindings=read(folder/'admitted-public-bindings.json');ctx=read(folder/'context.json')
    p=row['post_exposure_start'];domain=row['source_and_simulated_domains'];extent=row['full_target_extent'];o=e['outcomes'];a=plan['accounting'];h=m['history']
    need(row['status']=='completed_fixed_post_exposure_route' and row['route_complete'] and row['independent_episode_accepted'] and e['accepted'] and e['complete_episode'],'accepted complete row')
    need(e['geometry']['feasible'] and not e['geometry']['failures'] and e['geometry']['complete_tool_checked'] and e['geometry']['frontier_checked'],'saved complete geometry audit')
    need(m['terminated'] and not m['planning_estimator_only'] and 1<=m['steps']==len(h)<=24,'committed complete history')
    actions=[r['action_id'] for r in h]
    need(actions==plan['actions']==row['actions'] and actions[-1]=='STOP' and 'STOP' not in actions[:-1],'route joins and terminal STOP')
    need(sd(h)==e['committed_history_hash'],'history seal')
    need(row['source_hash']==m['source_hash']==e['source_hash']==bindings['source_binding']['source_hash']==ctx['source_hash'],'source joins')
    need(m['reference_hash']==e['reference_hash'] and m['decision_model_hash']==e['decision_model_hash']==ctx['decision_model_hash'],'objective/model joins')
    need(ctx['role']=='TRAIN' and not ctx['private_reference_in_task'] and not ctx['clinical_claim'],'public TRAIN only')
    need(bindings['protocol']['search']=={'max_calls':1,'beam_width':1,'seconds':60},'actual protocol budget')
    need(p==bindings['source_binding']['post_exposure']==ctx['post_exposure'],'post condition joins')
    need(sd(p)==domain['post_exposure_condition_hash'],'post condition seal')
    need(all(p[k]is False for k in ('source_domains_extended','source_material_deleted','source_target_deleted','physical_air_or_operative_opening_observed')),'explicit simulated start')
    need(p['source_Ds_hash']==domain['source_support_domain_hash'] and p['interaction_domain_hash']==domain['interaction_domain_hash'],'Ds/D separate hashes')
    need(p['source_S_hash']==extent['occupancy_derivation']['source_support_hash'] and p['occupancy_hash']==extent['occupancy_derivation']['derived_occupancy_hash'],'S/O ancestry')
    need(p['native_affine_hash']==bindings['native_grid_reconciliation']['derived_affine_hash'],'native frame')
    need(p['external_encounter_hash_rule']=='semantic_digest(sorted_unique_native_index_lists)' and p['seed_cells']>0 and p['initial_free_cells']>=p['seed_cells'],'E/K/F0 declaration')
    need(row['initial_removed_cells']==row['initial_contact_cells']==0 and row['live_source_or_task_after_cleanup']==0,'no initial cuts and case cleanup')
    need(not domain['source_domain_extended'] and not domain['anatomical_or_material_validation'] and domain['interaction_domain_rule']=='Ds OR (T > 0)','domain assumption')
    need(domain['source_support_unknown_voxels']-domain['target_in_unknown_support_voxels']==domain['unavailable_interaction_voxels'],'unknown partition')
    need(extent['target_modified']is False and extent['fraction_denominator']=='entire_unchanged_supplied_region','full immutable target')
    sampling=manifest['public_label_resampling']['whole_tumor'];need(extent['full_region_positive_voxels']==sampling['saved_positive_voxels'],'full placed target preserved')
    near(o['total_reference_target_mm3'],extent['full_region_membership_mm3'],'full denominator')
    need(inv['omitted_count']==0 and inv['accepted_count']==row['accepted_count'] and inv['emitted_count']==row['emitted_count']<=120,'initial inventory joins')
    need(inv['accepted_count']==sum(x['feasible'] for x in inv['emitted']),'initial feasible count')
    need(a==row['greedy_accounting'] and a['complete'] and a['time_budget_seconds']==60 and a['planning_seconds']<60,'greedy accounting')
    need(a['model_transition_calls']==len(a['decisions'])==len(h)==row['steps_taken'],'step joins')
    for d,record in zip(a['decisions'],h):
        scores=d['scores'];need(scores[0]['action_id']=='STOP' and scores[0]['reward']==0,'STOP tie precedence')
        best=max(scores,key=lambda s:s['reward'])
        need(best['action_id']==d['selected_action_id']==record['action_id'],'greedy exact maximum')
        near(best['reward'],record['reward'],'nominal/authoritative reward')
        need(d['all_current_legal_actions_scored'] and d['legal_nonstop_actions']==d['scored_nonstop_actions']==len(scores)-1,'all emitted legal scores')
    removed=set();contacts=set();previous_tool=None;sumreward=sumtarget=sumnormal=sumpath=0.;changes=0;unknowns=set();external_counts=[]
    volume=e['geometry']['source_voxel_volume_mm3'];near(volume,abs(det(bindings['native_grid_reconciliation']['derived_affine_ras_mm'])),'native cell volume')
    for record in h:
        if record['action_id']=='STOP':
            need(all(record[k]==0 for k in ('reward','target_removed_mm3','normal_removed_mm3','insertion_distance_mm','complete_tool_path_length_mm')),'STOP zero');continue
        R=cells(record['removed_indices_native']);C=cells(record['contact_indices_native'])
        need(not (removed&R),'no duplicate reward');removed|=R;contacts|=C
        microR=set();microC=set();last=record['physical_start_mm']
        for micro in record['microsteps']:
            for x,y in zip(last,micro['tip_start_mm']):near(x,y,'continuous saved microsteps')
            last=micro['tip_end_mm'];microR|=cells(micro['removed_indices_native']);microC|=cells(micro['contact_indices_native'])
        need(microR==R and microC==C,'macro/micro union')
        for x,y in zip(last,record['tip_mm']):near(x,y,'last physical tip')
        need(record['post_exposure_condition_hash']==sd(p) and record['inter_insertion_transfer']=='unassessed; each axial primitive withdraws identically','per insertion bound scope')
        tipr,shaftr,tipl={'generic_suction':(1.,1.4,3.),'generic_aspirator':(1.7,2.7,4.)}[record['tool_id']]
        axis=record['axis_unit'];alignment=sum(x*y for x,y in zip(axis,p['access']['normal_inward']));need(alignment>0,'inward axis')
        distance0=(max(tipr,shaftr-tipl*alignment)+1e-8)/alignment
        for saved,entry,direction in zip(record['physical_start_mm'],record['entry_mm'],axis):near(saved,entry-distance0*direction,'preentry formula')
        dist=math.dist(record['physical_start_mm'],record['tip_mm']);near(dist,record['insertion_distance_mm'],'approach charged');near(2*dist,record['complete_tool_path_length_mm'],'reverse charged')
        changed=int(previous_tool is not None and previous_tool!=record['tool_id']);changes+=changed;previous_tool=record['tool_id']
        reward=record['target_removed_mm3']-.2*record['normal_removed_mm3']-.03-.001*2*dist-.03*changed
        near(reward,record['reward'],'unchanged reward');near(len(R)*volume,record['target_removed_mm3']+record['normal_removed_mm3'],'removal partition')
        sumreward+=reward;sumtarget+=record['target_removed_mm3'];sumnormal+=record['normal_removed_mm3'];sumpath+=2*dist
        unknowns.update(record['geometry_unknowns']);external_counts.append(record['external_workspace_encounter_cells'])
    need(o['nonstop_actions']==len(h)-1==e['geometry']['action_count'],'native action count')
    for key,value in [('target_removed_mm3',sumtarget),('normal_removed_mm3',sumnormal),('total_reward',sumreward),('simulated_removed_volume_mm3',len(removed)*volume),('cumulative_contacted_tissue_upper_bound_mm3',len(contacts)*volume),('currently_retained_contacted_tissue_upper_bound_mm3',len(contacts-removed)*volume)]:near(o[key],value,key);near(m[key],value,'metrics '+key)
    near(o['complete_tool_path_length_mm'],sumpath,'total motor path');near(o['reference_target_fraction_removed'],sumtarget/o['total_reference_target_mm3'],'full target fraction')
    near(e['geometry']['contained_source_tissue_volume_mm3'],len(removed)*volume,'saved independent containment')
    need(o==row['full_route_outcomes'] and o['tool_changes']==changes,'outcome exact join')
    need(o['normal_removed_mm3']==0 and o['positive_target_source_cells_removed']==len(removed),'all saved removals target-positive as independently reported')
    need(e['target_access_success']==(len(removed)>=1) and o['partial_contact_reward_weight']==0 and o['clinical_deficit_probability']is None and o['motor_surrogate']is None and o['language_surrogate']is None,'no clinical/contact reward')
    total_steps+=len(h);total_previews+=row['native_preview_entries']
    summaries.append({'subject':subject,'initial_emitted':inv['emitted_count'],'initial_legal':inv['accepted_count'],'initial_cap_omissions':inv['omitted_count'],'nonstop_actions':len(h)-1,'removed_cells':len(removed),'full_target_cells':extent['full_region_positive_voxels'],'native_target_positive_cells':sampling['source_positive_voxels_from_bound_execution'],'target_removed_mm3':sumtarget,'outside_target_removed_mm3':sumnormal,'full_target_fraction':o['reference_target_fraction_removed'],'reward':sumreward,'path_mm':sumpath,'contact_cells':len(contacts),'retained_contact_cells':len(contacts-removed),'seed_cells':p['seed_cells'],'initial_free_cells':p['initial_free_cells'],'exterior_unknown_cells':p['external_unknown_cells'],'target_in_unknown_Ds_cells':domain['target_in_unknown_support_voxels'],'recorded_geometry_unknowns':sorted(unknowns),'per_insertion_E_encounter_count_sum_not_unique':sum(external_counts),'native_previews':row['native_preview_entries'],'greedy_seconds':a['planning_seconds']})
need(result['public_array_loads']==expected_loads and len(expected_loads)==20,'exact declared public load log')
need(total_steps==result['transition_calls']==result['planning_transition_calls']==29 and result['planning_clone_calls']==result['greedy_calls']==4,'exact all-case counters')
need(total_previews==budget['native_preview_entries'],'preview sum')
output_size=0
for path in RUN.rglob('*'):
    need(not path.is_symlink(),'no output symlink')
    if path.is_file():output_size+=len(raw(path))
need(output_size==receipt['output_bytes'],'exact closed output bytes')
for name,record in list(pins.items()):raw(name,record['sha256'])
elapsed=time.perf_counter()-started;peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
need(elapsed<60 and peak<512*1024**2,'saved audit resources')
summary={'status':'PASS_saved_metadata_and_arithmetic_only','checks':checks,'input_files':len(pins),'source_index_entries':len(index['files']),'audit_seconds':elapsed,'audit_peak_rss_bytes':peak,'result_sha256':receipt['result_sha256'],'parent_seconds':receipt['elapsed_seconds'],'worker_seconds':result['elapsed_seconds'],'parent_sampled_peak_rss_bytes':receipt['sampled_peak_rss_bytes'],'native_previews':total_previews,'planning_transitions':total_steps,'authoritative_transitions':total_steps,'greedy_calls':4,'model_updates_or_checkpoint_loads':0,'cases':summaries,'limitations':['No arrays reopened, cell memberships/geometry/E/K/F0 not recomputed; rely on source-reviewed executed independent geometric checker.','Zero catalog omissions verified for initial inventories only; later greedy decisions retain legal scores but not full proposal ledgers.','New assumed post-exposure condition changes aperture/start/free-space assumptions; historical starting-state failures remain valid.','Unknown exterior workspace and inter-insertion transfer remain unassessed; no clinical safety, surgery, full resection, global optimum or training admission claim.']}
for name,value in [('audit-result.json',summary),('input-hashes.json',pins)]:
    with (OUT/name).open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')
print(json.dumps({'status':summary['status'],'checks':checks,'inputs':len(pins),'elapsed_seconds':elapsed,'maxrss_bytes':peak}))
