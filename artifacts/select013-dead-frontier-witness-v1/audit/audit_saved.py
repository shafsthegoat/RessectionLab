"""Bounded independent saved JSON audit; no arrays, weights or scientific imports."""
import hashlib, json, math, os, resource, signal, stat, struct, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT/'build/select013-dead-frontier-witness-v1'
OUT = RUN/'attempt-01'
DEST = Path(__file__).resolve().parent
CHECKS = 0
READS = {}


def need(condition, reason):
    global CHECKS
    CHECKS += 1
    if not condition:
        raise ValueError(reason)
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > 512*1024**2:
        raise MemoryError('512 MiB saved audit cap')


def semantic(value):
    raw = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return 'sha256:' + hashlib.sha256(raw).hexdigest()


def bytes_read(path, expected=None):
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    relative = path.relative_to(ROOT)
    need('..' not in relative.parts and path.suffix in ('.json','.py'), 'Only workspace JSON/source allowed; no weight/array payload')
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    try:
        info=os.fstat(fd)
        need(stat.S_ISREG(info.st_mode) and 0<info.st_size<=8*1024**2, 'Bounded regular evidence')
        with os.fdopen(fd,'rb',closefd=False) as stream:raw=stream.read(8*1024**2+1)
        need(len(raw)==info.st_size and len(raw)<=8*1024**2,'Stable bounded read')
    finally:os.close(fd)
    digest = hashlib.sha256(raw).hexdigest()
    need(expected is None or digest == expected, 'Saved JSON digest mismatch: ' + str(relative))
    need(str(relative) not in READS or READS[str(relative)]['sha256']==digest,'Repeated evidence changed')
    READS[str(relative)] = {'sha256': digest, 'bytes': len(raw)}
    need(sum(r['bytes'] for r in READS.values()) <= 128*1024**2, 'Metadata read bound exceeded')
    return raw


def read(path, expected=None):
    need(Path(path).suffix=='.json','JSON only')
    return json.loads(bytes_read(path,expected),parse_constant=lambda x:(_ for _ in ()).throw(ValueError(x)))


def pinned(ref):
    return read(ref['path'], ref['sha256'])


def digest(path):
    return READS[str(Path(path).relative_to(ROOT))]['sha256']


def history_identity(rows):
    return semantic([{k: v for k, v in row.items() if k != 'outcome_scope'} for row in rows])


def close(a, b):
    return math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-9)


def array_hash(rows, dtype='<i8'):
    """Recreate core.array_digest from small saved lists, without NumPy."""
    need(isinstance(rows,list) and all(isinstance(r,list) for r in rows),'Saved rectangular list required')
    width=len(rows[0]) if rows else 3
    need(all(len(r)==width for r in rows),'Rectangular saved evidence')
    raw=json.dumps({'shape':[len(rows),width],'dtype':dtype},sort_keys=True).encode()
    fmt='<q' if dtype=='<i8' else '<d'
    for row in rows:
        for value in row:raw+=struct.pack(fmt,value)
    return 'sha256:'+hashlib.sha256(raw).hexdigest()


def cells(rows):
    need(isinstance(rows,list) and all(isinstance(r,list) and len(r)==3
         and all(type(v) is int for v in r) for r in rows),'Native index list')
    out={tuple(r) for r in rows}
    need(len(out)==len(rows),'No duplicate native indices')
    need(all(0<=v<n for c in out for v,n in zip(c,(131,154,117))),'Native indices in source grid')
    return out


def audit():
    pins={'result':'e23d75b1171936af24ff0834e37a46e8426bd8b28a0a5f5017848cd78e65c0a2',
          'receipt':'085f40abf2ccad5133e8ddd283c97a80ec6a57d1d2acf240e52b3cec7da96cb0',
          'release':'8866a10e14574b79c46f93be9f5f4f4f8f3e827e9f82c94a5742a18d988a1477',
          'source_index':'71879696c1194859f8280b3865804c18a545d09011929fb0a8c39a7a4274a008',
          'costs':'a237c5b0467a12552355ed2bd9319a054c9e1853ff201b9cdf87314678c78ce0'}
    parent=read(RUN/'attempt-01.supervision/receipt.json',pins['receipt'])
    need(parent['status']=='complete' and parent['exit_code']==0 and parent['stop_reason'] is None
         and parent['worker_termination_confirmed'] and not parent['final_owned_pids']
         and not parent['cleanup_errors'],'Clean owned terminal precedes saved reads')
    release=read(RUN/'root-release.json',pins['release'])
    index=pinned(release['source_index']);need(digest(RUN/'source-index.json')==pins['source_index'],'Exact index')
    need(index['head']==release['expected_head']=='593a148fe72d57641bb428c60eecaabbe768369e','Executed source HEAD')
    guard=read(DEST/'current-source-guard.json','d16f97740351c571de3ded98081b9a4215b5fecb457b26ad1ee58385c09bb0b7')
    need(guard['all_sources_match'] and guard['head']==index['head'] and guard['source_index_sha256']==pins['source_index']
         and {p:v['sha256'] for p,v in guard['source_files'].items()}==index['source_files']
         and len(index['source_files'])==94,'Independent 94-source guard captured before root released source hold')
    # Root may now change admission/factory. Historical source guard is immutable evidence;
    # do not substitute successor source bytes for the executed closure.
    for path,sha in index['metadata_files'].items():read(path,sha)
    need(len(index['metadata_files'])==50,'Exact metadata closure')
    result=read(OUT/'result.json',pins['result']);cost=read(OUT/'costs.json',pins['costs'])
    worker=read(RUN/'attempt-01.supervision/worker-final.json',parent['worker_final_sha256'])
    need(parent['release_sha256']==pins['release'] and parent['source_index']==release['source_index']
         and parent['result_sha256']==worker['result_sha256']==worker['canonical_result_sha256']==pins['result'],
         'Parent/worker/result joins')
    need(worker['status']=='complete_owned_SELECT013_dead_frontier'
         and result['status']==parent['canonical_result_status']=='complete_fixed_dead_frontier_diagnostic','Terminal identity')
    need(release['status']=='released_one_attempt' and release['arms']==['fixed_dead_frontier']
         and release['attempts']==1 and release['automatic_retry'] is False,'One diagnostic attempt')
    need(release['limits']=={'cells_per_sidecar':32,'classified_unique_cells':128,'closure_STOP_calls':1,
         'diagnostic_previews':120,'fixed_prefix_commits':11,'model_forwards':0,'native_previews':2000,'threads':1,'updates':0},'Owned limits')
    need(parent['caps']=={'attempts':1,'automatic_retry':False,'hard_total_wall_seconds':210,
         'output_bytes':33554432,'output_sample_interval_seconds':1.0,'sample_interval_seconds':.2,
         'sampled_owned_tree_rss_bytes':3221225472,'supervision_bytes':16777216,'threads':1,
         'worker_deadline_seconds':180,'worker_log_bytes':8388608},'Unchanged supervision caps')
    need(release['worker_seconds']==180 and release['parent_seconds']==210
         and result['wall_seconds']<180 and worker['wall_seconds']<180 and parent['elapsed_seconds']<210
         and parent['sampled_peak_rss_bytes']<=3*1024**3 and parent['samples']>0
         and parent['output_bytes']<=32*1024**2 and parent['supervision_bytes_before_receipt']<16*1024**2,'Actual costs within caps')
    need(result['subject']==parent['SELECT_subject']=='ReMIND-013' and result['role']=='SELECT'
         and result['planned_SELECT_denominator']==parent['planned_SELECT_denominator']==2
         and result['held_cases']==parent['held_cases']==['ReMIND-037'],'Fixed SELECT denominator')
    for key in ('held_inputs_opened','EVAL_opened','accepted_plan_claim','independent_replay_certificate'):
        need(result[key] is False,'Forbidden claim/input: '+key)
    for key in ('checkpoint_loads','model_forwards','updates','optimizer_attempts','gradient_attempts','private_reference_reads'):
        need(type(result[key]) is int and result[key]==0,'No model/training/private work: '+key)
    need(release['models_or_weights_loaded'] is False and release['training_or_private_reference_execution'] is False
         and parent['optimizer_updates_on_SELECT']==0 and parent['EVAL_execution'] is False
         and parent['private_reference_execution'] is False,'No runtime model capability')
    public=pinned(release['public_index']);oldpublic=pinned(public['original_select_ledger'])
    need(public['cases'][1]==oldpublic['cases'][1] and public['cases'][1]['patient_id']=='ReMIND-037'
         and public['cases'][1]['path'] is None and public['planned_case_denominator']==2,'Held037 unchanged')
    manifest=pinned(public['cases'][0]);need(public['cases'][0]['sha256']=='841261035a7d7be5c55984e717b8d65b712065a708e9260231ebcd49cc4d700e'
         and set(manifest['input_files'])=={'image','supplied_support','supplied_whole_tumor','whole_tumor_domain','supplied_support_domain'},'Exact qualified five-array manifest, metadata only')
    prior={k:pinned(v) for k,v in release['prior_pins'].items()}
    world=read(OUT/'world-control.json');need(world['exact'] and world['context_exact']
         and world['actual']==world['expected']==prior['world'],'Exact saved public world')
    w=world['actual'];shape=[131,154,117]
    need(w['max_steps']==24 and w['post_exposure']['source_shape']==shape
         and w['post_exposure']['version']=='public-union-post-exposure-axis0-v1'
         and w['post_exposure']['seed_cells']==89 and w['source_domains']['source_domain_extended'] is False
         and w['source_domains']['source_support_unknown_voxels']==176294,'Same partial-domain exposure, no fallback')
    for name in ('public-task-derivation.json','admitted-public-bindings.json','derived-occupancy-assumption.json'):
        read(OUT/'source'/name)
    need(result['world_equal'] and result['context_equal'] and result['prefix_exact']
         and result['unselected_negative_prefix'] and result['complete_episode'],'Exact unselected diagnostic prefix')
    selection=prior['selection']['accounting'];need(selection['completed_layers']==24
         and selection['model_transition_calls']==725 and prior['selection']['actions']==['STOP'],'Prior complete lazy search preserved')
    history=[];prefixrows=[];removed=set();reward=target=outside=depth=0.0
    for step in range(1,12):
        row=read(OUT/f'prefix-{step:02d}.json');h=row['history'];a=row['actual'];exp=row['expected']
        layer=selection['layers'][step-1];need(layer['completed'] and layer['depth']==step,'Saved completed layer')
        layerdiag=layer['retained_prefix_diagnostics']
        candidates=layerdiag['retained_nonterminal_prefixes']+layerdiag['best_terminal_prefixes']
        matches=[x for x in candidates if x['actions']==a['actions']]
        need(len(matches)==1 and {k:matches[0][k] for k in a}==a==exp,'Exact original prefix projection')
        need(len(a['actions'])==step and a['actions'][-1]==h['action_id']
             and a['actions'][:-1]==([r['action_id'] for r in history]),'Fixed eleven-action sequence')
        need(h['source_hash']==w['source_hash'] and h['source_shape']==shape and h['outcome_scope']=='permitted_nominal_model'
             and h['source_state_hash']==row['source_state_before'],'History source/context identity')
        if history:need(previous_state==row['source_state_before'],'Committed source-state chain')
        previous_state=row['source_state_after'];r=cells(h['removed_indices_native'])
        need(not (r&removed),'New committed cells are disjoint');removed|=r
        micro=set()
        for m in h['microsteps']:
            mr=cells(m['removed_indices_native']);need(not micro&mr,'Microstep removed sets disjoint');micro|=mr
        need(micro==r,'Complete microstep removal evidence')
        reward+=h['reward'];target+=h['target_removed_mm3'];outside+=h['normal_removed_mm3'];depth=max(depth,h['insertion_distance_mm'])
        for field,value in [('estimated_incremental_return',reward),('target_removed_mm3',target),
                            ('outside_supplied_target_removed_mm3',outside),('max_insertion_distance_mm',depth)]:
            need(close(a[field],value),'Cumulative prefix arithmetic: '+field)
        history.append(h);prefixrows.append({'step':step,**a,'source_state_after':previous_state,
                                             'removed_cell_count':len(r),'history_hash':semantic(h)})
    need(len(removed)==44 and target==0 and close(reward,result['public_prefix_return'])
         and close(outside,result['outside_supplied_target_removed_mm3']) and result['target_removed_mm3']==0
         and previous_state==result['final_source_state_hash'] and prefixrows[-1]['legal_nonstop_actions']==0,'Final negative dead frontier')
    need(read(OUT/'diagnostic-prefix-history.json')==history,'Full diagnostic history')
    closed=read(OUT/'closed-negative-diagnostic-history.json')
    need(closed[:-1]==history and closed[-1]=={'action_id':'STOP','complete_tool_path_length_mm':0.0,
         'insertion_distance_mm':0.0,'normal_removed_mm3':0.0,'reward':0.0,'target_removed_mm3':0.0},'Only zero-cost diagnostic STOP closes history')
    inventory=read(OUT/'dead-frontier-inventory.json')
    need(inventory['complete'] and inventory['ledger_complete'] and not inventory['terminal']
         and inventory['steps_taken']==11 and inventory['max_steps']==24 and inventory['remaining_steps']==13
         and inventory['candidate_cap']==120 and inventory['declared_slots']==len(inventory['ledger'])==130
         and inventory['emitted_count']==inventory['evaluated_slots']==inventory['rejected_count']==len(inventory['emitted'])==117
         and inventory['accepted_count']==inventory['omitted_count']==inventory['unavailable_count']==0
         and inventory['duplicate_count']==13 and inventory['crop_clipping'] is False,'Complete uncapped dead-frontier inventory')
    need(inventory['source_hash']==w['source_hash'] and inventory['decision_model_hash']==w['decision_model_hash']
         and inventory['cavity_state_hash']==previous_state,'Inventory world/cavity joins')
    affinehash=array_hash(history[0]['native_affine'],'<f8')
    need(affinehash==w['post_exposure']['native_affine_hash'],'Native affine metadata hash')
    rejected=[];union=set();censored=0
    for i,item in enumerate(inventory['emitted']):
        row=read(OUT/f'rejection-{i:03d}.json');d=row['diagnostic']
        for key in ('action_id','column_index','family','tool_id','entry_mm','tip_mm'):
            need(row[key]==item[key],'Rejection-to-emitted-ray join: '+key)
        need(row['feasible'] is False and row['reason']==item['reason']==d['reason']=='SHAFT_BLOCKED_BY_REMAINING_NATIVE_TISSUE',
             'Same actual first rejection')
        need(d['fingerprint']==semantic({k:v for k,v in d.items() if k!='fingerprint'}),'Sidecar content seal')
        need(d['version']=='native-first-shaft-obstruction-v1' and d['source_hash']==w['source_hash']
             and d['decision_model_hash']==history[0]['decision_model_hash'] and d['source_state_hash']==previous_state
             and d['native_affine_hash']==affinehash and d['source_shape']==shape,'Sidecar native world/state')
        need(d['entry_mm']==row['entry_mm'] and d['requested_tip_mm']==row['tip_mm'] and d['tool']['tool_id']==row['tool_id']
             and 0<=d['failure_interval_index']<d['planned_microsteps'] and d['prior_temporary_removals_committed'] is False,
             'Direct native interval identity, no discarded temporary cut committed')
        block=cells(d['blocked_indices_native']);need(not block&removed,'Blockers exclude already committed cavity cells')
        need(d['cell_limit']==32 and d['retained_cell_count']==len(block)<=32
             and d['blocked_cell_count']>=len(block)>0,'Bounded nonempty first-rejection set')
        need(d['complete_first_failure_set']==(d['blocked_cell_count']==len(block))
             and d['truncated']==(d['blocked_cell_count']>len(block)),'Truthful first-set censoring')
        if d['complete_first_failure_set']:need(d['blocked_indices_hash']==array_hash(d['blocked_indices_native']),'Complete blocked-array digest')
        censored+=int(d['truncated']);union|=block
        rejected.append({'action_id':row['action_id'],'column_index':row['column_index'],'family':row['family'],
             'tool_id':row['tool_id'],'blocked_cells':sorted(block),'first_set_complete':d['complete_first_failure_set'],
             'failure_interval_index':d['failure_interval_index'],'planned_microsteps':d['planned_microsteps'],
             'prior_temporary_removed_count':d['prior_temporary_removed_count'],'sidecar_fingerprint':d['fingerprint']})
    capacity=read(OUT/'actual-blocker-capacity.json');need(len(union)==19==capacity['retained_unique_first_blocker_cells']
         ==capacity['classified_unique_cells']==result['classified_cells'],'All unique retained cells classified')
    need(not capacity['classification_truncated'] and not capacity['sidecar_evidence_truncated'] and censored==0
         and not capacity['unclassified_retained_cells'] and not capacity['complete_corridor_claim']
         and not capacity['global_unreachability_claim'] and capacity['axis_set_state_independent']
         and capacity['axis_or_tool_changes_allowed'] is False,'Complete bounded sets; limited interpretation')
    offsets=[[0,0],[-2,0],[2,0],[0,-2],[0,2],[-2,-2],[-2,2],[2,-2],[2,2],[-3,0],[3,0],[0,-3],[0,3]]
    need(capacity['offsets_source_voxels']==offsets and len(capacity['pairs'])==26,'Exact fixed thirteen axes and two tools')
    applicability=capacity['numerical_applicability']
    need(applicability['applicable'] and applicability['coordinate_span_plus_max_tool_dimension_bound_mm']<=10000
         and applicability['maximum_allowed_mm']==10000 and applicability['minimum_active_tip_length_mm']==1
         and applicability['enlargement_mm']==capacity['reporting_roundoff_cushion_mm']==1e-6
         and capacity['whole_cell_inward_tolerance_mm']==1e-10,'Reviewed numerical envelope scope applies')
    tools={t['tool_id']:t for t in capacity['tools']};need(set(tools)=={'generic_suction','generic_aspirator'},'Exact generic tools')
    native=set();enlarged=set();pairkeys=set();kernelcalls=0;cell_pairs={cell:[] for cell in union}
    for pair in capacity['pairs']:
        key=(pair['column_index'],pair['tool_id']);need(key not in pairkeys,'Unique tool/axis pair');pairkeys.add(key)
        need(0<=key[0]<13 and key[1] in tools and pair['offset_source_voxels']==offsets[key[0]]
             and pair['positive_source_endpoints_checked']==129 and pair['native_zero_length_endpoints_refused']==0,'Complete positive endpoint enumeration')
        kernelcalls+=2*pair['positive_source_endpoints_checked']
        need({tuple(r['cell']) for r in pair['cell_capacity']}==union and len(pair['cell_capacity'])==len(union),'Each pair covers all retained cells')
        for r in pair['cell_capacity']:
            c=tuple(r['cell']);n=r['inside_some_full_active_sweep'];e=r['inside_enlarged_roundoff_envelope']
            need(type(n) is bool and type(e) is bool and (not n or e),'Native containment implies enlarged containment')
            need((r['first_envelope'] is not None)==n,'Native positive witness presence')
            if n:native.add(c)
            if e:enlarged.add(c);cell_pairs[c].append(list(key))
    excluded=union-enlarged;ambiguous=enlarged-native
    need(native==cells(capacity['inside_at_least_one_optimistic_envelope']) and excluded==cells(capacity['outside_every_fixed_axis_tip_envelope'])
         and ambiguous==cells(capacity['roundoff_ambiguous_cells']) and len(native)==6 and len(excluded)==13 and not ambiguous,'Independent pair-table classification aggregation')
    need(capacity['outside_every_fixed_axis_tip_envelope']==result['cells_outside_all_fixed_axis_envelopes']
         and kernelcalls==capacity['native_containment_kernel_calls']==result['additional_containment_kernel_calls']==6708
         and capacity['additional_native_previews']==0,'Kernel count and result classification join')
    for i,row in enumerate(rejected):
        d=read(OUT/f'rejection-{i:03d}.json')['diagnostic'];need(d['tool']==tools[row['tool_id']],'Coverage/actual shaft tool exact equality')
        row['excluded_blockers']=sorted(set(row['blocked_cells'])&excluded)
        row['optimistically_containable_blockers']=sorted(set(row['blocked_cells'])&native)
    cellrows=[{'cell':list(c),'classification':'excluded_by_all_enlarged_envelopes' if c in excluded else 'inside_optimistic_envelope',
               'enlarged_tool_axis_pairs':cell_pairs[c],'rejected_ray_count':sum(c in r['blocked_cells'] for r in rejected)} for c in sorted(union)]
    need(cost['native_budget']==result['native_budget'],'Cost and result budget equality')
    budget=cost['native_budget'];need(budget['native_preview_entries']==1521 and budget['counting_reliable']
         and budget['blocked_preview_attempts']==0 and budget['failure'] is None and budget['history_complete_caller_attestation']
         and budget['elapsed_seconds']<180 and budget['time_overshoot_seconds']==0
         and budget['limits']=={'native_preview_entries':2000,'planning_execution_seconds':180.0},'Complete within-budget history')
    expected={'public_construction':117,'fixed_prefix_replay':1287,'first_rejection_sidecars':117,'actual_cell_capacity':0,'diagnostic_STOP_closure':0}
    need(set(cost['phases'])==set(expected),'Exact cost phases')
    for name,n in expected.items():need(cost['phases'][name]['native_preview_entries']==n,'Phase preview accounting')
    need(sum(expected.values())==1521 and sum(x['complete_wall_seconds'] for x in cost['phases'].values())<=result['wall_seconds'],'Nonoverlapping phase totals')
    for name,n in [('fixed_prefix_replay',11),('diagnostic_STOP_closure',1)]:
        need(cost['phases'][name]['native_transition_calls']==cost['phases'][name]['native_transition_returned']==n,'Exact transition accounting')
    need(result['committed_actions']==11 and result['diagnostic_STOP_calls']==1
         and result['diagnostic_previews']==result['first_shaft_rejections']==117 and result['final_legal_motions']==0
         and result['state_unchanged_by_diagnostics'],'Top-level action/counter guards')
    profile=budget['native_preview_profile']['phases']['unclassified']
    need(profile['started']==profile['returned']==profile['feasible']+profile['rejected']==1521
         and profile['raised']==0 and profile['rejection_reasons']=={'SHAFT_BLOCKED_BY_REMAINING_NATIVE_TISSUE':profile['rejected']},'Native profile reconciles')
    output_total=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())
    need(output_total==parent['output_bytes'] and output_total<32*1024**2,'Closed output byte recount')
    return {'terminal_pins':pins,'source_head':index['head'],'source_files':94,'metadata_files':50,
            'source_guard_before_root_successor_edits':digest(DEST/'current-source-guard.json'),
            'prefix':prefixrows,'committed_unique_cells':len(removed),'committed_target_mm3':target,'committed_outside_target_mm3':outside,
            'prefix_return':reward,'rejected_rays':len(rejected),'ray_rows':rejected,'cell_rows':cellrows,
            'rays_with_at_least_one_excluded_blocker':sum(bool(r['excluded_blockers']) for r in rejected),
            'rays_with_only_optimistically_containable_blockers':sum(not r['excluded_blockers'] for r in rejected),
            'censored_first_rejection_sets':censored,'excluded_unique_cells':len(excluded),'optimistically_containable_unique_cells':len(native),
            'ambiguous_unique_cells':len(ambiguous),'maximum_first_set_size':max(len(r['blocked_cells']) for r in rejected),
            'rays_with_discarded_temporary_removal':sum(r['prior_temporary_removed_count']>0 for r in rejected),
            'containment_kernel_calls':kernelcalls,'phase_costs':cost['phases'],'native_preview_entries':1521,
            'parent_seconds':parent['elapsed_seconds'],'worker_seconds':result['wall_seconds'],
            'sampled_peak_rss_bytes':parent['sampled_peak_rss_bytes'],'output_bytes':output_total,
            'no_survivors':True,'models_weights_arrays_read_by_audit':False,
            'scope':'Saved-record integrity and aggregation only; native geometry is not recomputed. Fixed first-rejection evidence and optimistic coverage do not certify a legal clearing sequence, full corridor, global unreachability or an accepted plan.'}


if __name__=='__main__':
    start=time.monotonic();signal.alarm(30)
    record={'status':'HOLD'}
    try:
        compact=audit()
        for path,ref in list(READS.items()):bytes_read(path,ref['sha256'])
        record={'status':'PASS','checks':CHECKS,'input_files':len(READS),'input_bytes':sum(x['bytes'] for x in READS.values()),
                'elapsed_seconds':time.monotonic()-start,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                'all_saved_inputs_postguarded':True,'scientific_execution':False,
                'source_postguard_note':'All 94 executed source files independently hashed before releasing the root hold. Subsequent source promotion does not alter historical bindings.'}
        for name,data in [('compact-result.json',compact),('input-hashes.json',READS)]:
            with (DEST/name).open('x') as f:json.dump(data,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
    except Exception as error:
        record.update(error=type(error).__name__+': '+str(error),checks=CHECKS,elapsed_seconds=time.monotonic()-start)
    with (DEST/'audit-result-02.json').open('x') as f:json.dump(record,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(record,sort_keys=True));raise SystemExit(0 if record['status']=='PASS' else 1)
