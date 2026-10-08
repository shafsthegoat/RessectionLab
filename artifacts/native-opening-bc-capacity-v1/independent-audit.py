"""Saved bytes, JSON, scalar arithmetic and safely loaded tensor metadata only.

No project imports, model construction, policy forwards, rollout or optimizer execution.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, math, time, traceback

ROOT = Path.cwd()
B = ROOT / 'artifacts/native-opening-bc-capacity-v1'
OLD = ROOT / 'artifacts/native-opening-learning-v1'
OUT = ROOT / 'build/native-opening-bc-capacity-output-review-v1'
started = time.perf_counter()
checks = 0

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()

def read(name): return json.loads((B / name).read_text())
def oldread(name): return json.loads((OLD / name).read_text())
def canonical(value): return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)
def digest(value): return 'sha256:' + hashlib.sha256(canonical(value).encode()).hexdigest()
def require(condition, message):
    global checks
    checks += 1
    if not condition: raise AssertionError(message)
def close(a, b, message, tolerance=1e-10):
    require(math.isclose(float(a), float(b), abs_tol=tolerance, rel_tol=tolerance), f'{message}: {a} != {b}')
def cells(rows):
    require(all(isinstance(row,list) and len(row)==3 and all(type(x) is int for x in row) for row in rows),'bad source-cell schema')
    result={tuple(row) for row in rows};require(len(result)==len(rows),'duplicate source cells');return result

def audit():
    index = read('output-sha256.json'); index_sha = sha(B / 'output-sha256.json')
    require(len(index) == 51, 'expected execution inventory')
    for name, expected in index.items():
        path = (B / name).resolve()
        require(path.is_relative_to(B.resolve()), 'output escaped root')
        require(sha(path) == expected, 'output changed ' + name)
    declaration = read('declaration-input.json'); raw_sha = sha(B / 'declaration-input.json')
    require(raw_sha == '3812bb430b0537c83604c4e18d9236d51a7dc203cafc753e0dc6f52d688b9c84', 'declaration')
    require(sha(ROOT / 'manifests/experiments/native-opening-bc-capacity-v1.json') == raw_sha, 'prospective declaration changed')
    require(len(declaration['source_sha256']) == 32, 'source count')
    for name, expected in declaration['source_sha256'].items():
        require(sha(ROOT / name) == expected, 'current source changed ' + name)
        require(sha(B / 'source-snapshot' / name) == expected, 'source snapshot changed ' + name)
    require(read('inherited-input-sha256.json') == declaration['input_sha256'], 'inherited inventory')
    require(len(declaration['input_sha256']) == 15, 'inherited input count')
    for name, expected in declaration['input_sha256'].items():
        require(sha(OLD / name) == expected, 'inherited input changed ' + name)
    old_index = oldread('output-sha256.json')
    require(len(old_index) == 278, 'original execution count')
    for name, expected in old_index.items():
        require(sha(OLD / name) == expected, 'original experiment changed ' + name)
    require(oldread('declaration-input.json')['source_sha256'] == {k:v for k,v in declaration['source_sha256'].items() if k != 'scripts/run_native_opening_bc_capacity.py'}, 'old source closure mismatch')
    lineage = declaration['lineage']
    require(lineage['real_patient_count'] == 0 and lineage['human_roles_opened'] == [] and lineage['transitions'] == 'simulator_generated' and lineage['clinical_use'] is False, 'lineage')
    result=read('result.json'); supervisor=read('supervisor.json'); settings=declaration['settings']
    require(result['status']=='complete' and supervisor['status']=='complete' and supervisor['returncode']==0, 'terminal status')
    require(supervisor['declaration_sha256']==raw_sha and supervisor['automatic_retry'] is False and supervisor['timed_out'] is False and supervisor['termination_reason'] is None, 'supervision')
    require(supervisor['seconds'] <= settings['max_wall_seconds']==60 and result['elapsed_seconds']<supervisor['seconds'], 'wall budget')
    require(supervisor['sampled_peak_rss_bytes'] <= settings['max_rss_bytes']==2147483648 and supervisor['rss_samples'] > 0, 'sampled memory')
    require(settings['cpu_threads']==1 and result['real_patient_count']==0, 'scope')
    require(result['updates']==256 and result['loss_forward_calls']==1280 and result['diagnostic_forward_calls']==30 and result['update16_exact_reproduction'] is True, 'fixed work')
    require(set(result['methods'])=={'initial','BC256'}, 'fixed method denominator')
    old_result=oldread('result.json'); prep=oldread('preparation.json'); teacher=oldread('teacher.json')
    source_hash=prep['source_hash']; model_hash=prep['decision_model_hash']; reference_hash=prep['reference_hash']
    teacher_states={r['observation_hash']:r for r in teacher['state_rows']}
    proofs=oldread('teacher-demonstration-checks.json')
    require(len(teacher_states)==len(proofs)==5 and teacher['complete'] and teacher['reference_fields_used'] is False, 'teacher')
    require(result['reconstruction']['new_teacher_search_calls']==0 and result['reconstruction']['prefix_transitions']==4, 'no new search')
    reconstructed=result['reconstruction']['states']
    require(len(reconstructed)==5, 'reconstruction denominator')
    for i,(proof,state) in enumerate(zip(proofs,reconstructed)):
        expected=teacher_states[proof['observation_hash']]
        require(state=={'observation_hash':proof['observation_hash'],'prefix':proof['prefix'],'action_ids':expected['action_ids'],'label':proof['selected_action_id']}, 'immutable reconstruction state')
        episode=oldread(f'teacher-state-{i:02}.json')
        require(proof['accepted'] is True and episode['independent_evaluation']['accepted'] is True and episode['status']=='complete', 'inherited teacher audit')
        label=episode['decisions'][len(proof['prefix'])]
        require(label['observation_hash']==proof['observation_hash'] and label['action_id']==proof['selected_action_id'], 'teacher demonstration label')
        require([d['action_id'] for d in episode['decisions']]==proof['complete_demonstration'], 'teacher complete trajectory')
    expected_costs={'depth2_search_online_seconds':old_result['methods']['depth2-search']['online_seconds'],
        'original_shared_preparation':old_result['preparation'], 'depth2_search_audit_seconds':oldread('depth2-search-cost.json')['independent_audit_seconds'],
        'teacher_search_model_transitions':teacher['model_transition_calls'],'teacher_validation':old_result['offline_teacher_validation'],
        'teacher_search_preview_profile':oldread('depth2-search-cost.json')['online'], 'prior_BC16_training_seconds':old_result['training']['BC']['fit_seconds'],
        'scope':'previously paid teacher/search and audit costs; not repeated or counted as new runtime'}
    require(result['inherited_costs']==expected_costs, 'inherited costs accounting')
    require(result['context']['declaration_sha256']==raw_sha and result['context']['decision_model_hash']==model_hash and result['context']['source_ids']==[source_hash] and result['context']['real_patient_count']==0 and result['context']['max_steps']==2, 'context')
    support={(4,4,z) for z in range(1,6)}|{(5,5,1)}; target={(4,4,4),(4,4,5)}
    episode_names = ['initial', 'BC256']
    episode_results={};all_transitions=all_forwards=all_native_previews=0;online=audit_seconds=actor_seconds=native_seconds=0.
    for name in episode_names:
     ep=read(name+'.json');terminal=read(name+'-terminal.json');cost=read(name+'-cost.json');metrics=ep['metrics'];ev=ep['independent_evaluation'];out=ev['outcomes'];hist=metrics['history'];dec=ep['decisions']
     require(ep['status']=='complete' and terminal['status']=='awaiting_independent_check','episode terminal '+name)
     require(terminal['metrics']==metrics and terminal['decisions']==dec,'durable history differs '+name)
     require(ev['accepted'] is True and ev['complete_episode'] is True and ev['geometry']['feasible'] is True and ev['geometry']['failures']==[],'saved independent geometry rejection '+name)
     require(ev['geometry']['complete_tool_checked'] is True and ev['geometry']['frontier_checked'] is True,'incomplete saved audit '+name)
     require(metrics['source_hash']==source_hash and metrics['decision_model_hash']==model_hash and metrics['reference_hash']==reference_hash,'source/task/reference binding '+name)
     require(ev['decision_model_hash']==model_hash and ev['source_hash']==source_hash and ev['reference_hash']==reference_hash,'evaluator binding '+name)
     require(metrics['observation_track']=='synthetic_scan' and metrics['planning_estimator_only'] is False and metrics['terminated'] is True,'episode lineage '+name)
     require(len(hist)==len(dec)==metrics['steps']==ep['committed_transitions']==ep['attempted_actions'] and 1<=len(hist)<=2,'episode action counts '+name)
     require(ep['invalid_actions']==0 and dec[-1]['terminated'] is True,'invalid or incomplete episode '+name)
     removed=set();contact=set();normal=target_volume=reward=path_length=0.;tool=None;changes=nonstop=0
     for k,(h,d) in enumerate(zip(hist,dec)):
      require(d['status']=='returned' and d['committed_info']==h and d['action_id']==h['action_id'],'decision history '+name)
      require(d['step']==k and d['behavior_parameter_hash']==ep['behavior_parameter_hash'],'behavior binding '+name)
      observed=teacher_states[d['observation_hash']]
      require(d['action_ids']==observed['action_ids'] and all(d['action_mask']) and len(d['action_mask'])==len(d['action_ids']),'candidate evidence mismatch '+name)
      require(d['action_ids'][d['selected_index']]==d['action_id'],'selected action mismatch '+name)
      require(d['terminated']==(d['action_id']=='STOP' or k==1),'terminal flag '+name)
      if 'logits' in d:
       values=d['logits'];top=max(values);weights=[math.exp(x-top) for x in values];den=sum(weights)
       for expected,actual in zip([x/den for x in weights],d['probabilities']):require(abs(expected-actual)<2e-7,'saved softmax mismatch '+name)
       if ep['mode']=='argmax':require(d['selected_index']==values.index(top),'argmax mismatch '+name)
      if h['action_id']=='STOP':
       require(k==len(hist)-1,'actions after STOP '+name);close(h['reward'],0.,'STOP reward');close(h['complete_tool_path_length_mm'],0.,'STOP path')
      else:
       require(h['source_hash']==source_hash and h['source_shape']==[9,9,7],'native source binding '+name)
       require(h['native_affine']==[[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,1.]],'source physical scale '+name)
       removed_now=cells(h['removed_indices_native']);contacts_now=cells(h['contact_indices_native'])
       require(removed_now<=support and contacts_now<=support and not removed_now&removed,'out-of-support or duplicate reward '+name)
       micro_removed=set();micro_contact=set()
       for micro in h['microsteps']:
        mr=cells(micro['removed_indices_native']);mc=cells(micro['contact_indices_native'])
        require(not mr&micro_removed,'duplicate micro removal '+name);micro_removed|=mr;micro_contact|=mc
       require(micro_removed==removed_now and micro_contact==contacts_now,'microstep cell union '+name)
       tv=len(removed_now&target);nv=len(removed_now-target);length=2*math.dist(h['entry_mm'],h['tip_mm']);changed=tool is not None and tool!=h['tool_id']
       value=tv-.2*nv-.03-.001*length-.03*changed
       close(h['target_removed_mm3'],tv,'per-action target');close(h['normal_removed_mm3'],nv,'per-action normal');close(h['reward'],value,'per-action reward');close(h['complete_tool_path_length_mm'],length,'per-action path')
       require(h['outcome_scope']=='separate_evaluator_reference' and h['partial_contact_weight']==0.,'reward lineage/contact '+name)
       removed|=removed_now;contact|=contacts_now;target_volume+=tv;normal+=nv;reward+=value;path_length+=length;nonstop+=1;changes+=changed;tool=h['tool_id']
      close(d['reward'],h['reward'],'decision reward')
      actor_seconds+=d['decision_seconds'];native_seconds+=d['transition_seconds']
     sums={'target_removed_mm3':target_volume,'normal_removed_mm3':normal,'simulated_removed_volume_mm3':len(removed),'total_reward':reward,'cumulative_contacted_tissue_upper_bound_mm3':len(contact),'currently_retained_contacted_tissue_upper_bound_mm3':len(contact-removed)}
     for key,value in sums.items():close(metrics[key],value,name+' metrics '+key);close(out[key],value,name+' outcomes '+key)
     for key,value in {'complete_tool_path_length_mm':path_length,'nonstop_actions':nonstop,'tool_changes':changes,'positive_target_source_cells_removed':target_volume,'reference_target_fraction_removed':target_volume/2.,'total_reference_target_mm3':2.}.items():close(out[key],value,name+' '+key)
     for key in ['clinical_deficit_probability','motor_surrogate','language_surrogate']:require(out[key] is None,name+' unsupported clinical claim')
     close(ep['simulated_return'],reward,name+' episode return')
     budget=cost['online'];require(budget['failure'] is None and budget['counting_reliable'] is True and budget['history_complete_caller_attestation'] is True,'bad online budget '+name)
     require(budget['elapsed_seconds']<=10. and budget['native_preview_entries']<=2048 and budget['blocked_preview_attempts']==0,'per episode bounds '+name)
     phases=budget['native_preview_profile']['phases'];require(sum(x['started'] for x in phases.values())==budget['native_preview_entries'],'native preview accounting '+name)
     require(all(x['started']==x['returned'] and x['raised']==0 and x['feasible']+x['rejected']==x['returned'] for x in phases.values()),'preview phase accounting '+name)
     close(cost['independent_audit_seconds'],ep['independent_evaluation_seconds'],'audit cost '+name)
     require(budget['elapsed_seconds']>=ep['online_seconds'],'planning excluded from online cost '+name)
     if name in result['methods']:
      row=result['methods'][name];require(row['status']=='complete' and row['accepted'] is True and row['outcomes']==out,'method summary '+name)
      require(row['actions']==[d['action_id'] for d in dec] and row['parameter_hash']==ep['behavior_parameter_hash'],'method action/checkpoint '+name)
      close(row['online_seconds'],budget['elapsed_seconds'],'inclusive method seconds '+name);close(row['replay_seconds'],ep['online_seconds'],'method replay seconds '+name)
     episode_results[name]={'return':reward,'transitions':len(dec),'target_mm3':target_volume,'normal_mm3':normal,'behavior_parameter_hash':ep['behavior_parameter_hash'],'online_seconds':budget['elapsed_seconds'],'audit_seconds':ep['independent_evaluation_seconds'],'native_previews':budget['native_preview_entries'],'forwards':ep['policy_forward_calls']}
     all_transitions+=len(dec);all_forwards+=ep['policy_forward_calls'];all_native_previews+=budget['native_preview_entries'];online+=budget['elapsed_seconds'];audit_seconds+=ep['independent_evaluation_seconds']
    for name in episode_names:
        ep=read(name+'.json')
        require(digest(ep['metrics']['history'])==ep['independent_evaluation']['committed_history_hash'],'committed history digest')
        require(ep['policy_forward_calls']==2,'two actual inference decisions per episode')
    close(episode_results['initial']['return'],old_result['methods']['initial']['outcomes']['total_reward'],'initial preserved')
    require(result['methods']['initial']['actions']==old_result['methods']['initial']['actions'],'initial action reproduction')
    close(episode_results['BC256']['return'],teacher['nominal_optimum'],'fixed final reaches finite search optimum')
    require(result['methods']['BC256']['actions']==teacher['sequence'],'fixed final matches searched sequence')
    require(result['methods']['BC256']['outcomes']==old_result['methods']['depth2-search']['outcomes'],'same searched outcomes')
    close(old_result['methods']['BC']['outcomes']['total_reward'],0.,'prior BC STOP retained')
    require(old_result['methods']['BC']['actions']==['STOP'],'prior negative action retained')
    close(old_result['methods']['scratch-RL']['outcomes']['total_reward'],-.896,'prior RL negative retained')
    # Static checkpoint metadata allowlist, identical to the previous independent review.
    import torch
    import numpy as np
    torch.set_num_threads(1)
    def load_checkpoint(path):
        require(set(torch.serialization.get_unsafe_globals_in_checkpoint(path))=={'numpy._core.multiarray.scalar','numpy.dtype'},'unexpected checkpoint metadata globals')
        with torch.serialization.safe_globals([np._core.multiarray.scalar,np.dtype,np.dtypes.Float64DType]):
            return torch.load(path,map_location='cpu',weights_only=True)
    def state_hash(state):
        h=hashlib.sha256()
        for name,tensor in sorted(state.items()):
            require(torch.is_tensor(tensor) and torch.isfinite(tensor).all().item(),'nonfinite checkpoint tensor')
            array=tensor.detach().cpu().contiguous().numpy()
            h.update(name.encode());h.update(str(array.dtype).encode());h.update(str(array.shape).encode());h.update(array.tobytes())
        return 'sha256:'+h.hexdigest()
    def tensor_tree_equal(left,right):
        if torch.is_tensor(left):
            return torch.is_tensor(right) and left.dtype==right.dtype and left.shape==right.shape and torch.equal(left,right)
        if isinstance(left,dict):
            return isinstance(right,dict) and list(left)==list(right) and all(tensor_tree_equal(v,right[k]) for k,v in left.items())
        if isinstance(left,(list,tuple)):
            return type(left)==type(right) and len(left)==len(right) and all(tensor_tree_equal(a,b) for a,b in zip(left,right))
        return left==right
    original_initial=load_checkpoint(OLD/'initial.pt'); original16=load_checkpoint(OLD/'BC-latest.pt')
    initial=load_checkpoint(B/'BC-000.pt'); initial_hash=state_hash(initial['policy'])
    require(initial_hash==declaration['initial_parameter_hash']==initial['parameter_hash']==result['initial_parameter_hash'],'initial tensor identity')
    require(tensor_tree_equal(initial['policy'],original_initial['policy']),'original initial tensor equality')
    require(initial['updates']==0 and initial['optimizer'] is None,'fresh initial no optimizer')
    require(canonical(initial['architecture']['config'])==canonical(declaration['policy']),'architecture config')
    require(digest(initial['architecture'])==result['architecture_hash'],'architecture fingerprint')
    require(sum(t.numel() for t in initial['policy'].values())==result['parameter_count']==30827,'parameter count')
    updates=read('updates.json'); old_updates=oldread('BC-updates.json')
    require(len(updates)==256 and len(old_updates)==16,'update denominator')
    require([row['gradient'] for row in updates[:16]]==[row['gradient'] for row in old_updates],'all first16 exact gradient/hash records')
    require([row['loss'] for row in updates[:16]]==[row['loss'] for row in old_updates],'all first16 exact losses')
    current=initial_hash;chain={0:current}
    for i,row in enumerate(updates,1):
        grad=row['gradient'];loss=row['loss']
        require(row['update']==i and grad['optimizer_steps']==1 and grad['parameters_changed'] is True,'gradient update count')
        require(grad['initial_parameter_hash']==current and grad['updated_parameter_hash']!=current,'gradient hash chain')
        require(loss['kind']=='search_action_behavior_cloning' and loss['supervised_actions']==5 and loss['loss_forward_calls']==5,'exact sample count')
        require(math.isfinite(loss['loss']) and row['seconds']>0,'loss/update timing')
        require(grad['gradient_norm_after_clip']<=5.00001 and grad['encoder_gradient_norm_before_clip']>0 and grad['actor_gradient_norm_before_clip']>0 and grad['stop_gradient_norm_before_clip']>0,'nonzero actual gradients')
        close(grad['critic_gradient_norm_before_clip'],0.,'critic has no BC supervision')
        current=grad['updated_parameter_hash'];chain[i]=current
    require(sum(r['loss']['loss_forward_calls'] for r in updates)==1280,'loss forward sum')
    close(sum(r['seconds'] for r in updates),result['loss_update_seconds'],'loss/update seconds sum')
    require(result['fit_with_readout_export_seconds']>=result['loss_update_seconds'],'inclusive fit timing')
    checkpoints={}; readouts={};readout_errors={'probability':0.,'cross_entropy':0.,'margin':0.}
    require(set(result['readouts'])=={'0','16','32','64','128','256'},'six fixed readouts')
    parameter_names=list(initial['policy'])
    for update in (0,16,32,64,128,256):
        ck=initial if update==0 else load_checkpoint(B/f'BC-{update:03}.pt')
        computed=state_hash(ck['policy']);row=result['readouts'][str(update)]
        require(computed==ck['parameter_hash']==chain[update]==row['parameter_hash'],'checkpoint/readout/gradient identity')
        require(ck['updates']==update and ck['initial_parameter_hash']==initial_hash,'checkpoint lineage')
        require(canonical(ck['architecture'])==canonical(initial['architecture']),'architecture unchanged')
        require(canonical(ck['context'])==canonical(result['context']),'checkpoint context')
        require(ck['context']['transition_lineage']=='simulator_generated' and ck['context']['real_patient_count']==0,'checkpoint source scope')
        changed={group:any(not torch.equal(t,ck['policy'][name]) for name,t in initial['policy'].items() if name.startswith(group+'.')) for group in ('encoder','actor','stop','critic')}
        if update:
            require(changed=={'encoder':True,'actor':True,'stop':True,'critic':False},'BC module changes')
            opt=ck['optimizer']; groups=opt['param_groups']
            require(len(groups)==1 and groups[0]['params']==list(range(16)),'complete optimizer membership')
            require(groups[0]['lr']==.001 and groups[0]['betas']==(.9,.999) and groups[0]['eps']==1e-8 and groups[0]['weight_decay']==0 and groups[0]['amsgrad'] is False,'Adam settings')
            require(set(opt['state'])==set(range(12)),'only encoder/actor/stop receive Adam state')
            for parameter,meta in opt['state'].items():
                require(int(meta['step'].item())==update,'actual Adam step counter')
                for key in ('exp_avg','exp_avg_sq'):
                    require(meta[key].shape==ck['policy'][parameter_names[parameter]].shape and torch.isfinite(meta[key]).all().item(),'Adam moment metadata')
                require((meta['exp_avg_sq']>=0).all().item(),'Adam second moment')
        else:
            require(not any(changed.values()),'initial changed')
        if update==16:
            require(computed==declaration['update16_parameter_hash'],'old16 hash')
            require(tensor_tree_equal(ck['policy'],original16['policy']) and tensor_tree_equal(ck['optimizer'],original16['optimizer']),'exact original16 weights AND Adam moments')
        require(row['forward_calls']==len(row['states'])==5,'readout count')
        ces=[];correct=0
        for state,proof in zip(row['states'],proofs):
            expected=teacher_states[proof['observation_hash']]; logits=state['logits']; ids=state['action_ids']
            require(state['observation_hash']==proof['observation_hash'] and state['teacher_action']==proof['selected_action_id'] and ids==expected['action_ids'],'readout teacher binding')
            require(len(logits)==len(ids)==len(state['probabilities']) and all(math.isfinite(x) for x in logits) and math.isfinite(state['critic_value']),'readout finite values')
            label=ids.index(state['teacher_action']);top=max(logits);selected=logits.index(top)
            exps=[math.exp(value-top) for value in logits];den=sum(exps);probs=[v/den for v in exps]
            ce=(top-logits[label])+math.log(den);margin=logits[label]-max(v for i,v in enumerate(logits) if i!=label)
            stop_margin=logits[0]-max(logits[1:])
            for p,q in zip(probs,state['probabilities']):
                close(p,q,'softmax from saved logits',2e-7);readout_errors['probability']=max(readout_errors['probability'],abs(p-q))
            close(state['teacher_cross_entropy'],ce,'saved label CE',2e-7);readout_errors['cross_entropy']=max(readout_errors['cross_entropy'],abs(state['teacher_cross_entropy']-ce))
            close(state['teacher_probability'],probs[label],'label probability',2e-7)
            close(state['stop_probability'],probs[0],'STOP probability',2e-7)
            close(state['teacher_logit_margin'],margin,'label margin',2e-6)
            close(state['stop_minus_best_nonstop_logit'],stop_margin,'STOP margin',2e-6)
            readout_errors['margin']=max(readout_errors['margin'],abs(state['teacher_logit_margin']-margin),abs(state['stop_minus_best_nonstop_logit']-stop_margin))
            require(state['teacher_rank']==1+sum(v>logits[label] for v in logits),'teacher rank')
            require(state['selected_action']==ids[selected] and state['correct']==(selected==label),'argmax/correctness')
            ces.append(state['teacher_cross_entropy']);correct+=selected==label
        close(sum(ces)/5,row['mean_cross_entropy'],'mean CE')
        require(correct==row['correct_states'],'correct state count')
        if update<256:close(updates[update]['loss']['loss'],row['mean_cross_entropy'],'next update uses all five identical states',2e-7)
        checkpoints[str(update)]={'tensor_sha256':computed,'file_sha256':sha(B/f'BC-{update:03}.pt'),'actual_optimizer_steps':update,'changed_modules':changed}
        readouts[str(update)]={'correct_states':correct,'mean_cross_entropy':row['mean_cross_entropy'],'selected_actions':[s['selected_action'] for s in row['states']]}
    require(readouts['256']['correct_states']==5 and readouts['16']['correct_states']==2,'capacity outcome and original failure retained')
    require(all(s['selected_action']=='STOP' for s in result['readouts']['16']['states']),'old16 STOP at every teacher state')
    require(current==result['methods']['BC256']['parameter_hash'],'final inference exact fixed256 weights')
    for name,update in [('initial',0),('BC256',256)]:
        by_observation={s['observation_hash']:s for s in result['readouts'][str(update)]['states']}
        for decision in read(name+'.json')['decisions']:
            state=by_observation[decision['observation_hash']]
            require(decision['logits']==state['logits'] and decision['action_ids']==state['action_ids'] and decision['action_id']==state['selected_action'],'saved inference equals exact checkpoint readout')
    preview_counts={}
    for name in ('preparation','reconstruction'):
        phases=result[name]['preview_profile']['phases']
        require(all(x['started']==x['returned'] and x['raised']==0 and x['feasible']+x['rejected']==x['returned'] for x in phases.values()),'preparation/reconstruction native count')
        preview_counts[name]=sum(x['started'] for x in phases.values())
    for name,expected in index.items():require(sha(B/name)==expected,'output changed during audit '+name)
    require(sha(B/'output-sha256.json')==index_sha,'index changed')
    for name,expected in old_index.items():require(sha(OLD/name)==expected,'original changed during audit '+name)
    extras=sorted(str(p.relative_to(B)) for p in B.rglob('*') if p.is_file() and str(p.relative_to(B)) not in index and p.name!='output-sha256.json')
    return {'schema':'native-opening-bc-capacity-saved-output-audit-v1','created_at':datetime.now(timezone.utc).isoformat(),'status':'passed','checks':checks,
        'declaration_sha256':raw_sha,'output_inventory_sha256':index_sha,'result_sha256':sha(B/'result.json'),'updates_sha256':sha(B/'updates.json'),
        'source_files_verified':32,'inherited_inputs_verified':15,'output_files_verified':51,'prior_output_files_verified':278,'original_and_new_indexed_outputs_unchanged_before_after':True,
        'checkpoints':checkpoints,'readouts':readouts,'maximum_scalar_arithmetic_errors':readout_errors,'methods':episode_results,
        'counts':{'BC_updates':256,'loss_forwards':1280,'readout_forwards':30,'inference_forwards':all_forwards,'actual_audited_episode_transitions':all_transitions,'teacher_prefix_transitions':4,'new_teacher_search_calls':0,'native_previews':{**preview_counts,'audited_inference':all_native_previews,'total':sum(preview_counts.values())+all_native_previews}},
        'costs':{'loss_update_seconds':result['loss_update_seconds'],'fit_with_readout_export_seconds':result['fit_with_readout_export_seconds'],'preparation_seconds':result['preparation']['seconds'],'reconstruction_seconds':result['reconstruction']['seconds'],'readout_seconds_sum':sum(r['seconds'] for r in result['readouts'].values()),'guarded_inference_seconds_sum':online,'independent_inference_audit_seconds_sum':audit_seconds,'actor_decision_seconds_sum':actor_seconds,'native_transition_seconds_sum':native_seconds,'inherited_costs_verified_exact':True,'inherited_costs':expected_costs,'overlap_note':'Loss/update and diagnostic/export costs are contained within fit/runtime; replay and action timings are contained within guarded inference. Do not sum overlapping scopes.'},
        'bounds':{'worker_seconds':result['elapsed_seconds'],'supervisor_seconds':supervisor['seconds'],'wall_threshold_seconds':60,'sampled_peak_RSS_bytes':supervisor['sampled_peak_rss_bytes'],'RSS_threshold_bytes':2147483648,'RSS_samples':supervisor['rss_samples'],'automatic_retry':False,'termination_reason':None},
        'prior_negative_results_preserved':{'BC16_return':0.,'BC16_all_five_readout_actions':'STOP','scratch_RL16_return':-.896,'initial_return':-.464},
        'interpretation':'Same model, labels, weighting and optimizer fit all five training labels by fixed update256 and its two-action inference equals the complete finite search return1.1. This establishes capacity on this single generated training task only. The earlier16-update BC STOP outcome and negative RL outcome remain unchanged; no new RL improvement, generalization, physical realism or clinical claim follows.',
        'audit_scope':'Read-only source/JSON/checkpoint audit. Safe weights_only tensor load with exact NumPy scalar/dtype metadata allowlist, solely for hashes, equality and optimizer metadata. Recomputed saved softmax/CE/rank/margins and native-cell reward arithmetic. Inspected existing independent geometry audits and bindings; did not recompute geometric clearances or run any policy, task, simulator, gradient, search, patient or held-out evaluation.',
        'unindexed_reporting_files':extras,'audit_elapsed_seconds':time.perf_counter()-started,'findings':[]}

if __name__ == '__main__':
    try:
        report=audit()
        path=OUT/'verification.json'
        with path.open('x') as f: json.dump(report,f,indent=2,allow_nan=False); f.write('\n')
        print(json.dumps({'status':report['status'],'checks':report['checks'],'review_sha256':sha(path),'readouts':report['readouts'],'methods':report['methods'],'bounds':report['bounds'],'checkpoints':report['checkpoints']},indent=2))
    except BaseException as error:
        with (OUT/'failure-initial.json').open('x') as f:
            json.dump({'type':type(error).__name__,'message':str(error),'checks':checks,'traceback':traceback.format_exc()},f,indent=2)
        raise
