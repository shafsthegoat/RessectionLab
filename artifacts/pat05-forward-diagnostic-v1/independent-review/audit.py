"""Saved PAT05 metadata/bytes and scalar arithmetic only; no patient/model decode."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,importlib.metadata,json,math,sys,time,traceback
ROOT=Path.cwd();B=ROOT/'artifacts/pat05-forward-diagnostic-v1';OUT=ROOT/'build/pat05-forward-output-review-v1'
H='artifacts/pat05-real-geometric-learning-v1/';RL='artifacts/native-opening-rl-capacity-v1/'
started=time.perf_counter();checks=0
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()
def read(name):return json.loads((B/name).read_text())
def rootread(name):return json.loads((ROOT/name).read_text())
def require(condition,message):
    global checks
    checks+=1
    if not condition:raise AssertionError(message)
def close(a,b,message,tolerance=1e-10):require(math.isclose(float(a),float(b),abs_tol=tolerance,rel_tol=tolerance),f'{message}: {a} != {b}')
def determinant(a):return a[0][0]*(a[1][1]*a[2][2]-a[1][2]*a[2][1])-a[0][1]*(a[1][0]*a[2][2]-a[1][2]*a[2][0])+a[0][2]*(a[1][0]*a[2][1]-a[1][1]*a[2][0])
def solve3(a,b):
    d=determinant(a);require(abs(d)>1e-12,'invertible saved affine')
    return [determinant([[b[row] if column==axis else a[row][column] for column in range(3)] for row in range(3)])/d for axis in range(3)]
def audit():
    expected_result='9d27ac141f98f6345605d4e81c24b0a1ab3bba528dc4d6bcef39aa6975fa3c3d'
    expected_methods={'initial':'8a717bf240c02f4f23d8a64dbb1491138ba59a458b8a0d3f9676a77a341d8b47','RL256':'ed846cda00b5c1419eb4f540e9f50c667a03cc3b3b6ebb718efdb7769e6ac0eb'}
    require(sha(B/'result.json')==expected_result,'terminal result identity')
    for name,digest in expected_methods.items():require(sha(B/(name+'.json'))==digest,'method output identity')
    index=read('output-sha256.json');index_sha=sha(B/'output-sha256.json')
    require(len(index)==45,'direct index count')
    for name,expected in index.items():
        path=(B/name).resolve();require(path.is_relative_to(B.resolve()),'index path escapes output');require(sha(path)==expected,'indexed byte identity '+name)
    declaration=read('declaration-input.json');decl_sha=sha(B/'declaration-input.json')
    require(decl_sha=='54100dfb6cc8f54a3df4ac63120b19104c7ab75b82fb64f062978a2800c8d9dd','prospective declaration identity')
    require(sha(ROOT/'manifests/experiments/pat05-forward-diagnostic-v1.json')==decl_sha,'current declaration')
    require(len(declaration['source_sha256'])==28 and len(declaration['metadata_sha256'])==10,'input/source counts')
    for name,expected in declaration['source_sha256'].items():require(sha(ROOT/name)==expected and sha(B/'source-snapshot'/name)==expected,'source snapshot/current binding '+name)
    omitted=[]
    for name,expected in declaration['metadata_sha256'].items():
        snapshot='input-metadata/'+name
        require(sha(ROOT/name)==expected and sha(B/snapshot)==expected,'metadata original/snapshot binding '+name)
        if snapshot not in index:omitted.append(snapshot)
    require(len(omitted)==2 and all(Path(p).name=='output-sha256.json' for p in omitted),'only nested index snapshots excluded by basename')
    actual_files={str(p.relative_to(B)) for p in B.rglob('*') if p.is_file()}
    extras=sorted(actual_files-set(index)-set(omitted)-{'output-sha256.json'})
    runtime={'python':sys.version,'platform':sys.platform,**{n:importlib.metadata.version(n) for n in ('numpy','scipy','torch','nibabel')}}
    require(runtime==declaration['runtime'],'runtime matches pinned record')
    old=rootread(H+'declaration-input.json');receipt=rootread(H+'receipt.json');first=rootread(H+'initial_policy.json')['decisions'][0]
    readiness=rootread('artifacts/btc-train-transfer-readiness-v1/contract.json');entry=next(r for r in readiness['train_records'] if r['subject']=='sub-PAT05');binding=entry['bindings'];member=old['member']
    require(declaration['subject']==member['subject']==entry['subject']=='sub-PAT05' and declaration['role']==member['role']==entry['role']=='TRAIN','exact patient/role')
    for key in ('case_bundle','case_bundle_sha256','case_semantic_hash'):require(member[key]==binding[key],'historical readiness identity '+key)
    case_path=(ROOT/member['case_bundle']).resolve();require(case_path.is_relative_to(ROOT.resolve()),'bundle path scope');require(sha(case_path)==member['case_bundle_sha256'],'original bundle byte hash only')
    result=read('result.json');supervisor=read('supervisor.json');reconstruction=read('reconstruction.json');description=read('input-description.json');accounting=result['accounting']
    require(result['status']==supervisor['status']=='complete' and supervisor['returncode']==0,'terminal status')
    require(supervisor['declaration_sha256']==accounting['declaration_sha256']==decl_sha and supervisor['termination_reason'] is None and supervisor['timed_out'] is False and supervisor['automatic_retry'] is False,'bound successful one-attempt closure')
    require(accounting['checkpoint_attempts']==accounting['forward_attempts']==accounting['completed_forwards']==2 and accounting['attempted_checkpoints']==accounting['completed_checkpoints']==['RL256','initial'],'exact two forward attempts')
    require(accounting['native_previews']==accounting['executed_actions']==accounting['optimizer_updates']==0,'no native actions or learning')
    require(result['clinical_performance_measured'] is False and result['deployment_compatibility_established'] is False,'no performance claim')
    require(result['subject']=='sub-PAT05' and result['role']=='TRAIN','reported role')
    require(supervisor['seconds']<=60 and result['elapsed_seconds']<supervisor['seconds'] and supervisor['sampled_peak_rss_bytes']<=2*1024**3 and result['peak_worker_rss_bytes']<=2*1024**3,'actual resource bounds')
    require(supervisor['rss_samples']>0 and declaration['settings']['cpu_threads']==1,'sampled memory/single thread')
    require(set(result['methods'])=={'initial','RL256'} and all(r['status']=='complete' for r in result['methods'].values()),'method closure')
    require(reconstruction['source_hash']==declaration['expected_native_case_source_hash']==receipt['initial_task_metrics']['source_hash'],'historical full source identity')
    require(reconstruction['observation_hash']==declaration['expected_observation_hash']==first['observation_hash']=='sha256:645770594d7988980781df8324fb76ed347f47ff8d7452ca7d032d31efc93f90','full DTO fingerprint consistency')
    metrics=receipt['initial_task_metrics']
    for new,oldkey in [('native_grid_reconciliation','native_grid_reconciliation'),('normalization','intensity_normalization'),('support_provenance','support_provenance'),('crop','crop')]:require(reconstruction[new]==metrics[oldkey],'historical reconstruction metadata '+new)
    for key,value in member['expected_native_grid_binding'].items():require(reconstruction['native_grid_reconciliation'][key]==value,'historical frame binding '+key)
    support=reconstruction['support_provenance'];ack=support['acknowledgment']
    require(support['evidence_type']=='estimated' and support['review_status']=='review_required' and support['clinical_use_permitted'] is False and support['cortical_access_permitted'] is False,'estimate not anatomy admission')
    require(ack==member['research_support_acknowledgment'],'exact historical support acknowledgement')
    for key in ('evidence_hash','source_image_hash','source_frame_hash','mask_hash','model_sha256','run_sha256'):require(ack[key]==binding[key],'support/source lineage '+key)
    require(reconstruction['anatomy_admission'] is False and reconstruction['clinical_admission'] is False,'no anatomical/clinical admission')
    require(reconstruction['annotation']['availability_timestamp'] is None and reconstruction['annotation']['expert_review_status']=='unknown' and reconstruction['annotation']['learned_segmentation'] is False,'annotation limitations')
    require(reconstruction['unknown_evidence']==['motor','language','vessels','deformation','cutting_forces'],'unknowns retained')
    require('hypothetical empty' in reconstruction['cavity_lineage'] and 'simulator-feasibility-filtered' in reconstruction['proposal_lineage'],'input interpretation')
    require(reconstruction['crop']['origin_voxels']==[3,106,119] and reconstruction['crop']['shape']==[64,64,64] and reconstruction['source_shape']==[160,256,256],'exact crop/source shape')
    require(reconstruction['supplied_annotation_cells_full_source']==reconstruction['supplied_annotation_cells_in_crop']==11437,'recorded all target cells in crop')
    original_affine=reconstruction['native_grid_reconciliation']['original_affine_ras_mm'];volume_per_voxel=abs(determinant([r[:3] for r in original_affine[:3]]))
    historical_volume=receipt['episodes'][0]['independent_evaluation']['outcomes']['total_reference_target_mm3']
    close(historical_volume/volume_per_voxel,11437,'full-source target count from historical volume/frame',1e-9)
    normalization=reconstruction['normalization'];require(normalization['lower']==18. and normalization['upper']==282. and normalization['statistics_scope']=='entire_permitted_support' and normalization['crop_used_for_statistics'] is False,'normalization meaning')
    channels=description['channels'];require(len(channels)==6,'channel count')
    for i,row in enumerate(channels):
        require(row['available']==(i<4) and row['covered_voxels']==(64**3 if i<4 else 0) and row['coverage_fraction']==(1. if i<4 else 0.),'coverage/availability')
        if i>=4:require(row['min'] is None and row['max'] is None,'missing functional evidence')
        else:require(row['min']==0. and row['max']==(0. if i==3 else 1.),'recorded normalized range/empty cavity')
    require(description['shift']['generated_horizon']==2 and description['shift']['real_historical_horizon']==3 and description['shift']['generalization_or_performance_established'] is False,'declared shift')
    inv=receipt['initial_inventory'];accepted=[r for r in inv['emitted'] if r['feasible'] is True];ids=['STOP',*[r['action_id'] for r in accepted]]
    require(inv['complete'] is True and inv['accepted_count']==len(accepted)==70 and inv['emitted_count']==78 and ids==first['action_ids'],'original filtered action inventory')
    tools={r['tool_id']:r for r in old['tools']};geometry=[];inside=0
    affine=reconstruction['native_grid_reconciliation']['derived_affine_ras_mm'];a=[r[:3] for r in affine[:3]];origin=reconstruction['crop']['origin_voxels']
    for action in accepted:
        entry=action['entry_mm'];tip=action['tip_mm'];tool=tools[action['tool_id']];vector=[b-a for a,b in zip(entry,tip)];length=math.sqrt(sum(v*v for v in vector));axis=[v/length for v in vector]
        geometry.append([0.,*entry,*tip,*axis,tool['tip_radius_mm'],tool['shaft_radius_mm'],tool['working_length_mm'],tool['tip_length_mm'],tool['max_access_angle_deg'],0.])
        for k in range(5):
            point=[entry[i]+k/4*(tip[i]-entry[i]) for i in range(3)]
            voxel=solve3(a,[point[i]-affine[i][3] for i in range(3)])
            normalized=[2*(voxel[i]-origin[i])/63-1 for i in range(3)]
            inside+=all(abs(v)<=1+64*sys.float_info.epsilon for v in normalized)
    names=['stop','entry_x_mm','entry_y_mm','entry_z_mm','tip_x_mm','tip_y_mm','tip_z_mm','axis_x','axis_y','axis_z','tip_radius_mm','shaft_radius_mm','working_length_mm','tip_length_mm','max_access_angle_deg','tool_change']
    for i,name in enumerate(names):
        close(min(r[i] for r in geometry),description['geometry_ranges'][name][0],'geometry min '+name);close(max(r[i] for r in geometry),description['geometry_ranges'][name][1],'geometry max '+name)
    require(inside==description['ray_samples_inside_crop']==description['ray_samples_total']==350,'independent saved-ray crop containment')
    state=description['state'];require(state['steps_taken']==0 and state['max_steps']==3 and state['current_tool_present']==0,'hypothetical initial state')
    access=member['access']
    for i,axis in enumerate('xyz'):
        close(state['access_center_'+axis+'_mm'],access['center_mm'][i],'access center');close(state['access_normal_'+axis],access['normal_inward'][i],'access normal')
    close(state['access_radius_mm'],access['radius_mm'],'access radius')
    rl_review=rootread(RL+'independent-verification.json');rl_index=rootread(RL+'output-sha256.json');comparison={};ranks={};checkpoint_bytes={};max_errors={'softmax':0.,'entropy':0.}
    for name,update in [('initial',0),('RL256',256)]:
        row=read(name+'.json');spec=declaration['checkpoints'][name];recorded=result['methods'][name]
        require(row['status']=='complete' and row['weights_unchanged'] is True and row['updates']==update,'completed frozen checkpoint')
        for key,value in spec.items():require(row[key]==value,'checkpoint specification '+key)
        prior=rl_review['checkpoints'][str(update)]
        require(row['parameter_hash']==prior['tensor_sha256'] and row['file_sha256']==prior['file_sha256']==rl_index[Path(row['path']).name],'independently verified checkpoint ancestry')
        require(sha(ROOT/row['path'])==row['file_sha256'],'raw checkpoint fixity without load');checkpoint_bytes[row['path']]=row['file_sha256']
        require(row['architecture_hash']==declaration['architecture_hash'] and row['parameter_count']==30827,'fixed architecture')
        require(row['training_context_preserved']['declaration_sha256']=='14df1a1fbb9bb8187d97bc7dcf99cacf923a9e5e9f4182693de0e1d01a5e1285' and row['training_context_preserved']['max_steps']==2 and row['training_context_preserved']['real_patient_count']==0 and row['training_context_preserved']['transition_lineage']=='simulator_generated','generated training origin preserved')
        require(row['observation_hash']==reconstruction['observation_hash'] and row['action_ids']==ids and row['action_mask']==first['action_mask'],'same full DTO/inventory')
        logits=row['logits'];probabilities=row['probabilities'];require(len(logits)==len(probabilities)==len(ids)==71 and all(row['action_mask']),'71 unmasked initial action values')
        require(all(math.isfinite(v) for v in logits+probabilities+[row['entropy'],row['value_uncalibrated']]),'finite outputs')
        top=max(logits);weights=[math.exp(v-top) for v in logits];total=math.fsum(weights);expected_probs=[v/total for v in weights]
        for expected,actual in zip(expected_probs,probabilities):close(expected,actual,'independent softmax',2e-7);max_errors['softmax']=max(max_errors['softmax'],abs(expected-actual))
        close(math.fsum(probabilities),1.,'probability mass',2e-7)
        entropy=-math.fsum(p*math.log(p) for p in expected_probs if p>0)
        close(entropy,row['entropy'],'independent entropy',1e-6);max_errors['entropy']=max(max_errors['entropy'],abs(entropy-row['entropy']))
        order=sorted(range(71),key=lambda i:(-logits[i],i));top_index=order[0]
        require(ids[top_index]==row['highest_ranked_id_not_executed']==recorded['highest_ranked_id_not_executed'],'first-maximum ranking')
        require(recorded['outputs']==name+'.json' and recorded['parameter_hash']==row['parameter_hash'],'method result binding')
        for key in ('forward_seconds','load_and_forward_seconds'):close(row[key],recorded[key],'timing binding')
        require(0<row['forward_seconds']<=row['load_and_forward_seconds']<=result['elapsed_seconds'],'nested forward timing')
        comparison[name]={'highest_ranked_id_not_executed':ids[top_index],'STOP_probability':probabilities[0],'STOP_rank':order.index(0)+1,'STOP_minus_best_nonSTOP_logit':logits[0]-max(logits[1:]),'top_minus_second_logit':logits[order[0]]-logits[order[1]],'top_probability':probabilities[top_index],'entropy_nats':row['entropy'],'value_uncalibrated':row['value_uncalibrated'],'forward_seconds':row['forward_seconds'],'load_and_forward_seconds':row['load_and_forward_seconds'],'parameter_hash':row['parameter_hash']}
        ranks[name]=[{'rank':rank+1,'action_id':ids[i],'probability':probabilities[i],'logit':logits[i]} for rank,i in enumerate(order)]
    require(comparison['RL256']['highest_ranked_id_not_executed']=='STOP' and comparison['RL256']['STOP_rank']==1,'trained immediate STOP preference')
    require(comparison['initial']['highest_ranked_id_not_executed']!='STOP' and comparison['initial']['STOP_rank']>1,'initial nonSTOP preference')
    require(result['historical_preparation']['seconds']==receipt['preparation_seconds']==8.175235959002748 and result['historical_preparation']['new_previews']==0,'historical preparation retained separately')
    for name,expected in index.items():require(sha(B/name)==expected,'indexed bytes changed during review '+name)
    for name,expected in declaration['metadata_sha256'].items():require(sha(B/'input-metadata'/name)==expected,'input snapshot changed during review')
    for name,expected in checkpoint_bytes.items():require(sha(ROOT/name)==expected,'checkpoint changed during review')
    require(sha(B/'output-sha256.json')==index_sha,'index changed during review')
    require(sha(case_path)==member['case_bundle_sha256'],'case bundle changed during review')
    return {'schema':'pat05-forward-saved-output-independent-review-v1','status':'passed_with_nonblocking_inventory_note','created_at':datetime.now(timezone.utc).isoformat(),'checks':checks,'declaration_sha256':decl_sha,'result_sha256':expected_result,'method_output_sha256':expected_methods,'output_inventory_sha256':index_sha,'sources_verified':28,'metadata_files_and_snapshots_verified':10,'directly_indexed_files_verified':45,'additional_manifest_bound_index_snapshots_verified':2,'input_bundle_byte_sha256':member['case_bundle_sha256'],'checkpoint_files_byte_sha256':checkpoint_bytes,'runtime':runtime,'original_and_output_bytes_unchanged_before_after':True,'comparison':comparison,'all_action_rankings':ranks,'maximum_scalar_errors':max_errors,'observation':{'full_historical_fingerprint':reconstruction['observation_hash'],'source_hash':reconstruction['source_hash'],'action_count':71,'nonSTOP_count':70,'historical_proposals_before_filter':78,'crop_origin_voxels':[3,106,119],'crop_shape':[64,64,64],'source_shape':[160,256,256],'source_target_cells':11437,'recorded_target_cells_in_crop':11437,'full_source_target_count_crosscheck':'Historical total reference volume divided by saved source-frame voxel volume. Crop count is the authenticated reconstruction report; no patient arrays were decoded in this review.','ray_samples_inside_crop_recomputed_from_saved_geometry':350,'ray_samples_total':350,'grid_and_normalization_match_historical':True,'support_status':'estimated, unreviewed, synthetic-trained SynthStrip; not anatomy admission','annotation_status':'supplied preoperative compartment; availability/review unknown','missing_evidence':reconstruction['unknown_evidence']},'accounting':accounting,'costs':{'worker_seconds':result['elapsed_seconds'],'supervisor_seconds':supervisor['seconds'],'worker_peak_rss_bytes':result['peak_worker_rss_bytes'],'supervisor_sampled_peak_rss_bytes':supervisor['sampled_peak_rss_bytes'],'reconstruction_seconds':result['reconstruction_seconds'],'forward_seconds_sum':sum(r['forward_seconds'] for r in comparison.values()),'load_forward_validation_seconds_sum':sum(r['load_and_forward_seconds'] for r in comparison.values()),'prior_historical_preparation_seconds':result['historical_preparation']['seconds'],'timer_scope':'Forward time is nested within load/forward/validation time, which is nested in worker and supervisor time. Prior preparation was previously paid, excluded from this diagnostic. Do not add overlapping timers. No end-to-end planning latency established.','wall_threshold_seconds':60,'RSS_threshold_bytes':2147483648,'automatic_retry':False},'inventory_note':{'severity':'nonblocking','description':'The writer excludes every basename output-sha256.json from its outer index, so two copied inherited input-index snapshots are absent from the45-entry list. Both exist and match the hashes pinned in the indexed declaration; this audit verifies them separately.','files':omitted,'recommendation':'Future index writers can exclude only the root output index; preserve this completed run unchanged.'},'interpretation':'The trained checkpoint ranks immediate STOP first (probability0.8136946 versus initial0.00988367) on this single historical annotation-assisted TRAIN observation. This is a model preference over70 simulator-filtered proposals plus STOP. No action, planning endpoint, resection outcome, target reach, risk or benefit was executed/measured. Softmax probabilities are not clinical risk probabilities; value outputs are uncalibrated. Exact-input/checkpoint compatibility succeeded; real transfer and deployment performance remain unestablished.','scope':'Read-only JSON/source/runtime metadata, byte hashes of the bound case bundle and checkpoints, and scalar geometry/softmax/rank/entropy arithmetic. No patient-array or checkpoint-tensor decoding, policy forward, native preview, task/action, optimizer update, search, network or protected-patient records opened. Weight tensor identities are joined to the prior independent checkpoint audit by exact file hash.','unindexed_later_reporting_files':extras,'audit_elapsed_seconds':time.perf_counter()-started}

if __name__=='__main__':
    try:
        report=audit();p=OUT/'verification.json'
        with p.open('x') as f:json.dump(report,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
        print(json.dumps({k:report[k] for k in ['status','checks','comparison','maximum_scalar_errors','costs','inventory_note','audit_elapsed_seconds']},indent=2));print(json.dumps({'verification_sha256':sha(p)}))
    except BaseException as error:
        with (OUT/'failure-initial.json').open('x') as f:json.dump({'checks':checks,'error':str(error),'traceback':traceback.format_exc()},f,indent=2)
        raise
