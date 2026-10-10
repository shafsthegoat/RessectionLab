"""Independent corrected-attempt saved JSON audit; no scientific runtime or arrays."""
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'build/union025-missing-axis-v2';OUT=BASE/'attempt-02';SUP=BASE/'attempt-02.supervision'
pins={}
def raw(p):
    p=Path(p);assert p.is_file() and not p.is_symlink() and p.stat().st_size<8*1024**2
    b=p.read_bytes();pins[str(p.relative_to(ROOT))]=hashlib.sha256(b).hexdigest();return b
def load(p):return json.loads(raw(p))
def sha(p):return hashlib.sha256(raw(p)).hexdigest()
def semantic(v):return 'sha256:'+hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def physical(v):return {k:v[k] for k in ('tool_id','entry_mm','tip_mm')}
def without_ids(rows):return [{k:v for k,v in r.items() if k not in ('action_id','proposal_id')} for r in rows]
def close(a,b):assert math.isfinite(a) and math.isclose(a,b,rel_tol=1e-9,abs_tol=1e-9),(a,b)
def cells(v):
    assert all(len(c)==3 and all(type(x) is int for x in c) for c in v)
    s=set(map(tuple,v));assert len(s)==len(v);return s
def native_hash(v):
    cells(v);b=b''.join(struct.pack('<qqq',*c) for c in v)
    return hashlib.sha256(str((len(v),3)).encode()+b'int64'+b).hexdigest()

def main():
    assert sha(BASE/'root-release.json')=='9f8e7249b01a8bc73d34a2682976889c3e57b2fda72ece0ab7bb3fea284c82a7'
    assert sha(BASE/'source-index.json')=='cddb0f1ef363d7728479c53bd7896eb9a95ff174be0ce5426fd8339030175024'
    release=load(BASE/'root-release.json');index=load(BASE/'source-index.json')
    assert release['expected_head']==index['head']=='7947d23529a38d4cb8e8348fbba142c23467a6c3'
    # Current sources were independently verified before releasing root's source hold.
    # Pin canonical code to execution HEAD here so later authorized integration is harmless.
    for p,d in index['source_files'].items():
        if p.startswith('src/'):
            b=subprocess.check_output(['git','show',index['head']+':'+p],cwd=ROOT,timeout=5)
            assert hashlib.sha256(b).hexdigest()==d,p
        else:assert sha(ROOT/p)==d,p
    for p,d in index['metadata_files'].items():assert sha(ROOT/p)==d,p
    refs=load(BASE/'input-index.json')['files'];old=load(ROOT/refs['union_world']['path'])
    oldinv=load(ROOT/refs['union_inventory']['path']);learning=load(ROOT/refs['learning_release']['path'])
    priorfailure=load(ROOT/refs['failed_result']['path']);assert priorfailure['status']=='failed_or_unresolved'
    assert priorfailure['failure']['type']=='FileExistsError'
    assert sha(ROOT/'build/union025-missing-axis-independent-v1/audit.json')=='d736d7cd05fba851b62085cd5e0471868c848222cdc794fd6cba847d43533c13'
    world=load(OUT/'world.json');context=load(OUT/'context.json');inventory=load(OUT/'initial-inventory.json')
    assert semantic(context)==world['context_hash']
    for k,v in world['unchanged_public_checks'].items():assert v==old['common_public_world'][k],k
    for k in ('derived_occupancy','normalization','supplied_goal_extent'):assert world[k]==old[k]
    for k in ('source_hash','decision_model_hash','initial_observation_hash'):assert world[k]!=old[k]
    config=world['proposal_config'];prior=learning['learning_protocol']['cohort_execution']['proposal_config']
    assert config['offsets_source_voxels']==prior['offsets_source_voxels']+[[-1,4]]
    assert len(config['offsets_source_voxels'])==14 and config['max_candidates']==120
    assert {k:v for k,v in config.items() if k!='offsets_source_voxels'}=={k:v for k,v in prior.items() if k!='offsets_source_voxels'}
    oldrows=[r for r in inventory['ledger'] if r['column_index']<13]
    assert without_ids(oldrows)==without_ids(oldinv['ledger']) and len(oldrows)==130
    assert inventory['omitted_count']==0 and world['explicit_new_decision_model'] is True
    assert context['role']=='TRAIN' and context['subject']=='ReMIND-025' and context['max_steps']==24
    assert context['private_reference_in_task'] is False and context['derived_occupancy_anatomically_validated'] is False
    assert context['max_optimizer_updates']==context['budgets']['max_policy_forwards']==0
    result=load(OUT/'result.json');cost=load(OUT/'costs.json');receipt=load(SUP/'receipt.json');final=load(SUP/'worker-final.json')
    control=load(SUP/'endpoint-control.json');diagnostics=load(OUT/'diagnostics.json');rays=load(OUT/'prospective-rays.json')
    assert receipt['status']=='complete' and receipt['exit_code']==0 and receipt['worker_termination_confirmed']
    assert receipt['stop_reason'] is None and not receipt['final_owned_pids'] and not receipt['cleanup_errors']
    assert result['status']=='complete_missing_axis_diagnostic' and result['source_released']
    assert final['status']=='complete_owned_union025_missing_axis' and final['runtime']['threads']==1
    assert receipt['result_sha256']==final['result_sha256']==final['canonical_result_sha256']==sha(OUT/'result.json')
    assert receipt['worker_final_sha256']==sha(SUP/'worker-final.json')
    assert receipt['endpoint_control_sha256']==final['endpoint_control_sha256']==sha(SUP/'endpoint-control.json')
    assert receipt['release_sha256']==sha(BASE/'root-release.json')
    assert receipt['elapsed_seconds']<90 and final['wall_seconds']<60 and 0<receipt['sampled_peak_rss_bytes']<3*1024**3
    assert receipt['samples']>0 and receipt['output_bytes']==sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())<32*1024**2
    assert sum(p.stat().st_size for p in SUP.rglob('*') if p.is_file())<16*1024**2
    assert result['SELECT_EVAL_opened'] is False and result['post_hoc_TRAIN_diagnosis'] is True
    for k in ('policy_forwards','optimizer_updates','checkpoint_loads','private_reference_reads','search_calls'):assert result[k]==0
    assert all(v==0 for v in cost['prohibited_calls'].values())
    assert set(diagnostics)=={'clearance','target'} and result['clearance_status']==result['target_status']=='executable'
    action_ids=[]
    for name in ('clearance','target'):
        d=diagnostics[name];saved=load(OUT/name/'fixed-ray-preview.json');exam=load(OUT/name/'examination.json')
        inv=load(OUT/name/'inventory.json')
        assert d==exam and all(d[k]==v for k,v in saved.items())
        assert physical(d)==rays[name] and d['committed'] is False and d['temporary_removals_committed'] is False
        assert d['feasible'] and d['reason']=='NATIVE_CONNECTED_STROKE' and d['obstruction_diagnostic'] is None
        assert d['action_status']=='executable' and inv['omitted_count']==d['inventory_omitted_count']==0
        matches=[r for r in inv['emitted'] if physical(r)==rays[name]]
        assert matches==d['exact_inventory_matches'] and len(matches)==1 and matches[0]['feasible']
        assert matches[0]['action_id']==d['selected_action_id']
        assert inv['cavity_state_hash']==d['source_state_hash'] and d['source_hash']==world['source_hash']
        assert d['task_model_hash']==world['decision_model_hash']
        steps=d['prior_completed_microsteps'];removed=[s['removed_indices_native'] for s in steps]
        assert d['temporary_removed_count']==sum(map(len,removed))==len(cells(d['feasible_preview_removed_indices_native']))
        assert d['temporary_removed_hash']==semantic([native_hash(v) for v in removed])
        assert cells(d['feasible_preview_removed_indices_native'])==set().union(*(cells(v) for v in removed))
        action_ids.append(d['selected_action_id'])
    assert diagnostics['clearance']['source_state_hash']==world['initial_state_hash']
    assert diagnostics['target']['source_state_hash']!=world['initial_state_hash']
    assert rays['target']==release['target_ray'] and rays['clearance_voxel']==[121,44,72] and rays['appended_axis']==[-1,4]
    histories=result['histories'];assert len(histories)==len(control['histories'])==3
    w=world['unchanged_public_checks']['reward'];full=world['unchanged_public_checks']['full_target_mm3'];summary=[]
    for i,row in enumerate(histories):
        dest=OUT/f'history-{i:02d}';wrapper=load(dest/'plan.json');replay=load(dest/'replay.json')
        p=wrapper['plan'];hist=p['history'];audit=replay['independent_geometry'];metrics=replay['metrics']
        assert row['ordinal']==i and row['status']=='complete'
        assert p['actions']==row['actions']==action_ids[:i]+['STOP']==[h['action_id'] for h in hist]
        assert wrapper['plan_seal']==row['plan_seal']==semantic(p)
        assert hist==metrics['history'] and metrics['terminated'] and metrics['steps']==i+1
        assert audit['accepted'] and audit['complete_episode'] and audit['committed_history_hash']==semantic(hist)
        geometry=audit['geometry'];assert geometry['feasible'] and geometry['complete_tool_checked'] and geometry['frontier_checked']
        assert not geometry['failures'] and geometry['unsupported_source_tissue_volume_mm3']==0
        assert geometry['action_count']==i and audit['outcomes']==row['outcomes']
        for k in ('source_hash','decision_model_hash'):assert p[k]==metrics[k]==audit[k]==world[k]
        assert p['context_hash']==world['context_hash'] and p['initial_observation_hash']==world['initial_observation_hash']
        assert p['max_steps']==24 and p['learning_updates']==0 and p['parameter_hash'] is None
        reward=0.;distance=0.;previous=None;toolchanges=0;removed=set()
        for j,h in enumerate(hist):
            assert load(dest/f'attempt-{j:02d}.json')['action_id']==h['action_id']
            assert load(dest/f'returned-{j:02d}.json')==h
            assert load(dest/f'replay-attempt-{j:02d}.json')['action_id']==h['action_id']
            if h['action_id']=='STOP':assert h['reward']==h['target_removed_mm3']==h['normal_removed_mm3']==0;continue
            name=('clearance','target')[j];d=diagnostics[name]
            assert physical(h)==rays[name] and h['source_state_hash']==d['source_state_hash']
            one=cells(h['removed_indices_native']);assert not one&removed
            assert one==cells(d['feasible_preview_removed_indices_native']);removed|=one
            length=math.dist(h['entry_mm'],h['tip_mm']);changed=int(previous is not None and previous!=h['tool_id'])
            r=w['target_per_mm3']*h['target_removed_mm3']-w['normal_per_mm3']*h['normal_removed_mm3']-w['action_cost']-2*w['motion_per_mm']*length-w['tool_change_cost']*changed
            close(r,h['reward']);close(2*length,h['complete_tool_path_length_mm'])
            reward+=r;distance+=2*length;toolchanges+=changed;previous=h['tool_id']
        o=row['outcomes'];close(reward,o['total_reward']);close(distance,o['complete_tool_path_length_mm'])
        assert o['tool_changes']==toolchanges and o['normal_removed_mm3']==0
        assert len(removed)==o['positive_target_source_cells_removed']==i
        close(o['target_removed_mm3'],i*geometry['source_voxel_volume_mm3'])
        close(o['reference_target_fraction_removed'],o['target_removed_mm3']/full)
        assert o['total_reference_target_mm3']==full and row['target_access_success']==(i>0)
        c=control['histories'][i];assert c['ordinal']==i and c['seal']==wrapper['plan_seal']
        assert c['plan_sha256']==sha(dest/'plan.json') and c['replay_sha256']==sha(dest/'replay.json')
        summary.append({'ordinal':i,'actions':row['actions'],'return':o['total_reward'],'target_mm3':o['target_removed_mm3'],
            'target_cells':i,'normal_mm3':0,'full_target_mm3':full,'fraction':o['reference_target_fraction_removed'],
            'removed_cells':sorted(removed),'plan_seal':wrapper['plan_seal']})
    assert result['selected_ordinal']==control['selected_ordinal']==max(range(3),key=lambda i:summary[i]['return'])==2
    assert result['selected_plan_seal']==summary[2]['plan_seal'] and result['selected_return']==summary[2]['return']
    assert result['native_counts']==cost['native_counts']==control['native_counts']=={'rollout':6,'replay':6}
    assert cost['diagnostic_previews']==control['diagnostic_previews']==2 and cost['source_visits_attempted']==1
    b=cost['native_budget'];profile=b['native_preview_profile']['phases']['unclassified']
    assert b['counting_reliable'] and b['failure'] is None and b['blocked_preview_attempts']==0
    assert b['history_complete_caller_attestation'] and b['native_preview_entries']==profile['started']==profile['returned']==779
    assert profile['raised']==0 and profile['feasible']+profile['rejected']==779
    saved=dict(pins);assert all(sha(ROOT/p)==d for p,d in saved.items())
    report={'decision':'PASS_SAVED_CORRECTED_MISSING_AXIS_WITNESS','release_sha256':receipt['release_sha256'],
        'result_sha256':receipt['result_sha256'],'source_index_sha256':release['source_index']['sha256'],
        'source_guard':'All80 current source and16 metadata pins passed before root source hold released; canonical80-file closure also verified against execution HEAD as applicable.',
        'original_13_axes_preserved_physical_ledger_rows':130,'appended_axis':[-1,4],'original_failure_preserved':True,
        'underlying_public_world_unchanged':True,'proposal_model_identity_changed':True,'complete_independently_audited_histories':3,
        'histories':summary,'selected_ordinal':2,'native_preview_entries':779,'native_counts':cost['native_counts'],
        'diagnostic_previews':2,'parent_seconds':receipt['elapsed_seconds'],'sampled_peak_rss_bytes':receipt['sampled_peak_rss_bytes'],
        'clean_reap':True,'output_bytes':receipt['output_bytes'],'runtime':final['runtime'],'saved_file_sha256':saved,
        'interpretation':'The appended axis permits exact legal removal of the prior blocker; the subsequent original target ray becomes legal. Both prep alone and prep+target beat STOP after unchanged costs.',
        'limits':['Post-hoc TRAIN action-space diagnosis, not a learned-policy, transfer, or fair comparison to the old13-axis model.',
          'Only2 source cells/1.9073495904mm3 are removed out of77974.3586067mm3 full supplied target.',
          'S union T is an unvalidated material assumption; no clinical or anatomical correctness inferred.',
          'Independent review validates saved full-tool geometry receipts and arithmetic, without rerunning geometry or opening patient arrays.',
          'Observed feasibility under this one changed proposal model does not establish global reachability or optimality.']}
    report['audit_script_sha256']=sha(Path(__file__))
    with Path(__file__).with_name('audit.json').open('x') as f:json.dump(report,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    print(json.dumps({k:report[k] for k in ('decision','histories','parent_seconds','sampled_peak_rss_bytes')}))

if __name__=='__main__':main()
