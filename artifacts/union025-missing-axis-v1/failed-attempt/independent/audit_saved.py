"""Saved-only failure and partial-evidence audit; stdlib, no native or array loads."""
import hashlib
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'build/union025-missing-axis-v1'
OUT=BASE/'attempt-01';SUP=BASE/'attempt-01.supervision'
pins={}
def read(p):
    p=Path(p);assert p.is_file() and not p.is_symlink() and p.stat().st_size<8*1024**2
    data=p.read_bytes();pins[str(p.relative_to(ROOT))]=hashlib.sha256(data).hexdigest();return data
def load(p):return json.loads(read(p))
def sha(p):return hashlib.sha256(read(p)).hexdigest()
def semantic(v):return 'sha256:'+hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def physical(v):return {k:v[k] for k in ('tool_id','entry_mm','tip_mm')}
def without_ids(rows):return [{k:v for k,v in r.items() if k not in ('action_id','proposal_id')} for r in rows]
def close(a,b):assert math.isclose(a,b,rel_tol=1e-9,abs_tol=1e-9),(a,b)

def main():
    assert sha(BASE/'root-release.json')=='7fee8d087469790fb48e6b8289a048201feba605a173424a6a0e142896b971aa'
    assert sha(BASE/'source-index.json')=='2e12da647223d4ec5b26fd19ec70160afc6032721ebf7e152d049f00c4d9b7ef'
    release=load(BASE/'root-release.json');index=load(BASE/'source-index.json')
    assert release['expected_head']==index['head']=='7947d23529a38d4cb8e8348fbba142c23467a6c3'
    for section in ('source_files','metadata_files'):
        for p,d in index[section].items():assert sha(ROOT/p)==d,p
    refs=load(BASE/'input-index.json')['files']
    old=load(ROOT/refs['union_world']['path']);oldinv=load(ROOT/refs['union_inventory']['path'])
    learning=load(ROOT/refs['learning_release']['path'])
    world=load(OUT/'world.json');context=load(OUT/'context.json');inv=load(OUT/'initial-inventory.json')
    assert semantic(context)==world['context_hash']
    for k,v in world['unchanged_public_checks'].items():assert v==old['common_public_world'][k],k
    for k in ('derived_occupancy','supplied_goal_extent','normalization'):assert world[k]==old[k]
    for k in ('source_hash','decision_model_hash','initial_observation_hash'):assert world[k]!=old[k]
    config=world['proposal_config'];prior=learning['learning_protocol']['cohort_execution']['proposal_config']
    assert config['offsets_source_voxels']==prior['offsets_source_voxels']+[[-1,4]]
    assert len(config['offsets_source_voxels'])==14
    assert {k:v for k,v in config.items() if k!='offsets_source_voxels'}=={k:v for k,v in prior.items() if k!='offsets_source_voxels'}
    oldrows=[r for r in inv['ledger'] if r['column_index']<13]
    assert without_ids(oldrows)==without_ids(oldinv['ledger']) and inv['omitted_count']==0
    assert context['role']=='TRAIN' and context['subject']=='ReMIND-025'
    assert context['private_reference_in_task'] is False and context['max_optimizer_updates']==0
    assert context['budgets']['max_policy_forwards']==0 and context['derived_occupancy_anatomically_validated'] is False
    result=load(OUT/'result.json');cost=load(OUT/'costs.json');receipt=load(SUP/'receipt.json');final=load(SUP/'worker-final.json')
    assert receipt['status']==result['status']==final['status']=='failed_or_unresolved'
    assert receipt['exit_code']==1 and receipt['worker_termination_confirmed']
    assert not receipt['final_owned_pids'] and not receipt['cleanup_errors']
    assert receipt['result_sha256']==final['canonical_result_sha256']==sha(OUT/'result.json')
    assert receipt['worker_final_sha256']==sha(SUP/'worker-final.json')
    assert result['failure']['type']==final['failure']['type']=='FileExistsError'
    assert 'diagnostics.json' in result['failure']['message']
    log=read(SUP/'worker.log').decode()
    assert "diagnostics[name]=record;write(output/'diagnostics.json',diagnostics)" in log
    assert "path.open('xb')" in log
    assert not (SUP/'endpoint-control.json').exists()
    assert receipt['elapsed_seconds']<90 and final['wall_seconds']<60 and receipt['sampled_peak_rss_bytes']<3*1024**3
    assert receipt['output_bytes']==sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())<32*1024**2
    assert cost['native_counts']==result['native_counts']=={'rollout':2,'replay':1}
    assert cost['diagnostic_previews']==2 and cost['native_budget']['native_preview_entries']==224
    assert cost['native_budget']['counting_reliable'] and cost['native_budget']['failure']=='arm_raised_FileExistsError'
    assert all(result[k]==0 for k in ('optimizer_updates','policy_forwards','checkpoint_loads','search_calls','private_reference_reads'))
    assert all(v==0 for v in cost['prohibited_calls'].values()) and result['SELECT_EVAL_opened'] is False
    # Only the STOP alternative reached its complete independent audit.
    assert len(result['histories'])==1
    plan=load(OUT/'history-00/plan.json');replay=load(OUT/'history-00/replay.json')
    body=plan['plan'];audit=replay['independent_geometry']
    assert body['actions']==['STOP'] and body['history']==replay['metrics']['history']
    assert semantic(body)==plan['plan_seal']==result['histories'][0]['plan_seal']
    assert audit['accepted'] and audit['complete_episode'] and audit['committed_history_hash']==semantic(body['history'])
    assert audit['geometry']['complete_tool_checked'] and not audit['geometry']['failures']
    assert audit['outcomes']==result['histories'][0]['outcomes'] and audit['outcomes']['total_reward']==0
    # The prep committed, but its STOP, sealed plan and independent replay never ran.
    assert sorted(p.name for p in (OUT/'history-01').iterdir())==['attempt-00.json','returned-00.json']
    prep=load(OUT/'history-01/returned-00.json')
    diagnostics=load(OUT/'diagnostics.json');assert list(diagnostics)==['clearance']
    rays=load(OUT/'prospective-rays.json');partial={}
    for name in ('clearance','target'):
        d=load(OUT/name/'fixed-ray-preview.json');inventory=load(OUT/name/'inventory.json')
        assert physical(d)==rays[name] and d['feasible'] and d['reason']=='NATIVE_CONNECTED_STROKE'
        assert d['committed'] is False and d['temporary_removals_committed'] is False
        assert d['obstruction_diagnostic'] is None
        matches=[r for r in inventory['emitted'] if physical(r)==rays[name]]
        assert len(matches)==1 and matches[0]['feasible'] and inventory['omitted_count']==0
        assert inventory['cavity_state_hash']==d['source_state_hash']
        assert d['source_hash']==world['source_hash'] and d['task_model_hash']==world['decision_model_hash']
        if name=='clearance':
            assert d['source_state_hash']==world['initial_state_hash']
            assert prep['action_id']==matches[0]['action_id']==diagnostics['clearance']['selected_action_id']
            assert diagnostics['clearance']['action_status']=='executable'
            assert physical(prep)==rays[name]
            assert prep['removed_indices_native']==d['feasible_preview_removed_indices_native']
        else:assert d['source_state_hash']!=world['initial_state_hash']
        partial[name]={'preview_feasible':True,'exact_emitted_legal_matches':1,'selected_geometry':rays[name],
            'preview_removed_cells':d['feasible_preview_removed_indices_native'],'native_reason':d['reason']}
    w=world['unchanged_public_checks']['reward'];length=math.dist(prep['entry_mm'],prep['tip_mm'])
    r=w['target_per_mm3']*prep['target_removed_mm3']-w['normal_per_mm3']*prep['normal_removed_mm3']-w['action_cost']-2*w['motion_per_mm']*length
    close(prep['reward'],r);close(prep['complete_tool_path_length_mm'],2*length)
    assert 'selected_ordinal' not in result and result['source_released'] is False
    saved_pins=dict(pins);assert all(sha(ROOT/p)==d for p,d in saved_pins.items())
    report={'decision':'VALID_PRESERVED_SOFTWARE_FAILURE_UNRESOLVED_SCIENTIFIC_OUTCOME',
        'release_sha256':receipt['release_sha256'],'result_sha256':receipt['result_sha256'],
        'failure':'Second examine writes diagnostics.json through exclusive OutputBudget.write; FileExistsError before prep STOP/replay and target execution.',
        'source_files_verified':len(index['source_files']),'metadata_files_verified':len(index['metadata_files']),
        'original_13_axes_unchanged_physical_ledger_rows':len(oldrows),'changed_axis_config':config,
        'source_model_observation_ids_changed':True,'unchanged_public_arrays_tools_access_reward_verified_by_saved_bindings':True,
        'complete_audited_histories':1,'complete_history':'STOP0 only','partial_preparation':
          {k:prep[k] for k in ('action_id','reward','target_removed_mm3','normal_removed_mm3','complete_tool_path_length_mm')},
        'partial_diagnostics':partial,'native_counts':cost['native_counts'],'native_preview_entries':224,'diagnostic_previews':2,
        'parent_seconds':receipt['elapsed_seconds'],'sampled_peak_rss_bytes':receipt['sampled_peak_rss_bytes'],
        'clean_reap':True,'runtime':final['runtime'],'saved_file_sha256':saved_pins,
        'minimal_repair':'Use unique per-examination receipts and write aggregate diagnostics once; test two examinations with an exclusive output sink. Preserve this attempt unchanged.',
        'limits':['Partial native-feasible evidence is not a completed witness or independently audited preparation.',
          'No target action executed and no winner selected. Neither success nor global failure is established.',
          'Source_released=false is the interrupted weak-reference check; parent confirms OS worker termination.',
          'Post-hoc TRAIN action-space diagnosis only; S union T remains an unvalidated material assumption.',
          'Review reads saved JSON/source only; no patient arrays, native/model imports, or replay execution.']}
    report['audit_script_sha256']=sha(Path(__file__))
    with Path(__file__).with_name('audit.json').open('x') as f:json.dump(report,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    print(json.dumps({k:report[k] for k in ('decision','failure','original_13_axes_unchanged_physical_ledger_rows','partial_preparation','native_counts')}))

if __name__=='__main__':main()
