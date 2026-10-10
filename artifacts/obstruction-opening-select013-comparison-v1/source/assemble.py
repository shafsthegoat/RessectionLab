"""Compact terminal SELECT source/JSON package. Never read weights or arrays."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'build/obstruction-opening-select013-comparison-v1';OUT=BASE/'attempt-01'
AUDIT=ROOT/'build/obstruction-opening-select013-independent-v1'
PACKAGE=Path(__file__).resolve().parent/'artifacts/obstruction-opening-select013-comparison-v1'
ARMS=('observed_greedy','IL64','RL8','observed_beam')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):
    assert p.is_file() and not p.is_symlink() and p.stat().st_size<=2*1024**2
    return json.loads(p.read_text())
def write(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(v,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
result=read(OUT/'result.json');parent=read(BASE/'attempt-01.supervision/receipt.json');audit=read(AUDIT/'audit-attempt-01.json');release=read(BASE/'root-release.json')
assert sha(OUT/'result.json')=='597d6ad3816d87a071551ac594bdc513c573aaf790bb5023f68794d70b48824b'==parent['result_sha256']
assert sha(AUDIT/'audit-attempt-01.json')=='cb077fefbe30d609b8a3fe3d6ab46ae55973c8c0019229bc6fef94ae02c8ef9f'
assert audit['decision']=='PASS_SAVED_UNION_OBSTRUCTION_SELECT013_COMPARISON' and audit['complete_comparison'] is True
assert parent['status']=='complete' and not parent['cleanup_errors'] and not parent['final_owned_pids']
assert all(result['arms'][a]['actions']==['STOP'] and result['arms'][a]['public_return']==0 for a in ARMS)
assert not PACKAGE.exists();PACKAGE.mkdir(parents=True);entries=[]
def copy(source,target):
    assert source.is_file() and not source.is_symlink() and source.stat().st_size<=2*1024**2
    assert source.suffix not in ('.psckpt','.npz','.npy')
    raw=source.read_bytes();dest=PACKAGE/target;dest.parent.mkdir(parents=True,exist_ok=True)
    with dest.open('xb') as f:f.write(raw)
    entries.append({'path':target,'source_path':str(source.relative_to(ROOT)),'bytes':len(raw),'sha256':sha(dest),'kind':'exact_copy'})
for name in ('comparison_contract.py','select_worker.py','run_owned.py','freeze-runtime.py','test_worker_flow.py','run-controls.py'):
    copy(BASE/name,'source/'+name)
for name in ('controls-01.json','controls-01.log'):copy(BASE/name,'generated-control/'+name)
for name in ('root-release.json','release-template.json','source-index.json'):copy(BASE/name,'provenance/'+name)
for name in ('receipt.json','worker-final.json','declaration.json'):copy(BASE/'attempt-01.supervision'/name,'supervision/'+name)
for name in ('result.json','costs.json','IL64-lineage.json','RL8-lineage.json'):copy(OUT/name,'run/'+name)
for arm in ARMS:
    for name in ('summary.json','comparison-world.json','context.json','complete-trace.json','native-replay.json','plan.json','select-replay.json'):
        copy(OUT/arm/name,'run/'+arm+'/'+name)
    copy(OUT/arm/'source/public-task-derivation.json','run/'+arm+'/public-task-derivation.json')
    if arm.startswith('observed_'):
        for name in ('search-return.json','selection.json'):copy(OUT/arm/name,'run/'+arm+'/'+name)
for name in ('audit_saved.py','audit-attempt-01.json'):copy(AUDIT/name,'audit/'+name)
copy(Path(__file__),'source/assemble.py')
beam=read(OUT/'observed_beam/search-return.json')['accounting'];layers=[]
for layer in beam['layers']:
    diag=layer['retained_prefix_diagnostics'];progress=diag['retained_nonterminal_prefixes']+diag['best_terminal_prefixes']
    opening=layer['opening_depth_retention']
    layers.append({'depth':layer['depth'],'completed':layer['completed'],'model_transition_calls':layer['model_transition_calls'],
        'retained_progress_rows':len(progress),'best_retained_return':max((r['estimated_incremental_return'] for r in progress),default=None),
        'max_retained_target_mm3':max((r['target_removed_mm3'] for r in progress),default=None),
        'max_retained_outside_target_mm3':max((r['outside_supplied_target_removed_mm3'] for r in progress),default=None),
        'max_retained_insertion_mm':max((r['max_insertion_distance_mm'] for r in progress),default=None),
        'max_opening_depth_mm':max((r['opening_depth_mm'] for r in opening),default=None),
        'recorded_terminal_prefixes':len(diag['best_terminal_prefixes'])})
assert len(layers)==24 and all(r['completed'] for r in layers) and sum(r['model_transition_calls'] for r in layers)==519==beam['model_transition_calls']
assert beam['negative_prefixes_evaluated']==519 and beam['call_cap_reached'] is False and beam['time_cap_reached'] is False
progress={'source_path':'run/observed_beam/search-return.json','source_sha256':sha(OUT/'observed_beam/search-return.json'),
    'completed_layers':24,'model_transition_calls':519,'all_evaluated_transition_prefixes_negative':True,
    'retained_progress_rows':sum(r['retained_progress_rows'] for r in layers),'max_retained_target_mm3':max(r['max_retained_target_mm3'] for r in layers),
    'max_retained_outside_target_mm3':max(r['max_retained_outside_target_mm3'] for r in layers),
    'max_retained_insertion_mm':max(r['max_retained_insertion_mm'] for r in layers),
    'max_opening_depth_mm':max(r['max_opening_depth_mm'] for r in layers if r['max_opening_depth_mm'] is not None),
    'selected_plan':['STOP'],'selected_plan_return':0.,'scope':'Recorded bounded beam exploration, not exhaustive reachability or global optimality proof','layers':layers}
write(PACKAGE/'beam-progress-summary.json',progress)
costs=read(OUT/'costs.json');cost_table=[]
for arm in ARMS:
    row={'arm':arm,'public_construction_seconds':costs['phases'][arm+'.public_construction']['complete_wall_seconds'],
        'planning_collection_and_replay_seconds':costs['phases'][arm+'.planning_and_replay']['complete_wall_seconds'],
        'policy_forwards':costs['phases'][arm+'.planning_and_replay'].get('policy_forward_calls',0),
        'public_construction_previews':costs['phases'][arm+'.public_construction']['native_preview_entries'],
        'planning_collection_and_replay_previews':costs['phases'][arm+'.planning_and_replay']['native_preview_entries']}
    if arm.startswith('observed_'):
        selection=read(OUT/arm/'selection.json');row.update(selector_seconds=selection['wall_seconds'],selector_previews=selection['native_previews'])
    cost_table.append(row)
write(PACKAGE/'cost-summary.json',{'parent_seconds':parent['elapsed_seconds'],'sampled_peak_rss_bytes':parent['sampled_peak_rss_bytes'],
    'native_previews':result['native_budget']['native_preview_entries'],'total_policy_forwards':result['total_policy_forward_calls'],
    'checkpoint_reload_seconds':{n:costs['phases']['checkpoint_reload.'+n]['complete_wall_seconds'] for n in ('IL64','RL8')},
    'arms':cost_table,'scope':'Selector time/previews are subsets of planning+collection+replay; do not add them again. Upstream TRAIN cost is separate; no equal-compute claim.'})
summary={'version':'obstruction-opening-select013-comparison-v1','result_sha256':sha(OUT/'result.json'),
    'release_sha256':sha(BASE/'root-release.json'),'source_head':release['expected_head'],'status':result['status'],
    'conclusion':'No target-removal benefit demonstrated on executed SELECT013: all4 arms completed STOP with zero return/removal.',
    'arms':audit['arms'],'checkpoint_lineages':{n:read(OUT/(n+'-lineage.json')) for n in ('IL64','RL8')},
    'planned_SELECT_cases':2,'executed_SELECT_cases':1,'held_cases':['ReMIND-037'],'held_inputs_opened':False,'EVAL_opened':False,
    'optimizer_updates_on_SELECT':0,'gradient_attempts':0,'checkpoint_loads':2,'policy_forwards':2,'native_previews':61290,
    'occupancy_condition':result['occupancy_condition'],'full_supplied_target_voxels':35260,'raw_unsupported_target_voxels':33681,
    'simulated_unsupported_target_voxels_after_explicit_union':0,'beam_progress':{k:v for k,v in progress.items() if k!='layers'},
    'parent_seconds':parent['elapsed_seconds'],'sampled_peak_rss_bytes':parent['sampled_peak_rss_bytes'],
    'output_bytes':parent['output_bytes'],'clean_termination':True,
    'audit':{'decision':audit['decision'],'sha256':sha(AUDIT/'audit-attempt-01.json'),'checks':audit['checks'],'input_bindings':len(audit['evidence_files']),
        'limitations':audit['limits'],'information_scope':audit['information_scope']},
    'payloads_excluded':['checkpoint weights','patient arrays/images','repeated state inventories and frames'],
    'claim_limits':'One executed SELECT case of two planned. No clinical/material validation, global-optimality/reachability impossibility, generalization or equal-compute claim.'}
write(PACKAGE/'summary.json',summary)
readme='''# SELECT013: four complete zero-removal outcomes

Observed greedy, reloaded IL64, reloaded RL8 and observed beam all returned and fully replayed STOP on SELECT013. Every arm had return 0, supplied-target removal 0 and outside-target removal 0. No target-removal benefit was demonstrated. Valid geometric replay of STOP is a completion check, not evidence of task success or clinical safety.

The beam did finish its declared 24 layers: 519 simulated transitions, all with negative prefix returns, within the 300-second local cap. Selector time was 219.388330 seconds and 60,338 native previews. Its 48 recorded retained-progress rows showed no target removal; the deepest recorded opening was about 3.5 mm. The largest retained outside-target removal was 53.999995 mm³, but these are discarded search states, not the executed STOP outcome. This bounded beam result does not prove that no useful route exists. `beam-progress-summary.json` retains the compact layer/progress table, and the exact selector accounting is included.

Both learned endpoints were authenticated and reloaded from their own completed TRAIN runs. All four arms used identical permitted source/model/initial-observation hashes, public-target context, initial action inventory, h24/cap120 and footprint/obstruction proposals. The actor representation and planner computation differ; this is no claim of identical internal information processing. SEARCH actions carry no learned-policy authorship. There were two checkpoint loads, two actual policy forwards, zero SELECT optimizer/gradient attempts or updates, and no private-reference or EVAL access reported by the reviewed runtime.

The world explicitly assumes removable occupancy S OR (T > 0). The unchanged full supplied target has 35,260 positive voxels; 33,681 lie outside raw supplied support and are included only by this declared simulation assumption. The simulated unsupported count becomes zero without repairing the source support or shrinking the target denominator. This is not anatomical or material validation.

SELECT013 is one executed case of the original planned denominator two. SELECT037 remains held with no replacement and no input opened; it is not imputed as a zero-result case. EVAL067 remains closed.

The owned parent completed in 228.130311 seconds with sampled peak RSS 1,270,431,744 bytes, 61,290 native previews and clean termination. Observed-greedy selection alone took 0.008359 seconds after construction; this excludes construction, collection and native replay. `cost-summary.json` separates these scopes. Selector and instrumented subcall times are nested within arm times and must not be added again. Prior TRAIN cost is separate; no equal-compute claim is made.

The independent saved-output audit passed 726 checks across 150 source/JSON bindings. It verified terminal/source/reload lineage, public-world equality, zero-update roles, complete trace/plan/native-replay seals, raw-versus-derived target accounting and costs. No acquired array, checkpoint tensor, model or native simulator was opened or run by the auditor. Recorded geometry and tensor checks were not independently recomputed from payloads.

`MANIFEST.json` gives exact original paths/hashes for copied runtime and test source, releases, costs, checkpoint lineage, complete four-arm traces/plans/replays, selector progress and audit evidence. Checkpoint payloads, patient arrays and repeated inventories are omitted. Scripts are archival and retain original relative-path assumptions; do not run them from this package. Reproduction requires the original layout and a separate owned release.
'''
(PACKAGE/'README.md').write_text(readme)
for name in ('beam-progress-summary.json','cost-summary.json','summary.json','README.md'):
    p=PACKAGE/name;entries.append({'path':name,'source_path':None,'bytes':p.stat().st_size,'sha256':sha(p),'kind':'derived_from_packaged_evidence'})
manifest={'version':'compact-result-manifest-v1','destination':'artifacts/obstruction-opening-select013-comparison-v1','files':sorted(entries,key=lambda r:r['path']),
    'file_count':len(entries),'total_bytes':sum(r['bytes'] for r in entries),'payloads_excluded':True}
write(PACKAGE/'MANIFEST.json',manifest)
for row in entries:
    p=PACKAGE/row['path'];assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256']
    if row['source_path']:assert p.read_bytes()==(ROOT/row['source_path']).read_bytes()
check={'status':'PASS_exact_saved_package_parity','manifest_sha256':sha(PACKAGE/'MANIFEST.json'),'file_count':len(entries),
    'total_bytes_excluding_manifest':manifest['total_bytes'],'total_bytes_including_manifest':manifest['total_bytes']+(PACKAGE/'MANIFEST.json').stat().st_size,
    'checkpoint_or_patient_payloads_read':False,'destination':str(PACKAGE.relative_to(ROOT))}
write(Path(__file__).resolve().parent/'package-check.json',check);print(json.dumps(check,indent=2))
