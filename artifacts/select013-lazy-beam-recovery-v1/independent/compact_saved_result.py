"""Compact already-audited JSON evidence; no arrays, weights or scientific imports."""
import hashlib,json
from pathlib import Path
R=Path(__file__).resolve().parents[2]
D=R/'build/hybrid-lazy-beam-independent-v1'
A=json.loads((D/'audit-result-01.json').read_text())
assert A['decision']=='PASS_SAVED_LAZY_BEAM_RECOVERY'
F=R/'build/select013-lazy-beam-recovery-v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def put(p,v):
    with p.open('x') as s:json.dump(v,s,indent=2,sort_keys=True,allow_nan=False);s.write('\n')
def evidence(relative):
    p=R/relative;m=A['evidence_files'][relative]
    assert sha(p)==m['sha256'] and p.stat().st_size==m['bytes']
    return {'path':relative,**m}
rows=[]
for row in A['layer_progress']:
    prefixes=[]
    for kind in ('retained_nonterminal_prefixes','best_terminal_prefixes'):
        for p in row[kind]:
            q={k:v for k,v in p.items() if k!='actions'}
            q.update(kind=kind,action_count=len(p['actions']),action_sequence_sha256=hashlib.sha256(json.dumps(p['actions'],sort_keys=True,separators=(',',':')).encode()).hexdigest())
            lane=next((r for r in row['opening_depth_retention'] if r['actions']==p['actions']),None)
            if lane:q.update(retention_reason=lane['reason'],opening_depth_mm=lane['opening_depth_mm'],opening_removed_volume_mm3=lane['opening_removed_volume_mm3'])
            prefixes.append(q)
    rows.append({'depth':row['depth'],'completed':row['completed'],'transition_calls':row['calls'],
        'negative_prefixes':row['negative_prefixes'],'saved_progress':prefixes})
prior_path='build/post-exposure-select013-expanded-comparison-v1/attempt-01/observed_beam/search-unresolved.json'
prior=json.loads((R/prior_path).read_text())
index=json.loads((F/'source-index.json').read_text())
summary={k:A[k] for k in ['decision','complete_recovered_beam','original_four_arm_result_still_incomplete','arms','world',
    'prior_completed_layers_verified','layer_control_exclusions','prior_partial_fifth_layer_compared','search_accounting',
    'max_saved_retained_or_terminal_target_mm3','progress_scope','cost_phases','cost_scope','native_previews','policy_forwards',
    'checkpoint_loads','checkpoint_role','optimizer_updates_on_SELECT','gradient_attempts','private_reference_reads',
    'planned_SELECT_denominator','executed_SELECT_cases','held_cases','EVAL_closed','elapsed_seconds','sampled_peak_rss_bytes',
    'output_bytes','cleanup_errors','final_owned_pids','worker_termination_confirmed','limits']}
summary.update(schema='SELECT013-lazy-beam-saved-independent-result-v1',source_head=index['head'],
    source_files_verified=len(index['source_files']),metadata_pins_verified=len(index['metadata_files']),
    independent_checks=A['checks'],independent_input_files=len(A['evidence_files']),
    independent_audit_seconds=A['audit_seconds'],independent_audit_peak_rss_bytes=A['audit_peak_rss_bytes'],
    layer_progress=rows,scientific_outcome='Complete negative result: selected STOP, zero target removal, zero return. More search depth did not yield useful recorded target progress.',
    all_evaluated_prefix_returns_negative=A['search_accounting']['negative_prefixes_evaluated']==A['search_accounting']['evaluated_transition_prefixes'],
    original_capped_search={'binding':evidence(prior_path),'completed_layers':prior['accounting']['completed_layers'],
        'transitions':prior['accounting']['model_transition_calls'],'planning_seconds':prior['accounting']['planning_seconds'],
        'search_native_previews':28782,'accepted_plan':False},
    comparison_limits='One scheduling recovery under the same admitted world and ranking, with exact four completed-layer controls. Old and new searches evaluated different amounts of work; these timings are not an equal-work replicated speedup benchmark. Saved retained/terminal progress does not enumerate every discarded successor.',
    declared_caps={'worker_seconds':900,'parent_seconds':960,'sampled_tree_rss_bytes':3*1024**3,
        'output_bytes':128*1024**2,'supervision_bytes':16*1024**2,'threads':1,'attempts':1,'automatic_retry':False,
        'search_seconds':300,'search_calls':5640,'beam_width':2,'horizon':24,'native_previews':750000,
        'actual_actor_forwards':0,'historical_factory_forward_envelope':48},
    terminal_pins={k:evidence('build/select013-lazy-beam-recovery-v1/'+p) for k,p in {
        'release':'root-release.json','source_index':'source-index.json','parent':'attempt-01.supervision/receipt.json',
        'worker':'attempt-01.supervision/worker-final.json','result':'attempt-01/result.json',
        'selection':'attempt-01/observed_beam/selection.json','costs':'attempt-01/costs.json'}.items()})
put(D/'compact-result.json',summary)
put(D/'input-hashes.json',A['evidence_files'])
for name in ('plan.json','native-replay.json','select-replay.json','complete-trace.json'):
    relative='build/select013-lazy-beam-recovery-v1/attempt-01/observed_beam/'+name
    p=R/relative;evidence(relative)
    with (D/('accepted-'+name)).open('xb') as s:s.write(p.read_bytes())
report='''PASS — saved SELECT013 lazy-beam recovery; completed negative scientific result.

The recovered public beam completed all 24 layers and 725 model transitions in
79.719992250 seconds (300-second local cap), selected exactly ["STOP"], and passed
complete trace, sealed plan and independent native replay joins. Return, supplied
target removal, outside-target removal and path length are all zero. There were
zero actor forwards, optimizer attempts, gradient attempts or SELECT updates.
The one original IL64 checkpoint was loaded solely as the existing admission/
replay reference and did not author actions.

This resolves the measured eager-successor-inventory execution bottleneck, but
shows no useful target-progress outcome. All 725 evaluated cumulative prefixes
had negative return. All recorded retained and terminal candidates had zero target
removal. The opening lane reached approximately 4.5 mm by layer 2 and did not
exceed that depth; it accumulated up to 43.999995750 mm3 outside the supplied target.
The best saved 24-motion terminal candidate had return -5.984199518, zero target
removal and 24.999997585 mm3 outside the supplied target; STOP remained preferable.
The progress logs do not enumerate target removal in every discarded successor.
Neither a globally optimal STOP decision nor physical impossibility is established.

The exact source, decision model, proposal, initial observation, ordered action
inventory/mask, post-exposure condition and partial source-domain records equal
the prior expanded SELECT013 attempt. The original four completed layer projections
match exactly (37,72,66,61 transitions), excluding only legal_nonstop_actions and
successor_inventory_status, which reflect later inventory availability. The old
partial fifth layer is not compared. Horizon24, cap120, original13 axes, footprint,
world, reward, tools, depth-volume ranking, beam2 and max5640 calls are unchanged.
The original EMPTY_FIXED_K failure and later capped four-arm result remain intact.
This is a separate accepted beam result, not retroactive completion of that attempt.

Measured recovery costs: public construction 1.520595792 s /117 previews;
search phase 79.727862666 s /5382 previews (selector 79.719992250 s);
collection 0.095118791 s /0 previews; independent replay 1.351520500 s /117 previews;
reference checkpoint reload 0.013633958 s. Total 5616 previews. Parent elapsed
86.929392833 s, sampled peak RSS561709056 B, output690036 B, exit0, clean direct-child
termination with no cleanup errors or survivors. Original outer900/960 s,3GiB,
128MiB output and16MiB supervision caps remain. Nested inventory/transition/clone
timers are inclusive and must not be added to phase totals. Search inventory time
was 62.859830746 s versus293.554574491 s previously, while work increased from246
transitions/four complete layers to725/24. This is not a replicated equal-work
speedup estimate.

SELECT denominator remains2:013 executed,037 held unchanged;067 EVAL stays closed.
Full supplied target35260 voxels remains unchanged. Source-domain unknown176294
voxels and raw support/target mismatch33681 voxels remain explicit under simulated
S-union-T material and deterministic public post-exposure assumptions. No clinical,
anatomical accuracy, material truth or observed-surgical-demonstration claim follows.
Legality and independent replay use only the admitted public estimated world.

Independent audit:2206 checks over151 saved source/JSON files (5786248 bytes),
0.056608542 s,27475968 B high-water RSS, first execution PASS. Exact input hashes
were checked again at termination. No arrays, weights, model forwards, native
replay or geometric recomputation were performed by this review. Saved receipts
and reviewed source establish these recorded calls; no independent syscall trace
was collected. Full accepted plan/trace/replay certificates and all24 compact layer
progress rows accompany this report. Audit source preserves a separate unresolved
branch; the actual result used the complete branch.

Exact root bindings:
HEAD fe857eed61ad2fc3cace011bcef523f7b43a1923
release43e7cdf0e43a014154cc15236bbb4df1818ed1577b3cee5b8d75ac368e9aa3d9
source-index1a9793219207a94ff3c71f654d50b02c7c196ab4cf3f0198260d4ea45aa258e1
result96e0b98069689a7ad6f1b986423f527ce931e04034b33605d3ebaf48e9e9d761
parent83031cb5bdeb9c9ee6bd47dcf53f28e3c7ea7f83207bc94b174d8501b994f232
selection8f62656d1081ea247499cb1ff7ed8b803bf2249d3e700703d3806506a151c8f4
costsc7d0ea71d90c036cb66727f6cd96b6957ed03d5c299a7257ad06f7cb1ffcee17
'''
# Spacing around numeric labels is deliberate in the final human-readable text.
for a,b in [('RSS561','RSS 561'),('output690','output 690'),('exit0','exit 0'),('outer900','outer 900'),(',3GiB',', 3 GiB'),(',128MiB',', 128 MiB'),('and16MiB','and 16 MiB'),('versus293','versus 293'),('from246','from 246'),('to725','to 725'),('denominator remains2','denominator remains 2'),(':013',': 013'),(',037',', 037'),(';067','; 067'),('target35260','target 35260'),('unknown176294','unknown 176294'),('mismatch33681','mismatch 33681'),('audit:2206','audit: 2206'),('over151','over 151'),('s,274','s, 274'),('all24','all 24'),('release43','release 43'),('source-index1a','source-index 1a'),('result96','result 96'),('parent830','parent 830'),('selection8f','selection 8f'),('costsc7','costs c7')]:report=report.replace(a,b)
with (D/'SAVED_RESULT_REPORT.txt').open('x') as s:s.write(report)
files=['audit_saved.py','audit-result-01.json','compact_saved_result.py','compact-result.json','input-hashes.json','SAVED_RESULT_REPORT.txt']+['accepted-'+n for n in ('plan.json','native-replay.json','select-replay.json','complete-trace.json')]
idx={'schema':'saved-independent-review-file-index-v1','scope':'source and saved JSON only; no binary payloads',
     'files':{n:{'sha256':sha(D/n),'bytes':(D/n).stat().st_size} for n in files}}
put(D/'saved-review-index.json',idx)
print(json.dumps({'index_sha256':sha(D/'saved-review-index.json'),'files':idx['files'],'total_bytes':sum(m['bytes'] for m in idx['files'].values())},indent=2))
