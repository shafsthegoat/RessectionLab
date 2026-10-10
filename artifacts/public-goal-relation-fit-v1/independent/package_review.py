"""Package already-completed saved arithmetic; no run/checkpoint/model access."""
from pathlib import Path
import hashlib,json

out=Path(__file__).resolve().parent
audit_path=out/'audit.json';audit_bytes=audit_path.read_bytes();a=json.loads(audit_bytes)
assert a['status']=='PASS_SAVED_JSON_ONLY'
old=a['baseline_summaries']['after'];new=a['summaries']['after'];cost=a['cost_comparison']
changed=[]
for pair in a['paired_roots']:
    for goal in pair['goals']:
        if goal['new_choice']!=goal['baseline_choice']:
            changed.append({'layout_id':pair['layout_id'],**goal})
assert len(changed)==2 and all(x['layout_id']=='pcf-19' and x['baseline_correct'] and not x['new_correct'] and x['new_choice']=='STOP' for x in changed)
summary={
    'schema':'public-goal-relation-fit-saved-review-v1',
    'decision':'PASS_SAVED_JSON_ARITHMETIC',
    'performance_improvement_established':False,
    'trajectory_or_clinical_validation':False,
    'result_sha256':a['result_sha256'],'receipt_sha256':a['receipt_sha256'],
    'audit_sha256':hashlib.sha256(audit_bytes).hexdigest(),
    'run_head':a['run_head'],'experiment_hash':a['experiment_hash'],
    'source_index_sha256':a['source_index_sha256'],
    'initial_function_parity':{'logit_max_abs_difference':a['initial_function_max_logit_difference'],'value_max_abs_difference':a['initial_function_max_value_difference'],'declared_tolerance':1e-6},
    'initial_tensor_evidence':{'producer_shared_exact':True,'producer_added_zero_parameters':704,'independent_tensor_read':False},
    'counts':a['counts'],
    'classes':{k:{'n':new[k]['n'],'baseline_correct':old[k]['correct_count'],'new_correct':new[k]['correct_count'],'baseline_cross_entropy':old[k]['mean_cross_entropy'],'new_cross_entropy':new[k]['mean_cross_entropy'],'baseline_STOP':old[k]['greedy_STOP'],'new_STOP':new[k]['greedy_STOP']} for k in new},
    'new_aspiration_roots':{'exact_teacher_aspiration':2,'different_aspiration':2,'STOP':12,'denominator':16},
    'paired_root_tasks':{'layouts':12,'goals_per_layout':2,'unchanged_choice_vs_baseline':22,'changed_choice_vs_baseline':2,'changed_rows':changed,'cross_goal_action_id_caveat':'Action identities include source/task/engine state; differing IDs across goals do not establish differing physical geometry. Paired comparison here uses each goal\'s unchanged before/after/baseline inventory.'},
    'conditional_movement_rank1':{'new':25,'baseline':24,'denominator':32,'STOP_policy_removed':False},
    'costs':{**cost,'parent_wall_delta_seconds':cost['new_parent_wall_seconds']-cost['old_parent_wall_seconds'],'sampled_peak_rss_delta_bytes':cost['new_sampled_peak_rss_bytes']-cost['old_sampled_peak_rss_bytes']},
    'audit_resources':{'wall_seconds':a['audit_elapsed_seconds'],'observed_peak_self_rss_bytes':a['audit_peak_self_rss_bytes'],'wall_limit_seconds':60,'observed_rss_limit_bytes':512*1024**2,'files_rechecked':a['source_and_result_files_unchanged']},
    'limitations':a['limitations'],
}
(out/'result-summary.json').write_text(json.dumps(summary,sort_keys=True,indent=2,allow_nan=False)+'\n')
report='''Independent public goal-relation full40 fit audit — PASS_SAVED_JSON_ONLY

The retained result is internally consistent and complete, but the new fit does not establish a policy improvement. Mean teacher-state cross-entropy is 1.670608912 versus the unchanged full40 baseline's 1.700327130. Exact greedy choices decrease from 28/40 (70%) to 26/40 (65%). The common initial function is effectively unchanged: maximum legal-logit difference 7.450580597e-9, maximum value difference 0, within the predeclared 1e-6 tolerance across all 40 states.

All fixed denominators are retained; new versus baseline final correct choices are:
  all40              26/40 versus 28/40
  root24             10/24 versus 12/24
  second16           16/16 versus 16/16
  STOP8                8/8 versus 8/8
  movement32         18/32 versus 20/32
  aspiration_root16   2/16 versus 4/16

The 16 aspiration roots now choose 2 exact teacher aspirations, 2 different aspirations and 12 STOPs, versus 4 exact, 2 different and 10 STOPs in the baseline. All 40 new choices comprise 20 STOP, 4 aspiration and 16 probe actions. Conditional movement-only teacher rank improves 24/32 to 25/32; this is score arithmetic and does not remove STOP from the actual policy. The lowest class-specific CE does not imply the best greedy decision: aspiration-root CE is slightly worse (1.990516966 versus 1.989446860).

All 12 layouts and both named goals are included. Comparing the same goal's exact bound action inventory against the old baseline, 22/24 root choices are unchanged. Both pcf-19 goals change from a correct teacher aspiration to STOP; no previously wrong root choice becomes correct. The pcf-14 and pcf-23 surface roots retain the same nonteacher action IDs; their deep roots retain the exact teacher choice. Different action IDs across goals cannot by themselves prove different physical actions: the canonical identity includes source/task/engine-state bindings. The audit reports all paired score records without calling task-specific IDs a geometry comparison. No new trajectory, contact, removal, return or success evidence was generated. Earlier baseline trajectory results do not transfer to this checkpoint.

Source and provenance: run HEAD d2c68559288196bb571a7016fe4aadcb9eda9a99; experiment sha256:770de4707a3cde985c57fc214964b47c460ed5efd1985d83e6cf4277e3ff54a1. The exact ordered 40-state corpus, 24 TRAIN task identities, family/source role manifest, action masks, teacher indices and fixed optimization settings match the baseline. The expanded initialization receipt records all shared tensors exact and 704 appended parameters zero (39,099 -> 39,803 parameters). Saved initial logits independently substantiate the functional parity; this audit deliberately did not decode either checkpoint and therefore does not independently reconstruct tensor equality or zero columns.

All 32 ordered update receipts form a changed-parameter chain from the new initialization to the final parameter hash. Each update records all 40 supervised states: 1,280 loss forwards in total. The before/after endpoint has 80 readout forwards; no alternate endpoint was selected. Gradient receipt norms span 0.189457297 to 0.487855196, below the fixed norm-5 clip, and the critic IL gradient is recorded as zero. These are authenticated producer gradient records, not an independent backward pass. All legal logits independently reproduce softmax probabilities, CE, entropy, masked inventory-first argmax, STOP margins, movement ranks and every class aggregate with 1e-10 arithmetic tolerance; initial float32 loss agrees within 2e-7.

Reconstruction/cost records reconcile 24 teacher reconstructions, 40 saved native teacher steps (32 non-STOP), 760 previews, 24 geometry audits, 1,280 loss and 80 readout forwards, 32 optimizer updates, one reload of the newly saved checkpoint, zero prior-checkpoint loads and zero new SEARCH calls. Producer records report zero patient and SELECT/MEASUREMENT task reads. The prospective all-role manifest is metadata only; this audit did not inspect task inputs or anatomy. Equal update/forward counts do not mean equal total compute: there are 704 additional parameters and the actual recorded costs remain explicit.

Root supervision is complete exit 0: parent 113.679587625 s, worker 113.098729750 s, sampled owned-tree RSS 401,162,240 B, 555 samples, no stop reason, no cleanup actions/errors and no remaining owned PIDs. Limits were 180 s, 1 GiB, one attempt and one thread. Baseline parent/worker/RSS were 110.954839417 s / 110.515916542 s / 399,704,064 B. New parent time is +2.724748208 s and sampled peak is +1,458,176 B. These single-run observed costs do not establish a timing regression or matched-total-compute result. Resource sampling may miss transient peaks or detached descendants between samples.

The independent stdlib audit completed in 0.101459209 s at 26,083,328 B observed peak self RSS, within the 60 s / 512 MiB scope. It rechecked 264 exact source/evidence files unchanged and verified the complete 140-name run inventory; checkpoint names were inventoried but checkpoint bytes were never opened. No project/scientific/model import, checkpoint/tensor/array/input read, native reconstruction, replay, optimizer, patient or network/process action occurred. No blocked-access attempts occurred. Runtime binaries and actual gradient/tensor computations were not independently reproduced.

Result SHA-256: 4eed507e1ceb9160714cfe3e64101415fa7d26ae62b80406633e8f93ab2e70db
Receipt SHA-256: 0da4fef6cf7556db95a77c1191c6e5f895a130f710110579bc719b6a3e51f179
Source-index SHA-256: c2feb9288dcd15f4feeb85cb999d86fa4c190600c80069ac11cde3352b39274c

This is a complete, reproducible TRAIN teacher-state negative/mixed comparison. It supplies neither on-policy trajectory results, unseen-task generalization nor clinical validation. The completion/arithmetic PASS does not authorize a new rollout or model execution.
'''
(out/'REPORT.txt').write_text(report)
names=['AUDIT-SCOPE.txt','audit_saved.py','audit-console.txt','audit.json','source-and-output-hashes.json','package_review.py','result-summary.json','REPORT.txt']
files=[]
for name in names:
    data=(out/name).read_bytes();files.append({'path':name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
index={'schema':'goal-relation-fit-independent-index-v1','files':files,'total_indexed_bytes':sum(x['bytes'] for x in files),'no_checkpoint_tensor_or_array_copies':True}
(out/'index.json').write_text(json.dumps(index,sort_keys=True,indent=2)+'\n')
print(json.dumps({'report_sha256':next(x['sha256'] for x in files if x['path']=='REPORT.txt'),'summary_sha256':next(x['sha256'] for x in files if x['path']=='result-summary.json'),'index_sha256':hashlib.sha256((out/'index.json').read_bytes()).hexdigest(),'indexed_bytes':index['total_indexed_bytes']}))
