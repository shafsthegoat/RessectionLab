"""Independent stdlib audit of saved refit evidence; never imports project/ML code."""
from pathlib import Path
import hashlib
import json
import math
import statistics
import time

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'build/goal-conditioned-policy-v1/full-teacher-refit-run-v1'
SUP = BASE.with_name(BASE.name + '.supervision')
PREP = ROOT / 'build/goal-conditioned-policy-v1/full-teacher-refit-v1'
OUT = Path(__file__).resolve().parent
START = time.monotonic()
seen = {}

def sha(path):
    # Checkpoint files are opaque bytes for identity only; never decoded.
    value = hashlib.sha256(path.read_bytes()).hexdigest()
    seen[str(path.relative_to(ROOT))] = {'sha256': value, 'bytes': path.stat().st_size}
    return value

def read(path):
    sha(path)
    return json.loads(path.read_text())

def digest(value):
    return 'sha256:' + hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()

def close(left, right):
    assert math.isclose(left, right, rel_tol=1e-10, abs_tol=1e-10), (left, right)

result = read(BASE / 'result.json')
receipt = read(SUP / 'receipt.json')
declaration = read(SUP / 'declaration.json')
progress = read(SUP / 'progress.json')
sha(SUP / 'worker.log')
assert receipt['status'] == 'complete' and receipt['exit_code'] == 0
assert receipt['result_sha256'] == sha(BASE / 'result.json')
assert not receipt['final_owned_pids'] and receipt['worker_termination_confirmed']
assert not receipt['cleanup_errors'] and receipt['stop_reason'] is None
assert receipt['elapsed_seconds'] < 180 and receipt['sampled_peak_rss_bytes'] < 2**30
assert declaration['head'] == 'ad819927258f251446b08ba378de3d80d7355e4f'
source = read(PREP / 'source-index.json')
assert sha(PREP / 'source-index.json') == declaration['source_index_sha256'] == '3bd47a8d388ac5748cb12e75f33132e9014b4c98e82199ac5c7ff44dc322e4ff'
assert source['source_files'] == declaration['source_files']
for path, value in source['source_files'].items():
    assert sha(ROOT / path) == value, path
assert sha(PREP / 'input-index.json') == declaration['input_index_sha256'] == source['input_index']['sha256']
prospective = read(PREP / 'prospective-experiment.json')
experiment = read(BASE / 'experiment-freeze.json')
assert experiment == prospective
assert digest(experiment) == result['experiment_hash'] == source['refit_experiment_hash']
assert result['experiment_hash'] != result['baseline_experiment_hash'] == source['baseline_experiment_hash']
assert experiment['version'] == 'generated-public-contact-full-teacher-refit-v1'
corpus = experiment['teacher_states']
assert len(corpus) == 40 and sum(r['action_id'] == 'STOP' for r in corpus) == 8
assert sum(r['step'] == 0 for r in corpus) == 24 and sum(r['step'] == 1 for r in corpus) == 16
roles = {r['layout_id']: r['role'] for r in experiment['family_manifest']['source_bindings']}
assert all(roles[r['layout_id']] == 'TRAIN' for r in corpus)

reconstructions = [read(BASE / f'teacher-reconstruction-{i:02d}.json') for i in range(24)]
assert reconstructions == result['completed_teacher_reconstructions']
roots = [r for r in corpus if r['step'] == 0]
for i, (reconstruction, root) in enumerate(zip(reconstructions, roots)):
    assert reconstruction['teacher_index'] == i
    assert reconstruction['original_binding_hash'] == root['original_binding_hash']
    assert reconstruction['strategy_seal'] == root['strategy_seal']
    assert reconstruction['refit_binding_hash'] != reconstruction['original_binding_hash']

initial = result['initial_checkpoint']
final = result['final_checkpoint']
for meta, name, updates in [(initial, 'refit-initial.gmckpt', 0), (final, 'refit-IL-final.gmckpt', 32)]:
    path = BASE / name
    assert Path(meta['path']) == path and sha(path) == meta['sha256']
    assert path.stat().st_size == meta['bytes']
    lineage = meta['lineage']
    assert lineage['experiment_hash'] == result['experiment_hash']
    assert lineage['learning_contract_version'] == experiment['version']
    assert lineage['optimizer_updates'] == updates and lineage['real_patient_count'] == 0
    assert lineage['parameter_hash'] == meta['parameter_hash']
assert initial['parameter_hash'] == 'sha256:e7215950e221045b8e9612e4482837f9b1ff2c52278454b26efc746598cf1487'
assert final['parameter_hash'] != initial['parameter_hash']
assert final['lineage']['initial_parameter_hash'] == initial['parameter_hash']
assert {digest(b) for b in final['lineage']['training_bindings']} == {r['refit_binding_hash'] for r in reconstructions}
assert all(b['role'] == 'TRAIN' and b['experiment_hash'] == result['experiment_hash'] for b in final['lineage']['training_bindings'])

updates = [read(BASE / f'update-{i:02d}.json') for i in range(1, 33)]
previous = initial['parameter_hash']
for i, update in enumerate(updates, 1):
    u = update['update_receipt']
    assert update['update'] == u['cumulative_updates'] == i and u['optimizer_steps'] == 1
    assert u['version'] == experiment['version'] and update['experiment_hash'] == result['experiment_hash']
    assert update['all40_ordered_state_manifest_hash'] == digest(corpus)
    assert update['loss']['supervised_actions'] == update['loss']['loss_forward_calls'] == 40
    assert u['initial_parameter_hash'] == previous and u['parameters_changed']
    assert u['updated_parameter_hash'] != previous
    assert 0 < u['gradient_norm_before_clip'] < experiment['protocol']['max_gradient_norm']
    assert u['module_gradient_norms_before_clip']['critic'] == 0
    previous = u['updated_parameter_hash']
assert previous == final['parameter_hash']

readouts = {}
summaries = {}
fixed_keys = ('teacher_index', 'layout_id', 'goal_id', 'step', 'binding_hash', 'teacher_action', 'observation_hash', 'action_ids', 'action_modes', 'legal_mask', 'teacher_action_index')
for label, params in [('before', initial['parameter_hash']), ('after', final['parameter_hash'])]:
    rows = [read(BASE / f'{label}-state-{i:02d}.json') for i in range(40)]
    readouts[label] = rows
    for row, expected in zip(rows, corpus):
        assert row['parameter_hash'] == params
        for a, b in [('layout_id','layout_id'),('goal_id','goal_id'),('step','step'),('binding_hash','original_binding_hash'),('teacher_action','action_id'),('observation_hash','observation_hash')]:
            assert row[a] == expected[b]
        ids, modes, mask, logits = (row[k] for k in ('action_ids','action_modes','legal_mask','logits'))
        assert len(ids) == len(modes) == len(mask) == len(logits)
        assert ids[0] == 'STOP' and modes[0] == 'stop' and mask[0]
        legal = [i for i, allowed in enumerate(mask) if allowed]
        assert all(type(m) is bool for m in mask)
        assert all(math.isfinite(logits[i]) for i in legal)
        assert all(logits[i] is None for i, allowed in enumerate(mask) if not allowed)
        teacher = ids.index(row['teacher_action'])
        assert row['teacher_action_index'] == teacher and mask[teacher]
        peak = max(logits[i] for i in legal)
        total = sum(math.exp(logits[i] - peak) for i in legal)
        probs = {i: math.exp(logits[i] - peak)/total for i in legal}
        chosen = max(legal, key=lambda i: logits[i])
        assert row['greedy_action'] == ids[chosen] and row['greedy_mode'] == modes[chosen]
        assert row['correct'] == (chosen == teacher)
        close(row['teacher_probability'], probs[teacher]); close(row['STOP_probability'], probs[0])
        close(row['cross_entropy'], peak + math.log(total) - logits[teacher])
        close(row['entropy'], -sum(p*math.log(p) for p in probs.values() if p > 0))
        movement = sorted([i for i in legal if modes[i] != 'stop'], key=lambda i: -logits[i])
        assert row['teacher_rank_among_legal_movements'] == (None if teacher == 0 else movement.index(teacher)+1)
        if movement: close(row['STOP_minus_best_movement_margin'], logits[0]-logits[movement[0]])
        else: assert row['STOP_minus_best_movement_margin'] is None
        assert row['movement_inventory_counts'] == {mode:sum(mask[i] and modes[i] == mode for i in range(len(ids))) for mode in ('aspirate','probe')}
    groups = {'all40':rows, 'root24':[r for r in rows if r['step']==0], 'second16':[r for r in rows if r['step']==1], 'STOP8':[r for r in rows if r['teacher_action']=='STOP'], 'movement32':[r for r in rows if r['teacher_action']!='STOP'], 'aspiration_root16':[r for r in rows if r['step']==0 and r['teacher_action']!='STOP']}
    summaries[label] = {}
    for key, group in groups.items():
        calc = {'n':len(group), 'mean_cross_entropy':statistics.mean(r['cross_entropy'] for r in group), 'accuracy':statistics.mean(r['correct'] for r in group), 'greedy_STOP':sum(r['greedy_mode']=='stop' for r in group), 'teacher_movement_rank1':sum(r['teacher_rank_among_legal_movements']==1 for r in group), 'mean_STOP_probability':statistics.mean(r['STOP_probability'] for r in group)}
        for k, v in calc.items(): close(v, result[label][key][k])
        summaries[label][key] = {**calc, 'correct_count':sum(r['correct'] for r in group), 'greedy_modes':{m:sum(r['greedy_mode']==m for r in group) for m in ('stop','aspirate','probe')}}
for before, after in zip(readouts['before'], readouts['after']):
    assert all(before[k] == after[k] for k in fixed_keys)
assert abs(updates[0]['loss']['loss'] - summaries['before']['all40']['mean_cross_entropy']) < 2e-7  # float32 mean

expected_counts = {'optimizer_updates':32,'native_teacher_steps':40,'geometry_previews':760,'loss_forward_calls':1280,'readout_forward_calls':80,'new_checkpoint_loads':1,'prior_checkpoint_loads':0,'search_calls':0,'patient_reads':0,'SELECT_or_MEASUREMENT_reads':0}
assert all(result[k] == v for k,v in expected_counts.items())
assert sum(u['loss']['loss_forward_calls'] for u in updates) == 1280
costs = result['costs']; reconstruction = costs['refit.reconstruct40_native_teacher_states']
for key, value in {'geometry_preview_calls':760,'geometry_preview_returned_calls':760,'native_transition_calls':40,'native_transition_returned_calls':40,'native_nonstop_commit_calls':32,'native_nonstop_commit_returned_calls':32,'geometry_audit_calls':24,'geometry_audit_returned_calls':24,'task_clone_calls':24,'task_clone_returned_calls':24}.items(): assert reconstruction[key] == value
for scope, count in [('before.readout40',40),('loss40',1280),('after.readout40',40)]:
    assert costs['refit.'+scope]['actor_forward_calls'] == costs['refit.'+scope]['actor_forward_returned_calls'] == count
assert not result['clinical_validation'] and not result['greedy_native_rollout_executed']
assert result['wall_seconds'] < receipt['elapsed_seconds']
expected_names = {'result.json','experiment-freeze.json','refit-initial.gmckpt','refit-IL-final.gmckpt'} | {f'{label}-state-{i:02d}.json' for label in ('before','after') for i in range(40)} | {f'update-{i:02d}.json' for i in range(1,33)} | {f'teacher-reconstruction-{i:02d}.json' for i in range(24)}
assert {p.name for p in BASE.iterdir()} == expected_names
assert {p.name for p in SUP.iterdir()} == {'declaration.json','receipt.json','worker.log','progress.json'}
root_movement = [r for r in readouts['after'] if r['step']==0 and r['teacher_action']!='STOP']
audit = {'status':'PASS_SAVED_ONLY','summaries':summaries,'counts':expected_counts,'all32_parameter_transitions_changed_and_chained':True,'full40_manifest_hash':digest(corpus),'initial_parameter_hash':initial['parameter_hash'],'final_parameter_hash':final['parameter_hash'],'source_index_sha256':declaration['source_index_sha256'],'run_head':declaration['head'],'experiment_hash':result['experiment_hash'],'root_movement_failures':{'STOP':sum(r['greedy_mode']=='stop' for r in root_movement),'different_movement':sum(r['greedy_mode']!='stop' and not r['correct'] for r in root_movement)},'parent_wall_seconds':receipt['elapsed_seconds'],'worker_wall_seconds':result['wall_seconds'],'sampled_peak_rss_bytes':receipt['sampled_peak_rss_bytes'],'gradient_norm_range':[min(u['update_receipt']['gradient_norm_before_clip'] for u in updates),max(u['update_receipt']['gradient_norm_before_clip'] for u in updates)],'clipping_triggered_updates':0,'read_scope':'Saved run/supervision metadata, frozen prospective identity/source bytes, and two new opaque checkpoint hashes only. No checkpoint decoding or project/ML imports. No old payload or held-out task reads.','on_policy_rollout_evidence':False,'audit_elapsed_seconds':time.monotonic()-START}
(OUT/'audit.json').write_text(json.dumps(audit,indent=2,sort_keys=True)+'\n')
(OUT/'source-and-output-hashes.json').write_text(json.dumps(seen,indent=2,sort_keys=True)+'\n')
print(json.dumps({'status':audit['status'],'before':summaries['before'],'after':summaries['after'],'audited_files':len(seen),'elapsed_seconds':audit['audit_elapsed_seconds']},indent=2))
