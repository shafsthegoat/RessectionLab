"""Independent saved JSON arithmetic/binding audit; no scientific imports or source arrays."""
import hashlib
import json
import math
from pathlib import Path
import struct

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'build/union025-clearance-witness-v1'
OUT = BASE / 'attempt-01'
SUP = BASE / 'attempt-01.supervision'
PINS = {
    'root-release.json': '12f2b1dac613db9e577f23542828f0224ba4080b0362851565022a60db18094f',
    'source-index.json': '161116cc522d53dcf37c4cbb48813422e0c19bd1019f4c573891e02a228fcc67',
}
reads = {}

def raw(path):
    path = Path(path)
    assert not path.is_symlink() and path.is_file() and path.stat().st_size <= 8*1024**2, path
    data = path.read_bytes()
    reads[str(path.relative_to(ROOT))] = hashlib.sha256(data).hexdigest()
    return data

def load(path):
    return json.loads(raw(path))

def sha(path):
    return hashlib.sha256(raw(path)).hexdigest()

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)

def semantic(value):
    return 'sha256:' + hashlib.sha256(canonical(value).encode()).hexdigest()

def physical(row):
    return {key: row[key] for key in ('tool_id', 'entry_mm', 'tip_mm')}

def close(a, b):
    assert math.isfinite(a) and math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9), (a, b)

def cell_bytes(cells):
    assert len({tuple(c) for c in cells}) == len(cells)
    assert all(len(c) == 3 and all(type(x) is int for x in c) for c in cells)
    return b''.join(struct.pack('<qqq', *cell) for cell in cells)

def array_hash(cells):
    header = json.dumps({'shape': [len(cells), 3], 'dtype': '<i8'}, sort_keys=True).encode()
    return 'sha256:' + hashlib.sha256(header + cell_bytes(cells)).hexdigest()

def native_hash(cells):
    return hashlib.sha256(str((len(cells), 3)).encode() + b'int64' + cell_bytes(cells)).hexdigest()

def main():
    for name, expected in PINS.items():
        assert sha(BASE/name) == expected
    release, index = load(BASE/'root-release.json'), load(BASE/'source-index.json')
    assert release['expected_head'] == index['head'] == '177c3861cfdc48dc3ee33f966098a288cda858b7'
    for section in ('source_files', 'metadata_files'):
        for name, expected in index[section].items():
            assert sha(ROOT/name) == expected, name
    refs = load(BASE/'input-index.json')['files']
    prior = load(ROOT/refs['union_world']['path'])
    old_inventory = load(ROOT/refs['union_inventory']['path'])
    world, context = load(OUT/'world.json'), load(OUT/'context.json')
    inventory = load(OUT/'initial-inventory.json')
    receipt, final = load(SUP/'receipt.json'), load(SUP/'worker-final.json')
    result, costs = load(OUT/'result.json'), load(OUT/'costs.json')
    control = load(SUP/'endpoint-control.json')
    assert receipt['status'] == 'complete' and receipt['exit_code'] == 0
    assert receipt['worker_termination_confirmed'] and not receipt['final_owned_pids']
    assert not receipt['cleanup_errors'] and receipt['stop_reason'] is None
    assert receipt['result_sha256'] == final['result_sha256'] == final['canonical_result_sha256'] == sha(OUT/'result.json')
    assert receipt['worker_final_sha256'] == sha(SUP/'worker-final.json')
    assert receipt['endpoint_control_sha256'] == final['endpoint_control_sha256'] == sha(SUP/'endpoint-control.json')
    assert receipt['release_sha256'] == PINS['root-release.json']
    assert receipt['elapsed_seconds'] < 210 and final['wall_seconds'] < 180
    assert receipt['samples'] > 0 and 0 < receipt['sampled_peak_rss_bytes'] < 3*1024**3
    assert receipt['output_bytes'] == sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file()) < 64*1024**2
    assert sum(p.stat().st_size for p in SUP.rglob('*') if p.is_file()) < 16*1024**2
    assert final['status'] == 'complete_owned_union025_fixed_ray' and final['runtime']['threads'] == 1
    assert result['status'] == 'complete_fixed_ray_witness' and result['source_released']
    assert result['SELECT_EVAL_opened'] is False and context['private_reference_in_task'] is False
    assert context['subject'] == 'ReMIND-025' and context['role'] == 'TRAIN'
    assert context['max_optimizer_updates'] == context['budgets']['max_policy_forwards'] == 0
    assert context['derived_occupancy_anatomically_validated'] is False
    assert semantic(context) == world['context_hash']
    for key in ('source_hash', 'decision_model_hash', 'initial_observation_hash', 'derived_occupancy', 'normalization', 'supplied_goal_extent', 'common_public_world'):
        assert world[key] == prior[key], key
    assert world['common_public_world_sha256'] == semantic(world['common_public_world'])
    assert inventory == old_inventory and inventory['omitted_count'] == 0
    legal = [r for r in inventory['emitted'] if r['feasible']]
    assert legal == release['root_preparations'] and len(legal) == 15
    assert len({r['action_id'] for r in legal}) == 15
    ray = release['fixed_ray']; weights = world['common_public_world']['reward']
    assert result['fixed_ray'] == ray
    assert len(result['branches']) == len(control['branches']) == 16
    summaries = []
    for i, row in enumerate(result['branches']):
        dest = OUT/f'branch-{i:02d}'
        plan, replay = load(dest/'plan.json'), load(dest/'replay.json')
        diagnostic = load(dest/'fixed-ray-preview.json')
        assert row == load(dest/'branch-result.json') and row['status'] == 'complete' and row['ordinal'] == i
        body, metrics, audit = plan['plan'], replay['metrics'], replay['independent_geometry']
        hist, actions = body['history'], body['actions']
        assert row['plan_seal'] == plan['plan_seal'] == semantic(body)
        assert row['actions'] == actions == [r['action_id'] for r in hist]
        assert hist == metrics['history'] and metrics['terminated'] and audit['accepted']
        assert audit['complete_episode'] and audit['committed_history_hash'] == semantic(hist)
        assert audit['geometry']['feasible'] and audit['geometry']['complete_tool_checked'] and audit['geometry']['frontier_checked']
        assert not audit['geometry']['failures'] and audit['geometry']['unsupported_source_tissue_volume_mm3'] == 0
        assert row['outcomes'] == audit['outcomes']
        for key in ('source_hash', 'decision_model_hash'):
            assert body[key] == metrics[key] == audit[key] == world[key]
        assert body['context_hash'] == world['context_hash'] and body['initial_observation_hash'] == world['initial_observation_hash']
        assert body['learning_updates'] == 0 and body['parameter_hash'] is None
        assert actions == (['STOP'] if i == 0 else [legal[i-1]['action_id'], 'STOP'])
        assert row['native_counts'] == {'rollout':len(actions), 'replay':len(actions)}
        assert physical(diagnostic) == ray and diagnostic['committed'] is False
        assert diagnostic['temporary_removals_committed'] is False and not diagnostic['feasible']
        assert diagnostic['reason'] == 'SHAFT_BLOCKED_BY_REMAINING_NATIVE_TISSUE'
        assert row['ray_status'] == ('baseline_blocked' if i == 0 else 'blocked')
        if i:
            assert physical(hist[0]) == physical(legal[i-1])
            successor = load(dest/'successor-inventory.json')
            matches = [r for r in successor['emitted'] if physical(r) == ray]
            assert matches == row['successor_exact_ray_matches'] and len(matches) == 1
            assert all(not r['feasible'] and r['reason'] == diagnostic['reason'] for r in matches)
            assert successor['cavity_state_hash'] == diagnostic['source_state_hash']
        else:
            assert diagnostic['source_state_hash'] == world['initial_state_hash']
        d = diagnostic['obstruction_diagnostic']; cells = d['blocked_indices_native']
        assert d['fingerprint'] == semantic({k:v for k,v in d.items() if k != 'fingerprint'})
        assert d['source_hash'] == diagnostic['source_hash'] == world['source_hash']
        assert d['source_state_hash'] == diagnostic['source_state_hash']
        assert d['decision_model_hash'] == diagnostic['native_model_hash']
        assert d['native_affine_hash'] == world['common_public_world']['native_frame_hash']
        assert d['requested_tip_mm'] == ray['tip_mm'] and d['entry_mm'] == ray['entry_mm']
        assert d['failure_tip_mm'] == diagnostic['failure_tip_mm']
        assert d['cell_limit'] == 4096 and d['retained_cell_count'] == len(cells) <= 4096
        assert d['complete_first_failure_set'] == (d['blocked_cell_count'] == len(cells))
        assert d['truncated'] == (d['blocked_cell_count'] > len(cells))
        assert d['complete_first_failure_set'] and array_hash(cells) == d['blocked_indices_hash']
        steps = diagnostic['prior_completed_microsteps']
        removed = [s['removed_indices_native'] for s in steps]
        assert d['prior_temporary_removed_count'] == diagnostic['temporary_removed_count'] == sum(map(len, removed))
        assert d['prior_temporary_removals_hash'] == diagnostic['temporary_removed_hash'] == semantic([native_hash(c) for c in removed])
        assert d['prior_temporary_removals_committed'] is False
        assert d['failure_interval_index'] == len(steps)
        fraction = d['failure_interval_index'] / (d['planned_microsteps'] - 1)
        for a, b, actual in zip(ray['entry_mm'], ray['tip_mm'], d['failure_tip_mm']): close(actual, a+(b-a)*fraction)
        temporary = {tuple(c) for step in removed for c in step}
        committed = {tuple(c) for h in hist if h['action_id'] != 'STOP' for c in h['removed_indices_native']}
        assert not set(map(tuple, cells)) & (temporary | committed)
        classification = diagnostic['blocker_occupancy']
        assert classification['retained_count'] == len(cells) == classification['in_raw_S'] + classification['in_added_T_minus_S']
        rewards=[]; tool=None; changes=0; distance_total=0
        for j, h in enumerate(hist):
            assert load(dest/f'returned-{j:02d}.json') == h
            assert load(dest/f'attempt-{j:02d}.json')['action_id'] == h['action_id']
            assert load(dest/f'replay-attempt-{j:02d}.json')['action_id'] == h['action_id']
            if h['action_id'] == 'STOP':
                assert h['reward'] == 0
                continue
            length=math.dist(h['entry_mm'], h['tip_mm']); changed=int(tool is not None and tool != h['tool_id'])
            r=weights['target_per_mm3']*h['target_removed_mm3']-weights['normal_per_mm3']*h['normal_removed_mm3']-weights['action_cost']-2*weights['motion_per_mm']*length-weights['tool_change_cost']*changed
            close(r,h['reward']); close(2*length,h['complete_tool_path_length_mm'])
            rewards.append(r); distance_total+=2*length; changes+=changed; tool=h['tool_id']
        close(math.fsum(rewards),row['outcomes']['total_reward'])
        close(distance_total,row['outcomes']['complete_tool_path_length_mm'])
        assert changes == row['outcomes']['tool_changes']
        assert row['outcomes']['total_reference_target_mm3'] == world['common_public_world']['full_target_mm3']
        for kind in ('target_removed_mm3','normal_removed_mm3'):
            close(math.fsum(h[kind] for h in hist),row['outcomes'][kind])
        assert row['outcomes']['target_removed_mm3'] == 0 and not row['target_access_success']
        entry = control['branches'][i]
        assert entry['ordinal'] == i and entry['plan_seal'] == plan['plan_seal']
        for name, field in [('plan.json','plan_sha256'),('replay.json','replay_sha256'),('fixed-ray-preview.json','diagnostic_sha256')]:
            assert entry[field] == sha(dest/name)
        summaries.append({'ordinal':i,'preparation_action':row['preparation_action'],'return':row['outcomes']['total_reward'],
            'normal_removed_mm3':row['outcomes']['normal_removed_mm3'],'ray_status':row['ray_status'],
            'blocked_cells':cells,'failure_interval_index':d['failure_interval_index'],'temporary_removed_cells':diagnostic['temporary_removed_count'],
            'saved_blocker_classification':classification,'plan_seal':plan['plan_seal']})
    assert costs['native_counts'] == result['native_counts'] == {'rollout':31,'replay':31}
    assert costs['diagnostic_previews'] == 16 and costs['source_visits_attempted'] == 1
    assert all(result[k] == 0 for k in ('policy_forwards','optimizer_updates','checkpoint_loads','search_calls','private_reference_reads'))
    assert all(v == 0 for v in costs['prohibited_calls'].values())
    budget=costs['native_budget']; profile=budget['native_preview_profile']['phases']['unclassified']
    assert budget['failure'] is None and budget['counting_reliable'] and budget['blocked_preview_attempts'] == 0
    assert budget['history_complete_caller_attestation'] and budget['native_preview_entries'] == profile['started'] == profile['returned'] == 3252
    assert profile['raised'] == 0 and profile['feasible']+profile['rejected'] == 3252
    branch_previews=sum(r['native_previews'] for r in result['branches'])
    assert branch_previews <= 3252
    assert result['selected_ordinal'] == control['selected_ordinal'] == max(range(16),key=lambda i:summaries[i]['return']) == 0
    assert result['selected_return'] == control['selected_return'] == 0
    assert all(r['return'] < 0 for r in summaries[1:])
    pins=dict(reads)
    assert all(sha(ROOT/p)==s for p,s in pins.items())
    report={'decision':'PASS_SAVED_FIXED_RAY_WITNESS','release_sha256':PINS['root-release.json'],
        'result_sha256':receipt['result_sha256'],'source_files_verified':len(index['source_files']),
        'metadata_files_verified':len(index['metadata_files']),'saved_file_sha256':pins,
        'outcome':'All15 legal preparations remain shaft-blocked on the exact fixed ray; all finish prep+STOP, remove zero target, and have negative complete return. STOP0 wins.',
        'complete_histories':16,'independent_geometry_receipts_validated':16,'diagnostic_sets_truncated':0,
        'native_preview_entries':3252,'branch_preview_entries':branch_previews,'source_preparation_preview_entries':3252-branch_previews,
        'native_counts':costs['native_counts'],'parent_seconds':receipt['elapsed_seconds'],
        'sampled_peak_rss_bytes':receipt['sampled_peak_rss_bytes'],'output_bytes':receipt['output_bytes'],
        'clean_reap':True,'runtime':final['runtime'],'branches':summaries,
        'limits':['Saved-JSON audit validates prior full-tool geometry receipts; geometry was not recomputed here.',
          'Blocker S/T classification is bound worker evidence, not independently reopened annotation data.',
          'First rejected interval evidence is not exact contact time, a minimum clearing set, or the full corridor.',
          'Negative applies only to one fixed ray after each of15 existing root preparations; longer preparation and other paths remain untested.',
          'S union T remains an unvalidated material assumption; no clinical/anatomical correctness claim.',
          'Sampled RSS may miss transient peaks. No model, native, array, or patient reconstruction in this review.']}
    report['audit_script_sha256']=sha(Path(__file__))
    target=Path(__file__).with_name('audit.json')
    with target.open('x') as f: json.dump(report,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({k:report[k] for k in ('decision','outcome','native_preview_entries','branch_preview_entries','source_preparation_preview_entries','parent_seconds','sampled_peak_rss_bytes')}))

if __name__ == '__main__': main()
