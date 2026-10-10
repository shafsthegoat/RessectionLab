"""Saved JSON/source-only audit. Does not import science or open array files."""
import hashlib
import json
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
BASE = ROOT / 'build/lazy-copy-train025-pair-v2'
OUT = BASE / 'attempt-02'
PINS = {}
CHECKS = 0

def require(condition, message):
    global CHECKS
    CHECKS += 1
    if not condition:
        raise AssertionError(message)

def raw(path):
    path = Path(path)
    require(path.suffix in ('.json', '.py', '.log') and not path.is_symlink()
            and path.is_file() and path.stat().st_size <= 8*1024**2, str(path))
    value = path.read_bytes()
    PINS[str(path.relative_to(ROOT))] = hashlib.sha256(value).hexdigest()
    return value

def read(path):
    return json.loads(raw(path))

def digest(path):
    return hashlib.sha256(raw(path)).hexdigest()

def semantic(value):
    value = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return 'sha256:' + hashlib.sha256(value.encode()).hexdigest()

def normalized(value, kind):
    value = json.loads(json.dumps(value))
    if kind == 'search':
        del value['accounting']['planning_seconds']
    if kind == 'replay':
        del value['independent_geometry']['evaluation_seconds']
    return value

def main():
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError()))
    signal.alarm(30)
    started = time.monotonic()
    release = read(BASE/'root-release.json')
    release_sha = digest(BASE/'root-release.json')
    require(release_sha == '3de4f772674c3371de2484569f864bb499f284f4872eb501965c5952c55fcd0f', 'release')
    idx = read(ROOT/release['source_index']['path'])
    require(digest(ROOT/release['source_index']['path']) == release['source_index']['sha256'], 'index')
    require(idx['head'] == release['expected_head'] == 'd583008cb1ebc36b4ed34b5f722f3b4f8cc959ec', 'executed HEAD')
    archived_source = []
    for section in ('source_files', 'metadata_files'):
        for name, expected in idx[section].items():
            observed = digest(ROOT/name)
            if observed != expected and name.startswith('src/'):
                value = subprocess.check_output(['git','show',idx['head']+':'+name], cwd=ROOT, timeout=5)
                observed = hashlib.sha256(value).hexdigest()
                archived_source.append(name)
            require(observed == expected, 'bound file: '+name)
    for ref in release['correction']['preserved_v1'].values():
        require(digest(ROOT/ref['path']) == ref['sha256'], 'preserved first failure')
    pair = read(OUT/'pair-result.json')
    require(pair['status'] == 'complete_exact_parity' and pair['release_sha256'] == release_sha, 'pair terminal')
    proofs = []
    measures = []
    for name in ('baseline', 'lazy_copy'):
        root = OUT/name
        sup = OUT/(name+'.supervision')
        d = root/'arm-00'
        receipt = read(sup/'receipt.json')
        result = read(root/'result.json')
        worker = read(sup/'worker-final.json')
        control = read(sup/'endpoint-control.json')
        costs = read(root/'costs.json')
        require(receipt['status'] == 'complete' and receipt['exit_code'] == 0
                and receipt['worker_termination_confirmed'] and receipt['samples'] > 0
                and not receipt['cleanup_errors'] and not receipt['final_owned_pids'], name+' reaped')
        require(receipt['release_sha256'] == release_sha and receipt['source_index'] == release['source_index'], 'receipt inputs')
        for key, path in [('result_sha256',root/'result.json'),('worker_final_sha256',sup/'worker-final.json'),('endpoint_control_sha256',sup/'endpoint-control.json')]:
            require(receipt[key] == digest(path), key)
        require(worker['result_sha256'] == worker['canonical_result_sha256'] == receipt['result_sha256']
                and worker['endpoint_control_sha256'] == receipt['endpoint_control_sha256'], 'worker joins')
        require(control['historical_world_and_route_equal'] and control['zero_model_work'], 'historical endpoint')
        require(result['status'] == 'complete_paired_TRAIN_search' and result['implementation'] == name
                and result['global_resource_failure'] is None and not result['SELECT_EVAL_opened'], 'result')
        require(all(result[k] == 0 for k in ('policy_forwards','optimizer_updates','checkpoint_loads','private_reference_reads')), 'zero prohibited work')
        row = result['arms'][0]
        require(len(result['arms']) == 1 and row['subject'] == 'ReMIND-025'
                and row['status'] == 'complete' and row['source_released'], 'one source visit')
        world = read(d/'world.json')
        plan = read(d/'plan.json')
        search = read(d/'search.json')
        replay = read(d/'replay.json')
        require(semantic(plan['plan']) == plan['plan_seal'] == row['plan_seal'], 'plan seal')
        require(plan['plan']['actions'] == search['actions'] == row['actions']
                and len(row['actions']) == 4 and row['actions'][-1] == 'STOP', 'three motions plus STOP')
        require(plan['plan']['history'] == replay['metrics']['history'], 'full replay history')
        require(search['accounting']['complete'] and replay['independent_geometry']['accepted']
                and replay['independent_geometry']['geometry']['complete_tool_checked'], 'full completion')
        outcome = replay['independent_geometry']['outcomes']
        require(outcome['positive_target_source_cells_removed'] == 3 and outcome['normal_removed_mm3'] == 0
                and outcome['target_removed_mm3'] == 2.861024385655677
                and outcome['total_reward'] == 2.7583290722311435, 'frozen outcome')
        require(result['native_counts'] == costs['native_counts'] == {'search':4,'rollout':4,'replay':4}, 'transition counts')
        budget = costs['native_budget']
        copies = costs['temporary_mask_copies']
        require(budget['native_preview_entries'] == copies['preview_calls'] == 1224
                and budget['counting_reliable'] and budget['blocked_preview_attempts'] == 0 and budget['failure'] is None, 'preview accounting')
        require(copies['copy_calls'] == sum(r['calls'] for r in copies['by_mask'].values())
                and copies['copied_bytes'] == sum(r['bytes'] for r in copies['by_mask'].values())
                and set(copies['by_mask']) == {'remaining','connected_free'}, 'copy sums')
        require(all(r['bytes'] == r['calls']*2405039 for r in copies['by_mask'].values()), 'boolean mask byte count')
        require(receipt['elapsed_seconds'] < 90 and worker['wall_seconds'] < 60
                and receipt['sampled_peak_rss_bytes'] < 3*1024**3
                and receipt['output_bytes'] < 32*1024**2, 'resource limits')
        inventory_index = read(d/'search-inventories.json')
        inventories = []
        for entry in inventory_index:
            inventories.append(read(d/entry['file']))
            require(digest(d/entry['file']) == entry['sha256'], 'inventory digest')
        require(len(inventories) == 4, 'four observed states')
        states = {inv['cavity_state_hash']:inv for inv in inventories}
        for decision in search['accounting']['decisions']:
            inv = states[decision['source_state_hash']]
            require({r['action_id'] for r in inv['ledger'] if r['feasible']} ==
                    {r['action_id'] for r in decision['scores'] if r['action_id'] != 'STOP'}, 'every legal motion scored')
        identity_keys = ('source_hash','decision_model_hash','initial_observation_hash','context_hash','common_public_world','normalization','derived_occupancy','supplied_goal_extent','proposal_config','proposal_rule_hash')
        profile = {k:{x:y for x,y in v.items() if x != 'seconds'} for k,v in budget['native_preview_profile']['phases'].items()}
        proofs.append({'identity':{k:world[k] for k in identity_keys}, 'initial_inventory':read(d/'initial-inventory.json'),
            'inventories':inventories, 'search':normalized(search,'search'), 'plan':plan,
            'replay':normalized(replay,'replay'), 'final_state':row['final_native_state'],
            'native_counts':result['native_counts'], 'native_previews':1224, 'preview_outcome_profile':profile})
        require(pair['arms'][name]['receipt_sha256'] == digest(sup/'receipt.json')
                and pair['arms'][name]['result_sha256'] == digest(root/'result.json')
                and pair['arms'][name]['temporary_mask_copies'] == copies, 'pair joins')
        measures.append(pair['arms'][name])
    require(proofs[0] == proofs[1], 'independent full saved parity')
    prior = ROOT/'build/paired-obstruction-opening-search-v2/attempt-02/arm-07'
    require(proofs[0]['plan']['plan']['actions'] == read(prior/'plan.json')['plan']['actions']
            and proofs[0]['plan']['plan']['history'] == read(prior/'plan.json')['plan']['history'], 'original complete route')
    require(proofs[0]['search'] == normalized(read(prior/'search.json'),'search')
            and proofs[0]['replay'] == normalized(read(prior/'replay.json'),'replay'), 'original scores and full replay')
    before, after = measures
    reduction = before['temporary_mask_copies']['copied_bytes'] - after['temporary_mask_copies']['copied_bytes']
    report = {'status':'PASS','checks':CHECKS,'input_count':len(PINS),'wall_seconds':time.monotonic()-started,
        'executed_head':idx['head'],'release_sha256':release_sha,'pair_result_sha256':digest(OUT/'pair-result.json'),
        'source_files_checked':len(idx['source_files']),'metadata_files_checked':len(idx['metadata_files']),
        'canonical_files_checked_at_executed_commit_after_promotion':archived_source,
        'exact_saved_parity_fields':list(proofs[0]),'both_children_reaped':True,'preserved_v1_failure':True,
        'previews_each':1224,'transitions_each':{'search':4,'rollout':4,'replay':4},
        'copy_bytes':[before['temporary_mask_copies']['copied_bytes'],after['temporary_mask_copies']['copied_bytes']],
        'copy_calls':[before['temporary_mask_copies']['copy_calls'],after['temporary_mask_copies']['copy_calls']],
        'copy_bytes_reduction':reduction,'copy_bytes_reduction_percent':100*reduction/before['temporary_mask_copies']['copied_bytes'],
        'greedy_seconds':[before['greedy_seconds'],after['greedy_seconds']],
        'sampled_peak_rss_bytes':[before['sampled_peak_rss_bytes'],after['sampled_peak_rss_bytes']],
        'interpretation':'One ordered pair: reduced committed-mask copy traffic, exact observed behavior; no stable speedup or training-speed conclusion.',
        'scope':'Saved JSON and source hashes only; no patient array, model, native replay or checkpoint access.', 'inputs':PINS}
    with (HERE/'audit-result.json').open('x') as stream:
        json.dump(report,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'inputs'},indent=2))

if __name__ == '__main__':
    main()
