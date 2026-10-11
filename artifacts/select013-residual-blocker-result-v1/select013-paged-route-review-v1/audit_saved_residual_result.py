"""One saved-only stdlib audit. No native imports, arrays, or replay."""
import collections
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'build/select013-residual-blocker-v1'
P = RUN / 'attempt-01'
OUT = Path(__file__).resolve().parent
pins = {}

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read(path, expected=None):
    path = Path(path)
    if not path.is_absolute():
        path = P / path
    digest = sha(path)
    if expected:
        assert digest == expected, str(path)
    pins[str(path.relative_to(ROOT))] = digest
    return json.loads(path.read_text())

def digest(value):
    return 'sha256:' + hashlib.sha256(json.dumps(value, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()

def cells(rows):
    return set(map(tuple, rows))

def objects(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from objects(child)
    elif isinstance(value, list):
        for child in value:
            yield from objects(child)

r = read('result.json', '532581e440afdf656c8a3f24ed730e30beba45470588f9e5a337932e97ffb12c')
receipt = read(RUN/'attempt-01.supervision/receipt.json', 'cd744522216e3e6bf27fed44c1c5be67c225a4cd0f8d5e8aa4ad477dfbab213d')
release = read(RUN/'root-release.json', '7c0d91b63c5922285976660ffc4d9ac2d7377ab8a8147ec39b021eb64fc80db7')
index = read(RUN/'runtime-source-index.json', '83afce441ab2fd109e7e7e83b76d6ff20450f719838bba03096a4352622cc54e')
assert release['expected_head'] == index['head'] == '42fe588907626de402731a4f588d9936437a7a00'
assert r['release_sha256'] == receipt['release_sha256'] == sha(RUN/'root-release.json')
assert receipt['result_sha256'] == sha(P/'result.json')
assert receipt['status'] == 'complete' and receipt['exit_code'] == 0
assert not receipt['final_owned_pids'] and not receipt['cleanup_errors']
assert receipt['worker_termination_confirmed'] and receipt['stop_reason'] is None
for value, limit in [('elapsed_seconds','hard_total_wall_seconds'),
                     ('sampled_peak_rss_bytes','sampled_owned_tree_rss_bytes'),
                     ('output_bytes','output_bytes')]:
    assert receipt[value] < receipt['caps'][limit]

old = read('old-authority.json')
control = read('prefix-controls.json')
history = read('original-native-history.json')
costs = read('costs.json')
world = read('original-world-control.json')
for key in ['world', 'context']:
    pin = release['prior_pins'][key]
    assert world[key] == read(ROOT/pin['path'], pin['sha256'])
assert len(history) == len(control) == len(old['prefix']) == 47
assert old['post_exposure_authority'] == {'capability_serialized': False,
    'same_canonical_capability': True, 'same_prepared_object': True}
assert 'obstruction_selection_rule' not in old['proposal_config']
pin = release['prior_pins']['h48_result']
prior = read(ROOT/pin['path'], pin['sha256'])
layers = prior['arms']['paged_obstruction_lazy_beam']['selector_accounting']['layers']
reward = outside = target = distance = 0.0
for i, c in enumerate(control):
    step = read(f'prefix/step{i:02}.json')
    assert step['action_id'] == old['prefix'][i]
    assert {k: step[k] for k in history[i]} == history[i]
    reward += step['reward']; outside += step['normal_removed_mm3']
    target += step['target_removed_mm3']; distance = max(distance,step['insertion_distance_mm'])
    progress = c['progress']
    assert c['step'] == i+1 and c['exact']
    assert progress['actions'] == old['prefix'][:i+1]
    for key, value in [('estimated_incremental_return',reward),
                       ('outside_supplied_target_removed_mm3',outside),
                       ('target_removed_mm3',target),('max_insertion_distance_mm',distance)]:
        assert math.isclose(progress[key],value,rel_tol=0,abs_tol=1e-12)
    assert any(all(row.get(k) == v for k,v in progress.items())
               for row in objects(layers[i])), 'historical progress mismatch'
assert reward == r['prefix_return'] and target == 0
pin = release['prior_pins']['frontier_page']
assert read('old-frontier/page00.json') == read(ROOT/pin['path'], pin['sha256'])
stop = read('original-prefix-STOP.json')
assert stop['action_id'] == 'STOP' and stop['reward'] == 0

cat = read('prospective/prospective-catalog.json')
selection = read('prospective/branch-selection.json')
assert cat['base_physical_prefix_exact'] and cat['old_task_authority_unchanged']
assert cat['native_source_and_model_unchanged'] and not cat['new_rule_generated_prefix']
assert cat['old_rule'] == old['proposal_config']
base = [read(f'prospective/base-ray-{i:03}.json') for i in range(117)]
extras = [read(f'prospective/extra-{i:02}.json') for i in range(22)]
blockers = set()
for row in base:
    sidecar = row['sidecar']
    if sidecar:
        assert not sidecar['truncated'] and sidecar['complete_first_failure_set']
        blockers |= cells(sidecar['blocked_indices_native'])
assert len(blockers) == 35 and blockers == cells(r['sampled_first_blockers'])
feasible = [row for row in extras if row['feasible']]
assert len(feasible) == r['feasible_extras'] == 9
for row in extras:
    removed = cells((row['history_if_feasible'] or {}).get('removed_indices_native', []))
    assert cells(row['sampled_first_blockers_removed']) == removed & blockers
    assert row['selection_score'] == len(removed & blockers)
ranked = sorted(feasible,key=lambda x:(-x['selection_score'],x['ordinal']))
assert selection['chosen_ordinals'] == [x['ordinal'] for x in ranked[:3]] == [2,6,8]
assert selection['unchosen_ordinals'] == [x['ordinal'] for x in ranked[3:]]
assert all(row['nominal_increment']['target_removed_mm3'] == 0 for row in feasible)

branch_summary=[]
for i in range(3):
    branch = read(f'prospective/branch-{i:02}/result.json')
    assert branch['full_native_history'] == history + [branch['prep']]
    assert branch['prep'] == extras[branch['extra_ordinal']]['history_if_feasible']
    assert branch['native_state']['history_hash'] == digest(branch['full_native_history'])
    assert branch['deterministic_native_commit_replay_exact']
    assert not branch['accepted_plan_claim'] and not branch['complete_new_rule_task_episode']
    assert branch['prefix_plus_prep_STOP_return'] == reward + branch['nominal_prep']['reward']
    assert branch['nominal_prep']['target_removed_mm3'] == 0
    changes=[]; newly=[]; after_counts=collections.Counter()
    for k, before in enumerate(base):
        after = read(f'prospective/branch-{i:02}/ray-{k:03}.json')
        assert after['proposal'] == before['proposal']
        assert after['before_reason'] == before['reason']
        assert after['before_feasible'] == before['feasible']
        assert after['newly_feasible'] == (after['feasible'] and not before['feasible'])
        after_counts[after['reason']] += 1
        if after['reason'] != before['reason']:
            changes.append({'ray':k,'before':before['reason'],'after':after['reason']})
        if after['newly_feasible']:
            newly.append(k)
    assert changes == branch['reason_changes']
    assert newly == [x['ray'] for x in branch['newly_feasible_original_rays']]
    assert {k:v for k,v in branch.items() if k not in ['full_native_history','prep','nominal_prep']} == r['branch_results'][i]
    branch_summary.append({'branch':i,'extra':branch['extra_ordinal'],
        'removed':branch['first_blockers_removed'],'increment':branch['nominal_prep']['reward'],
        'total':branch['prefix_plus_prep_STOP_return'],'newly_feasible':newly,
        'after_reason_counts':dict(after_counts)})

certs=[]
for i, item in enumerate(r['independent_certificates']):
    cert=read(item['path'],item['sha256']); audit=cert['audit']
    h=history if i==0 else read(f'prospective/branch-{i-1:02}/result.json')['full_native_history']
    assert cert['history_hash'] == digest(h)
    assert audit['action_count'] == len(h) and audit['feasible'] and not audit['failures']
    assert audit['complete_tool_checked'] and audit['frontier_checked']
    assert audit['unsupported_source_tissue_volume_mm3'] == 0
    certs.append({'path':item['path'],'sha256':item['sha256'],'motions':len(h),'feasible':True})

assert r['native_budget'] == costs['native_budget']
assert sum(x.get('native_preview_entries',0) for x in costs['phases'].values()) == r['native_budget']['native_preview_entries'] == 6445
assert costs['phases']['prospective_catalog_and_branches']['native_preview_entries'] == 117+22+3*(2+117) == 496
assert r['native_commits'] == r['native_commit_attempts'] == 47+2*3 == 53
assert r['STOP_calls'] == 1 and r['native_commits']+r['STOP_calls'] <= 60
assert r['native_budget']['failure'] is None and r['native_budget']['blocked_preview_attempts'] == 0
assert r['native_budget']['counting_reliable'] and r['native_budget']['time_overshoot_seconds'] == 0
assert r['wall_seconds'] < 210 and r['native_budget']['native_preview_entries'] < 10000
for k in ['checkpoint_loads','gradient_attempts','model_forwards','optimizer_attempts','updates','private_reference_reads']:
    assert r[k] == 0
for k in ['EVAL_opened','held_inputs_opened','accepted_plan_claim','new_rule_generated_prefix']:
    assert not r[k]

witness=(7,92,85)
assert all(witness not in cells((x['history_if_feasible'] or {}).get('removed_indices_native',[])) for x in base+extras)
omitted=[x for x in cat['ledger'] if x['reason']=='OBSTRUCTION_SELECTION_CAP']
assert omitted == r['omitted_pairs'] and len(omitted)==17
assert len({tuple(x['voxel']) for x in omitted})==11
aperture=cat['accounting']['static_aperture_pairs']
assert len(aperture)==70 and sum(x['excluded'] for x in aperture)==31
assert len(aperture)-sum(x['excluded'] for x in aperture)==len(extras)+len(omitted)
witness_rows=[{'ray':i,'proposal':x['proposal'],'blocked_indices_native':x['sidecar']['blocked_indices_native']}
    for i,x in enumerate(base) if x['sidecar'] and witness in cells(x['sidecar']['blocked_indices_native'])]
summary={'status':'PASS_saved_only','claim':'one local exposed-opening unlock; no target gain or accepted route',
    'original47_exact':True,'original_return':reward,'base_reason_counts':dict(collections.Counter(x['reason'] for x in base)),
    'extras_reason_counts':dict(collections.Counter(x['reason'] for x in extras)),
    'branches':branch_summary,'independent_certificates':certs,
    'costs':{'previews':6445,'native_commits':53,'STOP':1,'worker_seconds':r['wall_seconds'],
        'parent_seconds':receipt['elapsed_seconds'],'sampled_peak_rss_bytes':receipt['sampled_peak_rss_bytes'],
        'output_bytes':receipt['output_bytes'],'clean':True},
    'censoring':{'retained_blocker_cells':35,'aperture_excluded_pairs':31,'surviving_pairs':39,
        'emitted_pairs':22,'selection_omitted_pairs':17,'selection_omitted_cells':11,
        'feasible_extras_not_branched':selection['unchosen_ordinals']},
    'candidate_witness':{'voxel':list(witness),'currently_feasible_base_or_extra_removes':False,
        'omitted_pairs':[x for x in omitted if tuple(x['voxel'])==witness],
        'aperture_rows':[x for x in aperture if tuple(x['voxel'])==witness],
        'affected_rays':witness_rows},
    'pins':{k:v for k,v in pins.items() if not '/prefix/step' in k and not '/ray-' in k and not '/base-ray-' in k}}
(OUT/'residual-saved-result-audit.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
print(json.dumps({k:summary[k] for k in ['status','claim','original47_exact','base_reason_counts','extras_reason_counts','branches','costs','censoring']},indent=2))
