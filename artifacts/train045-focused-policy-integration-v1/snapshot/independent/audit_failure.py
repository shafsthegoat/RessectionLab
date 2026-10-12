"""One bounded saved-JSON audit of the failed first focused-policy attempt.

No scientific imports, native execution, arrays, models or checkpoint decoding.
The incomplete planning prefix is checked as evidence, never as an endpoint.
"""
import argparse
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import signal
import time

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p/'src/resectionlab').is_dir())
STAGE = HERE.parent
PINS = {
    'report': 'c31b88e88c77b8f50d262193ce2424d4abf203eb23fbab14eed63de032229038',
    'launcher': '5dcad2ddc1f1ba893a1982d579598cf7dc18006cbb3703cd8d69e29f9bf10784',
    'index': '46e709594e58de75233d1cb993efd4a3ec24f510a4c903938aac6069396dfb6a',
}


def semantic(value):
    return 'sha256:' + hashlib.sha256(json.dumps(value, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


class Audit:
    def __init__(self):
        self.checks = 0
        self.files = {}
    def need(self, condition, label):
        self.checks += 1
        if not condition:
            raise AssertionError('check %d: %s' % (self.checks, label))
    def close(self, a, b, label):
        self.need(type(a) in (int, float) and type(b) in (int, float)
            and math.isfinite(a) and math.isfinite(b)
            and math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-9), label)
    def raw(self, path, digest=None, limit=32*1024**2):
        path = Path(path)
        self.need(path.is_file() and not path.is_symlink() and path.stat().st_size <= limit,
            'Regular bounded saved file: ' + path.name)
        raw = path.read_bytes()
        actual = hashlib.sha256(raw).hexdigest()
        if digest is not None:
            self.need(actual == digest, 'Exact bytes: ' + path.name)
        self.files[str(path.relative_to(ROOT))] = {'sha256': actual, 'bytes': len(raw)}
        return raw
    def read(self, path, digest=None):
        return json.loads(self.raw(path, digest), parse_constant=lambda x:
            (_ for _ in ()).throw(ValueError('Nonfinite JSON: ' + x)))
    def compressed(self, directory, reference):
        self.need(Path(reference['path']).name == reference['path'], 'Local artifact path')
        raw = self.raw(directory/reference['path'], reference['sha256'], 3*1024**2)
        self.need(len(raw) == reference['bytes'] and reference['decompressed_limit'] == 32*1024**2,
            'Declared compressed and decoded limits')
        with gzip.GzipFile(fileobj=io.BytesIO(raw), mode='rb') as stream:
            decoded = stream.read(32*1024**2 + 1)
        self.need(len(decoded) <= 32*1024**2 and len(decoded) == reference['decompressed_bytes']
            and hashlib.sha256(decoded).hexdigest() == reference['decompressed_sha256'],
            'Exact complete lossless decoded evidence')
        return json.loads(decoded)


def inspect(a):
    directory = STAGE/'run-01'
    index = a.read(STAGE/'runner/source-index.json', PINS['index'])
    report = a.read(directory/'report.json', PINS['report'])
    parent = a.read(directory/'launcher.json', PINS['launcher'])
    for name, digest in index['files'].items():
        # Source files only. Checkpoint and public-array sections are metadata.
        a.raw(ROOT/name, digest)
    a.need(index['execution_released'] is True and report['source_index_sha256'] ==
        parent['source_index_sha256'] == PINS['index'], 'Released unchanged source identity')
    a.need(parent['report_sha256'] == PINS['report'] and report['source_unchanged'] is True,
        'Terminal report and source binding')
    life = parent['owned_lifecycle']
    a.need(parent['status'] == 'incomplete_or_failed' and report['status'] == 'failed_or_incomplete'
        and parent['worker_exit_code'] == life['exit_code'] == 1, 'Original failed statuses retained')
    a.need(life['verified'] is True and life['child_reaped'] is True
        and life['remaining_owned_pids'] == [] and life['errors'] == []
        and life['remaining_state_unknown'] is False and parent['stop_reason'] is None,
        'Worker failure with verified clean process cleanup')
    a.need(report['failure']['phase'] == 'GREEDY_suffix_planning'
        and report['failure']['type'] == 'TimeoutError'
        and report['failure']['message'] == 'Focused rollout time cap', 'Actual planner deadline')
    a.need(report['branches'] == {} and report['selections'] == {}
        and report['plans_sealed_before_replay'] is False
        and not (directory/'plans.json').exists(), 'No complete plans, endpoint outcomes or accepted comparison')
    a.need(set(report['incomplete_selection']) == {'GREEDY'}, 'Only first arm attempted')
    a.need(report['closed_roles_opened'] is False and report['cache_loads'] ==
        report['head_optimizer_updates'] == report['total_frozen_policy_forwards'] == 0,
        'No fitting, policy forward, cache load or closed role')
    a.need(report['checkpoint_loads'] == report['checkpoint_loads_attempted'] == 2,
        'Two originally pinned model artifacts loaded')
    a.need(report['model_provenance']['checkpoints'] == index['checkpoints'], 'Exact checkpoint metadata')
    a.need(report['baseline_prefix_reproduced'] is True
        and report['frozen_models_unchanged'] is False, 'Prefix complete; overall acceptance flag remains false')
    partial = a.compressed(directory, report['incomplete_selection']['GREEDY'])
    a.need(partial['scoring'] is None, 'No hidden scoring attempt')
    policy = partial['policy']
    a.need(policy['method'] == 'GREEDY' and policy['complete'] is False
        and policy['training_updates'] == 0 and policy['initial_focus'] is None
        and policy['positive_immediate_guard'] is True, 'Honest incomplete fixed-method record')
    history = policy['explored_history']
    packets = policy['explored_action_records']
    prefix = a.read(directory/'fixed-prefix.json')
    frontier = report['fixed_frontier']
    a.need(frontier == prefix['focused_frontier'] and frontier['steps'] == 12
        and frontier['seal'] == semantic({k:v for k,v in frontier.items() if k != 'seal'}),
        'Authenticated common twelve-step frontier')
    a.need(len(prefix['joins']) == len(prefix['focused_packets']) == 12
        and packets[:12] == prefix['focused_packets'], 'Actual prefix packets retained intact')
    a.need(len(history) == len(packets) == 23 and len(policy['states']) == 11
        and policy['model_transition_calls'] == policy['model_transition_completed'] == 11,
        'Eleven committed suffix motions; incomplete absolute24 horizon')
    a.need(all(p['action_id'] != 'STOP' for p in packets), 'Incomplete prefix not replaced by STOP')
    a.need(semantic(history[:12]) == frontier['nominal_history_hash'], 'Full fixed prefix history hash')
    a.close(sum(row['reward'] for row in history[:12]), frontier['total_reward'], 'Prefix reward arithmetic')
    a.close(policy['fixed_prefix_return'], frontier['total_reward'], 'Incumbent prefix scalar')
    expected_tool = None
    for i, (packet, row) in enumerate(zip(packets, history)):
        a.need(packet['steps_taken'] == i and packet['current_tool'] == expected_tool
            and packet['action_id'] == row['action_id'] and packet['source_hash'] == frontier['source_hash']
            and packet['decision_model_hash'] == frontier['decision_model_hash'], 'Actual packet step/tool/source')
        a.need(packet['seal'] == semantic({k:v for k,v in packet.items() if k != 'seal'}), 'Packet semantic seal')
        a.need(packet['physical_action'] == {k:row[k] for k in ('tool_id','entry_mm','tip_mm')}
            and row['outcome_scope'] == 'permitted_nominal_model', 'Public nominal physical action')
        expected_tool = row['tool_id']
    for j, state in enumerate(policy['states']):
        i = 12 + j
        candidates = state['candidates']
        ids = [r['action_id'] for r in candidates]
        a.need(ids[0] == 'STOP' and len(ids) == len(set(ids)) <= 153
            and state['steps'] == i and state['complete_page_enumeration'] is True,
            'Current complete bounded unique actor catalog')
        pages = state['pages']
        a.need([p['page_index'] for p in pages] == list(range(len(pages)))
            and 1 <= len(pages) <= 5, 'All ordered bounded pages')
        emitted = pages[0]['additional_emitted_count']
        a.need(emitted <= 32 and len(pages) == 1+(emitted+7)//8
            and pages[0]['base_emitted_count'] <= 120, 'Original allocation caps')
        ordered = ['STOP']; owners = [0]
        for n, page in enumerate(pages):
            a.need(page['source_state_hash'] == state['source_state_hash']
                and page['steps_taken'] == i and page['source_hash'] == frontier['source_hash'],
                'Every page is from the same actual current state')
            ordered.extend(page['page_action_ids']); owners.extend([n]*len(page['page_action_ids']))
        a.need(ids == ordered and [r['owner_page'] for r in candidates] == owners,
            'Base once, extra ownership and stable ordering')
        positive = []
        for k, candidate in enumerate(candidates):
            r = candidate['immediate_reward']
            a.need(type(r) in (int,float) and math.isfinite(r)
                and candidate['raw_tail'] is None and candidate['clipped_tail'] is None
                and candidate['ranking_value'] == r and candidate['actual_positive'] is (r > 0.),
                'Unlearned exact positive immediate comparison')
            if k and r > 0.: positive.append((k,r))
        a.need(bool(positive), 'Committed positive action exists')
        winner = sorted(positive, key=lambda p:(-p[1],p[0]))[0][0]
        selected = state['selected']
        a.need(selected['action_id'] == ids[winner] and selected['packet'] == packets[i],
            'Independent scalar GREEDY winner and committed packet')
        a.close(selected['immediate_reward'], history[i]['reward'], 'Chosen reward once')
        a.close(selected['cumulative_return'], sum(r['reward'] for r in history[:i+1]), 'Full cumulative prefix reward')
        a.need(history[i]['reward'] > 0 and selected['packet']['query_catalog_digest'] == state['query_catalog_digest'],
            'Positive committed action in exact catalog')
        matches = [r for r in pages[owners[winner]]['ledger']
            if r.get('action_id') == ids[winner] and r.get('verification') == 'NATIVE_VERIFIED']
        a.need(bool(matches) and all(r['feasible'] is True and
            {k:r[k] for k in ('tool_id','entry_mm','tip_mm')} == selected['packet']['physical_action']
            for r in matches), 'Chosen action belongs to the verified owning page')
    sidecars = a.compressed(directory, report['sidecars'])
    a.need(len(sidecars['records']) == report['sidecars']['unique_count'] == 1051, 'All retained sidecars')
    for fingerprint, row in sidecars['records'].items():
        a.need(fingerprint == row['fingerprint'] == semantic({k:v for k,v in row.items() if k != 'fingerprint'}),
            'Authenticated complete failure sidecar')
    for state in policy['states']:
        for goal in state['goals']:
            row = goal['sidecar']
            if row is not None:
                a.need(sidecars['records'][row['fingerprint']] == row, 'Current goal sidecar retained losslessly')
    costs = report['costs']; budget = report['native_budget']; caps = report['config']['caps']
    a.need(budget['counting_reliable'] is True and budget['status'] == 'failed'
        and budget['history_complete_caller_attestation'] is None and budget['score'] is None,
        'Reliable failed guard, no false complete-history attestation')
    leaf_phases = ('saved_metadata_admission','load_saved_frozen_IL64_and_TAIL',
        'same045_source.public_reconstruction','genuinely_execute_original_and_focused_prefix','GREEDY_suffix_planning')
    preview_total = sum(costs[p]['native_preview_entries'] for p in leaf_phases)
    a.need(preview_total == budget['native_preview_entries'] == 4733
        and costs['same045_source']['native_preview_entries'] == 4642,
        'Nested scopes separated: final4733 includes91 source previews')
    a.need(report['native_transition_counts'] == {'attempted':35,'completed':35,'raised':0}
        and report['native_transition_attempts'] == 24+11, 'All prefix and suffix transitions paid once')
    a.need(not any('TAIL_' in p or 'replay' in p for p in costs), 'No second arm or endpoint replay')
    profile = budget['native_preview_profile']['phases']['unclassified']
    a.need(profile['started'] == profile['returned'] == preview_total and profile['raised'] == 0,
        'Final native profile joins reliable count')
    progress = parent['last_saved_costs']
    a.need(progress['depth'] == 23 and progress['native_budget']['native_preview_entries'] == 4642,
        'Last progress is earlier than final accounting')
    a.close(progress['cumulative_return'], sum(r['reward'] for r in history), 'Last provisional return matches partial history')
    a.need(report['worker_seconds'] < caps['work_seconds'] and parent['elapsed_seconds'] < caps['parent_wall_seconds']
        and max(report['self_peak_rss_bytes'],parent['sampled_peak_group_rss_bytes']) <= caps['worker_peak_rss_bytes']
        and preview_total <= caps['native_previews'], 'Outer resource caps held despite inner planner cap')
    a.need(costs['GREEDY_suffix_planning']['complete_wall_seconds'] >= report['config']['planner_seconds'],
        'Observed planner deadline, not outer resource refusal')
    a.need('preserved_failed_attempt' in report['retained_capability_reference']
        and 'preserved_output_failed_attempt' in report['retained_capability_reference']
        and report['retained_prefix']['prior_costs_not_erased'], 'Historical failures and costs remain referenced')
    return {
        'scientific_comparison': 'INCOMPLETE; no learned result or independently accepted endpoint',
        'failure': report['failure'], 'partial_GREEDY': {'fixed_prefix':12,'suffix_commits':11,
            'total_steps':23,'provisional_total_return':sum(r['reward'] for r in history),
            'provisional_suffix_return':sum(r['reward'] for r in history[12:]),
            'positive_suffix_actions':11,'independent_geometry_replay':False},
        'costs': {'native_previews_final':preview_total,'last_progress_previews':4642,
            'previews_after_last_progress':preview_total-4642,'native_transitions':35,
            'policy_forwards':0,'checkpoint_loads':2,'worker_seconds':report['worker_seconds'],
            'parent_seconds':parent['elapsed_seconds'],'peak_rss_bytes':parent['sampled_peak_group_rss_bytes'],
            'native_preview_inclusive_seconds':profile['seconds'],
            'GREEDY_phase_wall_seconds':costs['GREEDY_suffix_planning']['complete_wall_seconds'],
            'leaf_phase_costs':{p:costs[p] for p in leaf_phases},
            'scope_note':'Phase/preview timers are nested; do not sum inclusive timers as independent costs.'},
        'cleanup':life, 'retained_prior_costs': report['retained_prefix']['prior_costs_not_erased'],
        'prior_capability_costs_and_failures':report['retained_capability_reference'],
        'optimization_review_basis':'Source repeatedly calls task.metrics() only to read immutable prefix history; metrics performs outcome reductions over the full source. Removing those unrelated reductions can preserve authority/history seals. Saved timers do not isolate that overhead or establish a measured speedup.',
        'limits':'Saved packet/choice/hash/accounting checks only. No new geometry validation, model inference, array/checkpoint/cache decoding. The failed run is not relabeled successful.'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args()
    output=Path(args.output).resolve()
    if output.parent != HERE or output.exists():raise ValueError('Fresh independent output required')
    started=time.monotonic();a=Audit();receipt={'schema':'focused-policy-failed-attempt-independent-v1','pins':PINS}
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('30s saved audit')))
    signal.alarm(30)
    code=0
    try:receipt.update(status='PASS_failure_evidence',findings=inspect(a))
    except BaseException as error:
        code=1;receipt.update(status='FAIL_reader_or_evidence',failure={'type':type(error).__name__,'message':str(error)})
    finally:
        signal.alarm(0);receipt.update(checks=a.checks,elapsed_seconds=time.monotonic()-started,
            auditor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            saved_inputs=a.files,native_calls=0,model_calls=0,patient_payload_reads=0)
        raw=json.dumps(receipt,sort_keys=True,indent=2,allow_nan=False).encode()+b'\n'
        if len(raw)>65536:raise ValueError('64KiB receipt cap')
        with output.open('xb') as stream:stream.write(raw)
        print(json.dumps({'status':receipt['status'],'checks':a.checks,'bytes':len(raw),
            'sha256':hashlib.sha256(raw).hexdigest(),'elapsed_seconds':receipt['elapsed_seconds']}))
    return code


if __name__=='__main__':raise SystemExit(main())
