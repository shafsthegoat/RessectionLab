"""Fixed PAT25 saved-record report; no patient, native geometry or model imports."""
from collections import Counter
from pathlib import Path
import argparse
import hashlib
import json
import math

VERSION = 'pat25-screened-ingress-initial-comparison-v1'
PHASES = ('original', 'screened_selected')
EXITS = tuple((axis, sign) for axis in range(3) for sign in (-1, 1))
SEGMENTS = ('entry_to_tip', 'approach_shaft_centerline', 'deepest_shaft_centerline')
CHANNELS = ('structural_intensity', 'nominal_tissue', 'nominal_target',
            'observed_cavity', 'nominal_motor', 'nominal_language')


def mean(values):
    values = list(values)
    return None if not values else math.fsum(values) / len(values)


def phase_summary(record):
    result = {key: record[key] for key in ('status', 'exit', 'error', 'elapsed_seconds',
        'preview_calls', 'started_preview_calls', 'historical_inventory_exactly_reproduced',
        'reward', 'horizon', 'decision_model_hash') if key in record}
    result.update(inventory=None, actor=None, centerline_visibility=None)
    if record['status'] != 'complete':
        result['partial_returned_preview_count'] = len(record.get('partial_preview_trace', []))
        return result
    inventory = record['initial_inventory']
    if not inventory['complete'] or not inventory['ledger_complete']:
        raise ValueError('Complete phase requires its full declared inventory')
    emitted = inventory['emitted']
    tools = sorted({row['tool_id'] for row in inventory['ledger']})
    result['inventory'] = {key: value for key, value in inventory.items() if key not in ('ledger','emitted')}
    result['inventory']['per_tool'] = {}
    for tool in tools:
        rays = [row for row in emitted if row['tool_id'] == tool]
        slots = [row for row in inventory['ledger'] if row['tool_id'] == tool]
        result['inventory']['per_tool'][tool] = {
            'declared_slots': len(slots), 'emitted': len(rays),
            'accepted': sum(row['feasible'] for row in rays),
            'rejected': sum(not row['feasible'] for row in rays),
            'proposal_dispositions': dict(Counter(row['proposal_reason'] for row in slots)),
            'rejection_reasons': dict(Counter(row['reason'] for row in rays if not row['feasible']))}
    coverage, visibility = record['actor_coverage'], record['actor_visibility']
    result['actor'] = {key: value for key, value in coverage.items() if key != 'actions'}
    result['actor'].update({key: visibility[key] for key in
        ('cavity_source_cells','cavity_visible_cells','cavity_channel_available')})
    result['actor']['channel_order'] = list(CHANNELS)
    result['actor']['channel_available'] = visibility['view']['channel_available']
    result['actor']['coverage_fraction_range'] = visibility['view']['coverage_fraction_range']
    result['centerline_visibility'] = {}
    for name, rows in (('emitted', visibility['actions']),
                       ('accepted', [r for r in visibility['actions'] if r['accepted']])):
        result['centerline_visibility'][name] = {'action_count': len(rows), 'segments': {}}
        for segment in SEGMENTS:
            values = [row['segments'][segment] for row in rows]
            result['centerline_visibility'][name]['segments'][segment] = {
                'mean_continuous_center_domain_fraction': mean(v['continuous_center_domain_fraction'] for v in values),
                'mean_continuous_fullcell_extent_fraction': mean(v['continuous_fullcell_extent_fraction'] for v in values),
                'mean_five_sample_center_fraction': mean(mean(v['sample_inside_center_domain']) for v in values)}
    result['accepted_proposal_envelope'] = record['proposal_coverage']['accepted_envelope']
    result['invariants'] = record['invariants']
    if 'invariants_after' in record:
        result['invariants_after'] = record['invariants_after']
        result['invariants_equal'] = record['invariants'] == record['invariants_after']
    return result


def summarize(record):
    if (record.get('version') != VERSION or record.get('subject') != 'sub-PAT25'
            or record.get('role') != 'TRAIN' or record.get('status') not in ('complete','incomplete')):
        raise ValueError('Only an explicitly terminal fixed PAT25 TRAIN receipt may be summarized')
    screen = record['screening']
    saved = {(row['axis'], row['outward_sign']): row for row in screen.get('exits', [])}
    if len(saved) != len(screen.get('exits', [])) or not set(saved).issubset(EXITS):
        raise ValueError('Duplicate or unexpected source-axis exit')
    selected = screen.get('selected_exit')
    if selected is not None and not screen.get('complete'):
        raise ValueError('Incomplete screening cannot authorize a selected exit')
    derived = {(row['axis'],row['outward_sign']):row for row in record.get('derived_exits',[])}
    if derived and (set(derived) != set(EXITS) or len(record['derived_exits']) != 6):
        raise ValueError('Derived exit metadata must preserve all six exact slots')
    baseline = record['original'].get('exit', {})
    original = (baseline.get('axis'), baseline.get('outward_sign'))
    exits = []
    for key in EXITS:
        row = saved.get(key, {})
        complete = row.get('status') == 'screened'
        poses = row.get('poses', [])
        metadata = derived.get(key,{})
        exits.append({'axis':key[0], 'outward_sign':key[1],
            'distance_mm':row.get('distance_mm',metadata.get('distance_mm')),
            'derived_exit_metadata':metadata or None,
            'screen_status':row.get('status','not_recorded'),
            'eligibility':('eligible' if row['eligible'] else 'rejected') if complete else 'unknown',
            'reported_eligible_flag':row.get('eligible'),
            'original':key == original or metadata.get('selected_original') is True,
            'chosen':selected is not None and list(key) == selected,
            'returned_pose_count':len(poses),
            'admissible_pose_count':sum(p['admissible'] for p in poses) if complete else None,
            'rejected_pose_count':sum(not p['admissible'] for p in poses) if complete else None,
            'unique_returned_static_checks':sum(p['duplicate_of'] is None for p in poses),
            'reused_pose_count':sum(p['duplicate_of'] is not None for p in poses),
            'geometry_unknown_pose_count':sum(bool(p['geometry']['unknowns']) for p in poses),
            'geometry_unknown_reasons':dict(Counter(reason for p in poses for reason in p['geometry']['unknowns'])),
            'rejection_reasons':dict(Counter(reason for p in poses for reason in p['reasons']))})
    return {'schema':'pat25-ingress-access-saved-summary-v1','status':'unverified_outer_status',
        'worker_status':record['status'],
        'subject':'sub-PAT25','role':'TRAIN','six_exit_denominator':6,'exits':exits,
        'selection':{key:screen.get(key) for key in ('status','complete','selected_exit','selection','screen_count','max_screens','error')},
        'phases':{phase:phase_summary(record[phase]) for phase in PHASES},
        'settings':record['settings'],'execution_binding':record.get('execution_binding'),
        'operations':{key:record.get(key) for key in ('total_preview_calls','candidate_setup_seconds','screening_seconds',
            'executed_transitions','commits','searches','policy_forwards','checkpoints','optimizer_updates','automatic_retry')},
        'worker':{key:record.get(key) for key in ('elapsed_seconds','peak_rss_bytes','inputs_unchanged','error',
            'observer_violation','observer_health_verified')},
        'claim_boundaries':['Static ingress eligibility is not whole-stroke feasibility.',
            'Geometry unknowns are separate from eligibility and remain unknown.',
            'Center interpolation domains and full-cell extents are distinct from complete-tool clearance.',
            'Zero initial cavity is an empty procedure state, not unavailable evidence.',
            'No resection, learned policy benefit, clinical benefit or neurological-risk claim.'],
        'independent_audit':'pending separate saved-only audit'}


def attach_termination(summary, worker_raw, acceptance_raw, outer_raw):
    """Preserve outer authority; this is not an independent scientific audit."""
    acceptance, outer = json.loads(acceptance_raw), json.loads(outer_raw)
    links = {'worker_receipt_matches':acceptance.get('worker_receipt_sha256') == hashlib.sha256(worker_raw).hexdigest(),
             'inner_acceptance_matches':outer.get('acceptance_sha256') == hashlib.sha256(acceptance_raw).hexdigest()}
    elapsed = outer.get('elapsed_seconds')
    in_budget = type(elapsed) in (int,float) and math.isfinite(elapsed) and 0 <= elapsed < 180
    complete = (all(links.values()) and summary['worker_status'] == 'complete'
        and summary['worker']['observer_health_verified'] is True
        and summary['worker']['observer_violation'] is None
        and summary['worker']['inputs_unchanged'] is True and acceptance.get('inputs_unchanged') is True
        and acceptance.get('status') == 'complete' and outer.get('status') == 'complete'
        and type(outer.get('returncode')) is int and outer['returncode'] == 0
        and outer.get('group_gone') is True and outer.get('reason') is None and in_budget)
    summary['status'] = 'complete' if complete else 'incomplete'
    summary['termination'] = {'acceptance':acceptance,'outer':outer,'receipt_links':links,
        'acceptance_sha256':hashlib.sha256(acceptance_raw).hexdigest(),
        'outer_sha256':hashlib.sha256(outer_raw).hexdigest(),
        'scope':'Saved lifecycle receipt consistency only; root terminal confirmation and independent evidence audit remain required.'}
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('receipt', type=Path); parser.add_argument('output', type=Path)
    parser.add_argument('--acceptance',type=Path,required=True)
    parser.add_argument('--outer',type=Path,required=True)
    args = parser.parse_args(); raw = args.receipt.read_bytes()
    report = summarize(json.loads(raw)); report['receipt_sha256'] = hashlib.sha256(raw).hexdigest()
    report['receipt_path'] = str(args.receipt)
    attach_termination(report,raw,args.acceptance.read_bytes(),args.outer.read_bytes())
    report['termination']['acceptance_path'] = str(args.acceptance)
    report['termination']['outer_path'] = str(args.outer)
    with args.output.open('x') as stream:
        stream.write(json.dumps(report, sort_keys=True, indent=2, allow_nan=False)+'\n')
