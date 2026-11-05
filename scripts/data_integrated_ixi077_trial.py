#!/usr/bin/env python3
"""One exact conditional post-exposure IXI077 instrument protocol."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
import zipfile

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / 'src'
sys.path.insert(0, str(SRC))
from resectionlab.data_root import atomic_json, checked_root, safe_path, sha256_file

UUID = '9CC3623C-E38B-3EEE-B6BB-B9A06C193700'
INTEGRATED = Path('integrated-v1')
SUBJECT = 'IXI-077-T1-MRA-INTEGRATED-source-reviewed-superficial-v1'
NATIVE = INTEGRATED / 'cases' / SUBJECT / (SUBJECT + '.ressectionlab')
PROPOSAL = INTEGRATED / 'healthy/IXI077/source-reviewed-superficial-v1/proposal.json'
SWEEP = INTEGRATED / 'access/IXI077-source-reviewed-superficial-fixed-route-v1/report.json'
NATIVE_REVIEW = INTEGRATED / 'qc/independent/IXI077-source-reviewed-superficial-native-app-postreview-v1.json'
PREFLIGHT = INTEGRATED / 'probes/IXI077-production-fixed-preflight-v1/report.json'
AUTHORITY = INTEGRATED / 'qc/independent/IXI077-conditional-fixed-trial-authority-v1.json'
OUTPUT = INTEGRATED / 'access/IXI077-conditional-fixed-physical-trial-v1'
CLAIM = INTEGRATED / 'access/IXI077-conditional-fixed-physical-trial-claim-v1.json'
BINDINGS = {
    NATIVE: '4abbd29bac0e2b2633f477f779b3badbf620cd7350480ba760ffda05c8690cdd',
    PROPOSAL: '9d0db864b7d5f83cc283ffd5f241197802e90fe8a2c643aba530c5b5eee55afd',
    SWEEP: '0417e7ac8251d4b73f968e1a91c296d8ed4d3c6be22ae2d47ccaa317f7b12f8c',
    PREFLIGHT: 'a668365f4069bdea836eb29b27172f58fd2ce33556bf558a74eba2ae66cc22e4',
    NATIVE_REVIEW: '0c9b0f8646263a764f69e1059e659563fcea38f434f5f722111a0ea3eb009477',
}
MAX_APPLIES = 16
APPLY_SECONDS = 30.
CAPACITY_PER_SECOND = 2.
TOLERANCE_MM = 1e-7
FORWARD_WALL_SECONDS = 780.
TOTAL_WALL_SECONDS = 1200.


class TrialDeadline(Exception):
    pass


def read(path):
    return json.loads(path.read_text())


def json_data(value):
    return json.loads(json.dumps(value, allow_nan=False))


def json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def backend_binding():
    package = SRC / 'resectionlab'
    files = sorted(p for p in package.rglob('*') if p.suffix in {'.py', '.c', '.json'} and p.is_file())
    rows = [(str(p.relative_to(package)), sha256_file(p)) for p in files]
    return json_sha(rows)


def stock_kernels():
    from resectionlab.surgical_idle_native import strict_live_identity as idle_identity
    from resectionlab.surgical_prefix_json_native import strict_live_identity as prefix_identity
    identities = {'idle_thermal': idle_identity(), 'prefix_json': prefix_identity(full=True)}
    if any(value.get('mode') != 'python' or value.get('selection') != 'unconfigured-fixed-fallback'
           for value in identities.values()):
        raise ValueError('Expected unconfigured production Python kernels')
    return identities


def workspace_identity(path):
    """Compare retained workspace sources independent of ZIP member timestamps."""
    with zipfile.ZipFile(path) as archive:
        workspace = json.loads(archive.read('workspace.json'))
    del workspace['sessionHash']
    workspace['images'] = [row['source'] for row in workspace['images']]
    return workspace


def segment_fraction(pose, route):
    start, end = np.asarray(route['A_ras_mm']), np.asarray(route['T_ras_mm'])
    axis = end - start
    point = np.asarray(pose['position'], dtype=float)
    q = np.asarray(pose['quaternion'], dtype=float)
    target_q = np.asarray(route['orientation_xyzw'])
    if (point.shape != (3,) or q.shape != (4,) or not np.isfinite(point).all()
            or not np.isfinite(q).all() or min(np.linalg.norm(q - target_q), np.linalg.norm(q + target_q)) > 1e-10):
        raise ValueError('Actual instrument pose or orientation differs from the fixed route')
    fraction = float(np.dot(point - start, axis) / np.dot(axis, axis))
    if (not -1e-9 <= fraction <= 1. + 1e-9
            or np.linalg.norm(point - (start + fraction * axis)) > TOLERANCE_MM):
        raise ValueError('Actual instrument tip left the fixed line')
    return min(1., max(0., fraction))


def route_intervals(report, route):
    result = []
    prior = 0.
    for row in report['records']['production_cover_leaves']:
        start, end = (segment_fraction({'position': row[key], 'quaternion': route['orientation_xyzw']}, route)
                      for key in ('start', 'end'))
        cover = row['result']
        if (abs(start - prior) > 1e-9 or end <= start or cover['inside_native_FOV'] is not True
                or cover['FOV_minimum_margin_mm'] <= 0. or set(cover['parts']) != {'shaft', 'working'}
                or any(p['known_vessel_ids'] or p['intracranial_outside_MRA_ids'] for p in cover['parts'].values())):
            raise ValueError('The reviewed actual segment is outside the frozen complete source cover')
        working = cover['parts']['working']
        target = set(working['target_ids'])
        brain = set(working['brain_ids']) - target
        result.append((start, end, brain, target))
        prior = end
    if len(result) != 121 or abs(prior - 1.) > 1e-9:
        raise ValueError('The exact frozen route cover is incomplete')
    return result


def stock_slots(slots, fresh):
    def base(slot):
        return {key: value for key, value in slot.items() if key != 'contactGeometry'}
    if base(slots['left']) != fresh['left']:
        raise ValueError('Parked left tool changed')
    if slots['left'].get('contactGeometry') != [
            {'a': [0., 0., -40.], 'b': [0., 0., -2.], 'radiusMm': .5, 'part': 'shaft'},
            {'a': [0., 0., -2.], 'b': [0., 0., 0.], 'radiusMm': 2., 'part': 'working'}]:
        raise ValueError('Parked left tool geometry differs from the reviewed stock drill')
    right = slots['right']
    if right['instrument'] != 'cusa' or right['active'] or right['grip'] or right['parameters']:
        raise ValueError('Right tool changed, latched, gripped, or overridden')
    expected = [
        {'a': [0., 0., -40.], 'b': [0., 0., 0.], 'radiusMm': .5, 'part': 'shaft'},
        {'a': [0., 0., 0.], 'b': [0., 0., 0.], 'radiusMm': 1.5, 'part': 'working'},
    ]
    if right.get('contactGeometry') != expected:
        raise ValueError('Right CUSA geometry differs from stock revision 2')


def verify_preview(before, preview, action, route, intervals, slots, voxel_mm3):
    """Admit only checked effects on the actually attained fixed-line segment."""
    operation = action['operation']
    view = preview['view']
    metrics = view['metrics']
    prior = before['metrics']
    soft = (preview['status'], preview['reason']) == ('partial', 'TISSUE_ASPIRATION_IN_PROGRESS')
    if (preview['status'], preview['reason']) != ('accepted', None) and not (operation == 'apply' and soft):
        raise ValueError('NATIVE_REFUSAL:' + str(preview.get('reason')))
    if (view['stateId'] != preview['stateId'] or preview['deltas']['geometryRevision'] != preview['stateId']
            or view['toolMechanicsRevision'] != 2
            or view['physicalClock']['revision'] != 1
            or view['physics']['cusaV3']['profile']['capacityMm3PerSecond'] != CAPACITY_PER_SECOND
            or view['holds'] or view['moved'] or view['deformation']):
        raise ValueError('Unexpected physical mode or deformation')
    stock_slots(view['slots'], slots)
    zero_fields = ('vesselEncounters', 'activeVesselWounds', 'severedVesselConnections',
                   'modeledVesselFluidGeneratedMm3', 'brainMaxDisplacementMm', 'drillInputEnergyJ', 'collisions')
    if any(metrics[key] != 0 for key in zero_fields):
        raise ValueError('Unexpected vessel, deformation, collision, or drill effect')
    if set(metrics['removedByLabelMm3']) - {'brain', 'synthetic_target'}:
        raise ValueError('Unexpected removed label')
    prior_ids = set(before['removed'])
    delta = preview['deltas']['removedCells']
    removed = set(delta)
    if len(removed) != len(delta) or prior_ids & removed or set(view['removed']) != prior_ids | removed:
        raise ValueError('Actual source removal ledger differs')
    pending = {int(key): value for key, value in view['cusaV3']['progressMm3'].items()}
    prior_pending = {int(key): value for key, value in before.get('cusaV3', {}).get('progressMm3', {}).items()}
    capacity = metrics['cusaInputCapacityMm3'] - prior.get('cusaInputCapacityMm3', 0.)
    effective_seconds = view['cusaV3']['effectiveActivationSeconds'] - before.get('cusaV3', {}).get('effectiveActivationSeconds', 0.)
    if (not math.isclose(capacity, CAPACITY_PER_SECOND * effective_seconds, abs_tol=1e-8)
            or not math.isclose(effective_seconds, action['duration'] if operation == 'apply' else 0., abs_tol=1e-8)):
        raise ValueError('Unexpected effective activation accounting')
    if operation != 'apply' and (removed or pending != prior_pending or capacity != 0.):
        raise ValueError('Non-aspiration request changed CUSA effects')
    if operation == 'configure':
        if any(view['slots'][key]['position'] != slots[key]['position']
               or view['slots'][key]['quaternion'] != slots[key]['quaternion']
               or view['slots'][key]['inserted'] for key in slots):
            raise ValueError('Configuration changed an instrument pose')
        return {'removed_source_ids': []}

    after = segment_fraction(view['slots']['right'], route)
    start = 0. if operation == 'place' else segment_fraction(before['slots']['right'], route)
    cover_indices = [index for index, (low, high, _, _) in enumerate(intervals)
                     if high >= min(start, after) - 1e-9 and low <= max(start, after) + 1e-9]
    if operation in ('place', 'withdraw'):
        if after > 1e-9 or view['slots']['right']['inserted']:
            raise ValueError('Tool did not end withdrawn at the exact fixed A48 pose')
        return {'attained_fraction': [start, after], 'removed_source_ids': [],
                'cover_interval_indices': cover_indices}

    if (not view['slots']['right']['inserted'] or after + 1e-9 < start
            or not soft and after < 1. - 1e-9 or soft and after >= 1. - 1e-9):
        raise ValueError('Aspiration did not follow the fixed forward segment')
    brain, target = set(), set()
    for index in cover_indices:
        _, _, covered_brain, covered_target = intervals[index]
        brain.update(covered_brain)
        target.update(covered_target)
    new_pending = {cell for cell, value in pending.items() if value != prior_pending.get(cell)}
    reported = {cell for event in preview['events'] if event['kind'] == 'cusa_contact'
                for cell in event['removedCells']}
    if (not (removed | new_pending) <= brain | target or reported != removed
            or set(prior_pending) - set(pending) - removed
            or any(not 0. < pending[cell] < voxel_mm3 or pending[cell] <= prior_pending.get(cell, 0.) for cell in new_pending)
            or not math.isfinite(capacity) or not 0. < capacity <= CAPACITY_PER_SECOND * action['duration'] + 1e-8):
        raise ValueError('CUSA removal, pending work, or rate exceeded the attained segment authority')
    for key, cells in (('brain', brain), ('synthetic_target', target)):
        change = metrics['removedByLabelMm3'].get(key, 0.) - prior['removedByLabelMm3'].get(key, 0.)
        if not math.isclose(change, voxel_mm3 * len(cells & removed), rel_tol=0., abs_tol=1e-8):
            raise ValueError('Actual label removal volume differs from source IDs')
    pending_gain = sum(pending[cell] - prior_pending.get(cell, 0.) for cell in new_pending)
    if after - start <= 1e-9 and not removed and pending_gain <= 1e-10:
        raise ValueError('ASPIRATION_WITHOUT_ACHIEVED_TRAVEL_OR_WORK_PROGRESS')
    return {'attained_fraction': [start, after], 'removed_source_ids': sorted(removed),
            'new_pending_source_ids': sorted(new_pending), 'pending_gain_mm3': pending_gain,
            'covered_working_ids': len(brain | target), 'cover_interval_indices': cover_indices}


def prepare(root):
    for relative, expected in BINDINGS.items():
        if sha256_file(safe_path(root, relative)) != expected:
            raise ValueError(f'Frozen input changed: {relative}')
    route = read(root / PROPOSAL)['prospective_readonly_route_phase']['fixed_route']
    intervals = route_intervals(read(root / SWEEP), route)
    evidence = read(safe_path(root, PREFLIGHT))
    initial = preflight_initial(evidence, route)
    def action(operation, duration, **kwargs):
        return {'slot': 'right', 'operation': operation, 'duration': duration, **kwargs}
    pose = lambda key: {'position': route[key], 'quaternion': route['orientation_xyzw']}
    actions = {
        'configure': action('configure', .1, mechanicsRevision=2, clockRevision=1),
        'place': action('place', .1, pose=pose('A_ras_mm')),
        'apply': action('apply', APPLY_SECONDS, path=[pose('T_ras_mm')]),
        'withdraw': action('withdraw', 1., path=[pose('A_ras_mm')]),
    }
    authority = {
        'schema': 'independent-IXI077-conditional-fixed-trial-authority-v1',
        'reviewer': '/root/anatomy_review', 'physical_execution_approved': False,
        'executor_sha256': sha256_file(__file__), 'backend_source_inventory_sha256': backend_binding(),
        'native_kernel_identities': stock_kernels(),
        'runtime_versions': {'python': platform.python_version(), **{
            name: importlib.metadata.version(name) for name in ('numpy', 'scipy', 'nibabel')}},
        'inputs': {str(p): sha for p, sha in BINDINGS.items()},
        'initial_case_hash': initial['caseHash'], 'initial_state_id': initial['stateId'],
        'actions': actions, 'maximum_apply_attempts': MAX_APPLIES,
        'maximum_previews_and_commits_each': MAX_APPLIES + 3,
        'maximum_total_declared_seconds': 1.2 + APPLY_SECONDS * MAX_APPLIES,
        'maximum_CUSA_active_seconds': APPLY_SECONDS * MAX_APPLIES,
        'maximum_CUSA_nominal_capacity_mm3': CAPACITY_PER_SECOND * APPLY_SECONDS * MAX_APPLIES,
        'maximum_brain_source_cells': len(set().union(*(interval[2] for interval in intervals))),
        'maximum_target_source_cells': len(set().union(*(interval[3] for interval in intervals))),
        'progress': 'Forward tip travel, newly removed owned cells, or increased pending capacity on the attained portion.',
        'success_stop': 'Stop applying at first positive synthetic target removal; report exact target endpoint separately.',
        'maximum_forward_wall_seconds': FORWARD_WALL_SECONDS,
        'maximum_total_wall_seconds': TOTAL_WALL_SECONDS,
        'withdrawal': 'One fixed reverse preview from the last verified committed pose, with no alternate; blocked withdrawal leaves an unresolved diagnostic.',
        'output': str(OUTPUT), 'one_use_claim': str(CLAIM),
        'scope': 'Fixed forward aspiration and one fixed reverse withdrawal after hypothetical scalp, calvarial and dural exposure. Unknown exposure, alignment, and vascular coverage retained.',
    }
    return authority, route, intervals, initial


def preflight_initial(evidence, route):
    placement, cover = evidence['placement'], evidence['route']
    if (evidence['schema'] != 'ressectionlab-IXI077-production-fixed-preflight-v1'
            or evidence['preflight_status'] != 'geometry_prerequisites_only_physical_admission_withheld'
            or evidence['action_previews'] != 0 or evidence['actions_committed'] != 0
            or evidence['same_state_after_inspect'] is not True or evidence['source_unchanged'] is not True
            or evidence['reviewed_task_domain_present'] is not False
            or evidence['geometry_candidate'] != {'path': str(NATIVE), 'sha256': BINDINGS[NATIVE]}
            or evidence['prior_readonly_route'] != {'path': str(SWEEP), 'sha256': BINDINGS[SWEEP]}
            or evidence['independent_postreview'] != {'path': str(NATIVE_REVIEW), 'sha256': BINDINGS[NATIVE_REVIEW]}
            or evidence['startup']['parked_slots_are_inactive_uninserted'] is not True
            or placement['native_FOV_margin_mm'] <= 0 or placement['occupied_source_ids']
            or placement['stock_CUSA_vs_parked_drill_gap_mm'] <= 1.
            or placement['exact_production_capsule']['radius_mm'] != 1.5
            or placement['exact_production_capsule']['length_mm'] != 40.
            or placement['exact_production_capsule']['b'] != route['A_ras_mm']
            or cover['A_RAS_mm'] != route['A_ras_mm'] or cover['T_RAS_mm'] != route['T_ras_mm']
            or cover['orientation_xyzw'] != route['orientation_xyzw']
            or cover['reverse_only_same_curve_and_orientation'] is not True
            or cover['minimum_frozen_cover_FOV_margin_mm'] <= 0
            or cover['minimum_CUSA_vs_parked_drill_cover_gap_mm'] <= 0
            or cover['tool_pair_conflict_interval_indices']):
        raise ValueError('Exact production geometry preflight prerequisites missing')
    for name, expected in evidence['source_bindings'].items():
        if sha256_file(SRC / 'resectionlab' / name) != expected:
            raise ValueError('The current production geometry differs from preflight')
    return {'caseHash': evidence['case_hash'], 'stateId': evidence['initial_state_id'],
            'sceneId': evidence['scene_id'], 'slots': evidence['slots'],
            'protection': evidence['protection'], 'voxel_mm3': evidence['capacity']['voxel_mm3']}


def run(root, prepared):
    binding, route, intervals, initial = prepared
    authority = safe_path(root, AUTHORITY)
    approved = {**binding, 'physical_execution_approved': True}
    recorded = read(authority)
    if recorded != approved or recorded['physical_execution_approved'] is not True:
        raise ValueError('Exact independent authority required')
    authority_sha = sha256_file(authority)
    from resectionlab.desktop_bridge import BridgeSession, _Request
    from resectionlab.workspace_bundle import load_workspace
    from resectionlab.surgical_cusa_v3 import default_profile

    root = checked_root(root, required_bytes=2 * 1024**3, expected_uuid=UUID)
    output, claim = safe_path(root, OUTPUT), safe_path(root, CLAIM)
    if output.exists() or claim.exists():
        raise ValueError('One physical allocation already consumed or output exists')
    cache = safe_path(root, INTEGRATED / 'cache' / OUTPUT.name)
    for key in ('TMPDIR', 'XDG_CACHE_HOME', 'NUMBA_CACHE_DIR', 'MPLCONFIGDIR'):
        path = cache / key
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)
    import tempfile
    tempfile.tempdir = os.environ['TMPDIR']
    with claim.open('x') as stream:
        json.dump({**approved, 'authority_sha256': authority_sha}, stream, allow_nan=False, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    output.mkdir()
    started = time.monotonic()
    forward_deadline = started + FORWARD_WALL_SECONDS
    total_deadline = started + TOTAL_WALL_SECONDS
    records, committed = [], []
    report = {'schema': 'IXI077-conditional-fixed-physical-trial-result-v1', 'authority_sha256': authority_sha,
              'clinical_validation': False, 'workflow_promotion': False, 'all_tools_withdrawn': False,
              'target_endpoint_reached': False, 'target_removal_observed': False,
              'scope': binding['scope']}

    class BoundedRequest(_Request):
        def check(self):
            super().check()
            if time.monotonic() >= self.deadline:
                raise TrialDeadline('Cooperative public request wall deadline reached')

    def call(session, operation, *, deadline=total_deadline, **args):
        request = BoundedRequest(operation)
        request.deadline = deadline
        request.check()
        return json_data(session.execute(operation, args, request, lambda *_: None))

    def cmd(session, command, **args):
        return call(session, 'surgicalRehearsal', caseHash=initial['caseHash'], command=command, **args)

    def require_live_binding():
        if (sha256_file(authority) != authority_sha or sha256_file(__file__) != binding['executor_sha256']
                or backend_binding() != binding['backend_source_inventory_sha256']
                or stock_kernels() != binding['native_kernel_identities']
                or any(sha256_file(root / path) != sha for path, sha in binding['inputs'].items())):
            raise ValueError('Reviewed executor, backend, source, or authority changed')

    session = None
    current = None
    try:
        session = BridgeSession(output / 'transfers', run_dir=output / 'runs')
        loaded = call(session, 'loadCase', path=str(root / NATIVE))
        if loaded['caseHash'] != initial['caseHash'] or cmd(session, 'inspect') is not None:
            raise ValueError('Native case or absence of prior physical history differs')
        current = cmd(session, 'create')
        view = current['view']
        slots = {hand: {k: v for k, v in slot.items() if k != 'contactGeometry'}
                 for hand, slot in view['slots'].items()}
        if (view['stateId'] != initial['stateId'] or slots != initial['slots']
                or view['scene']['sceneId'] != initial['sceneId'] or view['protection'] != initial['protection']
                or view['removed'] or view['history'] or view.get('toolMechanicsRevision') is not None
                or view['physics'].get('cusaV3')):
            raise ValueError('Actual fresh zero and both installed slots differ')
        voxel_mm3 = view['scene']['environment']['grid']['voxelVolumeMm3']
        if voxel_mm3 != initial['voxel_mm3']:
            raise ValueError('Actual source cell volume differs from the reviewed case')

        def one(name):
            nonlocal current
            require_live_binding()
            action = binding['actions'][name]
            before = current['view']
            args = {'expectedStateId': before['stateId'], 'action': action}
            deadline = total_deadline if name == 'withdraw' else forward_deadline
            row = {'action': action, 'state_before': before['stateId'], 'preview_attempted': True, 'committed': False}
            records.append(row)
            atomic_json(output / 'actions.json', records)
            try:
                preview = cmd(session, 'preview', deadline=deadline, **args)
            except TrialDeadline as error:
                if cmd(session, 'inspect')['view'] != before:
                    raise ValueError('Interrupted detached preview changed actual public state')
                row['refusal'] = str(error)
                atomic_json(output / 'actions.json', records)
                return False
            row.update(preview_view_sha256=json_sha(preview['view']),
                       preview={k: v for k, v in preview.items() if k != 'view'},
                       preview_slots=preview['view']['slots'])
            atomic_json(output / 'actions.json', records)
            if cmd(session, 'inspect')['view'] != before:
                raise ValueError('Detached preview changed actual public state')
            try:
                row['achieved_cover'] = verify_preview(before, preview, action, route, intervals, initial['slots'], voxel_mm3)
                if preview['view']['physics']['cusaV3']['profile'] != default_profile():
                    raise ValueError('CUSA profile differs from stock')
                if preview['view']['protection'] != initial['protection']:
                    raise ValueError('Production protection contract changed')
            except ValueError as error:
                row['refusal'] = str(error)
                atomic_json(output / 'actions.json', records)
                return False
            atomic_json(output / 'actions.json', records)
            require_live_binding()
            if time.monotonic() >= deadline:
                row['refusal'] = 'Forward or total wall limit reached before commit'
                atomic_json(output / 'actions.json', records)
                return False
            row['step_attempted'] = True
            atomic_json(output / 'actions.json', records)
            actual = cmd(session, 'step', deadline=deadline, **args)
            row['actual'] = actual['outcome']
            row['actual_slots'] = actual['view']['slots']
            row['actual_view_sha256'] = json_sha(actual['view'])
            row['committed'] = actual['outcome']['status'] in ('accepted', 'partial')
            if row['committed']:
                committed.append(action)
            atomic_json(output / 'actions.json', records)
            if (not row['committed'] or actual['view'] != preview['view']
                    or any(actual['outcome'][k] != preview[k] for k in actual['outcome'])):
                raise ValueError('Actual public step differs from detached preview')
            current = actual
            return True

        if one('configure') and one('place'):
            report['placement_committed'] = True
            report['aspiration_stop'] = 'predeclared_attempt_cap'
            for _ in range(MAX_APPLIES):
                if not one('apply'):
                    report['aspiration_stop'] = records[-1]['refusal']
                    break
                if segment_fraction(current['view']['slots']['right'], route) >= 1. - 1e-9:
                    report['target_endpoint_reached'] = True
                if current['view']['metrics']['targetRemovedMm3'] > 0.:
                    report['aspiration_stop'] = 'first_positive_target_removal'
                    report['target_removal_observed'] = True
                    break
            report['all_tools_withdrawn'] = one('withdraw')
            if not report['all_tools_withdrawn']:
                report['withdrawal_unresolved'] = records[-1]['refusal']
        else:
            report['setup_refusal'] = records[-1]['refusal']
        report['final_metrics'] = current['view']['metrics']
        checked_root(root, expected_uuid=UUID)
        saved = output / 'replay.ressectionlab'
        call(session, 'saveCase', path=str(saved), caseHash=initial['caseHash'])
        saved_case, artifacts, workspace = load_workspace(saved)
        record = artifacts['surgical_toolbox']
        snapshot = record['branches'][record['branchId']]
        if (saved_case.semantic_hash != initial['caseHash'] or workspace_identity(saved) != workspace_identity(root / NATIVE)
                or snapshot['actions'] != committed
                or snapshot['stateId'] != current['view']['stateId'] or snapshot['checkpoint']['position'] != len(committed)):
            raise ValueError('Saved physical history or source differs')
        del saved_case, artifacts, workspace, record, snapshot
        session.close()
        session = BridgeSession(output / 'cold-transfers', run_dir=output / 'cold-runs')
        call(session, 'loadCase', path=str(saved))
        reopened = cmd(session, 'inspect')
        if reopened['view'] != current['view'] or reopened['timeline'] != current['timeline']:
            raise ValueError('Cold public state and history differ')
        require_live_binding()
        report.update(cold_public_reopen_exact=True, replay_sha256=sha256_file(saved),
                      replay=str(saved.relative_to(root)))
    except Exception as error:
        report['unresolved_error'] = str(error)
    finally:
        if session is not None:
            try:
                session.close()
            except Exception as error:
                report['session_close_error'] = str(error)
    report.update(preview_count=len(records), committed_action_count=len(committed),
                  committed_declared_seconds=sum(action['duration'] for action in committed),
                  wall_seconds=time.monotonic() - started)
    atomic_json(output / 'report.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, help='Guarded external root, defaults to RESSECTIONLAB_DATA_ROOT')
    parser.add_argument('--authority-candidate', action='store_true')
    args = parser.parse_args()
    data_root = checked_root(args.root, expected_uuid=UUID)
    prepared = prepare(data_root)
    print(json.dumps(prepared[0] if args.authority_candidate else run(data_root, prepared), indent=2))
