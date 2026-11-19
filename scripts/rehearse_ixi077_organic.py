#!/usr/bin/env python3
"""Record one fixed post-exposure simulation on the IXI077 organic development case."""
import argparse
from importlib.metadata import version
import json
import math
from pathlib import Path
import platform
from tempfile import TemporaryDirectory
import time

from build_ixi077_organic_rehearsal import CASE_ID, COLLECTION, OBSERVATION, UUID
from data_integrated_hosts import checked_file
from data_integrated_ixi077_trial import (
    APPLY_SECONDS, CAPACITY_PER_SECOND, backend_binding, json_data, segment_fraction,
    stock_kernels, verify_preview, workspace_identity,
)
from resectionlab.data_root import atomic_json, checked_root, safe_path, sha256_file
from resectionlab.desktop_bridge import BridgeSession, _Request
from resectionlab.native_backend import strict_live_identity
from resectionlab.surgical_cusa_v3 import default_profile
from resectionlab.workspace_bundle import load_workspace

SOURCE = f'{COLLECTION}/{CASE_ID}/receipt.json'
SOURCE_SHA = '19a58c3ce08cf718633e2fc6f366b4244815d65d954099456c46b47582f622a2'
OUTPUT = 'ixi077-organic-conditional-replay-v1'
MAX_APPLIES = 8
FORWARD_SECONDS, REVERSE_SECONDS, VERIFY_SECONDS = 120., 180., 240.


class Deadline(ValueError):
    pass


class BoundedRequest(_Request):
    def check(self):
        super().check()
        if time.monotonic() >= self.deadline:
            raise Deadline('Fixed trial wall deadline reached')


def runtime_binding():
    return {'backend_sha256': backend_binding(), 'executor_sha256': sha256_file(__file__),
            'native_kernels': stock_kernels(), 'numerical_backend': strict_live_identity(),
            'runtime_versions': {'python': platform.python_version(), **{
                name: version(name) for name in ('numpy', 'scipy', 'nibabel')}}}


def prepare(root):
    source = {'path': SOURCE, 'sha256': SOURCE_SHA}
    receipt = json.loads(checked_file(root, source).read_text())
    preflight = json.loads(checked_file(root, receipt['static_preflight']).read_text())
    package = checked_file(root, receipt['native_case'])
    if (receipt['case_id'] != CASE_ID or receipt['role'] != 'DEVELOPMENT'
            or preflight['source_case_hash'] != receipt['case_hash'] or preflight['case_id'] != CASE_ID
            or preflight['status'] != 'necessary_static_checks_pass_pending_fresh_execution_review'
            or not preflight['checks'] or not all(preflight['checks'].values())
            or preflight['instrument_admitted'] is not False or preflight['actions_executed'] != 0
            or preflight['visible_contour_total_cells'] != 282 or preflight['automatic_planning']['available']):
        raise ValueError('Expected the exact static-only organic development case')

    raw = preflight['route']
    route = {'A_ras_mm': raw['start_ras_mm'], 'T_ras_mm': raw['goal_ras_mm'],
             'orientation_xyzw': raw['orientation_xyzw']}
    intervals = []
    next_start = 0.
    for row in preflight['intervals']:
        first, last = (segment_fraction({'position': row[key], 'quaternion': raw['orientation_xyzw']}, route)
                       for key in ('start_ras_mm', 'end_ras_mm'))
        brain, target = (set(row[key]) for key in ('working_brain_ids', 'working_target_ids'))
        if (row['index'] != len(intervals) or abs(first - next_start) > 1e-9 or last <= first
                or row['native_fov_minimum_margin_mm'] <= 0 or brain & target):
            raise ValueError('Static cover is not contiguous inside the native field')
        intervals.append((first, last, brain, target))
        next_start = last
    if (len(intervals) != 121 or abs(next_start - 1.) > 1e-9
            or tuple(len(set().union(*(row[i] for row in intervals))) for i in (2, 3)) != (358, 96)):
        raise ValueError('Fixed cover endpoints or brain and contour bounds differ')

    start = {'position': route['A_ras_mm'], 'quaternion': raw['orientation_xyzw']}
    goal = {'position': route['T_ras_mm'], 'quaternion': raw['orientation_xyzw']}
    actions = {'configure': {'duration': .1, 'mechanicsRevision': 2, 'clockRevision': 1},
               'place': {'duration': .1, 'pose': start},
               'apply': {'duration': APPLY_SECONDS, 'path': [goal]},
               'withdraw': {'duration': 1., 'path': [start]}}
    contract = {'case_hash': receipt['case_hash'], 'scene_id': preflight['source_scene_id'],
                'initial_state_id': preflight['zero_action_state_id'],
                'input_bindings': [source, receipt['native_case'], receipt['static_preflight']],
                'actions': {name: {'slot': 'right', 'operation': name, **value} for name, value in actions.items()},
                'maximum_applies': MAX_APPLIES, 'maximum_active_seconds': MAX_APPLIES * APPLY_SECONDS,
                'maximum_nominal_capacity_mm3': MAX_APPLIES * APPLY_SECONDS * CAPACITY_PER_SECOND,
                'maximum_brain_cells': 358, 'maximum_contour_cells': 96,
                'forward_wall_seconds': FORWARD_SECONDS, 'reverse_wall_seconds': REVERSE_SECONDS,
                'verification_wall_seconds': VERIFY_SECONDS, 'runtime': runtime_binding(),
                'clinical_validation': False, 'patient_admission': False,
                'scope': 'Fixed development simulation after assumed scalp, skull and dural exposure. '
                         'Stop at the first accepted apply boundary with contour effect, endpoint, refusal or cap. '
                         'One fixed reverse attempt.'}
    return package, contract, route, intervals


def check_effect(before, prediction, action, route, intervals, slots, voxel_mm3):
    # The frozen verifier expects the original scenario's target name. Map
    # only its private metric inputs. Keep the production responses and IDs.
    def legacy(view):
        metrics = view['metrics']
        volumes = metrics['removedByLabelMm3']
        if set(volumes) - {'brain', OBSERVATION} or not math.isclose(
                metrics['targetRemovedMm3'], volumes.get(OBSERVATION, 0.), rel_tol=0., abs_tol=1e-8):
            raise ValueError('Removal must be accounted for by the visible contour and modeled brain')
        return {**view, 'metrics': {**metrics, 'removedByLabelMm3': {
            'synthetic_target' if key == OBSERVATION else key: value for key, value in volumes.items()}}}

    view = prediction['view']
    result = verify_preview(legacy(before), {**prediction, 'view': legacy(view)}, action, route, intervals, slots, voxel_mm3)
    if view['physics']['cusaV3']['profile'] != default_profile() or view['protection'] != before['protection']:
        raise ValueError('Preview changed the configured stock profile or protection')
    if action['operation'] == 'withdraw' and any(slot['inserted'] or slot['active'] for slot in view['slots'].values()):
        raise ValueError('Reverse ended with an inserted or active instrument')
    return result


def run(root, output, prepared):
    package, contract, route, intervals = prepared
    checked_root(root, expected_uuid=UUID, required_bytes=1024**3)
    output.mkdir()
    receipt = output / 'report.json'
    started = time.monotonic()
    case_hash = contract['case_hash']
    rows, committed = [], []
    status = {'contract': contract, 'status': 'in_progress', 'target_effect': False, 'withdrawn': False,
              'source_unchanged': False, 'cold_reopen_exact': False}
    atomic_json(receipt, status)

    def verify_live_inputs():
        for binding in contract['input_bindings']:
            checked_file(root, binding)
        if runtime_binding() != contract['runtime']:
            raise ValueError('Runtime implementation changed during the run')

    def call(session, operation, seconds=VERIFY_SECONDS, **arguments):
        request = BoundedRequest(operation)
        request.deadline = started + seconds
        request.check()
        return json_data(session.execute(operation, arguments, request, lambda *_: None))

    def command(session, name, seconds=VERIFY_SECONDS, **arguments):
        return call(session, 'surgicalRehearsal', seconds, caseHash=case_hash, command=name, **arguments)

    def exercise(session):
        loaded = call(session, 'loadCase', path=str(package))
        if loaded['caseHash'] != case_hash or command(session, 'inspect') is not None:
            raise ValueError('Expected the pristine organic workspace')
        current = command(session, 'create')
        initial = current['view']
        slots = {hand: {k: v for k, v in slot.items() if k != 'contactGeometry'} for hand, slot in initial['slots'].items()}
        if (initial['scene']['sceneId'] != contract['scene_id'] or initial['stateId'] != contract['initial_state_id']
                or initial['removed'] or initial['history'] or initial.get('toolMechanicsRevision') is not None
                or any(slot['inserted'] or slot['active'] for slot in slots.values())):
            raise ValueError('Production initial scene or parked tools differ from the static proof')
        voxel_mm3 = initial['scene']['environment']['grid']['voxelVolumeMm3']

        def checked_step(name):
            nonlocal current
            action = contract['actions'][name]
            before = current['view']
            seconds = REVERSE_SECONDS if name == 'withdraw' else FORWARD_SECONDS
            args = {'expectedStateId': before['stateId'], 'action': action}
            row = {'action': action, 'before_state_id': before['stateId'], 'committed': False}
            rows.append(row)
            atomic_json(output / 'actions.json', rows)
            try:
                preview = command(session, 'preview', seconds - 5., **args)
                row['preview'] = {key: value for key, value in preview.items() if key != 'view'}
                row['checked_cover'] = check_effect(before, preview, action, route, intervals, slots, voxel_mm3)
            except ValueError as error:
                row['refusal'] = str(error)
            try:
                if command(session, 'inspect', seconds)['view'] != before:
                    raise RuntimeError('Detached preview changed the accepted state')
            except Deadline as error:
                row['refusal'] = str(error)
            atomic_json(output / 'actions.json', rows)
            if 'refusal' in row:
                return False
            row['step_attempted'] = True
            atomic_json(output / 'actions.json', rows)
            try:
                actual = command(session, 'step', seconds, **args)
            except Exception:
                row['commit_verification_unresolved'] = True
                atomic_json(output / 'actions.json', rows)
                raise
            row.update(actual_state_id=actual['view']['stateId'], actual_outcome=actual['outcome'],
                       committed=actual['outcome']['status'] in ('accepted', 'partial'))
            if row['committed']:
                committed.append(action)
            atomic_json(output / 'actions.json', rows)
            if (actual['view'] != preview['view'] or actual['outcome']['status'] not in ('accepted', 'partial')
                    or any(preview.get(key) != value for key, value in actual['outcome'].items())):
                raise RuntimeError('Committed simulator outcome differs from the checked preview')
            current = actual
            return True

        if not checked_step('configure') or not checked_step('place'):
            status['forward_stop'] = 'setup_refusal'
            return current
        status['forward_stop'] = 'fixed_apply_cap'
        for _ in range(MAX_APPLIES):
            if not checked_step('apply'):
                status['forward_stop'] = 'refused'
                break
            view = current['view']
            status['attained_fraction'] = segment_fraction(view['slots']['right'], route)
            if view['metrics']['targetRemovedMm3'] > 0.:
                status.update(target_effect=True, target_effect_state_id=view['stateId'], forward_stop='first_target_effect')
                break
            if status['attained_fraction'] >= 1. - 1e-9:
                status['forward_stop'] = 'endpoint_without_target_effect'
                break
        status['withdrawn'] = checked_step('withdraw')
        if not status['withdrawn']:
            status['withdrawal_refusal'] = rows[-1]['refusal']
        return current

    try:
        verify_live_inputs()
        with TemporaryDirectory(prefix='.trial-scratch-', dir=output) as scratch:
            temp = Path(scratch)
            session = BridgeSession(temp / 'transfers', run_dir=temp / 'runs')
            try:
                current = exercise(session)
                final = current['view']
                status.update(final_metrics=final['metrics'], final_state_id=final['stateId'],
                              final_slots=final['slots'], final_removed_source_ids=final['removed'])
                saved = output / 'replay.ressectionlab'
                checked_root(root, expected_uuid=UUID)
                call(session, 'saveCase', path=str(saved), caseHash=case_hash)
            finally:
                session.close()
            case, artifacts, _ = load_workspace(saved)
            recording = artifacts['surgical_toolbox']
            branch = recording['branches'][recording['branchId']]
            if (case.semantic_hash != case_hash or workspace_identity(saved) != workspace_identity(package)
                    or branch['actions'] != committed or branch['stateId'] != final['stateId']
                    or branch['checkpoint']['position'] != len(committed)):
                raise ValueError('Saved rehearsal differs from the verified source and history')
            del case, artifacts, recording, branch
            cold = BridgeSession(temp / 'cold-transfers', run_dir=temp / 'cold-runs')
            try:
                call(cold, 'loadCase', path=str(saved))
                reopened = command(cold, 'inspect')
                if reopened['view'] != final or reopened['timeline'] != current['timeline']:
                    raise ValueError('Fresh session did not restore the exact recorded result')
            finally:
                cold.close()
            verify_live_inputs()
            outcome = 'target_effect_and_withdrawal' if status['target_effect'] and status['withdrawn'] else 'bounded_negative'
            status.update(source_unchanged=True, cold_reopen_exact=True, status=outcome,
                          replay=str(saved.relative_to(root)), replay_sha256=sha256_file(saved))
    except Exception as error:
        status.update(status='unverified', error=f'{type(error).__name__}: {error}')
    finally:
        status.update(committed_action_count=len(committed), attempts=len(rows), wall_seconds=time.monotonic() - started)
        atomic_json(receipt, status)
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--output-name', default=OUTPUT)
    parser.add_argument('--describe', action='store_true')
    args = parser.parse_args()
    if not args.output_name or Path(args.output_name).name != args.output_name or args.output_name.startswith('.'):
        parser.error('Use one new output directory name')
    root = checked_root(args.root, expected_uuid=UUID)
    prepared = prepare(root)
    if args.describe:
        print(json.dumps(prepared[1], indent=2))
        return
    result = run(root, safe_path(root, Path('qc') / args.output_name), prepared)
    print(json.dumps({key: result.get(key) for key in ('status', 'error', 'target_effect', 'withdrawn',
                                                        'cold_reopen_exact', 'replay', 'committed_action_count')}, indent=2))
    if result['status'] == 'unverified':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
