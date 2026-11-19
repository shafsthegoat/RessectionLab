#!/usr/bin/env python3
"""Rebuild a bounded CTA014 opening and aspiration from a pristine study."""
import argparse
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic

import numpy as np
from scipy.spatial.transform import Rotation

from data_integrated_hosts import ORDER, UUID, checked_file
from data_integrated_ixi077_trial import json_data, segment_fraction, workspace_identity
from data_integrated_probe import envelope_admission
from data_integrated_regression import allowed_status, contact_trace
from probe_cta014_route_conditioned import CASE, HISTORY, OUTPUT as CASE_DIR
from probe_cta_organic_rehearsal import OBSERVATION, cells, cover, tool_segments
from rehearse_ixi077_organic import BoundedRequest, runtime_binding
from resectionlab.core import array_digest, semantic_digest
from resectionlab.data_root import atomic_json, checked_root, safe_path, sha256_file
from resectionlab.desktop_bridge import BridgeSession
from resectionlab.geometry import GeometryScene
from resectionlab.surgical_environment import standardize
from resectionlab.surgical_motion import swept_envelopes
from resectionlab.surgical_scene import from_case
from resectionlab.surgical_tools import TOOLS
from resectionlab.workspace_bundle import load_workspace

STATIC = {'path': str(CASE_DIR / 'preflight.json'),
          'sha256': '91bf118df110ab71f79a26b6a4e2a0915592c81fa98cc436e1fa50a258e4a5f8'}
OUTPUT = 'cta014-organic-fixed-route-rehearsal-v1'
ACTIVE_SECONDS, MAX_APPLIES = 30., 5
EXECUTION_SECONDS, EXIT_SECONDS, VERIFY_SECONDS = 300., 360., 480.
BRAIN_CAP, TARGET_CAP = 617, 375
TEMP_CAP = 124.24613176586166
BRAIN, BONE, ARTERY, VEIN = ORDER


def outcome(row):
    return row['actual'].get('outcome', row['actual'])


def live_runtime():
    binding = runtime_binding()
    return {**binding, 'executor_sha256': sha256_file(__file__),
            'shared_runner_sha256': binding['executor_sha256']}


def prepare(root):
    static = json.loads(checked_file(root, STATIC).read_text())
    if (static['case_id'] != CASE or static['actions_executed'] != 0
            or static['existing_authority_or_state_inherited'] is not False
            or static['vote'] != 'candidate_for_fresh_execution_review'
            or static['cusa_working_cover_to_historical_attained_tip_cells'] != TARGET_CAP):
        raise ValueError('Expected the exact pristine case and fixed-route screen')
    native = static['case']['native_case']
    package = checked_file(root, native)
    archive = {name: json.loads(checked_file(root, ref).read_text()) for name, ref in HISTORY.items()}
    opening = [row for row in archive['opening'] if row['committed']]
    if [x['literal_index'] for x in opening] != [*range(23), 24]:
        raise ValueError('Expected the 24 accepted opening requests with the old refusal omitted')
    prior = archive['preflight']
    selected, = (row for row in archive['plan']['proposals'] if row['route']['sample'] == prior['route_sample'])
    old_route = selected['route']
    setup = archive['kerf'][:18]
    if ([x['action']['operation'] for x in setup] != ['exchange', *(['place', 'apply', 'withdraw'] * 5), 'exchange', 'place']
            or not all(x['committed'] for x in setup)
            or setup[0]['action']['instrument'] != 'craniotome' or setup[16]['action']['instrument'] != 'cusa'):
        raise ValueError('Expected five historical kerfs and their returns followed by CUSA setup')
    route = {'A_ras_mm': old_route['A'], 'T_ras_mm': prior['tip_position'],
             'orientation_xyzw': old_route['orientation']}
    start, end = ({'position': route[key], 'quaternion': route['orientation_xyzw']}
                  for key in ('A_ras_mm', 'T_ras_mm'))
    if segment_fraction(setup[-1]['action']['pose'], route) != 0.:
        raise ValueError('Archived placement does not match the bounded aspiration line')
    stock = {'slot': 'right'}
    active = {**stock, 'operation': 'apply', 'duration': ACTIVE_SECONDS, 'path': [end]}
    reverse = {**stock, 'operation': 'withdraw', 'duration': 22., 'path': [start]}
    bone = [set(cell for row in rows for cell in outcome(row)['deltas']['removedCells']) for rows in (opening, setup)]
    if [len(ids) for ids in bone] != [552, 22] or bone[0] & bone[1] or set(old_route['bone_ids']) - (bone[0] | bone[1]):
        raise ValueError('Historical bone volumes or guarded blockers differ')
    contract = {'schema': 'cta014-organic-fixed-rehearsal-contract-v1', 'case_id': CASE,
                'case_hash': static['case']['case_hash'], 'bindings': [STATIC, native, *HISTORY.values()],
                'runtime': live_runtime(), 'opening_actions': len(opening), 'kerf_and_setup_actions': len(setup),
                'historical_guarded_bone_ids': old_route['bone_ids'],
                'limits': {'opening_bone_cells': 552, 'additional_kerf_bone_cells': 22, 'brain_cells': BRAIN_CAP,
                           'target_cells': TARGET_CAP, 'applies': MAX_APPLIES, 'modeled_active_seconds': 150.,
                           'action_boundary_temperature_ceiling_C': TEMP_CAP},
                'aspirate': active, 'reverse': reverse,
                'scope': 'Bounded modeled skull opening, first contour effect and checked reverse. Scalp exposure and '
                         'dural opening are assumed. Unmapped tissue and vascular coverage remain unknown.',
                'clinical_validation': False, 'historical_state_or_authority_inherited': False}
    return package, static, contract, route, opening, setup, bone


def same_slots(actual, expected):
    for hand, a in actual.items():
        b = expected[hand]
        if (any(a[field] != b[field] for field in ('instrument', 'inserted', 'active', 'grip', 'parameters'))
                or not np.allclose(a['position'], b['position'], atol=1e-8, rtol=0.)
                or abs(float(np.dot(a['quaternion'], b['quaternion']))) < 1. - 1e-12):
            return False
    return True


class Checks:
    def __init__(self, case, static, route, bone):
        if (case.semantic_hash != static['case']['case_hash'] or set(case.compartments) != {*ORDER, OBSERVATION}
                or set(case.source_compartments) != set(case.compartments)
                or any(not np.array_equal(case.source_compartments[name], mask) for name, mask in case.compartments.items())
                or any(array_digest(case.compartments[name]) != digest
                       for name, digest in static['case']['source_membership_hashes'].items())
                or array_digest(case.compartments[OBSERVATION]) != static['contour_sha256']):
            raise ValueError('Pristine observation or source anatomy differs')
        self.scene = standardize(from_case(case))
        self.grid = GeometryScene(np.zeros(case.mri.shape, bool), case.affine)
        self.labels = self.scene.labels.ravel()
        self.regions = {row['name']: row['id'] for row in self.scene.regions}
        self.vessels = (case.compartments[ARTERY] | case.compartments[VEIN]).ravel()
        self.bone, self.route = bone, route
        self.voxel_mm3 = case.voxel_volume_mm3

    def validate_motion(self, geometry):
        if not envelope_admission(geometry, self.scene.affine, np.asarray(self.scene.shape))['admitted']:
            raise ValueError('Full prospective instrument envelope leaves native acquisition')
        ids = cells(self.grid, geometry)
        if self.vessels[ids].any():
            raise ValueError('Prospective whole-tool envelope meets a mapped vessel')
        return ids

    def check_setup_path(self, before, action, instrument):
        tool = TOOLS[instrument]
        current = before['slots'][action['slot']]
        parameters = {'radiusMm': tool['radiusMm'], 'depthMm': 2., 'contactGeometryRevision': 2}
        operation = action['operation']
        if operation in ('place', 'exchange'):
            pose = action['pose'] if operation == 'place' else current
            geometry = tool_segments(instrument, pose['position'], pose['quaternion'])
            if operation == 'place':
                tip = np.asarray(pose['position'])
                direction = Rotation.from_quat(pose['quaternion']).apply([0., 0., 1.])
                geometry.append((tip - 40. * direction, tip, max(tool['radiusMm'], tool['shaftRadiusMm']), 'placement'))
            return self.validate_motion(geometry).size
        # Each kerf stroke and reverse are multi-waypoint requests. Audit the
        # complete requested path while the engine resolves retained bone.
        points = [current, *action['path']]
        touched = set()
        for a, b in zip(points, points[1:]):
            ids = self.validate_motion(swept_envelopes(tool, parameters, a, b))
            occupied = self.labels[ids]
            if np.any((occupied != 0) & (occupied != self.regions[BONE])):
                raise ValueError('Requested kerf or reverse crosses modeled nonbone tissue')
            touched.update(map(int, ids))
        return len(touched)

    def ready_for_cusa(self, view, guarded):
        geometry = cover('cusa', self.route['A_ras_mm'], self.route['T_ras_mm'], self.route['orientation_xyzw'])
        ids = self.validate_motion(geometry)
        removed = set(view['removed'])
        blockers = {int(i) for i in ids if self.labels[i] == self.regions[BONE]}
        if removed != self.bone[0] | self.bone[1] or (blockers | set(guarded)) - removed:
            raise ValueError('Accepted fresh opening does not clear the complete bounded CUSA envelope')
        return sorted(blockers)

    def verify(self, phase, before, prediction, action, reference):
        view = prediction['view']
        instrument = action.get('instrument', before['slots']['right']['instrument'])
        if not allowed_status(action, instrument, prediction):
            raise ValueError('Native refusal: ' + str(prediction.get('reasonCode') or prediction.get('reason')))
        for name in ('vesselEncounters', 'activeVesselWounds', 'severedVesselConnections',
                     'modeledVesselFluidGeneratedMm3', 'collisions', 'brainMaxDisplacementMm'):
            if view['metrics'][name] != 0:
                raise ValueError('Unexpected mapped-vessel effect, collision or deformation: ' + name)
        if (view['holds'] or view['moved'] or view['deformation'] or view['protection'] != before['protection']
                or (view['slots']['left'] != before['slots']['left'] and action['operation'] != 'configure')
                or any(slot['grip'] or slot['active'] for slot in view['slots'].values())):
            raise ValueError('Immovable guard, parked tool or released grip changed')

        prior = set(before['removed'])
        delta = prediction['deltas']['removedCells']
        removed = set(delta)
        if len(removed) != len(delta) or removed & prior or set(view['removed']) != prior | removed:
            raise ValueError('Removed source-cell accounting differs')
        if action['operation'] != 'apply' and removed:
            raise ValueError('An inactive request removed source cells')
        volumes = view['metrics']['removedByLabelMm3']
        limits = {BONE: 574, BRAIN: BRAIN_CAP, OBSERVATION: TARGET_CAP}
        owned = self.labels[list(prior | removed)] if prior or removed else np.array([], np.uint16)
        if set(volumes) - set(limits):
            raise ValueError('Unsupported removed material')
        for key, cap in limits.items():
            count = int(np.count_nonzero(owned == self.regions[key]))
            if count > cap or not math.isclose(volumes.get(key, 0.), count * self.voxel_mm3, rel_tol=0., abs_tol=1e-7):
                raise ValueError('Physical removed-volume ledger or cap differs for ' + key)
        if not math.isclose(view['metrics']['targetRemovedMm3'], volumes.get(OBSERVATION, 0.), rel_tol=0., abs_tol=1e-7):
            raise ValueError('Target accounting exceeds the observable contour')
        maximum = view['metrics'].get('drillMaxTemperatureC', 37.)
        if not math.isfinite(maximum) or maximum > TEMP_CAP + 1e-8:
            raise ValueError('Fresh action-boundary thermal ledger exceeded the historical comparison ceiling')

        if reference is not None:
            old = outcome(reference)
            old_slots = reference['actual_slots_after'] if phase == 'opening' else reference['slots']
            if (any(prediction[key] != old[key] for key in ('status', 'reasonCode', 'completedFraction'))
                    or removed != set(old['deltas']['removedCells'])
                    or not same_slots(view['slots'], old_slots)
                    or removed - (self.bone[0] if phase == 'opening' else self.bone[1])):
                raise ValueError('Fresh preparation differs from the bound historical poses or bone-only IDs')
            if phase == 'opening':
                allowed = {'by_material': {'bone': self.bone[0], 'soft_tissue': set()}}
                passed, reason, trace = contact_trace(self.scene, action, before['slots'], prediction, prior, allowed)
                if not passed:
                    raise ValueError('Fresh native contact trace did not pass: ' + reason)
                for sample in trace:
                    self.validate_motion([(item['a'], item['b'], item['radiusMm'], item['part'])
                                          for item in sample['envelopes']])
                if action['operation'] == 'place':
                    self.check_setup_path(before, action, instrument)
                return {'source_bone_cells': len(removed), 'reconstructed_microsteps': len(trace)}
            size = self.check_setup_path(before, action, instrument)
            return {'source_bone_cells': len(removed), 'requested_sweep_cell_queries': size}
        return self.verify_treatment(before, view, prediction, action, removed)

    def verify_treatment(self, before, view, receipt, action, removed):
        right = view['slots']['right']
        if right['instrument'] != 'cusa' or right['parameters'] or view['toolMechanicsRevision'] != 2:
            raise ValueError('Treatment changed stock CUSA mechanics')
        start = segment_fraction(before['slots']['right'], self.route)
        end = segment_fraction(right, self.route)
        original = before['slots']['right']['position']
        a, b = (original, right['position']) if start <= end else (right['position'], original)
        geometry = cover('cusa', a, b, right['quaternion'])
        self.validate_motion(geometry)
        if view['metrics']['drillInputEnergyJ'] != before['metrics']['drillInputEnergyJ']:
            raise ValueError('CUSA or its reverse added drill energy')
        pending = {int(key): value for key, value in view['cusaV3']['progressMm3'].items()}
        old = {int(key): value for key, value in before['cusaV3']['progressMm3'].items()}
        capacity = view['metrics']['cusaInputCapacityMm3'] - before['metrics']['cusaInputCapacityMm3']
        if action['operation'] == 'withdraw':
            if end > 1e-9 or any(s['inserted'] for s in view['slots'].values()) or removed or pending != old or capacity:
                raise ValueError('Reverse did not restore the exterior pose with unchanged effect')
        else:
            if end + 1e-9 < start or not right['inserted']:
                raise ValueError('Aspiration did not stay on the fixed forward line')
            working = set(map(int, cells(self.grid, geometry, 'working')))
            changed = {cell for cell, value in pending.items() if old.get(cell) != value}
            declared = {cell for event in receipt['events'] if event['kind'] == 'cusa_contact' for cell in event['removedCells']}
            if ((removed | changed) - working or removed != declared or set(old) - set(pending) - removed
                    or any(not old.get(cell, 0.) < pending[cell] < self.voxel_mm3 for cell in changed)):
                raise ValueError('Active tissue effect or pending work left the attained working cover')
            if (not math.isclose(capacity, 2. * ACTIVE_SECONDS, rel_tol=0., abs_tol=1e-8)
                    or end - start <= 1e-9 and not removed and not changed):
                raise ValueError('Stock aspiration capacity or forward progress differs')
        return {'attained_fraction': [start, end], 'removed_cells': len(removed)}


def run(root, output, prepared):
    package, static, contract, route, opening, setup, bone = prepared
    checked_root(root, expected_uuid=UUID, required_bytes=2 * 1024**3)
    output.mkdir()
    started = monotonic()
    actions, rows = [], []
    receipt = output / 'report.json'
    report = {'contract': contract, 'status': 'in_progress', 'target_effect': False, 'withdrawn': False,
              'cold_reopen_exact': False, 'historical_prefix_actuated_from_zero': False}
    atomic_json(receipt, report)

    def call(session, operation, limit=VERIFY_SECONDS, **args):
        request = BoundedRequest(operation)
        request.deadline = started + limit
        request.check()
        return json_data(session.execute(operation, args, request, lambda *_: None))

    def command(session, name, limit=VERIFY_SECONDS, **args):
        return call(session, 'surgicalRehearsal', limit, caseHash=contract['case_hash'], command=name, **args)

    def exercise(session, checks):
        loaded = call(session, 'loadCase', path=str(package))
        if loaded['caseHash'] != contract['case_hash'] or command(session, 'inspect') is not None:
            raise ValueError('Expected no inherited replay or treatment effects')
        current = command(session, 'create')
        initial = current['view']
        if (initial['scene']['sceneId'] != checks.scene.identity or initial['removed'] or initial['history']
                or initial.get('toolMechanicsRevision') is not None
                or any(s['active'] or s['inserted'] for s in initial['slots'].values())):
            raise ValueError('Expected exact fresh production simulator scene')
        report['initial_state_id'] = initial['stateId']

        def trial(phase, action, reference=None):
            nonlocal current
            before = current['view']
            limit = EXIT_SECONDS if phase == 'reverse' else EXECUTION_SECONDS
            args = {'expectedStateId': before['stateId'], 'action': action}
            row = {'phase': phase, 'action': action, 'before_state_id': before['stateId'], 'committed': False}
            rows.append(row)
            atomic_json(output / 'actions.json', rows)
            try:
                preview = command(session, 'preview', limit - 5., **args)
                row['native_status'] = preview['status']
                row['native_reason'] = preview.get('reasonCode') or preview.get('reason')
                row['checked_effect'] = checks.verify(phase, before, preview, action, reference)
            except ValueError as error:
                row['refusal'] = str(error)
            if command(session, 'inspect', limit)['view'] != before:
                raise ValueError('Detached preview changed the originally accepted state')
            atomic_json(output / 'actions.json', rows)
            if 'refusal' in row:
                return False
            row['step_attempted'] = True
            atomic_json(output / 'actions.json', rows)
            try:
                actual = command(session, 'step', limit, **args)
            except Exception:
                row['commit_verification_unresolved'] = True
                atomic_json(output / 'actions.json', rows)
                raise
            result, after = actual['outcome'], actual['view']
            row.update(committed=result['status'] in ('accepted', 'partial'), actual_state_id=after['stateId'],
                       actual_status=result['status'], actual_reason=result.get('reasonCode'),
                       removed_ids=result.get('deltas', {}).get('removedCells', []),
                       right_pose=after['slots']['right'], thermal_boundary_C=after['metrics'].get('drillMaxTemperatureC'),
                       drill_input_energy_J=after['metrics']['drillInputEnergyJ'],
                       drilling_boundary_ledger_sha256=semantic_digest(after.get('drillingV2')))
            if row['committed']:
                actions.append(action)
            atomic_json(output / 'actions.json', rows)
            if (not row['committed'] or after != preview['view']
                    or any(preview.get(k) != value for k, value in result.items())):
                raise ValueError('Actual simulator result differs from the guarded preview')
            current = actual
            return True

        for phase, sequence in (('opening', opening), ('kerf_and_setup', setup)):
            for reference in sequence:
                if not trial(phase, reference['action'], reference):
                    report.update(stop=rows[-1]['refusal'], stopped_phase=phase)
                    return current
            report[phase + '_verified'] = True
        report['historical_prefix_actuated_from_zero'] = True
        report['fresh_cleared_bone_blockers'] = checks.ready_for_cusa(current['view'], contract['historical_guarded_bone_ids'])
        report['guarded_bone_blockers_verified_cleared'] = len(contract['historical_guarded_bone_ids'])
        report['stop'] = 'apply_cap'
        for _ in range(MAX_APPLIES):
            if not trial('aspiration', contract['aspirate']):
                report['stop'] = rows[-1]['refusal']
                break
            after = current['view']
            fraction = segment_fraction(after['slots']['right'], route)
            report['attained_fraction'] = fraction
            if after['metrics']['targetRemovedMm3'] > 0.:
                report.update(target_effect=True, target_effect_state_id=after['stateId'], stop='first_effect_at_action_boundary')
                break
            if fraction >= 1. - 1e-9:
                report['stop'] = 'endpoint_without_effect'
                break
        report['withdrawn'] = trial('reverse', contract['reverse'])
        if not report['withdrawn']:
            report['withdrawal_refusal'] = rows[-1]['refusal']
        return current

    try:
        if live_runtime() != contract['runtime']:
            raise ValueError('Runtime changed after preparation')
        case, artifacts, workspace = load_workspace(package)
        if artifacts or workspace['episode'] is not None:
            raise ValueError('Pristine case has pre-existing action history')
        checks = Checks(case, static, route, bone)
        del case, artifacts, workspace
        with TemporaryDirectory(prefix='.simulation-', dir=output) as tmp:
            scratch = Path(tmp)
            session = BridgeSession(scratch / 'transfer', run_dir=scratch / 'runs')
            try:
                current = exercise(session, checks)
                final = current['view']
                report.update(final_state_id=final['stateId'], final_metrics=final['metrics'],
                              final_slots=final['slots'], final_removed_source_ids=final['removed'],
                              final_any_tool_inserted=any(s['inserted'] for s in final['slots'].values()))
                saved = output / 'replay.ressectionlab'
                call(session, 'saveCase', path=str(saved), caseHash=contract['case_hash'])
            finally:
                session.close()
            del checks
            case, artifacts, _ = load_workspace(saved)
            record = artifacts['surgical_toolbox']
            branch = record['branches'][record['branchId']]
            if (case.semantic_hash != contract['case_hash'] or workspace_identity(package) != workspace_identity(saved)
                    or branch['actions'] != actions or branch['stateId'] != final['stateId']
                    or branch['checkpoint']['position'] != len(actions)):
                raise ValueError('Saved replay differs from source or accepted history')
            del case, artifacts, record, branch
            cold = BridgeSession(scratch / 'cold-transfer', run_dir=scratch / 'cold-runs')
            try:
                call(cold, 'loadCase', path=str(saved))
                reopened = command(cold, 'inspect')
                if reopened['view'] != final or reopened['timeline'] != current['timeline']:
                    raise ValueError('Cold production reopen changed verified state or timeline')
            finally:
                cold.close()
        for entry in contract['bindings']:
            checked_file(root, entry)
        if live_runtime() != contract['runtime']:
            raise ValueError('Pinned source or runtime implementations changed during trial')
        report.update(cold_reopen_exact=True, source_unchanged=True, replay=str(saved.relative_to(root)),
                      replay_sha256=sha256_file(saved),
                      status='target_effect_and_withdrawal' if report['target_effect'] and report['withdrawn'] else 'bounded_negative')
    except Exception as error:
        report.update(status='unverified', error=f'{type(error).__name__}: {error}')
    finally:
        temperatures = [r['thermal_boundary_C'] for r in rows if r.get('committed') and r.get('thermal_boundary_C') is not None]
        report.update(committed_action_count=len(actions), attempted_actions=len(rows), wall_seconds=monotonic() - started,
                      max_observed_action_boundary_temperature_C=max(temperatures, default=None))
        atomic_json(receipt, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--output-name', default=OUTPUT)
    parser.add_argument('--describe', action='store_true')
    args = parser.parse_args()
    if not args.output_name or Path(args.output_name).name != args.output_name or args.output_name.startswith('.'):
        parser.error('Choose one new output directory name under qc')
    root = checked_root(args.root, expected_uuid=UUID)
    prepared = prepare(root)
    if args.describe:
        print(json.dumps(prepared[2], indent=2))
        return
    report = run(root, safe_path(root, Path('qc') / args.output_name), prepared)
    print(json.dumps({key: report.get(key) for key in ('status', 'error', 'stop', 'stopped_phase', 'target_effect',
                                                       'withdrawn', 'cold_reopen_exact', 'committed_action_count')}, indent=2))
    if report['status'] == 'unverified':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
