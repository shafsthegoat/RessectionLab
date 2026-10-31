"""Independent native source-context audit, not a planner or clinical evaluator.

Reads a canonical report JSON or saved native study and a source case. Literal
CUSA/Frazier geometry and segment/orthogonal-box distances independently rebuild
every exported footprint. No planning/encounter implementation is imported.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from resectionlab.core import array_digest, semantic_digest
from resectionlab.evaluation import segment_box_distance_sq
from resectionlab.patient_anatomy import PatientImagingCollection
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_environment import load_environment
from resectionlab.surgical_task_admission import bind_admission, validate_admission
from resectionlab.workspace_bundle import load_workspace

SCHEMA = 'resectionlab.source-context-encounter.v1'
METHOD = 'unique-retained-source-cells-in-complete-nominal-axial-tool-envelope-v1'
BOUNDARY_TOLERANCE_MM = 1e-9
SCOPE = ('Whole-cell geometric near-encounter within a declared research margin; '
         'not tissue removal, mechanical injury, function or clinical deficit. '
         'Only the supplied source image grid is assessed; outside anatomy remains unknown.')


def require(ok, reason):
    if not ok:
        raise ValueError('INDEPENDENT_ANATOMICAL_ENCOUNTER: ' + reason)


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def physical_hash(engine):
    return semantic_digest({k: v for k, v in engine.state.items() if k != 'observations'})


def axis(slot):
    return Rotation.from_quat(slot['quaternion']).apply([0., 0., 1.])


def literal_capsules(slot):
    """Exact catalog geometry, deliberately independent of surgical_motion."""
    require(slot['instrument'] in ('cusa', 'frazier_suction'), 'unsupported instrument')
    params = slot.get('parameters', {})
    require(params.get('radiusMm', 1.5) == 1.5, 'catalog working radius must be 1.5mm')
    tip = np.asarray(slot['position'], float)
    direction = axis(slot)
    require(tip.shape == (3,) and np.isfinite(tip).all(), 'finite tip required')
    if slot['instrument'] == 'cusa':
        return [(tip - 40. * direction, tip, .5, 'shaft'), (tip, tip, 1.5, 'working')]
    require(params.get('depthMm', 2.) == 2., 'Frazier distal capsule must be 2mm')
    shoulder = tip - 2. * direction
    return [(tip - 40. * direction, shoulder, .5, 'shaft'), (shoulder, tip, 1.5, 'working')]


def axial_capsules(start, end):
    require(start['instrument'] == end['instrument'], 'instrument changed during a sweep')
    direction = axis(start)
    delta = np.asarray(end['position']) - start['position']
    require(np.allclose(direction, axis(end), atol=1e-10, rtol=0.) and
            np.linalg.norm(np.cross(delta, direction)) <= 1e-8,
            'only unchanged-axis collinear translation is supported')
    first, last = literal_capsules(start), literal_capsules(end)
    result = []
    for (a, b, radius, part), (c, d, other_radius, other_part) in zip(first, last):
        require(radius == other_radius and part == other_part, 'geometry changed during sweep')
        ordered = sorted((a, b, c, d), key=lambda p: float(p @ direction))
        result.append((ordered[0], ordered[-1], radius, part))
    return result


class IndependentFootprint:
    """Exact source boxes, unique retained source memberships, no engine covers."""
    def __init__(self, mask, affine, baseline_removed, limit):
        self.mask = np.asarray(mask)
        self.affine = np.asarray(affine, float)
        require(self.mask.dtype == np.bool_ and self.mask.ndim == 3 and self.mask.any(), 'nonempty boolean source mask required')
        require(self.affine.shape == (4, 4) and np.isfinite(self.affine).all() and
                np.array_equal(self.affine[3], [0., 0., 0., 1.]), 'finite homogeneous affine required')
        basis = self.affine[:3, :3]
        self.spacing = np.linalg.norm(basis, axis=0)
        require(np.all(self.spacing > 0.) and np.allclose(basis.T @ basis,
                np.diag(self.spacing ** 2), atol=1e-10, rtol=0.), 'sheared/singular source cells unsupported')
        self.rotation = basis / self.spacing
        require(type(limit) is dict and set(limit) == {'marginMm', 'maxNearMm3'}, 'exact geometric limit required')
        require(all(type(limit[k]) in (int, float) and np.isfinite(limit[k]) and 0. <= limit[k] <= cap
                for k, cap in (('marginMm', 5.), ('maxNearMm3', 1000.))), 'finite bounded research limit required')
        self.limit = {k: float(v) for k, v in limit.items()}
        self.baseline_removed = frozenset(map(int, baseline_removed))
        require(all(0 <= c < self.mask.size for c in self.baseline_removed), 'removed source cell outside grid')
        self.seen = {'shaft': set(), 'working': set()}
        self._queries = {}
        self.box_tests = 0

    def _cells(self, a, b, radius):
        key = (tuple(a), tuple(b), radius)
        if key in self._queries:
            return self._queries[key]
        origin = self.affine[:3, 3]
        a, b = (self.rotation.T @ (np.asarray(p) - origin) for p in (a, b))
        lo = np.maximum(0, np.floor((np.minimum(a, b) - radius) / self.spacing - .5).astype(int))
        hi = np.minimum(np.asarray(self.mask.shape) - 1, np.ceil((np.maximum(a, b) + radius) / self.spacing + .5).astype(int))
        found = set()
        if np.all(lo <= hi):
            region = tuple(slice(int(x), int(y) + 1) for x, y in zip(lo, hi))
            for voxel in np.argwhere(self.mask[region]) + lo:
                center = voxel * self.spacing
                self.box_tests += 1
                if segment_box_distance_sq(a, b, center - self.spacing / 2., center + self.spacing / 2.) <= (radius + BOUNDARY_TOLERANCE_MM) ** 2:
                    found.add(int(np.ravel_multi_index(voxel, self.mask.shape)))
        # This cache contains only pure source geometry; removal is filtered on
        # every use. Its lifetime is one bounded exported sequence audit.
        self._queries[key] = found
        return found

    def add(self, capsules, removed):
        removed = set(removed)
        for a, b, radius, part in capsules:
            require(part in self.seen, 'unknown instrument part')
            self.seen[part].update(self._cells(a, b, radius + self.limit['marginMm']) - removed)

    def observe(self, action, before, after, removed):
        for hand, slot in after.items():
            cover = (axial_capsules(before[hand], slot) if hand == action['slot'] and
                     action['operation'] in ('move', 'apply', 'withdraw') else literal_capsules(slot))
            self.add(cover, removed)

    def receipt(self, removed):
        flat = self.mask.ravel()
        union = sorted(self.seen['shaft'] | self.seen['working'])
        historical = sum(bool(flat[c]) for c in self.baseline_removed)
        new = set(removed) - self.baseline_removed
        voxel_mm3 = float(abs(np.linalg.det(self.affine[:3, :3])))
        volume = len(union) * voxel_mm3
        coordinates = np.asarray(np.unravel_index(np.asarray(union[:12], dtype=np.intp), self.mask.shape)).T.reshape(-1, 3)
        positions = coordinates @ self.affine[:3, :3].T + self.affine[:3, 3]
        return {'schema': SCHEMA, 'method': METHOD, **self.limit,
                'sourceContextMm3': int(flat.sum()) * voxel_mm3,
                'baselineRemovedContextMm3': historical * voxel_mm3,
                'baselineRetainedContextMm3': (int(flat.sum()) - historical) * voxel_mm3,
                'newRemovedContextMm3': sum(bool(flat[c]) for c in new) * voxel_mm3,
                'nearSourceCells': union, 'nearPositionsMm': positions.tolist(), 'nearMm3': volume,
                'shaftNearMm3': len(self.seen['shaft']) * voxel_mm3,
                'workingNearMm3': len(self.seen['working']) * voxel_mm3,
                'withinLimit': volume <= self.limit['maxNearMm3'], 'scope': SCOPE}


class CaptureEngine(SurgicalEngine):
    def clone(self):
        other = super().clone()
        other._audit_trace = list(self._audit_trace)
        return other

    def _substep(self, key, op, p, q, active, dt):
        require(not self.state['holds'] and not self.state.get('deformation') and not self.state.get('detachedFragments'),
                'geometry-changing microstep outside audit scope')
        before = copy.deepcopy(self.state['slots'])
        events = super()._substep(key, op, p, q, active, dt)
        self._audit_trace.append({'hand': key, 'operation': op, 'before': before,
                                  'after': copy.deepcopy(self.state['slots'])})
        return events


def certify_trace(action, before, after, trace):
    """Prove the accepted microsteps equal the collapsed nominal axial union."""
    require(bool(trace), 'accepted action has no captured microsteps')
    hand = action['slot']
    positions = []
    for frame in trace:
        require(frame['hand'] == hand and frame['operation'] == action['operation'], 'unexpected microstep control')
        for other in before:
            literal_capsules(frame['after'][other])
            if other != hand:
                require(np.allclose(frame['before'][other]['position'], frame['after'][other]['position'], atol=1e-10, rtol=0.) and
                        np.allclose(frame['after'][other]['position'], before[other]['position'], atol=1e-10, rtol=0.) and
                        np.allclose(axis(frame['before'][other]), axis(frame['after'][other]), atol=1e-10, rtol=0.) and
                        np.allclose(axis(frame['after'][other]), axis(before[other]), atol=1e-10, rtol=0.),
                        'stationary other hand moved')
        if action['operation'] in ('move', 'apply', 'withdraw'):
            axial_capsules(frame['before'][hand], frame['after'][hand])
            if positions:
                require(np.allclose(positions[-1], frame['before'][hand]['position'], atol=1e-9, rtol=0.), 'microstep path has a gap')
            else:
                positions.append(np.asarray(frame['before'][hand]['position']))
            positions.append(np.asarray(frame['after'][hand]['position']))
    if positions:
        start, end = before[hand], after[hand]
        axial_capsules(start, end)
        require(np.allclose(positions[0], start['position'], atol=1e-9, rtol=0.) and
                np.allclose(positions[-1], end['position'], atol=1e-9, rtol=0.), 'captured path endpoints differ')
        direction = axis(start)
        scalars = np.asarray([(p - positions[0]) @ direction for p in positions])
        differences = np.diff(scalars)
        require(np.all(differences >= -1e-9) or np.all(differences <= 1e-9), 'nonmonotone action cannot collapse to endpoint union')
        require(all(np.linalg.norm(np.cross(p - positions[0], direction)) <= 1e-8 for p in positions), 'nonaxial microstep path')
    else:
        require(action['operation'] in ('exchange', 'inspect', 'place', 'activate', 'deactivate'), 'unsupported operation')
        for frame in trace:
            for slot in before:
                require(np.allclose(frame['before'][slot]['position'], frame['after'][slot]['position'], atol=1e-10, rtol=0.),
                        'nonmoving operation moved in a microstep')


def load_source_context(engine, report):
    """Check the actual native source members and all exported binding fields."""
    contract = report['taskAdmission']
    bind_admission(engine, contract)
    validate_admission(engine, contract)
    binding = report['contextBinding']
    require(binding.get('bindingSeal') == semantic_digest({k: v for k, v in binding.items() if k != 'bindingSeal'}), 'context binding seal differs')
    collection = PatientImagingCollection(contract['verification']['nativeCollectionPath'])
    require(file_hash(collection.root / 'patient-anatomy.json') == binding['nativeManifestSha256'] == contract['verification']['nativeManifestSha256'], 'native manifest bytes differ')
    require(collection.identity() == binding['sourceCollectionHash'] == contract['sourceEvidence']['sourceCollectionHash'], 'native snapshot differs')
    matches = [r for r in collection.document['layers'] if r['id'] == binding['layerId']]
    require(len(matches) == 1, 'source context layer differs')
    row = matches[0]
    require(row['role'] == 'context' and row['kind'] == 'context' and type(row['label_value']) is int and row['label_value'] > 0 and
            row['provenance']['kind'] == 'observed_source_annotation', 'explicit supplied categorical context required')
    original = row['provenance']['source']
    require(file_hash(collection.path(original['path'])) == original['sha256'], 'original annotation bytes differ')
    require(file_hash(collection.path(row['data']['path'])) == row['data']['sha256'], 'native annotation array bytes differ')
    array = collection.array(row['data'])
    registered = collection.registration_matrix(row['source_series']) @ np.asarray(row['affine'])
    require(array.shape == engine.scene.shape and np.array_equal(registered, engine.scene.affine), 'source context is not the exact engine frame')
    require(row['review_status'] == 'reviewed' or contract['task']['allowUnreviewedEvidence'], 'unreviewed context not authorized')
    adapter = engine.scene.source['adapter']
    sibling = adapter['sourceLayers'].get(next((k for k, v in adapter['sourceLayers'].items() if v['layerId'] == binding['verifiedSiblingLayerId']), ''))
    require(sibling is not None and sibling['provenance'].get('kind') == 'observed_source_annotation' and
            sibling['sourceArray'] == row['data'] and sibling['sourceAffine'] == row['affine'] and
            sibling['provenance']['source'] == original and np.array_equal(sibling['registeredAffine'], registered), 'verified selected sibling differs')
    sibling_layer = collection.layer(sibling['layerId'])
    require(sibling_layer['source_series'] == row['source_series'], 'selected sibling source series differs')
    mask = np.asarray(array == row['label_value'])
    targets = [r['id'] for r in engine.scene.regions if r.get('target') and r['material'] == 'tumor']
    require(not np.any(mask & np.isin(engine.scene.labels, targets)), 'context overlaps owned target')
    expected_manifest = {'schema': SCHEMA, 'method': METHOD, 'maskHash': array_digest(mask),
        'affineHash': array_digest(registered), 'shape': list(mask.shape), 'role': 'supplied-categorical-context',
        'reviewStatus': 'reviewed' if row['review_status'] == 'reviewed' else 'unreviewed',
        'sampling': 'exact-identity-source-cell-membership', 'coverage': 'source-grid-only; outside-unassessed',
        'scope': SCOPE, 'evaluationReferencesExposed': False}
    expected = {'schema': SCHEMA, 'sourceCollectionHash': collection.identity(),
        'nativeManifestSha256': contract['verification']['nativeManifestSha256'], 'layerId': row['id'],
        'labelValue': row['label_value'], 'revision': row.get('revision', 0), 'sourceLabel': row.get('label'),
        'sourceArray': row['data'], 'original': original, 'registeredAffine': registered.tolist(),
        'registration': collection.document['transforms'], 'reviewStatus': row['review_status'],
        'coverage': row.get('coverage', 'unknown'), 'provenance': row['provenance'],
        'verifiedSiblingLayerId': sibling['layerId'], 'publicObservation': expected_manifest}
    expected['bindingSeal'] = semantic_digest(expected)
    require(semantic_digest(expected) == semantic_digest(binding), 'source context binding fields differ')
    require(report['publicPlan']['publicObservation']['anatomicalContext'] == expected_manifest, 'public source context manifest differs')
    return mask, registered


def replay_program(baseline, mask, affine, limit, program, exported=None):
    require(type(program) is list and len(program) <= 24, 'exported sequence exceeds bounded action family')
    require(all(type(a.get('duration')) in (int, float) and np.isfinite(a['duration']) and a['duration'] > 0.
                for a in program) and sum(a['duration'] for a in program) <= 10. + 1e-10,
            'exported sequence exceeds bounded duration')
    trial = baseline.clone()
    trial.__class__ = CaptureEngine
    trial._audit_trace = []
    tracker = IndependentFootprint(mask, affine, baseline.state['removed'], limit)
    base_removed = set(baseline.state['removed'])
    prefixes = []
    for index, action in enumerate(program):
        require(type(action) is dict and action.get('operation') in ('exchange', 'inspect', 'place', 'move', 'apply', 'activate', 'deactivate', 'withdraw') and not action.get('path'), 'unsupported exported action/path')
        before = copy.deepcopy(trial.state['slots'])
        trial._audit_trace = []
        result = trial.step(copy.deepcopy(action))
        partial = (action['slot'] == 'left' and action['operation'] == 'apply' and result['status'] == 'partial' and
                   result.get('reasonCode') == 'TISSUE_ASPIRATION_IN_PROGRESS' and abs(result['elapsedSeconds'] - action['duration']) <= 1e-8)
        require(result['status'] == 'accepted' or partial, 'native action refused/incomplete')
        certify_trace(action, before, trial.state['slots'], trial._audit_trace)
        tracker.observe(action, before, trial.state['slots'], trial.state['removed'])
        receipt = tracker.receipt(trial.state['removed'])
        require(receipt['newRemovedContextMm3'] == 0., 'new supplied context was removed')
        if exported is not None:
            require(physical_hash(trial) == exported['physicalStateHashes'][index], 'native physical prefix differs')
            actual = {k: result.get(k) for k in ('status', 'reason', 'completedFraction', 'elapsedSeconds')}
            require(semantic_digest(actual) == semantic_digest(exported['actionReceipts'][index]), 'native action receipt differs')
            require(semantic_digest(receipt) == exported['contextPrefixHashes'][index], 'independent anatomical prefix differs')
            for phase in exported['phases']:
                if phase['endAction'] == index + 1:
                    require(phase['physicalStateHash'] == physical_hash(trial) and
                            phase['contextHash'] == semantic_digest(receipt), 'exported phase endpoint differs')
        prefixes.append(semantic_digest(receipt))
    targets = {r['id'] for r in baseline.scene.regions if r.get('target') and r['material'] == 'tumor'}
    require(all(int(baseline.scene.labels.flat[c]) in targets for c in set(trial.state['removed']) - base_removed), 'new non-target removal')
    return trial, tracker, prefixes


def verify(report, source, *, branch=None, revision=1):
    source = Path(source).resolve()
    require(source.is_relative_to(ROOT / 'testcases'), 'source must stay in ignored testcases')
    source_sha = file_hash(source)
    _, artifacts, _ = load_workspace(source)
    record = artifacts['surgical_toolbox']
    saved_item = None
    if not isinstance(report, dict):
        report_path = Path(report)
        if report_path.suffix == '.ressectionlab':
            _, saved, _ = load_workspace(report_path)
            record = saved['surgical_toolbox']
            history = record.get('strategyComparisons', [])
            require(type(revision) is int and 1 <= revision <= len(history), 'choose a saved study revision')
            saved_item = history[revision - 1]
            report = saved_item['report']
            branch = branch or saved_item['sourceBranchId']
        else:
            report = json.loads(report_path.read_text())
    require(report.get('schema') == 'resectionlab.two-tool-comparison.v2', 'only source-context V2 reports are supported')
    require(report['reportSeal'] == semantic_digest({k: v for k, v in report.items() if k != 'reportSeal'}), 'report seal differs')
    plan = report['publicPlan']
    require(plan['publicPlanSeal'] == semantic_digest({k: v for k, v in plan.items() if k != 'publicPlanSeal'}), 'public plan seal differs')
    scene = load_environment(source)
    snapshot = record['branches'][branch or record['branchId']]
    if saved_item:
        require(semantic_digest(snapshot) == saved_item['sourceSnapshotHash'], 'saved study baseline differs')
    count = report['prefixActions']
    require(type(count) is int and 0 <= count <= len(snapshot['actions']), 'invalid source prefix')
    options = {}
    if snapshot.get('protectionContract'):
        from resectionlab.surgical_rl_protection import restore_protection_contract
        protection, config = restore_protection_contract(scene, snapshot['protectionContract'])
        options = {'protection': protection, 'guard_config': config}
    baseline = SurgicalEngine(scene, **options)
    for action in snapshot['actions'][:count]:
        require(baseline.step(copy.deepcopy(action))['status'] != 'rejected', 'source replay refused')
    require(scene.identity == report['sourceSceneId'] and baseline.state_id == report['startStateId'] and
            semantic_digest(baseline.snapshot()) == report['baselineSnapshotHash'] and
            semantic_digest(baseline.actions) == report['baselineActionHash'] and
            semantic_digest(baseline.history) == report['baselineHistoryHash'], 'native baseline binding differs')
    require(set(scene.properties) <= {'tissuePhysicsV2', 'cusaV3'} and 'tissuePhysicsV2' in scene.properties and
            baseline.state.get('toolMechanicsRevision') == 2 and baseline.state.get('physicalClockRevision') == 1, 'unsupported model/profile')
    require(not baseline.state['holds'] and not baseline.state.get('deformation') and not baseline.state.get('detachedFragments') and
            'patientPosition' not in baseline.state and all(not s['inserted'] and not s['active'] and not s['grip'] for s in baseline.state['slots'].values()), 'inactive released undeformed baseline required')
    require(not any(baseline.state.get('deformationModel', {}).get(k) for k in ('active', 'anchors', 'velocities')), 'pending mechanics unsupported')
    mask, affine = load_source_context(baseline, report)
    limit = plan['constraints']['encounter']
    rows = []
    require(type(plan['alternatives']) is list and 1 <= len(plan['alternatives']) <= 3 and
            len({a['id'] for a in plan['alternatives']}) == len(plan['alternatives']) and
            all(a['id'] in ('short', 'long', 'stop') for a in plan['alternatives']), 'exported alternative family differs')
    for alternative in plan['alternatives']:
        program = alternative['actionProgram']
        require(len(program) == len(alternative['physicalStateHashes']) == len(alternative['contextPrefixHashes']) and
                (not program or len(program) == len(alternative['actionReceipts'])), 'exported prefix count differs')
        native, tracker, _ = replay_program(baseline, mask, affine, limit, program, alternative)
        receipt = tracker.receipt(native.state['removed'])
        require(semantic_digest(receipt) == semantic_digest(alternative['anatomicalEncounter']) and
                alternative['feasible'] is receipt['withinLimit'], 'independent final encounter differs')
        require(not native.state['holds'] and all(not s['active'] and not s['inserted'] and not s['grip'] for s in native.state['slots'].values()), 'native withdrawal incomplete')
        require(SurgicalEngine.restore(scene, native.snapshot()).snapshot() == native.snapshot(), 'cold native replay differs')
        rows.append({'id': alternative['id'], 'passed': True, 'nativeActions': len(program),
                     'boxTests': tracker.box_tests, 'anatomicalEncounter': receipt, 'exactColdReplay': True,
                     'finalStateId': native.state_id})
    routes = {r['id']: r for r in plan['approaches']['routes']}
    refusals = []
    deferred = []
    for index, rejection in enumerate(plan['rejections']):
        if rejection.get('reason') != 'ANATOMICAL_CONTEXT_LIMIT:primary-access':
            deferred.append({'index': index, 'reason': rejection.get('reason')})
            continue
        route = routes[rejection['routeId']]
        native, tracker, _ = replay_program(baseline, mask, affine, limit, route['actionProgram'])
        receipt = tracker.receipt(native.state['removed'])
        require(physical_hash(native) == route['publicExecution']['approachPhysicalStateHash'], 'primary-access physical endpoint differs')
        require(not receipt['withinLimit'] and semantic_digest(receipt) == semantic_digest(rejection['anatomicalEncounter']), 'independent primary-access refusal differs')
        refusals.append({'index': index, 'routeId': route['id'], 'passed': True, 'anatomicalEncounter': receipt})
    require(file_hash(source) == source_sha, 'source case changed during audit')
    return {'schema': 'resectionlab.independent-anatomical-encounter-audit.v1', 'passed': True,
            'source': str(source), 'sourceSha256': source_sha, 'sourceUnchanged': True,
            'sourceContextBindingVerified': True, 'baselineStateId': baseline.state_id,
            'publicPlanSeal': plan['publicPlanSeal'], 'alternatives': rows, 'primaryAccessRefusals': refusals,
            'otherRejectionsNotIndependentlyAudited': deferred,
            'scope': 'Exported native programs and recorded primary-access refusals only; finite-family coverage and other refusals remain the canonical validator responsibility. No clinical inference.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path, help='canonical report JSON or saved native workspace')
    parser.add_argument('source', type=Path, help='source case containing the native scene')
    parser.add_argument('--branch')
    parser.add_argument('--revision', type=int, default=1)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    require(args.output.resolve().is_relative_to(ROOT / 'testcases') and not args.output.exists(), 'preserve evidence and keep output in testcases')
    result = verify(args.report, args.source, branch=args.branch, revision=args.revision)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'passed': result['passed'], 'alternatives': len(result['alternatives']), 'primaryAccessRefusals': len(result['primaryAccessRefusals'])}))
