#!/usr/bin/env python3
"""Execute one independently approved, exact-old-request software regression.

Importing this module and its validators performs no simulator work. Execution
requires an exact protocol/decision/driver binding and imports the unchanged
production backend only from its validated external snapshot. There is no route
generator, retry, reset, checkpoint transplant, or automatic approval.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
import zipfile

REPO = Path(__file__).resolve().parents[1]
OPS = frozenset({'configure', 'exchange', 'place', 'apply', 'withdraw'})
AUTHORITY_KEYS = frozenset({
    'schema', 'decision', 'reviewer', 'subject', 'protocol_sha256', 'source_sha256',
    'old_history_sha256', 'initial_state_id', 'old_action_list_sha256',
    'allowed_removal_ids_sha256', 'backend_manifest_sha256', 'executor_sha256',
    'maximum_main_requests', 'maximum_cleanup_commits', 'maximum_new_commits',
    'maximum_total_declared_action_seconds', 'maximum_brain_removed_mm3',
    'maximum_bone_removed_mm3', 'maximum_target_removed_mm3',
    'maximum_vessel_removed_mm3', 'physical_execution_approved',
})


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def token(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,127}', value):
        raise ValueError('Plain immutable output token required')
    return value


def contained(root, value):
    root = Path(root).resolve(strict=True)
    path = Path(value)
    path = (path if path.is_absolute() else root / path).resolve(strict=True)
    if not path.is_relative_to(root):
        raise ValueError('Input outside guarded data root')
    return path


def read_json(path):
    path = Path(path)
    if path.stat().st_size > 32 * 1024 * 1024:
        raise ValueError('Bounded JSON input exceeded')
    return json.loads(path.read_text())


def finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def validate_requests(actions, expected):
    if not isinstance(actions, list) or not actions or actions != expected:
        raise ValueError('Exact unchanged literal request list required')
    for action in actions:
        if (not isinstance(action, dict) or action.get('operation') not in OPS
                or action.get('slot') not in ('left', 'right')
                or not finite_number(action.get('duration', 1.))
                or not 0 < action.get('duration', 1.) <= 30
                or len(action.get('path', [])) > 1):
            raise ValueError('Unsupported frozen request contract')


def validate_authority(protocol, decision, *, protocol_hash, driver_hash):
    """Exact independent artifact binding; no inferred or inherited approval."""
    if set(decision) != AUTHORITY_KEYS:
        raise ValueError('Exact authority schema required')
    expected = {
        'schema': 'independent-exact-old-request-regression-decision-v1',
        'decision': 'accepted_for_one_pass_exact_old_request_regression',
        'reviewer': '/root/integration_review', 'subject': protocol['subject'],
        'protocol_sha256': protocol_hash, 'executor_sha256': driver_hash,
        'source_sha256': protocol['source']['sha256'],
        'old_history_sha256': protocol['old_reference']['history_sha256'],
        'initial_state_id': protocol['fresh_initialization']['initial_state_id'],
        'old_action_list_sha256': protocol['literal_requests']['sha256'],
        'allowed_removal_ids_sha256': protocol['allowed_removals']['sha256'],
        'backend_manifest_sha256': protocol['backend']['manifest_sha256'],
        'maximum_main_requests': protocol['budgets']['maximum_main_requests'],
        'maximum_cleanup_commits': 1,
        'maximum_new_commits': protocol['budgets']['maximum_new_commits'],
        'maximum_total_declared_action_seconds': protocol['budgets']['maximum_total_declared_action_seconds'],
        'maximum_brain_removed_mm3': protocol['allowed_removals']['brain_volume_cap_mm3_exact_historical'],
        'maximum_bone_removed_mm3': protocol['allowed_removals']['bone_volume_cap_mm3_exact_historical'],
        'maximum_target_removed_mm3': 0., 'maximum_vessel_removed_mm3': 0.,
        'physical_execution_approved': True,
    }
    if any(decision[key] != value for key, value in expected.items()):
        raise ValueError('Independent decision differs from exact protocol/driver/caps')
    if decision['physical_execution_approved'] is not True:
        raise ValueError('Physical execution has not been approved')
    for key, value in decision.items():
        if key.startswith('maximum_') and (not finite_number(value) or value < 0):
            raise ValueError('Authority limits must be finite numbers, not booleans')
    if any(type(decision[key]) is not int for key in ('maximum_main_requests', 'maximum_cleanup_commits', 'maximum_new_commits')):
        raise ValueError('Authority action limits must be integers')


def validate_limits(protocol, actions):
    budget = protocol['budgets']; n = len(actions)
    declared = sum(a.get('duration', 1.) for a in actions)
    withdrawals = [a.get('duration', 1.) for a in actions if a['operation'] == 'withdraw']
    if (not withdrawals or budget['maximum_main_requests'] != n
            or budget['maximum_cleanup_commits'] != 1
            or budget['maximum_new_commits'] != n + 1
            or budget['maximum_public_previews'] != n + 1
            or budget['maximum_main_declared_material_apply_seconds'] != sum(a.get('duration', 1.) for a in actions if a['operation'] == 'apply')
            or budget['maximum_additional_cleanup_seconds'] != max(withdrawals)
            or budget['maximum_total_declared_action_seconds'] != declared + max(withdrawals)):
        raise ValueError('No expanded, repeated or reset request budget allowed')


def metrics_of(response):
    return response.get('view', {}).get('metrics', response.get('metrics', {}))


def receipt_of(response):
    return response.get('outcome', response)


def allowed_status(action, instrument, response):
    receipt = receipt_of(response)
    if receipt.get('status') == 'accepted' and receipt.get('reason') is None:
        return True
    return (action['operation'] == 'apply' and receipt.get('status') == 'partial'
            and receipt.get('reason') == {
                'cranial_drill': 'BONE_CUTTING_IN_PROGRESS',
                'cusa': 'TISSUE_ASPIRATION_IN_PROGRESS',
            }.get(instrument))


def preview_predicate(protocol, allowed, action, instrument, before, preview, removed, voxel_volume=None):
    """Cumulative task limits supplement, and never replace, native guards."""
    if not allowed_status(action, instrument, preview):
        return False, 'HARD_NATIVE_REFUSAL'
    metrics = metrics_of(preview)
    required = {'removedByLabelMm3', 'targetRemovedMm3', 'vesselEncounters', 'elapsedSeconds', 'travelMm'}
    if not required <= set(metrics) or not isinstance(metrics['removedByLabelMm3'], dict):
        return False, 'MISSING_PREVIEW_METRICS'
    receipt = receipt_of(preview)
    delta = receipt.get('deltas', {}).get('removedCells')
    if (not isinstance(delta, list) or any(type(i) is not int or i < 0 for i in delta)
            or len(delta) != len(set(delta)) or set(delta) & set(removed)):
        return False, 'INVALID_NEW_REMOVAL_IDS'
    authority = {row['cell_id']: row for row in allowed['source_cells']}
    cumulative = set(removed) | set(delta)
    if not cumulative <= set(authority):
        return False, 'UNALLOWED_SOURCE_REMOVAL_ID'
    if action['operation'] == 'apply':
        material = {'cranial_drill': 'bone', 'cusa': 'soft_tissue'}.get(instrument)
        if material is None or any(authority[i]['original_material'] != material for i in delta):
            return False, 'WRONG_TOOL_ORIGINAL_MATERIAL_REMOVAL'
    old_vol = before.get('removedByLabelMm3', {}); new_vol = metrics.get('removedByLabelMm3', {})
    for key in set(old_vol) | set(new_vol):
        value, old = new_vol.get(key, 0.), old_vol.get(key, 0.)
        if not finite_number(value) or value < 0 or value < old:
            return False, 'NONMONOTONIC_OR_INVALID_REMOVAL_METRIC'
        if key not in ('brain', 'estimated_skull') and value != 0:
            return False, 'UNEXPECTED_MATERIAL_REMOVAL'
    caps = protocol['allowed_removals']
    if (new_vol.get('brain', 0.) > caps['brain_volume_cap_mm3_exact_historical']
            or new_vol.get('estimated_skull', 0.) > caps['bone_volume_cap_mm3_exact_historical']):
        return False, 'CUMULATIVE_MATERIAL_VOLUME_CAP'
    if voxel_volume is not None:
        expected_volume = {}
        for cell in sorted(cumulative):
            region = authority[cell]['original_region']
            expected_volume[region] = expected_volume.get(region, 0.) + voxel_volume
        if any(new_vol.get(key, 0.) != expected_volume.get(key, 0.) for key in set(new_vol) | set(expected_volume)):
            return False, 'REMOVAL_METRIC_SOURCE_ID_MISMATCH'
    for material, limit in caps['counts_are_also_hard_caps'].items():
        if sum(authority[i]['original_material'] == material for i in cumulative) > limit:
            return False, 'CUMULATIVE_MATERIAL_CELL_CAP'
    zero_fields = ('targetRemovedMm3', 'vesselEncounters', 'activeVesselWounds',
                   'severedVesselConnections', 'modeledVesselFluidGeneratedMm3')
    if any(not finite_number(metrics.get(key, 0.)) or metrics.get(key, 0.) != 0 for key in zero_fields):
        return False, 'TARGET_OR_VESSEL_EFFECT'
    vessel = metrics.get('vesselPhysicsV2', {})
    if any(vessel.get(key, 0.) != 0 for key in ('woundCount', 'damagedFaceCount', 'woundAreaMm2', 'activeWounds', 'generatedMm3', 'emittedMm3')):
        return False, 'VESSEL_PHYSICS_EFFECT'
    for key in ('brainMaxDisplacementMm', 'brainElasticEnergyNmm'):
        if metrics.get(key, 0.) != 0:
            return False, 'UNSUPPORTED_DEFORMATION_FOR_EXACT_TRACE'
    monotonic = ('elapsedSeconds', 'travelMm', 'cusaAspiratedSolidMm3', 'cusaInputCapacityMm3',
                 'drillInputEnergyJ', 'drillCuttingWorkJ', 'drillRemovedVolumeMm3')
    for key in monotonic:
        value = metrics.get(key, 0.)
        if not finite_number(value) or value < 0 or value < before.get(key, 0.):
            return False, 'INVALID_OR_DECREASING_PHYSICAL_METRIC'
    if action['operation'] != 'apply':
        if delta or any(new_vol.get(k, 0.) != old_vol.get(k, 0.) for k in set(old_vol) | set(new_vol)):
            return False, 'NONMATERIAL_REQUEST_REMOVED_TISSUE'
        for key in ('cusaAspiratedSolidMm3', 'cusaInputCapacityMm3', 'drillInputEnergyJ', 'drillCuttingWorkJ'):
            if metrics.get(key, 0.) != before.get(key, 0.):
                return False, 'NONMATERIAL_REQUEST_PERFORMED_WORK'
    return True, None


def capture_predicate(instrument, capture_ids, target_ids, vessel_ids, allowed):
    """Pending capture/work is restricted even before any voxel is removed."""
    if target_ids or vessel_ids:
        return False, 'TARGET_OR_VESSEL_CONTACT'
    material = {'cusa': 'soft_tissue', 'cranial_drill': 'bone'}.get(instrument)
    if material is None or not set(capture_ids) <= set(allowed['by_material'].get(material, [])):
        return False, 'UNALLOWED_BRAIN_OR_BONE_WORK_SOURCE_ID'
    return True, None


def select_cleanup(actions, failed_index, slots, open_hands, failed_hand):
    """Only the earliest remaining literal withdrawal for the actual posed hand."""
    intrusive = [hand for hand, slot in slots.items()
                 if slot.get('inserted') or slot.get('active') or slot.get('grip')]
    if len(intrusive) > 1:
        return None, 'MULTIPLE_HANDS_REQUIRE_CLEANUP'
    hand = intrusive[0] if intrusive else failed_hand if failed_hand in open_hands else None
    if hand is None:
        return None, 'NO_POSED_HAND_REQUIRES_CLEANUP'
    for index in range(failed_index + 1, len(actions)):
        if actions[index]['slot'] == hand and actions[index]['operation'] == 'withdraw':
            return index, None
    return None, 'NO_REMAINING_LITERAL_WITHDRAWAL'


def claim_once(path, payload):
    """A decision cannot be rerun with a new name to reset its finite budget."""
    with Path(path).open('x') as stream:
        json.dump(payload, stream, sort_keys=True, indent=2)
        stream.write('\n')


def reviewed_appledouble(path, expected, backend):
    """Recognize paired ExFAT metadata, never ignore an extra Python source."""
    path = Path(path)
    if (not path.name.startswith('._') or not path.resolve().is_relative_to(backend)
            or (path.parent / path.name[2:]).resolve() not in expected
            or not 26 <= path.stat().st_size <= 1024 * 1024):
        return False
    raw = path.read_bytes()
    if raw[:4] != b'\x00\x05\x16\x07' or raw[4:8] not in (b'\x00\x01\x00\x00', b'\x00\x02\x00\x00'):
        return False
    count = int.from_bytes(raw[24:26], 'big')
    if not 1 <= count <= 64 or 26 + 12 * count > len(raw):
        return False
    for offset in range(26, 26 + 12 * count, 12):
        start = int.from_bytes(raw[offset + 4:offset + 8], 'big')
        length = int.from_bytes(raw[offset + 8:offset + 12], 'big')
        if start < 26 + 12 * count or start + length > len(raw):
            return False
    return True


def verify_backend(root, protocol, override=None):
    manifest_path = contained(root, protocol['backend']['manifest_path'])
    if digest(manifest_path) != protocol['backend']['manifest_sha256']:
        raise ValueError('Reviewed backend manifest changed')
    manifest = read_json(manifest_path)
    backend = contained(root, protocol['backend']['snapshot_root'])
    if override is not None and contained(root, override) != backend:
        raise ValueError('Backend override must be the exact reviewed snapshot')
    expected = set()
    for row in manifest['files']:
        path = contained(backend, row['relative_path'])
        if digest(path) != row['sha256'] or path.stat().st_size != row['bytes']:
            raise ValueError('Reviewed production backend source/resource changed')
        expected.add(path)
    actual = set(); metadata = []
    for path in (backend / 'resectionlab').rglob('*'):
        if not path.is_file() or path.suffix not in ('.py', '.c', '.json'):
            continue
        if reviewed_appledouble(path, expected, backend):
            metadata.append({'path': str(path.relative_to(backend)), 'sha256': digest(path), 'role': 'validated paired AppleDouble filesystem metadata, not importable source'})
        else:
            actual.add(path.resolve())
    if actual != expected:
        raise ValueError('Unexpected or missing backend code/resource')
    for key, value in protocol['backend']['required_optional_native_environment'].items():
        if os.environ.get(key) != value:
            raise ValueError('Reviewed optional native selection changed')
    versions = {'python': platform.python_version(), **{
        name: importlib.metadata.version(name) for name in ('numpy', 'scipy', 'nibabel')}}
    if versions != protocol['backend']['required_runtime_versions']:
        raise ValueError('Reviewed installed runtime versions changed')
    return backend, {**manifest, 'validated_filesystem_metadata': metadata}, versions


def bootstrap(backend):
    existing = [name for name in sys.modules if name == 'resectionlab' or name.startswith('resectionlab.')]
    if existing:
        raise ValueError('Production package was imported before frozen backend validation')
    sys.dont_write_bytecode = True
    checkout_src = (REPO / 'src').resolve()
    sys.path[:] = [str(backend), *[value for value in sys.path if Path(value or '.').resolve() != checkout_src]]


def git_binding():
    def git(*args):
        return subprocess.check_output(['git', *args], cwd=REPO, stderr=subprocess.STDOUT)
    dirty = []
    paths = set(git('diff', '--name-only', '-z').split(b'\0') + git('ls-files', '--others', '--exclude-standard', '-z').split(b'\0'))
    for raw in sorted(paths):
        if raw:
            path = REPO / os.fsdecode(raw)
            dirty.append({'path': str(path.relative_to(REPO)), 'sha256': digest(path) if path.is_file() else None})
    return {'head': git('rev-parse', 'HEAD').decode().strip(),
            'branch': git('branch', '--show-current').decode().strip(),
            'porcelain': git('status', '--porcelain=v1').decode(), 'dirty_manifest': dirty,
            'interpretation': 'Shared checkout binding only; execution imports the reviewed external backend snapshot, not a clean milestone branch.'}


def historical_success_count(receipt, scheduled_count):
    """Read an existing hard-stop receipt without inventing its rejected motion."""
    if (receipt.get('status') == 'partial'
            and receipt.get('reason') not in (None, 'BONE_CUTTING_IN_PROGRESS', 'TISSUE_ASPIRATION_IN_PROGRESS')):
        fraction = receipt.get('completedFraction')
        if not finite_number(fraction) or not 0 <= fraction < 1:
            raise ValueError('Invalid historical hard-stop fraction')
        done = round(fraction * scheduled_count)
        if not 0 <= done < scheduled_count or abs(done / scheduled_count - fraction) > 1e-10:
            raise ValueError('Historical fraction does not bind whole successful microsteps')
        return done
    return scheduled_count


def contact_trace(scene, action, before_slots, preview, removed, allowed, *, historical_receipt=False):
    """Read-only reconstruction from actual public receipts, never effects.

    Instantaneous capture IDs supplement removal/metric gates, so pending CUSA
    work cannot consume newly added brain/target silently. An unsupported trace
    is withheld. Source occupancy is local derived receipt bookkeeping only.
    historical_receipt is only for read-only checks of existing hard-stop
    receipts; it checks their successful prefix. The executor never enables it
    and still rejects every hard preview before calling this function.
    """
    import numpy as np
    from scipy.spatial.transform import Rotation, Slerp
    from resectionlab.geometry import GeometryScene, capsule_voxel_indices
    from resectionlab.surgical_tools import TOOLS
    from resectionlab.surgical_motion import envelopes, swept_envelopes
    from resectionlab.surgical_cusa_v3 import contact_limited_pose as cusa_limit
    from resectionlab.surgical_drilling_v2 import contact_limited_pose as drill_limit

    key = action['slot']; slot = before_slots[key]; op = action['operation']
    instrument = action.get('instrument', slot['instrument']) if op == 'exchange' else slot['instrument']
    if any(s.get('active') or s.get('grip') for h, s in before_slots.items() if h != key):
        return False, 'UNSUPPORTED_OTHER_ACTIVE_HAND_TRACE', []
    tool = TOOLS[instrument]
    parameters = {'rpm': 40000., 'forceN': 1., 'duty': 1., 'radiusMm': tool['radiusMm'],
                  'depthMm': 2., 'flowMm3PerSecond': 20., 'power': 1., 'jawOpeningMm': 2.,
                  **({} if op == 'exchange' else slot.get('parameters', {})),
                  **action.get('parameters', {}), 'contactGeometryRevision': 2}
    grid = GeometryScene(np.zeros(scene.shape, bool), scene.affine)
    by_label = {r['id']: r for r in scene.regions}; labels = scene.labels.ravel()

    class SourceReceiptGeometry:
        def __init__(self):
            self.scene = scene; self.by_label = by_label
            self.state = {'removed': set(removed), 'toolMechanicsRevision': 2, 'holds': {}, 'deformation': {}}

        def touched(self, a, b, radius):
            indices = capsule_voxel_indices(grid, a, b, radius)
            ids = np.ravel_multi_index(indices.T, scene.shape) if len(indices) else []
            return [int(i) for i in ids if labels[i] and int(i) not in self.state['removed']]

    proxy = SourceReceiptGeometry(); rows = []

    def check_cover(start, end, *, capture=None):
        cover = swept_envelopes(tool, parameters, start, end)
        contacts = {'shaft': set(), 'working': set()}
        for a, b, radius, part in cover:
            contacts[part].update(proxy.touched(a, b, radius))
        all_ids = contacts['shaft'] | contacts['working']
        targets = sorted(i for i in all_ids if by_label[int(labels[i])].get('target'))
        vessels = sorted(i for i in all_ids if by_label[int(labels[i])]['material'] == 'vessel')
        row = {'from': copy.deepcopy(start), 'contact_limited_candidate': copy.deepcopy(end),
               'envelopes': [{'a': a.tolist(), 'b': b.tolist(), 'radiusMm': float(r), 'part': p} for a, b, r, p in cover],
               'target_contact_source_ids': targets, 'vessel_contact_source_ids': vessels,
               'capture_or_bone_work_source_ids': sorted(capture or [])}
        rows.append(row)
        if targets or vessels:
            return False, 'TARGET_OR_VESSEL_CONTACT'
        if op == 'apply':
            return capture_predicate(instrument, capture or [], targets, vessels, allowed)
        return True, None

    if op == 'configure':
        return True, None, []  # Explicit nonphysical setup; initial unposed slots are not an ingress.
    if op in ('place', 'exchange'):
        pose = action['pose'] if op == 'place' else slot
        ok, why = check_cover(pose, pose)
        return ok, why, rows
    points = action.get('path') or ([action['pose']] if 'pose' in action else [])
    if len(points) != 1:
        return False, 'UNSUPPORTED_LITERAL_PATH_TRACE', rows
    start = {'position': slot['position'], 'quaternion': slot['quaternion']}; end = points[0]
    a, b = np.asarray(start['position']), np.asarray(end['position'])
    rotations = Rotation.from_quat([start['quaternion'], end['quaternion']])
    angle = (rotations[1] * rotations[0].inv()).magnitude()
    n = max(1, math.ceil((np.linalg.norm(b - a) + 40 * angle) / min(.4, float(scene.spacing.min()) / 3)), math.ceil(action.get('duration', 1.) / .1))
    interpolation = Slerp([0., 1.], rotations); current = start
    events = [event for event in receipt_of(preview).get('events', [])
              if event.get('kind') == ('cusa_contact' if instrument == 'cusa' else 'drill_work')]
    successful = historical_success_count(receipt_of(preview), n) if historical_receipt else n
    if successful < n:
        fraction = (successful + 1) / n
        rejected_requested = (1 - fraction) * a + fraction * b
        recorded = receipt_of(preview).get('contactPosition')
        if recorded is None or not np.allclose(rejected_requested, recorded, atol=1e-9, rtol=0):
            return False, 'HISTORICAL_REJECTED_REQUESTED_SAMPLE_MISMATCH', rows
    cursor = 0
    for index in range(successful):
        t = (index + 1) / n; requested = (1 - t) * a + t * b
        position = requested; capture = []
        if op == 'apply':
            if instrument == 'cusa':
                position, capture = cusa_limit(proxy, current['position'], requested, parameters['radiusMm'])
            elif instrument == 'cranial_drill':
                position, _ = drill_limit(proxy, current['position'], requested, parameters['radiusMm'])
                capture = [i for i in proxy.touched(current['position'], position, parameters['radiusMm'])
                           if by_label[int(labels[i])]['material'] == 'bone']
            else:
                return False, 'UNSUPPORTED_STOCK_MATERIAL_TRACE', rows
        candidate = {'position': np.asarray(position).tolist(), 'quaternion': interpolation(t).as_quat().tolist()}
        ok, why = check_cover(current, candidate, capture=capture)
        rows[-1].update(sample_index=index, requested_position=requested.tolist(),
                        scheduled_sample_count=n, historical_successful_sample_count=successful)
        if not ok:
            return ok, why, rows
        if op == 'apply' and (instrument == 'cusa' or capture):
            if cursor >= len(events):
                return False, 'MISSING_NATIVE_WORK_RECEIPT_FOR_TRACE', rows
            event = events[cursor]; cursor += 1
            proxy.state['removed'].update(event.get('removedCells', []))
        current = candidate
    if cursor != len(events):
        return False, 'EXCESS_NATIVE_WORK_RECEIPTS_FOR_TRACE', rows
    actual = preview['view']['slots'][key]
    if (not np.allclose(current['position'], actual['position'], atol=1e-8, rtol=0)
            or abs(float(np.dot(current['quaternion'], actual['quaternion']))) < 1 - 1e-12):
        return False, 'ACTUAL_REACHED_POSE_TRACE_MISMATCH', rows
    return True, None, rows


def execute(root, protocol_path, decision_path, name, backend_override=None):
    """No caller can substitute actions or supply a source checkpoint."""
    token(name); root = Path(root).resolve(strict=True)
    protocol_path, decision_path = contained(root, protocol_path), contained(root, decision_path)
    protocol, decision = read_json(protocol_path), read_json(decision_path)
    driver_hash, protocol_hash, decision_hash = digest(__file__), digest(protocol_path), digest(decision_path)
    validate_authority(protocol, decision, protocol_hash=protocol_hash, driver_hash=driver_hash)
    backend, manifest, versions = verify_backend(root, protocol, backend_override)
    bootstrap(backend)
    from resectionlab.data_root import checked_root, safe_path, atomic_json
    root = checked_root(root, required_bytes=2 * 1024 ** 3)
    runtime_cache = safe_path(root, 'integrated-v1/regressions/runtime-cache/' + name)
    for key, suffix in (('TMPDIR', 'tmp'), ('XDG_CACHE_HOME', 'xdg'),
                        ('MPLCONFIGDIR', 'matplotlib'), ('NUMBA_CACHE_DIR', 'numba'), ('TORCH_HOME', 'torch')):
        path = runtime_cache / suffix; path.mkdir(parents=True, exist_ok=True); os.environ[key] = str(path)
    import tempfile
    tempfile.tempdir = os.environ['TMPDIR']
    from resectionlab.core import semantic_digest, array_digest
    from resectionlab.workspace_bundle import load_workspace
    from resectionlab.surgical_environment import standardize, default_target_names
    from resectionlab.surgical_scene import from_case
    from resectionlab.desktop_bridge import BridgeSession, _Request
    from resectionlab.surgical_idle_native import strict_live_identity as idle_identity
    from resectionlab.surgical_prefix_json_native import strict_live_identity as prefix_identity
    native_identity = {'idle_thermal': idle_identity(), 'prefix_json': prefix_identity(full=True)}
    # These sealed protocols explicitly selected unconfigured Python kernels.
    # Never compile a missing cache or silently change that fixed selection.
    if any(identity.get('mode') != 'python' or identity.get('selection') != 'unconfigured-fixed-fallback'
           for identity in native_identity.values()):
        raise ValueError('Selected native artifact was not sealed for these protocols')
    if any(not Path(module.__file__).resolve().is_relative_to(backend) for key, module in sys.modules.items()
           if key.startswith('resectionlab') and getattr(module, '__file__', None)):
        raise ValueError('Production imports escaped the frozen reviewed backend')
    bindings = [(protocol['source']['path'], protocol['source']['sha256']),
                (protocol['old_reference']['history_path'], protocol['old_reference']['history_sha256']),
                (protocol['literal_requests']['path'], protocol['literal_requests']['sha256']),
                (protocol['allowed_removals']['path'], protocol['allowed_removals']['sha256']),
                (protocol['fresh_initialization']['fresh_zero_snapshot_path'], protocol['fresh_initialization']['fresh_zero_snapshot_sha256']),
                (protocol['backend']['authority_schema_path'], protocol['backend']['authority_schema_sha256']),
                (protocol['source']['native_QC_path'], protocol['source']['native_QC_sha256']),
                (protocol['source']['native_geometry_acceptance_path'], protocol['source']['native_geometry_acceptance_sha256'])]
    for value, expected in bindings:
        if digest(contained(root, value)) != expected:
            raise ValueError('Frozen source/authority/reference input changed: ' + value)
    actions = read_json(contained(root, protocol['literal_requests']['path']))
    allowed = read_json(contained(root, protocol['allowed_removals']['path']))
    validate_requests(actions, copy.deepcopy(actions)); validate_limits(protocol, actions)
    if semantic_digest(actions) != protocol['literal_requests']['semantic_hash']:
        raise ValueError('Literal action semantic hash changed')
    with zipfile.ZipFile(contained(root, protocol['old_reference']['history_path'])) as archive:
        old_record = json.loads(archive.read('manifest.json'))['artifacts']['surgical_toolbox']
    old_snapshot = old_record['branches'][old_record['branchId']]
    validate_requests(actions, old_snapshot['actions'])
    if set(old_snapshot['checkpoint']['state']['removed']) != {row['cell_id'] for row in allowed['source_cells']}:
        raise ValueError('Allowed IDs are not the exact old actual removed set')
    if allowed['by_material'] != protocol['allowed_removals']['allowed_source_ids_by_original_material']:
        raise ValueError('Original-material source-ID authority changed')
    source = contained(root, protocol['source']['path']); source_case, source_artifacts, source_workspace = load_workspace(source)
    if 'surgical_toolbox' in source_artifacts or source_case.semantic_hash != protocol['source']['case_hash']:
        raise ValueError('Fresh new source must have no saved physical checkpoint')
    scene = standardize(from_case(source_case), default_target_names(source_case))
    if scene.identity != protocol['fresh_initialization']['initial_scene_id'] or array_digest(scene.labels) != protocol['source']['owned_label_array_hash']:
        raise ValueError('New source material geometry differs')
    source_regions = {row['id']: row for row in scene.regions}
    allowed_rows = allowed['source_cells']
    if len({row['cell_id'] for row in allowed_rows}) != len(allowed_rows):
        raise ValueError('Duplicate source-ID authority')
    for row in allowed_rows:
        cell = row['cell_id']
        if type(cell) is not int or not 0 <= cell < scene.labels.size or not scene.labels.flat[cell]:
            raise ValueError('Invalid original-material source ID')
        region = source_regions[int(scene.labels.flat[cell])]
        if (row['original_region'] != region['name'] or row['original_material'] != region['material']
                or region['material'] not in ('bone', 'soft_tissue') or region.get('target')):
            raise ValueError('Original-material source ID changed')
    directory = safe_path(root, 'integrated-v1/regressions/executions/' + name)
    if directory.exists():
        raise ValueError('Preserve immutable previous execution')
    directory.mkdir(parents=True)
    recipe = safe_path(root, 'integrated-v1/recipes/' + digest(__file__) + '-' + Path(__file__).name)
    if not recipe.exists():
        recipe.write_bytes(Path(__file__).read_bytes())
    if digest(recipe) != digest(__file__):
        raise ValueError('Executor source snapshot differs')
    initial_binding = {'argv': sys.argv, 'cwd': str(Path.cwd()), 'driver_sha256': driver_hash,
                       'driver_snapshot': str(recipe.relative_to(root)), 'protocol_sha256': digest(protocol_path),
                       'decision_sha256': digest(decision_path), 'protocol_path': str(protocol_path),
                       'decision_path': str(decision_path), 'backend_manifest_sha256': digest(contained(root, protocol['backend']['manifest_path'])),
                       'runtime_versions': versions, 'runtime_git': git_binding(), 'source_sha256': digest(source),
                       'actual_optional_kernel_identities': native_identity,
                       'validated_snapshot_filesystem_metadata': manifest.get('validated_filesystem_metadata', []),
                       'old_history_sha256': protocol['old_reference']['history_sha256'],
                       'environment': {k: os.environ.get(k) for k in ('RESSECTIONLAB_DATA_ROOT', 'RESSECTIONLAB_IDLE_THERMAL_BUILD', 'RESSECTIONLAB_PREFIX_JSON_BUILD', 'TMPDIR', 'XDG_CACHE_HOME', 'MPLCONFIGDIR', 'NUMBA_CACHE_DIR', 'TORCH_HOME')},
                       'scope': 'Fresh software counterfactual; no old physical identity or thermal state is imported.'}
    atomic_json(directory / 'reproduction.json', initial_binding)
    claims = safe_path(root, 'integrated-v1/regressions/decision-use'); claims.mkdir(parents=True, exist_ok=True)
    claim_once(claims / (digest(decision_path) + '.json'), {'output': str(directory.relative_to(root)), **initial_binding})
    try:
        session = BridgeSession(directory / 'transfers', run_dir=directory / 'runs')
    except Exception as error:
        result = {'schema': 'one-pass-exact-old-request-software-regression-result-v1',
                  'subject': protocol['subject'], 'pending_negative_session_error': str(error),
                  'committed_action_count': 0, 'preview_count': 0, 'driver_sha256': driver_hash,
                  'protocol_sha256': protocol_hash, 'decision_sha256': decision_hash}
        atomic_json(directory / 'report.json', result)
        return result
    def call(operation, args):
        return session.execute(operation, args, _Request(operation), lambda *_: None)
    records = []; committed_actions = []; removed = set(); open_hands = set()
    preview_count = 0; requested_seconds = 0.; current = None; identity = None
    stopped = None; failure_index = None; cleanup_result = None; started = time.perf_counter(); initial_metrics = None

    def one(index, cleanup=False):
        nonlocal current, preview_count, requested_seconds
        action = copy.deepcopy(actions[index]); before = current['view']; hand = action['slot']
        if digest(__file__) != driver_hash or digest(protocol_path) != protocol_hash or digest(decision_path) != decision_hash:
            raise ValueError('Execution driver/protocol/authority changed during the run')
        if preview_count >= protocol['budgets']['maximum_public_previews']:
            raise ValueError('Frozen preview budget exhausted')
        preview_count += 1
        preview = call('surgicalRehearsal', {'caseHash': identity, 'command': 'preview',
                      'expectedStateId': before['stateId'], 'action': copy.deepcopy(action)})
        row = {'literal_index': index, 'cleanup': cleanup, 'action': action, 'state_before': before['stateId'],
               'actual_slots_before': before['slots'], 'preview': preview, 'committed': False}
        records.append(row); atomic_json(directory / 'actions.json', records)
        inspected = call('surgicalRehearsal', {'caseHash': identity, 'command': 'inspect'})
        if inspected['view']['stateId'] != before['stateId']:
            row['task_reason'] = 'PUBLIC_PREVIEW_CHANGED_ACTUAL_STATE'
            raise ValueError(row['task_reason'])
        instrument = before['slots'][hand]['instrument']
        ok, why = preview_predicate(protocol, allowed, action, instrument, before['metrics'], preview, removed, scene.voxel_mm3)
        if ok:
            try:
                ok, why, trace = contact_trace(scene, action, before['slots'], preview, removed, allowed)
                row['actual_contact_reconstruction'] = trace
                row['contact_trace_scope'] = 'Actual achieved motion inside detached public preview; physical only if the matching commit below succeeds.'
            except Exception as error:
                ok, why = False, 'ACTUAL_CONTACT_TRACE_UNAVAILABLE:' + str(error)
        row['task_predicate_passed'] = ok; row['task_reason'] = why
        atomic_json(directory / 'actions.json', records)
        if not ok:
            return False, why
        if cleanup and action['operation'] != 'withdraw':
            raise ValueError('Cleanup is not a literal withdrawal')
        if len(committed_actions) >= protocol['budgets']['maximum_new_commits']:
            raise ValueError('Frozen commit budget exhausted')
        duration = action.get('duration', 1.)
        if requested_seconds + duration > protocol['budgets']['maximum_total_declared_action_seconds']:
            raise ValueError('Frozen declared-duration budget exhausted')
        if digest(__file__) != driver_hash:
            raise ValueError('Execution driver changed before commit')
        actual = call('surgicalRehearsal', {'caseHash': identity, 'command': 'step',
                     'expectedStateId': before['stateId'], 'action': copy.deepcopy(action)})
        current = actual; did_commit = receipt_of(actual).get('status') in ('accepted', 'partial')
        row.update(step_attempted=True, committed=did_commit, actual=actual, state_after=current['view']['stateId'],
                   actual_slots_after=current['view']['slots'], actual_metrics_after=current['view']['metrics'])
        if did_commit:
            requested_seconds += duration; committed_actions.append(copy.deepcopy(action))
            removed.update(receipt_of(actual).get('deltas', {}).get('removedCells', []))
            if action['operation'] == 'place': open_hands.add(hand)
            if action['operation'] == 'withdraw': open_hands.discard(hand)
        atomic_json(directory / 'actions.json', records)
        if (not did_commit or current['view']['stateId'] != preview.get('stateId')
                or receipt_of(actual).get('deltas') != receipt_of(preview).get('deltas')):
            raise ValueError('Actual step differs from approved preview; no further material action')
        validate_requests(actions, read_json(contained(root, protocol['literal_requests']['path'])))
        return True, None

    try:
        identity = call('loadCase', {'path': str(source)})['caseHash']
        current = call('surgicalRehearsal', {'caseHash': identity, 'command': 'create'})
        if current['view']['stateId'] != protocol['fresh_initialization']['initial_state_id']:
            raise ValueError('Public fresh-zero state differs; no reset or state transplant allowed')
        initial_metrics = current['view']['metrics']
        if initial_metrics != protocol['fresh_initialization']['initial_metrics']:
            raise ValueError('Fresh initial metrics differ')
        for index in range(len(actions)):
            ok, why = one(index)
            if not ok:
                stopped = why; failure_index = index
                cleanup_index, cleanup_reason = select_cleanup(actions, index, current['view']['slots'], open_hands, actions[index]['slot'])
                cleanup_result = {'selected_literal_index': cleanup_index, 'selection_reason': cleanup_reason}
                if cleanup_index is not None:
                    cleanup_ok, cleanup_reason = one(cleanup_index, cleanup=True)
                    cleanup_result.update(committed=cleanup_ok, reason=cleanup_reason)
                break
    except Exception as error:
        stopped = 'PENDING_NEGATIVE_EXECUTION_ERROR:' + str(error)
        if identity is not None:
            try:
                current = call('surgicalRehearsal', {'caseHash': identity, 'command': 'inspect'})
            except Exception as inspection_error:
                stopped += ';INSPECT_FAILED:' + str(inspection_error)
    result = {'schema': 'one-pass-exact-old-request-software-regression-result-v1',
              'subject': protocol['subject'], 'scope': protocol['scope'], 'stopped_reason': stopped,
              'initial_metrics': initial_metrics,
              'failed_literal_index': failure_index, 'cleanup': cleanup_result,
              'preview_count': preview_count, 'committed_action_count': len(committed_actions),
              'committed_declared_action_seconds': requested_seconds,
              'source_sha256': protocol['source']['sha256'], 'protocol_sha256': protocol_hash,
              'decision_sha256': decision_hash, 'driver_sha256': driver_hash,
              'old_outcomes_required_to_match': False, 'old_checkpoint_or_thermal_identity_transferred': False,
              'clinical_validation': False, 'workflow_promotion': False}
    try:
        if identity is None or current is None:
            raise ValueError('No verified public state available to save')
        saved = directory / 'replay.ressectionlab'
        call('saveCase', {'caseHash': identity, 'path': str(saved)})
        result.update(saved_path=str(saved.relative_to(root)), saved_sha256=digest(saved),
                      final_state_id=current['view']['stateId'], final_metrics=current['view']['metrics'],
                      all_tools_withdrawn=all(not s['inserted'] and not s['active'] and not s['grip'] for s in current['view']['slots'].values()))
        owncase, ownartifacts, ownworkspace = load_workspace(saved)
        record = ownartifacts['surgical_toolbox']; snap = record['branches'][record['branchId']]
        if snap['actions'] != committed_actions or len(snap['actions']) > protocol['budgets']['maximum_new_commits']:
            raise ValueError('Saved own literal history differs')
        if (owncase.semantic_hash != source_case.semantic_hash
                or [(o.name, o.data) for o in ownworkspace['originals']] != [(o.name, o.data) for o in source_workspace['originals']]):
            raise ValueError('Saved own source arrays/original bytes differ')
        cold = BridgeSession(directory / 'cold-transfers', run_dir=directory / 'cold-runs')
        try:
            cold_id = cold.execute('loadCase', {'path': str(saved)}, _Request('loadCase'), lambda *_: None)['caseHash']
            inspected = cold.execute('surgicalRehearsal', {'caseHash': cold_id, 'command': 'inspect'}, _Request('surgicalRehearsal'), lambda *_: None)
            if inspected['view']['stateId'] != current['view']['stateId']:
                raise ValueError('Cold public own-record verification differs')
        finally:
            cold.close()
        result.update(source_arrays_and_originals_preserved=True, own_saved_literal_history_exact=True,
                      cold_public_own_record_exact=True, own_checkpoint_semantic_hash=semantic_digest(snap['checkpoint']['state']))
    except Exception as error:
        result['pending_negative_save_or_cold_error'] = str(error)
    finally:
        try:
            session.close()
        except Exception as error:
            result['pending_negative_session_close_error'] = str(error)
    own_temperatures = [row.get('actual_metrics_after', {}).get('drillMaxTemperatureC', 37.) for row in records if row.get('committed')]
    result['own_trajectory_max_recorded_spatial_temperature_C'] = max([37., *own_temperatures])
    result['own_costs_only'] = True; result['runtime_wall_seconds'] = time.perf_counter() - started
    try:
        verify_backend(root, protocol, backend_override)
        if digest(__file__) != driver_hash or digest(protocol_path) != protocol_hash or digest(decision_path) != decision_hash:
            raise ValueError('Executor/protocol/decision identity changed')
        if native_identity != {'idle_thermal': idle_identity(), 'prefix_json': prefix_identity(full=True)}:
            raise ValueError('Actual selected optional kernel identity changed')
        if digest(source) != protocol['source']['sha256'] or digest(contained(root, protocol['old_reference']['history_path'])) != protocol['old_reference']['history_sha256']:
            raise ValueError('Immutable source/history changed')
        result['frozen_backend_and_sources_stable_before_after'] = True
    except Exception as error:
        result['pending_negative_binding_error'] = str(error)
    atomic_json(directory / 'report.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True); parser.add_argument('--protocol', required=True)
    parser.add_argument('--decision', required=True); parser.add_argument('--name', required=True)
    parser.add_argument('--backend-override', help='Must equal the protocol-bound frozen package root')
    args = parser.parse_args()
    print(json.dumps(execute(args.root, args.protocol, args.decision, args.name, args.backend_override), indent=2))
