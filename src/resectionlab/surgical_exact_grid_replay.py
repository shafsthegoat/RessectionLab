"""Read-only join for a fixed post-exposure CUSA recording on its original image lattice."""
from __future__ import annotations

import hashlib
import json

import nibabel as nib
import numpy as np

from .anatomy_operative_field import _saved_route
from .anatomy_planning import manifest_target_binding
from .patient_anatomy import digest, file_hash
from .surgical_tools import TOOLS


SCHEMA = 'resectionlab.exact-grid-fixed-cusa-replay.v1'
# Only this archived producer report has been reviewed for the fixed-route join.
# Its body binds the source case, numerical engine, script, actions and observed endpoint.
FIXED_CUSA_REPORT_SHA256 = '28da0e97f025b4916f9867c9cbeb9902e424793bdd1cd8160ba11e1aadc543b6'


def fixed_route_identity(case, scene, snapshot, exact):
    """Bind only the supported recorded script shape, without admitting new actions."""
    if (not isinstance(exact, dict) or exact.get('schema') != SCHEMA or
            scene.source['adapter'].get('type') == 'native-patient-research-derivative' or
            case.metadata.get('integrated_rehearsal', {}).get('stage') != 'conditional_exposed_brain_geometry'):
        raise ValueError('Unsupported source for exact-grid post-exposure replay')
    manifest = exact['sourceManifest']
    target_id = exact['targetLayerId']
    target = next(r for r in manifest['layers'] if r['id'] == target_id)
    if (manifest['patient_id'] != case.metadata.get('patient_id') or target['kind'] != 'synthetic-target'
            or target['provenance'].get('kind') != 'simulated'
            or [r['name'] for r in scene.regions if r['target']] != [target_id]
            or manifest['snapshot_id'] != digest({k: v for k, v in manifest.items() if k != 'snapshot_id'})):
        raise ValueError('Recorded patient, target or source snapshot differs')
    route = exact['recordedApproach']
    generator = exact['actionGenerator']
    report_text = exact.get('executionReport')
    if not isinstance(report_text, str) or hashlib.sha256(report_text.encode()).hexdigest() != FIXED_CUSA_REPORT_SHA256:
        raise ValueError('Unknown fixed CUSA execution report')
    report = json.loads(report_text)
    contract = report['contract']
    if generator != {'kind': 'scripted', 'name': 'Fixed post-exposure CUSA sequence',
                     'executionReportSha256': FIXED_CUSA_REPORT_SHA256, 'scriptSha256': contract['executor_sha256']}:
        raise ValueError('Fixed CUSA generator differs from its verified execution report')
    actions = snapshot['actions']
    program = contract['actions']
    expected = [program['configure'], program['place'], *[program['apply']] * (report['committed_action_count'] - 3),
                program['withdraw']]
    if (case.semantic_hash != contract['case_hash'] or scene.identity != contract['scene_id']
            or snapshot['stateId'] != report['final_state_id'] or actions != expected):
        raise ValueError('Recorded engine branch differs from the fixed CUSA execution report')
    slot = snapshot['checkpoint']['state']['slots']['right']
    if (slot['instrument'] != 'cusa' or slot['inserted'] or slot['active'] or
            route['radiusMm'] != slot['parameters'].get('radiusMm', TOOLS['cusa']['radiusMm'])):
        raise ValueError('Recorded instrument or route radius differs')
    if program['place']['pose']['position'] != route['entry'] or program['apply']['path'][0]['position'] != route['target']:
        raise ValueError('Recorded tool requests do not follow the selected approach')
    return dict(patientId=manifest['patient_id'], sourceSnapshotId=manifest['snapshot_id'],
                targetBinding=manifest_target_binding(manifest, target),
                recordedApproach=route, actionGenerator=generator)


def verify_selected_route(collection, exact, approach_id):
    route = exact['recordedApproach']
    if approach_id != route['id']:
        raise ValueError('Select the recorded approach for this exact-grid episode')
    args = dict(approachId=approach_id, targetLayerId=exact['targetLayerId'], snapshotId=collection.identity(),
                parameterHash=route['parameterHash'])
    selected, _, _ = _saved_route(collection, args)
    keys = ('id', 'name', 'entry', 'target', 'radiusMm', 'measurementHash', 'parameterHash')
    if {key: selected[key] for key in keys} != route:
        raise ValueError('The selected native route differs from the recorded path or measurements')


def verify_exact_source(collection, case, scene, exact, approach_id, *, check):
    manifest = exact['sourceManifest']
    if manifest != collection.document:
        raise ValueError('Recorded patient or native study revision differs from the open study')
    verify_selected_route(collection, exact, approach_id)
    reg = case.metadata['workspace_registration']
    reference = exact['referenceSeriesId']
    moving = exact['movingSeriesId']
    registered = exact['registeredSeriesId']
    hashes = {reference: reg['source_sha256_by_modality']['T1'], moving: reg['original_moving_image_sha256'],
              registered: reg['source_sha256_by_modality']['MRA']}
    if (len(hashes) != 3 or manifest['reference_series'] != reference or set(hashes) != {s['id'] for s in manifest['series']}
            or next(s.sha256 for s in case.source_refs if s.source_id == 'structural') != hashes[reference]):
        raise ValueError('Recorded reference or registered source identity differs')
    for sid, sha in hashes.items():
        check()
        row = collection.series(sid)
        path = collection.path(row['original']['path'])
        if row['original']['sha256'] != sha or file_hash(path) != sha:
            raise ValueError('Original scan bytes differ from the recorded sources')
        if sid != reference:
            registration = manifest['transforms'][sid]
            quality = registration.get('quality', {})
            if (registration['status'] != 'unreviewed' or quality.get('reviewDecision') != reg['review_decision']
                    or quality.get('reviewEvidenceSha256') != reg['evidence_sha256'] or row['modality'] != 'MRA'):
                raise ValueError('The recorded MRA registration supports overlay inspection only')
        elif row['modality'] != 'T1':
            raise ValueError('Exact-grid source reference must be the recorded T1 acquisition')
        transform = collection.registration_matrix(sid, require_geometry=True)
        expected = reg['original_moving_to_reference_ras_mm'] if sid == moving else np.eye(4)
        scan = collection.array(row['data'])
        if (not np.array_equal(transform, expected) or
                not np.array_equal(scan, nib.load(path).get_fdata(dtype=np.float32))):
            raise ValueError('Registered source transform or decoded source pixels differ')
        if sid == reference and (not np.array_equal(scan, case.mri) or not np.array_equal(row['affine'], case.affine)):
            raise ValueError('Recorded reference image differs from the exact native grid')
    if (scene.shape != case.mri.shape or not np.array_equal(scene.affine, case.affine)
            or {row['id'] for row in manifest['layers']} != set(case.compartments)
            or {r['name'] for r in scene.regions} != set(case.compartments)):
        raise ValueError('Recorded grid or modeled compartment set differs')
    for name, mask in case.compartments.items():
        check()
        layer = collection.layer(name)
        original = case.metadata['anatomy_layers'][name]
        src = layer['source_series']
        if original['kind'] == 'vessels':
            provenance = layer['provenance']
            if (layer['kind'] != 'vessels' or src != registered or provenance.get('evidence_use') != 'context_only'
                    or provenance.get('registrationReviewSha256') != reg['evidence_sha256']):
                raise ValueError('The recorded partial vascular annotation supports registered overlay inspection only')
        elif src != reference:
            raise ValueError('Recorded brain and target labels must share the original T1 image frame')
        # Equality of recorded voxel geometry permits replay inspection only.
        # The native quantitative registration gate remains closed for overlay-only MRA.
        collection.require_native_geometry(layer)
        layer_affine = collection.registration_matrix(layer['source_series'], require_geometry=True) @ layer['affine']
        if (not np.array_equal(layer_affine, case.affine)
                or not np.array_equal(collection.mask(layer), mask)):
            raise ValueError('Recorded material membership differs from the native layer: ' + name)
