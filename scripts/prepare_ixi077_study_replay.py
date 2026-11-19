#!/usr/bin/env python3
"""Build a native study companion to the existing, fixed IXI077 engine recording."""
import argparse
import json
from pathlib import Path

import numpy as np

from resectionlab.anatomy_approaches import dispatch_plan, read_approaches
from resectionlab.anatomy_planning import native_target_binding
from resectionlab.anatomy_workspace import save_view
from resectionlab.data_root import checked_root, sha256_file
from resectionlab.patient_anatomy import PatientImagingCollection
from resectionlab.surgical_exact_grid_replay import FIXED_CUSA_REPORT_SHA256, SCHEMA
from resectionlab.surgical_study_replay import verify_recording
from resectionlab.workspace_bundle import load_workspace, save_workspace
from data_integrated_hosts import UUID, checked_file


REPORT = 'qc/ixi077-organic-conditional-replay-v1/report.json'
TARGET = 'synthetic_target_observation'
APPROACH = 'recorded-post-exposure'
GENERATOR = 'Fixed post-exposure CUSA sequence'


def prepare(root, output):
    checked_root(root, expected_uuid=UUID)
    report_text = checked_file(root, {'path': REPORT, 'sha256': FIXED_CUSA_REPORT_SHA256}).read_text()
    report = json.loads(report_text)
    contract = report['contract']
    path = checked_file(root, {'path': report['replay'], 'sha256': report['replay_sha256']})
    inputs = [checked_file(root, item) for item in contract['input_bindings']]
    source = json.loads(inputs[0].read_text())
    preflight = json.loads(inputs[2].read_text())
    registration = json.loads(checked_file(root, source['source']['registration']).read_text())
    case, artifacts, workspace = load_workspace(path)
    tb = artifacts['surgical_toolbox']
    snapshot = tb['branches'][tb['branchId']]
    program = contract['actions']
    expected = [program['configure'], program['place'],
                *[program['apply']] * (report['committed_action_count'] - 3), program['withdraw']]
    if (report['status'] != 'target_effect_and_withdrawal' or not report['cold_reopen_exact']
            or source['case_hash'] != contract['case_hash'] or case.semantic_hash != contract['case_hash']
            or preflight['source_scene_id'] != snapshot['sceneId'] or snapshot['stateId'] != report['final_state_id']
            or not all(preflight['checks'].values()) or snapshot['actions'] != expected
            or preflight['route']['start_ras_mm'] != program['place']['pose']['position']
            or preflight['route']['goal_ras_mm'] != program['apply']['path'][0]['position']):
        raise ValueError('Recorded trajectory differs from its fixed source, screened route or execution contract')
    output.mkdir(parents=True, exist_ok=False)
    study = PatientImagingCollection.create(output/'native', case.metadata['patient_id'], timepoint='source-session-unknown')
    def add_scan(binding, sid, modality):
        return study.add_series(checked_file(root, binding), series_id=sid, modality=modality,
                                timepoint='source-session-unknown')
    add_scan(registration['fixed_T1'], 't1', 'T1')
    add_scan(registration['moving_MRA'], 'mra', 'MRA')
    registered = next(item for item in registration['outputs'] if item['path'].endswith('/registered-MRA.nii.gz'))
    add_scan(registered, 'mra-registered', 'MRA')
    study.set_registration('mra', registration['moving_MRA_RAS_to_fixed_T1_RAS_forward'], status='unreviewed',
                           method='Independently reviewed approximate rigid transform for overlay inspection only')
    study.set_registration('mra-registered', np.eye(4), status='unreviewed',
                           method='MRA resampled on the T1 grid with the reviewed approximate transform')
    for sid in ('mra', 'mra-registered'):
        study.document['transforms'][sid]['quality'] = {
            'warnings': ['Reviewed for overlay inspection only. Alignment uncertainty is unquantified.'],
            'reviewDecision': case.metadata['workspace_registration']['review_decision'],
            'reviewEvidenceSha256': case.metadata['workspace_registration']['evidence_sha256'],
            'coverage': 'Acquired MRA slab only'}
    study.document['missing'] = list(case.metadata['data_case']['missing_anatomy'])
    for name, mask in case.compartments.items():
        meta = case.metadata['anatomy_layers'][name]
        target = name == TARGET
        vessel = meta['kind'] == 'vessels'
        provenance = {'kind': 'simulated' if target else 'registered_source_annotation' if vessel else 'model_estimate'}
        if vessel:
            provenance['source'] = study.original(checked_file(root, source['source']['mask_bindings']['vessel']))
            provenance['evidence_use'] = 'context_only'
            provenance['registrationReviewSha256'] = case.metadata['workspace_registration']['evidence_sha256']
        elif not target:
            provenance['model'] = 'SynthStrip'
        layer = study.add_layer(mask.astype(np.uint8), case.affine, layer_id=name,
            source_series='mra-registered' if vessel else 't1', kind='synthetic-target' if target else meta['kind'],
            role='target' if target else 'critical' if vessel else 'context', label=meta['label'],
            color=meta['color'], provenance=provenance)
        if 'coverage' in meta:
            layer['coverage'] = meta['coverage']
    # Use the existing display pyramid for an acquired MRA larger than the full-resolution viewer budget.
    study.preview_pyramid('mra', levels=(2,))
    study.save('Imported same-person source images, registered partial vessels, estimated brain and simulated target')
    selection = native_target_binding(study, study.layer(TARGET))
    dispatch_plan(study, {'action': 'selectTarget', 'targetLayerId': TARGET, 'candidateBinding': selection})
    dispatch_plan(study, {'action': 'upsert', 'approach': {'id': APPROACH, 'name': 'Recorded post-exposure CUSA approach',
        'targetLayerId': TARGET, 'entry': preflight['route']['start_ras_mm'], 'target': preflight['route']['goal_ras_mm'],
        'radiusMm': preflight['stock_parameters']['radiusMm']}})
    route = next(row for row in read_approaches(study) if row['id'] == APPROACH)
    save_view(study, dict(schema='patient-anatomy-view-v1', patientId=study.document['patient_id'], snapshotId=study.identity(),
                          seriesId='t1', states={}, selectedPlanningTargetId=TARGET, selectedApproachId=APPROACH,
                          selectedApproachIds={TARGET: APPROACH}))
    record_route = {key: route[key] for key in ('id', 'name', 'entry', 'target', 'radiusMm', 'measurementHash', 'parameterHash')}
    tb['exactGridReplay'] = {'schema': SCHEMA, 'sourceManifest': study.document,
        'executionReport': report_text, 'recordedApproach': record_route, 'referenceSeriesId': 't1', 'movingSeriesId': 'mra',
        'registeredSeriesId': 'mra-registered', 'targetLayerId': TARGET,
        'actionGenerator': {'kind': 'scripted', 'name': GENERATOR, 'executionReportSha256': FIXED_CUSA_REPORT_SHA256,
                            'scriptSha256': contract['executor_sha256']}}
    save_workspace(case, output/'replay.ressectionlab', sources=workspace['sources'], imaging=workspace['imagingState'],
        episode=workspace['episode'], selection=workspace['selection'], artifacts=artifacts, originals=workspace['originals'])
    audit = {'sourceReplaySha256': sha256_file(path), 'sourceReportSha256': FIXED_CUSA_REPORT_SHA256,
             'sourceSnapshotId': study.identity(), 'recordedApproach': record_route, 'caseHash': case.semantic_hash,
             'engineSceneId': snapshot['sceneId'], 'engineEndpoint': snapshot['stateId'],
             'originalTargetCells': int(case.compartments[TARGET].sum()), 'finalMetrics': report['final_metrics']}
    (output/'preparation.json').write_text(json.dumps(audit, indent=2, allow_nan=False) + '\n')


def verify(output):
    study = PatientImagingCollection(output/'native')
    case, artifacts, _ = load_workspace(output/'replay.ressectionlab')
    receipt, sampling = verify_recording(study, TARGET, case, artifacts['surgical_toolbox'], approach_id=APPROACH)
    (output/'verified-binding.json').write_text(json.dumps({'receipt': receipt, 'sampling': sampling}, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to(Path(__file__).resolve().parents[1]/'testcases'):
        parser.error('Keep generated testing data under testcases/')
    prepare(args.data_root, args.output)
    verify(args.output)
