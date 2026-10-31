#!/usr/bin/env python3
"""Inventory declared evidence and canonically replay an existing development episode.

Writes compact reports only. No dataset download, inference, or training.
"""
from pathlib import Path
from collections import Counter
import argparse
import copy
import hashlib
import json
import platform
import re
import resource
import subprocess
import sys
import time
from zipfile import ZipFile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from resectionlab.core import array_digest
from resectionlab.simulator_qualification import (
    SCHEMA, file_sha256, native_coverage, removal_phases, role_reservations, validate_removed_ids)
from resectionlab.surgical_deformation_v2 import diagnostic_scope


def inventory(directories):
    versions = []
    excluded = []
    paths = sorted({p.resolve() for directory in directories for p in Path(directory).rglob('patient-anatomy.json')})
    for path in paths:
        if path.stat().st_size > 16 * 1024**2:
            raise ValueError('Oversized native manifest: ' + str(path))
        raw = path.read_bytes()
        if len(raw) > 16 * 1024**2:
            raise ValueError('Oversized native manifest: ' + str(path))
        manifest = json.loads(raw)
        patient = manifest.get('patient_id', '')
        if not re.fullmatch(r'(UCSF-PDGM-\d{4}|IXI\d+-[\w-]+|ZENODO-17486320-meningioma-sub-\d+|TOPBRAIN-2025-TopCoW-CTA-\d+)', patient):
            excluded.append({'path': str(path), 'patientId': patient,
                'reason': 'outside current acquired native-study identity inventory; not counted as a patient'})
            continue
        row = native_coverage(manifest)
        row.update(manifestPath=str(path), manifestSha256=hashlib.sha256(raw).hexdigest())
        row['role'] = ('FROZEN_ASSESSMENT_NO_NEW_PAYLOAD_READ' if patient == 'UCSF-PDGM-0008'
            else 'EXISTING_DEVELOPMENT_WITH_FROZEN_NEGATIVE_ASSESSMENTS' if patient == 'UCSF-PDGM-0005'
            else 'EXISTING_DEVELOPMENT' if patient in ('UCSF-PDGM-0004', 'UCSF-PDGM-0007')
            else 'TRAIN_HEALTHY_NON_GLIOMA' if patient.startswith('IXI265-')
            else 'EXISTING_VASCULAR_DEVELOPMENT_NON_GLIOMA' if patient.startswith('TOPBRAIN-2025-')
            else 'DEVELOPMENT_NON_GLIOMA' if patient.startswith('ZENODO-17486320-') else 'UNKNOWN_NO_PAYLOAD_ADMISSION')
        versions.append(row)
    return {'scope': 'Existing native-study metadata below requested roots; legacy bundles and absent historical catalogs are not requalified',
        'roots': [str(Path(p).resolve()) for p in directories],
        'uniquePatients': len({v['patientId'] for v in versions}), 'versions': versions,
        'excludedManifests': excluded, 'missingVesselsMeanSafe': False,
        'crossVersionAnatomyFusion': False, 'pixelArraysRead': False}


def baseline(path, branch, output):
    from resectionlab.workspace_bundle import load_workspace
    from resectionlab.surgical_scene import from_case
    from resectionlab.surgical_environment import standardize
    from resectionlab.surgical_engine import SurgicalEngine
    started = time.perf_counter()
    before_hash = file_sha256(path)
    contract = json.loads((ROOT / 'manifests/experiments/simulator-qualification-v1.json').read_text())
    # Bind the ENTIRE archive before the loader can decode auxiliary images or
    # original files. A primary case ID alone cannot protect held-out payloads.
    if before_hash != contract['baseline_archive_sha256'] or branch != contract['baseline_branch']:
        raise ValueError('Frozen baseline archive/branch identity required before payload access')
    with ZipFile(path) as archive:
        if archive.getinfo('manifest.json').file_size > 16 * 1024**2:
            raise ValueError('Oversized case manifest')
        declared = json.loads(archive.read('manifest.json'))
        if declared.get('case_id') != 'UCSF-PDGM-0007-surgical-research':
            raise ValueError('This frozen baseline is limited to the existing 0007 development case')
    case, artifacts, _ = load_workspace(path)
    # Existing development identity only; this runner grants no held-out access.
    if case.case_id != 'UCSF-PDGM-0007-surgical-research':
        raise ValueError('This frozen baseline is limited to the existing 0007 development case')
    record = artifacts['surgical_toolbox']
    scene = standardize(from_case(case, record['mapping']), physics_profile=record['physicsProfile'])
    if scene.identity != 'sha256:b7c1e36161051be29b5e694aab35c9b11828f63b0c3444016824f6d506c0db12':
        raise ValueError('Baseline source scene changed')
    snapshot = record['branches'][branch]
    if snapshot['stateId'] != contract['baseline_state_id']:
        raise ValueError('Frozen baseline endpoint changed')
    setup = record['researchSetup']
    exposure_end, work_start, work_end = (setup[k] for k in ('exposureActions', 'finiteCusaStart', 'finiteCusaEnd'))
    boundaries = {0, exposure_end, work_start, work_end, len(snapshot['actions'])}
    if not (0 < exposure_end <= work_start < work_end <= len(snapshot['actions'])):
        raise ValueError('Invalid frozen baseline boundaries')
    labels_hash, image_hash = array_digest(scene.labels), array_digest(case.mri)
    engine = SurgicalEngine(scene)
    states = {0: copy.deepcopy(engine.state)}
    timings, statuses = [], []
    previous_removed = set()
    with diagnostic_scope() as diagnostics:
        for index, action in enumerate(snapshot['actions'], 1):
            t = time.perf_counter()
            result = engine.step(action)
            timings.append(time.perf_counter() - t)
            statuses.append(result['status'])
            if result['status'] == 'rejected':
                raise ValueError(f'Canonical replay rejected action {index}: {result.get("reason")}')
            previous_removed = validate_removed_ids(scene, engine.state['removed'], previous=previous_removed)
            if index in boundaries:
                states[index] = copy.deepcopy(engine.state)
            (output / 'progress.json').write_text(json.dumps({'action': index,
                'totalActions': len(snapshot['actions']), 'stateId': engine.state_id}) + '\n')
    accounting = removal_phases(scene, states, exposure_end=exposure_end, work_start=work_start, work_end=work_end)
    metrics = engine.metrics()
    checks = {'canonicalStateId': engine.state_id == snapshot['stateId'],
        'allActionBoundarySourceIdentityMonotone': True,
        'fullCheckpointState': engine.state == snapshot['checkpoint']['state'],
        'sourceLabelsUnchanged': labels_hash == array_digest(scene.labels),
        'mriUnchanged': image_hash == array_digest(case.mri),
        'archiveUnchanged': before_hash == file_sha256(path),
        'independentTargetAccounting': abs(accounting['fullEpisode']['targetRemovedMm3'] - metrics['targetRemovedMm3']) < 1e-9,
        'independentNonTargetAccounting': abs(accounting['fullEpisode']['nonTargetRemovedMm3'] - metrics['nonTargetRemovedMm3']) < 1e-9}
    from resectionlab.surgical_cusa_v3 import profile as cusa_profile
    return {'status': 'pass' if all(checks.values()) else 'fail', 'checks': checks,
        'statusScope': 'Stage 0 canonical replay and source accounting only; contract physical gates remain open',
        'input': str(path), 'archiveSha256': before_hash, 'sceneId': scene.identity,
        'finalStateId': engine.state_id, 'branch': branch,
        'taskScope': 'post-exposure engineering baseline; original historical exposure cost retained',
        'accounting': accounting, 'assumedPhysics': scene.properties['tissuePhysicsV2'],
        'assumedCusa': copy.deepcopy(cusa_profile(engine)),
        'mechanicsDiagnostics': diagnostics,
        'deformationExercised': diagnostics['substeps'] > 0,
        'timing': {'actions': len(timings), 'accepted': statuses.count('accepted'), 'partial': statuses.count('partial'),
            'stepP50Seconds': float(np.median(timings)),
            'stepP95Seconds': float(np.percentile(timings, 95)),
            'wallSeconds': time.perf_counter() - started,
            'processPeakRssBytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024)},
        'measuredResponseAgreement': 'not established; this checks replay/accounting only',
        'meaningfulPhysiologicalRLReady': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native-root', type=Path, action='append', required=True)
    parser.add_argument('--case', type=Path)
    parser.add_argument('--branch', default='operative-cusa')
    parser.add_argument('--mechanics', action='store_true', help='Run generated analytical and grid-sensitivity controls')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / 'testcases') or output.exists():
        parser.error('Choose a NEW output directory under testcases/')
    contract_path = ROOT / 'manifests/experiments/simulator-qualification-v1.json'
    contract = json.loads(contract_path.read_text())
    output.mkdir(parents=True)
    report = {'schema': SCHEMA, 'gitRevision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'sourceSha256': {name: file_sha256(ROOT / name) for name in (
            'src/resectionlab/surgical_deformation_v2.py', 'src/resectionlab/surgical_engine.py',
            'src/resectionlab/simulator_qualification.py', 'scripts/qualify_surgical_simulator.py')},
        'hardware': {'platform': platform.platform(), 'machine': platform.machine(), 'python': platform.python_version()},
        'contractSha256': file_sha256(contract_path), 'contract': contract,
        'coverage': inventory(args.native_root), 'reservations': role_reservations(ROOT),
        'trainingUpdates': 0, 'newDatasetBytes': 0, 'largeScaleRL': 'hold'}
    (output / 'coverage.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    results = {q['id']: {'status': 'unvalidated', 'reason': q.get('status', 'not exercised')} for q in contract['questions']}
    if args.case:
        result = baseline(args.case.resolve(), args.branch, output)
        (output / 'baseline.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
        if result['status'] != 'pass':
            raise SystemExit('Baseline validation failed; inspect baseline.json')
        for key in ('A-source', 'A-replay'):
            results[key] = {'status': 'pass', 'evidence': 'baseline.json', 'level': 'numerical correctness only'}
    if args.mechanics:
        from qualify_surgical_mechanics import run
        mechanics = run()
        (output / 'mechanics.json').write_text(json.dumps(mechanics, indent=2, allow_nan=False) + '\n')
        checks = mechanics['checks']
        energy_pass = checks['backwardEulerMatchesIndependentFormula'] and checks['forceResidual'] and checks['discreteEnergyBalance']
        results['A-energy'] = {'status': 'pass' if energy_pass else 'fail', 'evidence': 'mechanics.json', 'level': 'analytical generated coupon only'}
        results['A-domain'] = {'status': 'partial', 'evidence': 'mechanics.json',
            'reason': 'Projection/force residual instrumentation verified; no measured tissue envelope has been established'}
        results['A-convergence'] = {'status': 'fail', 'evidence': 'mechanics.json',
            'reason': 'Time convergence passes analytical control; unchanged per-node profile changes energy density with physical grid refinement'}
    suite = {'schema': 'resectionlab.simulator-validation-suite.v1',
        'questions': results, 'counts': dict(Counter(r['status'] for r in results.values())),
        'physicalGatePassed': False, 'largeScaleRL': 'hold',
        'scoreMeaning': 'Question status coverage, not a clinical accuracy score; numerical passes do not satisfy measured-response gates'}
    (output / 'validation-suite.json').write_text(json.dumps(suite, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'output': str(output), 'patients': report['coverage']['uniquePatients'], 'largeScaleRL': 'hold'}))


if __name__ == '__main__':
    main()
