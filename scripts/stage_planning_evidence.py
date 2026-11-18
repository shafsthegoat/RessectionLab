#!/usr/bin/env python3
"""Stage a private, portable native comparison without changing its source study."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import time

import nibabel as nib
import numpy as np

from resectionlab.anatomy_approaches import NativeRouteGeometry, dispatch_plan
from resectionlab.anatomy_package import open_package, save_package
from resectionlab.anatomy_planning import native_target_options
from resectionlab.patient_anatomy import PatientAnatomySnapshot, PatientImagingCollection

ROOT = Path(__file__).resolve().parents[1]


def generated(out):
    x, y, z = np.indices((32, 32, 32))
    cerebral = (3 <= y) & (y < 28) & (4 <= z) & (z < 28)
    right = cerebral & (4 <= x) & (x < 16)
    left = cerebral & (18 <= x) & (x < 27)
    target = right & (x >= 9) & (x <= 11) & (y >= 10) & (y <= 12) & (z >= 14) & (z <= 16)
    image = nib.Nifti1Image(np.where(target, 140, np.where(right | left, 80, 0)).astype(np.float32), np.eye(4))
    image.header.set_xyzt_units('mm')
    source = out / 'generated-source.nii.gz'
    nib.save(image, source)
    study = PatientImagingCollection.create(out / 'study', 'Generated cortical access control')
    study.add_series(source, series_id='t1', modality='T1')
    for name, mask, kind, role in [('brain', right | left, 'brain', 'context'),
                                  ('target', target, 'synthetic-target', 'target')]:
        study.add_layer(mask.astype(np.uint8), np.eye(4), layer_id=name, label=name.title(),
                        source_series='t1', kind=kind, role=role, color='#d073a9')
    for role, mask in {
        'right-cerebral-hemisphere': right, 'left-cerebral-hemisphere': left,
        'cortical-entry-surface': cerebral & ((x == 4) | (x == 26)),
        'supratentorial-compartment': y >= 3, 'posterior-fossa': y < 2,
        'falx-cerebri': cerebral & (x == 17), 'tentorium-cerebelli': y == 2,
    }.items():
        row = study.add_layer(mask.astype(np.uint8), np.eye(4), layer_id=role, label=role,
                              source_series='t1', kind='context', role='context', color='#abcdee',
                              provenance={'cortical_access_role': role, 'cortical_access_evidence': 'generated-reference'})
        row['review_status'] = 'generated-engineering'
    study.save('Generated cortical compartments and surface')
    return study


def stage(out, source, target_id, *, seconds, radius):
    deadline = time.monotonic() + seconds

    def check():
        if time.monotonic() >= deadline:
            raise TimeoutError('Private native comparison reached its wall limit')

    if source is None:
        study = generated(out)
    elif source.suffix == '.rsworkspace':
        study = PatientImagingCollection(open_package(source, out / 'working', check=check).parent)
    else:
        work = out / 'study'
        work.mkdir()
        shutil.copyfile(source, work / source.name)
        for member in ('arrays', 'originals', 'revisions'):
            check()
            shutil.copytree(source.parent / member, work / member, ignore=shutil.ignore_patterns('._*'))
        study = PatientImagingCollection(work)
    options = native_target_options(study, layer_id=target_id, check=check)
    if len(options) != 1 or options[0].get('target') is None:
        raise ValueError(f'Choose one nonempty aligned target: {target_id}')
    option, = options
    dispatch_plan(study, {'action': 'selectTarget', 'targetLayerId': target_id,
                          'candidateBinding': option['candidateBinding']}, check=check)
    target = [10., 11., 15.] if source is None else option['target']
    # Match the existing geometric editor's initial sample when the source
    # supplies no entry coordinate. The access evaluator gives it no credit.
    entry = [3.5, 6., 14.] if source is None else option.get('entry', [target[0], target[1], target[2] + 40.])
    task = {'action': 'suggest', 'start': entry, 'end': target, 'radiusMm': radius,
            'targetLayerId': target_id, 'patchRadiusMm': 0. if source is None else 6.}
    geometry = NativeRouteGeometry(PatientAnatomySnapshot.capture(study), check=check)
    result = dispatch_plan(study, task, geometry=geometry, check=check)
    # Keep an explicit assessed geometry when all sampled alternatives are
    # vetoed. Its assessment may show contact and incomplete access evidence.
    if not result['approaches'] or not any(row['assessmentStatus'] == 'current' for row in result['approaches']):
        result = dispatch_plan(study, {'action': 'upsert', 'approach': {
            **task, 'entry': entry, 'target': target, 'name': 'Assessed reference corridor'}},
            geometry=geometry, check=check)
    summary = result['proposalReport'] if 'proposalReport' in result else None
    print(f"Retained measured alternatives: {sum(row['assessmentStatus'] == 'current' for row in result['approaches'])}")
    if summary is not None:
        print(f"Sampled alternatives: {summary.get('enumerated')}")
    if source is not None and source.name == 'patient-anatomy.json':
        print(study.root / source.name)
    else:
        destination = out / 'comparison.rsworkspace'
        save_package(study, destination, check=check)
        print(destination)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--target', default='target')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--radius', type=float, default=.4)
    parser.add_argument('--seconds', type=int, choices=(90, 180, 300), default=180)
    args = parser.parse_args()
    destination = args.output.resolve()
    if not destination.is_relative_to(ROOT / 'testcases'):
        parser.error('Stage portable comparisons under ignored testcases/')
    destination.mkdir(parents=True, exist_ok=False)
    source = args.source.resolve(strict=True) if args.source else None
    if source is not None and source.suffix != '.rsworkspace' and source.name != 'patient-anatomy.json':
        parser.error('Use a prepared patient-anatomy.json or .rsworkspace')
    stage(destination, source, args.target, seconds=args.seconds, radius=args.radius)
