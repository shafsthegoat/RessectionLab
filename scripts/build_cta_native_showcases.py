#!/usr/bin/env python3
"""Prepare focused native display companions from checksum-bound CTA cases."""
from __future__ import annotations

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import nibabel as nib
import numpy as np

from build_cta_organic_gliomas import COLLECTION, OBSERVATION, checked_mask, read_json
from data_integrated_hosts import ORDER, UUID, binding, checked_file
from resectionlab.anatomy_planning import is_native_target
from resectionlab.core import array_digest
from resectionlab.data_root import atomic_json, checked_root, safe_path, sha256_file
from resectionlab.patient_anatomy import PatientImagingCollection, nifti_geometry, reconstruction_lock
from resectionlab.workspace_bundle import load_workspace


SUBJECTS = ('014', '006', '012')
OUTPUT = 'prepared/cta-native-showcases-v1'
SCHEMA = 'cta-native-showcase-v1'
TIMEPOINT = 'source-session-unknown'
SERIES = 'ct'
BRAIN, LOCAL_SKULL, ARTERIES, VEINS = ORDER
SKULL = 'broader_skull_display'
LIMITS = [
    'Source CTA is the acquired baseline. The contour is a generated, approximate observation.',
    'Brain and skull are partial estimates. The brain envelope can include CSF. The skull can omit '
    'low-density bone and include calcification; source crop faces remain unknown.',
    'The supplied vascular labels cover selected arteries and veins or sinuses.',
    'The display companion carries no route or operative authority and no privileged lesion compartments.',
]


def skull_display_root(subject):
    suffix = '' if subject == '006' else f'-CTA{subject}'
    return f'integrated-v1/skull-display{suffix}-v1'


def read_scenario(root, subject):
    scenario_id = f'TopCoW-{subject}-CTA-ORGANIC-01-v1'
    receipt_path = safe_path(root, f'{COLLECTION}/{scenario_id}/receipt.json')
    receipt = read_json(receipt_path)
    package = checked_file(root, receipt['native_case'])
    case, artifacts, workspace = load_workspace(package)
    if (receipt['case_id'] != scenario_id or receipt['person_id'] != f'TopCoW:{subject}'
            or receipt['role'] != 'DEVELOPMENT' or not receipt['production_workspace_reopen']
            or case.case_id != scenario_id or case.frame != 'RAS+' or artifacts or not workspace
            or workspace['sources'] or workspace['episode'] is not None
            or set(case.compartments) != {*ORDER, OBSERVATION}
            or set(case.source_compartments) != {*ORDER, OBSERVATION}
            or array_digest(case.compartments[OBSERVATION]) != receipt['observation']['array_hash']
            or any(not np.array_equal(mask, case.source_compartments[name])
                   for name, mask in case.compartments.items())):
        raise ValueError('Expected the exact intact development observation package')
    return case, receipt, binding(root, receipt_path)


def prepare_inputs(root, subject, case, receipt):
    evidence = receipt['source']
    source_ct, source_labels = evidence['source_CTA_and_v1_labels']
    ct_path, label_path = (checked_file(root, row) for row in (source_ct, source_labels))
    source_ref = next(ref for ref in case.source_refs if ref.source_id == 'structural')
    if source_ref.provenance != 'observed' or source_ref.sha256 != source_ct['sha256']:
        raise ValueError('Scenario must declare the same observed CTA source')
    labels = nib.load(label_path)
    geometry = nifti_geometry(labels)
    if (labels.shape != case.mri.shape or not np.array_equal(labels.affine, case.affine)
            or geometry['spatial_units'] not in {'mm', 'unknown'} or geometry['frame_status'] != 'coded'
            or any(not np.allclose(geometry[form], case.affine, rtol=0, atol=1e-5)
                   for form in ('qform', 'sform') if geometry[form + '_code'])):
        raise ValueError('Supplied V1 annotations must match the acquired CTA grid')
    categorical = np.asanyarray(labels.dataobj)
    if (not np.isin(categorical, np.arange(41)).all()
            or not np.array_equal((categorical >= 1) & (categorical <= 34), case.compartments[ARTERIES])
            or not np.array_equal(categorical >= 35, case.compartments[VEINS])):
        raise ValueError('Native artery and vein unions must exactly reproduce the supplied labels')

    brain = evidence['anatomy_masks']['brain_estimate']
    checked_file(root, evidence['host_review'])
    if not np.array_equal(checked_mask(root, brain, case), case.compartments[BRAIN]):
        raise ValueError('Brain estimate differs from the reviewed host mask')

    skull_report_path = safe_path(root, f'{skull_display_root(subject)}/report.json')
    skull_report = read_json(skull_report_path)
    skull_review = skull_report['preparation_review']
    checked_file(root, skull_review)
    if (skull_report['family_id'] != receipt['person_id'] or skull_report['source_ct'] != source_ct
            or skull_report['primary_host']['sha256'] != evidence['host']['sha256']
            or skull_report['instrument_task_authorized'] is not False):
        raise ValueError('Broader skull estimate must bind the same CTA and host with display-only authority')
    skull = skull_report['estimated_mask']
    broader = checked_mask(root, skull, case)
    local = case.compartments[LOCAL_SKULL]
    added = broader & ~local
    if (np.any(local & ~broader) or int(np.count_nonzero(added)) != skull_report['added_voxels']
            or array_digest(broader) != skull_report['estimated_mask_array_hash']
            or any(np.any(broader & case.compartments[name]) for name in (ARTERIES, VEINS, OBSERVATION))
            or np.any(added & case.compartments[BRAIN])):
        raise ValueError('Broader skull estimate changes protected memberships or the approved geometry')
    source = {'cta': source_ct, 'publisher_v1_labels': source_labels, 'brain_estimate': brain,
              'host_review': evidence['host_review'], 'skull_estimate': skull,
              'skull_preparation_review': skull_review, 'skull_record': binding(root, skull_report_path),
              'observation_case': receipt['native_case']}
    return ct_path, label_path, source_ref.license, source, broader


def establish_study(directory, showcase_id, case, source, ct_path, label_path, license_name, broader):
    study = PatientImagingCollection.create(directory, showcase_id, timepoint=TIMEPOINT)
    scan = study.add_series(ct_path, series_id=SERIES, modality='CTA', timepoint=TIMEPOINT,
                            patient_id=showcase_id)
    if (scan['original']['sha256'] != source['cta']['sha256'] or scan['units_status'] != 'mm'
            or not np.array_equal(scan['affine'], case.affine)
            or not np.array_equal(study.array(scan['data']), case.mri)):
        raise ValueError('Native study changed the acquired CTA image, grid or bytes')
    scan.update(label='TopBrain CTA · acquired baseline before simulated contour',
                publisher_dataset='https://zenodo.org/records/21972006', source_license=license_name)
    supplied = study.original(label_path)
    if supplied['sha256'] != source['publisher_v1_labels']['sha256']:
        raise ValueError('Native study changed the supplied annotation original')
    label_units = nib.load(label_path).header.get_xyzt_units()[0]
    masks = {**case.compartments, SKULL: broader}
    rows = [
        (BRAIN, 'brain', 'Estimated brain envelope', '#d09bb6',
         {'kind': 'image_derived_estimate', 'source': source['brain_estimate']}),
        (SKULL, 'skull', 'Estimated mineralized skull', '#ddc5a9',
         {'kind': 'image_derived_estimate', 'source': source['skull_estimate'],
          'evidence_use': 'reviewed only for display'}),
        (ARTERIES, 'artery', 'Supplied TopBrain v1 arteries', '#f06455',
         {'kind': 'supplied_annotation', 'source': supplied, 'source_label_values': list(range(1, 35))}),
        (VEINS, 'vein', 'Supplied TopBrain v1 veins and sinuses', '#71b7ef',
         {'kind': 'supplied_annotation', 'source': supplied, 'source_label_values': list(range(35, 41))}),
        (OBSERVATION, 'tumor', 'Simulated tumor contour', '#f18fbd',
         {'kind': 'simulated_observation', 'source': source['observation_case'],
          'evidence_use': 'generated observation for display',
          'planning_target_candidate': False, 'observation_array_hash': array_digest(masks[OBSERVATION])}),
    ]
    for (name, kind, label, color, provenance), coverage in zip(
        rows, (LIMITS[1], LIMITS[1], LIMITS[2], LIMITS[2], LIMITS[0]),
    ):
        row = study.add_layer(masks[name].astype(np.uint8), case.affine, layer_id=name,
                              source_series=SERIES, kind=kind, role='context', color=color,
                              label=label, provenance=provenance)
        row['coverage'] = coverage
        if name in (ARTERIES, VEINS):
            row['units_status'] = label_units
        if name == OBSERVATION:
            row['review_status'] = 'generated inspection contour'
    centroid = np.mean(np.argwhere(masks[OBSERVATION]), axis=0)
    study.document['display'] = {'initial_layers': [row[0] for row in rows],
                                 'cursor_ras_mm': case.voxel_to_world(centroid).tolist()}
    study.document['missing'] = [
        'Complete brain and skull coverage', 'Complete arterial and venous segmentation',
        'MRI and tractography', 'Validated planning and instrument authority',
    ]
    study.document['data_case'] = {'source_family_id': case.metadata['data_case']['family_id'],
                                   'role': 'DEVELOPMENT', 'case_type': 'display_companion',
                                   'clinical_validation': False, 'limitations': LIMITS}
    return study.save('Prepare source-bound CTA display companion')


def verify_reopened(directory, case, broader, source):
    study = PatientImagingCollection(directory)
    scan = study.series(SERIES)
    masks = {**case.compartments, SKULL: broader}
    expected = {BRAIN, SKULL, ARTERIES, VEINS, OBSERVATION}
    if (study.identity() != study.document['snapshot_id']
            or sha256_file(study.path(scan['original']['path'])) != source['cta']['sha256']
            or not np.array_equal(study.array(scan['data']), case.mri)
            or not np.array_equal(scan['affine'], case.affine)
            or {row['id'] for row in study.document['layers']} != expected):
        raise ValueError('Native reopen changed the CTA or display inventory')
    stats = []
    for row in study.document['layers']:
        if (is_native_target(row) or row['role'] != 'context' or row['source_series'] != SERIES
                or not np.array_equal(row['affine'], case.affine)
                or not np.array_equal(study.mask(row), masks[row['id']])):
            raise ValueError('Native reopen changed a source mask, grid or display-only authority')
        if row['id'] in (ARTERIES, VEINS):
            original = row['provenance']['source']
            original_path = study.path(original['path'])
            if (sha256_file(original_path) != source['publisher_v1_labels']['sha256']
                    or row.get('units_status') != nib.load(original_path).header.get_xyzt_units()[0]):
                raise ValueError('Stored publisher annotation original or units have changed')
        count = int(np.count_nonzero(masks[row['id']]))
        stats.append({'id': row['id'], 'kind': row['kind'], 'voxels': count,
                      'volume_mm3': count * case.voxel_volume_mm3,
                      'array_sha256': array_digest(masks[row['id']])})
    return study.identity(), stats


def build(root, subject):
    case, receipt, receipt_record = read_scenario(root, subject)
    ct_path, label_path, license_name, source, broader = prepare_inputs(root, subject, case, receipt)
    source['scenario_receipt'] = receipt_record
    showcase_id = f'{case.case_id}-NATIVE-SHOWCASE'
    destination = safe_path(root, f'{OUTPUT}/{showcase_id}')
    manifest = destination / 'patient-anatomy.json'
    if destination.exists():
        report = read_json(destination / 'receipt.json')
        if report['schema'] != SCHEMA or report['source'] != source:
            raise ValueError('An existing showcase has different source evidence')
        if manifest != checked_file(root, report['manifest']):
            raise ValueError('An existing showcase points to a different manifest')
        snapshot, stats = verify_reopened(destination, case, broader, source)
        if snapshot != report['snapshot_id'] or stats != report['layers']:
            raise ValueError('An existing showcase differs from its verified receipt')
        return report

    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f'.{showcase_id}-', dir=destination.parent) as temporary:
        staging = Path(temporary) / showcase_id
        establish_study(staging, showcase_id, case, source, ct_path, label_path, license_name, broader)
        snapshot, stats = verify_reopened(staging, case, broader, source)
        report = {'schema': SCHEMA, 'case_id': showcase_id, 'source_person_id': receipt['person_id'],
                  'role': 'DEVELOPMENT', 'new_source_people': 0, 'new_synthetic_scenarios': 0,
                  'source': source, 'manifest': {'path': str(manifest.relative_to(root)),
                                                'sha256': sha256_file(staging / manifest.name)},
                  'snapshot_id': snapshot, 'layers': stats, 'limits': LIMITS}
        atomic_json(staging / 'receipt.json', report)
        checked_root(root, expected_uuid=UUID)
        staging.rename(destination)
    return report


def register(root, library, report):
    manifest = checked_file(root, report['manifest'])
    library.mkdir(parents=True, exist_ok=True)
    link = library / f'{report["case_id"]}.anatomy.json'
    if not link.exists() and not link.is_symlink():
        link.symlink_to(manifest)
    if not link.is_symlink() or link.resolve() != manifest:
        raise ValueError(f'Conflicting Cases card: {link}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--subject', action='append', choices=SUBJECTS)
    parser.add_argument('--register', type=Path, help='Optional local Cases directory for native study links')
    args = parser.parse_args()
    root = checked_root(args.root, expected_uuid=UUID, required_bytes=3 * 1024**3)
    with reconstruction_lock():
        for subject in dict.fromkeys(args.subject or SUBJECTS):
            report = build(root, subject)
            if args.register is not None:
                register(root, args.register, report)
            print(report['manifest']['path'], flush=True)


if __name__ == '__main__':
    main()
