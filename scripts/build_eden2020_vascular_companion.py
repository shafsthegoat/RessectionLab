#!/usr/bin/env python3
"""Stage one restricted EDEN2020 native TOF vascular review study on the verified HDD."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import tempfile

import numpy as np

from resectionlab.anatomy_display import display_affine
from resectionlab.anatomy_evidence import vascular_evidence_supported
from resectionlab.anatomy_package import referenced_files
from resectionlab.data_root import ROOT_ENV, atomic_json, checked_root, safe_path, sha256_file
from resectionlab.patient_anatomy import PatientImagingCollection, projection_or_localizer

HDD_ROOT = Path('/Volumes/shafrir/old coding projects/RessectionLabData')
HDD_UUID = '9CC3623C-E38B-3EEE-B6BB-B9A06C193700'
SOURCE = Path('prepared/Eden 2020 ResectionLab Projects (restricted)')
RUN = Path('qc/eden2020-glioma-5714812/vascular-reconstruction-20261123-v1')
MANIFEST = 'patient-anatomy.json'
LINKED_MANIFEST = 'vascular-review.anatomy.json'
SCHEMA = 'eden2020-restricted-vascular-companion-v1'
CANDIDATE = 'vesselfm-cleanup'
PARTITION_CODES = ('background', 'retained', 'uncertain_retained', 'hidden_candidate')
COPY_CHUNK_BYTES = 16 * 1024**2


def native_vessels(source):
    scans = [row for row in source.document['series']
             if row['modality'] == 'TOF' and not projection_or_localizer(row)]
    if len(scans) != 1:
        raise ValueError('Expected one original native TOF source volume')
    scan = scans[0]
    source.require_native_geometry(scan)
    baselines = [row for row in source.document['layers']
                 if row['source_series'] == scan['id'] and row['kind'] == 'vessels'
                 and row.get('provenance', {}).get('model') in {'vesselfm', 'vesselfm-base'}
                 and row.get('availability', 'available') == 'available'
                 and row.get('probability')]
    if len(baselines) != 1:
        raise ValueError('Expected one available native vesselFM mask with retained scores')
    baseline = baselines[0]
    if (baseline['label_value'] != 1 or baseline['data']['dtype'] != 'uint8'
            or not np.array_equal(baseline['affine'], scan['affine'])
            or baseline['data']['shape'] != scan['data']['shape']
            or baseline['probability']['shape'] != scan['data']['shape']):
        raise ValueError('TOF, vesselFM mask and scores must share the exact acquired grid')
    return scan, baseline


def research_layer(row):
    result = deepcopy(row)
    result.update(role='context', revision=0)
    result.pop('preparation_input', None)
    result['provenance'].update(evidence_use='context_only', clinical_validation=False,
                                simulation_admission=False)
    return result


def structural_context(source, tof):
    reference = source.series(source.document['reference_series'])
    prior = source.document['transforms'].get(tof['id'])
    if (reference['id'] == tof['id'] or not prior
            or prior['status'] not in {'unreviewed', 'reviewed', 'source-provided'}
            or prior.get('quality', {}).get('status') in {'failed', 'rejected', 'refused'}):
        raise ValueError('No usable prior structural-to-TOF alignment preview')
    source.registration_matrix(reference['id'], require_review=True, require_geometry=True)
    tof_to_reference = source.registration_matrix(tof['id'], require_geometry=True)
    transform = {
        'status': 'unreviewed',
        'method': 'inverse of prior source-bound TOF-to-structural alignment',
        'matrix': np.linalg.inv(tof_to_reference).tolist(),
        'uncertainty_mm': None,
        'direction': 'source-RAS-mm-to-reference-RAS-mm',
        'source_sha256': reference['original']['sha256'],
        'reference_sha256': tof['original']['sha256'],
        'prior_status': prior['status'],
        'prior_method': prior.get('method'),
        'prior_transform_digest': prior.get('transform_digest'),
        'quality': {'status': prior.get('quality', {}).get('status', 'unassessed'),
                    'warnings': deepcopy(prior.get('quality', {}).get('warnings', [])),
                    'meaning': 'Prior automated checks in the original registration direction'},
    }
    preferred = set(source.document.get('display', {}).get('initial_layers', []))
    layers = []
    for kind in ('brain', 'skull'):
        available = [row for row in source.document['layers']
                     if row['kind'] == kind and row['source_series'] == reference['id']
                     and row.get('availability', 'available') == 'available']
        if available:
            row = research_layer(min(available, key=lambda item: item['id'] not in preferred))
            row['label'] += ' · unreviewed TOF alignment'
            row.update(transform_dependencies=[reference['id']], source_transform_at_admission=None)
            layers.append(row)
    if not layers:
        raise ValueError('No existing structural brain or skull context to show')
    return reference, transform, layers


def package_members(document):
    members = {}
    for name, digest in referenced_files(document):
        if members.setdefault(name, digest) != digest:
            raise ValueError(f'Conflicting package member identities: {name}')
    return members


def prepare_members(source, destination, *, native_arrays=()):
    native_arrays = {ref['path']: ref for ref in native_arrays}
    for name, digest in package_members(destination.document).items():
        target = destination.path(name)
        if source.root != destination.root:
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.path(name).open('rb') as original, target.open('xb') as output:
                shutil.copyfileobj(original, output, length=COPY_CHUNK_BYTES)
        if name in native_arrays:
            destination.array(native_arrays[name])
        elif sha256_file(target) != digest:
            raise ValueError(f'Package member differs from its source record: {name}')


def partition_membership(partition, source_mask):
    if partition.dtype != np.uint8 or partition.shape != source_mask.shape:
        raise ValueError('Partition must contain native-grid uint8 codes')
    counts = np.zeros(len(PARTITION_CODES), dtype=np.int64)
    slab_size = max(1, 2_000_000 // (partition.shape[1] * partition.shape[2]))
    for start in range(0, partition.shape[0], slab_size):
        slab = slice(start, start + slab_size)
        labels = partition[slab]
        baseline = source_mask[slab]
        if labels.max() > 3 or baseline.max() > 1 or not np.array_equal(labels > 0, baseline == 1):
            raise ValueError('Partition codes must account for precisely the frozen original foreground')
        for code in range(len(counts)):
            counts[code] += np.count_nonzero(labels == code)
    if counts[1] + counts[2] == 0:
        raise ValueError('Partition leaves no vessel candidates visible')
    return dict(zip(PARTITION_CODES, map(int, counts)))


def verify_partition_report(report, counts):
    analysis = report['analysis']
    expected = {
        'retained_voxels': counts['retained'],
        'uncertain_voxels': counts['uncertain_retained'],
        'hidden_voxels': counts['hidden_candidate'],
        'foreground_voxels': sum(counts[name] for name in PARTITION_CODES[1:]),
        'new_positive_voxels': 0,
    }
    for key, value in expected.items():
        if key in analysis and analysis[key] != value:
            raise ValueError(f'Case report disagrees with the native partition: {key}')
    if report['status'] == 'candidate' and not counts['hidden_candidate']:
        raise ValueError('Candidate report has no hidden foreground')


def candidate_layer(study, baseline, partition, counts, source_binding):
    row = deepcopy(baseline)
    row.update(id=CANDIDATE, label='vesselFM cleanup · research display candidate',
               coverage='Native vesselFM positives selected for research display review')
    row.pop('probability')
    row.pop('score_meaning', None)
    if counts['hidden_candidate']:
        candidate = np.empty(partition.shape, dtype=np.uint8)
        for start in range(0, partition.shape[0], 8):
            slab = slice(start, start + 8)
            labels = partition[slab]
            candidate[slab] = (labels > 0) & (labels < 3)
        row['data'] = study.put_array(candidate)
    row['provenance'] = {
        'kind': 'image_derived_estimate',
        'method': 'native vesselFM foreground visibility partition',
        'source_series': baseline['source_series'],
        'source': source_binding['source_tof_sha256'],
        'evidence_use': 'context_only',
        'clinical_validation': False,
        'simulation_admission': False,
        'vascular_cleanup': {
            'source_layer_id': baseline['id'],
            'source_mask_sha256': baseline['data']['sha256'],
            'source_probability_sha256': baseline['probability']['sha256'],
            'result_mask_sha256': row['data']['sha256'],
            'partition_sha256': source_binding['partition_sha256'],
        },
    }
    return row


def case_partition(root, subject):
    case = RUN / 'cases' / subject
    proposed_path = safe_path(root, case / 'report.json')
    proposed = json.loads(proposed_path.read_text())
    reviewed_path = safe_path(root, case / 'reviewed' / 'report.json')
    reviewed_partition = safe_path(root, case / 'reviewed' / 'partition.npy')
    if reviewed_path.is_file() != reviewed_partition.is_file():
        raise ValueError('Reviewed case requires both the report and partition')
    if not reviewed_path.is_file():
        if safe_path(root, case / '.companion-hold').exists():
            raise ValueError('Source review is pending for this case')
        return proposed, proposed_path, safe_path(root, case / 'partition.npy'), None

    reviewed = json.loads(reviewed_path.read_text())
    manual_review = reviewed.get('manual_review', {})
    if (manual_review.get('parent_partition_sha256') != proposed['partition_sha256']
            or manual_review.get('parent_report_sha256') != sha256_file(proposed_path)):
        raise ValueError('Reviewed partition does not bind its frozen proposal')
    return reviewed, reviewed_path, reviewed_partition, proposed


def companion_directory(root, subject, partition_sha256, proposed):
    destination = safe_path(root, RUN / 'studies' / subject)
    if not proposed or not destination.exists():
        return destination
    existing = json.loads((destination / 'receipt.json').read_text())['inputs']['partition_sha256']
    if existing == partition_sha256:
        return destination
    if existing != proposed['partition_sha256']:
        raise ValueError('Existing companion does not match the frozen or reviewed partition')
    return safe_path(root, RUN / 'studies' / f'{subject}-reviewed-{partition_sha256[:12]}')


def verified_existing(destination, expected, case_report):
    study = PatientImagingCollection(destination)
    receipt = json.loads((destination / 'receipt.json').read_text())
    if (receipt.get('schema') != SCHEMA or receipt.get('inputs') != expected
            or receipt.get('snapshot_id') != study.identity()
            or receipt['manifest']['path'] != str(RUN / 'studies' / destination.name / MANIFEST)
            or receipt['manifest']['sha256'] != sha256_file(study.path(MANIFEST))):
        raise ValueError('Existing companion inputs or snapshot differ from its receipt')
    verify_partition_report(case_report, receipt['partition_counts'])
    for name, digest in package_members(study.document).items():
        if sha256_file(study.path(name)) != digest:
            raise ValueError(f'Existing companion member differs from its receipt: {name}')
    return receipt


def stage_derivatives(study, source, scan, baseline, report, partition_path, source_binding):
    prepare_members(source, study, native_arrays=(scan['data'], baseline['data']))
    partition = np.load(partition_path, mmap_mode='r', allow_pickle=False)
    counts = partition_membership(partition, study.array(baseline['data']))
    verify_partition_report(report, counts)
    study.document['layers'].insert(1, candidate_layer(study, baseline, partition, counts, source_binding))
    study.document['vascular_review']['partition_counts'] = counts
    if not scan.get('preview_pyramid'):
        study.preview_pyramid(scan['id'], levels=(2,))
    return counts


def verify_scope(study, scan, source_baseline, context_layers):
    if (not np.array_equal(study.patient_affine(study.series(scan['id'])), scan['affine'])
            or study.layer(source_baseline['id'])['probability'] != source_baseline['probability']
            or any(vascular_evidence_supported(study, row) for row in study.document['layers']
                   if row['kind'] == 'vessels')):
        raise ValueError('Reopened companion changed native geometry or vascular evidence scope')
    if not all(display_affine(study, study.layer(row['id']), preview=True)[1] for row in context_layers):
        raise ValueError('Structural context must remain an unreviewed display preview')


def publication_receipt(study, manifest, published_path, root, scan, baseline, inputs, counts):
    return {
        'schema': SCHEMA,
        'inputs': inputs,
        'source_grid_shape': scan['data']['shape'],
        'snapshot_id': study.identity(),
        'baseline_layer_id': baseline['id'],
        'candidate_layer_id': CANDIDATE,
        'partition_counts': counts,
        'manifest': {'path': str(published_path.relative_to(root)), 'sha256': sha256_file(manifest)},
    }


def linked_paths(root, source_path, subject, partition_sha256):
    manifest = source_path.with_name(LINKED_MANIFEST)
    receipt_dir = safe_path(root, RUN / 'linked-studies' / subject)
    if manifest.is_symlink():
        raise ValueError('Linked companion manifest must be a regular study file')
    if manifest.exists():
        document = json.loads(manifest.read_text())
        previous = document.get('vascular_review', {})
        if previous.get('linked_source') is not True:
            raise ValueError('Existing alternate manifest does not identify a linked vascular companion')
        if previous['partition']['sha256'] != partition_sha256:
            suffix = partition_sha256[:12]
            manifest = source_path.with_name(f'vascular-review.{suffix}.anatomy.json')
            receipt_dir = safe_path(root, RUN / 'linked-studies' / f'{subject}-reviewed-{suffix}')
            if manifest.is_symlink():
                raise ValueError('Linked companion manifest must be a regular study file')
    return manifest, receipt_dir / 'receipt.json'


def build_linked(root, subject, source_path, source, scan, baseline, document, report,
                 partition_path, inputs, context_layers, required_bytes):
    checked_root(root, expected_uuid=HDD_UUID, required_bytes=required_bytes)
    lock_dir = safe_path(root, RUN / 'linked-studies')
    lock_dir.mkdir(parents=True, exist_ok=True)
    with (lock_dir / f'.{subject}.lock').open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        manifest, receipt_path = linked_paths(root, source_path, subject, inputs['partition_sha256'])
        if manifest.exists():
            reopened = PatientImagingCollection(source.root, json.loads(manifest.read_text()))
            review = reopened.document.get('vascular_review', {})
            if (review.get('linked_source') is not True or review.get('context', False) != inputs['context']
                    or review['source_manifest']['sha256'] != inputs['source_manifest_sha256']
                    or review['partition']['sha256'] != inputs['partition_sha256']):
                raise ValueError('Existing linked companion differs from its frozen inputs')
            counts = review['partition_counts']
            verify_partition_report(report, counts)
            for name, digest in package_members(reopened.document).items():
                if sha256_file(reopened.path(name)) != digest:
                    raise ValueError(f'Linked companion member differs from its source record: {name}')
        else:
            study = PatientImagingCollection(source.root, document)
            counts = stage_derivatives(study, source, scan, baseline, report, partition_path, inputs)
            study.document['revision'] = 1
            study.document['history'] = [{
                'event': 'Stage read-only linked native TOF vascular review',
                'author': 'local researcher',
                'at': datetime.now(timezone.utc).isoformat(),
            }]
            study.document['snapshot_id'] = study.identity()
            reopened = PatientImagingCollection(source.root, study.document)
            verify_scope(reopened, scan, baseline, context_layers)
            checked_root(root, expected_uuid=HDD_UUID)
            if sha256_file(source_path) != inputs['source_manifest_sha256']:
                raise ValueError('Frozen source manifest changed while staging the linked companion')
            atomic_json(manifest, reopened.document)
        verify_scope(reopened, scan, baseline, context_layers)
        receipt = publication_receipt(reopened, manifest, manifest, root, scan, baseline, inputs, counts)
        if receipt_path.exists():
            if json.loads(receipt_path.read_text()) != receipt:
                raise ValueError('Existing linked companion receipt differs from its manifest')
        else:
            receipt_path.parent.mkdir(parents=True, exist_ok=True)
            atomic_json(receipt_path, receipt)
        return receipt


def build(root, subject, *, context=False, linked_source=False):
    if not re.fullmatch(r'PTNT_[0-9]+', subject):
        raise ValueError('Provide a restricted EDEN2020 subject id such as PTNT_4509')
    source_path = safe_path(root, SOURCE / subject / MANIFEST)
    source = PatientImagingCollection(source_path.parent)
    source_doc = source.document
    if (source_doc['patient_id'] != f'EDEN2020:{subject}' or source_doc.get('exposure') != 'private'
            or source_doc.get('source_url') != 'https://zenodo.org/records/5714812'):
        raise ValueError('Expected the same-patient restricted EDEN2020 source manifest')
    scan, source_baseline = native_vessels(source)
    report, report_path, partition_path, proposed = case_partition(root, subject)
    status = report.get('status')
    if status not in {'candidate', 'retained_baseline'}:
        raise ValueError('Case report has no usable completed vascular partition')
    source_binding = {
        'subject': subject,
        'source_manifest_sha256': sha256_file(source_path),
        'source_tof_sha256': scan['original']['sha256'],
        'source_mask_sha256': source_baseline['data']['sha256'],
        'source_probability_sha256': source_baseline['probability']['sha256'],
        'partition_sha256': sha256_file(partition_path),
    }
    for key, value in source_binding.items():
        if report.get(key) != value:
            raise ValueError(f'Case report does not bind the exact partition input: {key}')
    inputs = {**source_binding, 'context': context, 'partition_status': status}
    if not linked_source:
        destination = companion_directory(root, subject, source_binding['partition_sha256'], proposed)
        if destination.exists():
            return verified_existing(destination, inputs, report)

    baseline = research_layer(source_baseline)
    baseline.update(label='vesselFM baseline · original native prediction',
                    transform_dependencies=[scan['id']], source_transform_at_admission='reference')
    series, layers, context_layers = [scan], [baseline], []
    transforms = {scan['id']: {'status': 'reference', 'matrix': np.eye(4).tolist(), 'uncertainty_mm': None}}
    if context:
        structural, alignment, context_layers = structural_context(source, scan)
        series.append(structural)
        transforms[structural['id']] = alignment
        layers.extend(context_layers)
    document = {
        'schema': source_doc['schema'],
        'patient_id': source_doc['patient_id'],
        'study_timepoint': source_doc.get('study_timepoint', scan['timepoint']),
        'revision': 0,
        'history': [],
        'label': f'EDEN2020 {subject} · native TOF vascular review',
        'source_url': source_doc['source_url'],
        'role': 'DEVELOPMENT',
        'exposure': 'private',
        'clinical_validation': False,
        'training_admitted': False,
        'simulation_admission': False,
        'reference_series': scan['id'],
        'series': series,
        'layers': layers,
        'transforms': transforms,
        'missing': ['Vascular classification, complete vascular coverage and simulation admission are unestablished.'],
        'display': {'initial_layers': [CANDIDATE if status == 'candidate' else baseline['id']]},
        'vascular_review': {
            'schema': SCHEMA,
            'use': 'private source-bound vascular research display',
            'source_manifest': {'hdd_relative_path': str(source_path.relative_to(root)),
                                'sha256': source_binding['source_manifest_sha256'],
                                'snapshot_id': source.identity()},
            'partition_report': {'hdd_relative_path': str(report_path.relative_to(root))},
            'partition': {'hdd_relative_path': str(partition_path.relative_to(root)),
                          'sha256': source_binding['partition_sha256']},
            **({'manual_review': deepcopy(report['manual_review'])} if proposed else {}),
            'status': status,
            **({'linked_source': True, 'context': context} if linked_source else {}),
            'source_grid_preserved': True,
            'score_meaning': 'uncalibrated model score',
        },
    }
    preview_shape = [(axis + 1) // 2 for axis in scan['data']['shape']]
    preview_bytes = 0 if scan.get('preview_pyramid') else 4 * int(np.prod(preview_shape))
    required_bytes = partition_path.stat().st_size + preview_bytes + 16 * 1024**2
    if linked_source:
        return build_linked(root, subject, source_path, source, scan, baseline, document, report,
                            partition_path, inputs, context_layers, required_bytes)
    source_bytes = sum(source.path(name).stat().st_size for name in package_members(document))
    checked_root(root, expected_uuid=HDD_UUID,
                 required_bytes=source_bytes + required_bytes)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent, prefix=f'.{subject}-vascular-') as temporary:
        stage = Path(temporary) / 'study'
        stage.mkdir()
        study = PatientImagingCollection(stage, document)
        counts = stage_derivatives(study, source, scan, baseline, report, partition_path, source_binding)
        study.save('Stage native TOF and reversible vesselFM cleanup for research display review')
        reopened = PatientImagingCollection(stage)
        verify_scope(reopened, scan, source_baseline, context_layers)
        receipt = publication_receipt(reopened, reopened.path(MANIFEST), destination / MANIFEST,
                                      root, scan, baseline, inputs, counts)
        atomic_json(stage / 'receipt.json', receipt)
        checked_root(root, expected_uuid=HDD_UUID)
        if sha256_file(source_path) != source_binding['source_manifest_sha256']:
            raise ValueError('Frozen source manifest changed while staging the companion')
        stage.rename(destination)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(os.environ.get(ROOT_ENV) or HDD_ROOT))
    parser.add_argument('--subject', required=True, help='Restricted source subject, such as PTNT_4509')
    parser.add_argument('--context', action='store_true', help='Include unreviewed prior structural preview')
    parser.add_argument('--linked-source', action='store_true',
                        help='Publish a read-only companion referencing exact arrays in the source study')
    args = parser.parse_args()
    root = checked_root(args.root, expected_uuid=HDD_UUID)
    report = build(root, args.subject, context=args.context, linked_source=args.linked_source)
    print(safe_path(root, report['manifest']['path']))


if __name__ == '__main__':
    main()
