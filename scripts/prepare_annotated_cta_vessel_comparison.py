#!/usr/bin/env python3
"""Save portable CTA comparison studies with native publisher vessel anatomy."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import zlib

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize

from benchmark_annotated_cta_vessels import BASELINE, EVIDENCE, inputs, prediction, read, scores
from build_cta_native_showcases import register
from data_integrated_hosts import UUID, binding, checked_file
from resectionlab.anatomy_display import source_lattice_annotation_preview
from resectionlab.anatomy_package import open_package, save_package
from resectionlab.data_root import atomic_json, checked_root, safe_path
from resectionlab.patient_anatomy import PatientImagingCollection

DESTINATION = 'prepared/annotated-cta-vascular-comparison-v1'
TOPBRAIN = 'https://zenodo.org/records/21972006'
TOPCOW = 'https://zenodo.org/records/15692630'
TIMEPOINT = 'source-session-unknown'
DISTAL = {13, 14, *range(16, 23), *range(25, 33)}
ROI_SCOPE = 'Publisher Circle of Willis evaluation box; native CTA lattice'


def within(mask, region):
    cropped = np.zeros(mask.shape, dtype=np.uint8)
    cropped[region] = mask[region]
    return cropped


def pick_branch(agreement):
    options = [(int(key), row['skeleton_points'] - row['skeleton_points_supported'])
               for key, row in agreement.items() if int(key) in DISTAL]
    value, missed = max(options, key=lambda item: (item[1], -item[0]), default=(0, 0))
    if missed == 0:
        raise ValueError('No documented distal annotation with unsupported baseline centerline points')
    return value


def branch_view(mask, baseline, region, spacing):
    xyz = np.argwhere(mask)
    padding = np.ceil(8 / spacing).astype(int)
    lower = np.maximum(0, xyz.min(0) - padding)
    upper = np.minimum(mask.shape, xyz.max(0) + padding + 1)
    window = tuple(slice(int(a), int(b)) for a, b in zip(lower, upper))
    local = mask[window]
    points = np.argwhere(skeletonize(local))
    distances = ndi.distance_transform_edt(~baseline[window], sampling=spacing)
    centroid = xyz.mean(0) - lower
    # Show a documented gap near the body of the labelled branch, away from the CoW box.
    native_points = points + lower
    outside = ~np.logical_and.reduce([(native_points[:, axis] >= slab.start) & (native_points[:, axis] < slab.stop)
                                     for axis, slab in enumerate(region)])
    gaps = points[outside & (distances[tuple(points.T)] > 2)]
    if not len(gaps):
        gaps = points[outside]
    if not len(gaps):
        raise ValueError('The selected branch has no centreline outside the publisher CoW box')
    center = gaps[np.argmin(np.linalg.norm((gaps - centroid) * spacing, axis=1))] + lower
    return window, center


def label_map(root):
    metadata = read(safe_path(root, 'catalog/source-research-topbrain-remote.json'))
    name, content = next((key, value) for key, value in metadata['metadata'].items()
                         if key.endswith('/labels_topbrain_v1_ct.json'))
    member = next(row for row in metadata['members'] if row['name'] == name)
    payload = content.encode('utf-8')
    if len(payload) != member['bytes'] or f'{zlib.crc32(payload):08x}' != member['crc32']:
        raise ValueError('Publisher CTA label dictionary differs from its retained ZIP index')
    labels = {value: key for key, value in json.loads(content)['labels'].items()}
    if set(labels) != set(range(41)):
        raise ValueError('Publisher TopBrain v1 CTA label dictionary differs')
    return labels, name


def check_layer_geometry(study, layer):
    if layer.get('units_status') == 'unknown':
        if not source_lattice_annotation_preview(study, layer, study.series('cta')):
            raise ValueError(f'Layer {layer["id"]} must match the exact acquired CTA lattice')
    else:
        study.require_native_geometry(layer)


def populate(directory, root, subject, protocol, coverage, report, evidence):
    scan, topbrain, cow, region = inputs(root, subject, protocol, coverage)
    source = protocol['sources'][subject]
    values = scan.get_fdata(dtype=np.float32)
    spacing = nib.affines.voxel_sizes(scan.affine)
    choices = (BASELINE, report['choice'])
    predictions = {}
    for choice in choices:
        settings = protocol['selection_candidates'][choice]
        blend = settings['blend']
        run = report['source_inference'][f'{subject}-{blend}']
        probability = scores(root, run, source['image'], scan, blend)
        predictions[choice] = prediction(probability, values, settings)
    roi_baseline, roi_candidate = (within(predictions[choice], region) for choice in choices)
    roi_reference = within((topbrain > 0) | (cow > 0), region)
    baseline = predictions[BASELINE]
    supported = report['cases'][subject][BASELINE]['topbrain_positive_path_agreement']
    branch_id = pick_branch(supported)
    branch = topbrain == branch_id
    window, branch_cursor = branch_view(branch, baseline, region, spacing)
    branch_baseline = within(baseline, window)
    added = int(np.count_nonzero(branch & ~baseline))
    if not branch_baseline.any() or not added:
        raise ValueError('Branch comparison requires visible model anatomy and new supplied voxels')
    labels, legend_name = label_map(root)
    branch_name = labels[branch_id]
    del predictions, probability, values

    case = f'TopCoW-{subject}-CTA-VESSEL-COMPARISON-v1'
    study = PatientImagingCollection.create(directory, case, timepoint=TIMEPOINT)
    series = study.add_series(checked_file(root, source['image']), series_id='cta', modality='CTA', timepoint=TIMEPOINT)
    if series['original']['sha256'] != source['image']['sha256'] or not np.array_equal(series['affine'], scan.affine):
        raise ValueError('Imported CTA changed the frozen source image or physical lattice')
    study.require_native_geometry(series)
    series.update(label='Acquired CTA', publisher_dataset=TOPBRAIN,
                  source_license='Noncommercial use with source attribution')
    label_path = checked_file(root, source['topbrain'])
    published = study.original(label_path)
    label_units = nib.load(label_path).header.get_xyzt_units()[0]
    original_cow = study.original(checked_file(root, coverage['cases'][subject]['cow']))
    roi_original = study.original(checked_file(root, coverage['cases'][subject]['roi']))
    bound_verification = study.original(evidence)
    if published['sha256'] != source['topbrain']['sha256']:
        raise ValueError('Imported publisher annotation bytes changed')

    def add(id_, label, mask, color, scope, provenance, *, kind='vessels', value=1):
        layer = study.add_layer(mask, scan.affine, layer_id=id_, label=label, source_series='cta',
                                kind=kind, role='context', color=color, label_value=value, provenance=provenance)
        layer['coverage'] = scope
        if provenance.get('kind') == 'observed_source_annotation':
            layer['units_status'] = label_units
        check_layer_geometry(study, layer)

    estimated = {'kind': 'image_derived_estimate', 'benchmark': bound_verification, 'evidence_use': 'context_only'}
    supplied = {'kind': 'observed_source_annotation', 'source': published, 'publisher_dataset': TOPBRAIN,
                'publisher_label_dictionary': legend_name, 'evidence_use': 'context_only'}
    for id_, choice, mask in zip(('roi-baseline', 'roi-candidate'), choices, (roi_baseline, roi_candidate)):
        add(id_, choice, mask, '#ef7991', ROI_SCOPE, {**estimated, 'benchmark_setting': choice})
    add('roi-reference', 'Publisher vessel reference', roi_reference, '#ef7991', ROI_SCOPE,
        {**supplied, 'original_cow': original_cow, 'roi_definition': roi_original,
         'compared_union': 'TopBrain v1 positive labels and original TopCoW positive labels'})
    add('local-baseline', 'Baseline vessels near ' + branch_name, branch_baseline, '#ef7991',
        'Local branch comparison window', {**estimated, 'benchmark_setting': BASELINE})
    add('distal-branch', 'Supplied ' + branch_name, topbrain, '#65e9ba', 'Native TopBrain v1 named landmark',
        {**supplied, 'source_label_value': branch_id}, kind='artery', value=branch_id)
    add('publisher-arteries', 'Supplied TopBrain arteries', ((topbrain > 0) & (topbrain <= 34)).astype(np.uint8),
        '#ef7991', 'Native TopBrain v1 positive arterial landmarks',
        {**supplied, 'source_label_values': list(range(1, 35))}, kind='artery')
    add('publisher-veins', 'Supplied TopBrain veins and sinuses', (topbrain >= 35).astype(np.uint8),
        '#86bff5', 'Native TopBrain v1 positive venous landmarks',
        {**supplied, 'source_label_values': list(range(35, 41))}, kind='vein')
    roi_center = np.argwhere(roi_reference)
    roi_center = roi_center[np.argmin(np.linalg.norm((roi_center - roi_center.mean(0)) * spacing, axis=1))]
    roi_cursor, distal_cursor = (nib.affines.apply_affine(scan.affine, p).tolist() for p in (roi_center, branch_cursor))
    study.document.update(
        display={'initial_layers': ['publisher-arteries', 'publisher-veins'], 'cursor_ras_mm': roi_cursor},
        missing=['Exhaustive vessel annotations', 'Surgical planning and instrument authority'],
        data_case={'source_family_id': f'TopCoW:{subject}', 'role': 'DEVELOPMENT', 'case_type': 'display_companion',
                   'clinical_validation': False, 'publisher_dataset': TOPBRAIN},
        vascular_comparison={'verification': bound_verification, 'original_cow_release': TOPCOW,
                             'settings_promoted': report['promote_roi_variant'], 'roi_ijk': [[s.start, s.stop] for s in region],
                             'labelled_distal_branch': {'label': branch_name, 'id': branch_id, 'native_voxels': int(branch.sum()),
                                                       'voxels_absent_in_baseline': added, 'outside_cow_roi': supported[str(branch_id)]},
                             'views': [
                                 {'focus': 'roi-reference', 'cursor_ras_mm': roi_cursor, 'states': [
                                     {'name': 'cow-baseline', 'layers': ['roi-baseline']},
                                     {'name': 'cow-candidate', 'layers': ['roi-candidate']},
                                     {'name': 'cow-publisher', 'layers': ['roi-reference']}]},
                                 {'focus': 'distal-branch', 'cursor_ras_mm': distal_cursor, 'states': [
                                     {'name': 'distal-baseline', 'layers': ['local-baseline']},
                                     {'name': 'distal-supplied', 'layers': ['local-baseline', 'distal-branch']}]}]},
    )
    study.save('Build annotation-bound CTA vessel comparison')
    return study


def build(root, subject, library=None):
    qc = safe_path(root, EVIDENCE)
    protocol, coverage = (read(qc / name) for name in ('protocol.json', 'annotation-coverage.json'))
    evidence = qc / 'verification.json'
    report = read(evidence)
    if subject not in report['cases'] or subject not in protocol['split']['verification']:
        raise ValueError('Only frozen verification subjects are eligible for comparison studies')
    for record in [*report['inputs'].values(), report['frozen_selection']]:
        checked_file(root, record)
    case = f'TopCoW-{subject}-CTA-VESSEL-COMPARISON-v1'
    final = safe_path(root, Path(DESTINATION) / case)
    final.parent.mkdir(parents=True, exist_ok=True)
    if final.exists():
        receipt = read(final / 'receipt.json')
        for key in ('manifest', 'portable', 'verification'):
            checked_file(root, receipt[key])
        if receipt['verification'] != binding(root, evidence):
            raise ValueError('Existing companion binds a different verification receipt')
    else:
        with TemporaryDirectory(dir=final.parent, prefix='.cta-vascular-') as scratch:
            stage = Path(scratch) / case
            study = populate(stage, root, subject, protocol, coverage, report, evidence)
            archive = stage / f'{case}.rsworkspace'
            save_package(study, archive)
            cold = PatientImagingCollection(open_package(archive, Path(scratch) / 'cold').parent)
            if cold.document != study.document:
                raise ValueError('Portable study did not reopen with the same snapshot')
            # Package opening checks every referenced file hash. Check decoded anatomy as well.
            for layer in cold.document['layers']:
                check_layer_geometry(cold, layer)
                if not np.array_equal(cold.mask(layer), study.mask(layer)):
                    raise ValueError(f'Cold reopen changed {layer["id"]}')
            receipt = {'case_id': case, 'snapshot_id': study.identity(), 'verification': binding(root, evidence),
                       'branch': study.document['vascular_comparison']['labelled_distal_branch'], 'cold_reopen': True}
            for key, file in [('manifest', stage / 'patient-anatomy.json'), ('portable', archive)]:
                receipt[key] = {**binding(root, file), 'path': str((final / file.name).relative_to(root))}
            atomic_json(stage / 'receipt.json', receipt)
            checked_root(root, expected_uuid=UUID)
            stage.rename(final)
    if library is not None:
        register(root, library, receipt)
    print(final / 'patient-anatomy.json', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--subject', action='append', required=True)
    parser.add_argument('--register', type=Path)
    args = parser.parse_args()
    data_root = checked_root(args.root, expected_uuid=UUID, required_bytes=3 * 1024**3)
    for subject in dict.fromkeys(args.subject):
        build(data_root, subject, args.register)
