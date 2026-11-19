#!/usr/bin/env python3
"""Compare fixed native CTA vessel settings within publisher-annotated coverage."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize

from data_integrated_hosts import UUID, binding, checked_file
from resectionlab.anatomy_evaluation import evaluate
from resectionlab.data_root import atomic_json, checked_root, safe_path
from resectionlab.poc_models import MODEL_CONFIG

EVIDENCE = 'qc/annotated-cta-vascular-benchmark-v1'
BASELINE = 'gaussian-050'
STRUCTURE = ndi.generate_binary_structure(3, 3)


def read(path):
    return json.loads(path.read_text())


def publisher_roi(text, shape):
    size = re.search(r'^Size \(Voxels\):\s+(\d+)\s+(\d+)\s+(\d+)\s*$', text, re.M)
    start = re.search(r'^Location \(Voxels\):\s+(\d+)\s+(\d+)\s+(\d+)\s*$', text, re.M)
    if not size or not start or len(shape) != 3:
        raise ValueError('Expected publisher voxel size and location')
    size, start = (np.asarray(row.groups(), dtype=int) for row in (size, start))
    stop = start + size
    if np.any(size <= 0) or np.any(stop > shape):
        raise ValueError('Annotation box exceeds its original CTA lattice')
    return tuple(slice(int(i), int(j)) for i, j in zip(start, stop))


def native_image(path, scan):
    image = nib.load(path)
    if image.shape != scan.shape or not np.allclose(image.affine, scan.affine, rtol=0, atol=1e-5):
        raise ValueError('An annotation or output is outside the bound native CTA lattice')
    return image


def inputs(root, case, protocol, coverage):
    sources = protocol['sources'][case]
    row = coverage['cases'][case]
    scan = nib.load(checked_file(root, sources['image']))
    if sources['image'] != row['input'] or tuple(nib.aff2axcodes(scan.affine)) != ('L','P','S'):
        raise ValueError('Publisher ROI must use the exact original LPS-oriented CTA')
    region = publisher_roi(checked_file(root, row['roi']).read_text(), scan.shape)
    v1 = np.asanyarray(native_image(checked_file(root, sources['topbrain']), scan).dataobj)
    cow = np.asanyarray(native_image(checked_file(root, row['cow']), scan).dataobj)
    if (not np.isin(v1, np.arange(41)).all() or not np.isin(cow, [*range(13), 15]).all()
            or not np.any(v1[region] > 0) or not np.any(cow[region] > 0)):
        raise ValueError('Published label ontology or coverage differs')
    return scan, v1, cow, region


def scores(root, record, source, scan, blend):
    result = read(checked_file(root, record))
    model = MODEL_CONFIG['vesselfm-base']
    parameters = {**model['parameters'], 'blend': blend, 'threshold': .5, 'retain_probabilities': True}
    checkpoint = safe_path(root, 'cache/models/poc-vesselfm-base/vesselFM_base.pt')
    if (result['status'] != 'succeeded' or result['input']['sha256'] != source['sha256']
            or result['model_id'] != 'vesselfm-base' or result['input']['modality'] != 'CTA'
            or result['model']['parameters'] != parameters or result['model']['version'] != model['version']
            or result['device'] != 'mps' or len(result['weights']) != 1):
        raise ValueError('Inference receipt differs from the frozen CTA protocol')
    checked_file(root, {'path': str(checkpoint.relative_to(root)), 'sha256': result['weights'][0]['sha256']})
    probability = result['probability']
    binary = next(row for row in result['outputs'] if row['name'] == 'vessels_mask.nii.gz')
    arrays = []
    for member in (probability, binary):
        file = Path(member['path']).resolve()
        bound = checked_file(root, {'path':str(file.relative_to(root)), 'sha256':member['sha256']})
        arrays.append(np.asanyarray(native_image(bound, scan).dataobj))
    result, mask = arrays
    if (not np.isfinite(result).all() or result.min() < 0 or result.max() > 1
            or not np.isin(mask, (0,1)).all() or not np.array_equal(result > .5, mask > 0)):
        raise ValueError('The retained probability and threshold-0.5 mask disagree')
    return result


def prediction(probability, values, settings):
    if 'remove_hu_at_or_above' in settings:
        return (probability > settings['threshold']) & (values < settings['remove_hu_at_or_above'])
    if 'seed' not in settings:
        return probability > settings['threshold']
    labels, count = ndi.label(probability > settings['low'], structure=STRUCTURE)
    seeded = np.zeros(count + 1, bool)
    seeded[labels[probability > settings['seed']]] = True
    seeded[0] = False
    return seeded[labels]


def roi_metrics(p, v1, cow, values, affine, tolerance):
    v1 = v1 > 0
    cow = cow > 0
    known = v1 | cow
    result = evaluate(p, known, affine, tolerance_mm=tolerance)
    core = evaluate(p, cow, affine, tolerance_mm=tolerance)
    precision = result['centerline_precision'] or 0.
    recall = result['centerline_recall'] or 0.
    bins = [row for row in result.get('diameter_bins', []) if row['max_mm'] is not None and row['max_mm'] <= 2]
    n = sum(row['reference_points'] for row in bins)
    unmatched = p & ~known
    result.update(
        dice=2 * np.count_nonzero(p & known) / (np.count_nonzero(p) + np.count_nonzero(known)),
        centerline_f1=2 * precision * recall / (precision + recall) if precision + recall else 0.,
        small_vessel_recall=(sum(row['reference_points']*(row['centerline_recall'] or 0.) for row in bins) / n
                             if n else None),
        small_vessel_reference_points=n,
        annotated_voxels=int(known.sum()),
        predicted_voxels=int(p.sum()),
        unmatched_prediction_voxels=int(unmatched.sum()),
        unmatched_high_attenuation_voxels=int(np.count_nonzero(unmatched & (values >= 650))),
        original_cow_centerline_recall=core['centerline_recall'],
        original_cow_complete_branch_recall=core['complete_branch_segment_recall'],
    )
    return result


def positive_agreement(predicted, labels, affine, region, tolerance):
    """Score supported labelled paths only outside the documented CoW box."""
    native = nib.affines.apply_affine
    centers = native(affine, np.argwhere(skeletonize(predicted)))
    index = cKDTree(centers) if len(centers) else None
    results = {}
    for value in np.unique(labels):
        if not value:
            continue
        mask = labels == value
        xyz = np.nonzero(mask)
        crop = tuple(slice(int(axis.min()), int(axis.max())+1) for axis in xyz)
        offset = np.asarray([s.start for s in crop])
        points = np.argwhere(skeletonize(mask[crop])) + offset
        inside = np.logical_and.reduce([(points[:, axis] >= slab.start) & (points[:, axis] < slab.stop)
                                       for axis, slab in enumerate(region)])
        points = points[~inside]
        mask[region] = False
        if not mask.any():
            continue
        distances = index.query(native(affine, points))[0] if index and len(points) else []
        hits = int(np.count_nonzero(np.asarray(distances) <= tolerance))
        results[str(int(value))] = {'label_voxels': int(mask.sum()), 'label_voxels_covered': int(np.count_nonzero(predicted[mask])),
                                    'skeleton_points': len(points), 'skeleton_points_supported': hits}
    return results


def assess(root, case, choices, protocol, coverage, runs, *, include_broader=False):
    scan, v1, cow, region = inputs(root, case, protocol, coverage)
    values = scan.get_fdata(dtype=np.float32)
    subaffine = scan.slicer[region].affine
    probability = {}
    result = {}
    for name in choices:
        settings = protocol['selection_candidates'][name]
        blend = settings['blend']
        if blend not in probability:
            probability[blend] = scores(root, runs[f'{case}-{blend}'], protocol['sources'][case]['image'], scan, blend)
        p = prediction(probability[blend], values, settings)
        row = roi_metrics(p[region], v1[region], cow[region], values[region], subaffine, protocol['tolerance_mm'])
        if include_broader:
            row['topbrain_positive_path_agreement'] = positive_agreement(p, v1, scan.affine, region, protocol['tolerance_mm'])
        result[name] = row
    return result


def comparison(cases, name, policy):
    paired = [(row[BASELINE], row[name]) for row in cases.values()]
    f1 = [b['centerline_f1'] - a['centerline_f1'] for a,b in paired]
    dice = [b['dice'] - a['dice'] for a,b in paired]
    branch = [(b['complete_branch_segment_recall'] or 0.) - (a['complete_branch_segment_recall'] or 0.)
              for a,b in paired]
    small = [b['small_vessel_recall'] - a['small_vessel_recall'] for a,b in paired
             if a['small_vessel_recall'] is not None and b['small_vessel_recall'] is not None]
    hd95 = [b['boundary_hd95_mm'] - a['boundary_hd95_mm'] for a,b in paired
            if a['boundary_hd95_mm'] is not None and b['boundary_hd95_mm'] is not None]
    return {
        'eligible': bool(min(f1) >= -policy['max_subject_f1_loss'] and min(dice) >= -policy['max_subject_dice_loss']
                     and min(small, default=0.) >= -policy['max_subject_small_vessel_recall_loss']
                     and len(hd95) == len(paired) and max(hd95) <= policy['max_subject_hd95_increase_mm']
                     and (np.mean(f1) >= policy['min_mean_f1_gain']
                          or np.mean(branch) >= policy['min_mean_branch_recall_gain'])),
        'mean_f1_gain': float(np.mean(f1)),
        'mean_dice_gain': float(np.mean(dice)),
        'mean_branch_recall_gain': float(np.mean(branch)),
        'mean_small_vessel_recall_gain': float(np.mean(small)) if small else None,
        'f1_by_subject': dict(zip(cases, f1)),
        'ranking_delta': float(np.mean(f1) + .25*np.mean(dice) + .25*np.mean(branch)),
    }


def build(root, phase):
    output = safe_path(root, EVIDENCE)
    protocol_file = output/'protocol.json'
    coverage_file = output/'annotation-coverage.json'
    protocol, coverage, runs = (read(file) for file in (protocol_file, coverage_file, output/'inference.json'))
    split = protocol['split']
    if set(split['selection']) & set(split['verification']):
        raise ValueError('Settings selection and verification require different subjects')
    inputs_used = {'protocol':binding(root,protocol_file),'coverage':binding(root,coverage_file)}
    selection_path = output/'selection.json'
    if phase == 'selection':
        if selection_path.exists():
            raise ValueError('Settings selection is already frozen')
        choices = tuple(protocol['selection_candidates'])
        records = {case:assess(root,case,choices,protocol,coverage,runs) for case in split['selection']}
        rankings = {name:comparison(records,name,protocol) for name in choices if name != BASELINE}
        ranked = sorted(rankings, key=lambda key:rankings[key]['ranking_delta'], reverse=True)
        winner = next((name for name in ranked if rankings[name]['eligible']), None)
        provisional = winner or ranked[0]
        report = {'schema':'annotated-cta-vascular-selection-v1','inputs':inputs_used,
                  'selection_subjects':split['selection'], 'verification_subjects_unscored':split['verification'],
                  'winner':winner,'frozen_comparison':provisional,'rankings':rankings,'cases':records,
                  'source_inference':{key:val for key,val in runs.items() if key.split('-')[0] in split['selection']}}
        path = selection_path
    else:
        selection = read(selection_path)
        if selection['inputs'] != inputs_used or selection['selection_subjects'] != split['selection']:
            raise ValueError('Verification differs from the frozen selection split')
        choice = selection['frozen_comparison']
        records = {case:assess(root,case,(BASELINE,choice),protocol,coverage,runs,include_broader=True)
                   for case in split['verification']}
        gate = comparison(records,choice,protocol)
        report = {'schema':'annotated-cta-vascular-verification-v1','inputs':inputs_used,
                  'frozen_selection':binding(root,selection_path),'choice':choice,
                  'promote_roi_variant':bool(selection['winner'] == choice and gate['eligible']),
                  'comparison':gate,'cases':records,
                  'source_inference':{key:val for key,val in runs.items() if key.split('-')[0] in split['verification']},
                  'scope':'Agreement with supplied labels in publisher ROI. Outside ROI, only positive labelled path support is measured.'}
        path = output/'verification.json'
    if path.exists():
        if read(path) != report:
            raise ValueError(f'Existing benchmark receipt differs: {path}')
    else:
        atomic_json(path,report)
    print(path,flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True,type=Path)
    parser.add_argument('phase',choices=('selection','verification'))
    args = parser.parse_args()
    build(checked_root(args.root,expected_uuid=UUID),args.phase)
