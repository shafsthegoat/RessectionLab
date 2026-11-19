#!/usr/bin/env python3
"""Audit protected superior display conditioning on frozen ISLES and TAIR studies."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates

from data_integrated_hosts import UUID, binding
from probe_cads_vault_conditioning import (
    MAX_ADDITIONAL_MM,
    MAX_FROM_SOURCE_CELLS,
    MAX_FROM_SOURCE_MM,
    assess,
    contrast_pair,
    normal_turn_pair,
    retain_result,
)
from probe_cta_showcase_conditioning import CROP_MARGIN, CROP_TAPER, smoothstep
from resectionlab.anatomy_geometry import mesh
from resectionlab.core import array_digest
from resectionlab.data_root import checked_root, safe_path, sha256_file
from resectionlab.patient_anatomy import PatientImagingCollection, reconstruction_lock

CASE = 'ISLES24-v7:sub-stroke0146'
SERIES = 'ncct'
SKULL = 'totalsegmentator-skull-a64fb148aae8'
BRAIN = 'totalsegmentator-skull-13d01e3cd6a4-brain'
SNAPSHOT = '4e08cc0d00a066774edf3300504ca5e875e10db9fed0db270d5dbf4aeb2b9b2b'
DESTINATION = Path('prepared/isles24-v7-native/sub-stroke0146')
OUTPUT = 'isles0146-skull-conditioning-v1'
SUPERIOR_DEPTH_MM = 70
SUPERIOR_TAPER_MM = 15
MAX_ADDED_MASK_RESIDUAL = .03


@dataclass(frozen=True)
class Study:
    patient: str
    directory: Path
    snapshot: str
    series: str
    modality: str
    layer: str
    brain: str
    output: str


STUDIES = {
    'isles0146-skull': Study(CASE, DESTINATION, SNAPSHOT, SERIES, 'CT', SKULL, BRAIN, OUTPUT),
    'isles0146-brain': Study(CASE, DESTINATION, SNAPSHOT, SERIES, 'CT', BRAIN, BRAIN,
                            'isles0146-brain-conditioning-v1'),
    'tair00273-skull': Study(
        'TAIR:00273', Path('prepared/tair-v1-cta-native/00273'),
        '0fa9f579dea14dc6118d527ec1bdfe8b7f1ab2b73bd8ee7b6d8d901b59f0e0ab',
        'tair-cta-display', 'CTA', 'totalsegmentator-skull-d7eea0c1b2e5',
        'totalsegmentator-skull-e80ddcce5db3-brain', 'tair00273-skull-conditioning-v1',
    ),
}


def open_stock(root, study):
    directory = safe_path(root, study.directory)
    frozen = directory / 'revisions' / f'{study.snapshot}.json'
    collection = PatientImagingCollection(directory, json.loads(frozen.read_text()))
    scan, layer, brain = collection.series(study.series), collection.layer(study.layer), collection.layer(study.brain)
    matrix = np.asarray(scan['affine'])
    linear = matrix[:3, :3]
    if (collection.document['patient_id'] != study.patient or collection.identity() != study.snapshot
            or scan['modality'] != study.modality or collection.document['reference_series'] != study.series
            or not np.array_equal(linear, np.diag(np.diag(linear))) or linear[2, 2] <= 0
            or brain['kind'] != 'brain' or layer['kind'] not in ('brain', 'skull')
            or any(row['source_series'] != study.series or row['role'] != 'context'
                   or row['affine'] != scan['affine'] or row['data']['shape'] != scan['data']['shape']
                   or row['label_value'] != 1 or row['revision'] != 0
                   or row.get('edits') or row.get('conditioned_surface') or row.get('probability')
                   for row in (layer, brain))):
        raise ValueError('Expected the exact frozen unedited model layers on their CT-family source grid')

    original = scan['original']
    publisher = scan.get('publisher_source', original)
    for ref in (original,) if original == publisher else (original, publisher):
        if sha256_file(collection.path(ref['path'])) != ref['sha256']:
            raise ValueError(f'Source volume checksum differs for {study.patient}')
    return collection, layer, brain, scan, binding(root, frozen), publisher


def categorical_residual(mask, positions, samples):
    """Sample trilinear categorical boundary residuals at matched mesh indices."""
    values = map_coordinates(mask, positions[samples].T, output=np.float64, order=1)
    residual = np.abs(values - .5)
    return {'samples': len(samples), 'median': float(np.median(residual)),
            'p95': float(np.percentile(residual, 95))}


def crop_counts(mask, label):
    counts = {}
    for axis in range(mask.ndim):
        for edge, index in (('min', 0), ('max', -1)):
            slab = tuple(index if coordinate == axis else slice(None) for coordinate in range(mask.ndim))
            counts[f'{axis}_{edge}'] = int(np.count_nonzero(mask[slab] == label))
    return counts


def guard_categorical_boundary(mask, baseline, candidate):
    changed = np.flatnonzero(np.any(baseline != candidate, axis=1))
    origin = baseline[changed]
    delta = candidate[changed] - origin

    def residual(positions):
        return np.abs(map_coordinates(mask, positions.T, output=np.float64, order=1) - .5)

    permitted = residual(origin) + MAX_ADDED_MASK_RESIDUAL / 2
    for fraction in (1., .5, .25):
        proposed = origin + fraction * delta
        accepted = residual(proposed) <= permitted
        candidate[changed[accepted]] = proposed[accepted]
        waiting = ~accepted
        changed, origin, delta, permitted = changed[waiting], origin[waiting], delta[waiting], permitted[waiting]
    candidate[changed] = origin
    return candidate


def probe(root: Path, output: Path, case: str = 'isles0146-skull', *, categorical_guard=False) -> dict:
    if output.exists():
        raise FileExistsError(output)
    study = STUDIES[case]
    collection, layer, brain, scan, manifest, publisher = open_stock(root, study)
    mask = collection.array(layer['data'])
    ct = collection.array(scan['data'])
    brain_levels = np.flatnonzero(collection.mask(brain).any(axis=(0, 1)))
    if not len(brain_levels):
        raise ValueError('The frozen brain support is empty')
    roof = int(brain_levels[-1])
    matrix = np.asarray(scan['affine'])
    linear = matrix[:3, :3]
    lower = int(np.ceil(roof - SUPERIOR_DEPTH_MM / linear[2, 2]))
    upper = lower + int(np.ceil(SUPERIOR_TAPER_MM / linear[2, 2]))
    geometry = mesh(collection, layer)
    raw, stock, faces = geometry['source'], geometry['positions'], geometry['indices']
    crop_distance = np.minimum(raw, np.asarray(mask.shape) - 1 - raw).min(axis=1)
    crop = crop_distance <= CROP_MARGIN
    base = raw[:, 2] <= lower
    protected = base | crop
    measured = (raw[:, 2] >= upper) & (crop_distance >= CROP_MARGIN + CROP_TAPER)
    if not protected.any() or np.count_nonzero(measured) < 3000:
        raise ValueError('Insufficient retained surface in the protected or measured regions')

    shift = stock - raw
    shift_mm = np.linalg.norm(shift @ linear.T, axis=1)
    shift_cells = np.linalg.norm(shift, axis=1)
    over = (shift_mm > MAX_FROM_SOURCE_MM) | (shift_cells > MAX_FROM_SOURCE_CELLS)
    pinned_over = int(np.count_nonzero(over & protected))
    raw_phys, stock_phys = raw @ linear.T, stock @ linear.T
    raw_to_stock_edges, turns = normal_turn_pair(raw_phys, stock_phys, faces, measured)
    del raw_phys, stock_phys
    contrast = contrast_pair(ct, raw, stock, faces, measured, linear)
    samples = np.flatnonzero(measured)[::13]
    categorical = {name: categorical_residual(mask, points, samples)
                   for name, points in (('raw', raw), ('stock', stock))}
    report = dict(
        schema=study.output, case_id=study.patient, source_person=study.patient, source_manifest=manifest,
        source_ct=publisher, source_display_original=scan['original'], source_ct_array=scan['data'],
        source_display_sampling_ijk=scan.get('display_sampling_ijk', [1, 1, 1]),
        mask=layer['data'], layer=layer['id'], snapshot=study.snapshot, affine=matrix.tolist(),
        triangles=len(faces), vertices=len(raw), indices_hash=array_digest(faces),
        source_mesh_hash=array_digest(raw), stock_conditioned_hash=array_digest(stock),
        estimated_brain_superior_index=roof, protected_native_plane_inclusive=lower,
        full_effect_native_plane=upper, protected_crop_margin_cells=CROP_MARGIN, crop_taper_cells=CROP_TAPER,
        protected_vertices_exact=int(np.count_nonzero(protected)), measured_vertices=int(np.count_nonzero(measured)),
        native_crop_face_foreground_voxels=crop_counts(mask, layer['label_value']),
        stock_protected_vertices_over_attachment_limits=pinned_over,
        stock_base_vertices_over_attachment_limits=int(np.count_nonzero(over & base)),
        stock_crop_vertices_over_attachment_limits=int(np.count_nonzero(over & crop)),
        stock_vertices_over_attachment_limits=int(np.count_nonzero(over)),
        stock_source_displacement_mm=float(shift_mm.max()), stock_source_displacement_cells=float(shift_cells.max()),
        attachment_limits={'source_mm': MAX_FROM_SOURCE_MM, 'source_cells': MAX_FROM_SOURCE_CELLS,
                           'additional_mm': MAX_ADDITIONAL_MM},
        raw_to_stock={
            'shared_edges': raw_to_stock_edges, 'turn': dict(zip(('raw', 'stock'), turns)),
            'ct_contrast_probe': {name: dict(samples=count, median=value)
                                  for name, (count, value) in zip(('raw', 'stock'), contrast)},
        },
        categorical_contour_residual=categorical, max_added_categorical_residual=MAX_ADDED_MASK_RESIDUAL,
        trial_attempted=False, passed_numeric_gate=False,
        scope='Display positions only. Source values, categorical labels, crop and indexed topology are retained.',
    )
    if pinned_over:
        report['refusal_reasons'] = [
            f'{pinned_over:,} protected stock vertices already exceed the attachment displacement limits. '
            'Retaining them exactly and admitting a new display surface are incompatible.'
        ]
        return retain_result(root, output, report, None)

    report['trial_attempted'] = True
    report['categorical_guard_during_fit'] = categorical_guard
    weight = smoothstep((raw[:, 2] - lower) / (upper - lower)) * smoothstep((crop_distance - CROP_MARGIN) / CROP_TAPER)
    try:
        guard = (lambda a, b: guard_categorical_boundary(mask, a, b)) if categorical_guard else None
        trial, candidate = assess(geometry, ct, linear, protected, measured, weight, position_guard=guard)
    except ValueError as error:
        report['refusal_reasons'] = [f'The one bounded geometry trial failed verification. {error}']
        return retain_result(root, output, report, None)

    report.update(trial)
    categorical['candidate'] = categorical_residual(mask, candidate, samples)
    refusal = []
    if not trial['passed_numeric_gate']:
        refusal.append('The trial failed the shared roughness or paired CT-contrast gates.')
    if any(categorical['candidate'][metric] > categorical['stock'][metric] + MAX_ADDED_MASK_RESIDUAL
           for metric in ('median', 'p95')):
        refusal.append('The trial moved the sampled display surface farther from the retained categorical contour.')
    if refusal:
        report.update(passed_numeric_gate=False, refusal_reasons=refusal)
    return retain_result(root, output, report, candidate)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--case', choices=tuple(STUDIES), default='isles0146-skull')
    parser.add_argument('--output-name')
    parser.add_argument('--guard-categorical', action='store_true')
    args = parser.parse_args()
    name = args.output_name if args.output_name is not None else STUDIES[args.case].output
    if Path(name).name != name or name in ('', '.', '..'):
        parser.error('--output-name must be a fresh directory name under qc')
    root = checked_root(args.root, expected_uuid=UUID, required_bytes=1024**3)
    output = safe_path(root, f'qc/{name}')
    with reconstruction_lock():
        result = probe(root, output, args.case, categorical_guard=args.guard_categorical)
    print(f"Numeric gate {result['passed_numeric_gate']}. Report at {output / 'report.json'}", flush=True)


if __name__ == '__main__':
    main()
