#!/usr/bin/env python3
"""Measure an alternative conditioned surface on the superior CADS 465 vault."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from scipy.ndimage import map_coordinates
from vtkmodules.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray, vtk_to_numpy
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData
from vtkmodules.vtkFiltersCore import vtkWindowedSincPolyDataFilter

from data_integrated_hosts import UUID, binding, checked_file
from prepare_cads_brainct import DESTINATION, SERIES
from resectionlab.anatomy_geometry import (
    CONDITIONED_MAX_ADDITIONAL_MM as MAX_ADDITIONAL_MM,
    CONDITIONED_MAX_SOURCE_CELLS as MAX_FROM_SOURCE_CELLS,
    CONDITIONED_MAX_SOURCE_MM as MAX_FROM_SOURCE_MM,
    mesh,
    normals,
)
from resectionlab.core import array_digest
from resectionlab.data_root import atomic_json, checked_root, safe_path
from resectionlab.patient_anatomy import PatientImagingCollection, reconstruction_lock
from resectionlab.surface_fairing import intersection_vertices

OUTPUT = 'cads465-vault-conditioning-v1'
LOWER_INDEX = 116
UPPER_INDEX = 136


def filter_positions(positions, faces):
    """Run a stronger low-pass in physical coordinates with the existing topology."""
    points = vtkPoints()
    points.SetData(numpy_to_vtk(np.ascontiguousarray(positions, dtype=np.float64), deep=True))
    polys = vtkCellArray()
    offsets = np.arange(0, 3 * len(faces) + 1, 3, dtype=np.int64)
    polys.SetData(numpy_to_vtkIdTypeArray(offsets, deep=True),
                  numpy_to_vtkIdTypeArray(np.ascontiguousarray(faces.ravel(), dtype=np.int64), deep=True))
    surface = vtkPolyData()
    surface.SetPoints(points)
    surface.SetPolys(polys)
    smoother = vtkWindowedSincPolyDataFilter()
    smoother.SetInputData(surface)
    smoother.SetNumberOfIterations(32)
    smoother.SetPassBand(.025)
    smoother.NormalizeCoordinatesOn()
    smoother.BoundarySmoothingOff()
    smoother.FeatureEdgeSmoothingOff()
    smoother.NonManifoldSmoothingOff()
    smoother.Update()
    result = smoother.GetOutput()
    if (result.GetNumberOfPoints() != len(positions)
            or not np.array_equal(vtk_to_numpy(result.GetPolys().GetConnectivityArray()), faces.ravel())):
        raise ValueError('Windowed sinc filter changed indexed topology')
    return vtk_to_numpy(result.GetPoints().GetData()).astype(np.float64)


def cap_candidate(source, baseline, target, linear, weight=None):
    """Taper to exact baseline under the protected skull-base boundary."""
    if weight is None:
        t = np.clip((source[:, 2] - LOWER_INDEX) / (UPPER_INDEX - LOWER_INDEX), 0, 1)
        weight = t * t * (3 - 2 * t)
    inverse = np.linalg.inv(linear)
    delta = (target - baseline @ linear.T) * weight[:, None]
    delta *= np.minimum(1., (MAX_ADDITIONAL_MM - .0001) /
                        np.maximum(np.linalg.norm(delta, axis=1), 1.e-20))[:, None]
    result = np.asarray(baseline + delta @ inverse.T, dtype=np.float32)
    # Project excess motion back towards the categorical contour. The original
    # conditioning is always within the tighter display budget in pinned areas.
    moved = result - source
    fractions = np.minimum((MAX_FROM_SOURCE_MM - .0001) /
                           np.maximum(np.linalg.norm(moved @ linear.T, axis=1), 1.e-20),
                           (MAX_FROM_SOURCE_CELLS - .0001) /
                           np.maximum(np.linalg.norm(moved, axis=1), 1.e-20))
    result = np.asarray(source + moved * np.minimum(1., fractions)[:, None], dtype=np.float32)
    result[weight == 0] = baseline[weight == 0]
    return result


def signed_normals(points, faces):
    triangles = points[faces]
    return np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])


def remove_bad_faces(baseline, candidate, faces):
    """Roll back intersecting or failed vertices to the previously audited surface."""
    original = signed_normals(baseline, faces)
    min_area_squared = .01 * np.einsum('ij,ij->i', original, original)
    backed_off = set()
    for _ in range(5):
        relocated = signed_normals(candidate, faces)
        invalid = (np.einsum('ij,ij->i', original, relocated) <= 0) | (
            np.einsum('ij,ij->i', relocated, relocated) < min_area_squared)
        contacts = intersection_vertices(baseline, candidate, faces)
        rollback = np.union1d(faces[invalid].ravel(), contacts)
        if not len(rollback):
            return len(backed_off)
        candidate[rollback] = baseline[rollback]
        backed_off.update(rollback.tolist())
    raise ValueError('The candidate retains new triangle intersections or orientation failures')


def normal_turn_pair(baseline, candidate, faces, vault):
    """Compare mean squared normal turn on the same shared vault edges."""
    active = faces[vault[faces].all(axis=1)]
    n = len(baseline)
    pieces = [np.sort(active[:, [a, b]], axis=1).astype(np.int64) for a, b in [(0, 1), (1, 2), (0, 2)]]
    edges = np.concatenate([pair[:, 0] * n + pair[:, 1] for pair in pieces])
    order = np.argsort(edges, kind='stable')
    keys = edges[order]
    lower = np.flatnonzero((keys[:-1] == keys[1:]) &
                           (keys[:-1] != np.r_[-1, keys[:-2]]) & (keys[1:] != np.r_[keys[2:], -1]))
    first, second = order[lower] % len(active), order[lower + 1] % len(active)
    results = []
    for points in (baseline, candidate):
        directions = signed_normals(points, active)
        directions /= np.maximum(np.linalg.norm(directions, axis=1)[:, None], 1.e-20)
        cosine = np.clip(np.einsum('ij,ij->i', directions[first], directions[second]), -1, 1)
        turn = np.degrees(np.arccos(cosine))
        results.append({'mean_squared_deg': float(np.mean(turn**2)), 'p95_deg': float(np.percentile(turn, 95))})
    return len(lower), results


def contrast_pair(ct, baseline, candidate, faces, vault, linear):
    physical_normals = normals(baseline @ linear.T, faces)
    stride = 13
    which = np.flatnonzero(vault)[::stride]
    offset = physical_normals[which] @ np.linalg.inv(linear).T * .5
    def contrast(points):
        p = points[which]
        within = ((p - np.abs(offset) > 0) & (p + np.abs(offset) < np.asarray(ct.shape) - 1)).all(axis=1)
        pos = map_coordinates(ct, (p[within] + offset[within]).T, order=1)
        neg = map_coordinates(ct, (p[within] - offset[within]).T, order=1)
        return int(within.sum()), float(np.median(np.abs(pos - neg)))
    return [contrast(points) for points in (baseline, candidate)]


def assess(old, ct, linear, protected, vault, weight=None):
    """Measure display-only relocation on fixed, source-bound indices."""
    raw, baseline, faces = old['source'], old['positions'], old['indices']
    raw_phys, baseline_phys = raw @ linear.T, baseline @ linear.T
    target = filter_positions(baseline_phys, faces)
    candidate = cap_candidate(raw, baseline, target, linear, weight)
    del target
    print(f'Prepared {len(faces):,} existing triangles; checking independent geometry limits', flush=True)
    candidate_phys = candidate @ linear.T
    rolled_back = remove_bad_faces(baseline_phys, candidate_phys, faces)
    rolled_back_nodes = np.all(candidate_phys == baseline_phys, axis=1)
    candidate = (candidate_phys @ np.linalg.inv(linear).T).astype(np.float32)
    candidate[rolled_back_nodes] = baseline[rolled_back_nodes]
    candidate_phys = candidate @ linear.T
    if len(intersection_vertices(baseline_phys, candidate_phys, faces)):
        raise ValueError('Quantized candidate introduced triangle contacts')
    origin_change = np.linalg.norm(candidate_phys - raw_phys, axis=1)
    extra_change = np.linalg.norm(candidate_phys - baseline_phys, axis=1)
    cell_change = np.linalg.norm(candidate - raw, axis=1)
    if (origin_change.max() > MAX_FROM_SOURCE_MM or extra_change.max() > MAX_ADDITIONAL_MM
            or cell_change.max() > MAX_FROM_SOURCE_CELLS
            or not np.array_equal(candidate[protected], baseline[protected])):
        raise ValueError('Candidate changed protected anatomy or exceeded its grid-relative displacement bounds')

    edge_count, turn = normal_turn_pair(baseline_phys, candidate_phys, faces, vault)
    contrast = contrast_pair(ct, baseline, candidate, faces, vault, linear)
    reduction = float(1 - turn[1]['mean_squared_deg'] / turn[0]['mean_squared_deg'])
    contrast_ratio = contrast[1][1] / max(contrast[0][1], 1.e-10)
    passed = reduction >= .15 and contrast_ratio >= .95 and contrast[0][0] == contrast[1][0]
    report = {'triangles': len(faces),
              'indices_hash': array_digest(faces), 'source_mesh_hash': array_digest(raw),
              'stock_conditioned_hash': array_digest(baseline), 'candidate_hash': array_digest(candidate),
              'protected_vertices_exact': int(np.count_nonzero(protected)),
              'additional_displacement_mm': float(extra_change.max()),
              'source_displacement_mm': float(origin_change.max()),
              'source_displacement_cells': float(cell_change.max()),
              'vertices_returned_to_baseline': rolled_back, 'new_contacts': 0,
              'vault_shared_edges': edge_count, 'turn': dict(zip(('stock', 'candidate'), turn)),
              'mean_squared_turn_reduction': reduction,
              'ct_contrast_probe': dict(zip(('stock', 'candidate'), [dict(samples=n, median=v) for n, v in contrast])),
              'contrast_ratio': contrast_ratio, 'passed_numeric_gate': passed,
              'scope': 'Display surface only. Image values, label memberships and triangle indices are unchanged.'}
    return report, candidate


def retain_result(root, output, report, candidate):
    with TemporaryDirectory(prefix='.cads-vault-', dir=output.parent) as temporary:
        stage = Path(temporary) / output.name
        stage.mkdir()
        if report['passed_numeric_gate']:
            positions = stage / 'conditioned-vault.npy'
            np.save(positions, candidate, allow_pickle=False)
            report['positions'] = {**binding(root, positions), 'path': str((output / positions.name).relative_to(root))}
        atomic_json(stage / 'report.json', report)
        stage.rename(output)
    return report


def probe(root, directory, source_manifest, output):
    collection = PatientImagingCollection(directory, json.loads(checked_file(root, source_manifest).read_text()))
    scan = collection.series(SERIES)
    layer = collection.layer('head-bone')
    source_ref = scan['publisher_source']
    if (collection.identity() != collection.document['snapshot_id'] or layer['revision'] != 0
            or not np.array_equal(layer['affine'], scan['affine'])
            or binding(root, collection.path(source_ref['path']))['sha256'] != source_ref['sha256']):
        raise ValueError('The native study or source CT binding differs from the accepted baseline')
    collection.array(layer['data'])
    ct = collection.array(scan['data'])
    matrix = np.asarray(layer['affine'])
    linear = matrix[:3, :3]
    if tuple(np.diag(linear)) != (-.42578125, .42578125, 1.) or not np.array_equal(linear, np.diag(np.diag(linear))):
        raise ValueError('The frozen protected levels require the original orthogonal CADS 465 grid')
    old = mesh(collection, layer)
    levels = old['source'][:, 2]
    report, candidate = assess(old, ct, linear, levels <= LOWER_INDEX, levels >= UPPER_INDEX)
    report.update(schema=OUTPUT, source_manifest=source_manifest, source_ct=source_ref,
                  source_ct_array=scan['data'], mask=layer['data'], snapshot=collection.identity(),
                  layer=layer['id'], affine=matrix.tolist(), protected_native_plane_inclusive=LOWER_INDEX,
                  full_effect_native_plane=UPPER_INDEX)
    return retain_result(root, output, report, candidate)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--output-name', default=OUTPUT)
    args = parser.parse_args()
    if Path(args.output_name).name != args.output_name or args.output_name in ('', '.', '..'):
        parser.error('--output-name must be a fresh directory name under qc')
    root = checked_root(args.root, expected_uuid=UUID, required_bytes=1024**3)
    study = safe_path(root, DESTINATION)
    receipt = json.loads((study / 'receipt.json').read_text())
    if checked_file(root, receipt['manifest']) != study / 'patient-anatomy.json':
        raise ValueError('Expected the released CADS 465 study')
    stock = receipt.get('stock_manifest', receipt['manifest'])
    output = safe_path(root, f'qc/{args.output_name}')
    with reconstruction_lock():
        if output.exists():
            raise FileExistsError(output)
        result = probe(root, study, stock, output)
    print(f"Numeric gate {result['passed_numeric_gate']}. Report at {output / 'report.json'}", flush=True)


if __name__ == '__main__':
    main()
