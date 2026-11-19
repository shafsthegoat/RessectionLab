#!/usr/bin/env python3
"""Prepare and screen one CTA014 organic contour on a fixed proposed line."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

from build_cta_organic_gliomas import (
    RELEASE, UUID, checked_file, load_host, observed_contour, placement_domain, read_json, register, save_variant,
)
from probe_cta_organic_rehearsal import cells, cover, tool_segments
from resectionlab.core import array_digest
from resectionlab.data_root import atomic_json, checked_root, safe_path
from resectionlab.geometry import GeometryScene
from resectionlab.organic_tumors import organic_geometry, signed_domain
from resectionlab.patient_anatomy import reconstruction_lock
from resectionlab.surgical_motion import swept_envelopes
from resectionlab.surgical_tools import TOOLS

CASE = 'TopCoW-014-CTA-ORGANIC-FIXED-ROUTE-01-v1'
OUTPUT = Path('prepared/cta014-organic-fixed-route-v1')
OLD = 'integrated-v1/access/CTA014-stock-target-access-v1'
PREFIX = 'integrated-v1/regressions/executions/CTA014-model-brain-oldrequest-regression-v1-continuation-v1'
HISTORY = {
    'plan': {'path': OLD + '/plan.json', 'sha256': '6557f9fedc316f89f989adaa30429b4dae4fb315d3dce1a08ce4f45fd7746167'},
    'preflight': {'path': OLD + '/preflight-pass.json', 'sha256': 'b936748ff1af27c19a09d7af45e3f378e62bc1643bc5e55c942e4e0e4014a700'},
    'kerf': {'path': OLD + '/public/actions.json', 'sha256': '95ddfb00f4acfc131f66c0e7411f3496b4893ba689337cad69ee036f22a0bdc0'},
    'opening': {'path': PREFIX + '/actions.json', 'sha256': 'daeb8b42352788fb11411fa58c12f110aa6218516351953e510b38b429cd6aa7'},
}
OFFSET_MM, SEED, SIGMA_MM = 5., 3303391082, .8


def fixed_candidate(host, support, affine, route):
    entry, goal = (np.asarray(route[key]) for key in ('A', 'T'))
    axis = (goal - entry) / np.linalg.norm(goal - entry)
    point = goal - OFFSET_MM * axis
    center = np.rint(np.linalg.solve(affine[:3, :3], point - affine[:3, 3])).astype(int)
    low, high = center - 28, center + 29
    if np.any(low < 0) or np.any(high > support.shape):
        raise ValueError('Fixed tissue crop does not fit the source support')
    region = tuple(slice(a, b) for a, b in zip(low, high))
    crop_affine = affine.copy()
    crop_affine[:3, 3] = affine[:3, :3] @ low + affine[:3, 3]
    domain = support[region]
    signed = signed_domain(domain, crop_affine)
    masks, recipe = organic_geometry(domain, crop_affine, center_voxel=center - low, seed=SEED,
                                     radii_mm=(3.2, 3.8, 4.2), signed=signed, profile='lobulated')
    recipe.update(placement=['right', 'fixed-route', 'development'],
                  conditioning='5 mm upstream of the historical proposed endpoint',
                  signed_center_clearance_mm=float(signed[tuple(center - low)]), nominal_center_ras_mm=point.tolist())
    exact = masks['synthetic_core'] | masks['synthetic_rim']
    contour, dice = observed_contour(exact, domain, host.spacing_mm, sigma_mm=SIGMA_MM)
    standard = (gaussian_filter(exact.astype(np.float32), 1.3 / np.asarray(host.spacing_mm)) >= .54) & domain
    standard_dice = 2 * np.count_nonzero(standard & exact) / (standard.sum() + exact.sum())
    return masks, recipe, region, contour, dice, float(standard_dice)


def opening_samples(history):
    applied = [row for row in history if row['committed']]
    if ([row['literal_index'] for row in applied] != [*range(23), 24]
            or [row['literal_index'] for row in history if not row['committed']] != [23]):
        raise ValueError('Historical opening accepted/failed partition differs')
    for row in applied:
        for sample in row['actual_contact_reconstruction']:
            yield [(p['a'], p['b'], p['radiusMm'], p['part']) for p in sample['envelopes']]


def kerf_samples(history):
    tool = TOOLS['craniotome']
    parameters = {'radiusMm': tool['radiusMm'], 'depthMm': 2., 'contactGeometryRevision': 2}
    previous = None
    for row in history:
        if not row['committed']:
            raise ValueError('Expected the committed historical kerf prefix')
        action = row['action']
        kind = action['operation']
        if kind == 'exchange' and action['instrument'] == 'cusa':
            break
        current = row['slots']['right']
        if kind in ('exchange', 'place'):
            yield tool_segments('craniotome', current['position'], current['quaternion'])
        else:
            poses = [previous, *action['path']]
            for a, b in zip(poses, poses[1:]):
                yield swept_envelopes(tool, parameters, a, b)
        previous = current


def contact(grid, geometries, target):
    overlap = set()
    count = 0
    for geometry in geometries:
        indices = cells(grid, geometry)
        overlap.update(map(int, indices[target[indices]]))
        count += 1
    if count < 100:
        raise ValueError('Expected the full historical geometry sample coverage')
    return {'swept_queries': count, 'observed_contour_contact_ids': sorted(overlap)}


def build(root, directory, library):
    final = directory / CASE
    proof = directory / 'preflight.json'
    if proof.exists():
        report = read_json(proof)
        if report['case'] is not None:
            for member in ('native_case', 'truth', 'qa'):
                checked_file(root, report['case'][member])
            register(root, [report['case']], library)
        return report
    history = {name: read_json(checked_file(root, row)) for name, row in HISTORY.items()}
    prior = history['preflight']
    route, = (r['route'] for r in history['plan']['proposals'] if r['route']['sample'] == prior['route_sample'])
    if (prior['route_sample'] != 13451 or not prior['passed'] or route['A'] != prior['cusa_entry']
            or route['T'] != prior['target_endpoint'] or prior['plan_sha256'] != HISTORY['plan']['sha256']):
        raise ValueError('Expected the exact historical route used solely as a geometry proposal')
    host, originals, source_masks, evidence = load_host(root, '014', read_json(safe_path(root, RELEASE)))
    support, affine, z, midline = placement_domain(root, host, source_masks)
    masks, recipe, region, contour, dice, standard = fixed_candidate(host, support, affine, route)
    target = np.zeros(host.mri.shape, bool)
    target[(*region[:2], slice(region[2].start + z, region[2].stop + z))] = contour
    grid = GeometryScene(np.zeros(host.mri.shape, bool), host.affine)
    flat = target.ravel()
    opening = contact(grid, opening_samples(history['opening']), flat)
    kerfs = contact(grid, kerf_samples(history['kerf']), flat)
    working = cover('cusa', route['A'], prior['tip_position'], route['orientation'])
    contacts = cells(grid, working, 'working')
    count = int(flat[contacts].sum())
    eligible = count > 0 and not any(x['observed_contour_contact_ids'] for x in (opening, kerfs))
    directory.mkdir(parents=True, exist_ok=True)
    report = {'schema': 'cta014-organic-fixed-route-static-v1', 'case_id': CASE, 'history': HISTORY,
              'source_case_hash': host.semantic_hash, 'source': evidence['host'], 'route_sample': prior['route_sample'],
              'recipe': recipe, 'contour_sha256': array_digest(target), 'contour_cells': int(contour.sum()),
              'truth_overlap_dice': dice, 'original_1_3_mm_blur_dice': standard,
              'center_lateral_to_estimated_midline_mm': abs(recipe['center_ras_mm'][0] - midline),
              'historical_opening_coverage': opening, 'historical_requested_kerf_coverage': kerfs,
              'cusa_working_cover_to_historical_attained_tip_cells': count,
              'vote': 'candidate_for_fresh_execution_review' if eligible else 'static_refused',
              'actions_executed': 0, 'existing_authority_or_state_inherited': False, 'case': None}
    if eligible:
        if final.exists():
            report['case'] = read_json(final / 'receipt.json')
            for member in ('native_case', 'truth', 'qa'):
                checked_file(root, report['case'][member])
        else:
            report['case'] = save_variant(root, final, CASE, host, originals, evidence, (masks, recipe, region),
                                          support, z, observation_sigma_mm=SIGMA_MM)
        if (report['case']['case_id'] != CASE or report['case']['source']['host'] != evidence['host']
                or report['case']['observation']['array_hash'] != report['contour_sha256']
                or report['case']['observation']['filter_sigma_mm'] != SIGMA_MM):
            raise ValueError('Packaged contour differs from the screened contour')
        register(root, [report['case']], library)
    atomic_json(proof, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--register', action='append', default=[], type=Path)
    args = parser.parse_args()
    root = checked_root(args.root, expected_uuid=UUID, required_bytes=1024**3)
    with reconstruction_lock():
        result = build(root, safe_path(root, OUTPUT), args.register)
    print(json.dumps({key: result[key] for key in ('case_id', 'vote', 'contour_cells',
                                                   'cusa_working_cover_to_historical_attained_tip_cells')}, indent=2))


if __name__ == '__main__':
    main()
