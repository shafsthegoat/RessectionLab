#!/usr/bin/env python3
"""Public production-bridge anatomy controls; never a patient or clinical claim."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import nibabel as nib
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / 'src')]
from resectionlab.core import CaseData, SourceRef, array_digest, semantic_digest
from resectionlab.data_root import checked_root, atomic_json, sha256_file
from resectionlab.desktop_bridge import BridgeSession, _Request
from resectionlab.surgical_motion import envelopes, swept_envelopes
from resectionlab.surgical_tools import TOOLS
from resectionlab.workspace_bundle import load_workspace, save_workspace, view_state


def envelope_admission(segments, affine, shape, observed=None):
    """Conservative capsule containment in measured grid and reviewed mask.

    Endpoint boxes bound the capsule, including radius. This can refuse a valid
    route; it cannot certify unsupported surrounding anatomy as empty air.
    """
    inverse = np.linalg.inv(affine)
    margins = np.linalg.norm(inverse[:3, :3], axis=1)
    reasons = []
    for a, b, radius, part in segments:
        points = np.asarray([a, b]) @ inverse[:3, :3].T + inverse[:3, 3]
        low = np.floor(points.min(0) - radius * margins).astype(int)
        high = np.ceil(points.max(0) + radius * margins).astype(int)
        if np.any(low < 0) or np.any(high >= shape):
            reasons.append(part + ':outside-source-grid')
        elif observed is not None and not np.all(observed[tuple(slice(l, h + 1) for l, h in zip(low, high))]):
            reasons.append(part + ':outside-reviewed-observation')
    return {'admitted': not reasons, 'reasons': sorted(set(reasons)),
            'policy': 'conservative full swept capsule bounding boxes; independent task restriction, not automatic engine FOV safety'}


def control_case():
    shape = (96, 96, 144)
    brain = np.zeros(shape, bool); brain[30:65, 30:65, 63:70] = True
    bone = np.zeros(shape, bool); bone[25:70, 25:70, 75:77] = True
    tumor = np.zeros(shape, bool); tumor[47:50, 47:50, 65:68] = True
    vessels = np.zeros(shape, bool); vessels[74:77, 30:65, 69:71] = True
    masks = dict(brain=brain, tumor=tumor, bone=bone, vessels=vessels)
    return CaseData(case_id='integrated-generated-control-v1', mri=np.zeros(shape, np.float32),
        compartments=masks, affine=np.eye(4), source_refs=(SourceRef('generated-control',
            'generated://integrated-v1/labelled-control', semantic_digest({'recipe': 'integrated-generated-control-v1'})),),
        metadata={'anatomy_layers': {n: {'kind': {'bone': 'skull'}.get(n, n),
            'provenance': 'generated control; no patient'} for n in masks},
            'surgical_adapter': {'ownershipPrecedence': list(masks)},
            'description': 'Generated labelled full-instrument control; zero source people; no anatomical plausibility claim'})


def encode_segments(segments):
    return [{'aRasMm': np.asarray(a).tolist(), 'bRasMm': np.asarray(b).tolist(),
             'radiusMm': float(r), 'part': part} for a, b, r, part in segments]


def bounded_motion_envelopes(tool, parameters, start, end):
    """Continuous conservative cover without one long-stroke inflated box.

    Every interval retains the production capsule inflation proof; partitioning
    only tightens an independent observation restriction, never collision guards.
    """
    from scipy.spatial.transform import Rotation, Slerp
    a, b = np.asarray(start['position']), np.asarray(end['position'])
    rotations = Rotation.from_quat([start['quaternion'], end['quaternion']])
    angle = (rotations[1] * rotations[0].inv()).magnitude()
    count = max(1, int(np.ceil(np.linalg.norm(b-a)/.25)), int(np.ceil(angle/np.deg2rad(1.))))
    interpolate = Slerp([0., 1.], rotations); result = []; previous = start
    for fraction in np.linspace(0., 1., count+1)[1:]:
        pose = {'position': ((1-fraction)*a+fraction*b).tolist(), 'quaternion': interpolate(fraction).as_quat().tolist()}
        result.extend(swept_envelopes(tool, parameters, previous, pose)); previous = pose
    return result


def admissible_bone_preview(before, outcome, predicted):
    """A task-only gate; native geometry/material guards remain authoritative."""
    bone = predicted.get('drillRemovedVolumeMm3', 0)-before.get('drillRemovedVolumeMm3', 0)
    non_target = predicted.get('nonTargetRemovedMm3', 0)-before.get('nonTargetRemovedMm3', 0)
    return (outcome.get('status') in ('accepted', 'partial') and
        outcome.get('reason') in (None, 'BONE_CUTTING_IN_PROGRESS') and bone > 0 and
        abs(non_target-bone) < 1e-8 and
        predicted.get('targetRemovedMm3', 0) == before.get('targetRemovedMm3', 0) and
        predicted.get('vesselEncounters', 0) == before.get('vesselEncounters', 0))


def propose_routes(source, proposal, *, root=None, name='006-route-proposals-v1'):
    """Read-only observed-calvarium ray proposals for independent route review."""
    from scipy.spatial.transform import Rotation
    from resectionlab.geometry import GeometryScene, capsule_voxel_indices
    root = checked_root(root, required_bytes=1024**3)
    source, proposal = Path(source).resolve(), Path(proposal).resolve()
    if not source.is_relative_to(root) or not proposal.is_relative_to(root): raise ValueError('Inputs outside guarded root')
    case, _, _ = load_workspace(source); center = np.asarray(case.metadata['data_case']['recipe']['center_ras_mm'])
    provenance = json.loads((proposal / 'provenance.json').read_text())
    def mask(name):
        row = provenance['outputs'][name]; path = root / row['path']
        if sha256_file(path) != row['sha256']: raise ValueError('Source proposal mask changed')
        image = nib.load(path)
        if image.shape != case.mri.shape or not np.array_equal(image.affine, case.affine): raise ValueError('Mask native frame differs')
        return np.asanyarray(image.dataobj) > 0
    skull = mask('observed_skull'); enclosed = mask('enclosed_coverage')
    vessels = np.logical_or.reduce([v for k, v in case.compartments.items() if k not in ('brain', 'synthetic_target', 'estimated_skull')])
    grid = GeometryScene(np.zeros(case.mri.shape, bool), case.affine)
    def cells(a, b, radius):
        ijk = capsule_voxel_indices(grid, a, b, radius)
        return np.ravel_multi_index(ijk.T, case.mri.shape) if len(ijk) else np.array([], int)
    count = 2048; indices = np.arange(count); z = 1-2*(indices+.5)/count
    phi = indices*np.pi*(3-np.sqrt(5)); directions = np.column_stack([np.sqrt(1-z*z)*np.cos(phi), np.sqrt(1-z*z)*np.sin(phi), z])
    ts = np.arange(0, 250, .5); rows = []; vessel_rejections = []
    for d in directions:
        world = center + ts[:, None]*d; voxel = case.world_to_voxel(world)
        valid = np.all((voxel >= 2.) & (voxel < np.asarray(case.mri.shape)-3.), axis=1)
        ijk = np.rint(voxel[valid]).astype(int); t = ts[valid]
        hits = np.flatnonzero(skull[tuple(ijk.T)])
        if not len(hits): continue
        inner, outer = float(t[hits[0]]), float(t[hits[-1]])
        if not 3 < outer < 35 or t[-1]-outer < 45: continue
        start = center + d*(outer+4.)
        quaternion = Rotation.align_vectors([-d], [[0., 0., 1.]])[0].as_quat().tolist()
        params = {'radiusMm': 2., 'depthMm': 2., 'contactGeometryRevision': 2}
        drill_placement = envelopes(TOOLS['cranial_drill'], params, start, quaternion)
        cusa_placement = envelopes(TOOLS['cusa'], {**params, 'radiusMm': 1.5}, start, quaternion)
        coverage = envelope_admission(drill_placement+cusa_placement, case.affine, np.array(case.mri.shape))
        if not coverage['admitted']: continue
        placement_cells = set()
        for a, b, radius, _ in drill_placement+cusa_placement:
            placement_cells.update(map(int, cells(a, b, radius)))
        placement = np.array(sorted(placement_cells), int)
        skull_contacts = int(skull.ravel()[placement].sum()); brain_contacts = int(enclosed.ravel()[placement].sum())
        if skull_contacts or brain_contacts or vessels.ravel()[placement].any(): continue
        corridor = cells(center, start, 2.)
        intrabrain = cells(center, center+d*max(0,inner-3), 1.5)
        vessel_hits = corridor[vessels.ravel()[corridor]]
        if len(vessel_hits):
            vessel_rejections.append({'outward_unit': d.tolist(), 'start_ras_mm': start.tolist(),
                'quaternion': quaternion, 'target_ras_mm': center.tolist(),
                'supplied_vessel_contact_cells': int(len(vessel_hits)),
                'first_contact_ras_mm': case.voxel_to_world(np.array(np.unravel_index(vessel_hits[0], case.mri.shape))).tolist(),
                'placement_grid_admitted': True, 'candidate_decision': 'restricted_supplied_vessel_geometry_conflict',
                'scope': 'Source-bound public capsule geometry screen; not an executed instrument action or complete vessel coverage claim'})
            continue
        rows.append({'outward_unit': d.tolist(), 'quaternion': quaternion, 'target_ras_mm': center.tolist(),
            'inner_skull_distance_mm': inner, 'outer_skull_distance_mm': outer,
            'outer_skull_ras_mm': (center+d*outer).tolist(), 'start_ras_mm': start.tolist(),
            'drill_requested_endpoint_ras_mm': (center+d*max(0,inner-3.)).tolist(),
            'full_placement_envelopes': encode_segments(drill_placement+cusa_placement),
            'placement_grid_admission': coverage, 'placement_source_CT_percentiles': np.percentile(case.mri.ravel()[placement], [0, 5, 50, 95, 100]).tolist(),
            'placement_nonair_CT_ge_minus200_voxels': int(np.count_nonzero(case.mri.ravel()[placement] >= -200)),
            'placement_source_cells': len(placement), 'intrabrain_within_enclosed_fraction': float(enclosed.ravel()[intrabrain].mean()),
            'corridor_supplied_vessel_contacts': 0,
            'remaining_observed_ray_after_bone_mm': float(t[-1]-outer)})
    rows.sort(key=lambda r: (r['placement_nonair_CT_ge_minus200_voxels'], -r['intrabrain_within_enclosed_fraction'], r['outer_skull_distance_mm']))
    directory = root / 'integrated-v1/probes' / name
    if directory.exists(): raise ValueError('Immutable route evidence exists')
    directory.mkdir(parents=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.ndimage import map_coordinates
    for index, row in enumerate(rows[:6]):
        d = np.asarray(row['outward_unit']); side = np.cross(d, [0., 0., 1.]); side /= np.linalg.norm(side)
        along = np.linspace(-8, 80, 705); across = np.linspace(-15, 15, 241)
        aa, bb = np.meshgrid(along, across)
        points = center + aa[:, :, None]*d + bb[:, :, None]*side
        voxel = case.world_to_voxel(points); coords = np.moveaxis(voxel, -1, 0)
        image = map_coordinates(case.mri, coords, order=1, mode='constant', cval=np.nan)
        fig, axes = plt.subplots(3, 1, figsize=(13, 10))
        for pane, (lo, hi) in zip(axes, [(-1024, 200), (-10, 100), (-10, 100)]):
            pane.imshow(image, origin='lower', extent=[along[0], along[-1], across[0], across[-1]], cmap='gray', vmin=lo, vmax=hi, aspect='equal')
            pane.set_xlabel('mm outward along exact target-to-calvarium ray; target center=0'); pane.set_ylabel('mm across source oblique plane')
        for array, color in [(enclosed, 'lime'), (skull, 'cyan'), (vessels, 'red'), (case.compartments['synthetic_target'], 'yellow')]:
            plane = map_coordinates(array.astype(np.uint8), coords, order=0, mode='constant', cval=0)
            if plane.any() and not plane.all(): axes[2].contour(plane, levels=[.5], colors=[color], linewidths=.6, extent=[along[0], along[-1], across[0], across[-1]])
        for pane in axes:
            start_t = row['outer_skull_distance_mm'] + 4.
            pane.plot([start_t, start_t+40], [0, 0], color='deepskyblue', linewidth=2, label='full shaft at proposed free placement')
            pane.add_patch(plt.Rectangle((start_t-2, -2), 6, 4, fill=False, edgecolor='deepskyblue'))
            pane.add_patch(plt.Rectangle((start_t, -.5), 40, 1, fill=False, edgecolor='deepskyblue'))
            pane.plot([0, 40], [0, 0], color='magenta', linewidth=1, linestyle='--', label='shaft at target pose (requires actual executed clearance)')
            pane.axvline(row['inner_skull_distance_mm'], color='orange', linewidth=.7)
            pane.axvline(row['outer_skull_distance_mm'], color='orange', linewidth=.7)
        axes[0].legend(fontsize=8, loc='upper right')
        fig.suptitle(f'{case.case_id} source proposal {index}; NOT an admitted route\nBlue proposed 40mm tool+2mm tip; magenta target pose after hypothetical genuine removal. Source scalp/other unmapped tissue remains unknown.\nActual source CT oblique display interpolation only; source/native masks unchanged. NaN/outside-FOV white.', fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, .92)); destination = directory / f'route-{index:02d}-source-full-tool.png'
        fig.savefig(destination, dpi=125); plt.close(fig)
        row['source_full_tool_contact'] = {'path': str(destination.relative_to(root)), 'sha256': sha256_file(destination)}
    report = {'schema': 'integrated-read-only-source-route-proposals-v1', 'source': str(source),
        'source_sha256': sha256_file(source), 'proposal_sha256': sha256_file(proposal / 'provenance.json'),
        'count_tested_directions': count, 'routes': rows[:12], 'accepted_routes': 0,
        'supplied_vessel_conflict_candidate_count': len(vessel_rejections),
        'supplied_vessel_conflict_candidates': vessel_rejections[:12],
        'scope': 'Read-only proposals; observed source air intensity and inferred cavity are evidence for independent review, not automated anatomical clearance or surgical access',
        'source_masks_unchanged': True, 'actions_executed': 0}
    atomic_json(directory / 'report.json', report)
    return report


def make_centerline_contract(source, route_report, route_index, *, root=None, name):
    """Propose a single working-hand drill/exchange/CUSA task for review."""
    root = checked_root(root, required_bytes=256*1024**2)
    source, route_report = Path(source).resolve(), Path(route_report).resolve()
    if not source.is_relative_to(root) or not route_report.is_relative_to(root): raise ValueError('Inputs outside guarded root')
    case, _, _ = load_workspace(source)
    if case.metadata.get('integrated_rehearsal', {}).get('stage') != 'conditional_post_scalp_exposure':
        raise ValueError('Explicit conditional scalp/dura stage required')
    routes = json.loads(route_report.read_text())
    if routes['source_sha256'] != sha256_file(source): raise ValueError('Route source differs')
    route = routes['routes'][route_index]; q = route['quaternion']; start = route['start_ras_mm']
    def pose(point): return {'position': list(point), 'quaternion': q}
    def action(label, operation, **kw): return {'label': label, 'action': {'slot': 'right', 'operation': operation, **kw}}
    actions = [action('public-revised-mechanics', 'configure', mechanicsRevision=2, clockRevision=1, duration=.1),
        action('external-cusa-conditional', 'place', pose=pose(start), duration=.1),
        {**action('skull-blocks-cusa', 'apply', path=[pose(route['drill_requested_endpoint_ras_mm'])], duration=1.), 'preview': True},
        action('checked-cusa-withdraw', 'withdraw', path=[pose(start)], duration=.1),
        action('withdrawn-exchange-to-catalog-drill', 'exchange', instrument='cranial_drill', duration=.1),
        action('external-drill-conditional', 'place', pose=pose(start), duration=.1),
        {**action('genuine-calvarial-drilling', 'apply', path=[pose(route['drill_requested_endpoint_ras_mm'])], duration=2.,
            parameters={'rpm': 60000., 'forceN': 2., 'duty': 1.}), 'repeat': 40,
            'stop_when': 'accepted_or_physical_refusal', 'continue_after_refusal': True},
        action('checked-drill-withdraw', 'withdraw', path=[pose(start)], duration=1.),
        action('withdrawn-exchange-to-catalog-cusa', 'exchange', instrument='cusa', duration=.1),
        action('original-centerline-cusa-reentry', 'place', pose=pose(start), duration=.1),
        {**action('genuine-intracranial-target-approach', 'apply', path=[pose(route['target_ras_mm'])], duration=2.),
            'repeat': 40, 'stop_when': 'target_effect_or_physical_refusal', 'continue_after_refusal': True},
        action('checked-final-withdraw', 'withdraw', path=[pose(start)], duration=1.)]
    all_segments = []
    for tool in (TOOLS['cusa'], TOOLS['cranial_drill']):
        params = {'radiusMm': tool['radiusMm'], 'depthMm': 2., 'contactGeometryRevision': 2}
        for point in (start, route['target_ras_mm'], route['drill_requested_endpoint_ras_mm']):
            all_segments.extend(envelopes(tool, params, point, q))
    inverse = np.linalg.inv(case.affine); margins = np.linalg.norm(inverse[:3, :3], axis=1)
    lows, highs = [], []
    for aa, bb, radius, part in all_segments:
        points = np.asarray([aa, bb]) @ inverse[:3, :3].T + inverse[:3, 3]
        lows.append(np.floor(points.min(0)-radius*margins-1).astype(int)); highs.append(np.ceil(points.max(0)+radius*margins+1).astype(int))
    low = np.maximum(np.min(lows, axis=0), 0); high = np.minimum(np.max(highs, axis=0), np.asarray(case.mri.shape)-1)
    mask = np.zeros(case.mri.shape, np.uint8); mask[tuple(slice(l, h+1) for l, h in zip(low, high))] = 1
    # This pipeline's native axial cavity lower plane is an artificial cutoff.
    # A bounded workspace may not turn it into a portal.
    if not np.allclose(case.affine[:3, :3], np.diag(np.diag(case.affine[:3, :3]))) or case.affine[2, 2] <= 0:
        raise ValueError('Current proposal needs native superior-increasing axial grid')
    occupied_z = np.where(np.any(case.compartments['brain'], axis=(0, 1)))[0]
    forbidden_below = occupied_z[0] + int(np.ceil(3./case.spacing_mm[2]))
    mask[:, :, :forbidden_below] = 0
    coverage = envelope_admission(all_segments, case.affine, np.array(case.mri.shape), mask)
    directory = root / 'integrated-v1/probes' / name
    if directory.exists(): raise ValueError('Immutable proposal exists')
    directory.mkdir(parents=True); mask_path = directory / 'conditional-workspace-proposal.nii.gz'
    nib.save(nib.Nifti1Image(mask, case.affine), mask_path)
    contract = {'schema': 'integrated-single-working-hand-conditional-proposal-v1', 'status': 'pending_independent_review',
        'source_sha256': sha256_file(source), 'source_path': str(source.relative_to(root)),
        'scope': 'Conditional post-scalp-exposure rigid-tool test; scalp exposure/dural opening unmodeled. Catalog tool exchange only while withdrawn. Real skull/brain/vessel/tool guards and fullsourceFOV apply; no intacthead or clinicalsafeclaim.',
        'reviewed_observation_mask': {'path': str(mask_path.relative_to(root)), 'sha256': sha256_file(mask_path),
            'review_evidence': None, 'meaning': 'Proposed finite engineering workspace within native FOV and above artificial lower cap; not measured absence of superficial anatomy'},
        'route_report': str(route_report.relative_to(root)), 'route_report_sha256': sha256_file(route_report),
        'route_index': route_index, 'static_full_tool_scope_check': coverage, 'actions': actions}
    atomic_json(directory / 'contract-proposal.json', contract)
    return {'contract_path': str(directory / 'contract-proposal.json'), 'proposal_sha256': sha256_file(directory / 'contract-proposal.json'),
        'scope': coverage, 'mask': contract['reviewed_observation_mask']}


def render_one_route(source, route_report, route_index, *, root=None, name):
    """Additional exact source/full-tool review view; never modify prior evidence."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.ndimage import map_coordinates
    root = checked_root(root, required_bytes=256*1024**2)
    source, route_report = Path(source).resolve(), Path(route_report).resolve()
    if not source.is_relative_to(root) or not route_report.is_relative_to(root): raise ValueError('Inputs outside guarded root')
    report = json.loads(route_report.read_text())
    if report['source_sha256'] != sha256_file(source): raise ValueError('Route source snapshot differs')
    case, _, _ = load_workspace(source); route = report['routes'][route_index]
    center, d = np.asarray(route['target_ras_mm']), np.asarray(route['outward_unit'])
    side = np.cross(d, [0., 0., 1.]); side /= np.linalg.norm(side)
    along, across = np.linspace(-8, 80, 705), np.linspace(-15, 15, 241)
    aa, bb = np.meshgrid(along, across); world = center + aa[:, :, None]*d + bb[:, :, None]*side
    coordinates = np.moveaxis(case.world_to_voxel(world), -1, 0)
    image = map_coordinates(case.mri, coordinates, order=1, mode='constant', cval=np.nan)
    fig, axes = plt.subplots(3, 1, figsize=(13, 10)); extent = [along[0], along[-1], across[0], across[-1]]
    for pane, (low, high) in zip(axes, [(-1024, 200), (-10, 100), (-10, 100)]):
        pane.imshow(image, origin='lower', cmap='gray', extent=extent, vmin=low, vmax=high, aspect='equal')
        pane.set_xlabel('mm outward along exact target-to-calvarium ray; target center=0'); pane.set_ylabel('mm across source oblique plane')
        t = route['outer_skull_distance_mm']+4.
        pane.plot([t, t+40], [0, 0], color='deepskyblue', linewidth=1)
        pane.add_patch(plt.Rectangle((t, -.5), 40, 1, fill=False, edgecolor='deepskyblue'))
        pane.add_patch(plt.Rectangle((t-2, -2), 6, 4, fill=False, edgecolor='deepskyblue'))
        pane.plot([0, 40], [0, 0], color='magenta', linestyle='--', linewidth=1)
        pane.axvline(route['inner_skull_distance_mm'], color='orange', linewidth=.7); pane.axvline(route['outer_skull_distance_mm'], color='orange', linewidth=.7)
    for layer, mask in case.compartments.items():
        color = {'brain': 'lime', 'synthetic_target': 'yellow', 'estimated_skull': 'cyan'}.get(layer, 'red')
        plane = map_coordinates(mask.astype(np.uint8), coordinates, order=0, mode='constant', cval=0)
        if plane.any() and not plane.all(): axes[2].contour(plane, levels=[.5], colors=[color], linewidths=.6, extent=extent)
    fig.suptitle(case.case_id+f' | exact route {route_index}; NOT yet an admitted action\nBlue unchanged catalog40mmshaft+.5mmradius/drill2mmtip; magenta targetpose requiresgenuineexecutedclearance\nActual baseline CTA displayed obliquely; scalp exposure/dural opening unmodeled; allsource masks retained; outside-FOV white', fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, .92)); directory = root / 'integrated-v1/probes' / name
    if directory.exists(): raise ValueError('Immutable review view exists')
    directory.mkdir(parents=True); destination = directory / f'route-{route_index:02d}-source-full-tool.png'
    fig.savefig(destination, dpi=125); plt.close(fig)
    evidence = {'path': str(destination.relative_to(root)), 'sha256': sha256_file(destination),
        'source_sha256': sha256_file(source), 'route_report_sha256': sha256_file(route_report), 'route_index': route_index,
        'route': route, 'actions_executed': 0, 'source_masks_modified': False}
    atomic_json(directory / 'report.json', evidence); return evidence


def run_case(source, contract_path, *, root=None, name):
    """Execute an explicit bounded source program with independent FOV restriction.

    Contract observations require a hash-bound native NIfTI observation mask.
    Without it, every trajectory is restricted, even when the engine could
    otherwise traverse zero-labelled grid cells. No outside-image allowance is
    implicit. A contract is a research probe declaration, not clinical approval.
    """
    root = checked_root(root, required_bytes=1024**3)
    source, contract_path = Path(source).resolve(), Path(contract_path).resolve()
    if not source.is_relative_to(root) or not contract_path.is_relative_to(root):
        raise ValueError('Probe source and contract must remain on guarded external root')
    contract = json.loads(contract_path.read_text()); source_sha = sha256_file(source)
    if contract.get('source_sha256') != source_sha: raise ValueError('Probe contract source checksum differs')
    declarations = contract['actions']
    if not isinstance(declarations, list) or len(declarations) > 128: raise ValueError('At most128 explicit action groups')
    actions = []
    for group, declaration in enumerate(declarations):
        repeat = declaration.get('repeat', 1)
        if type(repeat) is not int or not 1 <= repeat <= 64: raise ValueError('Finite repeat bound1..64 required')
        for iteration in range(repeat):
            actions.append({**declaration, '_group': group, '_iteration': iteration})
    if len(actions) > 128: raise ValueError('At most128 total bounded actions')
    case, _, _ = load_workspace(source)
    observed = None; observed_binding = contract.get('reviewed_observation_mask')
    if observed_binding is not None:
        path = (root / observed_binding['path']).resolve()
        if not path.is_relative_to(root) or sha256_file(path) != observed_binding['sha256']:
            raise ValueError('Observation mask checksum/path differs')
        image = nib.load(path)
        if image.shape != case.mri.shape or not np.array_equal(image.affine, case.affine):
            raise ValueError('Observation mask differs from source physical frame')
        raw = np.asanyarray(image.dataobj)
        if not np.isin(raw, [0, 1]).all(): raise ValueError('Binary observation mask required')
        observed = raw.astype(bool)
        evidence = observed_binding.get('review_evidence')
        if not isinstance(evidence, dict) or set(evidence) != {'path', 'sha256'}:
            raise ValueError('Observation mask needs hash-bound independent review evidence')
        review_path = (root / evidence['path']).resolve()
        if not review_path.is_relative_to(root) or sha256_file(review_path) != evidence['sha256']:
            raise ValueError('Independent observation review bytes differ')
    prefix_burden = contract.get('temporal_burden_prefix_evidence')
    if prefix_burden is not None:
        prefix_path = (root / prefix_burden['path']).resolve()
        if not prefix_path.is_relative_to(root) or sha256_file(prefix_path) != prefix_burden['sha256']:
            raise ValueError('Saved physical prefix burden evidence differs')
        actual_burden = json.loads(prefix_path.read_text())['trajectory_max_recorded_spatial_temperature_C']
        if actual_burden != prefix_burden['trajectory_max_recorded_spatial_temperature_C']:
            raise ValueError('Declared temporal prefix burden differs from actual diagnostic')
    directory = root / 'integrated-v1/probes' / name
    if directory.exists(): raise ValueError('Preserve prior probe output; choose a new name')
    directory.mkdir(parents=True); started = time.perf_counter()
    records = []; session = BridgeSession(directory / 'transfers', run_dir=directory / 'runs')
    def call(operation, args): return session.execute(operation, args, _Request(operation), lambda *_: None)
    try:
        identity = call('loadCase', {'path': str(source)})['caseHash']
        current = call('surgicalRehearsal', {'caseHash': identity,
            'command': 'inspect' if contract.get('resume_saved_state', False) else 'create'})
        initial_view = current['view']
        if contract.get('initial_state_id') is not None and contract['initial_state_id'] != initial_view['stateId']:
            raise ValueError('Saved public physical prefix differs from admitted initial state')
        stopped_groups = set()
        for declaration in actions:
            if declaration['_group'] in stopped_groups: continue
            value = declaration['action']; hand = value['slot']; before = current['view']; slot = before['slots'][hand]
            tool = TOOLS[slot['instrument']]
            params = {'radiusMm': tool['radiusMm'], 'depthMm': 2., 'contactGeometryRevision': 2,
                      **slot.get('parameters', {}), **value.get('parameters', {})}
            if 'radiusMm' in value.get('parameters', {}) and value['parameters']['radiusMm'] != tool['radiusMm']:
                raise ValueError('Probe cannot change catalog tool dimensions')
            poses = value.get('path', [value['pose']] if 'pose' in value else [])
            if value['operation'] == 'withdraw' and not poses:
                raise ValueError('Explicit finite withdrawal path required for FOV review')
            geometry = []; previous = slot
            if value['operation'] == 'place':
                if len(poses) != 1: raise ValueError('One explicit external placement required')
                geometry = envelopes(tool, params, poses[0]['position'], poses[0]['quaternion'])
            else:
                for end in poses:
                    geometry.extend(bounded_motion_envelopes(tool, params, previous, end)); previous = end
                if not poses:
                    geometry = envelopes(tool, params, slot['position'], slot['quaternion'])
            trajectory = value['operation'] not in ('configure', 'exchange', 'inspect')
            if trajectory:
                if observed is None:
                    coverage = {'admitted': False, 'reasons': ['missing-reviewed-observation-mask']}
                else:
                    coverage = envelope_admission(geometry, case.affine, np.array(case.mri.shape), observed)
            else:
                coverage = {'admitted': True, 'reasons': [], 'policy': 'no physical trajectory'}
            if coverage['admitted']:
                if declaration.get('preview_before_commit', False):
                    candidate = call('surgicalRehearsal', {'caseHash': identity, 'command': 'preview',
                        'expectedStateId': before['stateId'], 'action': value})
                    outcome = candidate.get('outcome', candidate); metrics = candidate.get('view', {}).get('metrics', outcome.get('metrics', {}))
                    accept = admissible_bone_preview(before['metrics'], outcome, metrics)
                    records.append({'label': declaration['label']+'-candidate-preview', 'repeatIteration': declaration['_iteration'],
                        'action': value, 'previewOnly': True, 'coverageAdmission': coverage,
                        'fullToolEnvelopes': encode_segments(geometry), 'stateBefore': before['stateId'], 'stateAfter': before['stateId'],
                        'slotsBefore': before['slots'], 'slotsAfter': before['slots'], 'outcome': outcome,
                        'candidateMetrics': metrics, 'metricsAfter': before['metrics'], 'admittedForCommit': accept})
                    atomic_json(directory / 'actions.json', records)
                    if not accept:
                        stopped_groups.add(declaration['_group']); continue
                result = call('surgicalRehearsal', {'caseHash': identity,
                    'command': 'preview' if declaration.get('preview', False) else 'step',
                    'expectedStateId': before['stateId'], 'action': value})
                if not declaration.get('preview', False): current = result
            else:
                result = {'outcome': {'status': 'restricted', 'reason': 'INSUFFICIENT_OBSERVED_FULL_TOOL_ENVELOPE'}}
            row = {'label': declaration['label'], 'repeatIteration': declaration['_iteration'], 'action': value, 'previewOnly': declaration.get('preview', False),
                'coverageAdmission': coverage, 'fullToolEnvelopes': encode_segments(geometry),
                'slotsBefore': before['slots'], 'slotsAfter': current['view']['slots'],
                'stateBefore': before['stateId'], 'stateAfter': current['view']['stateId'],
                'outcome': result.get('outcome', result), 'metricsAfter': current['view']['metrics']}
            records.append(row); atomic_json(directory / 'actions.json', records)
            stop_rule = declaration.get('stop_when')
            physical_stop = row['outcome'].get('reason') not in (None, 'BONE_CUTTING_IN_PROGRESS', 'TISSUE_ASPIRATION_IN_PROGRESS')
            if stop_rule == 'accepted_or_physical_refusal' and (physical_stop or row['outcome'].get('status') == 'accepted'):
                stopped_groups.add(declaration['_group'])
            elif stop_rule == 'target_effect_or_physical_refusal' and (physical_stop or current['view']['metrics'].get('targetRemovedMm3', 0) > 0):
                stopped_groups.add(declaration['_group'])
            elif stop_rule not in (None, 'accepted_or_physical_refusal', 'target_effect_or_physical_refusal'):
                raise ValueError('Unknown explicit repeat stop rule')
            # A failed preparation cannot silently become a later successful
            # access assertion. The contract must explicitly request continuation.
            if (not declaration.get('preview', False) and row['outcome'].get('status') in ('restricted', 'rejected')
                    and not declaration.get('continue_after_refusal', False)): break
        saved = directory / 'replay.ressectionlab'
        call('saveCase', {'caseHash': identity, 'path': str(saved)})
        reopened, artifacts, _ = load_workspace(saved)
        from resectionlab.surgical_engine import SurgicalEngine
        from resectionlab.surgical_environment import standardize
        from resectionlab.surgical_scene import from_case
        record = artifacts['surgical_toolbox']; snapshot = record['branches'][record['branchId']]
        replayed = SurgicalEngine.restore(standardize(from_case(reopened), physics_profile=record['physicsProfile']), snapshot, use_cache=False)
        report = {'schema': 'resectionlab.integrated-source-program.v1', 'source': str(source),
            'source_sha256': source_sha, 'source_case_hash': case.semantic_hash,
            'contract_path': str(contract_path), 'contract_sha256': sha256_file(contract_path),
            'saved': str(saved), 'saved_sha256': sha256_file(saved),
            'source_case_preserved': reopened.semantic_hash == case.semantic_hash and sha256_file(source) == source_sha,
            'exact_replay': replayed.state_id == current['view']['stateId'], 'state_id': replayed.state_id,
            'metrics': replayed.metrics(), 'deformation_hash': semantic_digest(replayed.state.get('deformation', {})),
            'initial_state_id': initial_view['stateId'], 'initial_metrics': initial_view['metrics'],
            'saved_physical_prefix_resumed': bool(contract.get('resume_saved_state', False)),
            'observed_mask': observed_binding, 'requested_actions': len(actions), 'recorded_actions': len(records),
            'FOV_restricted': any(not row['coverageAdmission']['admitted'] for row in records),
            'all_tools_withdrawn': all(not slot['inserted'] and not slot['active'] and not slot['grip'] for slot in replayed.state['slots'].values()),
            'elapsed_seconds': time.perf_counter() - started,
            'scope': 'Explicit source-bound engineering probe; outside-FOV unknown is independently restricted; no clinical safety or anatomy completeness claim',
            'independent_promotion': False}
        report['integrated_rehearsal'] = dict(case.metadata.get('integrated_rehearsal', {}))
        report['observation_contract_scope'] = contract.get('scope')
        report['temporal_burden_prefix_evidence'] = contract.get('temporal_burden_prefix_evidence')
        report['trajectory_max_recorded_spatial_temperature_C'] = max(
            [row['metricsAfter'].get('drillMaxTemperatureC', 37.) for row in records] +
            [contract.get('temporal_burden_prefix_evidence', {}).get('trajectory_max_recorded_spatial_temperature_C', 37.)])
        atomic_json(directory / 'report.json', report); return report
    finally:
        session.close()


def run_control(root=None, name='control'):
    root = checked_root(root, required_bytes=1024**3)
    directory = root / 'integrated-v1/probes' / name
    if directory.exists():
        raise ValueError('Preserve previous evidence; choose a fresh --name')
    directory.mkdir(parents=True)
    case = control_case(); source = directory / 'source.ressectionlab'
    save_workspace(case, source, sources=[], imaging=view_state(case, []), episode=None,
                   selection=None, artifacts={}, originals=[])
    session = BridgeSession(directory / 'transfers', run_dir=directory / 'runs')
    records = []; started = time.perf_counter(); current = None
    def call(operation, args):
        return session.execute(operation, args, _Request(operation), lambda *_: None)
    try:
        installed = call('loadCase', {'path': str(source)}); identity = installed['caseHash']
        current = call('surgicalRehearsal', {'caseHash': identity, 'command': 'create'})
        def action(value, label, preview=False):
            nonlocal current
            before = current['view']; hand = value['slot']; slot = before['slots'][hand]
            tool = TOOLS[slot['instrument']]
            parameters = {'radiusMm': tool['radiusMm'], 'depthMm': 2., 'contactGeometryRevision': 2,
                          **slot.get('parameters', {}), **value.get('parameters', {})}
            poses = value.get('path', [value['pose']] if 'pose' in value else [])
            geometry = []; pose = slot
            for end in poses:
                geometry.extend(swept_envelopes(tool, parameters, pose, end)); pose = end
            if not poses: geometry = envelopes(tool, parameters, slot['position'], slot['quaternion'])
            coverage = envelope_admission(geometry, case.affine, np.array(case.mri.shape))
            # place is an explicit external relocation while withdrawn; its
            # endpoint still requires observed full geometry and native checks.
            if value['operation'] == 'place':
                geometry = envelopes(tool, parameters, poses[0]['position'], poses[0]['quaternion'])
                coverage = envelope_admission(geometry, case.affine, np.array(case.mri.shape))
            if value['operation'] in ('configure', 'exchange', 'inspect'):
                coverage = {'admitted': True, 'reasons': [], 'policy': 'no instrument trajectory'}
            if not coverage['admitted']:
                result = {'outcome': {'status': 'restricted', 'reason': 'INSUFFICIENT_OBSERVED_FULL_TOOL_ENVELOPE'}}
            else:
                result = call('surgicalRehearsal', {'caseHash': identity,
                    'command': 'preview' if preview else 'step', 'expectedStateId': before['stateId'], 'action': value})
                if not preview: current = result
            records.append({'label': label, 'previewOnly': preview, 'action': value,
                'coverageAdmission': coverage, 'fullToolEnvelopes': encode_segments(geometry),
                'stateBefore': before['stateId'], 'stateAfter': current['view']['stateId'],
                'slotsBefore': before['slots'], 'slotsAfter': current['view']['slots'],
                'outcome': result.get('outcome', result), 'metricsAfter': current['view']['metrics']})
            atomic_json(directory / 'actions.json', records)
            return result.get('outcome', result)
        q = [1., 0., 0., 0.]
        def pose(x, z): return {'position': [x, 48., z], 'quaternion': q}
        action({'slot': 'right', 'operation': 'configure', 'mechanicsRevision': 2, 'clockRevision': 1, 'duration': .1}, 'public-revised-mechanics')
        action({'slot': 'right', 'operation': 'place', 'pose': pose(48, 82), 'duration': .1}, 'external-cusa')
        action({'slot': 'right', 'operation': 'apply', 'path': [pose(48, 74)], 'duration': 1.}, 'skull-blocks-cusa', True)
        action({'slot': 'right', 'operation': 'withdraw', 'path': [pose(48, 82)], 'duration': .1}, 'cusa-withdraw-before-opening')
        action({'slot': 'right', 'operation': 'place', 'pose': pose(80, 82), 'duration': .1}, 'park-withdrawn-cusa')
        action({'slot': 'left', 'operation': 'place', 'pose': pose(48, 82), 'duration': .1}, 'external-drill')
        for index in range(16):
            outcome = action({'slot': 'left', 'operation': 'apply', 'path': [pose(48, 73)], 'duration': 1.,
                'parameters': {'rpm': 60000., 'forceN': 2., 'duty': 1.}}, 'genuine-bone-opening-' + str(index))
            if outcome.get('status') == 'rejected': break
            if current['view']['slots']['left']['position'][2] <= 73.01: break
        action({'slot': 'left', 'operation': 'withdraw', 'path': [pose(48, 83)], 'duration': 1.}, 'checked-drill-withdraw')
        action({'slot': 'left', 'operation': 'place', 'pose': pose(20, 83), 'duration': .1}, 'park-withdrawn-drill')
        action({'slot': 'right', 'operation': 'place', 'pose': pose(48, 82), 'duration': .1}, 'cusa-reentry')
        for index in range(100):
            outcome = action({'slot': 'right', 'operation': 'apply', 'path': [pose(48, 66)], 'duration': 2.}, 'bounded-target-approach-' + str(index))
            if outcome.get('status') == 'rejected' or current['view']['metrics'].get('targetRemovedMm3', 0) > 0: break
        action({'slot': 'right', 'operation': 'withdraw', 'path': [pose(48, 82)], 'duration': 1.}, 'checked-cusa-withdraw')
        # Vessel conflict from source-supported inferior free region; skull is
        # not on this line. No vessel/skull is erased to create this candidate.
        up = [0., 0., 0., 1.]
        action({'slot': 'right', 'operation': 'place', 'pose': {'position': [75., 48., 60.], 'quaternion': up}, 'duration': .1}, 'vessel-conflict-external')
        action({'slot': 'right', 'operation': 'apply', 'path': [{'position': [75., 48., 71.], 'quaternion': up}], 'duration': 2.}, 'vessel-conflicting-approach', True)
        action({'slot': 'right', 'operation': 'withdraw', 'path': [{'position': [75., 48., 60.], 'quaternion': up}], 'duration': .1}, 'checked-final-withdraw')
        saved = directory / 'replay.ressectionlab'; call('saveCase', {'caseHash': identity, 'path': str(saved)})
        reopened, artifacts, workspace = load_workspace(saved)
        snapshot = artifacts['surgical_toolbox']['branches']['main']
        from resectionlab.surgical_engine import SurgicalEngine
        from resectionlab.surgical_environment import standardize
        from resectionlab.surgical_scene import from_case
        replayed = SurgicalEngine.restore(standardize(from_case(reopened)), snapshot, use_cache=False)
        by_label = {row['label']: row for row in records}
        checks = {
            'skullBlocksCusa': by_label['skull-blocks-cusa']['outcome'].get('reason') in ('UNSUPPORTED_MATERIAL', 'SHAFT_COLLISION'),
            'genuineDrillRemoval': replayed.metrics()['drillRemovedVolumeMm3'] > 0,
            'targetEffect': replayed.metrics()['targetRemovedMm3'] > 0,
            'vesselConflictBlocksCusa': by_label['vessel-conflicting-approach']['outcome'].get('reason') in ('UNSUPPORTED_MATERIAL', 'SHAFT_COLLISION'),
            'nativeCheckedWithdrawal': by_label['checked-cusa-withdraw']['outcome']['status'] == 'accepted',
            'finalBothToolsWithdrawn': all(not slot['inserted'] and not slot['active'] and not slot['grip'] for slot in replayed.state['slots'].values()),
            'previewPreservesState': all(row['stateBefore'] == row['stateAfter'] for row in records if row['previewOnly']),
            'allRequestedMotionObserved': all(row['coverageAdmission']['admitted'] for row in records),
            'exactReplay': replayed.state_id == current['view']['stateId'],
            'sourceCasePreserved': reopened.semantic_hash == case.semantic_hash}
        report = {'schema': 'resectionlab.integrated-public-control.v1', 'sourcePeople': 0,
            'checks': checks, 'passed': all(checks.values()),
            'sourcePath': str(source), 'savedPath': str(saved), 'sourceSha256': sha256_file(source),
            'savedSha256': sha256_file(saved), 'caseHashPreserved': reopened.semantic_hash == case.semantic_hash,
            'sourceArrayHashes': {'image': array_digest(case.mri), **{k: array_digest(v) for k, v in case.compartments.items()}},
            'exactReplayStateId': replayed.state_id, 'savedStateId': current['view']['stateId'],
            'exactReplay': replayed.state_id == current['view']['stateId'], 'metrics': replayed.metrics(),
            'deformationHash': semantic_digest(replayed.state.get('deformation', {})),
            'elapsedSeconds': time.perf_counter() - started, 'actionCount': len(records),
            'scope': 'Labelled generated software control only; no acquired person, clinical or anatomical validity claim'}
        atomic_json(directory / 'report.json', report)
        return report
    finally:
        session.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--root'); parser.add_argument('--name', default='control')
    parser.add_argument('--case'); parser.add_argument('--contract'); parser.add_argument('--proposal')
    parser.add_argument('--route-report'); parser.add_argument('--route-index', type=int, default=0)
    parser.add_argument('--render-route', action='store_true')
    args = parser.parse_args()
    if args.route_report:
        if not args.case or args.contract or args.proposal: parser.error('--route-report requires --case only')
        factory = render_one_route if args.render_route else make_centerline_contract
        result = factory(args.case, args.route_report, args.route_index, root=args.root, name=args.name)
    elif args.proposal:
        if not args.case or args.contract: parser.error('--proposal requires --case and excludes --contract')
        result = propose_routes(args.case, args.proposal, root=args.root, name=args.name)
    else:
        if bool(args.case) != bool(args.contract): parser.error('--case and --contract must be paired')
        result = run_case(args.case, args.contract, root=args.root, name=args.name) if args.case else run_control(args.root, args.name)
    print(json.dumps(result, indent=2))
