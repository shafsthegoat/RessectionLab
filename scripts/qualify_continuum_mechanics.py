#!/usr/bin/env python3
"""Continuum numerical controls and optional exposed-specimen comparison.

No fitting, raw specimen access, reserved mode reads, patient decoding or RL.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse
import json
import platform
import sys
import time

import numpy as np
from scipy.sparse.linalg import spsolve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from resectionlab.surgical_scene import SurgicalScene
from resectionlab.surgical_physics_v2 import default_profile
from resectionlab import surgical_continuum as continuum
from resectionlab import surgical_deformation_v2 as legacy
from resectionlab.simulator_qualification import file_sha256


def material(mu=258.77, nu=.45):
    return {'shearPa': mu, 'poissonRatio': nu, 'relaxationShearPa': [],
        'source': 'Hinrichsen2023 Table11 preconditioned cortex, assumed nu0.45; postmortem cohort scenario, not patient measurement'}


def source(labels, affine):
    return SurgicalScene(labels, affine, [{'id': 1, 'name': 'Numerical specimen', 'material': 'soft_tissue'}],
        {'provenance': 'generated numerical geometry; not a patient'},
        {'tissuePhysicsV2': default_profile(), continuum.PROPERTY: continuum.research_profile({'1': material()})})


def prescribed_solve(matrix, prescribed):
    """Separate static Dirichlet checker, independent of runtime step/holds."""
    u = np.zeros(matrix.shape[0]); free = np.ones(len(u), bool)
    for dof, value in prescribed.items(): u[dof] = value; free[dof] = False
    if free.any(): u[free] = spsolve(matrix[free][:, free], -(matrix @ u)[free])
    force = matrix @ u
    if not np.isfinite(u).all() or np.max(np.abs(force[free]), initial=0.) > 1e-8:
        raise ValueError('Independent static checker residual failure')
    return u, force


def grid_control(spacing):
    side = 2.; n = round(side / spacing)
    case = source(np.ones((n, n, n), np.uint16), np.diag([spacing]*3 + [1.]))
    ids = list(range(case.labels.size)); mesh = continuum.voxel_mesh(case, ids)
    matrix = continuum.assemble(case, mesh, case.properties[continuum.PROPERTY]['materials'])[0][0]
    gradient = np.diag([.01, 0., 0.])
    u = (mesh.nodes @ gradient.T).ravel()
    energy = float(.5 * u @ (matrix @ u))
    mu = 258.77e-6; nu = .45; lam = 2 * mu * nu / (1 - 2 * nu)
    expected = (mu + .5 * lam) * .01**2 * side**3
    forces = (matrix @ u).reshape((-1, 3))
    right = np.isclose(mesh.nodes[:, 0], mesh.nodes[:, 0].max())
    load = float(forces[right, 0].sum())
    expected_load = (2 * mu + lam) * .01 * side**2
    old = SimpleNamespace(scene=case, state={'holds': {}}, by_label={1: {'material': 'soft_tissue'}})
    bonds = sorted((c, other) for c in ids for other, _ in legacy._neighbors(old, c) if c < other)
    old_matrix = legacy._system(old, legacy.profile(), ids, [], bonds)[-1]
    centers = case.points(ids); old_u = (centers @ gradient.T).ravel()
    old_energy = float(.5 * old_u @ (old_matrix @ old_u))
    rotation = np.array([[0., -.01, 0.], [.01, 0., 0.], [0., 0., 0.]])
    rigid = (mesh.nodes @ rotation.T).ravel()
    old_rigid = (centers @ rotation.T).ravel()
    return {'spacingMm': spacing, 'physicalSideMm': side, 'sourceCells': len(ids), 'cornerDofs': len(u),
        'continuumEnergyNmm': energy, 'analyticEnergyNmm': expected,
        'continuumResultantN': load, 'analyticResultantN': expected_load,
        'energyRelativeError': abs(energy - expected) / expected,
        'resultantRelativeError': abs(load - expected_load) / expected_load,
        'continuumInfinitesimalRotationEnergyNmm': float(.5 * rigid @ (matrix @ rigid)),
        'legacyInfinitesimalRotationEnergyNmm': float(.5 * old_rigid @ (old_matrix @ old_rigid)),
        'legacyEnergyNmm': old_energy,
        'comparisonScope': 'Same source-owned physical voxel cube; old lattice samples centers and includes its original weak foundation and divergence penalty'}


def uniaxial_control(spacing):
    side = 2.; n = round(side / spacing)
    case = source(np.ones((n, n, n), np.uint16), np.diag([spacing]*3 + [1.]))
    mesh = continuum.voxel_mesh(case, range(case.labels.size))
    matrix = continuum.assemble(case, mesh, case.properties[continuum.PROPERTY]['materials'])[0][0]
    x = mesh.nodes; minimum = x.min(axis=0); maximum = x.max(axis=0)
    left = np.isclose(x[:, 0], minimum[0]); right = np.isclose(x[:, 0], maximum[0])
    prescribed = {3*int(i): 0. for i in np.flatnonzero(left)}
    prescribed.update({3*int(i): .01 * side for i in np.flatnonzero(right)})
    origin = int(np.flatnonzero(np.all(np.isclose(x, minimum), axis=1))[0])
    prescribed[3*origin+1] = 0.; prescribed[3*origin+2] = 0.
    other = int(np.flatnonzero(left & np.isclose(x[:, 1], maximum[1]) & np.isclose(x[:, 2], minimum[2]))[0])
    prescribed[3*other+2] = 0.
    u, force = prescribed_solve(matrix, prescribed)
    expected_u = (x - minimum) * [.01, -.45 * .01, -.45 * .01]
    expected_force = 2 * 258.77e-6 * (1 + .45) * .01 * side**2
    resultant = float(force.reshape((-1, 3))[right, 0].sum())
    return {'spacingMm': spacing, 'tractionFreeLateralMaxDisplacementErrorMm': float(np.max(np.abs(u.reshape((-1, 3)) - expected_u))),
        'resultantN': resultant, 'analyticResultantN': expected_force,
        'forceRelativeError': abs(resultant - expected_force) / expected_force}


def specimen_comparison(protocol):
    declaration = protocol['measured_development_comparison']; path = ROOT / declaration['source']
    if file_sha256(path) != declaration['source_sha256']:
        raise ValueError('Exposed development receipt changed; no reference values admitted')
    measured = json.loads(path.read_text())
    if measured['held_out_values_used'] is not False:
        raise ValueError('Only the already exposed axial development receipt is admitted')
    height = 4.89159; radius = 4.; results = []
    for spacing, layers in ((1., 5), (.5, 10)):
        width = round(2 * radius / spacing); shape = (width, width, layers)
        axis = (np.arange(width) + .5) * spacing - radius
        labels = np.broadcast_to((axis[:, None]**2 + axis[None, :]**2 <= radius**2)[:, :, None], shape).astype(np.uint16).copy()
        affine = np.diag([spacing, spacing, height / layers, 1.]); affine[:3, 3] = [axis[0], axis[0], height / layers / 2.]
        case = source(labels, affine); ids = list(map(int, np.flatnonzero(labels)))
        mesh = continuum.voxel_mesh(case, ids)
        matrix = continuum.assemble(case, mesh, case.properties[continuum.PROPERTY]['materials'])[0][0]
        bottom = np.isclose(mesh.nodes[:, 2], 0.); top = np.isclose(mesh.nodes[:, 2], height)
        prescribed = {3*int(node)+a: (1. if top[node] and a == 2 else 0.)
            for node in np.flatnonzero(top | bottom) for a in range(3)}
        u, reaction = prescribed_solve(matrix, prescribed)
        revised_stiffness = float(reaction.reshape((-1, 3))[top, 2].sum())
        old = SimpleNamespace(scene=case, state={'holds': {}}, by_label={1: {'material': 'soft_tissue'}})
        owned = set(ids)
        bonds = sorted((c, other) for c in ids for other, _ in legacy._neighbors(old, c) if c < other and other in owned)
        old_matrix = legacy._system(old, legacy.profile(), ids, [], bonds)[-1]
        centers = case.points(ids)
        old_bottom = np.isclose(centers[:, 2], centers[:, 2].min()); old_top = np.isclose(centers[:, 2], centers[:, 2].max())
        fixed = {3*int(node)+a: (1. if old_top[node] and a == 2 else 0.)
            for node in np.flatnonzero(old_top | old_bottom) for a in range(3)}
        old_u, old_reaction = prescribed_solve(old_matrix, fixed)
        baseline_stiffness = float(old_reaction.reshape((-1, 3))[old_top, 2].sum())
        modes = {}
        for name, data in measured['modes'].items():
            coordinate = np.asarray(data['coordinate']) * 1000.
            observed = np.asarray(data['observed'])
            chosen = np.abs(coordinate / height) <= .02
            if not chosen.any(): raise ValueError('No declared small-strain reference points')
            x, y = coordinate[chosen], observed[chosen]
            endpoint = int(np.argmax(np.abs(x)))
            predictions = {'legacyDefault': baseline_stiffness * x, 'continuumPublishedScenario': revised_stiffness * x}
            modes[name] = {'role': 'already exposed development, not independent holdout', 'pointCount': len(x),
                'displacementMm': x.tolist(), 'observedForceN': y.tolist(),
                'predictions': {model: {'forceN': value.tolist(), 'forceRmseN': float(np.sqrt(np.mean((value-y)**2))),
                    'signedEndpointErrorN': float(value[endpoint]-y[endpoint])} for model, value in predictions.items()}}
        actual_volume = len(ids) * case.voxel_mm3
        results.append({'spacingXyMm': spacing, 'spacingZMm': height / layers, 'sourceElements': len(ids),
            'cornerDofs': len(u), 'voxelVolumeMm3': actual_volume,
            'nominalCylinderVolumeMm3': float(np.pi * radius**2 * height),
            'volumeRelativeError': actual_volume / (np.pi * radius**2 * height) - 1.,
            'legacyAxialStiffnessNPerMm': baseline_stiffness, 'continuumAxialStiffnessNPerMm': revised_stiffness,
            'modes': modes})
    return {'source': declaration['source'], 'sourceSha256': declaration['source_sha256'],
        'parameterFitting': False, 'newHeldOutReads': 0,
        'sourceScope': 'Independent acquired specimen measurements already exposed for development; published cohort parameters overlap the donor',
        'limits': ['Processed quasi-static curves do not validate relaxation.',
            'Material parameters and boundary representation differ between models; error differences cannot isolate discretization quality.',
            'Voxelized cylinder geometry, no experimental metrology error bars, no clinical acceptance tolerance.',
            'Unit-displacement static solve is a linear response basis, not an admitted 1mm runtime deformation.'],
        'runtimeScope': 'Static specimen matrix comparison; finest2080 elements exceed the current2000-source-cell runtime patch cap',
        'results': results, 'physicalValidationPassed': False}


def run(include_specimen=False):
    path = ROOT / 'manifests/experiments/continuum-mechanics-v1.json'
    protocol = json.loads(path.read_text()); started = time.perf_counter()
    grids = [grid_control(h) for h in (1., .5, .25)]
    uniaxial = [uniaxial_control(h) for h in (1., .5, .25)]
    result = {'schema': 'resectionlab.continuum-mechanics-result.v1', 'protocolSha256': file_sha256(path),
        'sourceSha256': file_sha256(ROOT / 'src/resectionlab/surgical_continuum.py'), 'platform': platform.platform(),
        'grid': grids, 'tractionFreeUniaxial': uniaxial,
        'checks': {'affineEnergy': all(r['energyRelativeError'] < 1e-10 for r in grids),
            'affineReaction': all(r['resultantRelativeError'] < 1e-10 for r in grids),
            'tractionFreeUniaxial': all(r['forceRelativeError'] < 1e-10 and r['tractionFreeLateralMaxDisplacementErrorMm'] < 1e-10 for r in uniaxial)},
        'largeScaleRL': 'hold', 'patientMechanicsValidated': False}
    if include_specimen: result['specimen'] = specimen_comparison(protocol)
    result['wallSeconds'] = time.perf_counter() - started
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--include-exposed-specimen', action='store_true')
    args = parser.parse_args(); output = args.output.resolve()
    if not output.is_relative_to(ROOT / 'testcases') or output.exists(): parser.error('Choose a NEW output under testcases/')
    report = run(args.include_exposed_specimen); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'output': str(output), 'checks': report['checks'], 'wallSeconds': report['wallSeconds']}))
    if not all(report['checks'].values()): raise SystemExit(1)
