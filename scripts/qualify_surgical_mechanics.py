#!/usr/bin/env python3
"""Small independent numerical controls of the current app lattice; no tissue fit."""
from pathlib import Path
from types import SimpleNamespace
import argparse
import json
import platform
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from resectionlab import surgical_deformation_v2 as brain
from resectionlab.simulator_qualification import file_sha256


def single_node(h):
    p = brain.profile(); p['fixedStepSeconds'] = h
    labels = np.zeros((7, 7, 7), np.uint16); labels[3, 3, 3] = 1
    e = SimpleNamespace(scene=SimpleNamespace(shape=labels.shape, affine=np.eye(4), inverse=np.eye(4),
        labels=labels, properties={'tissuePhysicsV2': {'brain': p}}), by_label={1: {'material': 'soft_tissue'}},
        state={'removed': [], 'holds': {}, 'cutEdges': [], 'transectedEdges': []})
    c = int(np.ravel_multi_index((3, 3, 3), labels.shape))
    brain.begin_hold(e, 'left', [c], [3., 3., 3.]); brain.set_target(e, 'left', [3.1, 3., 3.])
    duration = .1
    with brain.diagnostic_scope() as diagnostics: brain.tick(e, duration)
    # Derived independently for c*u' + (k_foundation+k_hold)*u=k_hold*x.
    k = p['brainStiffnessNPerMm'] * .05 + p['holdStiffnessNPerMm']
    equilibrium = p['holdStiffnessNPerMm'] * .1 / k
    exact = equilibrium * (1 - np.exp(-duration * k / p['dragNsPerMm']))
    backward = equilibrium * (1 - (1 + h * k / p['dragNsPerMm']) ** -round(duration / h))
    observed = e.state['deformation'][str(c)][0]
    return {'stepSeconds': h, 'durationSeconds': duration, 'displacementMm': observed,
        'continuousReferenceMm': float(exact), 'continuousAbsoluteErrorMm': abs(observed - float(exact)),
        'independentBackwardEulerMm': backward, 'backwardEulerAbsoluteErrorMm': abs(observed - backward),
        'diagnostics': diagnostics}


def affine_energy(spacing):
    """Fixed 4 mm cube, prescribed 1% affine extension, same old profile.

    This is an energy sampling test, not a continuum simulation or a fit.
    Zero bulk penalty isolates the vector springs and disclosed foundation.
    """
    side = 4.; n = round(side / spacing) + 1
    labels = np.ones((n, n, n), np.uint16)
    a = np.diag([spacing, spacing, spacing, 1.])
    p = brain.profile(); p['bulkPenaltyRatio'] = 0.
    e = SimpleNamespace(scene=SimpleNamespace(shape=labels.shape, affine=a, inverse=np.linalg.inv(a), labels=labels),
        by_label={1: {'material': 'soft_tissue'}}, state={'holds': {}})
    ids = list(range(labels.size))
    bonds = sorted((c, other) for c in ids for other, _ in brain._neighbors(e, c) if c < other)
    _, _, _, _, matrix = brain._system(e, p, ids, [], bonds)
    rest = np.column_stack(np.unravel_index(ids, labels.shape)) * spacing
    displacement = np.zeros_like(rest); displacement[:, 0] = .01 * (rest[:, 0] - side / 2)
    total = float(.5 * displacement.ravel() @ (matrix @ displacement.ravel()))
    # Independent exact count: only x-directed bonds carry this displacement.
    spring = .5 * p['brainStiffnessNPerMm'] * ((n - 1) * n**2) * (.01 * spacing)**2
    foundation = .5 * .05 * p['brainStiffnessNPerMm'] * float(np.sum(displacement**2))
    return {'spacingMm': spacing, 'physicalSideMm': side, 'nodes': len(ids), 'strain': .01,
        'energyNmm': total, 'energyDensityNPerMm2': total / side**3,
        'independentBondEnergyNmm': spring, 'independentFoundationEnergyNmm': foundation,
        'independentEnergyAbsoluteErrorNmm': abs(total - spring - foundation),
        'profileScaling': 'unchanged per-node/per-bond coefficients, as in the current app'}


def run():
    temporal = [single_node(h) for h in (1 / 60, 1 / 120, 1 / 240)]
    spatial = [affine_energy(h) for h in (1., .5, .25)]
    return {'schema': 'resectionlab.lattice-numerical-qualification.v1',
        'sourceSha256': file_sha256(ROOT / 'src/resectionlab/surgical_deformation_v2.py'),
        'platform': platform.platform(), 'scope': 'A: generated numerical controls only; no measured tissue or patient validation',
        'timeStep': temporal, 'fixedPhysicalGridEnergy': spatial,
        'checks': {'backwardEulerMatchesIndependentFormula': all(r['backwardEulerAbsoluteErrorMm'] < 1e-10 for r in temporal),
            'forceResidual': all(r['diagnostics']['maxFreeResidualN'] < 1e-10 for r in temporal),
            'discreteEnergyBalance': all(r['diagnostics']['maxAbsDiscreteEnergyBalanceNmm'] < 1e-10 for r in temporal),
            'temporalErrorDecreases': temporal[2]['continuousAbsoluteErrorMm'] < temporal[1]['continuousAbsoluteErrorMm'] < temporal[0]['continuousAbsoluteErrorMm'],
            'independentEnergyMatches': all(r['independentEnergyAbsoluteErrorNmm'] < 1e-10 for r in spatial)},
        'gridEnergyDensityRatiosTo1mm': [r['energyDensityNPerMm2'] / spatial[0]['energyDensityNPerMm2'] for r in spatial],
        'gridIndependentMaterialMeaning': 'not established; frozen coefficients change physical energy under refinement',
        'gridTestScope': 'Static stiffness/energy kernel audit, not patient timestep execution. Finest audit has 4913 nodes, beyond the 2000-node runtime patch limit.',
        'measuredResponseAgreement': 'unvalidated', 'meaningfulPhysiologicalRLReady': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); output = args.output.resolve()
    if not output.is_relative_to(ROOT / 'testcases') or output.exists():
        parser.error('Choose a NEW JSON output under testcases/')
    report = run(); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'output': str(output), 'checks': report['checks'], 'gridEnergyRatios': report['gridEnergyDensityRatiosTo1mm']}))
    if not all(report['checks'].values()): raise SystemExit(1)
