#!/usr/bin/env python3
"""Three-resolution work study of the actual canonical native action path.

Only the diagnostic worker varies maneuver.DT. Production phases, durations,
endpoints, source geometry, materials and supports are unchanged. This study
never grants physical, patient-specific, planning or RL admission.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]


def relative(a, b):
    return abs(a-b)/max(abs(b), 1e-20)


def continuous_uniform_maneuver():
    """Closed-form contact ramps/hold and unloaded recovery, independently.

    F=k0*d+k1*e, e_dot=v-e/tau in contact. Release occurs at F=0;
    separated d decays with tau*(k0+k1)/k0. Integrate F*v and k1*e^2/tau
    analytically, splitting withdrawal at its zero-force event.
    """
    import math
    from scipy.optimize import brentq
    k0, k1, tau = .008, .004, .5
    def segment(d, e, v, duration):
        a, b = tau*v, e-tau*v
        decay = math.exp(-duration/tau)
        integral_e = a*duration+b*tau*(1-decay)
        integral_e2 = a*a*duration+2*a*b*tau*(1-decay)+b*b*tau/2*(1-decay*decay)
        work = k0*v*(d*duration+v*duration*duration/2)+k1*v*integral_e
        return d+v*duration, a+b*decay, work, k1/tau*integral_e2
    d, e, indent_work, indent_viscous = segment(0., 0., .1, .2)
    indent_force = k0*d+k1*e
    d, e, _, hold_viscous = segment(d, e, 0., .2)
    hold_force = k0*d+k1*e
    def force(t):
        x, y, _, _ = segment(d, e, -.6, t)
        return k0*x+k1*y
    release = brentq(force, 0., .2, xtol=1e-15)
    d, e, withdrawal_work, withdrawal_viscous = segment(d, e, -.6, release)
    recovery_tau = tau*(k0+k1)/k0
    recovery_time = .2-release
    recovery_viscous = k1/tau*e*e*recovery_tau/2*(1-math.exp(-2*recovery_time/recovery_tau))
    d *= math.exp(-recovery_time/recovery_tau)
    e *= math.exp(-recovery_time/recovery_tau)
    stored = .5*k0*d*d+.5*k1*e*e
    work = indent_work+withdrawal_work
    viscous = indent_viscous+hold_viscous+withdrawal_viscous+recovery_viscous
    return {'actuatorWorkNmm': work, 'indentReactionN': indent_force, 'holdReactionN': hold_force,
        'releaseSecondsAfterWithdrawalStart': release, 'finalDisplacementMm': d,
        'storedEnergyNmm': stored, 'viscousDissipationNmm': viscous,
        'energyBalanceResidualNmm': work-stored-viscous}


def scalar_history(captured):
    """Independent uniaxial BE equilibrium, memory and endpoint-work integral.

    The accepted time partition is evidence, not a copied state/force oracle.
    Units: mu in Pa, d/z in mm, area16mm2/height4mm; work in Nmm.
    """
    z = 0.; old_offset = -.1; work = 0.; rows = []
    for row in captured:
        count = row['receipt'].get('internalSubsteps', 1)
        h = row['dtS']/count; ratio = .5/(.5+h)
        for i in range(count):
            offset = old_offset + (row['offsetMm']-old_offset)*(i+1)/count
            free = ratio*500*z/(1000+ratio*500)
            d = max(offset, free)
            z = ratio*z+(1-ratio)*d
            force = 8e-6*(1000*d+500*(d-z)) if offset > free else 0.
            work += force*(row['offsetMm']-old_offset)/count
        old_offset = row['offsetMm']
        rows.append({'normalReactionN': force, 'maxDisplacementMm': d, 'workNmm': work})
    return rows


def run(integration, dt):
    import numpy as np
    from resectionlab import surgical_planar_contact as contact
    from resectionlab import surgical_planar_maneuver as maneuver
    from resectionlab.surgical_engine import SurgicalEngine
    from resectionlab.surgical_scene import SurgicalScene
    from resectionlab.core import semantic_digest

    class ObservedEngine(SurgicalEngine):
        def __init__(self, *args):
            super().__init__(*args); self.observed = []
        def clone(self):
            candidate = super().clone(); candidate.observed = copy.deepcopy(self.observed)
            return candidate
        def _substep(self, *args):
            events = super()._substep(*args)
            raw = self.state[maneuver.PROPERTY]
            self.observed.append({'phase': raw['activePhase'], 'dtS': args[-1],
                'timeS': self.state['elapsedSeconds'], 'offsetMm': raw['kernel']['offsetMm'],
                'receipt': copy.deepcopy(raw['receipt'])})
            return events

    original_dt = maneuver.DT
    started = time.perf_counter()
    try:
        # Isolated diagnostic override; scene validation binds this exact schedule.
        # Default .025 s remains the only production maneuver declaration.
        maneuver.DT = dt
        scene = maneuver.engineering_scene()
        if integration == contact.LEGACY_INTEGRATION:
            props = copy.deepcopy(scene.properties)
            props[maneuver.PROPERTY]['schedule'].pop('integration')
            scene = SurgicalScene(scene.labels, scene.affine, scene.regions, scene.source, props)
        engine = ObservedEngine(scene); phase_rows = []
        for phase, duration in zip(maneuver.PHASES, maneuver.DURATIONS):
            result = engine.step({'slot': 'left', 'operation': 'bladeManeuver',
                'maneuverPhase': phase, 'duration': duration})
            if result['status'] != 'accepted':
                return {'integrationMethod': integration, 'dtS': dt, 'passed': False,
                    'failure': result, 'wallS': time.perf_counter()-started}
            raw = engine.state[maneuver.PROPERTY]
            phase_rows.append({'phase': phase, 'actuatorWorkNmm': raw['actuatorWorkNmm'],
                'receipt': copy.deepcopy(raw['receipt']), 'energyBalanceResidualNmm': raw['cumulativeEnergyBalanceNmm']})
        captured = engine.observed; scalar = scalar_history(captured)
        raw = engine.state[maneuver.PROPERTY]
        independent = {
            'maxReactionDifferenceN': max(abs(r['receipt']['normalReactionN']-e['normalReactionN']) for r, e in zip(captured, scalar)),
            'maxDisplacementDifferenceMm': max(abs(r['receipt']['maxDisplacementMm']-e['maxDisplacementMm']) for r, e in zip(captured, scalar)),
            'workDifferenceNmm': abs(raw['actuatorWorkNmm']-scalar[-1]['workNmm'])}
        # Compare the complete ledger independently of the backend's balance field.
        fields = ('actuatorWorkNmm', 'viscousDissipationNmm', 'numericalDissipationNmm', 'contactActivationDissipationNmm')
        ledger = {key: sum(row['receipt'][key] for row in captured) for key in fields}
        ledger['finalStoredEnergyNmm'] = captured[-1]['receipt']['storedEnergyNmm']
        ledger['independentBalanceResidualNmm'] = ledger['actuatorWorkNmm']-ledger['finalStoredEnergyNmm']-sum(ledger[k] for k in fields[1:])
        passed = (independent['maxReactionDifferenceN'] <= 1e-9 and independent['maxDisplacementDifferenceMm'] <= 1e-7
            and independent['workDifferenceNmm'] <= 1e-10 and abs(ledger['independentBalanceResidualNmm']) <= 1e-10)
        return {'integrationMethod': integration, 'dtS': dt, 'passed': passed,
            'geometryHash': maneuver.geometry_hash(scene),
            'materialHash': semantic_digest(scene.properties['brainContinuumV1']),
            'declaration': scene.properties[maneuver.PROPERTY], 'phases': phase_rows, 'steps': captured,
            'ledger': ledger, 'independentScalarCheck': independent,
            'peakReactionN': max(row['receipt']['normalReactionN'] for row in captured),
            'peakDisplacementMm': max(row['receipt']['maxDisplacementMm'] for row in captured),
            'maxEnergyBalanceResidualNmm': max(abs(row['receipt']['energyBalanceResidualNmm']) for row in captured),
            'internalSubsteps': sum(row['receipt'].get('internalSubsteps', 1) for row in captured),
            'integrationEvaluations': sum(row['receipt'].get('integrationEvaluations', 1) for row in captured),
            'workErrorEstimateSumNmm': sum(row['receipt'].get('workErrorEstimateNmm', 0.) for row in captured),
            'wallS': time.perf_counter()-started}
    finally:
        maneuver.DT = original_dt


def worker(output):
    from resectionlab import surgical_planar_contact as contact
    started = time.perf_counter()
    report = {'schema': 'resectionlab.planar-blade-temporal-study.v1',
        'role': 'GENERATED_NUMERICAL_VERIFICATION_ONLY', 'productionDtS': .025,
        'note': 'Finer dt overrides are diagnostic only; same four native actions, physical path, duration and source assumptions.',
        'clinicalValidation': False, 'patientPredictionValidation': False, 'physiologicalRL': 'hold',
        'platform': platform.platform(), 'python': sys.version, 'runs': [], 'comparisons': [], 'failures': []}
    continuous = continuous_uniform_maneuver()
    report['independentContinuousUniformSolution'] = continuous
    files = [Path(__file__), ROOT/'src/resectionlab/surgical_planar_contact.py', ROOT/'src/resectionlab/surgical_planar_maneuver.py']
    archive = output.parent/'sources'; archive.mkdir(exist_ok=True)
    report['sourceSha256'] = {}
    for path in files:
        raw = path.read_bytes(); digest = hashlib.sha256(raw).hexdigest()
        report['sourceSha256'][str(path.relative_to(ROOT))] = digest
        destination = archive/(digest+'-'+path.name)
        if destination.exists():
            assert destination.read_bytes() == raw
        else:
            destination.write_bytes(raw)
    for method in (contact.LEGACY_INTEGRATION, contact.ADAPTIVE_INTEGRATION):
        group = [run(method, dt) for dt in (.025, .0125, .00625)]
        report['runs'].extend(group)
        for row in group:
            if row['passed']:
                value = row['ledger']['actuatorWorkNmm']
                row['continuousWorkAbsoluteErrorNmm'] = abs(value-continuous['actuatorWorkNmm'])
                row['continuousWorkRelativeError'] = relative(value, continuous['actuatorWorkNmm'])
            if not row['passed']: report['failures'].append({'method': method, 'dtS': row['dtS'], 'check': 'acceptedCompleteNumericalHistory'})
            print(json.dumps({k: row.get(k) for k in ('integrationMethod', 'dtS', 'passed', 'ledger', 'wallS')}), flush=True)
        if not all(row['passed'] for row in group): continue
        assert len({row['geometryHash'] for row in group}) == len({row['materialHash'] for row in group}) == 1
        for a, b in zip(group, group[1:]):
            wa, wb = [row['ledger']['actuatorWorkNmm'] for row in (a, b)]
            hold_a, hold_b = [row['phases'][2]['receipt'] for row in (a, b)]
            comparison = {'method': method, 'coarseDtS': a['dtS'], 'fineDtS': b['dtS'],
                'absoluteWorkDifferenceNmm': abs(wa-wb), 'relativeWorkDifference': relative(wa, wb),
                'absoluteHoldReactionDifferenceN': abs(hold_a['normalReactionN']-hold_b['normalReactionN']),
                'relativeHoldReactionDifference': relative(hold_a['normalReactionN'], hold_b['normalReactionN']),
                'absolutePeakDisplacementDifferenceMm': abs(a['peakDisplacementMm']-b['peakDisplacementMm']),
                'workLimit': .05, 'workSensitivityPassed': relative(wa, wb) <= .05}
            report['comparisons'].append(comparison)
            if method == contact.ADAPTIVE_INTEGRATION and not comparison['workSensitivityPassed']:
                report['failures'].append(comparison)
    report['passed'] = not report['failures']; report['wallS'] = time.perf_counter()-started
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return 0 if report['passed'] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(); output = args.output.resolve()
    if not output.is_relative_to(ROOT/'testcases') or output.exists():
        parser.error('Choose a new output filename beneath testcases/')
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.worker: return worker(output)
    env = os.environ.copy()
    for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        env[key] = '1'
    return subprocess.run([sys.executable, __file__, '--worker', '--output', str(output)], cwd=ROOT, env=env).returncode


if __name__ == '__main__':
    raise SystemExit(main())
