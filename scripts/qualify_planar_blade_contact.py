#!/usr/bin/env python3
"""Fresh-process numerical qualification under the frozen planar contact protocol.

All generated evidence stays beneath testcases/. No live-engine admission occurs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from planar_blade_contact_fixtures import MANIFEST_SHA256, frozen_protocol, generated_fixture, prescribed_schedule


def json_value(value):
    import numpy as np
    if isinstance(value, np.ndarray):
        converted = value.tolist()
        _preserve_nonfinite(converted)
        return converted
    if isinstance(value, np.generic):
        converted = value.item()
        return str(converted) if isinstance(converted, float) and not math.isfinite(converted) else converted
    raise TypeError(type(value).__name__)


def _preserve_nonfinite(value):
    """Record invalid scalars explicitly, without duplicating the evidence tree."""
    if isinstance(value, dict):
        items = value.items()
    elif isinstance(value, list):
        items = enumerate(value)
    else:
        return
    for key, item in items:
        if isinstance(item, float) and not math.isfinite(item):
            value[key] = str(item)
        else:
            _preserve_nonfinite(item)


def write_report(path, report, *, indent=None):
    # Strict checker decisions happen first. Stream the retained arrays and
    # receipts without constructing another full Python/JSON copy in memory.
    _preserve_nonfinite(report)
    with path.open('w') as handle:
        json.dump(report, handle, default=json_value, indent=indent, allow_nan=False)
        handle.write('\n')


def finite_number(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def receipt_failures(receipt, tolerances):
    """Keep every missing, nonfinite or excessive strict receipt check."""
    failures = []
    mapping = {'maxPenetrationMm': 'penetrationMm', 'maxTensileForceN': 'tensileMultiplierN',
        'maxEquilibriumResidualN': 'equilibriumResidualN', 'maxComplementarityNmm': 'complementarityNmm',
        'energyBalanceResidualNmm': 'energyBalanceNmm', 'maxTangentialForceN': 'tangentialContactForceN',
        'maxElementGradientNorm': 'maximumElementGradientNorm'}
    for field, limit in mapping.items():
        value = receipt.get(field)
        if not finite_number(value) or abs(value) > tolerances[limit]:
            failures.append({'check': field, 'observed': value, 'maximum': tolerances[limit]})
    ledger = ['actuatorWorkNmm', 'storedEnergyNmm', 'previousStoredEnergyNmm', 'viscousDissipationNmm',
              'numericalDissipationNmm', 'contactActivationDissipationNmm']
    if all(finite_number(receipt.get(k)) for k in ledger):
        balance = receipt[ledger[0]] - receipt[ledger[1]] + receipt[ledger[2]] - sum(receipt[k] for k in ledger[3:])
        if abs(balance) > tolerances['energyBalanceNmm']:
            failures.append({'check': 'independentReceiptLedgerNmm', 'observed': balance,
                             'maximum': tolerances['energyBalanceNmm']})
        for k in ledger[1:]:
            if receipt[k] < -tolerances['energyBalanceNmm']:
                failures.append({'check': 'nonnegative-' + k, 'observed': receipt[k], 'minimum': 0.})
    else:
        failures.append({'check': 'completeFiniteEnergyLedger', 'observed': {k: receipt.get(k) for k in ledger}})
    for k in ('normalReactionN', 'contactAreaMm2', 'footprintAreaMm2', 'minimumGapMm'):
        if not finite_number(receipt.get(k)):
            failures.append({'check': 'finite-' + k, 'observed': receipt.get(k)})
    if finite_number(receipt.get('minimumGapMm')) and receipt['minimumGapMm'] < -tolerances['penetrationMm']:
        failures.append({'check': 'minimumGapMm', 'observed': receipt['minimumGapMm'], 'minimum': -tolerances['penetrationMm']})
    if all(finite_number(receipt.get(k)) for k in ('contactAreaMm2', 'footprintAreaMm2')):
        if not -1e-12 <= receipt['contactAreaMm2'] <= receipt['footprintAreaMm2'] + 1e-10:
            failures.append({'check': 'boundedContactArea', 'observed': receipt['contactAreaMm2']})
    return failures


def scenario_specs():
    f = frozen_protocol()['fixtures']
    specs = []
    for n in f['meshCellsPerAxis']:
        for partial in (False, True):
            for maxwell in (False, True):
                specs.append(dict(n=n, partial=partial, maxwell=maxwell, dt_s=.1, order=2, rotated=False))
    for partial in (False, True):
        for dt in (.05, .025):
            specs.append(dict(n=4, partial=partial, maxwell=True, dt_s=dt, order=2, rotated=False))
    for maxwell in (False, True):
        for order in (1, 3, 4):
            specs.append(dict(n=4, partial=True, maxwell=maxwell, dt_s=.1, order=order, rotated=False))
    for partial in (False, True):
        specs.append(dict(n=4, partial=partial, maxwell=True, dt_s=.1, order=2, rotated=True))
    return specs


def uniform_analytical(schedule, maxwell):
    """Scalar uniaxial BE recurrence, independently derived for nu=0 coupon."""
    f = frozen_protocol()['fixtures']
    mu, branch, tau = f['equilibriumShearPa'], f['maxwellBranchShearPa'], f['maxwellRelaxationSeconds']
    z = 0.
    rows = []
    for step in schedule:
        ratio = tau / (tau + step['dtS'])
        # Compression positive; a released surface recovers against zero load.
        free = ratio * branch * z / (mu + ratio * branch) if maxwell else 0.
        d = max(step['offsetMm'], free)
        z = ratio * z + (1 - ratio) * d if maxwell else 0.
        reaction = 2e-6 * 16 / 4 * (mu * d + (branch * (d - z) if maxwell else 0.))
        if free >= step['offsetMm']: reaction = 0.
        rows.append({'indentationMm': d, 'memoryIndentationMm': z, 'normalReactionN': reaction})
    return rows


def _compare(run, check, observed, maximum, **extra):
    if not finite_number(observed) or observed > maximum:
        run['failures'].append({'check': check, 'observed': observed, 'maximum': maximum, **extra})


def _run_scenario(spec, protocol):
    import numpy as np
    from resectionlab import surgical_planar_contact as contact
    import planar_contact_dense_oracle as oracle
    fixture = generated_fixture(spec['n'], partial=spec['partial'], maxwell=spec['maxwell'], rotated=spec['rotated'])
    run = {'id': fixture.name + f"-q{spec['order']}-dt{spec['dt_s']}", 'spec': spec, 'steps': [],
           'failures': [], 'diagnosticOnly': spec['order'] == 1, 'status': 'running'}
    started = time.perf_counter()
    try:
        kernel = contact.build(fixture.scene, fixture.cells, contact_faces=fixture.contact_faces,
            support_faces=fixture.support_faces, blade=fixture.blade, quadrature_order=spec['order'])
        state = kernel.initial_state()
        source_identity = fixture.scene.identity
        source_labels_sha256 = hashlib.sha256(fixture.scene.labels.tobytes()).hexdigest()
        ownership_before = kernel.mesh.node_owners
        nodes_before = kernel.mesh.nodes.copy()
        run['geometry'] = {'affine': fixture.scene.affine, 'sourceCells': fixture.cells,
            'contactFaces': fixture.contact_faces, 'supportFaces': fixture.support_faces,
            'blade': vars(fixture.blade), 'nodesMm': kernel.mesh.nodes,
            'elementNodes': kernel.mesh.element_nodes, 'freeDofs': kernel.free,
            'contactWeightsMm2': kernel.weights_mm2, 'nodeOwners': kernel.mesh.node_owners,
            'contactSourceIds': kernel.source_ids, 'sourceIdentity': source_identity,
            'sourceLabelsSha256': source_labels_sha256}
        schedule = prescribed_schedule(spec['dt_s'], partial=spec['partial'])
        run['resources'] = {'sourceCells': len(fixture.cells), 'dofs': int(kernel.mesh.nodes.size),
                            'contactPoints': len(kernel.weights_mm2), 'scheduleSteps': len(schedule)}
        for name, maximum in [('sourceCells', 'maxSourceCells'), ('dofs', 'maxDofs'),
                              ('contactPoints', 'maxContactPoints'), ('scheduleSteps', 'maxScheduleSteps')]:
            _compare(run, name, run['resources'][name], protocol['resources'][maximum])
        analytical = uniform_analytical(schedule, spec['maxwell']) if not spec['partial'] else None
        oracle_state = None
        if spec['partial'] and spec['n'] == 2 and spec['order'] == 2:
            element_index = {c: i for i, c in enumerate(kernel.mesh.cells)}
            rows = oracle.rectangular_surface_rows(kernel.mesh.nodes, kernel.mesh.element_nodes,
                [(element_index[c], a, s) for c, a, s in fixture.contact_faces],
                fixture.blade.origin_mm, fixture.blade.axis_u, fixture.blade.axis_v,
                fixture.blade.inward_normal, fixture.blade.size_mm, order=spec['order'], include_guards=True)
        for i, step in enumerate(schedule):
            clock = time.perf_counter()
            result = kernel.step(state, offset_mm=step['offsetMm'], dt_s=step['dtS'])
            wall = time.perf_counter() - clock
            state = result['state']; receipt = result['receipt']
            recorded = {**step, 'wallS': wall, 'receipt': receipt,
                'displacementMm': np.asarray(state.displacement_mm), 'memoryMm': np.asarray(state.memory_mm),
                'normalForcesN': result['normal_forces_n'], 'gapsMm': result['gaps_mm'],
                'contactNodalForceN': result['contact_nodal_force_n']}
            run['steps'].append(recorded)
            for failure in receipt_failures(receipt, protocol['tolerances']):
                run['failures'].append({'step': i + 1, **failure})
            _compare(run, 'stepWallS', wall, protocol['resources']['maxStepWallSeconds'], step=i + 1)
            for field, cap in [('optimizerIterations', 'maxDualIterations'), ('optimizerFunctionCalls', 'maxDualFunctionCalls')]:
                if field in receipt: _compare(run, field, receipt[field], protocol['resources'][cap], step=i + 1)
            u = np.asarray(state.displacement_mm).reshape((-1, 3))
            if analytical:
                expected = analytical[i]
                local_nodes = (kernel.mesh.nodes - fixture.translation_mm) @ fixture.rotation
                expected_u = np.zeros_like(u)
                expected_u[:, 2] = -expected['indentationMm'] * local_nodes[:, 2] / 4
                expected_u = expected_u @ fixture.rotation.T
                scale = max(expected['indentationMm'], 1e-12)
                error = float(np.max(np.abs(u - expected_u))) / scale
                force_error = abs(receipt['normalReactionN'] - expected['normalReactionN']) / max(expected['normalReactionN'], 2e-6 * 16 / 4 * 1000 * .04)
                recorded['analytical'] = {**expected, 'relativeDisplacementError': error, 'relativeReactionError': force_error}
                _compare(run, 'analyticalUniformDisplacementRelative', error, protocol['tolerances']['analyticalUniformRelative'], step=i + 1)
                _compare(run, 'analyticalUniformReactionRelative', force_error, protocol['tolerances']['analyticalUniformRelative'], step=i + 1)
            if spec['partial'] and spec['n'] == 2 and spec['order'] == 2:
                oracle_state = oracle.solve(kernel.mesh.nodes, kernel.mesh.element_nodes, 1000., .3,
                    [500.] if spec['maxwell'] else [], [.5] if spec['maxwell'] else [],
                    kernel.free, rows['contact_rows'], rows['targets_at_zero_mm'] + step['offsetMm'], step['dtS'],
                    None if oracle_state is None else oracle_state['u_mm'],
                    None if oracle_state is None else oracle_state['z_mm'])
                displacement_error = float(np.max(np.abs(u.ravel() - oracle_state['u_mm'])))
                oracle_force = oracle_state['contact_nodal_force_n'].reshape((-1, 3)).sum(axis=0)
                force_error = float(np.max(np.abs(np.asarray(result['contact_nodal_force_n']).reshape((-1, 3)).sum(axis=0) - oracle_force)))
                recorded['denseOracle'] = {'displacementMm': oracle_state['u_mm'], 'memoryMm': oracle_state['z_mm'],
                    'aggregateTissueForceN': oracle_force, 'diagnostics': oracle_state['diagnostics'],
                    'maximumDisplacementDifferenceMm': displacement_error, 'maximumForceDifferenceN': force_error}
                _compare(run, 'denseOracleDisplacementMm', displacement_error, protocol['tolerances']['denseOracleDisplacementMm'], step=i + 1)
                _compare(run, 'denseOracleReactionN', force_error, protocol['tolerances']['denseOracleReactionN'], step=i + 1)
        fixture.scene.require_cache_source()
        owner_cells = [cell for owners in kernel.mesh.node_owners for cell, _ in owners]
        ownership_ok = (kernel.mesh.node_owners == ownership_before and
            np.array_equal(kernel.mesh.nodes, nodes_before) and
            set(owner_cells) == set(fixture.cells) and
            all(owner_cells.count(cell) == 8 for cell in fixture.cells) and
            set(map(int, kernel.source_ids)) <= {face[0] for face in fixture.contact_faces} and
            fixture.scene.identity == source_identity and
            hashlib.sha256(fixture.scene.labels.tobytes()).hexdigest() == source_labels_sha256)
        run['sourceOwnershipPreserved'] = ownership_ok
        if not ownership_ok: run['failures'].append({'check': 'sourceIdentityAndNodeOwnerPreservation'})
        last = run['steps'][-1]
        _compare(run, 'withdrawalZeroAttractionN', abs(last['receipt']['normalReactionN']), protocol['tolerances']['tensileMultiplierN'])
        _compare(run, 'withdrawalContactAreaMm2', abs(last['receipt']['contactAreaMm2']), 1e-10)
        run['status'] = 'pass' if not run['failures'] else 'fail'
    except Exception as exc:
        run['status'] = 'refused'
        run['refusal'] = {'type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc()}
        if not run['diagnosticOnly']: run['failures'].append({'check': 'admittedFrozenScenario', 'exception': str(exc)})
    run['wallS'] = time.perf_counter() - started
    return run


def cross_run_checks(runs, protocol):
    import numpy as np
    findings = []
    lookup = {(r['spec']['n'], r['spec']['partial'], r['spec']['maxwell'], r['spec']['dt_s'], r['spec']['order'], r['spec']['rotated']): r for r in runs}
    def compare(label, observed, maximum):
        findings.append({'check': label, 'observed': observed, 'maximum': maximum, 'pass': finite_number(observed) and observed <= maximum})
    def successful(key):
        r = lookup.get(key)
        return r if r and r['status'] == 'pass' else None
    def relative(a, b): return abs(a - b) / max(abs(b), 1e-30)
    def final_hold(r): return [s for s in r['steps'] if s['phase'] == 'hold'][-1]
    for partial in (False, True):
        for maxwell in (False, True):
            a, b = successful((4, partial, maxwell, .1, 2, False)), successful((8, partial, maxwell, .1, 2, False))
            if a and b:
                for field, tol in [('normalReactionN', 'meshFineReactionRelativeChange'), ('storedEnergyNmm', 'meshFineEnergyRelativeChange')]:
                    compare(f'meshFine-{partial}-{maxwell}-{field}', relative(final_hold(a)['receipt'][field], final_hold(b)['receipt'][field]), protocol['tolerances'][tol])
        a, b = successful((4, partial, True, .05, 2, False)), successful((4, partial, True, .025, 2, False))
        if a and b:
            compare(f'timestepFineHold-{partial}', relative(final_hold(a)['receipt']['normalReactionN'], final_hold(b)['receipt']['normalReactionN']), protocol['tolerances']['timeStepFineHoldReactionRelativeChange'])
            compare(f'timestepFineWork-{partial}', relative(sum(s['receipt']['actuatorWorkNmm'] for s in a['steps']), sum(s['receipt']['actuatorWorkNmm'] for s in b['steps'])), protocol['tolerances']['timeStepFineWorkRelativeChange'])
        a, b = successful((4, partial, True, .1, 2, False)), successful((4, partial, True, .1, 2, True))
        if a and b:
            from planar_blade_contact_fixtures import rigid_rotation
            rotation = rigid_rotation()
            compare(f'rotationDisplacement-{partial}', max(float(np.max(np.abs(np.asarray(x['displacementMm']).reshape((-1, 3)) @ rotation.T - np.asarray(y['displacementMm']).reshape((-1, 3))))) for x, y in zip(a['steps'], b['steps'])), protocol['tolerances']['rotationDisplacementMm'])
            compare(f'rotationReaction-{partial}', max(float(np.max(np.abs(np.asarray(x['receipt']['reactionOnBladeN']) @ rotation.T - np.asarray(y['receipt']['reactionOnBladeN'])))) for x, y in zip(a['steps'], b['steps'])), protocol['tolerances']['rotationReactionN'])
            compare(f'rotationEnergy-{partial}', max(abs(x['receipt'][field] - y['receipt'][field]) for x, y in zip(a['steps'], b['steps']) for field in ('storedEnergyNmm', 'actuatorWorkNmm', 'viscousDissipationNmm', 'numericalDissipationNmm', 'contactActivationDissipationNmm')), protocol['tolerances']['rotationEnergyNmm'])
    for maxwell in (False, True):
        a, b = successful((4, True, maxwell, .1, 2, False)), successful((4, True, maxwell, .1, 4, False))
        if a and b:
            for field, tol in [('normalReactionN', 'surfaceOrder2To4ReactionRelativeChange'), ('contactAreaMm2', 'surfaceOrder2To4AreaRelativeChange')]:
                compare(f'surface2to4-{maxwell}-{field}', max(relative(x['receipt'][field], y['receipt'][field]) for x, y in zip(a['steps'], b['steps']) if y['receipt'][field] > 1e-12), protocol['tolerances'][tol])
    # Continuous Maxwell stress for prescribed linear ramp followed by a hold.
    f = protocol['fixtures']; tau = f['maxwellRelaxationSeconds']; ramp = f['schedule']['rampSeconds']; hold = f['schedule']['holdSeconds']
    continuous = 2e-6 * 16 / 4 * f['uniformIndentationMm'] * (f['equilibriumShearPa'] + f['maxwellBranchShearPa'] * tau / ramp * (1 - math.exp(-ramp / tau)) * math.exp(-hold / tau))
    errors = []
    for dt in f['timeStepsSeconds']:
        r = successful((4, False, True, dt, 2, False))
        if r: errors.append({'dtS': dt, 'absoluteErrorN': abs(final_hold(r)['receipt']['normalReactionN'] - continuous), 'continuousReactionN': continuous})
    if len(errors) == 3:
        compare('continuousHoldErrorRefines', max(errors[i + 1]['absoluteErrorN'] - errors[i]['absoluteErrorN'] for i in range(2)), 0.)
    return findings, errors


def negative_controls():
    from resectionlab import surgical_planar_contact as contact
    fixture = generated_fixture(2, partial=True)
    cases = [('unsupportedBody', dict(support_faces=[])),
             ('interiorContactFace', dict(contact_faces=[(fixture.cells[0], 2, 1)])),
             ('nonintegerQuadrature', dict(quadrature_order=2.5)),
             ('duplicateContactFace', dict(contact_faces=[fixture.contact_faces[0]] * 2))]
    results = []
    for name, change in cases:
        args = dict(contact_faces=fixture.contact_faces, support_faces=fixture.support_faces,
                    blade=fixture.blade, quadrature_order=2)
        args.update(change)
        try:
            contact.build(fixture.scene, fixture.cells, **args)
            results.append({'check': name, 'pass': False, 'failure': 'configuration unexpectedly admitted'})
        except ValueError as exc:
            results.append({'check': name, 'pass': True, 'refusal': str(exc)})
        except Exception as exc:
            results.append({'check': name, 'pass': False, 'failure': f'{type(exc).__name__}: {exc}'})
    try:
        contact.build(fixture.scene, (fixture.cells[0], fixture.cells[-1]),
            contact_faces=[(fixture.cells[-1], 2, 1)], support_faces=[(fixture.cells[0], 2, -1)],
            blade=fixture.blade, quadrature_order=2)
        results.append({'check': 'unsupportedTouchingDisconnectedBody', 'pass': False, 'failure': 'configuration unexpectedly admitted'})
    except ValueError as exc:
        results.append({'check': 'unsupportedTouchingDisconnectedBody', 'pass': 'UNSUPPORTED_RIGID_COMPONENT' in str(exc), 'refusal': str(exc)})
    except Exception as exc:
        results.append({'check': 'unsupportedTouchingDisconnectedBody', 'pass': False, 'failure': f'{type(exc).__name__}: {exc}'})
    from dataclasses import replace
    kernel = contact.build(fixture.scene, fixture.cells, contact_faces=fixture.contact_faces,
        support_faces=fixture.support_faces, blade=fixture.blade, quadrature_order=2)
    for name in ('stateModelMismatch', 'sourceMetadataMismatch'):
        initial = kernel.initial_state()
        if name == 'stateModelMismatch': initial = replace(initial, model_id='wrong-source-model')
        else: fixture.scene.source['unversioned-control-mutation'] = True
        try:
            kernel.step(initial, offset_mm=.002, dt_s=.1)
            results.append({'check': name, 'pass': False, 'failure': 'mismatched input unexpectedly admitted'})
        except ValueError as exc:
            expected = 'STATE_OR_SCHEDULE' if name == 'stateModelMismatch' else 'Source scene changed'
            results.append({'check': name, 'pass': expected in str(exc), 'refusal': str(exc)})
        except Exception as exc:
            results.append({'check': name, 'pass': False, 'failure': f'{type(exc).__name__}: {exc}'})
    return results


def run_worker(output):
    protocol = frozen_protocol(); start = time.perf_counter()
    source_files = sorted((ROOT / 'src/resectionlab').glob('*.py')) + [
        Path(__file__).resolve(), ROOT / 'scripts/planar_blade_contact_fixtures.py',
        ROOT / 'scripts/planar_contact_dense_oracle.py']
    source_sha256 = {}
    archive = output.parent / 'sources'
    archive.mkdir(exist_ok=True)
    for path in source_files:
        raw = path.read_bytes(); digest = hashlib.sha256(raw).hexdigest()
        source_sha256[str(path.relative_to(ROOT))] = digest
        destination = archive / (digest + '-' + path.name)
        if destination.exists():
            if destination.read_bytes() != raw: raise RuntimeError('source archive hash collision')
        else: destination.write_bytes(raw)
    report = {'schema': 'resectionlab.planar-blade-contact-qualification.v1',
        'role': protocol['role'], 'manifestSha256': MANIFEST_SHA256, 'protocolPredecessor': protocol['predecessor'], 'clinicalValidation': False,
        'patientPredictionValidation': False, 'physiologicalRL': 'hold',
        'identity': {'python': sys.version, 'executable': sys.executable, 'platform': platform.platform(),
            'machine': platform.machine(), 'processor': platform.processor(), 'logicalCpuCount': os.cpu_count(),
            'numpy': __import__('numpy').__version__, 'scipy': __import__('scipy').__version__,
            'gitStatus': subprocess.run(['git', 'status', '--porcelain'], cwd=ROOT, capture_output=True, text=True).stdout.splitlines(),
            'hardwareModel': subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], capture_output=True, text=True).stdout.strip() if sys.platform == 'darwin' else platform.processor(),
            'gitHead': subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip()},
        'sourceSha256': source_sha256, 'sourceArchiveRelativeDirectory': 'sources',
        'blasEnvironment': {k: os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')},
        'runs': [], 'allFailures': []}
    for spec in scenario_specs():
        run = _run_scenario(spec, protocol); report['runs'].append(run)
        if not run['diagnosticOnly']:
            report['allFailures'].extend({'run': run['id'], **f} for f in run['failures'])
        print(f"{run['id']}: {run['status']}, {len(run['steps'])} steps, {run['wallS']:.2f}s", flush=True)
        write_report(output.with_suffix('.partial.json'), report)
    report['crossRunChecks'], report['continuousHoldSensitivity'] = cross_run_checks(report['runs'], protocol)
    report['negativeControls'] = negative_controls()
    report['allFailures'].extend(f for f in report['crossRunChecks'] + report['negativeControls'] if not f['pass'])
    report['workerWallS'] = time.perf_counter() - start
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report['freshProcessPeakMiB'] = peak / (1024**2 if sys.platform == 'darwin' else 1024)
    for check, observed, maximum in [('qualificationWallS', report['workerWallS'], protocol['resources']['maxQualificationWallSeconds']), ('freshProcessPeakMiB', report['freshProcessPeakMiB'], protocol['resources']['maxFreshProcessPeakMiB'])]:
        if observed > maximum: report['allFailures'].append({'check': check, 'observed': observed, 'maximum': maximum})
    report['pass'] = not report['allFailures']
    write_report(output, report, indent=2)
    output.with_suffix('.partial.json').unlink(missing_ok=True)
    return 0 if report['pass'] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'testcases/outputs/planar-blade-contact-20261030/qualification.json')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(); output = args.output.resolve()
    if not output.is_relative_to((ROOT / 'testcases').resolve()): parser.error('generated evidence must remain under testcases/')
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists(): parser.error('preserve existing evidence; select a new output filename')
    if args.worker: return run_worker(output)
    env = os.environ.copy()
    for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        env[name] = str(frozen_protocol()['resources']['blasThreads'])
    start = time.perf_counter()
    try:
        result = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker', '--output', str(output)], env=env, cwd=ROOT, timeout=frozen_protocol()['resources']['maxQualificationWallSeconds'])
    except subprocess.TimeoutExpired:
        partial = output.with_suffix('.partial.json')
        report = json.loads(partial.read_text()) if partial.exists() else {'allFailures': [], 'runs': []}
        report.update({'pass': False, 'freshProcessTotalWallS': time.perf_counter() - start, 'freshProcessPeakMiB': None})
        report['allFailures'].append({'check': 'freshProcessTimeout', 'maximum': frozen_protocol()['resources']['maxQualificationWallSeconds']})
        output.write_text(json.dumps(report, indent=2) + '\n')
        return 1
    if output.exists():
        report = json.loads(output.read_text()); report['freshProcessTotalWallS'] = time.perf_counter() - start
        if report['freshProcessTotalWallS'] > frozen_protocol()['resources']['maxQualificationWallSeconds']:
            report['allFailures'].append({'check': 'freshProcessTotalWallS', 'observed': report['freshProcessTotalWallS'], 'maximum': frozen_protocol()['resources']['maxQualificationWallSeconds']}); report['pass'] = False
        output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        print(json.dumps({'pass': report['pass'], 'failures': len(report['allFailures']), 'output': str(output), 'wallS': report['freshProcessTotalWallS'], 'peakMiB': report['freshProcessPeakMiB']}))
        return 0 if report['pass'] else 1
    return result.returncode


if __name__ == '__main__':
    raise SystemExit(main())
