#!/usr/bin/env python3
"""Bounded native-bridge checks for an explicitly prepared planar maneuver.

All evidence and saved derivatives must stay in a fresh ignored testcases/
folder. Input source arrays/materials are never edited by this checker. Python
BridgeSession timings are reported separately from hidden Electron draw timings.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import statistics
import subprocess
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from resectionlab.core import array_digest, thaw_json
from resectionlab.desktop_bridge import BridgeSession, _Request
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_environment import standardize
from resectionlab.surgical_scene import from_case
from resectionlab.workspace_bundle import load_workspace


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ignored(path):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT / 'testcases'):
        raise ValueError('Evidence/input must remain beneath ignored testcases/')
    return path


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def source_arrays(case):
    result = {'mri': array_digest(case.mri), 'affine': array_digest(case.affine),
              'compartments': {k: array_digest(v) for k, v in case.compartments.items()},
              'sourceCompartments': {k: array_digest(v) for k, v in case.source_compartments.items()}}
    if case.brain_mask is not None:
        result['brainMask'] = array_digest(case.brain_mask)
    return result


def scene_for(case, record):
    return standardize(from_case(case, record['mapping']), physics_profile=record['physicsProfile'])


def audit_sources(output):
    """Inventory historical incompatibilities without altering a source case."""
    rows = []
    for name in ('operative-demo', 'continuum-research', 'consequence-comparison'):
        path = ROOT / f'testcases/cases/UCSF-PDGM-0007-{name}.ressectionlab'
        started = time.perf_counter()
        case, artifacts, _ = load_workspace(path)
        record = artifacts['surgical_toolbox']
        scene = scene_for(case, record)
        profile = scene.properties.get('brainContinuumV1')
        row = {'path': str(path.relative_to(ROOT)), 'archiveSha256': sha(path),
               'caseHash': case.semantic_hash, 'sourceArrays': source_arrays(case),
               'sourceLabelsSha256': array_digest(scene.labels), 'sourceSceneId': scene.identity,
               'shape': list(scene.shape), 'affine': scene.affine.tolist(),
               'continuumProfile': profile, 'selectedBranch': record['branchId'],
               'selectedPosition': record['position'], 'branches': {}}
        for key, saved in record['branches'].items():
            state = saved['checkpoint']['state']
            reasons = []
            if profile is None:
                reasons.append('Missing explicit continuum material spectrum; unchanged source refuses')
            if state['removed']:
                reasons.append('Historical source removal topology cannot be reset to intact reference')
            if state['holds']:
                reasons.append('Existing native holds cannot be silently rebound to planar contact')
            continuum = state.get('brainContinuumV1') or {}
            if state['deformation'] or continuum.get('cornerDisplacementMm') or continuum.get('memoryDisplacementMm'):
                reasons.append('Existing displacement/Maxwell history requires validated transfer; no reset')
            row['branches'][key] = {'actions': len(saved['actions']), 'stateId': saved['stateId'],
                'removedCells': len(state['removed']), 'holds': len(state['holds']),
                'deformationCells': len(state['deformation']),
                'continuumCornerEntries': len(continuum.get('cornerDisplacementMm', {})),
                'compatibility': 'refused' if reasons else 'clean-history-candidate-only',
                'reasons': reasons or ['Requires a declared physical footprint and independently anchored support; none stored in source archive']}
        # Exercise the clean archived main through the actual canonical engine;
        # this must refuse before any scene/history/clock mutation.
        engine = SurgicalEngine.restore(scene, record['branches']['main'])
        before = engine.snapshot()
        refused = engine.step({'slot': 'left', 'operation': 'bladeManeuver',
            'maneuverPhase': 'approach', 'duration': .1})
        assert refused['status'] == 'rejected'
        assert 'PLANAR_MANEUVER_EXPLICIT_DECLARATION_REQUIRED' in refused['reason']
        assert engine.snapshot() == before
        row['actualCanonicalMainProbe'] = {'action': 'bladeManeuver/approach',
            'status': refused['status'], 'reason': refused['reason'],
            'beforeStateId': before['stateId'], 'afterStateId': engine.state_id,
            'completeSnapshotPreserved': True}
        row['auditWallS'] = time.perf_counter() - started
        assert sha(path) == row['archiveSha256']
        rows.append(row)
    report = {'schema': 'resectionlab.planar-blade-source-audit.v1', 'sources': rows,
              'sourceArraysAndMaterialsChanged': False,
              'claim': 'Read-only source compatibility audit; clean history alone does not admit a maneuver'}
    write(output, report)
    return report




def prepare_engineering(output):
    """Persist a clearly generated coupon through the ordinary case adapter."""
    from resectionlab import surgical_planar_maneuver as maneuver
    from resectionlab.core import CaseData, SourceRef
    from resectionlab.imaging import save_case
    from resectionlab.surgical_environment import SCHEMA
    output = ignored(output)
    if output.exists():
        raise ValueError('Choose a NEW engineering fixture directory')
    output.mkdir(parents=True)
    scene = maneuver.engineering_scene()
    declared = copy.deepcopy(scene.properties[maneuver.PROPERTY])
    base_properties = copy.deepcopy(scene.properties)
    base_properties.pop(maneuver.PROPERTY)
    case = CaseData(case_id='generated-planar-blade-maneuver-coupon',
        mri=scene.labels.astype(np.float32),
        compartments={region['name']: scene.labels == region['id'] for region in scene.regions},
        affine=scene.affine,
        source_refs=(SourceRef('generated-planar-blade-maneuver-coupon',
            'synthetic://resectionlab/generated-planar-blade-maneuver-coupon',
            native_frame='RAS+', provenance='simulated'),),
        unknowns=('Generated numerical engineering fixture; no patient anatomy or measured response',),
        metadata={'is_synthetic': True, 'generated': True,
            'description': 'Generated4mm nu0 Maxwell numerical coupon; not patient anatomy',
            'anatomy_layers': {region['name']: {'kind': region['material'], 'color': region.get('color', '#c799b0'),
                'provenance': 'generated numerical engineering coupon'} for region in scene.regions},
            'surgical_properties': base_properties})
    # Bind to stable physical source evidence produced by the actual case
    # adapter. Derived case hashes must be excluded from this geometry digest,
    # because case metadata itself contains this declaration.
    from resectionlab.surgical_planar_contact import PlanarBlade
    base = standardize(from_case(case, scene.regions))
    blade = declared['blade']
    properties = copy.deepcopy(base.properties)
    properties[maneuver.PROPERTY] = maneuver.declaration(base, declared['cells'],
        declared['contactFaces'], declared['supportFaces'],
        PlanarBlade.create(blade['originMm'], blade['axisU'], blade['axisV'],
            blade['inwardNormal'], blade['sizeMm']), declared['binding'])
    metadata = thaw_json(case.metadata)
    metadata['surgical_properties'] = properties
    case = case.revised(metadata=metadata)
    adapted = standardize(from_case(case, base.regions))
    np.testing.assert_array_equal(adapted.labels, scene.labels)
    np.testing.assert_array_equal(adapted.affine, scene.affine)
    engine = SurgicalEngine(adapted)
    initial = engine.snapshot()
    record = {'environmentSchema': SCHEMA, 'physicsProfile': copy.deepcopy(adapted.properties['tissuePhysicsV2']),
        'mapping': adapted.regions, 'branches': {'main': initial, 'planar-maneuver': copy.deepcopy(initial)},
        'parents': {'main': None, 'planar-maneuver': {'branchId': 'main', 'position': 0}},
        'branchId': 'planar-maneuver', 'position': 0,
        'researchSetup': {'kind': 'generated-planar-blade-numerical-engineering-fixture',
            'description': 'Uniform4mm nu0 coupon with explicit16mm2 finite blade and independent bottom clamp. No acquired or patient anatomy; cycle-work sensitivity remains unqualified.'}}
    destination = output / 'generated-planar-blade-maneuver.ressectionlab'
    save_case(case, destination, artifacts={'surgical_toolbox': record})
    steps = [{'phase': phase, 'action': {'slot': 'left', 'operation': 'bladeManeuver',
        'maneuverPhase': phase, 'duration': duration}}
        for phase, duration in [('approach', .1), ('indent', .2), ('hold', .2), ('withdraw', .2)]]
    write(output / 'actions.json', {'steps': steps})
    result = {'schema': 'resectionlab.planar-blade-engineering-preparation.v1',
        'case': str(destination), 'caseSha256': sha(destination), 'sourceArrays': source_arrays(case),
        'sceneId': adapted.identity, 'initialStateId': engine.state_id,
        'actions': str(output / 'actions.json'), 'role': 'generated numerical engineering coupon only',
        'patientAnatomy': False, 'trainingUpdates': 0, 'newAcquiredDatasetBytes': 0}
    write(output / 'preparation.json', result)
    return result


def uniform_maxwell_oracle(actions, partitions=None):
    """Independent scalar BE recurrence for the frozen nu=0,4mm coupon.

    Compression-positive free recovery satisfies mu*d + mu1*(d-z)=0.
    No native/kernel function or archived expected output enters this oracle.
    """
    phase_start = -.1
    memory = 0.
    elapsed = 0.
    rows = []
    cursor = 0
    endings = {'approach': 0., 'indent': .02, 'hold': .02, 'withdraw': -.1}
    for row in actions:
        action = row.get('action', row)
        phase = action['maneuverPhase']
        count = round(action['duration'] / .025)
        for step in range(1, count + 1):
            begin = phase_start + (endings[phase] - phase_start) * (step-1) / count
            substeps = partitions[cursor] if partitions is not None else 1
            ratio = .5 / (.5 + .025/substeps)
            for inner in range(1, substeps+1):
                offset = begin + (endings[phase] - phase_start)/count * inner/substeps
                free = ratio * 500. * memory / (1000. + ratio * 500.)
                compression = max(offset, free)
                memory = ratio * memory + (1 - ratio) * compression
                reaction = 2e-6 * 16. / 4. * (1000. * compression + 500. * (compression - memory))
                if free >= offset:
                    reaction = 0.
            cursor += 1
            elapsed += .025
        rows.append({'phase': phase, 'offsetMm': endings[phase],
                     'maxDisplacementMm': compression, 'memoryIndentationMm': memory,
                     'normalReactionN': reaction, 'elapsedS': elapsed,
                     'contactAreaMm2': 16. if offset >= free - 1e-9 else 0.})
        phase_start = endings[phase]
    return rows


def check_uniform_endpoint(view, expected):
    actual = view['planarBladeManeuver']
    for key in ('offsetMm', 'maxDisplacementMm', 'normalReactionN', 'contactAreaMm2'):
        if not np.isfinite(actual[key]) or abs(actual[key] - expected[key]) > 1e-9:
            raise AssertionError(f'Independent uniform Maxwell endpoint {key}: {actual[key]} != {expected[key]}')
    return {'independentMethod': 'Scalar uniaxial BE compression/recovery recurrence; mu1000Pa/nu0,branch500Pa,tau.5s,4mm block/16mm2 footprint',
            'expected': expected, 'actual': {key: actual[key] for key in ('offsetMm', 'maxDisplacementMm', 'normalReactionN', 'contactAreaMm2')}}


def run_bridge(case_path, actions_path, output, *, uniform_oracle=False):
    case_path = ignored(case_path)
    output = ignored(output)
    if output.exists():
        raise ValueError('Choose a NEW output directory; preserve prior evidence')
    output.mkdir(parents=True)
    raw_actions = json.loads(Path(actions_path).read_text())
    actions = raw_actions['steps'] if isinstance(raw_actions, dict) else raw_actions
    if not isinstance(actions, list) or not 1 <= len(actions) <= 120:
        raise ValueError('One bounded explicit maneuver schedule required')
    oracle = None
    original_hash = sha(case_path)
    case, original_artifacts, _ = load_workspace(case_path)
    original_record = original_artifacts['surgical_toolbox']
    original_arrays = source_arrays(case)
    scene = scene_for(case, original_record)
    if uniform_oracle:
        # Record only accepted time partitions; expected displacement, force and
        # memory are then calculated by the independent scalar equations above.
        from resectionlab import surgical_planar_maneuver as maneuver
        class PartitionEngine(SurgicalEngine):
            def __init__(self, scene):
                super().__init__(scene); self.partitions = []
            def clone(self):
                candidate = super().clone(); candidate.partitions = list(self.partitions)
                return candidate
            def _substep(self, *args):
                events = super()._substep(*args)
                self.partitions.append(self.state[maneuver.PROPERTY]['receipt'].get('internalSubsteps', 1))
                return events
        numerical = PartitionEngine(scene)
        for row in actions:
            result = numerical.step(row.get('action', row))
            assert result['status'] == 'accepted', result
        oracle = uniform_maxwell_oracle(actions, numerical.partitions)
    original_labels = array_digest(scene.labels)
    original_properties = copy.deepcopy(scene.properties)
    report = {'schema': 'resectionlab.planar-blade-bridge-check.v1', 'input': str(case_path),
              'sourceSha256': original_hash, 'sourceArrays': original_arrays, 'steps': [],
              'actionsSha256': sha(actions_path),
              'implementationSha256': {str(path.relative_to(ROOT)): sha(path) for path in [
                  Path(__file__), ROOT / 'desktop/electron/verify-planar-blade-maneuver.cjs',
                  *[ROOT / 'src/resectionlab' / name for name in ('surgical_planar_maneuver.py',
                      'surgical_planar_contact.py', 'surgical_engine.py', 'surgical_continuum.py',
                      'surgical_motion.py', 'surgical_bridge.py', 'surgical_verified_cache.py')]]},
              'numpyVersion': np.__version__,
              'numericalThreads': {key: os.environ.get(key) for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')},
              'timingScope': 'Python BridgeSession load/inspect, preview and action only; excludes Electron, GPU and physical display',
              'trainingUpdates': 0, 'newAcquiredDatasetBytes': 0, 'checks': [], 'passed': False}
    snapshots = output.parent / 'source-snapshots'
    snapshots.mkdir(exist_ok=True)
    report['implementationSnapshots'] = {}
    for relative, digest in report['implementationSha256'].items():
        source = ROOT / relative
        snapshot = snapshots / (digest + '-' + source.name)
        if snapshot.exists():
            assert sha(snapshot) == digest
        else:
            snapshot.write_bytes(source.read_bytes())
        report['implementationSnapshots'][relative] = str(snapshot.relative_to(ROOT))
    session = BridgeSession(output / 'transfers')
    counter = 0
    def call(operation, args):
        nonlocal counter
        counter += 1
        started = time.perf_counter()
        result = session.execute(operation, args, _Request(str(counter)), lambda *_: None)
        return result, time.perf_counter() - started
    try:
        installed, report['coldLoadWallS'] = call('loadCase', {'path': str(case_path)})
        case_hash = installed['caseHash']
        current, report['coldInspectPreparationWallS'] = call('surgicalRehearsal', {'caseHash': case_hash, 'command': 'inspect'})
        report['initialStateId'] = current['view']['stateId']
        initial_position = current['position']
        report['blockedUnrelatedCommands'] = []
        for command in ('plan', 'fullCase', 'recoverExit', 'attachProtection'):
            before = thaw_json(session.cases[case_hash].artifacts)
            started = time.perf_counter()
            try:
                call('surgicalRehearsal', {'caseHash': case_hash, 'command': command,
                    'expectedStateId': current['view']['stateId']})
            except ValueError as error:
                assert 'PLANAR_MANEUVER_ONLY' in str(error), str(error)
                report['blockedUnrelatedCommands'].append({'command': command,
                    'refusal': str(error), 'wallS': time.perf_counter() - started})
            else:
                raise AssertionError('Unrelated command was accepted: ' + command)
            assert thaw_json(session.cases[case_hash].artifacts) == before, 'Refused command changed accepted archive'
        for index, row in enumerate(actions):
            action = row.get('action', row)
            before = thaw_json(session.cases[case_hash].artifacts)
            before_id = current['view']['stateId']
            preview, preview_wall = call('surgicalRehearsal', {'caseHash': case_hash, 'command': 'preview', 'expectedStateId': before_id, 'action': action})
            assert thaw_json(session.cases[case_hash].artifacts) == before, 'Preview changed accepted source/history'
            current, step_wall = call('surgicalRehearsal', {'caseHash': case_hash, 'command': 'step', 'expectedStateId': before_id, 'action': action})
            assert current['outcome']['status'] != 'rejected', current['outcome']
            record = thaw_json(session.cases[case_hash].artifacts)['surgical_toolbox']
            saved = record['branches'][current['branchId']]
            replay_started = time.perf_counter()
            replayed = SurgicalEngine.restore(scene, json.loads(json.dumps(saved)))
            replay_wall = time.perf_counter() - replay_started
            assert replayed.state_id == current['view']['stateId']
            assert replayed.state == saved['checkpoint']['state']
            assert not replayed.state['removed'] and not replayed.state['holds'], 'Planar coupon must not remove source or create native holds'
            retained = {'phase': row.get('phase'), 'action': action, 'stateId': replayed.state_id,
                        'position': current['position'], 'previewWallS': preview_wall, 'bridgeStepWallS': step_wall,
                        'freshExactReplayWallS': replay_wall, 'preview': preview,
                        'view': current['view'], 'outcome': current['outcome'], 'canonicalState': replayed.state}
            if oracle is not None:
                retained['independentUniformOracle'] = check_uniform_endpoint(current['view'], oracle[index])
            report['steps'].append(retained)
            write(output / f'step-{index+1:03d}.json', retained)
        final_id = current['view']['stateId']
        final_position = current['position']
        report['finalStateId'] = final_id
        report['finalPosition'] = final_position
        branch = current['branchId']
        for position, wanted in [(initial_position, report['initialStateId']), (final_position, final_id)]:
            current, wall = call('surgicalRehearsal', {'caseHash': case_hash, 'command': 'seek', 'expectedStateId': current['view']['stateId'], 'position': position})
            assert current['view']['stateId'] == wanted
            report.setdefault('seekTimings', []).append({'position': position, 'wallS': wall, 'stateId': wanted})
        saved_path = output / 'bridge-saved.ressectionlab'
        _, report['saveWallS'] = call('saveCase', {'caseHash': case_hash, 'path': str(saved_path)})
        saved_case, saved_artifacts, _ = load_workspace(saved_path)
        assert source_arrays(saved_case) == original_arrays
        saved_scene = scene_for(saved_case, saved_artifacts['surgical_toolbox'])
        assert array_digest(saved_scene.labels) == original_labels
        assert saved_scene.properties == original_properties
        for key, snapshot in original_record['branches'].items():
            if key != branch:
                assert saved_artifacts['surgical_toolbox']['branches'][key] == snapshot
        report['saved'] = str(saved_path)
        session.close()
        session = BridgeSession(output / 'cold-reopen-transfers')
        installed, report['coldReopenLoadWallS'] = call('loadCase', {'path': str(saved_path)})
        current, report['coldReopenReplayWallS'] = call('surgicalRehearsal', {'caseHash': installed['caseHash'], 'command': 'inspect'})
        assert current['view']['stateId'] == final_id
        assert sha(case_path) == original_hash
        for relative, digest in report['implementationSha256'].items():
            assert sha(ROOT / relative) == digest, 'Implementation changed during bridge proof: ' + relative
        report['checks'] = ['Every preview preserves accepted archive', 'Every action has fresh exact full-state replay',
                            'No source removal/native hold', 'Timeline initial/final identities exact',
                            'Native save and cold reopen exact', 'Source arrays, labels, material declarations and older branches preserved']
        report['passed'] = True
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        session.close()
        write(output / 'report.json', report)
    return report



def benchmark_bridge(case_path, actions_path, output, repeats):
    """Fresh BridgeSessions with no preview or replay before the first commit."""
    if not 1 <= repeats <= 10:
        raise ValueError('Benchmark repeats must be between1 and10')
    case_path, output = ignored(case_path), ignored(output)
    if output.exists():
        raise ValueError('Choose a NEW benchmark output directory')
    output.mkdir(parents=True)
    raw = json.loads(Path(actions_path).read_text())
    rows = raw['steps'] if isinstance(raw, dict) else raw
    hardware = {'platform': platform.platform(), 'machine': platform.machine(),
        'pythonVersion': platform.python_version(), 'numpyVersion': np.__version__,
        'logicalCpuCount': os.cpu_count()}
    if sys.platform == 'darwin':
        for key in ('hw.model', 'machdep.cpu.brand_string', 'hw.memsize', 'hw.physicalcpu'):
            result = subprocess.run(['sysctl', '-n', key], capture_output=True, text=True)
            hardware[key] = result.stdout.strip() if result.returncode == 0 else None
    report = {'schema': 'resectionlab.planar-blade-bridge-benchmark.v1',
        'input': str(case_path), 'sourceSha256': sha(case_path), 'actionsSha256': sha(actions_path),
        'samples': repeats, 'hardware': hardware, 'sequences': [],
        'numericalThreads': {key: os.environ.get(key) for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')},
        'scope': 'Fresh BridgeSession per sequence, within one fresh Python process. Fixture construction, Python imports and source file hashing excluded. No previews, replay or explicit kernel preparation before commits. Cold inspect assembles/validates model but first action includes lazy timestep factor preparation. Later sessions may benefit from process-level pure operator/implementation caches. Excludes renderer/GPU/physical display.',
        'passed': False}
    identifiers = []
    for repeat in range(repeats):
        directory = output / f'sequence-{repeat+1:02d}'
        session = BridgeSession(directory / 'transfers')
        sequence = {'index': repeat+1, 'actions': []}
        request_number = 0
        def call(operation, args):
            nonlocal request_number
            request_number += 1
            started = time.perf_counter()
            result = session.execute(operation, args, _Request(str(request_number)), lambda *_: None)
            return result, time.perf_counter() - started
        try:
            installed, sequence['loadWallS'] = call('loadCase', {'path': str(case_path)})
            case_hash = installed['caseHash']
            current, sequence['inspectAssemblyWallS'] = call('surgicalRehearsal', {'caseHash': case_hash, 'command': 'inspect'})
            for row in rows:
                current, wall = call('surgicalRehearsal', {'caseHash': case_hash, 'command': 'step',
                    'expectedStateId': current['view']['stateId'], 'action': row.get('action', row)})
                assert current['outcome']['status'] != 'rejected'
                sequence['actions'].append({'phase': row['phase'], 'wallS': wall,
                    'stateId': current['view']['stateId'],
                    'normalReactionN': current['view']['planarBladeManeuver']['normalReactionN']})
            identifiers.append([row['stateId'] for row in sequence['actions']])
            assert identifiers[-1] == identifiers[0], 'Fresh benchmark sequences changed canonical state'
            report['sequences'].append(sequence)
            write(directory / 'timings.json', sequence)
        finally:
            session.close()
    report['summary'] = {}
    values = {key: [r[key] for r in report['sequences']] for key in ('loadWallS', 'inspectAssemblyWallS')}
    for index, row in enumerate(rows):
        values[row['phase'] + 'ActionWallS'] = [r['actions'][index]['wallS'] for r in report['sequences']]
    for key, observations in values.items():
        report['summary'][key] = {'n': len(observations), 'minimum': min(observations),
            'median': statistics.median(observations), 'maximum': max(observations), 'raw': observations}
    assert sha(case_path) == report['sourceSha256']
    report['passed'] = True
    write(output / 'report.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-engineering', type=Path, help='Create a NEW ignored generated coupon directory')
    parser.add_argument('--audit-source', type=Path, help='Write a NEW read-only historical source audit JSON')
    parser.add_argument('--case', type=Path)
    parser.add_argument('--actions', type=Path, help='JSON explicit steps with optional phase labels')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--benchmark-repeats', type=int, help='Separate no-preview fresh-session benchmark,1–10 repeats')
    parser.add_argument('--uniform-oracle', action='store_true', help='Check declared frozen nu0 Maxwell coupon analytically')
    args = parser.parse_args()
    if args.prepare_engineering:
        print(json.dumps(prepare_engineering(args.prepare_engineering)))
    elif args.audit_source:
        destination = ignored(args.audit_source)
        if destination.exists():
            parser.error('Preserve existing source audit; choose a fresh path')
        destination.parent.mkdir(parents=True, exist_ok=True)
        report = audit_sources(destination)
        print(json.dumps({'audit': str(destination), 'sources': len(report['sources'])}))
    elif args.case and args.actions and args.output:
        if args.benchmark_repeats:
            report = benchmark_bridge(args.case, args.actions, args.output, args.benchmark_repeats)
            print(json.dumps({'passed': report['passed'], 'samples': report['samples'], 'summary': report['summary']}))
        else:
            report = run_bridge(args.case, args.actions, args.output, uniform_oracle=args.uniform_oracle)
            print(json.dumps({'passed': report['passed'], 'steps': len(report['steps']), 'saved': report['saved']}))
    else:
        parser.error('Use --audit-source or all of --case/--actions/--output')


if __name__ == '__main__':
    main()
