#!/usr/bin/env python3
"""Verify opt-in V2 physics on the unchanged source patient's actual anatomy.

Run after prepare_patient_physics_v2.py, with PYTHONPATH=src .venv/bin/python.
Saved V1 patient access trajectories are only candidate tool poses: every V2
movement and interaction is rechecked by public engine transitions. Separate
initially intact episodes exercise vessel injury/sealing, drilling, and brain
retraction. No hidden tissue removal, vessel seeding or anatomy edits occur.
"""
from pathlib import Path
import argparse
import copy
import json
import math
import sys
import subprocess
import time
import numpy as np
from resectionlab.core import array_digest
from resectionlab.imaging import load_case, save_case, read_case_artifacts
from resectionlab.surgical_scene import from_case
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab import surgical_vessels_v2, surgical_drilling_v2
from prepare_patient_physics_v2 import file_hash, array_hashes


def _recipe(directory, filename):
    data = json.loads((directory / filename).read_text())
    return [copy.deepcopy(record['action']) for record in data['actions']]


def _step(engine, action, records):
    before = engine.state_id; started = time.perf_counter()
    preview = engine.preview(action)
    if engine.state_id != before: raise AssertionError('Preview mutated committed patient state')
    result = engine.step(action)
    if result['stateId'] != preview['stateId'] or result['status'] != preview['status']:
        raise AssertionError('Preview and committed transition differ')
    record = {'action': copy.deepcopy(action), 'status': result['status'], 'reason': result.get('reason'),
        'stateId': result['stateId'], 'events': result['events'], 'metrics': result['metrics'],
        'previewEqualsCommit': True, 'wallSeconds': time.perf_counter() - started}
    records.append(record)
    print(action['operation'], result['status'], result.get('reason'), file=sys.stderr, flush=True)
    if result['status'] == 'rejected': raise AssertionError('Public action rejected: ' + str(result.get('reason')))
    return result


def _finish(engine, records, output, name, case):
    snapshot = engine.snapshot()
    encoded = json.dumps(snapshot, allow_nan=False)
    native = save_case(case, output / ('patient-' + name + '-verified.ressectionlab'), artifacts={'surgical_toolbox': {
        'mapping': engine.scene.regions, 'branches': {'main': json.loads(encoded)}, 'branchId': 'main',
        'position': len(engine.actions), 'parents': {'main': None}}})
    reopened = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--reopen', str(native)],
        capture_output=True, text=True, check=True)
    fresh = json.loads(reopened.stdout)
    if fresh['stateId'] != engine.state_id: raise AssertionError('Fresh-process patient action replay differs')
    (output / (name + '-snapshot.json')).write_text(json.dumps(snapshot, indent=2, allow_nan=False) + '\n')
    return {'actions': copy.deepcopy(records), 'metrics': engine.metrics(), 'stateId': engine.state_id,
            'previewVerified': True, 'jsonReplayVerified': True, 'freshProcessReopenVerified': True,
            'nativeCase': str(native), 'nativeCaseSha256': file_hash(native), 'preRemovedCells': 0}


def _conserved_fluid(engine):
    state = engine.state
    trapped = sum(s.get('reservoirMm3', 0.) for s in state['sources'])
    accounted = sum(state['fluid'].values()) + state['fluidRemovedMm3'] + state['fluidEscapedMm3'] + trapped
    if not math.isclose(state['fluidAddedMm3'], accounted, rel_tol=1e-8, abs_tol=1e-8):
        raise AssertionError('Fluid emission/removal/escape/reservoir ledger differs')
    return state['fluidAddedMm3'] - accounted


def _vessel(scene, references, output, case):
    engine = SurgicalEngine(scene); records = []; recipe = _recipe(references, 'vessel-verification.json')
    initial = surgical_vessels_v2.metrics(engine)
    if initial.get('generatedMm3', 0.) != 0 or initial.get('woundCount', 0) != 0:
        raise AssertionError('Intact patient spontaneously bleeds')
    # Cut, then explicitly dwell with the same closed scissors.
    for action in recipe[:3]: _step(engine, action, records)
    _step(engine, {'slot':'left', 'operation':'apply', 'duration':1.}, records)
    injured = surgical_vessels_v2.metrics(engine)
    if injured['generatedMm3'] <= 0 or injured['woundCount'] == 0: raise AssertionError('Patient vessel was not injured')
    # The reference next performs withdrawal, bipolar exchange, placement,
    # approach and a 3-second energy dwell at the same observed wound.
    for action in recipe[3:8]: _step(engine, action, records)
    sealed = surgical_vessels_v2.metrics(engine)
    if sealed['activeWounds'] != 0: raise AssertionError('Bipolar dwell did not seal the observed wound')
    _step(engine, {'slot':'left', 'operation':'deactivate', 'duration':1.}, records)
    _step(engine, {'slot':'left', 'operation':'inspect', 'duration':2.}, records)
    resting = surgical_vessels_v2.metrics(engine)
    if not math.isclose(resting['generatedMm3'], sealed['generatedMm3'], abs_tol=1e-9):
        raise AssertionError('Sealed wound restarted bleeding without exposure')
    if any(face['patent'] for face in engine.state['vesselsV2']['faces'].values()):
        raise AssertionError('Coagulation restored severed vessel patency')
    residual = _conserved_fluid(engine)
    if abs(resting['conservationResidualMm3']) > 1e-8: raise AssertionError('Vascular compartment lost volume')
    result = _finish(engine, records, output, 'vessel', case)
    result.update(injured=injured, sealed=sealed, afterRest=resting, fluidConservationResidualMm3=residual)
    return result


def _drill(scene, references, output, case):
    engine = SurgicalEngine(scene); records = []; recipe = _recipe(references, 'verification.json')
    place = recipe[0]; place['parameters'].update(forceN=5., rpm=60000., duty=1.)
    _step(engine, place, records)
    advance = recipe[1]; advance['duration'] = 5.
    for _ in range(8):
        _step(engine, advance, records)
        m = surgical_drilling_v2.metrics(engine)
        if m['drillRemovedVolumeMm3'] > 0 and m['drillHeatGeneratedJ'] > 0: break
    else: raise AssertionError('Patient bone did not receive sufficient work to remove a cell')
    m = surgical_drilling_v2.metrics(engine); state = engine.state['drillingV2']
    # Heat is a declared fraction of cutting work, not extra input energy.
    energy_residual = m['drillInputEnergyJ'] - m['drillCuttingWorkJ'] - m['drillUnusedEnergyJ']
    cutting_residual = m['drillCuttingWorkJ'] - m['drillPendingWorkJ'] - m['drillRemovedVolumeMm3'] * scene.properties['tissuePhysicsV2']['bone']['specificEnergyJPerMm3']
    if abs(cutting_residual) > 1e-7: raise AssertionError('Drill cutting work/volume ledger differs')
    if abs(energy_residual) > 1e-7: raise AssertionError('Drill energy ledger differs')
    removed = engine.state['removed']
    if not removed or any(engine.by_label[int(scene.labels.flat[i])]['material'] != 'bone' for i in removed):
        raise AssertionError('Drill removal was absent or crossed a protected material')
    if not state['temperatureC'] or m['drillMaxTemperatureC'] <= scene.properties['tissuePhysicsV2']['bone']['ambientTemperatureC']:
        raise AssertionError('Drill heating absent')
    result = _finish(engine, records, output, 'drill', case)
    result.update(drilling=m, energyConservationResidualJ=energy_residual, cuttingWorkResidualJ=cutting_residual,
                  removalRequiresAccumulatedWork=True, protectedMaterialsRetained=True)
    return result


def _brain(scene, references, output, case):
    engine = SurgicalEngine(scene); records = []; recipe = _recipe(references, 'retraction-verification.json')
    for action in recipe[:3]: _step(engine, action, records)
    motion = recipe[3]
    # Sub-voxel commanded displacement demonstrates the continuous surrogate.
    origin = np.array(recipe[2]['pose']['position']); end = np.array(motion['pose']['position'])
    direction = (end - origin) / np.linalg.norm(end - origin)
    motion['pose']['position'] = (origin + .35 * direction).tolist(); motion['duration'] = .5
    _step(engine, motion, records)
    held_cells = set(engine.state['holds']['left']['cells'])
    displacements = engine.state.get('deformation', {})
    peak = max((float(np.linalg.norm(v)) for v in displacements.values()), default=0.)
    neighbors = [float(np.linalg.norm(v)) for k, v in displacements.items() if int(k) not in held_cells]
    if peak <= 0 or not neighbors or max(neighbors) <= 0: raise AssertionError('Brain deformation did not propagate to neighbors')
    if engine.state['removed']: raise AssertionError('Retraction removed patient tissue')
    moved = {'maxDisplacementMm': peak, 'deformedNodes': sum(bool(np.linalg.norm(v)>1e-10) for v in displacements.values()),
             'nonHeldMaxDisplacementMm': max(neighbors), 'model': copy.deepcopy(engine.state['deformationModel'])}
    peak_artifact = _finish(engine, records, output, 'brain-retracted', case)
    _step(engine, {'slot':'left', 'operation':'release', 'duration':.5}, records)
    _step(engine, {'slot':'left', 'operation':'inspect', 'duration':2.}, records)
    relaxed = max((float(np.linalg.norm(v)) for v in engine.state.get('deformation', {}).values()), default=0.)
    if relaxed >= peak: raise AssertionError('Brain did not relax after release')
    result = _finish(engine, records, output, 'brain', case)
    result.update(underRetraction=moved, underRetractionNativeCase=peak_artifact['nativeCase'],
                  underRetractionFreshProcessReopenVerified=peak_artifact['freshProcessReopenVerified'],
                  relaxedMaxDisplacementMm=relaxed,
                  continuousSubvoxelResponse=True, neighboringTissueResponse=True)
    return result


def verify(case_path='outputs/surgical-patient-v2/patient.ressectionlab', output='outputs/surgical-patient-v2',
           source_path='outputs/surgical-patient-sub06/patient.ressectionlab', reference_directory=None):
    started = time.perf_counter(); output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=True)
    source_path = Path(source_path).resolve(); case_path = Path(case_path).resolve()
    references = Path(reference_directory).resolve() if reference_directory else source_path.parent
    original_hash = file_hash(source_path); derivative_hash = file_hash(case_path)
    original = load_case(source_path); case = load_case(case_path); scene = from_case(case)
    if array_hashes(original) != array_hashes(case) or original.source_refs != case.source_refs:
        raise AssertionError('Derivative anatomy or SourceRefs differ from original patient')
    if 'tissuePhysicsV2' not in scene.properties or 'tissuePhysics' in scene.properties:
        raise AssertionError('Explicit V2-only profile required')
    label_hash = array_digest(scene.labels); image_hash = array_digest(case.mri)
    results = {}
    for name, run in (('vessel', _vessel), ('drill', _drill), ('brain', _brain)):
        print('Verifying patient', name, file=sys.stderr, flush=True)
        results[name] = run(scene, references, output, case)
        (output / (name + '-verification.json')).write_text(json.dumps(results[name], indent=2, allow_nan=False) + '\n')
    if file_hash(source_path) != original_hash or file_hash(case_path) != derivative_hash:
        raise AssertionError('A patient bundle changed during verification')
    if array_digest(scene.labels) != label_hash or array_digest(case.mri) != image_hash:
        raise AssertionError('Patient anatomy or intensity changed during verification')
    report = {'schema':'resectionlab.patient-physics-verification.v2', 'status':'passed',
        'sourceCase':str(source_path), 'sourceFileSha256':original_hash, 'casePath':str(case_path),
        'caseFileSha256':derivative_hash, 'sceneId':scene.identity, 'labelArrayHash':label_hash,
        'intensityArrayHash':image_hash, 'sourceUnchanged':True, 'arraysUnchanged':True,
        'preRemovedCells':0, 'candidateTrajectoryDirectory':str(references), 'episodes':results,
        'elapsedWallSeconds':time.perf_counter()-started,
        'limitations':['Software transition and conservation verification, not clinical validation',
                       'Engineering assumptions do not predict patient force, tissue damage or blood loss',
                       'Trajectories are test access poses, not surgical recommendations']}
    (output / 'verification.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('case', nargs='?', default='outputs/surgical-patient-v2/patient.ressectionlab')
    parser.add_argument('output', nargs='?', default='outputs/surgical-patient-v2')
    parser.add_argument('--source', default='outputs/surgical-patient-sub06/patient.ressectionlab')
    parser.add_argument('--reference-directory')
    parser.add_argument('--reopen', help='Internal fresh-process native artifact replay check')
    args = parser.parse_args()
    if args.reopen:
        reopened_case = load_case(args.reopen); toolbox = read_case_artifacts(args.reopen)['surgical_toolbox']
        reopened_scene = from_case(reopened_case, toolbox['mapping'])
        restored = SurgicalEngine.restore(reopened_scene, toolbox['branches'][toolbox['branchId']])
        print(json.dumps({'stateId': restored.state_id, 'actions': len(restored.actions)}))
        raise SystemExit(0)
    report = verify(args.case, args.output, args.source, args.reference_directory)
    print(json.dumps({k:report[k] for k in ('status','casePath','sourceUnchanged','arraysUnchanged','elapsedWallSeconds')}, indent=2))
