#!/usr/bin/env python3
"""Patient-geometry sensitivity checks for explicit uncalibrated V2 assumptions.

Uses only public engine actions to produce state changes. Variant scenes retain
identical patient labels and affine, while explicitly declaring the changed
profile value. This checks direction of model response, not clinical accuracy.
"""
from dataclasses import replace
from pathlib import Path
import argparse
import copy
import json
import time
import numpy as np
from resectionlab.core import array_digest
from resectionlab.imaging import load_case
from resectionlab.surgical_scene import from_case
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab import surgical_vessels_v2 as vessel, surgical_drilling_v2 as drill
from prepare_patient_physics_v2 import file_hash
from verify_patient_physics_v2 import _recipe, _step, _conserved_fluid


def variant(scene, component, changes):
    properties = copy.deepcopy(scene.properties)
    properties['tissuePhysicsV2'][component].update(changes)
    properties['tissuePhysicsV2']['provenance']['description'] += ' Explicit sensitivity variant: ' + component + ' ' + json.dumps(changes, sort_keys=True)
    result = replace(scene, properties=properties)
    if array_digest(result.labels) != array_digest(scene.labels) or not np.array_equal(result.affine, scene.affine):
        raise AssertionError('Variant changed patient geometry')
    return result


def pressure_response(scene, references, pressure):
    changed = variant(scene, 'vessel', {'pressurePa': pressure})
    engine = SurgicalEngine(changed); records = []
    for action in _recipe(references, 'vessel-verification.json')[:3]: _step(engine, action, records)
    _step(engine, {'slot':'left', 'operation':'apply', 'duration':1.}, records)
    metrics = vessel.metrics(engine)
    assert metrics['generatedMm3'] > 0 and abs(metrics['conservationResidualMm3']) < 1e-8
    return {'pressurePa':pressure, 'sceneId':changed.identity, 'actions':records, 'metrics':metrics,
            'fluidConservationResidualMm3':_conserved_fluid(engine)}


def drill_response(scene, references, rpm, force):
    engine = SurgicalEngine(scene); records = []; recipe = _recipe(references, 'verification.json')
    place = recipe[0]; place['parameters'].update(rpm=rpm, forceN=force, duty=1.)
    _step(engine, place, records)
    advance = recipe[1]; advance['duration'] = 5.
    _step(engine, advance, records)
    m = drill.metrics(engine)
    assert abs(m['drillInputEnergyJ'] - m['drillCuttingWorkJ'] - m['drillUnusedEnergyJ']) < 1e-8
    return {'rpm':rpm, 'forceN':force, 'durationSeconds':5., 'actions':records, 'metrics':m}


def brain_response(scene, references, scale):
    settings = scene.properties['tissuePhysicsV2']['brain']
    changes = {k:settings[k]*scale for k in ('brainStiffnessNPerMm', 'tumorStiffnessNPerMm')}
    changed = variant(scene, 'brain', changes); engine = SurgicalEngine(changed); records = []
    recipe = _recipe(references, 'retraction-verification.json')
    for action in recipe[:3]: _step(engine, action, records)
    motion = recipe[3]; origin = np.array(recipe[2]['pose']['position']); end = np.array(motion['pose']['position'])
    motion['pose']['position'] = (origin + .35*(end-origin)/np.linalg.norm(end-origin)).tolist()
    motion['duration'] = .5; _step(engine, motion, records)
    values = [float(np.linalg.norm(v)) for v in engine.state.get('deformation',{}).values()]
    assert values and max(values)>0 and not engine.state['removed']
    return {'stiffnessMultiplier':scale, 'declaredChanges':changes, 'sceneId':changed.identity,
            'maxDisplacementMm':max(values), 'meanDisplacementMm':float(np.mean(values)),
            'actions':records, 'metrics':engine.metrics()}


def mechanical_contact(scene, references):
    engine = SurgicalEngine(scene); records = []; recipe = _recipe(references, 'vessel-verification.json')
    # Open jaws suppress the cutting mechanism. Force-free approach separates
    # geometric traversal from subsequent explicit contact loading.
    recipe[1]['parameters'].update(jawOpeningMm=2., forceN=0.)
    for action in recipe[:3]: _step(engine, action, records)
    initial = vessel.metrics(engine)
    if initial['damagedFaceCount'] or engine.state['transectedEdges']:
        raise AssertionError('Open-jaw force-free approach produced vessel damage')
    exposures = []
    elapsed = 0.
    for dwell in (.25, 1.5, 1., 1.):
        elapsed += dwell
        _step(engine, {'slot':'left', 'operation':'apply', 'duration':dwell,
                       'parameters':{'forceN':20., 'jawOpeningMm':2.}}, records)
        faces = engine.state['vesselsV2']['faces']
        severity = max((f['damage'] for f in faces.values()), default=0.)
        exposures.append({'appliedSeconds':elapsed, 'maxInjury':severity,
                          'metrics':vessel.metrics(engine)})
        if severity >= 1. and any(0 < x['maxInjury'] < 1 for x in exposures): break
    if engine.state['transectedEdges']: raise AssertionError('Open jaws recorded a geometric transection')
    if not any(0 < x['maxInjury'] < 1 for x in exposures) or exposures[-1]['maxInjury'] != 1.:
        raise AssertionError('Mechanical dose did not show progressive partial-to-full injury')
    assert exposures[-1]['metrics']['generatedMm3'] > 0
    return {'instrument':'micro_scissors', 'jawOpeningMm':2., 'forceN':20.,
            'initial':initial, 'exposures':exposures, 'actions':records,
            'transectedEdgeCount':len(engine.state['transectedEdges']),
            'fluidConservationResidualMm3':_conserved_fluid(engine)}


def verify(case_path='outputs/surgical-patient-v2/patient.ressectionlab', output='outputs/surgical-patient-v2',
           source_path='outputs/surgical-patient-sub06/patient.ressectionlab'):
    started=time.perf_counter(); source=Path(source_path).resolve(); case_path=Path(case_path).resolve()
    before=file_hash(source); derivative_before=file_hash(case_path)
    case=load_case(case_path); scene=from_case(case); labels_hash=array_digest(scene.labels)
    references=source.parent; pressure=[pressure_response(scene,references,p) for p in (500.,1500.)]
    if not pressure[0]['metrics']['generatedMm3'] < pressure[1]['metrics']['generatedMm3']:
        raise AssertionError('Higher declared pressure did not increase leak generation')
    drilling=[drill_response(scene,references,rpm,force) for rpm,force in ((30000.,2.),(60000.,5.))]
    if not drilling[0]['metrics']['drillCuttingWorkJ'] < drilling[1]['metrics']['drillCuttingWorkJ']:
        raise AssertionError('Higher drill drive did not increase cutting work')
    if not drilling[0]['metrics']['drillRemovedVolumeMm3'] <= drilling[1]['metrics']['drillRemovedVolumeMm3']:
        raise AssertionError('Higher drill drive reduced removed volume')
    brain=[brain_response(scene,references,scale) for scale in (.5,2.)]
    if not brain[0]['maxDisplacementMm'] > brain[1]['maxDisplacementMm']:
        raise AssertionError('Higher brain stiffness did not reduce displacement')
    mechanical=mechanical_contact(scene,references)
    if file_hash(source)!=before or file_hash(case_path)!=derivative_before or array_digest(scene.labels)!=labels_hash:
        raise AssertionError('Patient source or labels changed')
    report={'schema':'resectionlab.patient-physics-sensitivity.v2','status':'passed',
        'casePath':str(case_path),'sourceCase':str(source),'sourceFileSha256':before,
        'sourceUnchanged':True,'labelArrayHash':labels_hash,'labelsUnchanged':True,
        'pressure':pressure,'drilling':drilling,'brainStiffness':brain,'mechanicalVesselContact':mechanical,
        'elapsedWallSeconds':time.perf_counter()-started,
        'limitations':['Sensitivity checks concern explicit engineering assumptions, not calibrated patient measurements',
                       'Thermal vessel injury is not claimed as coupled to drill temperatures',
                       'Drilling comparison changes rpm and force together; individual causal contributions are not isolated']}
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    (output/'sensitivity.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case',nargs='?',default='outputs/surgical-patient-v2/patient.ressectionlab')
    parser.add_argument('output',nargs='?',default='outputs/surgical-patient-v2')
    parser.add_argument('--source',default='outputs/surgical-patient-sub06/patient.ressectionlab')
    args=parser.parse_args(); result=verify(args.case,args.output,args.source)
    print(json.dumps({k:result[k] for k in ('status','casePath','sourceUnchanged','labelsUnchanged','elapsedWallSeconds')},indent=2))
