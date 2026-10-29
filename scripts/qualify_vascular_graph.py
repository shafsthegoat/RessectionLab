#!/usr/bin/env python3
"""Generated source-bound vascular maneuvers with an independent expm checker.

No acquired anatomy, patient parameters, fitting, downloads or policy training.
Outputs are app-testing evidence and belong exclusively under testcases/.
"""
from pathlib import Path
import argparse
import copy
import json
import sys
import time

import numpy as np
from scipy.linalg import expm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from resectionlab.core import CaseData, SourceRef, array_digest
from resectionlab.imaging import save_case
from resectionlab.surgical_scene import from_case
from resectionlab.surgical_environment import standardize, SCHEMA as ENVIRONMENT_SCHEMA
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_physics_v2 import default_profile
from resectionlab.surgical_modeled_outcomes import summarize_engine_outcomes
from resectionlab.simulator_qualification import removed_accounting, file_sha256
from resectionlab import surgical_vascular_graph as graph

ASSUMED = 'Generated numerical-control coefficient; no patient/device calibration'
QUAD_X, QUAD_W = np.polynomial.legendre.leggauss(8)


def make_case():
    shape = (24,24,24); labels = np.zeros(shape, np.uint16)
    labels[9:14,8:13,14:17] = 1; labels[9:14,8:13,12:14] = 2
    def edge(name, a, b, kind, y, label):
        cells = [int(np.ravel_multi_index((x,y,12), shape)) for x in (4,5)]
        labels.ravel()[cells] = label
        return {'id':name, 'from':a, 'to':b, 'kind':kind, 'conductanceMm3PerPaS':.0001,
            'sourceCells':cells, 'sourceFace':cells[:], 'woundConductanceMm3PerPaS':[.0001,.0001], 'source':ASSUMED}
    edges = [edge('artery','a','b','artery',6,3), edge('vein','b','c','vein',10,4), edge('sinus','c','d','sinus',14,5)]
    profile = {'schema':graph.SCHEMA, 'provenance':{'kind':'assumed-research','description':ASSUMED},
        'coverage':{'sourceCaseId':'generated-vascular-coupon-v1','status':'generated-control',
            'missing':['No patient imaging, physiological boundaries, microvascular anatomy or neurological outcomes']},
        'fieldPressurePa':0., 'compressionForceN':1., 'coagulationDose':2., 'clotRatePerSecond':0.,
        'nodes':[{'id':name,'complianceMm3PerPa':.001,'initialStorageMm3':v,'source':ASSUMED}
            for name,v in [('a',8.),('b',6.),('c',4.),('d',2.)]], 'edges':edges,
        'boundaries':[{'id':name,'node':node,'pressurePa':pressure,'conductanceMm3PerPaS':.0001,'source':ASSUMED}
            for name,node,pressure in [('inlet','a',10000.),('outlet','d',0.)]],
        'regions':[{'id':'declared-tissue-interface','supply':[{'edge':'artery','direction':1}],
            'drainage':[{'edge':'vein','direction':1}], 'storageNodes':['b'],
            'referenceSupplyMm3PerSecond':.2,'referenceDrainageMm3PerSecond':.2,'source':ASSUMED}]}
    mapping = [{'id':i,'name':name,'material':material,'target':i==2} for i,name,material in [
        (1,'Generated brain','soft_tissue'),(2,'Generated target','tumor'),
        (3,'Generated artery','vessel'),(4,'Generated vein','vessel'),(5,'Generated sinus','vessel')]]
    layers = {r['name']:{'kind': 'tumor' if r['target'] else r['material'], 'rehearsal_target':r['target'],
        'provenance':{'kind':'simulated','description':'Generated post-exposure engineering coupon; not anatomy'}} for r in mapping}
    case = CaseData(case_id='generated-vascular-coupon-v1', mri=(labels*25).astype(np.float32), affine=np.eye(4),
        compartments={r['name']:labels==r['id'] for r in mapping},
        source_refs=(SourceRef(source_id='vascular-coupon-v1', uri='synthetic://resectionlab/vascular-coupon-v1',
            license='CC0-1.0', native_frame='RAS+', provenance='simulated'),),
        unknowns=('not_a_patient','no_clinical_validation','no_functional_anatomy','no_acquired_vascular_coverage'),
        metadata={'is_synthetic':True,'input_mode':'synthetic','clinical_use_status':'research_only',
            'description':'Generated post-exposure numerical coupon; no surgical exposure or patient anatomy is represented.',
            'anatomy_layers':layers, 'surgical_adapter':{'ownershipPrecedence':[r['name'] for r in mapping]},
            'surgical_properties':{'tissuePhysicsV2':default_profile(),graph.PROPERTY:profile}})
    return case, standardize(from_case(case,mapping))


class CheckedEngine(SurgicalEngine):
    """Separate matrix-exponential solution after each committed native substep.

    It reads action-induced conductances, not the runtime derivative or solver.
    Clotting=0 in these maneuvers. Analytic clotting is checked separately.
    """
    def __init__(self, scene, **kwargs):
        super().__init__(scene, **kwargs)
        self.independent = {'steps':0,'maxStorageErrorMm3':0.,'maxWoundIntegralErrorMm3':0.,
            'maxFlowDeficitErrorMm3':0., 'regions':{r['id']:{'supplyDeficitMm3':0.,'drainageDeficitMm3':0.}
                for r in scene.properties[graph.PROPERTY]['regions']},
            'storage':[n['initialStorageMm3'] for n in scene.properties[graph.PROPERTY]['nodes']],
            'generatedMm3':0., 'boundaryInMm3':0., 'boundaryOutMm3':0.}

    def clone(self):
        result = super().clone(); result.independent = copy.deepcopy(self.independent); return result

    def _substep(self, key, op, point, quaternion, active, dt):
        events = super()._substep(key, op, point, quaternion, active, dt)
        p = self.scene.properties[graph.PROPERTY]; state = self.state[graph.PROPERTY]
        names = {node['id']:i for i,node in enumerate(p['nodes'])}; n = len(names)
        c = np.array([node['complianceMm3PerPa'] for node in p['nodes']])
        # Independent continuous linear system [V, integral(V), constant].
        matrix = np.zeros((2*n+1,2*n+1)); matrix[n:2*n,:n] = np.eye(n)
        edge_coefficients = {}
        for edge in p['edges']:
            row = state['edges'][edge['id']]; a,b = names[edge['from']],names[edge['to']]
            g = edge['conductanceMm3PerPaS']*(1-row['compressionFraction'])*(1-row['occlusionFraction']) if row['cutFace'] is None else 0.
            edge_coefficients[edge['id']] = (g,a,b)
            matrix[a,a] -= g/c[a]; matrix[b,b] -= g/c[b]; matrix[a,b] += g/c[b]; matrix[b,a] += g/c[a]
        for boundary in p['boundaries']:
            i = names[boundary['node']]; g = boundary['conductanceMm3PerPaS']
            matrix[i,i] -= g/c[i]; matrix[i,-1] += g*boundary['pressurePa']
        for port in state['ports'].values():
            i = names[port['node']]; g = port['conductanceMm3PerPaS']*(1-port['sealFraction'])*(1-port['compressionFraction'])
            matrix[i,i] -= g/c[i]
        before = np.zeros(2*n+1); before[:n] = self.independent['storage']; before[-1] = 1.
        expected = expm(matrix*dt) @ before
        for x,w in zip(QUAD_X,QUAD_W):
            pressure = (expm(matrix*(dt*(1+x)/2)) @ before)[:n]/c
            flows = {name:g*(pressure[a]-pressure[b]) for name,(g,a,b) in edge_coefficients.items()}
            for region in p['regions']:
                for title in ('supply','drainage'):
                    delivery = sum(item['direction']*flows[item['edge']] for item in region[title])
                    target = region['referenceSupplyMm3PerSecond' if title == 'supply' else 'referenceDrainageMm3PerSecond']
                    self.independent['regions'][region['id']][title+'DeficitMm3'] += float(w*dt/2*max(0.,target-max(0.,delivery)))
        for region in p['regions']:
            for title in ('supply','drainage'):
                error = abs(self.independent['regions'][region['id']][title+'DeficitMm3']-state['regions'][region['id']][title+'DeficitMm3'])
                self.independent['maxFlowDeficitErrorMm3'] = max(self.independent['maxFlowDeficitErrorMm3'],error)
                if error > 2e-7: raise ValueError('INDEPENDENT_VASCULAR_DEFICIT_CHECK_FAILED')
        actual = np.array([state['storageMm3'][name] for name in names])
        error = float(np.max(np.abs(expected[:n]-actual)))
        self.independent['maxStorageErrorMm3'] = max(self.independent['maxStorageErrorMm3'],error)
        if error > 2e-8: raise ValueError('INDEPENDENT_VASCULAR_STORAGE_CHECK_FAILED')
        emitted = 0.
        for port in state['ports'].values():
            i = names[port['node']]
            emitted += port['conductanceMm3PerPaS']*(1-port['sealFraction'])*(1-port['compressionFraction'])*expected[n+i]/c[i]
        self.independent['generatedMm3'] += float(emitted)
        for boundary in p['boundaries']:
            i = names[boundary['node']]
            signed = boundary['conductanceMm3PerPaS']*(boundary['pressurePa']*dt-expected[n+i]/c[i])
            # This coupon has only extremal 10000/0Pa boundaries. Its positive
            # semigroup and initial pressures in that range preclude reversal.
            self.independent['boundaryInMm3' if boundary['pressurePa'] else 'boundaryOutMm3'] += float(abs(signed))
        wound_error = abs(self.independent['generatedMm3']-state['generatedMm3'])
        self.independent['maxWoundIntegralErrorMm3'] = max(self.independent['maxWoundIntegralErrorMm3'],wound_error)
        if wound_error > 2e-8: raise ValueError('INDEPENDENT_VASCULAR_WOUND_CHECK_FAILED')
        self.independent['storage'] = expected[:n].tolist(); self.independent['steps'] += 1
        return events


def prepare(output):
    output = output.resolve()
    if not output.is_relative_to(ROOT/'testcases') or output.exists(): raise ValueError('Choose a NEW output directory under testcases/')
    output.mkdir(parents=True); started = time.perf_counter(); case,scene = make_case()
    labels_hash = array_digest(scene.labels); mri_hash = array_digest(case.mri)
    rows = []
    def act(e, op, point=None, *, slot='left', duration=.1, quaternion=None, **kwargs):
        action = {'slot':slot,'operation':op,'duration':duration,**kwargs}
        if point is not None: action['pose'] = {'position':point,'quaternion':quaternion or [0.,0.,0.,1.]}
        result = e.step(action)
        rows.append({'action':action,'status':result['status'],'reason':result.get('reason'),'stateId':e.state_id})
        (output/'actions.json').write_text(json.dumps(rows,indent=2)+'\n')
        if result['status'] == 'rejected': raise ValueError(str(rows[-1]))
        if point is not None and op != 'apply' and not np.allclose(e.state['slots'][slot]['position'],point,rtol=0.,atol=1e-8):
            raise ValueError('Requested endpoint not reached')
        return result
    base = CheckedEngine(scene); act(base,'configure',mechanicsRevision=2)
    branches = {'main':SurgicalEngine(scene).snapshot(),'configured-start':base.snapshot()}
    parents = {'main':None,'configured-start':{'branchId':'main','position':0}}; results = []
    for name,y in [('preserve-flow',None),('occlude-artery',6.),('occlude-vein',10.),('transect-and-seal',6.)]:
        e = base.clone()
        if y is not None:
            outside = [4.5,y,-5.]; contact = [4.5,y,11.8]
            if name == 'transect-and-seal':
                q = [0.,0.,2**-.5,2**-.5]
                act(e,'exchange',instrument='micro_scissors'); act(e,'place',outside,quaternion=q,
                    parameters={'jawOpeningMm':0.,'forceN':0.,'depthMm':1.})
                act(e,'apply',contact,quaternion=q,duration=1.)
                act(e,'deactivate'); act(e,'inspect',duration=2.); act(e,'withdraw',outside,quaternion=q,duration=1.)
            act(e,'exchange',instrument='bipolar_forceps'); act(e,'place',outside,
                parameters={'jawOpeningMm':.5,'forceN':0.,'power':0.,'depthMm':1.})
            act(e,'apply',contact,duration=1.); act(e,'apply',parameters={'power':1.},duration=2.1)
            act(e,'deactivate'); act(e,'withdraw',outside,duration=1.)
        else: act(e,'inspect',duration=5.5)
        act(e,'inspect',duration=5.)
        act(e,'place',[11.,10.,-5.],slot='right',parameters={'radiusMm':1.5,'power':1.,'duty':1.})
        for point in ([11.,10.,11.2],None,[11.,10.,11.2],None): act(e,'apply',point,slot='right',duration=3.)
        act(e,'deactivate',slot='right'); act(e,'withdraw',[11.,10.,-5.],slot='right',duration=1.)
        # Freeze a common modeled horizon so STOP does not truncate one branch's
        # deficit clock early. Tools are already withdrawn during this tail.
        remaining = 30.-e.state['elapsedSeconds']
        if remaining <= 0: raise ValueError('Maneuver exceeded frozen 30-second comparison horizon')
        act(e,'inspect',duration=remaining)
        saved = e.snapshot(); restored = SurgicalEngine.restore(scene,json.loads(json.dumps(saved)),force_verify=True)
        if restored.state != e.state: raise ValueError('Canonical replay mismatch')
        if any(s['inserted'] or s['active'] for s in e.state['slots'].values()) or e.state['holds']: raise ValueError('STOP cleanup incomplete')
        m = graph.metrics(e); accounting = removed_accounting(scene,e.state)
        if name == 'preserve-flow' and (m['cutEdgeCount'] or m['occludedEdgeCount'] or m['generatedMm3']): raise ValueError('Preserved control changed vessels')
        if name.startswith('occlude') and (m['occludedEdgeCount'] != 1 or m['cutEdgeCount'] or m['generatedMm3']): raise ValueError('Intact occlusion control failed')
        if name == 'transect-and-seal' and (m['cutEdgeCount'] != 1 or m['woundPortCount'] != 2 or m['generatedMm3'] <= 0
                or any(r['bleedingActive'] for r in e.state['sources'])): raise ValueError('Two-port bleed/seal control failed')
        if accounting['targetRemovedMm3'] <= 0: raise ValueError('Maneuver did no target work')
        if abs(m['conservationResidualMm3']) > 1e-8: raise ValueError('Independent mass budget failed')
        field = sum(e.state['fluid'].values())+e.state['fluidEscapedMm3']+e.state['fluidRemovedMm3']+m['blockedOutletReservoirMm3']
        if abs(field-m['emittedMm3']) > 1e-8: raise ValueError('Independent fluid budget failed')
        for key in ('boundaryInMm3','boundaryOutMm3'):
            if abs(e.independent[key]-m[key]) > 2e-8: raise ValueError('Independent boundary ledger failed')
        branches[name] = saved; parents[name] = {'branchId':'configured-start','position':len(base.actions)}
        results.append({'branch':name,'actions':len(e.actions),'stateId':e.state_id,'removed':accounting,'vascular':m,
            'independentMatrixExponential':e.independent,'exactCanonicalReplay':True,'stopToolsWithdrawn':True,
            'outcomes':summarize_engine_outcomes(e)})
    if len({r['removed']['sourceIdsSha256'] for r in results}) != 1: raise ValueError('Maneuvers removed different source cells')
    record = {'environmentSchema':ENVIRONMENT_SCHEMA,'physicsProfile':copy.deepcopy(scene.properties['tissuePhysicsV2']),
        'mapping':scene.regions,'branches':branches,'parents':parents,'branchId':'preserve-flow','position':len(base.actions),
        'researchSetup':{'kind':'generated-vascular-numerical-control','description':'Generated post-exposure coupon, not patient anatomy; explicit RC parameters and identical CUSA contact work. No clinical validation.'}}
    destination = output/'vascular-rehearsal.ressectionlab'; save_case(case,destination,artifacts={'surgical_toolbox':record})
    if array_digest(scene.labels) != labels_hash or array_digest(case.mri) != mri_hash: raise ValueError('Source arrays changed')
    report = {'schema':'resectionlab.vascular-qualification.v1','case':str(destination),'sceneId':scene.identity,
        'sourceLabelsHash':labels_hash,'mriHash':mri_hash,'sourceArraysPreserved':True,'generatedNotPatient':True,
        'branches':results,'comparisonHorizonSeconds':30.,'validation':{'A':'numerical RC/source/fluid/replay controls','B':'unvalidated',
            'C':'unsupported generated coupon','D':'no recorded operation','E':'no outcome model'},'RL':'hold',
        'wallSeconds':time.perf_counter()-started,'sourceSha256':{name:file_sha256(ROOT/name) for name in (
            'src/resectionlab/surgical_vascular_graph.py','scripts/qualify_vascular_graph.py')}}
    (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'case':str(destination),'wallSeconds':report['wallSeconds'],
        'branches':[{k:r[k] for k in ('branch','actions','stateId')} for r in results]}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('output',type=Path); args=parser.parse_args()
    prepare(args.output)
