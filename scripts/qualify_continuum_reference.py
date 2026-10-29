#!/usr/bin/env python3
"""Supervised offline larger-domain reference; no live engine mode or RL."""
from pathlib import Path
from dataclasses import replace
from unittest.mock import patch
import argparse
import gzip
import json
import os
import resource
import subprocess
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from resectionlab.core import semantic_digest
from resectionlab.workspace_bundle import load_workspace
from resectionlab.surgical_scene import from_case
from resectionlab.surgical_environment import standardize
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab import surgical_continuum as c, surgical_deformation_v2 as brain
from resectionlab import surgical_continuum_reference as refmod
from resectionlab.simulator_qualification import file_sha256, removed_accounting
from febio_runtime import supervise

PROTOCOL=ROOT/'manifests/experiments/continuum-contact-reference-v1.json'
SOURCES=[Path(__file__),ROOT/'src/resectionlab/surgical_continuum_reference.py',
         ROOT/'src/resectionlab/surgical_continuum.py',ROOT/'src/resectionlab/surgical_deformation_v2.py']


def write(path,value): path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def bound(row):
    path=ROOT/row['path']
    if file_sha256(path)!=row['sha256']: raise ValueError('Frozen source binding differs: '+row['path'])
    return path


def capture_targets(base, actions, expected_snapshot, *, check):
    """A transient observer of complete accepted actions; no state modification."""
    h=brain._settings(base)['fixedStepSeconds']; before_removed=base.state['removed'][:]
    before_blocked=brain._blocked(base); order=tuple((key,cell) for key,hold in sorted(base.state['holds'].items()) for cell in sorted(hold['cells']))
    engine=base.clone(); rows=[]; tick=c.tick
    def observe(current,dt,diagnostic=None):
        model=current.state['deformationModel']; n=int(np.floor((model['remainderSeconds']+dt+1e-12)/h))
        if (current.state['removed']!=before_removed or brain._blocked(current)!=before_blocked
                or tuple((k,cell) for k,hold in sorted(current.state['holds'].items()) for cell in sorted(hold['cells']))!=order):
            raise ValueError('Reference load capture changed topology/contact ownership')
        target=[]
        for key,cell in order:
            hold=current.state['holds'][key]
            target.extend(np.subtract(hold['target'],hold['origin'])+np.asarray(hold.get('baseDisplacements',{}).get(str(cell),[0.,0.,0.])))
        result=tick(current,dt,diagnostic)
        rows.extend([target[:]]*n)
        return result
    with patch.object(c,'tick',observe):
        for action in actions:
            result=engine.step(action,check=check)
            if result['status']!='accepted' or result.get('reason') is not None or result.get('completedFraction')!=1.:
                raise ValueError('Only complete accepted native load histories are supported')
    if engine.snapshot()!=expected_snapshot: raise ValueError('Observed load capture differs from original nominal probe')
    return np.asarray(rows),h,engine.state_id,order


def relative(a,b):
    if any(type(v) not in (int,float) or not np.isfinite(v) or v<0 for v in (a,b)):
        raise ValueError('Finite nonnegative magnitudes required')
    return abs(a-b)/max(a,b) if max(a,b) else 0.


def stability(rows,protocol):
    expected={(branch,radius) for branch in protocol['branches'] for radius in protocol['referenceRadiiMm']}
    if len(rows)!=len(expected) or {(r['branchId'],r['radiusMm']) for r in rows}!=expected:
        raise ValueError('Every declared reference world is required')
    outcomes=[]; limits=protocol['stability']; low,high=limits['lastTwoRadiiMm']
    fields={'reaction':'reactionNormN','fixedProbeStrainSquared':'fixedProbeStrainSquaredMm3',
        'fixedProbeStressSquared':'fixedProbeStressSquaredPa2Mm3','fixedProbeCumulativeStrainSquared':'fixedProbeCumulativeStrainSquaredMm3S',
        'fixedProbeCumulativeStressSquared':'fixedProbeCumulativeStressSquaredPa2Mm3S'}
    for branch in protocol['branches']:
        subset=[r for r in rows if r['branchId']==branch]
        if any(r.get('status') not in ('completed','refused','failed') for r in subset):
            raise ValueError('Every reference world must have a terminal status')
        if any(r.get('status')!='completed' for r in subset):
            outcomes.append({'branchId':branch,'status':'not_estimable','radiusStable':False}); continue
        if len({r['commonSourceIdsHash'] for r in subset})!=1 or len({r['loadHistoryHash'] for r in subset})!=1:
            raise ValueError('Reference comparison changed source support or load history')
        probes=[np.asarray(r['probeCenterDisplacementMm']) for r in subset]
        if any(u.dtype.kind not in 'fiu' or u.ndim!=2 or u.shape[1]!=3 or not len(u)
               or u.shape!=probes[0].shape or not np.isfinite(u).all() for u in probes):
            raise ValueError('Identical finite source-center vector fields required')
        for row in subset:
            for field in fields.values(): relative(row[field],row[field])
        a,b=[next(r for r in subset if r['radiusMm']==radius) for radius in (low,high)]
        differences={'centerDisplacementRmsDifferenceMm':float(np.sqrt(np.mean(np.sum((np.asarray(a['probeCenterDisplacementMm'])-b['probeCenterDisplacementMm'])**2,axis=1))))}
        for key in ['reaction','fixedProbeStrainSquared','fixedProbeStressSquared','fixedProbeCumulativeStrainSquared','fixedProbeCumulativeStressSquared']:
            differences[key+'RelativeDifference']=relative(a[fields[key]],b[fields[key]])
        checks={key:value<=limits['maximum'+key[0].upper()+key[1:]] for key,value in differences.items()}
        outcomes.append({'branchId':branch,'status':'scored','radiusStable':all(checks.values()),'differences':differences,'checks':checks})
    return outcomes


def export_domain(ref,engine,common,path):
    p=engine.scene.properties[c.PROPERTY]; labels=np.array([int(engine.scene.labels.flat[cell]) for cell in ref.mesh.cells])
    np.savez_compressed(path,cells=ref.mesh.cells,shape=engine.scene.shape,nodes=ref.mesh.nodes,elementNodes=ref.mesh.element_nodes,
        affine=engine.scene.affine,freeDofs=ref.free,basis=ref.basis,compliance=ref.compliance,
        contactCells=np.asarray([cell for _,cell in ref.contact_cells]),spring=ref.spring,
        alpha=ref.alpha,times=ref.times,commonCells=common,labels=labels,
        muPa=np.asarray([p['materials'][str(label)]['shearPa'] for label in labels]),
        nu=np.asarray([p['materials'][str(label)]['poissonRatio'] for label in labels]))


def measure(ref,engine,common,targets,h,trajectory,path,limits,check):
    started=time.perf_counter(); mesh=ref.mesh; indices={cell:i for i,cell in enumerate(mesh.cells)}
    if not set(common)<=set(indices): raise ValueError('Fixed source probe is not supported')
    selected=[indices[cell] for cell in common]; subset=replace(mesh,cells=tuple(common),element_nodes=mesh.element_nodes[selected])
    gradients=c.element_operators(engine.scene.affine)[2]; materials=engine.scene.properties[c.PROPERTY]['materials']
    rates=[]; reactions=[]; gradient_norms=[]; displacement_norms=[]
    for q,z,target in zip(trajectory['coordinates'],trajectory['memoryCoordinates'],targets):
        check(); u=ref.basis@q; memories=np.asarray([ref.basis@w for w in z]).reshape((len(z),len(u)))
        local=u.reshape((-1,3))[mesh.element_nodes]
        gradient=np.einsum('eai,gaj->egij',local,gradients)
        largest_gradient=float(np.linalg.norm(gradient,axis=(2,3)).max()); largest_displacement=float(np.linalg.norm(local.mean(axis=1),axis=1).max())
        reaction=(ref.spring*(target-ref.compliance@q)).reshape((-1,3)).sum(axis=0)
        if (largest_gradient>limits['maximumGlobalGradientNorm']+1e-12 or largest_displacement>limits['maximumCenterDisplacementMm']
                or np.linalg.norm(reaction)>limits['maximumReactionN']):
            raise ValueError('Reference field exceeded unchanged gradient/displacement/force domain; no projection')
        instant=c.exposure_increment(engine.scene,subset,gradients,u,list(memories),materials,1.)
        rates.append([sum(v['strainSquaredVolumeTimeMm3Seconds'] for v in instant.values()),sum(v['stressSquaredVolumeTimePa2Mm3Seconds'] for v in instant.values())])
        reactions.append(reaction); gradient_norms.append(largest_gradient); displacement_norms.append(largest_displacement)
    rates=np.asarray(rates); reactions=np.asarray(reactions)
    centers=u.reshape((-1,3))[mesh.element_nodes[selected]].mean(axis=1)
    np.savez_compressed(path,coordinates=trajectory['coordinates'],memoryCoordinates=trajectory['memoryCoordinates'],
        displacement=u.reshape((-1,3)),memories=memories.reshape((len(memories),len(mesh.nodes),3)),
        fixedProbeRates=rates,reactionsN=reactions,maxGradientNorm=gradient_norms,maxCenterDisplacementMm=displacement_norms)
    return {'status':'completed','commonSourceIdsHash':semantic_digest(common),'loadHistoryHash':semantic_digest(targets.tolist()),
        'probeCenterDisplacementMm':centers.tolist(),'reactionNormN':float(np.linalg.norm(reactions[-1])),
        'peakReactionNormN':float(np.linalg.norm(reactions,axis=1).max()),
        'fixedProbeStrainSquaredMm3':float(rates[-1,0]),'fixedProbeStressSquaredPa2Mm3':float(rates[-1,1]),
        'fixedProbeCumulativeStrainSquaredMm3S':float(rates[:,0].sum()*h),
        'fixedProbeCumulativeStressSquaredPa2Mm3S':float(rates[:,1].sum()*h),
        'maximumGradientNorm':max(gradient_norms),'maximumCenterDisplacementMm':max(displacement_norms),
        'fieldVerificationWallSeconds':time.perf_counter()-started,
        'rawFields':path.name,'rawFieldsSha256':file_sha256(path),
        **{k:float(trajectory[k]) for k in ['maxCoefficientResidualN','maxEnergyBalanceResidualNmm','dissipatedEnergyNmm','numericalDissipationNmm']}}


def worker(output,launch):
    output.mkdir(parents=True,exist_ok=False); protocol=json.loads(PROTOCOL.read_text()); started=time.perf_counter()
    identity=json.loads(launch.read_text()); rows=[]
    report={'schema':'resectionlab.continuum-contact-reference-result.v1','status':'started','sourceRole':protocol['sourceRole'],
        'protocolSha256':file_sha256(PROTOCOL),'sourceHashes':identity['sourceHashes'],'sourceCommit':identity['commit'],
        'liveModelCapsChanged':False,'completeSurgicalPlans':False,'physicalValidationPassed':False,'clinicalValidation':False,
        'physiologicalRL':'hold','trainingUpdates':0,'newAcquiredDatasetBytes':0,
        'exposureScope':'Fixed-source instantaneous fields and time integrals over the exact post-prefix load clock; complete surgical-plan and patient-outcome claims are unsupported.'}
    def check():
        if time.perf_counter()-started>protocol['caps']['wallSeconds']-10: raise TimeoutError('Reference cooperative wall cap')
        if sum(p.stat().st_size for p in output.iterdir() if p.is_file())>protocol['caps']['outputBytes']-65536:
            raise ValueError('Reference storage cap; evidence retained')
    try:
        if identity['protocolSha256']!=file_sha256(PROTOCOL): raise ValueError('Protocol changed since launch')
        for path,digest in identity['sourceHashes'].items():
            if file_sha256(ROOT/path)!=digest: raise ValueError('Source changed since launch')
        source=bound(protocol['sourceArchive']); geometry=json.loads(bound(protocol['geometryFeasibility']).read_text())
        case,artifacts,_=load_workspace(source); record=artifacts['surgical_toolbox']
        scene=standardize(from_case(case,record['mapping']),physics_profile=record['physicsProfile'])
        if scene.identity!=protocol['sourceSceneId']: raise ValueError('Source scene differs')
        original=record['branches'][protocol['prefixBranch']]
        if len(original['actions'])!=protocol['prefixActions']: raise ValueError('Prefix count differs')
        base=SurgicalEngine.restore(scene,original,check=check)
        common=sorted(base.state['deformationModel']['active']); schedules={}; schedule_meta={}
        for branch in protocol['branches']:
            program=record['branches'][branch]['actions'][:protocol['probeActions']]
            if len(program)!=protocol['probeActions'] or program[:protocol['prefixActions']]!=original['actions']:
                raise ValueError('Exact source action program differs')
            saved=json.loads(gzip.decompress(bound(protocol['nativePriorEndpoints'][branch]['snapshot4']).read_bytes()))
            target,h,state_id,order=capture_targets(base,program[protocol['prefixActions']:],saved,check=check)
            schedules[branch]=target; schedule_meta[branch]={'h':h,'steps':len(target),'nominalStateId':state_id,
                'nativeActions':protocol['probeActions'],'actionHash':semantic_digest(program),'contactOrder':[list(v) for v in order]}
        np.savez_compressed(output/'load-schedules.npz',**{branch.replace('-','_'):v for branch,v in schedules.items()})
        report.update(loadSchedules=schedule_meta,sourceSceneId=scene.identity,sourceArchiveSha256=file_sha256(source),
                      fullSourceRemoval=removed_accounting(scene,base.state),domains=[])
        for radius in protocol['referenceRadiiMm']:
            check(); cells=refmod.source_domain(base,radius,max_cells=protocol['caps']['maxSourceCells'])
            expected=next(d for d in geometry['domains'] if d['radiusMm']==radius)
            if semantic_digest(cells)!=expected['cellIdsHash']: raise ValueError('Reference source geometry differs from feasibility freeze')
            ref=refmod.build_reference(base,cells,max_dofs=protocol['caps']['maxDofs'],check=check)
            if len(ref.free)!=expected['cornerDofs']: raise ValueError('Reference DOF count differs')
            domain_path=output/f'domain-{radius:g}.npz'; export_domain(ref,base,common,domain_path)
            report['domains'].append({'radiusMm':radius,'sourceCells':len(cells),'cornerDofs':len(ref.free),
                'sourceCellsHash':semantic_digest(cells),'rawGeometry':domain_path.name,'rawGeometrySha256':file_sha256(domain_path),**ref.diagnostics})
            for branch in protocol['branches']:
                target=schedules[branch]; h=schedule_meta[branch]['h']; begin=time.perf_counter(); trajectory=refmod.integrate(ref,target,h,check=check)
                core_seconds=time.perf_counter()-begin
                row={'branchId':branch,'radiusMm':radius,'status':'started','coreIntegrationWallSeconds':core_seconds}
                rows.append(row)
                if radius==8.:
                    prior=np.load(bound(protocol['nativePriorEndpoints'][branch]['fields']),allow_pickle=False)
                    if (not np.array_equal(prior['cells'],ref.mesh.cells) or not np.array_equal(prior['elementNodes'],ref.mesh.element_nodes)
                            or not np.array_equal(prior['nodes'],ref.mesh.nodes) or not np.array_equal(prior['affine'],scene.affine)):
                        raise ValueError('Native comparison changed mesh/source nodes')
                    endpoint=(ref.basis@trajectory['coordinates'][-1]).reshape((-1,3))
                    memory=np.asarray([ref.basis@z for z in trajectory['memoryCoordinates'][-1]]).reshape(prior['memories'].shape)
                    native_u=float(np.max(np.abs(endpoint-prior['displacement']),initial=0.)); native_z=float(np.max(np.abs(memory-prior['memories']),initial=0.))
                    if native_u>protocol['numericalGates']['maximumNativeEndpointDisplacementMm'] or native_z>protocol['numericalGates']['maximumNativeEndpointMemoryMm']:
                        raise ValueError('Reference reduction disagrees with exposed native endpoint')
                    prepared=refmod.prepare_full(ref,h); timings=[]; errors=[]
                    for repeat in range(3):
                        begin=time.perf_counter(); detailed=refmod.integrate_full(ref,target,h,prepared=prepared,check=check); full_seconds=time.perf_counter()-begin
                        begin=time.perf_counter(); reduced=refmod.integrate(ref,target,h,check=check); reduced_seconds=time.perf_counter()-begin
                        u_error=float(np.max(np.abs(reduced['coordinates']@ref.basis.T-detailed['displacement']),initial=0.))
                        z_error=float(np.max(np.abs(np.einsum('tbm,nm->tbn',reduced['memoryCoordinates'],ref.basis)-detailed['memories']),initial=0.))
                        if u_error>protocol['numericalGates']['maximumFullVsReducedDisplacementMm'] or z_error>protocol['numericalGates']['maximumFullVsReducedMemoryMm']:
                            raise ValueError('Detailed/reduced history disagreement')
                        timings.append({'detailedSeconds':full_seconds,'reducedSeconds':reduced_seconds}); errors.append({'maxDisplacementMm':u_error,'maxMemoryMm':z_error})
                    row.update(nativeDisplacementErrorMm=native_u,nativeMemoryErrorMm=native_z,detailedHistoryErrors=errors,coreTimings=timings)
                    prior.close()
                    del prepared,detailed,reduced
                row.update(measure(ref,base,common,target,h,trajectory,output/f'{branch}-radius-{radius:g}.npz',protocol['numericalGates'],check))
                report['samples']=rows; write(output/'report.json',report); check()
                print(json.dumps({'radiusMm':radius,'branch':branch,'status':row['status'],'elapsedWallSeconds':time.perf_counter()-started}),flush=True)
        if file_sha256(source)!=protocol['sourceArchive']['sha256']: raise ValueError('Original archive changed')
        report.update(status='completed',samples=rows,outcomes=stability(rows,protocol),sourceArchivePreserved=True)
        write(output/'report.json',report); check()
    except BaseException as error:
        report.update(status='invalid_or_incomplete',samples=rows,error=type(error).__name__+': '+str(error)); raise
    finally:
        report.update(wallSeconds=time.perf_counter()-started,peakRssBytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024))
        write(output/'report.json',report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('output',type=Path)
    parser.add_argument('--worker',action='store_true');parser.add_argument('--launch',type=Path)
    args=parser.parse_args(); output=args.output.resolve()
    if not output.is_relative_to(ROOT/'testcases'): raise ValueError('All evidence must stay under testcases/')
    if args.worker:
        if args.launch is None: raise ValueError('Supervised launch context required')
        worker(output,args.launch); return
    if output.exists(): raise ValueError('Use a NEW output path; no automatic retry')
    protocol=json.loads(PROTOCOL.read_text())
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():
        raise ValueError('Commit the prospective source/protocol before executing acquired-source reference')
    launch=output.with_name(output.name+'-launch.json')
    if launch.exists(): raise ValueError('Launch path already exists')
    source_hashes={str(p.relative_to(ROOT)):file_sha256(p) for p in SOURCES}
    write(launch,{'protocolSha256':file_sha256(PROTOCOL),'sourceHashes':source_hashes,
                  'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()})
    environment=os.environ.copy();environment.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1')
    receipt=supervise([sys.executable,str(Path(__file__)),'--worker','--launch',str(launch),str(output/'result')],output,
        cwd=ROOT,environment=environment,seconds=protocol['caps']['wallSeconds'],rss_bytes=protocol['caps']['rssBytes'])
    valid=receipt['status']=='completed'
    if valid:
        r=json.loads((output/'result/report.json').read_text()); valid=r['status']=='completed' and len(r['samples'])==len(protocol['branches'])*len(protocol['referenceRadiiMm'])
        if sum(p.stat().st_size for p in output.rglob('*') if p.is_file())>protocol['caps']['outputBytes']:valid=False
    write(output/'acceptance.json',{'numericalExecutionComplete':valid,'physicalValidationPassed':False,'physiologicalRL':'hold',
        'supervisionStatus':receipt['status'],'protocolSha256':file_sha256(PROTOCOL)})
    if not valid: raise RuntimeError('Reference execution failed or incomplete; evidence retained')
    print(json.dumps({'outcomes':r['outcomes'],'wallSeconds':r['wallSeconds']}))


if __name__=='__main__': main()
