#!/usr/bin/env python3
"""Paired public programs: current-interface detailed vs history compliance."""
from pathlib import Path
from contextlib import contextmanager
from unittest.mock import patch
import argparse
import copy
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
from resectionlab.core import thaw_json,array_digest
from resectionlab.workspace_bundle import load_workspace,save_workspace
from resectionlab.imaging import save_case
from resectionlab.surgical_scene import from_case
from resectionlab.surgical_environment import standardize,SCHEMA
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab import surgical_continuum as c
from resectionlab import surgical_continuum_solve as solve
from resectionlab.simulator_qualification import file_sha256,removed_accounting
from febio_runtime import supervise

PROTOCOL=ROOT/'manifests/experiments/continuum-history-qualification-v1.json'
SOURCES=[Path(__file__),ROOT/'src/resectionlab/surgical_continuum.py',ROOT/'src/resectionlab/surgical_continuum_solve.py',
         *[ROOT/'src/resectionlab'/name for name in ['surgical_verified_cache.py','surgical_engine.py','surgical_scene.py',
           'surgical_deformation_v2.py','surgical_cusa_v3.py','surgical_motion.py','surgical_tools.py']]]


def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def category(path):
    if 'Pa2' in path:return 'stressExposurePa2Mm3S'
    if 'strainSquared' in path:return 'strainExposureMm3S'
    if 'Energy' in path or 'energy' in path or 'Nmm' in path:return 'energyNmm'
    if 'ReactionsN' in path or 'ResidualN' in path:return 'forceN'
    if 'isplacement' in path or 'deformation.' in path:return 'displacementMm'
    return 'otherFloat'


def compare_state(a,b,limits,maximum,path='state'):
    if type(a)!=type(b):raise ValueError('State type differs: '+path)
    if isinstance(a,dict):
        if set(a)!=set(b):raise ValueError('State keys differ: '+path)
        for key in a:compare_state(a[key],b[key],limits,maximum,path+'.'+key)
    elif isinstance(a,list):
        if len(a)!=len(b):raise ValueError('State length differs: '+path)
        for index,(x,y) in enumerate(zip(a,b)):compare_state(x,y,limits,maximum,path+'.'+str(index))
    elif type(a) is float:
        if not np.isfinite(a) or not np.isfinite(b):raise ValueError('Nonfinite state: '+path)
        key=category(path);error=abs(a-b)
        maximum[key]=max(maximum.get(key,0.),error)
        if error>limits[key]:raise ValueError('Physical state exceeds '+key+' tolerance: '+path)
    elif a!=b:raise ValueError('Discrete state differs: '+path)


def variant(case,record,method,radius,tensor_contraction=None):
    metadata=thaw_json(case.metadata);physics=copy.deepcopy(record['physicsProfile']);physics['brain']['radiusMm']=radius
    metadata['surgical_properties'][solve.PROPERTY]=solve.profile(method,tensor_contraction=tensor_contraction)
    metadata['surgical_properties']['tissuePhysicsV2']=physics
    derived=case.revised(metadata=metadata)
    scene=standardize(from_case(derived,record['mapping']),physics_profile=physics)
    return derived,scene


@contextmanager
def observe_history(directory,branch,events,samples,cap):
    original=c.tick
    def observed(engine,dt,diagnostic=None):
        before=engine.state.get(c.PROPERTY,{})
        gone=sorted((int(key) for key in before.get('cornerDisplacementMm',{}) if int(key) in engine.state['removed']))
        ledger=float(before.get('removedStoredEnergyNmm',0.));count=before.get('removedElementCount',0)
        removed_fields=[]
        for cell in gone:
            m=engine.scene.properties[c.PROPERTY]['materials'][str(int(engine.scene.labels.flat[cell]))]
            removed_fields.append({'cell':cell,'displacement':copy.deepcopy(before['cornerDisplacementMm'][str(cell)]),
                'memories':[copy.deepcopy(z[str(cell)]) for z in before['memoryDisplacementMm']], 'material':copy.deepcopy(m)})
        previous=getattr(engine,'_continuum_system_v1',None)
        result=original(engine,dt,diagnostic)
        current=getattr(engine,'_continuum_system_v1',None);after=engine.state.get(c.PROPERTY,{})
        if gone:
            events.append({'branchId':branch,'sourceCells':removed_fields,'affine':engine.scene.affine.tolist(),
                'energyDeltaNmm':after['removedStoredEnergyNmm']-ledger,'countDelta':after['removedElementCount']-count,
                'elapsedSeconds':engine.state['elapsedSeconds']})
        elif float(after.get('removedStoredEnergyNmm',0.))!=ledger or after.get('removedElementCount',0)!=count:
            raise ValueError('Removal export changed without a removed owned element')
        if current is not None and current is not previous:
            rows=[r for r in samples if r['branchId']==branch]
            if len(rows)>=cap:raise ValueError('Captured system cap')
            system=current[1];mesh=system['mesh'];state=engine.state[c.PROPERTY];p=engine.scene.properties[c.PROPERTY]
            cells=list(mesh.cells);labels=np.array([int(engine.scene.labels.flat[cell]) for cell in cells])
            targets=[]
            for key,cell,nodes,k in system['contacts']:
                h=engine.state['holds'][key]
                targets.append(np.subtract(h['target'],h['origin'])+np.asarray(h.get('baseDisplacements',{}).get(str(cell),[0.,0.,0.])))
            gravity=np.zeros((len(mesh.nodes),3))
            if 'patientPosition' in engine.state:
                from resectionlab.surgical_position import gravity_forces_n
                for nodes,force in zip(mesh.element_nodes,gravity_forces_n(engine,mesh.cells)):
                    for node in nodes:gravity[node]+=force/8.
            filename=f'{branch}-system-{len(rows):02d}.npz';destination=directory/filename
            np.savez_compressed(destination,cells=cells,shape=engine.scene.shape,affine=engine.scene.affine,nodes=mesh.nodes,
                elementNodes=mesh.element_nodes,free=system['free'],labels=labels,
                gravity=gravity,patientPositionJson=json.dumps(engine.state.get('patientPosition')),
                displacement=c._unpack(mesh,state['cornerDisplacementMm']).reshape((-1,3)),
                memories=np.asarray([c._unpack(mesh,z) for z in state['memoryDisplacementMm']]).reshape((len(p['relaxationSeconds']),len(mesh.nodes),3)),
                cellCorners=np.asarray([state['cornerDisplacementMm'][str(cell)] for cell in cells]),
                cellMemories=np.asarray([[z[str(cell)] for cell in cells] for z in state['memoryDisplacementMm']]),
                contactNodes=np.asarray([nodes for _,_,nodes,_ in system['contacts']],dtype=np.int64).reshape((-1,8)),
                contactK=np.asarray([k for _,_,_,k in system['contacts']]),contactTargets=np.asarray(targets).reshape((-1,3)),
                muPa=np.asarray([p['materials'][str(label)]['shearPa'] for label in labels]),
                nu=np.asarray([p['materials'][str(label)]['poissonRatio'] for label in labels]),
                branchMuPa=np.asarray([p['materials'][str(label)]['relaxationShearPa'] for label in labels]))
            samples.append({'branchId':branch,'file':filename,'sha256':file_sha256(destination),'activeSourceElements':len(cells),
                'cornerDofs':len(mesh.nodes)*3,'freeDofs':int(system['free'].sum()),'removedStoredEnergyNmm':state['removedStoredEnergyNmm'],
                'removedElementCount':state['removedElementCount'],'elapsedSeconds':engine.state['elapsedSeconds']})
        return result
    with patch.object(c,'tick',observed):yield


def worker(output,launch):
    output.mkdir(parents=True,exist_ok=False);p=json.loads(PROTOCOL.read_text());identity=json.loads(launch.read_text())
    started=time.perf_counter();rows=[];snapshots={};summary=[];maximum={};events=[];samples=[];traces={}
    report={'schema':'resectionlab.continuum-history-qualification.v1','status':'started','sourceCommit':identity['commit'],
        'sourceHashes':identity['sourceHashes'],'protocolSha256':file_sha256(PROTOCOL),'sourceRole':p['sourceRole'],
        'physicalValidationPassed':False,'patientMechanicsValidated':False,'largeScaleRL':'hold','physiologicalRL':'hold',
        'trainingUpdates':0,'newAcquiredDatasetBytes':0,'phase':'source-admission',
        'observerBranchesComplete':[],'legacyRecordings':[]}
    def check():
        if time.perf_counter()-started>p['caps']['wallSeconds']-10:raise TimeoutError('Qualification wall cap')
        if sum(f.stat().st_size for f in output.rglob('*') if f.is_file())>p['caps']['outputBytes']-65536:
            raise ValueError('Qualification storage cap; evidence retained')
    try:
        if identity['protocolSha256']!=file_sha256(PROTOCOL):raise ValueError('Launch protocol changed')
        for name,digest in identity['sourceHashes'].items():
            if file_sha256(ROOT/name)!=digest:raise ValueError('Launch source changed')
        source=ROOT/p['sourceArchive']
        if file_sha256(source)!=p['sourceArchiveSha256']:raise ValueError('Frozen acquired source differs')
        case,artifacts,workspace=load_workspace(source);record=artifacts['surgical_toolbox']
        original=standardize(from_case(case,record['mapping']),physics_profile=record['physicsProfile'])
        if original.identity!=p['sourceSceneId']:raise ValueError('Frozen source scene differs')
        mri_hash=array_digest(case.mri);label_hash=array_digest(original.labels)
        cases={};scenes={}
        for method in p['methods']:
            cases[method],scenes[method]=variant(case,record,method,p['patchRadiusMm'],p.get('tensorContraction'))
            if array_digest(scenes[method].labels)!=label_hash:raise ValueError('Source labels changed')
        fast='proportional-history-compliance';detailed='detailed'
        report['phase']='paired'
        for repeat in range(p['repetitions']):
            for branch in p['branches']:
                engines={name:SurgicalEngine(scenes[name]) for name in p['methods']};times={name:[] for name in p['methods']}
                order=p['methods'][:] if repeat%2==0 else p['methods'][::-1]
                for index,action in enumerate(record['branches'][branch]['actions']):
                    results={}
                    for method in order:
                        check();begin=time.perf_counter();results[method]=engines[method].step(action,check=check);times[method].append(time.perf_counter()-begin)
                        if results[method]['status']=='rejected':raise ValueError('Recorded program refused: '+branch+'/'+method+'/'+str(index)+': '+str(results[method]))
                        key=(branch,method)
                        if repeat==0:traces.setdefault(key,[]).append(copy.deepcopy(results[method]))
                        elif results[method]!=traces[key][index]:raise ValueError('Repeated per-action state identity or complete receipt differs')
                    for key in ['status','reason']:
                        if results[detailed][key]!=results[fast][key]:raise ValueError('Paired action receipt differs: '+str((branch,index,key)))
                    for key in ['completedFraction','elapsedSeconds','events']:
                        compare_state(results[detailed][key],results[fast][key],p['tolerances'],maximum,'receipt.'+key)
                    compare_state(engines[detailed].state,engines[fast].state,p['tolerances'],maximum)
                    if engines[detailed].state['elapsedSeconds']!=engines[fast].state['elapsedSeconds']:raise ValueError('Physical clocks differ')
                    if repeat==0 and index+1 in (4,p['commonActions']):
                        name='authored-exposure' if index+1==4 else 'retraction-start';snap=engines[fast].snapshot()
                        if name in snapshots and snapshots[name]!=snap:raise ValueError('Source common prefix differs across programs')
                        snapshots[name]=snap
                for method,engine in engines.items():
                    snap=engine.snapshot();key=(branch,method)
                    if repeat==0:snapshots[key]=snap
                    elif snap!=snapshots[key]:raise ValueError('Independent full recording repeat differs')
                    rows.append({'branch':branch,'method':method,'repetition':repeat,'actionWallSeconds':sum(times[method]),
                        'actionSamplesSeconds':times[method],'nativeActions':len(engine.actions),'stateId':engine.state_id,
                        'fullEpisode':removed_accounting(engine.scene,engine.state),
                        'removedStoredEnergyNmm':engine.state[c.PROPERTY]['removedStoredEnergyNmm'],
                        'removedElementCount':engine.state[c.PROPERTY]['removedElementCount']})
                print(json.dumps({'phase':'paired','branch':branch,'repetition':repeat,'elapsedWallSeconds':time.perf_counter()-started}),flush=True)
                report.update(pairedRuns=rows,maximumStateDifferences=maximum);write(output/'report.json',report)
        trace_path=output/'paired-action-traces.json.gz'
        trace_path.write_bytes(gzip.compress(json.dumps({branch:{method:traces[(branch,method)] for method in p['methods']} for branch in p['branches']},sort_keys=True,allow_nan=False).encode(),mtime=0))
        prefix=snapshots['retraction-start'];prefix_removed=removed_accounting(scenes[fast],prefix['checkpoint']['state'])
        for branch in p['branches']:
            report.update(phase='observer',activeBranch=branch);write(output/'report.json',report)
            engine=SurgicalEngine(scenes[fast])
            with observe_history(output,branch,events,samples,p['caps']['maximumCapturedSystemsPerBranch']):
                for index,action in enumerate(record['branches'][branch]['actions']):
                    receipt=engine.step(action,check=check)
                    if receipt['status']=='rejected':raise ValueError('Verification replay refused')
                    if receipt!=traces[(branch,fast)][index]:raise ValueError('Observed per-action identity or receipt differs')
            if engine.snapshot()!=snapshots[(branch,fast)]:raise ValueError('Observed canonical source replay differs')
            if engine.state['holds'] or any(s['active'] or s['inserted'] for s in engine.state['slots'].values()):raise ValueError('Program did not clean up tools')
            account=removed_accounting(scenes[fast],engine.state)
            if branch!='stop-at-model-limit':
                if (account['targetRemovedMm3']-prefix_removed['targetRemovedMm3']!=p['expectedAdditionalTargetMm3']
                    or account['nonTargetRemovedMm3']-prefix_removed['nonTargetRemovedMm3']!=p['expectedAdditionalNonTargetMm3']
                    or engine.state[c.PROPERTY]['removedElementCount']!=p['expectedRemovedActiveElements']
                    or engine.state[c.PROPERTY]['removedStoredEnergyNmm']<=0):raise ValueError('Expected source-owned removal/history export not exercised')
                summary.append({'branch':branch,'finalStateId':engine.state_id,'nativeActions':len(engine.actions),'fullEpisode':account,
                    'resectionDeltaMm3':{'targetRemovedMm3':account['targetRemovedMm3']-prefix_removed['targetRemovedMm3'],
                                       'nonTargetRemovedMm3':account['nonTargetRemovedMm3']-prefix_removed['nonTargetRemovedMm3']},
                    'removedStoredEnergyNmm':engine.state[c.PROPERTY]['removedStoredEnergyNmm'],
                    'removedElementCount':engine.state[c.PROPERTY]['removedElementCount'],'exactReplay':True})
            report['observerBranchesComplete'].append(branch)
            write(output/'removal-energy-events.json',events);write(output/'system-samples.json',samples)
            print(json.dumps({'phase':'observer','branch':branch,'elapsedWallSeconds':time.perf_counter()-started}),flush=True)
        write(output/'removal-energy-events.json',events);write(output/'system-samples.json',samples)
        legacy=[]
        for name,snapshot in record['branches'].items():
            report.update(phase='legacy',activeBranch=name);write(output/'report.json',report)
            replay=SurgicalEngine.restore(original,snapshot,check=check)
            if replay.snapshot()!=snapshot:raise ValueError('Original untagged recording changed: '+name)
            legacy.append({'branch':name,'stateId':replay.state_id,'exactFullSnapshot':True})
            report['legacyRecordings']=legacy
            print(json.dumps({'phase':'legacy','branch':name,'elapsedWallSeconds':time.perf_counter()-started}),flush=True)
        main=SurgicalEngine(scenes[fast]).snapshot();native_branches={'main':main,'authored-exposure':snapshots['authored-exposure'],'retraction-start':prefix,
            **{branch:snapshots[(branch,fast)] for branch in p['branches']}}
        parents={'main':None,'authored-exposure':{'branchId':'main','position':0},'retraction-start':{'branchId':'authored-exposure','position':4},
            **{branch:{'branchId':'retraction-start','position':p['commonActions']} for branch in p['branches']}}
        native_record={'environmentSchema':SCHEMA,'physicsProfile':copy.deepcopy(scenes[fast].properties['tissuePhysicsV2']),
            'mapping':scenes[fast].regions,'branches':native_branches,'parents':parents,'branchId':'short-hold','position':p['commonActions'],
            'researchSetup':{'kind':'explicit-continuum-history-solve-research','exposureActions':4,
                'description':'Same acquired source and native programs;8mm assumed artificial domain, same unvalidated material/contact/relaxation law, current retained-interface support identity and proportional distributed-history solve. Radius adequacy and patient physiology remain unqualified.',
                'sourceSha256':p['sourceArchiveSha256']}}
        destination=output/'continuum-history.ressectionlab';derived=cases[fast]
        if workspace:save_workspace(derived,destination,sources=workspace['sources'],imaging=workspace['imagingState'],episode=workspace['episode'],selection=workspace['selection'],artifacts={'surgical_toolbox':native_record},originals=workspace['originals'])
        else:save_case(derived,destination,artifacts={'surgical_toolbox':native_record})
        if array_digest(case.mri)!=mri_hash or array_digest(scenes[fast].labels)!=label_hash or file_sha256(source)!=p['sourceArchiveSha256']:
            raise ValueError('Original source arrays/archive changed')
        for name,digest in identity['sourceHashes'].items():
            if file_sha256(ROOT/name)!=digest:raise ValueError('Source changed during qualification')
        stopped=native_branches['stop-at-model-limit']
        report.update(status='completed',phase='completed',case=str(destination),sceneId=scenes[fast].identity,commonActions=p['commonActions'],branches=summary,
            stop={'finalStateId':stopped['stateId'],'nativeActions':len(stopped['actions']),'acceptedPrefixStateId':prefix['stateId'],'toolsWithdrawn':True,'clinicalSafetyClaim':False},
            pairedRuns=rows,maximumStateDifferences=maximum,sourceArraysUnchanged=True,sourceArchiveSha256=p['sourceArchiveSha256'],
            legacyRecordings=legacy,systemSamples=samples,removalEnergyEventCount=len(events),linearSolve=solve.profile(tensor_contraction=p.get('tensorContraction')),
            actionTraceFile=trace_path.name,actionTraceSha256=file_sha256(trace_path),perActionRepeatExact=True,
            materialAndBoundaryAcceptance=False,artificialRadiusStability='previous16-to-20mm stress gates remain failed')
        write(output/'report.json',report);check()
    except BaseException as error:
        report.update(status='invalid_or_incomplete',pairedRuns=rows,error=type(error).__name__+': '+str(error));raise
    finally:
        # Preserve usable source/history exports even when an observer replay
        # reaches its cooperative cap before the complete qualification gate.
        write(output/'removal-energy-events.json',events);write(output/'system-samples.json',samples)
        report.update(wallSeconds=time.perf_counter()-started,peakRssBytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024))
        write(output/'report.json',report)
    return report


def main():
    global PROTOCOL
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('output',type=Path);parser.add_argument('--worker',action='store_true');parser.add_argument('--launch',type=Path)
    parser.add_argument('--protocol',type=Path,default=PROTOCOL)
    args=parser.parse_args();output=args.output.resolve();PROTOCOL=args.protocol.resolve()
    if not output.is_relative_to(ROOT/'testcases'):raise ValueError('Evidence stays under testcases/')
    if args.worker:
        if args.launch is None:raise ValueError('Supervised launch required')
        worker(output,args.launch);return
    if output.exists():raise ValueError('Use a NEW output path; no automatic retry')
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():raise ValueError('Commit prospective source before execution')
    p=json.loads(PROTOCOL.read_text());launch=output.with_name(output.name+'-launch.json')
    if launch.exists():raise ValueError('Launch already exists')
    write(launch,{'protocolSha256':file_sha256(PROTOCOL),'sourceHashes':{str(f.relative_to(ROOT)):file_sha256(f) for f in SOURCES},
        'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()})
    environment=os.environ.copy();environment.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1')
    receipt=supervise([sys.executable,str(Path(__file__)),'--worker','--protocol',str(PROTOCOL),'--launch',str(launch),str(output/'result')],output,
        cwd=ROOT,environment=environment,seconds=p['caps']['wallSeconds'],rss_bytes=p['caps']['rssBytes'])
    valid=receipt['status']=='completed';r={}
    if valid:
        r=json.loads((output/'result/report.json').read_text());valid=r['status']=='completed' and len(r['pairedRuns'])==len(p['branches'])*len(p['methods'])*p['repetitions']
        if sum(f.stat().st_size for f in output.rglob('*') if f.is_file())>p['caps']['outputBytes']:valid=False
    write(output/'acceptance.json',{'numericalExecutionComplete':valid,'physicalValidationPassed':False,'physiologicalRL':'hold','supervisionStatus':receipt['status']})
    if not valid:raise RuntimeError('History solve qualification failed or incomplete; evidence retained')
    print(json.dumps({'case':r['case'],'maximumStateDifferences':r['maximumStateDifferences'],'wallSeconds':r['wallSeconds']}))


if __name__=='__main__':main()
