#!/usr/bin/env python3
"""Two fresh state-driven conditional histories and independent export checks."""
from pathlib import Path
import argparse
import gc
import gzip
import hashlib
import io
import json
import os
import resource
import subprocess
import sys
import time
import weakref

import numpy as np
import scipy

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from resectionlab.surgical_closed_loop import CusaContinuumEpisode,CoupledPreparation,hold_contacts
from resectionlab.surgical_continuum_topology import SourceDomain
from resectionlab.surgical_corridor_research import prepare_corridor_domain,create_corridor_episode
from resectionlab.surgical_continuum_program import contact_matrix,field_hash,record_hash
from closed_loop_source import seed
from qualify_continuum_exterior import field_metrics
from profile_continuum_closed_loop import SOURCES as PROFILE_SOURCES
from profile_continuum_exterior import bound,sha,write
from febio_runtime import supervise

PROTOCOL=ROOT/'manifests/experiments/continuum-closed-loop-pilot-v3.json'
CHECKER=ROOT/'scripts/check_continuum_closed_loop.py'
SOURCES=[Path(__file__),CHECKER,ROOT/'scripts/continuum_quadrature.py',*[ROOT/'src/resectionlab'/n for n in ('surgical_continuum_corridor.py','surgical_corridor_research.py')],*PROFILE_SOURCES]


class CappedOutput:
    """Own every worker file; enforce the byte cap before each mutation."""
    def __init__(self,directory,limit):
        self.directory=directory;self.limit=limit;self.sizes={};self.total=0
        if any(directory.iterdir()):raise ValueError('Output accounting requires an empty directory')

    def put(self,path,content):
        if path.parent!=self.directory or not isinstance(content,bytes):raise ValueError('Unowned output write')
        size=len(content);total=self.total-self.sizes.get(path.name,0)+size
        if total>self.limit:raise ValueError('Closed-loop pilot output cap')
        old=self.sizes.get(path.name,0)
        try:
            if path.write_bytes(content)!=size:raise OSError('Incomplete artifact write')
        except BaseException:
            actual=path.stat().st_size if path.exists() else 0
            self.total=self.total-old+actual
            if path.exists():self.sizes[path.name]=actual
            else:self.sizes.pop(path.name,None)
            raise
        self.sizes[path.name]=size;self.total=total

    def json(self,path,value):self.put(path,(json.dumps(value,indent=2,allow_nan=False)+'\n').encode())

    def audit(self):
        actual={str(p.relative_to(self.directory)):p.stat().st_size for p in self.directory.rglob('*') if p.is_file()}
        if actual!=self.sizes or sum(actual.values())!=self.total or self.total>self.limit:
            raise ValueError('Closed-loop output accounting changed')


def summary(episode):
    s=episode.state
    return {'arrayHash':field_hash(s['u'],s['z']),'stateHash':episode.state_hash(),'energyNmm':s['energyNmm'],
        'initialEnergyNmm':s['initialEnergyNmm'],'removedEnergyNmm':s['removedEnergyNmm'],'contactJumpNmm':s['contactJumpNmm'],
        'workNmm':s['workNmm'],'lossNmm':s['lossNmm'],'exposure':s['exposure'].tolist(),
        'newRemovedSourceIds':sorted(s['sourceLedger'].removed_source_ids),'removedSourceIds':s['removed'][:],
        'cusa':s['cusa'],'elapsedSeconds':s['elapsedSeconds'],'remainderSeconds':s['remainderSeconds'],
        'fixedSteps':s['fixedSteps'],'ticks':s['ticks'],'memoryNonzero':bool(np.max(np.abs(s['z']))>0)}


def worker(output,launch):
    output.mkdir(parents=True,exist_ok=False);p=json.loads(PROTOCOL.read_text());identity=json.loads(launch.read_text());started=time.perf_counter()
    artifacts=CappedOutput(output,p['caps']['outputBytes']-65536)
    inputs={};metrics=[];samples=[];ticks=[];passes=[];geometry_entries={};payloads={};digests={};replay={'records':[],'ticks':[]}
    report={'schema':'resectionlab.closed-loop-pilot-result.v1','status':'started','phase':'admission',
        'allStepReplayExact':False,'outputAccounting':'prewrite-owned-byte-ledger-and-final-filesystem-audit',
        'sourceCommit':identity['commit'],'sourceHashes':identity['sourceHashes'],'protocolSha256':sha(PROTOCOL),
        'physicalValidationPassed':False,'physiologicalRL':'hold','largeScaleRL':'hold','largerDomainPublicActionsAdmitted':False,
        'newAcquiredDatasetBytes':0,'trainingUpdates':0,'numpyVersion':np.__version__,'scipyVersion':scipy.__version__,
        'sourceRole':p['sourceRole'],
        'numericalThreads':{key:os.environ.get(key) for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS')},
        'algebraicCorePolicy':p['algebraicCorePolicy'],
        'scope':p['scope']}
    def check():
        if time.perf_counter()-started>p['workerWallSeconds']-5:raise TimeoutError('Closed-loop pilot cooperative cap')
    def frozen(entry):
        path=bound(entry);inputs[str(path.relative_to(ROOT))]=entry['sha256'];return path
    def check_sources():
        if sha(PROTOCOL)!=identity['protocolSha256']:raise ValueError('Launch protocol changed')
        for path,digest in identity['sourceHashes'].items():
            if sha(ROOT/path)!=digest:raise ValueError('Launch source changed')
        for path,digest in inputs.items():
            if sha(ROOT/path)!=digest:raise ValueError('Frozen input changed')
    def export_geometry(stage,operator_identity,pass_index):
        if operator_identity in geometry_entries:return geometry_entries[operator_identity]
        if pass_index:raise ValueError('Fresh replay generated a different source operator')
        name=f'geometry-{len(geometry_entries):03d}.npz';g=stage['geometry']
        buffer=io.BytesIO();np.savez_compressed(buffer,**g,coreFreeDofs=stage['coreFreeDofs']);artifacts.put(output/name,buffer.getvalue())
        entry={'file':name,'sha256':sha(output/name)};geometry_entries[operator_identity]=entry;return entry
    def export_fields(fields,prefix):
        buffer=io.BytesIO();np.savez_compressed(buffer,**fields);content=buffer.getvalue();digest=hashlib.sha256(content).hexdigest()
        if digest not in payloads:
            name=prefix+'.npz';artifacts.put(output/name,content);payloads[digest]=name
        return payloads[digest],digest
    def persist_records():
        artifacts.put(output/'steps.json.gz',gzip.compress(json.dumps(metrics,sort_keys=True,allow_nan=False).encode(),mtime=0))
        artifacts.json(output/'samples.json',samples);artifacts.json(output/'replay-digests.json',replay)
        report.update(allStepCount=len(metrics),sampleWorlds=len(samples),verifiedReplayRecords=len(replay['records']),
            verifiedReplayTicks=len(replay['ticks']),geometryEntries=geometry_entries,
            metricsFile='steps.json.gz',metricsSha256=sha(output/'steps.json.gz'),samplesFile='samples.json',samplesSha256=sha(output/'samples.json'),
            replayFile='replay-digests.json',replaySha256=sha(output/'replay-digests.json'))
    try:
        check_sources();profile=json.loads(frozen(p['costProfileReport']).read_text());accepted=json.loads(frozen(p['costProfileAcceptance']).read_text())
        if profile['status']!='completed' or not profile['sourceInputsUnchanged']:raise ValueError('Incomplete measured source cost profile')
        if p['algebraicCorePolicy']=='conservative-corridor-retained':
            independent=json.loads(frozen(p['costProfileIndependent']).read_text())
            if (not accepted['corridorCostProfileComplete'] or accepted['supervisionStatus']!='completed'
                    or accepted['independentSupervisionStatus']!='completed' or independent['status']!='passed'
                    or not profile['hypothetical']['exactRetainedExteriorAndBothCouplings']
                    or profile['rootCoreFreeDofsSha256']!=p['rootCoreFreeDofsSha256']
                    or profile['corridorSourceIdsSha256']!=p['corridorSourceIdsSha256']
                    or profile['estimatedTwoPassWorkerSeconds']>p['workerWallSeconds']-10):
                raise ValueError('Measured conservative corridor admission missing')
            numerical={path:digest for path,digest in profile['sourceHashes'].items() if path.startswith('src/resectionlab/')}
            if not all('src/resectionlab/'+name in numerical for name in ('surgical_continuum_corridor.py','surgical_closed_loop.py','surgical_continuum_exterior.py')):
                raise ValueError('Missing measured numerical source identity')
            for path,digest in numerical.items():frozen({'path':path,'sha256':digest})
        else:
            if not accepted['sourceCostProfileComplete'] or profile['selectedFastestPolicy']!=p['algebraicCorePolicy']:raise ValueError('Measured source cost admission missing')
            measured=next(v for v in profile['variants'] if v['policy']==p['algebraicCorePolicy'])
            if measured['rootCoreFreeDofsSha256']!=p['rootCoreFreeDofsSha256']:raise ValueError('Profiled algebraic core differs')
        frozen(p['independentQuadrature']);frozen(p['independentChecker'])
        for name in ('baselineFailureReport','baselineFailureDiagnosis','referencePilotReport','referencePilotAcceptance'):
            if name in p:frozen(p[name])
        scene,grid,original,initial,alpha,times,h,common,commands,metadata=seed(p,inputs,check)
        root=original.geometry;root_free=np.flatnonzero(root['freeDofs'])
        if p['algebraicCorePolicy']=='contact-core-fresh':
            core=np.unique(contact_matrix(root,hold_contacts(scene,initial['holds']))[:,root_free].indices)
            domain=SourceDomain(root,core);fresh=True
        elif p['algebraicCorePolicy']=='original-core-retained':domain=original;fresh=False
        elif p['algebraicCorePolicy']=='conservative-corridor-retained':
            domain,selection,partition=prepare_corridor_domain(scene,original,initial,commands,p['limits'])
            owners=hashlib.sha256(partition['corridorSourceIds'].tobytes()).hexdigest()
            if owners!=p['corridorSourceIdsSha256']:raise ValueError('Current corridor owners changed')
            fresh=False;report.update(partitionSelection=selection,corridorSourceIdsSha256=owners)
        else:raise ValueError('Unqualified algebraic core policy')
        core_digest=hashlib.sha256(domain.original_core.astype(np.int64).tobytes()).hexdigest()
        if core_digest!=p['rootCoreFreeDofsSha256']:raise ValueError('Current root core changed')
        report.update(seed=metadata,rootCoreFreeDofsSha256=core_digest,inputHashes=inputs);artifacts.json(output/'report.json',report)
        def measure(g,free,u,z):
            lookup={int(cell):i for i,cell in enumerate(g['cells'])};rows=np.asarray([lookup[int(cell)] for cell in common if int(cell) in lookup],int)
            return field_metrics(g,free,u,z,rows,alpha)
        first_final=None;initial_hash=None
        for pass_index in range(p['passes']):
            begin=time.perf_counter();check();report.update(phase='initial-preparation',passIndex=pass_index);artifacts.json(output/'report.json',report)
            if p['algebraicCorePolicy']=='conservative-corridor-retained':
                episode,current_selection,current_partition=create_corridor_episode(scene,grid,original,initial,alpha,times,h,measure,p['limits'],commands,check=check)
                if current_selection!=selection or not np.array_equal(current_partition['coreFreeDofs'],domain.original_core):raise ValueError('Fresh research corridor changed')
                backend=episode.backend
            else:
                backend=CoupledPreparation(domain,alpha,times,allow_fresh_exterior_updates=fresh,check=check)
                episode=CusaContinuumEpisode(scene,grid,domain,initial,alpha,times,h,measure,p['limits'],backend=backend,check=check)
            if pass_index==0:
                s=episode.state;free=np.flatnonzero(s['stage']['geometry']['freeDofs']);geo=export_geometry(s['stage'],s['sourceOperatorIdentity'],pass_index)
                file,digest=export_fields({'freeIndices':free,'displacement':s['u'][free],'memories':s['z'][:,free],
                    'target':s['target'],'h':np.asarray(h)},'initial-fields')
                initial_hash=episode.state_hash();report['initial']={'geometry':geo,'fieldsFile':file,'fieldsSha256':digest,
                    'arrayHash':field_hash(s['u'],s['z']),'stateHash':initial_hash,'energyNmm':s['energyNmm'],
                    'contacts':copy_json(s['contacts']),'cusa':copy_json(s['cusa']),'removedSourceIds':s['removed'][:],
                    'holds':copy_json(s['holds']),'slots':copy_json(s['slots']),'parametersBySlot':copy_json(s['parametersBySlot']),
                    'elapsedSeconds':s['elapsedSeconds'],'remainderSeconds':s['remainderSeconds']}
            elif episode.state_hash()!=initial_hash:raise ValueError('Fresh replay initial causal state changed')
            pass_ticks=[]
            for index,command in enumerate(commands):
                check();report.update(phase='physical-microticks',passIndex=pass_index,activeTick=index,passes=passes);artifacts.json(output/'report.json',report)
                result=episode.advance(command);s=episode.state
                report['lastPhysicalCommit']={'passIndex':pass_index,'completedTicks':index+1,'completedFixedSteps':s['fixedSteps'],
                    'lastStateHash':result['afterStateHash'],'lastArrayHash':field_hash(s['u'],s['z'])}
                artifacts.json(output/'report.json',report);batch=episode.last_committed_batch()
                geo=export_geometry(batch['stage'],s['sourceOperatorIdentity'],pass_index)
                receipt={**result,'geometry':geo,'contacts':copy_json(s['contacts']),'cusaAfter':copy_json(s['cusa']),
                    'removedSourceIds':s['removed'][:],'arrayHash':field_hash(s['u'],s['z']),'energyNmm':s['energyNmm'],
                    'cumulativeRemovedEnergyNmm':s['removedEnergyNmm'],'cumulativeContactJumpNmm':s['contactJumpNmm'],
                    'cumulativeWorkNmm':s['workNmm'],'cumulativeLossNmm':s['lossNmm'],'exposure':s['exposure'].tolist()}
                tick_metrics=[];tick_samples=[];tick_digests={};tick_replay=[]
                for record,fields in batch['records']:
                    row={**record,'h':h,'contacts':copy_json(s['contacts'])};digest=record_hash(row);key=row['fixedStep']
                    if pass_index==0:
                        if key in digests or key in tick_digests:raise ValueError('Duplicate fixed step')
                        tick_digests[key]=digest;tick_metrics.append(row)
                        if row['withinTick'] in (0,result['fixedSteps']-1):
                            name,fingerprint=export_fields({**fields,'previousTarget':fields['target']},f'tick-{index:02d}-step-{key:03d}')
                            tick_samples.append({'tick':index,'withinTick':row['withinTick'],'fixedStep':key,'geometry':geo,
                                'fieldsFile':name,'fieldsSha256':fingerprint,'contacts':copy_json(s['contacts']),
                                'sourceOperatorIdentity':s['sourceOperatorIdentity']})
                    else:
                        if digests.get(key)!=digest:raise ValueError('Fresh full-field/scalar replay changed at step '+str(key))
                        tick_replay.append({'fixedStep':key,'recordSha256':digest,'arrayHash':row['arrayHash']})
                if pass_index==0:
                    metrics.extend(tick_metrics);samples.extend(tick_samples);digests.update(tick_digests)
                    ticks.append(receipt);report.update(ticks=ticks,sampleWorlds=len(samples));artifacts.json(output/'samples.json',samples)
                else:
                    if receipt!=ticks[index]:raise ValueError('Fresh replay changed a complete causal microtick receipt')
                    replay['records'].extend(tick_replay)
                    replay['ticks'].append(receipt)
                pass_ticks.append(receipt);del batch
                report['committedProgress']={'passIndex':pass_index,'completedTicks':index+1,'completedFixedSteps':s['fixedSteps'],
                    'lastStateHash':result['afterStateHash'],'lastArrayHash':receipt['arrayHash'],
                    'verifiedReplayRecords':len(replay['records']),'verifiedReplayTicks':len(replay['ticks']),'builds':copy_json(backend.builds)}
                if pass_index:artifacts.json(output/'replay-digests.json',replay)
                artifacts.json(output/'report.json',report)
            final=summary(episode)
            if (final['ticks']!=p['expectedTicks'] or final['fixedSteps']!=p['expectedFixedSteps'] or not final['memoryNonzero']
                    or not p['minimumNewRemovals']<=len(final['newRemovedSourceIds'])<=p['maximumNewRemovals']
                    or len(pass_ticks)!=len(commands) or not any(float(np.linalg.norm(np.subtract(t['resolvedPose']['position'],initial['slots'][t['command']['slot']]['position'])))>1e-8
                        for t in pass_ticks if t['command']['operation']=='apply')):
                raise ValueError('Frozen dynamic removal/release/relaxation/advance coverage incomplete')
            if pass_index==0:first_final=final
            elif final!=first_final:raise ValueError('Fresh replay final causal state changed')
            builds=copy_json(backend.builds)
            if p['algebraicCorePolicy']=='conservative-corridor-retained':
                events=sum(t['sourceEvent'] is not None for t in pass_ticks)
                if len(builds)!=1+events or builds[0]['retainedExterior'] or not all(b['retainedExterior'] for b in builds[1:]):
                    raise ValueError('Corridor did not retain every actual removal exterior')
            reference=weakref.ref(backend.prepared);del episode,backend;gc.collect()
            if reference() is not None:raise ValueError('Previous pass preparation remains resident')
            passes.append({'wallSeconds':time.perf_counter()-begin,'finalStateHash':final['stateHash'],'finalArrayHash':final['arrayHash'],
                'ticks':final['ticks'],'fixedSteps':final['fixedSteps'],'builds':builds,'preparationReleased':True})
            report.update(passes=passes,final=first_final);persist_records();artifacts.json(output/'report.json',report)
        if len(samples)!=p['expectedSamples'] or len(metrics)!=p['expectedFixedSteps'] or len(replay['records'])!=len(metrics):raise ValueError('Sample/replay coverage incomplete')
        persist_records()
        check_sources();check();report.update(status='completed',phase='completed',activeTick=None,sourceInputsUnchanged=True,
            allStepReplayExact=True,allStepCount=len(metrics),sampleWorlds=len(samples),distinctFieldPayloads=len(payloads),
            metricsFile='steps.json.gz',metricsSha256=sha(output/'steps.json.gz'),samplesFile='samples.json',samplesSha256=sha(output/'samples.json'),
            replayFile='replay-digests.json',replaySha256=sha(output/'replay-digests.json'),geometryEntries=geometry_entries)
    except BaseException as error:
        report.update(status='invalid_or_incomplete',error=type(error).__name__+': '+str(error))
        try:persist_records()
        except Exception as evidence_error:report['partialEvidenceError']=type(evidence_error).__name__+': '+str(evidence_error)
        raise
    finally:
        report.update(inputHashes=inputs,wallSeconds=time.perf_counter()-started,
            peakRssBytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024))
        artifacts.json(output/'report.json',report);artifacts.audit()


def copy_json(value):return json.loads(json.dumps(value,allow_nan=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('output',type=Path);parser.add_argument('--worker',action='store_true');parser.add_argument('--launch',type=Path)
    args=parser.parse_args();output=args.output.resolve()
    if not output.is_relative_to(ROOT/'testcases'):raise ValueError('Evidence stays under testcases/')
    if args.worker:
        if args.launch is None:raise ValueError('Supervised launch required')
        worker(output,args.launch);return
    if output.exists():raise ValueError('Use fresh output; no automatic retry')
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():raise ValueError('Commit prospective pilot first')
    launch=output.with_name(output.name+'-launch.json')
    if launch.exists():raise ValueError('Launch already exists')
    write(launch,{'protocolSha256':sha(PROTOCOL),'sourceHashes':{str(f.relative_to(ROOT)):sha(f) for f in SOURCES},
        'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()})
    p=json.loads(PROTOCOL.read_text());env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1')
    start=time.perf_counter();receipt=supervise([sys.executable,str(Path(__file__)),'--worker','--launch',str(launch),str(output/'result')],
        output,cwd=ROOT,environment=env,seconds=p['workerWallSeconds'],rss_bytes=p['caps']['rssBytes'])
    valid=receipt['status']=='completed';checker=None
    if valid:
        r=json.loads((output/'result/report.json').read_text());valid=r['status']=='completed' and r['allStepReplayExact'] and r['peakRssBytes']<=p['caps']['rssBytes']
    if valid:
        remaining=p['caps']['wallSeconds']-(time.perf_counter()-start)-1.
        if remaining<=1:valid=False
        else:
            checker=supervise([sys.executable,str(CHECKER),str(output/'result'),str(PROTOCOL)],output/'independent',
                cwd=ROOT,environment=env,seconds=min(p['checkerWallSeconds'],remaining),rss_bytes=p['caps']['rssBytes'])
            valid=checker['status']=='completed' and json.loads((output/'result/checker-result.json').read_text())['status']=='passed'
    if sum(f.stat().st_size for f in output.rglob('*') if f.is_file())>p['caps']['outputBytes']:valid=False
    combined=time.perf_counter()-start
    if combined>p['caps']['wallSeconds']:valid=False
    acceptance={'conditionalClosedLoopPilotComplete':valid,'largerDomainPublicActionsAdmitted':False,
        'physicalValidationPassed':False,'physiologicalRL':'hold','largeScaleRL':'hold','supervisionStatus':receipt['status'],
        'independentSupervisionStatus':None if checker is None else checker['status'],'combinedWallSeconds':combined}
    acceptance_bytes=len((json.dumps(acceptance,indent=2,allow_nan=False)+'\n').encode())
    if sum(f.stat().st_size for f in output.rglob('*') if f.is_file())+acceptance_bytes>p['caps']['outputBytes']:
        valid=False;acceptance['conditionalClosedLoopPilotComplete']=False
    write(output/'acceptance.json',acceptance)
    if not valid:raise RuntimeError('Conditional closed-loop pilot failed/incomplete; evidence retained')
    print(json.dumps({'combinedWallSeconds':time.perf_counter()-start,'workerSeconds':r['wallSeconds']}))


if __name__=='__main__':main()
