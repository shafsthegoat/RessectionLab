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
from resectionlab.surgical_continuum_program import contact_matrix,field_hash,record_hash
from closed_loop_source import seed
from qualify_continuum_exterior import field_metrics
from profile_continuum_closed_loop import SOURCES as PROFILE_SOURCES
from profile_continuum_exterior import bound,sha,write
from febio_runtime import supervise

PROTOCOL=ROOT/'manifests/experiments/continuum-closed-loop-pilot-v1.json'
CHECKER=ROOT/'scripts/check_continuum_closed_loop.py'
SOURCES=[Path(__file__),CHECKER,ROOT/'scripts/continuum_quadrature.py',*PROFILE_SOURCES]


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
    inputs={};metrics=[];samples=[];ticks=[];passes=[];geometry_entries={};payloads={};digests={};replay={'records':[],'ticks':[]}
    report={'schema':'resectionlab.closed-loop-pilot-result.v1','status':'started','phase':'admission',
        'sourceCommit':identity['commit'],'sourceHashes':identity['sourceHashes'],'protocolSha256':sha(PROTOCOL),
        'physicalValidationPassed':False,'physiologicalRL':'hold','largeScaleRL':'hold','largerDomainPublicActionsAdmitted':False,
        'newAcquiredDatasetBytes':0,'trainingUpdates':0,'numpyVersion':np.__version__,'scipyVersion':scipy.__version__,
        'sourceRole':p['sourceRole'],
        'numericalThreads':{key:os.environ.get(key) for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS')},
        'algebraicCorePolicy':p['algebraicCorePolicy'],
        'scope':p['scope']}
    def check():
        if time.perf_counter()-started>p['workerWallSeconds']-5:raise TimeoutError('Closed-loop pilot cooperative cap')
        if sum(f.stat().st_size for f in output.rglob('*') if f.is_file())>p['caps']['outputBytes']-65536:raise ValueError('Closed-loop pilot output cap')
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
        np.savez_compressed(output/name,**g,coreFreeDofs=stage['coreFreeDofs'])
        entry={'file':name,'sha256':sha(output/name)};geometry_entries[operator_identity]=entry;return entry
    def export_fields(fields,prefix):
        buffer=io.BytesIO();np.savez_compressed(buffer,**fields);content=buffer.getvalue();digest=hashlib.sha256(content).hexdigest()
        if digest not in payloads:
            name=prefix+'.npz';(output/name).write_bytes(content);payloads[digest]=name
        return payloads[digest],digest
    try:
        check_sources();profile=json.loads(frozen(p['costProfileReport']).read_text());accepted=json.loads(frozen(p['costProfileAcceptance']).read_text())
        if (profile['status']!='completed' or not profile['sourceInputsUnchanged'] or not accepted['sourceCostProfileComplete']
                or profile['selectedFastestPolicy']!=p['algebraicCorePolicy']):raise ValueError('Measured source cost admission missing')
        measured=next(v for v in profile['variants'] if v['policy']==p['algebraicCorePolicy'])
        if measured['rootCoreFreeDofsSha256']!=p['rootCoreFreeDofsSha256']:raise ValueError('Profiled algebraic core differs')
        frozen(p['independentQuadrature']);frozen(p['independentChecker'])
        scene,grid,original,initial,alpha,times,h,common,commands,metadata=seed(p,inputs,check)
        root=original.geometry;root_free=np.flatnonzero(root['freeDofs'])
        if p['algebraicCorePolicy']=='contact-core-fresh':
            core=np.unique(contact_matrix(root,hold_contacts(scene,initial['holds']))[:,root_free].indices)
            domain=SourceDomain(root,core);fresh=True
        elif p['algebraicCorePolicy']=='original-core-retained':domain=original;fresh=False
        else:raise ValueError('Unqualified algebraic core policy')
        core_digest=hashlib.sha256(domain.original_core.astype(np.int64).tobytes()).hexdigest()
        if core_digest!=p['rootCoreFreeDofsSha256']:raise ValueError('Current root core changed')
        report.update(seed=metadata,rootCoreFreeDofsSha256=core_digest,inputHashes=inputs);write(output/'report.json',report)
        def measure(g,free,u,z):
            lookup={int(cell):i for i,cell in enumerate(g['cells'])};rows=np.asarray([lookup[int(cell)] for cell in common if int(cell) in lookup],int)
            return field_metrics(g,free,u,z,rows,alpha)
        first_final=None;initial_hash=None
        for pass_index in range(p['passes']):
            begin=time.perf_counter();check();report.update(phase='initial-preparation',passIndex=pass_index);write(output/'report.json',report)
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
                check();report.update(phase='physical-microticks',passIndex=pass_index,activeTick=index,passes=passes);write(output/'report.json',report)
                result=episode.advance(command);batch=episode.last_committed_batch();s=episode.state
                geo=export_geometry(batch['stage'],s['sourceOperatorIdentity'],pass_index)
                receipt={**result,'geometry':geo,'contacts':copy_json(s['contacts']),'cusaAfter':copy_json(s['cusa']),
                    'removedSourceIds':s['removed'][:],'arrayHash':field_hash(s['u'],s['z']),'energyNmm':s['energyNmm'],
                    'cumulativeRemovedEnergyNmm':s['removedEnergyNmm'],'cumulativeContactJumpNmm':s['contactJumpNmm'],
                    'cumulativeWorkNmm':s['workNmm'],'cumulativeLossNmm':s['lossNmm'],'exposure':s['exposure'].tolist()}
                for record,fields in batch['records']:
                    row={**record,'h':h,'contacts':copy_json(s['contacts'])};digest=record_hash(row);key=row['fixedStep']
                    if pass_index==0:
                        if key in digests:raise ValueError('Duplicate fixed step')
                        digests[key]=digest;metrics.append(row)
                        if row['withinTick'] in (0,result['fixedSteps']-1):
                            name,fingerprint=export_fields({**fields,'previousTarget':fields['target']},f'tick-{index:02d}-step-{key:03d}')
                            samples.append({'tick':index,'withinTick':row['withinTick'],'fixedStep':key,'geometry':geo,
                                'fieldsFile':name,'fieldsSha256':fingerprint,'contacts':copy_json(s['contacts']),
                                'sourceOperatorIdentity':s['sourceOperatorIdentity']})
                    else:
                        if digests.get(key)!=digest:raise ValueError('Fresh full-field/scalar replay changed at step '+str(key))
                        replay['records'].append({'fixedStep':key,'recordSha256':digest,'arrayHash':row['arrayHash']})
                if pass_index==0:
                    ticks.append(receipt);report.update(ticks=ticks,sampleWorlds=len(samples));write(output/'samples.json',samples)
                else:
                    if receipt!=ticks[index]:raise ValueError('Fresh replay changed a complete causal microtick receipt')
                    replay['ticks'].append(receipt)
                pass_ticks.append(receipt);del batch
            final=summary(episode)
            if (final['ticks']!=p['expectedTicks'] or final['fixedSteps']!=p['expectedFixedSteps'] or not final['memoryNonzero']
                    or not p['minimumNewRemovals']<=len(final['newRemovedSourceIds'])<=p['maximumNewRemovals']
                    or len(pass_ticks)!=len(commands) or not any(float(np.linalg.norm(np.subtract(t['resolvedPose']['position'],initial['slots'][t['command']['slot']]['position'])))>1e-8
                        for t in pass_ticks if t['command']['operation']=='apply')):
                raise ValueError('Frozen dynamic removal/release/relaxation/advance coverage incomplete')
            if pass_index==0:first_final=final
            elif final!=first_final:raise ValueError('Fresh replay final causal state changed')
            builds=copy_json(backend.builds);reference=weakref.ref(backend.prepared);del episode,backend;gc.collect()
            if reference() is not None:raise ValueError('Previous pass preparation remains resident')
            passes.append({'wallSeconds':time.perf_counter()-begin,'finalStateHash':final['stateHash'],'finalArrayHash':final['arrayHash'],
                'ticks':final['ticks'],'fixedSteps':final['fixedSteps'],'builds':builds,'preparationReleased':True})
            report.update(passes=passes,final=first_final);write(output/'report.json',report)
        if len(samples)!=p['expectedSamples'] or len(metrics)!=p['expectedFixedSteps'] or len(replay['records'])!=len(metrics):raise ValueError('Sample/replay coverage incomplete')
        (output/'steps.json.gz').write_bytes(gzip.compress(json.dumps(metrics,sort_keys=True,allow_nan=False).encode(),mtime=0));write(output/'replay-digests.json',replay)
        check_sources();check();report.update(status='completed',phase='completed',activeTick=None,sourceInputsUnchanged=True,
            allStepReplayExact=True,allStepCount=len(metrics),sampleWorlds=len(samples),distinctFieldPayloads=len(payloads),
            metricsFile='steps.json.gz',metricsSha256=sha(output/'steps.json.gz'),samplesFile='samples.json',samplesSha256=sha(output/'samples.json'),
            replayFile='replay-digests.json',replaySha256=sha(output/'replay-digests.json'),geometryEntries=geometry_entries)
    except BaseException as error:
        report.update(status='invalid_or_incomplete',error=type(error).__name__+': '+str(error));raise
    finally:
        report.update(inputHashes=inputs,wallSeconds=time.perf_counter()-started,
            peakRssBytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024));write(output/'report.json',report)


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
    write(output/'acceptance.json',{'conditionalClosedLoopPilotComplete':valid,'largerDomainPublicActionsAdmitted':False,
        'physicalValidationPassed':False,'physiologicalRL':'hold','largeScaleRL':'hold','supervisionStatus':receipt['status'],
        'independentSupervisionStatus':None if checker is None else checker['status'],'combinedWallSeconds':combined})
    if not valid:raise RuntimeError('Conditional closed-loop pilot failed/incomplete; evidence retained')
    print(json.dumps({'combinedWallSeconds':time.perf_counter()-start,'workerSeconds':r['wallSeconds']}))


if __name__=='__main__':main()
