#!/usr/bin/env python3
"""Freeze native old-state CUSA evidence at the16 existing removal transitions."""
from pathlib import Path
import argparse
import gzip
import hashlib
import io
import json
import os
import resource
import subprocess
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from resectionlab.workspace_bundle import load_workspace
from resectionlab.surgical_scene import from_case
from resectionlab.surgical_environment import standardize
from resectionlab.geometry import GeometryScene
from resectionlab.surgical_pre_effect_capture import PreEffectCaptureEngine,capture_pre_effects
from resectionlab.surgical_pre_effect_witness import pre_effect_prediction,require_native_prediction_exact,require_pre_effect_context
from profile_continuum_exterior import bound,sha,write
from febio_runtime import supervise

PROTOCOL=ROOT/'manifests/experiments/continuum-pre-effect-capture-v1.json'
SOURCES=[Path(__file__),ROOT/'scripts/profile_continuum_exterior.py',ROOT/'scripts/febio_runtime.py',
    *[ROOT/'src/resectionlab'/name for name in ('surgical_pre_effect_capture.py','surgical_pre_effect_witness.py',
        'surgical_source_contacts.py','surgical_contact_witness.py','surgical_geometry_view.py','surgical_continuum_program.py',
        'surgical_continuum_capture.py','surgical_engine.py','surgical_continuum.py','surgical_continuum_solve.py',
        'surgical_deformation_v2.py','surgical_cusa_v3.py','surgical_motion.py','geometry.py','independent_geometry_batch.py',
        'surgical_tools.py','surgical_occupancy.py','surgical_physics_v2.py','workspace_bundle.py','surgical_scene.py')]]


def worker(output,launch):
    output.mkdir(parents=True,exist_ok=False);p=json.loads(PROTOCOL.read_text());identity=json.loads(launch.read_text());started=time.perf_counter()
    frames=[];branches=[];payloads={};inputs={}
    report={'schema':'resectionlab.native-pre-effect-capture-result.v1','status':'started','phase':'admission',
        'sourceCommit':identity['commit'],'sourceHashes':identity['sourceHashes'],'protocolSha256':sha(PROTOCOL),
        'sourceRole':p['sourceRole'],'physicalValidationPassed':False,'physiologicalRL':'hold','largeScaleRL':'hold',
        'largerDomainPublicActionsAdmitted':False,'newAcquiredDatasetBytes':0,'trainingUpdates':0}
    def input_path(entry):
        path=bound(entry);inputs[str(path.relative_to(ROOT))]=entry['sha256'];return path
    def check():
        if time.perf_counter()-started>p['caps']['wallSeconds']-10:raise TimeoutError('Pre-effect capture cooperative cap')
        if sum(f.stat().st_size for f in output.rglob('*') if f.is_file())>p['caps']['outputBytes']-65536:raise ValueError('Pre-effect capture output cap')
    def check_sources():
        if identity['protocolSha256']!=sha(PROTOCOL):raise ValueError('Launch protocol changed')
        for path,digest in identity['sourceHashes'].items():
            if sha(ROOT/path)!=digest:raise ValueError('Launch source changed')
        for path,digest in inputs.items():
            if sha(ROOT/path)!=digest:raise ValueError('Frozen input changed')
    try:
        check_sources();full_path=input_path(p['fullProgramReport']);full=json.loads(full_path.read_text())
        if full['status']!='completed' or not full['deterministicAllStepReplayExact']:raise ValueError('Full prescribed-field gate incomplete')
        samples_path=input_path({'path':str((full_path.parent/'samples.json').relative_to(ROOT)),'sha256':full['samplesSha256']})
        samples={(s['branch'],s['step']):s for s in json.loads(samples_path.read_text())}
        metrics_path=input_path({'path':str((full_path.parent/full['metricsFile']).relative_to(ROOT)),'sha256':full['metricsSha256']})
        rows=[row for row in json.loads(gzip.decompress(metrics_path.read_bytes())) if row['event'] is not None]
        if (len(rows)!=p['expectedEvents'] or any(row['withinTick']!=0 or len(row['event']['removedSourceIds'])!=1
                or (row['branch'],row['step']) not in samples for row in rows)):raise ValueError('Frozen removal phase coverage differs')
        case,artifacts,_=load_workspace(input_path(p['nativeCase']));record=artifacts['surgical_toolbox']
        scene=standardize(from_case(case,record['mapping']),physics_profile=record['physicsProfile'])
        if scene.identity!=p['sourceSceneId']:raise ValueError('Native source scene differs')
        grid=GeometryScene(np.zeros(scene.shape,bool),scene.affine)
        outcomes=json.loads(gzip.decompress(input_path(p['qualifiedActionTraces']).read_bytes()))
        for branch,entry in p['nativeInputs'].items():
            check();begin=time.perf_counter();native=json.loads(gzip.decompress(input_path(entry).read_bytes()))
            prefix=sum(tick['actionIndex']<p['commonActions'] for tick in native['ticks'])
            selections={prefix+row['suffixTick']:row for row in rows if row['branch']==branch}
            if len(selections)!=p['expectedEventsByBranch'][branch]:raise ValueError('Branch event coverage differs')
            engine=PreEffectCaptureEngine(scene,selected_ticks=selections)
            report.update(phase='native-replay',activeBranch=branch,branches=branches,inputHashes=inputs);write(output/'report.json',report)
            with capture_pre_effects():
                for index,action in enumerate(record['branches'][branch]['actions']):
                    if engine.step(action,check=check)!=outcomes[branch]['proportional-history-compliance'][index]:
                        raise ValueError('Pre-effect observer changed public receipt')
            if engine.snapshot()!=record['branches'][branch]:raise ValueError('Pre-effect observer changed native snapshot')
            captured=engine.mechanics_inputs()
            if any(captured[key]!=native[key] for key in captured):raise ValueError('Native mechanics projection differs')
            observed=engine.pre_effect_frames()
            if {f['nativeTickIndex'] for f in observed}!=set(selections) or len(observed)!=len(selections):raise ValueError('Native pre-effect coverage incomplete')
            for frame in observed:
                check();row=selections[frame['nativeTickIndex']];sample=samples[(branch,row['step'])];require_pre_effect_context(frame)
                if (frame['mechanics']!=native['ticks'][frame['nativeTickIndex']]
                        or any(frame[key]!=row[key] for key in ('actionIndex','slot','operation'))
                        or sorted(set(frame['removedSourceIdsAfter'])-set(frame['removedSourceIds']))!=row['event']['removedSourceIds']
                        or frame['elapsedSecondsBefore']+frame['durationSeconds']!=frame['mechanics']['timeSeconds']):
                    raise ValueError('Native/coupled pre-effect phase join differs')
                before=frame['removedSourceIds'];after=frame['removedSourceIdsAfter']
                if not set(before)<=set(after):raise ValueError('Native source removal is not monotone')
                prediction=pre_effect_prediction(scene,grid,frame,np.asarray(frame['sourceCells']),frame['cornerDisplacementMm'],check=check)
                require_native_prediction_exact(frame,prediction)
                buffer=io.BytesIO();np.savez_compressed(buffer,sourceCells=np.asarray(frame['sourceCells'],np.int64),cornerDisplacementMm=frame['cornerDisplacementMm'])
                content=buffer.getvalue();digest=hashlib.sha256(content).hexdigest()
                if digest not in payloads:
                    name=f'{branch}-event-{row["tissueState"]}-native-before.npz';(output/name).write_bytes(content);payloads[digest]=name
                metadata={key:value for key,value in frame.items() if key not in ('sourceCells','cornerDisplacementMm')}
                frames.append({**metadata,'branch':branch,'coupledStep':row['step'],'tissueStateAfter':row['tissueState'],
                    'tissueStateBefore':row['tissueState']-1,'sampleFieldsFile':sample['file'],'sampleFieldsSha256':sample['sha256'],
                    'previousArrayHash':row['previousArrayHash'],'sourceOperatorIdentityAfter':row['sourceOperatorIdentity'],
                    'nativeFieldsFile':payloads[digest],'nativeFieldsSha256':digest,'nativePredictionExact':True})
            branches.append({'branch':branch,'canonicalAllReceiptsExact':True,'canonicalFullSnapshotExact':True,'mechanicsProjectionExact':True,
                'sampledEvents':len(observed),'nativeTicks':len(captured['ticks']),'wallSeconds':time.perf_counter()-begin})
            report.update(branches=branches,sampledEvents=len(frames));write(output/'report.json',report);print(json.dumps(branches[-1]),flush=True)
        if len(frames)!=p['expectedEvents']:raise ValueError('Pre-effect event capture incomplete')
        (output/'frames.json.gz').write_bytes(gzip.compress(json.dumps(frames,sort_keys=True,allow_nan=False).encode(),mtime=0))
        check_sources();check();report.update(status='completed',phase='completed',activeBranch=None,sourceInputsUnchanged=True,
            sampleEvents=len(frames),distinctNativeFieldPayloads=len(payloads),nativeAllPredictionsExact=True,
            framesFile='frames.json.gz',framesSha256=sha(output/'frames.json.gz'),
            scope='Native old-state observations and exact native one-step CUSA oracle only; no larger-domain action admission')
    except BaseException as error:
        report.update(status='invalid_or_incomplete',error=type(error).__name__+': '+str(error));raise
    finally:
        report.update(inputHashes=inputs,wallSeconds=time.perf_counter()-started,
            peakRssBytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024));write(output/'report.json',report)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('output',type=Path);parser.add_argument('--worker',action='store_true');parser.add_argument('--launch',type=Path)
    args=parser.parse_args();output=args.output.resolve()
    if not output.is_relative_to(ROOT/'testcases'):raise ValueError('Evidence stays under testcases/')
    if args.worker:
        if args.launch is None:raise ValueError('Supervised launch required')
        worker(output,args.launch);return
    if output.exists():raise ValueError('Use fresh output; no automatic retry')
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():raise ValueError('Commit prospective capture first')
    launch=output.with_name(output.name+'-launch.json')
    if launch.exists():raise ValueError('Launch already exists')
    write(launch,{'protocolSha256':sha(PROTOCOL),'sourceHashes':{str(f.relative_to(ROOT)):sha(f) for f in SOURCES},
        'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()})
    p=json.loads(PROTOCOL.read_text());env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1')
    receipt=supervise([sys.executable,str(Path(__file__)),'--worker','--launch',str(launch),str(output/'result')],
        output,cwd=ROOT,environment=env,seconds=p['caps']['wallSeconds'],rss_bytes=p['caps']['rssBytes'])
    valid=receipt['status']=='completed'
    if valid:
        r=json.loads((output/'result/report.json').read_text());valid=r['status']=='completed' and r['sampleEvents']==p['expectedEvents'] and r['nativeAllPredictionsExact']
    if sum(f.stat().st_size for f in output.rglob('*') if f.is_file())>p['caps']['outputBytes']:valid=False
    write(output/'acceptance.json',{'nativePreEffectCaptureComplete':valid,'largerDomainPublicActionsAdmitted':False,
        'physicalValidationPassed':False,'physiologicalRL':'hold','largeScaleRL':'hold','supervisionStatus':receipt['status']})
    if not valid:raise RuntimeError('Pre-effect capture failed/incomplete; evidence retained')
    print(json.dumps({'wallSeconds':r['wallSeconds'],'sampleEvents':r['sampleEvents']}))


if __name__=='__main__':main()
