#!/usr/bin/env python3
"""Capture resolved native poses/BE geometry for frozen conditional sample joins."""
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

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from resectionlab.workspace_bundle import load_workspace
from resectionlab.surgical_scene import from_case
from resectionlab.surgical_environment import standardize
from resectionlab.surgical_contact_capture import ContactCaptureEngine,capture_contact_geometry
from profile_continuum_exterior import bound,sha,write
from febio_runtime import supervise

PROTOCOL=ROOT/'manifests/experiments/continuum-contact-capture-v1.json'
SOURCES=[Path(__file__),ROOT/'scripts/profile_continuum_exterior.py',ROOT/'scripts/febio_runtime.py',
    *[ROOT/'src/resectionlab'/name for name in ('surgical_contact_capture.py','surgical_continuum_capture.py',
        'surgical_engine.py','surgical_continuum.py','surgical_continuum_solve.py','surgical_deformation_v2.py',
        'surgical_cusa_v3.py','surgical_motion.py','surgical_tools.py','surgical_physics_v2.py','workspace_bundle.py','surgical_scene.py')]]


def worker(output,launch):
    output.mkdir(parents=True,exist_ok=False);p=json.loads(PROTOCOL.read_text());identity=json.loads(launch.read_text());started=time.perf_counter()
    frames=[];branches=[];payloads={}
    report={'schema':'resectionlab.native-contact-capture-result.v1','status':'started','phase':'admission',
        'sourceCommit':identity['commit'],'sourceHashes':identity['sourceHashes'],'protocolSha256':sha(PROTOCOL),
        'sourceRole':p['sourceRole'],'physicalValidationPassed':False,'physiologicalRL':'hold','largeScaleRL':'hold',
        'largerDomainPublicActionsAdmitted':False,'newAcquiredDatasetBytes':0,'trainingUpdates':0}
    def check():
        if time.perf_counter()-started>p['caps']['wallSeconds']-10:raise TimeoutError('Contact capture cooperative cap')
        if sum(f.stat().st_size for f in output.rglob('*') if f.is_file())>p['caps']['outputBytes']-65536:
            raise ValueError('Contact capture output cap')
    def check_sources():
        if identity['protocolSha256']!=sha(PROTOCOL):raise ValueError('Launch protocol changed')
        for path,digest in identity['sourceHashes'].items():
            if sha(ROOT/path)!=digest:raise ValueError('Launch source changed')
        for entry in [p['nativeCase'],p['qualifiedActionTraces'],p['fullProgramReport'],
                      *p['nativeInputs'].values()]:bound(entry)
    try:
        check_sources();full_path=bound(p['fullProgramReport']);full=json.loads(full_path.read_text())
        if not full['deterministicAllStepReplayExact'] or full['status']!='completed':raise ValueError('Full prescribed-field gate incomplete')
        samples_path=full_path.parent/'samples.json'
        if sha(samples_path)!=full['samplesSha256']:raise ValueError('Sample index changed')
        samples=json.loads(samples_path.read_text());metrics_path=full_path.parent/full['metricsFile']
        if sha(metrics_path)!=full['metricsSha256']:raise ValueError('Metric records changed')
        metrics={(row['branch'],row['step']):row for row in json.loads(gzip.decompress(metrics_path.read_bytes()))}
        case,artifacts,_=load_workspace(bound(p['nativeCase']));record=artifacts['surgical_toolbox']
        scene=standardize(from_case(case,record['mapping']),physics_profile=record['physicsProfile'])
        if scene.identity!=p['sourceSceneId']:raise ValueError('Native source scene differs')
        outcomes=json.loads(gzip.decompress(bound(p['qualifiedActionTraces']).read_bytes()))
        for branch,entry in p['nativeInputs'].items():
            check();begin=time.perf_counter();native=json.loads(gzip.decompress(bound(entry).read_bytes()))
            prefix_ticks=sum(tick['actionIndex']<p['commonActions'] for tick in native['ticks'])
            selections={};selected=[]
            for sample in samples:
                if sample['branch']!=branch:continue
                row=metrics[(branch,sample['step'])];key=(prefix_ticks+row['suffixTick'],row['withinTick'])
                if key in selections:raise ValueError('Duplicate sampled native phase')
                selections[key]=sample;selected.append(key)
            engine=ContactCaptureEngine(scene,selected_phases=selected)
            with capture_contact_geometry():
                for index,action in enumerate(record['branches'][branch]['actions']):
                    result=engine.step(action,check=check)
                    if result!=outcomes[branch]['proportional-history-compliance'][index]:
                        raise ValueError('Geometry observer changed canonical public receipt')
            if engine.snapshot()!=record['branches'][branch]:raise ValueError('Geometry observer changed full native snapshot')
            captured=engine.mechanics_inputs()
            if any(captured[key]!=native[key] for key in captured):raise ValueError('Native mechanics projection changed')
            observed=engine.contact_frames()
            if {(frame['nativeTickIndex'],frame['withinTick']) for frame in observed}!=set(selections):
                raise ValueError('Native geometry phase coverage incomplete')
            for frame in observed:
                sample=selections[(frame['nativeTickIndex'],frame['withinTick'])];row=metrics[(branch,sample['step'])]
                if (frame['actionIndex']!=row['actionIndex'] or frame['operation']!=row['operation'] or frame['slot']!=row['slot']
                        or frame['mechanics']!=native['ticks'][frame['nativeTickIndex']]
                        or frame['fixedStepSeconds']!=row['h']):raise ValueError('Native/coupled exact phase join differs')
                buffer=io.BytesIO();np.savez_compressed(buffer,sourceCells=np.asarray(frame['sourceCells'],dtype=np.int64),
                    cornerDisplacementMm=frame['cornerDisplacementMm']);content=buffer.getvalue();digest=hashlib.sha256(content).hexdigest()
                if digest not in payloads:
                    filename=f'{branch}-step-{sample["step"]:04d}-native.npz';(output/filename).write_bytes(content);payloads[digest]=filename
                metadata={key:value for key,value in frame.items() if key not in ('sourceCells','cornerDisplacementMm')}
                frames.append({**metadata,'branch':branch,'coupledStep':sample['step'],'sampleFieldsFile':sample['file'],
                    'sampleFieldsSha256':sample['sha256'],'nativeFieldsFile':payloads[digest],'nativeFieldsSha256':digest,
                    'nativeSourceCells':len(frame['sourceCells']),'tissueState':row['tissueState'],
                    'sourceOperatorIdentity':row['sourceOperatorIdentity']})
            branches.append({'branch':branch,'canonicalAllReceiptsExact':True,'canonicalFullSnapshotExact':True,
                'mechanicsProjectionExact':True,'nativeTicks':len(captured['ticks']),'sampledFrames':len(observed),
                'wallSeconds':time.perf_counter()-begin})
            report.update(phase='native-capture',branches=branches);write(output/'report.json',report)
            print(json.dumps(branches[-1]),flush=True)
        if len(frames)!=p['expectedSampleWorlds']:raise ValueError('Conditional geometry samples incomplete')
        (output/'frames.json.gz').write_bytes(gzip.compress(json.dumps(frames,sort_keys=True,allow_nan=False).encode(),mtime=0))
        check_sources();check();report.update(status='completed',phase='completed',branches=branches,
            sourceInputsUnchanged=True,sampleWorlds=len(frames),distinctNativeFieldPayloads=len(payloads),
            framesFile='frames.json.gz',framesSha256=sha(output/'frames.json.gz'),
            scope='Resolved native poses and exact conditional BE corner fields only; no larger-domain action or collision admission')
    except BaseException as error:
        report.update(status='invalid_or_incomplete',error=type(error).__name__+': '+str(error));raise
    finally:
        report.update(wallSeconds=time.perf_counter()-started,
            peakRssBytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024))
        write(output/'report.json',report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('output',type=Path)
    parser.add_argument('--worker',action='store_true');parser.add_argument('--launch',type=Path)
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
        r=json.loads((output/'result/report.json').read_text());valid=r['status']=='completed' and r['sampleWorlds']==p['expectedSampleWorlds']
    if sum(f.stat().st_size for f in output.rglob('*') if f.is_file())>p['caps']['outputBytes']:valid=False
    write(output/'acceptance.json',{'nativeGeometryCaptureComplete':valid,'largerDomainPublicActionsAdmitted':False,
        'physicalValidationPassed':False,'physiologicalRL':'hold','supervisionStatus':receipt['status']})
    if not valid:raise RuntimeError('Geometry capture failed/incomplete; evidence retained')
    print(json.dumps({'wallSeconds':r['wallSeconds'],'sampleWorlds':r['sampleWorlds']}))


if __name__=='__main__':main()
