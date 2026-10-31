#!/usr/bin/env python3
"""Bounded geometry-only survey of frozen native/coupled conditional fields."""
from pathlib import Path
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
from resectionlab.geometry import GeometryScene
from resectionlab.workspace_bundle import load_workspace
from resectionlab.surgical_scene import from_case
from resectionlab.surgical_environment import standardize
from resectionlab.surgical_contact_survey import ConditionalContactSurvey,require_sample_join
from resectionlab.surgical_contact_witness import GEOMETRY_TOLERANCE_MM
from profile_continuum_exterior import bound,sha,write
from febio_runtime import supervise

PROTOCOL=ROOT/'manifests/experiments/continuum-contact-survey-v1.json'
SOURCES=[Path(__file__),ROOT/'scripts/profile_continuum_exterior.py',ROOT/'scripts/febio_runtime.py',
    *[ROOT/'src/resectionlab'/name for name in ('surgical_contact_survey.py','surgical_contact_witness.py',
        'surgical_geometry_view.py','surgical_source_contacts.py','independent_geometry_batch.py','surgical_motion.py','geometry.py',
        'surgical_tools.py','surgical_occupancy.py','surgical_physics_v2.py','workspace_bundle.py','surgical_scene.py')]]


def corner_fields(geometry,payload):
    cells=geometry['cells'];en=geometry['elementNodes'];nodes=geometry['nodes'];free=np.flatnonzero(geometry['freeDofs'])
    if (en.shape!=(len(cells),8) or nodes.ndim!=2 or nodes.shape[1]!=3 or np.any(en<0) or np.any(en>=len(nodes))
            or not np.array_equal(payload['freeIndices'],free) or payload['displacement'].shape!=(len(free),)
            or not np.isfinite(payload['displacement']).all()):raise ValueError('CONTACT_SURVEY_FIELD_SUPPORT')
    full=np.zeros(nodes.size);full[free]=payload['displacement']
    return full.reshape((-1,3))[en]


def aligned_native_field(cells,native):
    ids=native['sourceCells'];u=native['cornerDisplacementMm'];lookup={int(cell):i for i,cell in enumerate(cells)}
    if (ids.ndim!=1 or ids.dtype.kind not in 'iu' or len(np.unique(ids))!=len(ids)
            or u.shape!=(len(ids),8,3) or not np.isfinite(u).all() or any(int(i) not in lookup for i in ids)):
        raise ValueError('CONTACT_SURVEY_NATIVE_SUPPORT')
    result=np.zeros((len(cells),8,3));result[[lookup[int(i)] for i in ids]]=u
    return result


def compare_worlds(native,coupled):
    rows=[]
    if len(native['capsules'])!=len(coupled['capsules']):raise ValueError('CONTACT_SURVEY_CAPSULE_JOIN')
    for a,b in zip(native['capsules'],coupled['capsules']):
        if any(a[key]!=b[key] for key in ('slot','part','envelopeIndex','startMm','endMm','radiusMm')):
            raise ValueError('CONTACT_SURVEY_POSE_JOIN')
        row={key:a[key] for key in ('slot','part','envelopeIndex')}
        for key in ('rasterSourceIds','witnessedSourceIds','indeterminateSourceIds','forbiddenWitnessedSourceIds',
                    'forbiddenIndeterminateSourceIds','rasterForbiddenSourceIds'):
            if key in a:row[key+'Added']=sorted(set(b[key])-set(a[key]));row[key+'Lost']=sorted(set(a[key])-set(b[key]))
        rows.append(row)
    return {'capsules':rows,'postFieldCusaCandidateOrderChanged':
        native['instantaneousPostFieldCusaCandidateOrders']!=coupled['instantaneousPostFieldCusaCandidateOrders']}


def worker(output,launch,preflight):
    output.mkdir(parents=True,exist_ok=False);p=json.loads(PROTOCOL.read_text());identity=json.loads(launch.read_text());started=time.perf_counter()
    report={'schema':'resectionlab.continuum-contact-survey-result.v1','status':'started','phase':'admission',
        'sourceCommit':identity['commit'],'sourceHashes':identity['sourceHashes'],'protocolSha256':sha(PROTOCOL),
        'mode':'preflight' if preflight else 'complete-frozen-samples','physicalValidationPassed':False,
        'physiologicalRL':'hold','largeScaleRL':'hold','largerDomainPublicActionsAdmitted':False,
        'completeActionChecks':False,'cusaEffectValidated':False,'newAcquiredDatasetBytes':0,'trainingUpdates':0,
        'scope':'Conditional endpoint geometry only, prescribed native events/poses; intermediate BE phases are not separately accepted actions'}
    worlds=[];inputs={}
    def input_path(entry):
        path=bound(entry);inputs[str(path.relative_to(ROOT))]=entry['sha256'];return path
    def checked_file(directory,name,digest):
        path=(directory/name).resolve()
        if path.parent!=directory.resolve():raise ValueError('CONTACT_SURVEY_EXPORT_PATH')
        return input_path({'path':str(path.relative_to(ROOT)),'sha256':digest})
    def check():
        if time.perf_counter()-started>p['caps']['wallSeconds']-5:raise TimeoutError('Contact survey cooperative wall cap')
        if sum(f.stat().st_size for f in output.rglob('*') if f.is_file())>p['caps']['outputBytes']-65536:raise ValueError('Contact survey output cap')
    def check_sources():
        if identity['protocolSha256']!=sha(PROTOCOL):raise ValueError('Launch protocol changed')
        for path,digest in identity['sourceHashes'].items():
            if sha(ROOT/path)!=digest:raise ValueError('Launch source changed')
        for path,digest in inputs.items():
            if sha(ROOT/path)!=digest:raise ValueError('Frozen evidence changed')
    try:
        check_sources()
        if p['geometryToleranceMm']!=GEOMETRY_TOLERANCE_MM:raise ValueError('Frozen geometry tolerance differs')
        full_path=input_path(p['fullProgramReport']);full=json.loads(full_path.read_text())
        if full['status']!='completed' or not full['deterministicAllStepReplayExact']:raise ValueError('Full input qualification incomplete')
        samples=json.loads(checked_file(full_path.parent,'samples.json',full['samplesSha256']).read_text())
        metrics=json.loads(gzip.decompress(checked_file(full_path.parent,full['metricsFile'],full['metricsSha256']).read_bytes()))
        rows={(r['branch'],r['step']):r for r in metrics}
        capture_path=input_path(p['nativeCaptureReport']);capture=json.loads(capture_path.read_text())
        accepted=json.loads(input_path(p['nativeCaptureAcceptance']).read_text())
        if (capture['status']!='completed' or not capture['sourceInputsUnchanged'] or not accepted['nativeGeometryCaptureComplete']
                or capture['protocolSha256']!=p['nativeCaptureProtocolSha256'] or capture['sampleWorlds']!=p['expectedSampleWorlds']):
            raise ValueError('Native geometry capture incomplete')
        frames=json.loads(gzip.decompress(checked_file(capture_path.parent,capture['framesFile'],capture['framesSha256']).read_bytes()))
        frame_by_key={(f['branch'],f['coupledStep']):f for f in frames}
        if (len(frame_by_key)!=len(frames) or len(frames)!=p['expectedSampleWorlds'] or len(samples)!=p['expectedSampleWorlds']
                or set(frame_by_key)!={(s['branch'],s['step']) for s in samples}):raise ValueError('Conditional sample coverage incomplete')
        for sample in samples:require_sample_join(frame_by_key[(sample['branch'],sample['step'])],sample,rows[(sample['branch'],sample['step'])])
        case,artifacts,_=load_workspace(input_path(p['nativeCase']));record=artifacts['surgical_toolbox']
        scene=standardize(from_case(case,record['mapping']),physics_profile=record['physicsProfile'])
        if scene.identity!=p['sourceSceneId']:raise ValueError('Native source scene differs')
        grid=GeometryScene(np.zeros(scene.shape,bool),scene.affine)
        if preflight:samples=[samples[index] for index in p['preflightSampleIndices']]
        setup=time.perf_counter()-started;geometries={}
        for sample in samples:
            check();begin=time.perf_counter();key=(sample['branch'],sample['step']);frame=frame_by_key[key];row=rows[key]
            report.update(phase='query',activeWorld={'branch':sample['branch'],'step':sample['step']},worlds=worlds,inputHashes=inputs)
            write(output/'report.json',report)
            geometry_path=input_path(sample['geometry'])
            if geometry_path not in geometries:
                with np.load(geometry_path,allow_pickle=False) as f:geometries[geometry_path]={k:f[k] for k in ('cells','shape','affine','nodes','elementNodes','freeDofs')}
            g=geometries[geometry_path]
            if not np.array_equal(g['shape'],scene.shape) or not np.array_equal(g['affine'],scene.affine):raise ValueError('Source field frame differs')
            field_path=checked_file(full_path.parent,sample['file'],sample['sha256'])
            with np.load(field_path,allow_pickle=False) as f:d={k:f[k] for k in ('displacement','freeIndices','h')}
            if d['h'].shape!=() or float(d['h'])!=row['h']:raise ValueError('CONTACT_SURVEY_FIELD_PHASE')
            coupled=corner_fields(g,d)
            native_path=checked_file(capture_path.parent,frame['nativeFieldsFile'],frame['nativeFieldsSha256'])
            with np.load(native_path,allow_pickle=False) as f:native=aligned_native_field(g['cells'],{k:f[k] for k in ('sourceCells','cornerDisplacementMm')})
            result={};timings={}
            for name,u in [('native8',native),('coupled20',coupled)]:
                t=time.perf_counter();survey=ConditionalContactSurvey(scene,grid,frame,g['cells'],u,check=check)
                result[name]=survey.run();timings[name]=time.perf_counter()-t
            difference=compare_worlds(result['native8'],result['coupled20'])
            world={'branch':sample['branch'],'step':sample['step'],'tissueState':row['tissueState'],
                'actionIndex':row['actionIndex'],'operation':row['operation'],'slot':row['slot'],
                'withinTick':row['withinTick'],'atCommittedTickEnd':frame['atCommittedTickEnd'],
                'conditionalSampleSha256':sample['sha256'],'nativeFieldsSha256':frame['nativeFieldsSha256'],
                'models':result,'difference':difference,'queryWallSeconds':timings,'wallSeconds':time.perf_counter()-begin}
            name=f'{sample["branch"]}-step-{sample["step"]:04d}.json';write(output/name,world)
            worlds.append({'branch':sample['branch'],'step':sample['step'],'file':name,'sha256':sha(output/name),
                'wallSeconds':world['wallSeconds'],'queryWallSeconds':timings})
            report.update(worlds=worlds,sampleWorlds=len(worlds),inputHashes=inputs);write(output/'report.json',report)
            print(json.dumps(worlds[-1]),flush=True)
        check_sources();check()
        report.update(status='completed',phase='completed',sourceInputsUnchanged=True,inputHashes=inputs,
            activeWorld=None,
            preflightComplete=bool(preflight),conditionalSurveyComplete=not preflight,
            sampleWorlds=len(worlds),worlds=worlds,setupWallSeconds=setup,
            projectedFullSeconds=setup+max(w['wallSeconds'] for w in worlds)*p['expectedSampleWorlds'],
            sampleSelection='frozen preflight indices' if preflight else 'all26frozen samples')
    except BaseException as error:
        report.update(status='invalid_or_incomplete',error=type(error).__name__+': '+str(error));raise
    finally:
        report.update(worlds=worlds,inputHashes=inputs,sampleWorlds=len(worlds),wallSeconds=time.perf_counter()-started,
            peakRssBytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024))
        write(output/'report.json',report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('output',type=Path);parser.add_argument('--preflight',action='store_true')
    parser.add_argument('--worker',action='store_true');parser.add_argument('--launch',type=Path)
    args=parser.parse_args();output=args.output.resolve()
    if not output.is_relative_to(ROOT/'testcases'):raise ValueError('Evidence stays under testcases/')
    if args.worker:
        if args.launch is None:raise ValueError('Supervised launch required')
        worker(output,args.launch,args.preflight);return
    if output.exists():raise ValueError('Use fresh output; no automatic retry')
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():raise ValueError('Commit prospective survey first')
    launch=output.with_name(output.name+'-launch.json')
    if launch.exists():raise ValueError('Launch already exists')
    write(launch,{'protocolSha256':sha(PROTOCOL),'sourceHashes':{str(f.relative_to(ROOT)):sha(f) for f in SOURCES},
        'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()})
    p=json.loads(PROTOCOL.read_text());env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1')
    command=[sys.executable,str(Path(__file__)),'--worker','--launch',str(launch),str(output/'result')]
    if args.preflight:command.append('--preflight')
    receipt=supervise(command,output,cwd=ROOT,environment=env,seconds=p['caps']['wallSeconds'],rss_bytes=p['caps']['rssBytes'])
    valid=receipt['status']=='completed'
    if valid:
        r=json.loads((output/'result/report.json').read_text());valid=r['status']=='completed' and (r['preflightComplete'] if args.preflight else r['conditionalSurveyComplete'])
    if sum(f.stat().st_size for f in output.rglob('*') if f.is_file())>p['caps']['outputBytes']:valid=False
    write(output/'acceptance.json',{'preflightComplete':valid and args.preflight,'conditionalSurveyComplete':valid and not args.preflight,
        'largerDomainPublicActionsAdmitted':False,'physicalValidationPassed':False,'physiologicalRL':'hold','largeScaleRL':'hold',
        'supervisionStatus':receipt['status']})
    if not valid:raise RuntimeError('Conditional geometry query failed/incomplete; evidence retained')
    print(json.dumps({'wallSeconds':r['wallSeconds'],'sampleWorlds':r['sampleWorlds'],'projectedFullSeconds':r['projectedFullSeconds']}))


if __name__=='__main__':main()
