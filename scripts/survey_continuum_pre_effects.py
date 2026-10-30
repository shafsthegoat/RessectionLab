#!/usr/bin/env python3
"""Bounded independent-event pre-effect comparisons; no20mm action sequence."""
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

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from resectionlab.workspace_bundle import load_workspace
from resectionlab.surgical_scene import from_case
from resectionlab.surgical_environment import standardize
from resectionlab.geometry import GeometryScene
from resectionlab.surgical_pre_effect_witness import pre_event_fields,pre_effect_prediction,require_native_prediction_exact,require_pre_effect_context
from resectionlab.surgical_contact_witness import GEOMETRY_TOLERANCE_MM
from survey_continuum_contacts import aligned_native_field
from profile_continuum_exterior import bound,sha,write
from febio_runtime import supervise

PROTOCOL=ROOT/'manifests/experiments/continuum-pre-effect-survey-v1.json'
SOURCES=[Path(__file__),ROOT/'scripts/profile_continuum_exterior.py',ROOT/'scripts/febio_runtime.py',ROOT/'scripts/survey_continuum_contacts.py',
    *[ROOT/'src/resectionlab'/name for name in ('surgical_pre_effect_witness.py','surgical_source_contacts.py',
        'surgical_contact_witness.py','surgical_contact_survey.py','surgical_geometry_view.py','surgical_continuum_program.py','surgical_cusa_v3.py',
        'surgical_motion.py','geometry.py','independent_geometry_batch.py','surgical_tools.py','surgical_occupancy.py',
        'surgical_physics_v2.py','workspace_bundle.py','surgical_scene.py')]]


def require_join(frame,row,sample,native,prefix):
    require_pre_effect_context(frame);index=frame['nativeTickIndex']
    if (row['withinTick']!=0 or row['event'] is None or index!=prefix+row['suffixTick'] or index<=0
            or frame['mechanics']!=native['ticks'][index]
            or any(frame[a]!=row[b] for a,b in [('branch','branch'),('coupledStep','step'),('actionIndex','actionIndex'),
                ('slot','slot'),('operation','operation'),('previousArrayHash','previousArrayHash'),('tissueStateAfter','tissueState'),
                ('sourceOperatorIdentityAfter','sourceOperatorIdentity')])
            or frame['tissueStateBefore']!=row['tissueState']-1
            or any(frame[a]!=sample[b] for a,b in [('sampleFieldsFile','file'),('sampleFieldsSha256','sha256')])
            or frame['elapsedSecondsBefore']!=native['ticks'][index-1]['timeSeconds']
            or frame['removedSourceIds']!=native['topologies'][native['ticks'][index-1]['topologyId']]['removedSourceIds']
            or frame['removedSourceIdsAfter']!=native['topologies'][native['ticks'][index]['topologyId']]['removedSourceIds']
            or sorted(set(frame['removedSourceIdsAfter'])-set(frame['removedSourceIds']))!=row['event']['removedSourceIds']):
        raise ValueError('PRE_EFFECT_SOURCE_PHASE_JOIN')


def compare(native,coupled):
    values={key:native[key]==coupled[key] for key in ('limiter','orderedSourceIds','candidateSourceIds','cusaAfter','removedSourceIdsAfter','events')}
    values['allCusaDecisionsExact']=all(values.values())
    values['removedSourceIdsAdded']=sorted(set(coupled['removedSourceIdsAfter'])-set(native['removedSourceIdsAfter']))
    values['removedSourceIdsLost']=sorted(set(native['removedSourceIdsAfter'])-set(coupled['removedSourceIdsAfter']))
    return values


def worker(output,launch,preflight):
    output.mkdir(parents=True,exist_ok=False);p=json.loads(PROTOCOL.read_text());identity=json.loads(launch.read_text());started=time.perf_counter()
    events=[];inputs={};report={'schema':'resectionlab.continuum-pre-effect-survey-result.v1','status':'started','phase':'admission',
        'sourceCommit':identity['commit'],'sourceHashes':identity['sourceHashes'],'protocolSha256':sha(PROTOCOL),
        'mode':'preflight' if preflight else 'all-frozen-events','physicalValidationPassed':False,
        'physiologicalRL':'hold','largeScaleRL':'hold','largerDomainPublicActionsAdmitted':False,'predictedStatePropagated':False,
        'newAcquiredDatasetBytes':0,'trainingUpdates':0,
        'scope':'Independent one-step events conditioned on captured native before-ledgers; cover overlaps do not establish actual swept collisions'}
    def input_path(entry):
        path=bound(entry);inputs[str(path.relative_to(ROOT))]=entry['sha256'];return path
    def checked_file(directory,name,digest):
        path=(directory/name).resolve()
        if path.parent!=directory.resolve():raise ValueError('PRE_EFFECT_EXPORT_PATH')
        return input_path({'path':str(path.relative_to(ROOT)),'sha256':digest})
    def check():
        if time.perf_counter()-started>p['caps']['wallSeconds']-5:raise TimeoutError('Pre-effect survey cooperative cap')
        if sum(f.stat().st_size for f in output.rglob('*') if f.is_file())>p['caps']['outputBytes']-65536:raise ValueError('Pre-effect survey output cap')
    def check_sources():
        if identity['protocolSha256']!=sha(PROTOCOL):raise ValueError('Launch protocol changed')
        for path,digest in identity['sourceHashes'].items():
            if sha(ROOT/path)!=digest:raise ValueError('Launch source changed')
        for path,digest in inputs.items():
            if sha(ROOT/path)!=digest:raise ValueError('Frozen input changed')
    try:
        check_sources()
        if p['geometryToleranceMm']!=GEOMETRY_TOLERANCE_MM:raise ValueError('Frozen geometry tolerance differs')
        full_path=input_path(p['fullProgramReport']);full=json.loads(full_path.read_text())
        if full['status']!='completed' or not full['deterministicAllStepReplayExact']:raise ValueError('Full prescribed-field gate incomplete')
        samples=json.loads(checked_file(full_path.parent,'samples.json',full['samplesSha256']).read_text());sample_by_key={(s['branch'],s['step']):s for s in samples}
        metrics=json.loads(gzip.decompress(checked_file(full_path.parent,full['metricsFile'],full['metricsSha256']).read_bytes()))
        rows=[r for r in metrics if r['event'] is not None];keys={(r['branch'],r['step']) for r in rows}
        capture_path=input_path(p['nativeCaptureReport']);capture=json.loads(capture_path.read_text());accept=json.loads(input_path(p['nativeCaptureAcceptance']).read_text())
        if (capture['status']!='completed' or not capture['nativeAllPredictionsExact'] or not capture['sourceInputsUnchanged']
                or not accept['nativePreEffectCaptureComplete'] or capture['protocolSha256']!=p['nativeCaptureProtocolSha256']):
            raise ValueError('Native pre-effect gate incomplete')
        frames=json.loads(gzip.decompress(checked_file(capture_path.parent,capture['framesFile'],capture['framesSha256']).read_bytes()))
        by_key={(f['branch'],f['coupledStep']):f for f in frames}
        if (len(rows)!=p['expectedEvents'] or len(keys)!=len(rows) or len(frames)!=len(by_key) or set(by_key)!=keys):
            raise ValueError('PRE_EFFECT_EVENT_COVERAGE')
        native_inputs={name:json.loads(gzip.decompress(input_path(entry).read_bytes())) for name,entry in p['nativeInputs'].items()}
        for row in rows:
            key=(row['branch'],row['step']);native=native_inputs[row['branch']];prefix=sum(t['actionIndex']<p['commonActions'] for t in native['ticks'])
            require_join(by_key[key],row,sample_by_key[key],native,prefix)
        case,artifacts,_=load_workspace(input_path(p['nativeCase']));record=artifacts['surgical_toolbox']
        scene=standardize(from_case(case,record['mapping']),physics_profile=record['physicsProfile'])
        if scene.identity!=p['sourceSceneId']:raise ValueError('Native source scene differs')
        grid=GeometryScene(np.zeros(scene.shape,bool),scene.affine);entries={}
        for s in samples:
            if s['tissueState'] in entries and entries[s['tissueState']]!=s['geometry']:raise ValueError('PRE_EFFECT_GEOMETRY_STAGE_IDENTITY')
            entries[s['tissueState']]=s['geometry']
        geometries={}
        def geometry(stage):
            path=input_path(entries[stage])
            if stage not in geometries:
                with np.load(path,allow_pickle=False) as f:geometries[stage]={k:f[k] for k in ('cells','nodes','elementNodes','freeDofs','shape','affine')}
                if not np.array_equal(geometries[stage]['shape'],scene.shape) or not np.array_equal(geometries[stage]['affine'],scene.affine):raise ValueError('PRE_EFFECT_SOURCE_FRAME')
            return geometries[stage]
        if preflight:rows=[rows[index] for index in p['preflightEventIndices']]
        setup=time.perf_counter()-started
        for row in rows:
            check();begin=time.perf_counter();key=(row['branch'],row['step']);frame=by_key[key];sample=sample_by_key[key]
            report.update(phase='query',activeEvent={'branch':row['branch'],'step':row['step']},events=events,inputHashes=inputs);write(output/'report.json',report)
            current,old=geometry(row['tissueState']),geometry(row['tissueState']-1)
            with np.load(checked_file(full_path.parent,sample['file'],sample['sha256']),allow_pickle=False) as f:
                payload={k:f[k] for k in ('freeIndices','beforeDisplacement','beforeMemories','orphanOldNodeIndices','orphanOldDisplacement','orphanOldMemories')}
            coupled,_=pre_event_fields(current,old,payload,row)
            with np.load(checked_file(capture_path.parent,frame['nativeFieldsFile'],frame['nativeFieldsSha256']),allow_pickle=False) as f:
                native=aligned_native_field(old['cells'],{k:f[k] for k in ('sourceCells','cornerDisplacementMm')})
            predictions={};timings={}
            for name,u in [('native8',native),('coupled20',coupled)]:
                t=time.perf_counter();predictions[name]=pre_effect_prediction(scene,grid,frame,old['cells'],u,check=check);timings[name]=time.perf_counter()-t
            require_native_prediction_exact(frame,predictions['native8']);difference=compare(predictions['native8'],predictions['coupled20'])
            result={'branch':row['branch'],'step':row['step'],'tissueStateBefore':row['tissueState']-1,'tissueStateAfter':row['tissueState'],
                'previousArrayHash':row['previousArrayHash'],'nativeOracleExact':True,'oldFieldHashExact':True,
                'predictions':predictions,'comparison':difference,'queryWallSeconds':timings,'wallSeconds':time.perf_counter()-begin}
            name=f'{row["branch"]}-event-{row["tissueState"]}.json';write(output/name,result)
            events.append({'branch':row['branch'],'step':row['step'],'file':name,'sha256':sha(output/name),
                'allCusaDecisionsExact':difference['allCusaDecisionsExact'],'wallSeconds':result['wallSeconds'],'queryWallSeconds':timings})
            report.update(events=events,sampleEvents=len(events),inputHashes=inputs);write(output/'report.json',report);print(json.dumps(events[-1]),flush=True)
        check_sources();check();report.update(status='completed',phase='completed',activeEvent=None,sourceInputsUnchanged=True,
            preflightComplete=bool(preflight),conditionalEventSurveyComplete=not preflight,
            allConditionalCusaDecisionsExact=all(e['allCusaDecisionsExact'] for e in events),setupWallSeconds=setup,
            projectedFullSeconds=setup+max(e['wallSeconds'] for e in events)*p['expectedEvents'])
    except BaseException as error:
        report.update(status='invalid_or_incomplete',error=type(error).__name__+': '+str(error));raise
    finally:
        report.update(events=events,inputHashes=inputs,sampleEvents=len(events),wallSeconds=time.perf_counter()-started,
            peakRssBytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024));write(output/'report.json',report)


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
        r=json.loads((output/'result/report.json').read_text());valid=r['status']=='completed' and (r['preflightComplete'] if args.preflight else r['conditionalEventSurveyComplete'])
    if sum(f.stat().st_size for f in output.rglob('*') if f.is_file())>p['caps']['outputBytes']:valid=False
    write(output/'acceptance.json',{'preflightComplete':valid and args.preflight,'conditionalEventSurveyComplete':valid and not args.preflight,
        'largerDomainPublicActionsAdmitted':False,'physicalValidationPassed':False,'physiologicalRL':'hold','largeScaleRL':'hold','supervisionStatus':receipt['status']})
    if not valid:raise RuntimeError('Pre-effect survey failed/incomplete; evidence retained')
    print(json.dumps({'wallSeconds':r['wallSeconds'],'sampleEvents':r['sampleEvents'],'projectedFullSeconds':r['projectedFullSeconds']}))


if __name__=='__main__':main()
