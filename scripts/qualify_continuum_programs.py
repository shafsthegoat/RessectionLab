#!/usr/bin/env python3
"""Full captured-input fields and exact replay, with independent source samples."""
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
import zipfile

import numpy as np
import scipy

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from resectionlab.surgical_continuum_exterior import ExteriorOperator,ExteriorCoreFactor,ExteriorHistorySolve,source_stiffness
from resectionlab.surgical_continuum_program import Episode,prepare_program,contact_key,contact_matrix,spring
from resectionlab.surgical_continuum_events import source_operator_identity
from profile_continuum_exterior import load_geometry,free_partition,bound,sha,write
from qualify_continuum_exterior import field_metrics
from febio_runtime import supervise

PROTOCOL=ROOT/'manifests/experiments/continuum-program-qualification-v1.json'
SOURCES=[Path(__file__),*[ROOT/'scripts'/name for name in ('check_continuum_programs.py',
    'profile_continuum_exterior.py','qualify_continuum_exterior.py','febio_runtime.py')],
    *[ROOT/'src/resectionlab'/name for name in ('surgical_continuum_program.py','surgical_continuum_exterior.py',
        'surgical_continuum_events.py','surgical_continuum_partition.py','surgical_continuum.py')]]


def load_stages(p):
    initial,_,partition=load_geometry(p);_,core=free_partition(initial,partition)
    path=bound(p['topologyReport']);topology=json.loads(path.read_text())
    if not topology['exactFactorReuseCandidate']:raise ValueError('Missing source topology admission')
    stages=[]
    for index,row in enumerate(topology['states']):
        if index==0:g=initial;lineage=np.arange(int(g['freeDofs'].sum()));entry=p['geometry']['reference'];i=core
        else:
            entry={'path':str((path.parent/row['geometry']['path']).relative_to(ROOT)),'sha256':row['geometry']['sha256']}
            with np.load(bound(entry),allow_pickle=False) as raw:
                g={key:raw[key] for key in initial};i=raw['coreFreeDofs'];lineage=raw['originalFreeDofForNew']
        stages.append({'geometry':g,'entry':entry,'core':i,'lineage':lineage,
            'mask':row['removedSourceIds'],'freshExterior':row['newExteriorGroup']})
    return stages


def run_pass(programs,stages,alpha,times,common,p,emit,check,progress):
    episodes={name:Episode(program,stages[0]['geometry'],alpha,times) for name,program in programs.items()}
    prepared=None;bindings={};old_lineage=None;builds=[];removed_prefix=set(next(iter(programs.values()))['prefix']['removedSourceIds'])
    for state,stage in enumerate(stages):
        check();g=stage['geometry'];free=np.flatnonzero(g['freeDofs']);core=stage['core'];lineage=stage['lineage']
        active=[name for name,program in programs.items() if state in program['segments']]
        if not active:raise ValueError('Frozen tissue stage has no episode')
        for name in active:
            if state:episodes[name].enter_geometry(g,state,programs[name]['segments'][state][0][1]['contacts'])
        begin=time.perf_counter();full=source_stiffness(g,source_order=stages[0]['geometry']['cells'],check=check)
        k=full[free][:,free].tocsr();del full
        released=None;bindings.clear()
        if stage['freshExterior']:
            if prepared is not None:
                reference=weakref.ref(prepared.operator);prepared=None;gc.collect()
                released=reference() is None
                if not released:raise ValueError('Previous exterior remains resident at source reset')
            prepared=ExteriorCoreFactor(ExteriorOperator(k,core,check=check),check=check)
        else:
            inverse={int(value):i for i,value in enumerate(old_lineage)}
            parents=np.asarray([inverse[int(value)] for value in lineage])
            prepared=prepared.rebuild(k,core,parents,check=check)
        del k
        identity=source_operator_identity(g,alpha,times,core_free_dofs=core,removed_source_ids=sorted(removed_prefix|set(stage['mask'])))
        lookup={int(cell):i for i,cell in enumerate(g['cells'])};common_rows=np.asarray([lookup[int(cell)] for cell in common])
        def measure(geometry,f,u,z):return field_metrics(geometry,f,u,z,common_rows,alpha)
        for name in active:
            ep=episodes[name];segment=programs[name]['segments'][state]
            for tick_index,tick in segment:
                check();changed=ep.bind(tick['contacts']);key=contact_key(tick['contacts'])
                if key not in bindings:
                    c=contact_matrix(g,tick['contacts'])[:,free]
                    bindings[key]=ExteriorHistorySolve(prepared.operator,c,spring(tick['contacts']),alpha,times,prepared=prepared,check=check)
                for within in range(tick['nativeFixedSteps']):
                    check();record,digest,fields=ep.advance(bindings[key],tick,tick_index,within,measure,p['tolerances'],p['domain'])
                    record['sourceOperatorIdentity']=identity
                    # Identity participates in the exact step and branch chain.
                    # Episode already chains its mechanics record; this emitted
                    # digest additionally binds source/world identity.
                    from resectionlab.surgical_continuum_program import record_hash
                    digest=record_hash(record)
                    reasons=[]
                    if ep.steps==1:reasons.append('first-step')
                    if state==0 and (tick_index,within)==(segment[-1][0],segment[-1][1]['nativeFixedSteps']-1):reasons.append('initial-stage-end')
                    if state>0 and tick_index==segment[0][0] and within==0:reasons.append('source-event')
                    if changed and not tick['contacts'] and within==0:reasons.append('release')
                    if ep.next_tick==len(programs[name]['ticks']):reasons.append('final')
                    emit(record,digest,fields,reasons,stage)
        builds.append({'tissueState':state,'freshExterior':stage['freshExterior'],
            'buildAndStageSeconds':time.perf_counter()-begin,'operator':dict(prepared.operator.stats),
            'cachedContactBindings':[{'contacts':len(value.spring)//3,'fullBasisBytes':value.full_basis.nbytes,
                'constructionSeconds':value.binding_seconds} for value in bindings.values()],
            'sourceOperatorIdentity':identity,'previousExteriorReleased':released,'activeBranches':active})
        progress(state,builds,{name:ep.steps for name,ep in episodes.items()})
        old_lineage=lineage
    bindings.clear();prepared=None;gc.collect()
    return {name:episode.summary() for name,episode in episodes.items()},builds


def worker(output,launch):
    output.mkdir(parents=True,exist_ok=False);p=json.loads(PROTOCOL.read_text());identity=json.loads(launch.read_text())
    started=time.perf_counter();samples=[];metrics=[];digests={};payloads={};passes=[]
    report={'schema':'resectionlab.continuum-full-program-result.v1','status':'started','phase':'admission',
        'sourceCommit':identity['commit'],'sourceHashes':identity['sourceHashes'],'protocolSha256':sha(PROTOCOL),
        'scope':p['scope'],'physicalValidationPassed':False,'physiologicalRL':'hold','largeScaleRL':'hold',
        'numpyVersion':np.__version__,'scipyVersion':scipy.__version__,'pythonVersion':sys.version,
        'numericalThreads':{key:os.environ.get(key) for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS')},
        'largerDomainPublicActionsAdmitted':False,'newAcquiredDatasetBytes':0,'trainingUpdates':0}
    def check():
        if time.perf_counter()-started>p['caps']['wallSeconds']-15:raise TimeoutError('Full program cooperative wall cap')
        if sum(f.stat().st_size for f in output.rglob('*') if f.is_file())>p['caps']['outputBytes']-65536:
            raise ValueError('Full program output cap')
    def check_sources():
        if identity['protocolSha256']!=sha(PROTOCOL):raise ValueError('Launch protocol changed')
        for path,digest in identity['sourceHashes'].items():
            if sha(ROOT/path)!=digest:raise ValueError('Launch source changed')
        for entry in [*p['geometry'].values(),p['partition'],p['topologyReport'],p['retainedPilotAcceptance'],
            p['nativeCaptureReport'],p['independentQuadrature'],*p['capturedPrograms'].values()]:bound(entry)
        bound(p['nativeCase'])
    try:
        check_sources()
        if not json.loads(bound(p['retainedPilotAcceptance']).read_text())['retainedSourcePilotComplete']:
            raise ValueError('Retained source resource admission missing')
        capture=json.loads(bound(p['nativeCaptureReport']).read_text())
        if capture['status']!='completed':raise ValueError('Native input capture incomplete')
        with zipfile.ZipFile(bound(p['nativeCase'])) as archive:
            mapping=json.loads(archive.read('manifest.json'))['artifacts']['surgical_toolbox']['mapping']
        if sorted(row['id'] for row in mapping if row.get('target'))!=p['targetLabels']:
            raise ValueError('Native source target mapping differs')
        stages=load_stages(p);masks=[stage['mask'] for stage in stages];programs={}
        for name,entry in p['capturedPrograms'].items():
            data=json.loads(gzip.decompress(bound(entry).read_bytes()));programs[name]=prepare_program(name,data,masks)
        prefixes=[program['prefix']['removedSourceIds'] for program in programs.values()]
        if any(prefix!=prefixes[0] for prefix in prefixes):raise ValueError('Branches do not share the frozen exposure prefix')
        if set(prefixes[0])&set(map(int,stages[0]['geometry']['cells'])):raise ValueError('Unmodeled prefix owner reappears')
        with np.load(bound(p['geometry']['reference']),allow_pickle=False) as raw:
            alpha,times,common,labels,cells=(raw[key] for key in ('alpha','times','commonCells','labels','cells'))
        label_by_cell={int(cell):int(label) for cell,label in zip(cells,labels)}
        volume=abs(float(np.linalg.det(stages[0]['geometry']['affine'][:3,:3])))
        for name,program in programs.items():
            accounting=program['capture']['summary']['fullSourceAccounting'];prefix=program['prefix']['fullSourceAccounting']
            added=program['capture']['summary']['additionalRemovedSourceIds']
            target=sum(volume for cell in added if label_by_cell[cell] in p['targetLabels'])
            if (target!=accounting['targetRemovedMm3']-prefix['targetRemovedMm3']
                    or len(added)*volume-target!=accounting['nonTargetRemovedMm3']-prefix['nonTargetRemovedMm3']):
                raise ValueError('Source-label/whole-source conservation differs from native receipts')
        for pass_index in range(2):
            begin=time.perf_counter()
            def emit(record,digest,fields,reasons,stage):
                key=(record['branch'],record['step'])
                if pass_index==0:
                    if key in digests:raise ValueError('Duplicate branch fixed step')
                    digests[key]=digest;metrics.append({**record,'recordHash':digest})
                    if reasons:
                        buffer=io.BytesIO();np.savez_compressed(buffer,**fields);content=buffer.getvalue()
                        fingerprint=hashlib.sha256(content).hexdigest()
                        if fingerprint not in payloads:
                            filename=f"{record['branch']}-step-{record['step']:04d}.npz"
                            (output/filename).write_bytes(content);payloads[fingerprint]=filename
                        samples.append({'branch':record['branch'],'step':record['step'],'tissueState':record['tissueState'],
                            'reasons':reasons,'file':payloads[fingerprint],'sha256':fingerprint,'geometry':stage['entry']})
                        write(output/'samples.json',samples)
                elif digests.get(key)!=digest:
                    write(output/'replay-failure.json',{'branch':key[0],'step':key[1],'expectedHash':digests.get(key),
                        'actualHash':digest,'actualRecord':record})
                    raise ValueError('Full-field/scalar deterministic replay differs')
            def progress(stage,builds,counts):
                report.update(phase='captured-programs',passIndex=pass_index,tissueState=stage,
                    currentBuilds=builds,branchStepCounts=counts)
                write(output/'report.json',report)
                print(json.dumps({'pass':pass_index,'stage':stage,'counts':counts,'wallSeconds':time.perf_counter()-started}),flush=True)
            summaries,builds=run_pass(programs,stages,alpha,times,common,p,emit,check,progress)
            if pass_index==1 and summaries!=passes[0]['branches']:raise ValueError('Final branch replay summaries differ')
            passes.append({'branches':summaries,'builds':builds,'wallSeconds':time.perf_counter()-begin})
        if len(metrics)!=p['expectedFixedSteps'] or len(samples)!=p['expectedSampleWorlds']:
            raise ValueError('Full program/sample coverage incomplete')
        if len(payloads)>p['maxDistinctFieldPayloads']:raise ValueError('Frozen field output admission exceeded')
        payload=gzip.compress(json.dumps(metrics,sort_keys=True,separators=(',',':'),allow_nan=False).encode(),mtime=0)
        (output/'step-metrics.json.gz').write_bytes(payload);write(output/'samples.json',samples)
        check_sources();check()
        report.update(status='completed',phase='completed',passes=passes,sourceInputsUnchanged=True,
            deterministicAllStepReplayExact=True,allStepCount=len(metrics),sampleWorlds=len(samples),
            distinctFieldPayloads=len(payloads),metricsFile='step-metrics.json.gz',metricsSha256=sha(output/'step-metrics.json.gz'),
            samplesSha256=sha(output/'samples.json'),completeCapturedInputPrograms=True,
            prefixInterpretation='All native prefix costs/receipts retained; no energy invented for5500owners absent from post-exposure mesh.')
    except BaseException as error:
        report.update(status='invalid_or_incomplete',error=type(error).__name__+': '+str(error));raise
    finally:
        if metrics and report['status']!='completed':
            (output/'step-metrics-partial.json.gz').write_bytes(gzip.compress(
                json.dumps(metrics,sort_keys=True,separators=(',',':'),allow_nan=False).encode(),mtime=0))
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
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():raise ValueError('Commit frozen full-program protocol first')
    launch=output.with_name(output.name+'-launch.json')
    if launch.exists():raise ValueError('Launch already exists')
    write(launch,{'protocolSha256':sha(PROTOCOL),'sourceHashes':{str(f.relative_to(ROOT)):sha(f) for f in SOURCES},
        'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()})
    p=json.loads(PROTOCOL.read_text());env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1')
    started=time.perf_counter();receipt=supervise([sys.executable,str(Path(__file__)),'--worker','--launch',str(launch),str(output/'result')],
        output,cwd=ROOT,environment=env,seconds=p['caps']['wallSeconds'],rss_bytes=p['caps']['rssBytes'])
    valid=receipt['status']=='completed';independent=None
    if valid:
        result=json.loads((output/'result/report.json').read_text());valid=result['status']=='completed'
    if valid:
        remaining=p['caps']['wallSeconds']-(time.perf_counter()-started)
        if remaining<=1:valid=False
        else:
            independent=supervise([sys.executable,str(ROOT/'scripts/check_continuum_programs.py'),str(output/'result'),str(PROTOCOL)],
                output/'independent',cwd=ROOT,environment=env,seconds=min(remaining-1,90),rss_bytes=p['caps']['rssBytes'])
            valid=independent['status']=='completed'
    if sum(f.stat().st_size for f in output.rglob('*') if f.is_file())>p['caps']['outputBytes']:valid=False
    if time.perf_counter()-started>p['caps']['wallSeconds']-1:valid=False
    write(output/'acceptance.json',{'completeCapturedInputQualification':valid,'largerDomainPublicActionsAdmitted':False,
        'physicalValidationPassed':False,'physiologicalRL':'hold','supervisionStatus':receipt['status'],
        'independentStatus':None if independent is None else independent['status'],'combinedWallSeconds':time.perf_counter()-started})
    if not valid:raise RuntimeError('Captured program qualification failed/incomplete; evidence retained')
    print(json.dumps({'workerSeconds':result['wallSeconds'],'combinedSeconds':time.perf_counter()-started}))


if __name__=='__main__':main()
