#!/usr/bin/env python3
"""Bounded source cost/field check for a conservative algebraic corridor."""
from pathlib import Path
import argparse,hashlib,io,json,os,resource,subprocess,sys,time
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from resectionlab.surgical_continuum_corridor import corridor_partition
from resectionlab.surgical_continuum_topology import SourceDomain
from resectionlab.surgical_closed_loop import CoupledPreparation,CusaContinuumEpisode,hold_contacts
from resectionlab.surgical_continuum_events import SourceHistoryLedger
from resectionlab.surgical_continuum_program import field_hash
from closed_loop_source import seed
from qualify_continuum_exterior import field_metrics
from qualify_continuum_closed_loop import CappedOutput
from profile_continuum_closed_loop import SOURCES as PRIOR_SOURCES
from profile_continuum_exterior import bound,sha,write
from febio_runtime import supervise

PROTOCOL=ROOT/'manifests/experiments/continuum-corridor-profile-v1.json'
CHECKER=ROOT/'scripts/check_continuum_corridor_profile.py'
SOURCES=[Path(__file__),CHECKER,ROOT/'scripts/qualify_continuum_closed_loop.py',ROOT/'scripts/continuum_quadrature.py',
    ROOT/'scripts/check_continuum_closed_loop.py',ROOT/'src/resectionlab/surgical_continuum_corridor.py',*PRIOR_SOURCES]


def worker(output,launch):
    output.mkdir(exist_ok=False);p=json.loads(PROTOCOL.read_text());identity=json.loads(launch.read_text());started=time.perf_counter()
    store=CappedOutput(output,p['caps']['outputBytes']-65536);inputs={}
    report={'schema':'resectionlab.continuum-corridor-profile.v1','status':'started','phase':'source-seed',
        'sourceCommit':identity['commit'],'sourceHashes':identity['sourceHashes'],'protocolSha256':sha(PROTOCOL),
        'sourceRole':p['sourceRole'],'scope':p['scope'],'completePilot':False,'physicalValidationPassed':False,
        'largerDomainPublicActionsAdmitted':False,'physiologicalRL':'hold','largeScaleRL':'hold','trainingUpdates':0,'newAcquiredDatasetBytes':0,
        'numpyVersion':np.__version__,'numericalThreads':{k:os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS')}}
    def check():
        if time.perf_counter()-started>p['workerWallSeconds']-5:raise TimeoutError('Corridor profile cooperative cap')
    def frozen(entry):
        path=bound(entry);inputs[str(path.relative_to(ROOT))]=entry['sha256'];return path
    def unchanged():
        if sha(PROTOCOL)!=identity['protocolSha256']:raise ValueError('Changed corridor protocol')
        for path,digest in {**identity['sourceHashes'],**inputs}.items():
            if sha(ROOT/path)!=digest:raise ValueError('Changed corridor source/input '+path)
    def arrays(name,values):
        b=io.BytesIO();np.savez_compressed(b,**values);store.put(output/name,b.getvalue());return {'file':name,'sha256':sha(output/name)}
    try:
        unchanged()
        for name in ('geometryPreflight','independentChecker','quadratureChecker','independentQuadrature'):frozen(p[name])
        baseline_path=frozen(p['qualifiedPilotReport']);baseline=json.loads(baseline_path.read_text())
        admission=json.loads(frozen(p['qualifiedPilotAcceptance']).read_text())
        if not admission['conditionalClosedLoopPilotComplete'] or baseline['status']!='completed' or not baseline['allStepReplayExact']:
            raise ValueError('Incomplete qualified conditional baseline')
        scene,grid,original,initial,alpha,times,h,common,commands,metadata=seed(p,inputs,check)
        active=[c for c in commands if c['operation']=='apply']
        if not active:raise ValueError('Corridor has no active request')
        slot=active[0]['slot']
        if (any(c['slot']!=slot or c['pose']!=active[0]['pose'] for c in active)
                or initial['slots'][slot]['instrument']!='cusa' or initial['parametersBySlot'][slot]['contactGeometryRevision']!=2):
            raise ValueError('Corridor requires repeated absolute spherical-tip requests')
        selection,partition=corridor_partition(original.geometry,hold_contacts(scene,initial['holds']),
            initial['slots'][slot]['position'],active[0]['pose']['position'],initial['parametersBySlot'][slot]['radiusMm'],p['limits']['maxCenterDisplacementMm'])
        core_sha=hashlib.sha256(partition['coreFreeDofs'].tobytes()).hexdigest()
        owner_sha=hashlib.sha256(partition['corridorSourceIds'].tobytes()).hexdigest()
        if core_sha!=p['rootCoreFreeDofsSha256'] or owner_sha!=p['corridorSourceIdsSha256']:raise ValueError('Changed frozen corridor geometry')
        report.update(seed=metadata,selection=selection,rootCoreFreeDofsSha256=core_sha,corridorSourceIdsSha256=owner_sha,
            partition=arrays('partition.npz',partition),phase='initial-preparation');store.json(output/'report.json',report)
        domain=SourceDomain(original.geometry,partition['coreFreeDofs'])
        def measure(g,free,u,z):
            lookup={int(c):i for i,c in enumerate(g['cells'])};rows=np.asarray([lookup[int(c)] for c in common if int(c) in lookup],int)
            return field_metrics(g,free,u,z,rows,alpha)
        begin=time.perf_counter();backend=CoupledPreparation(domain,alpha,times,check=check)
        episode=CusaContinuumEpisode(scene,grid,domain,initial,alpha,times,h,measure,p['limits'],backend=backend,check=check)
        report.update(initialPreparationSeconds=time.perf_counter()-begin,phase='one-actual-microtick');store.json(output/'report.json',report)
        begin=time.perf_counter();receipt=episode.advance(commands[0]);report['firstMicrotickSeconds']=time.perf_counter()-begin
        batch=episode.last_committed_batch();s=episode.state;g=s['stage']['geometry']
        if receipt['sourceEvent'] is not None:raise ValueError('Unexpected first actual tick removal')
        for key in ('command','limiter','orderedSourceIds','cusaEvents','resolvedPose','fixedSteps','remainderSeconds','elapsedSeconds','postEndpointCovers'):
            if receipt[key]!=baseline['ticks'][0][key]:raise ValueError('Algebraic partition changed first causal decision '+key)
        samples_path=frozen({'path':str((baseline_path.parent/baseline['samplesFile']).relative_to(ROOT)),'sha256':baseline['samplesSha256']})
        samples=json.loads(samples_path.read_text());sample=next(x for x in samples if x['tick']==0 and x['withinTick']==11)
        old_fields=frozen({'path':str((baseline_path.parent/sample['fieldsFile']).relative_to(ROOT)),'sha256':sample['fieldsSha256']})
        row,fields=batch['records'][-1]
        with np.load(old_fields,allow_pickle=False) as prior:
            if not np.array_equal(fields['freeIndices'],prior['freeIndices']):raise ValueError('Baseline physical support changed')
            difference=max(float(np.max(np.abs(fields[k]-prior[k]),initial=0.)) for k in ('displacement','memories','beforeDisplacement','beforeMemories'))
        if difference>p['tolerances']['displacementMm']:raise ValueError('Complete field comparison budget')
        report.update(firstMicrotick=receipt,completeFieldDifferenceMm=difference,
            geometry=arrays('geometry.npz',{**g,'coreFreeDofs':s['stage']['coreFreeDofs']}),fields=arrays('fields.npz',{**fields,'previousTarget':fields['target'],'alpha':alpha,'times':times}),
            sampleRecord={**row,'h':h,'contacts':s['contacts']},phase='hypothetical-current-owner-removal')
        store.json(output/'report.json',report);check();before=episode.state_hash();checkpoint=backend.checkpoint();owner=receipt['orderedSourceIds'][0]
        if owner not in partition['corridorSourceIds']:raise ValueError('Current source owner escaped declared corridor')
        removed=sorted(set(s['removed'])|{owner});candidate=domain.after(sorted(set(removed)&set(map(int,domain.cells))))
        begin=time.perf_counter();transfer=SourceHistoryLedger().transfer(g,candidate['geometry'],s['u'],s['z'],alpha);transfer_seconds=time.perf_counter()-begin
        begin=time.perf_counter();backend.prepare(candidate,s['contacts'],removed);preparation_seconds=time.perf_counter()-begin
        if not backend.builds[-1]['retainedExterior']:raise ValueError('Candidate did not certify exact retained exterior')
        backend.rollback(checkpoint)
        if episode.state_hash()!=before:raise ValueError('Hypothetical matrix probe changed physical state')
        report.update(hypothetical={'sourceOwner':owner,'isEpisodeEvent':False,'appliedToPhysicalState':False,
            'transferSeconds':transfer_seconds,'preparationSeconds':preparation_seconds,'event':transfer['event'],
            'exactRetainedExteriorAndBothCouplings':True,'rollbackStateHash':before,
            'geometry':arrays('candidate-geometry.npz',{**candidate['geometry'],'coreFreeDofs':candidate['coreFreeDofs']}),
            'fields':arrays('candidate-fields.npz',{'displacement':transfer['displacement'],'memories':transfer['memories']})},builds=backend.builds,
            estimatedTwoPassWorkerSeconds=2*(report['initialPreparationSeconds']+len(commands)*report['firstMicrotickSeconds']+
                p['maximumNewRemovals']*(transfer_seconds+preparation_seconds)),phase='completed',status='completed')
        unchanged();check();report['sourceInputsUnchanged']=True
    except BaseException as error:report.update(status='invalid_or_incomplete',error=type(error).__name__+': '+str(error));raise
    finally:
        report.update(inputHashes=inputs,wallSeconds=time.perf_counter()-started,
            peakRssBytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024))
        store.json(output/'report.json',report);store.audit()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('output',type=Path);parser.add_argument('--worker',action='store_true');parser.add_argument('--launch',type=Path)
    args=parser.parse_args();output=args.output.resolve()
    if not output.is_relative_to(ROOT/'testcases'):raise ValueError('Evidence stays under testcases')
    if args.worker:
        if args.launch is None:raise ValueError('Supervised launch required')
        worker(output,args.launch);return
    if output.exists():raise ValueError('Fresh output path required; no retry')
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():raise ValueError('Commit corridor profile first')
    launch=output.with_name(output.name+'-launch.json')
    if launch.exists():raise ValueError('Launch already exists')
    write(launch,{'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'protocolSha256':sha(PROTOCOL),'sourceHashes':{str(f.relative_to(ROOT)):sha(f) for f in SOURCES}})
    p=json.loads(PROTOCOL.read_text());env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1');start=time.perf_counter()
    receipt=supervise([sys.executable,str(Path(__file__)),'--worker','--launch',str(launch),str(output/'result')],output,
        cwd=ROOT,environment=env,seconds=p['workerWallSeconds'],rss_bytes=p['caps']['rssBytes']);valid=receipt['status']=='completed';checker=None
    if valid:
        remaining=p['caps']['wallSeconds']-(time.perf_counter()-start)-1
        if remaining<=1:valid=False
        else:
            checker=supervise([sys.executable,str(CHECKER),str(output/'result'),str(PROTOCOL)],output/'independent',cwd=ROOT,environment=env,
                seconds=min(p['checkerWallSeconds'],remaining),rss_bytes=p['caps']['rssBytes'])
            valid=checker['status']=='completed' and json.loads((output/'result/checker-result.json').read_text())['status']=='passed'
    elapsed=time.perf_counter()-start
    acceptance={'corridorCostProfileComplete':valid,'completePilot':False,'physicalValidationPassed':False,'largerDomainPublicActionsAdmitted':False,
        'physiologicalRL':'hold','largeScaleRL':'hold','combinedWallSeconds':elapsed,'supervisionStatus':receipt['status'],
        'independentSupervisionStatus':None if checker is None else checker['status']}
    total=sum(f.stat().st_size for f in output.rglob('*') if f.is_file())+len((json.dumps(acceptance,indent=2)+'\n').encode())
    if elapsed>p['caps']['wallSeconds'] or total>p['caps']['outputBytes']:valid=False;acceptance['corridorCostProfileComplete']=False
    write(output/'acceptance.json',acceptance)
    if not valid:raise RuntimeError('Corridor profile incomplete; evidence preserved')
    print(json.dumps(acceptance))


if __name__=='__main__':main()
