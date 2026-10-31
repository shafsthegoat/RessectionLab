#!/usr/bin/env python3
"""Profile two algebraic partitions of the same full physical source model."""
from pathlib import Path
import argparse
import gc
import hashlib
import json
import os
import resource
import subprocess
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from resectionlab.surgical_closed_loop import CusaContinuumEpisode,CoupledPreparation,hold_contacts
from resectionlab.surgical_continuum_program import contact_matrix
from resectionlab.surgical_continuum_topology import SourceDomain
from resectionlab.surgical_continuum_events import SourceHistoryLedger
from closed_loop_source import seed
from qualify_continuum_exterior import field_metrics
from profile_continuum_exterior import bound,sha,write
from febio_runtime import supervise

PROTOCOL=ROOT/'manifests/experiments/continuum-closed-loop-profile-v1.json'
SOURCES=[Path(__file__),*[ROOT/'scripts'/name for name in ('closed_loop_source.py','survey_continuum_pre_effects.py',
    'survey_continuum_contacts.py','qualify_continuum_exterior.py','profile_continuum_exterior.py','febio_runtime.py')],
    *[ROOT/'src/resectionlab'/name for name in ('surgical_closed_loop.py','surgical_continuum_topology.py','surgical_continuum_exterior.py',
        'surgical_continuum_events.py','surgical_continuum_partition.py','surgical_continuum_program.py','surgical_pre_effect_witness.py','surgical_source_contacts.py',
        'surgical_contact_survey.py','surgical_contact_witness.py','surgical_geometry_view.py','surgical_cusa_v3.py',
        'surgical_deformation_v2.py','surgical_continuum.py','surgical_motion.py','geometry.py','independent_geometry_batch.py',
        'surgical_tools.py','surgical_occupancy.py','surgical_physics_v2.py','workspace_bundle.py','surgical_scene.py','surgical_environment.py','core.py')]]


def worker(output,launch,seed_only=False):
    output.mkdir(parents=True,exist_ok=False);p=json.loads(PROTOCOL.read_text());identity=json.loads(launch.read_text());started=time.perf_counter()
    caps=p['seedPreflightCaps'] if seed_only else p['caps']
    inputs={};variants=[];report={'schema':'resectionlab.closed-loop-profile-result.v1','status':'started','phase':'admission',
        'sourceCommit':identity['commit'],'sourceHashes':identity['sourceHashes'],'protocolSha256':sha(PROTOCOL),
        'physicalValidationPassed':False,'physiologicalRL':'hold','largeScaleRL':'hold','largerDomainPublicActionsAdmitted':False,
        'newAcquiredDatasetBytes':0,'trainingUpdates':0,'completePilot':False,'mode':'seed-only' if seed_only else 'partition-costs',
        'scope':('Provenance and field reconstruction only; no episode, factor or solve' if seed_only else
            'One actual conditional microtick and a separately labeled hypothetical removal preparation per algebraic partition')}
    def check():
        if time.perf_counter()-started>caps['wallSeconds']-5:raise TimeoutError('Closed-loop source profile cooperative cap')
    def check_sources():
        if sha(PROTOCOL)!=identity['protocolSha256']:raise ValueError('Launch protocol changed')
        for path,digest in identity['sourceHashes'].items():
            if sha(ROOT/path)!=digest:raise ValueError('Launch source changed')
        for path,digest in inputs.items():
            if sha(ROOT/path)!=digest:raise ValueError('Frozen input changed')
    try:
        check_sources();scene,grid,original,initial,alpha,times,h,common,commands,metadata=seed(p,inputs,check)
        loaded=time.perf_counter()-started;report.update(phase='source-loaded',seed=metadata,inputHashes=inputs);write(output/'report.json',report)
        if seed_only:
            check_sources();check();report.update(status='completed',phase='completed',seedPreflightComplete=True,
                sourceInputsUnchanged=True,seedLoadSeconds=loaded,mechanicsSolves=0);return
        def measure(g,free,u,z):
            lookup={int(cell):i for i,cell in enumerate(g['cells'])};rows=np.asarray([lookup[int(cell)] for cell in common if int(cell) in lookup],int)
            return field_metrics(g,free,u,z,rows,alpha)
        reference=None;reference_z=None
        for policy in p['variants']:
            check();begin=time.perf_counter();report.update(phase='preparing-'+policy,variants=variants);write(output/'report.json',report)
            if policy=='original-core-retained':domain=original;fresh=False
            elif policy=='contact-core-fresh':
                root=original.geometry;free=np.flatnonzero(root['freeDofs'])
                core=np.unique(contact_matrix(root,hold_contacts(scene,initial['holds']))[:,free].indices)
                domain=SourceDomain(root,core);fresh=True
            else:raise ValueError('Frozen partition policy unsupported')
            backend=CoupledPreparation(domain,alpha,times,allow_fresh_exterior_updates=fresh,check=check)
            episode=CusaContinuumEpisode(scene,grid,domain,initial,alpha,times,h,measure,p['limits'],backend=backend,check=check)
            preparation=time.perf_counter()-begin;t=time.perf_counter();first=episode.advance(commands[0]);tick=time.perf_counter()-t
            if first['orderedSourceIds']!=metadata['coupledOrderAtSeed'] or first['sourceEvent'] is not None:
                raise ValueError('First actual microtick did not preserve audited seed CUSA decision')
            u=episode.state['u'];z=episode.state['z'];difference=0.
            if reference is None:reference=u.copy();reference_z=z.copy()
            else:
                difference=max(float(np.max(np.abs(u-reference))),float(np.max(np.abs(z-reference_z))))
                if difference>p['tolerances']['displacementMm']:raise ValueError('Algebraic partition changed complete source fields')
            before=episode.state_hash();checkpoint=backend.checkpoint();owner=int(first['orderedSourceIds'][0])
            removed=sorted(set(episode.state['removed'])|{owner});within=sorted(set(removed)&set(map(int,domain.cells)))
            candidate=domain.after(within);t=time.perf_counter()
            transfer=SourceHistoryLedger().transfer(episode.state['stage']['geometry'],candidate['geometry'],u,z,alpha)
            transfer_seconds=time.perf_counter()-t;t=time.perf_counter();backend.prepare(candidate,episode.state['contacts'],removed)
            update=time.perf_counter()-t;stats=dict(backend.builds[-1]);backend.rollback(checkpoint)
            if episode.state_hash()!=before:raise ValueError('Matrix-only candidate modified committed physical state')
            estimate=loaded+2*(preparation+len(commands)*tick+p['maximumNewRemovals']*(update+transfer_seconds))
            variants.append({'policy':policy,'rootCoreFreeDofs':len(domain.original_core),
                'rootCoreFreeDofsSha256':hashlib.sha256(domain.original_core.astype(np.int64).tobytes()).hexdigest(),
                'initialPreparationSeconds':preparation,'firstMicrotickSeconds':tick,'firstMicrotick':first,
                'candidateSourceOwner':owner,'candidateIsEpisodeEvent':False,'candidateAppliedToPhysicalState':False,
                'hypotheticalTransferSeconds':transfer_seconds,'hypotheticalRemovedEnergyNmm':transfer['event']['removedStoredEnergyNmm'],
                'hypotheticalPreparationSeconds':update,'hypotheticalPreparation':stats,
                'completeFieldDifferenceMm':difference,'estimatedTwoPassWorkerSeconds':estimate,
                'builds':backend.builds,'wallSeconds':time.perf_counter()-begin})
            report.update(variants=variants);write(output/'report.json',report);print(json.dumps({k:variants[-1][k] for k in ('policy','rootCoreFreeDofs','initialPreparationSeconds','hypotheticalPreparationSeconds','estimatedTwoPassWorkerSeconds')}),flush=True)
            del episode,backend,checkpoint,candidate,transfer,domain;gc.collect()
        check_sources();check();report.update(status='completed',phase='completed',sourceInputsUnchanged=True,inputHashes=inputs,
            seedLoadSeconds=loaded,selectedFastestPolicy=min(variants,key=lambda x:x['estimatedTwoPassWorkerSeconds'])['policy'])
    except BaseException as error:
        report.update(status='invalid_or_incomplete',error=type(error).__name__+': '+str(error));raise
    finally:
        report.update(variants=variants,inputHashes=inputs,wallSeconds=time.perf_counter()-started,
            peakRssBytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024));write(output/'report.json',report)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('output',type=Path);parser.add_argument('--worker',action='store_true');parser.add_argument('--launch',type=Path)
    parser.add_argument('--seed-preflight',action='store_true')
    args=parser.parse_args();output=args.output.resolve()
    if not output.is_relative_to(ROOT/'testcases'):raise ValueError('Evidence stays under testcases/')
    if args.worker:
        if args.launch is None:raise ValueError('Supervised launch required')
        worker(output,args.launch,args.seed_preflight);return
    if output.exists():raise ValueError('Use fresh output; no automatic retry')
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():raise ValueError('Commit source profile first')
    launch=output.with_name(output.name+'-launch.json')
    if launch.exists():raise ValueError('Launch already exists')
    write(launch,{'protocolSha256':sha(PROTOCOL),'sourceHashes':{str(f.relative_to(ROOT)):sha(f) for f in SOURCES},
        'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()})
    p=json.loads(PROTOCOL.read_text());caps=p['seedPreflightCaps'] if args.seed_preflight else p['caps']
    env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1')
    command=[sys.executable,str(Path(__file__)),'--worker','--launch',str(launch),str(output/'result')]
    if args.seed_preflight:command.append('--seed-preflight')
    receipt=supervise(command,output,cwd=ROOT,environment=env,seconds=caps['wallSeconds'],rss_bytes=caps['rssBytes'])
    valid=receipt['status']=='completed'
    if valid:
        r=json.loads((output/'result/report.json').read_text());valid=r['status']=='completed' and (r.get('seedPreflightComplete',False) if args.seed_preflight else len(r['variants'])==2) and r['peakRssBytes']<=caps['rssBytes']
    if sum(f.stat().st_size for f in output.rglob('*') if f.is_file())>caps['outputBytes']:valid=False
    write(output/'acceptance.json',{'seedPreflightComplete':valid and args.seed_preflight,'sourceCostProfileComplete':valid and not args.seed_preflight,
        'completePilot':False,'largerDomainPublicActionsAdmitted':False,
        'physicalValidationPassed':False,'physiologicalRL':'hold','largeScaleRL':'hold','supervisionStatus':receipt['status']})
    if not valid:raise RuntimeError('Closed-loop source profile failed/incomplete; evidence retained')
    print(json.dumps({'wallSeconds':r['wallSeconds'],'selectedFastestPolicy':r.get('selectedFastestPolicy')}))


if __name__=='__main__':main()
