#!/usr/bin/env python3
"""Bounded two-exterior-block pilot; prescribed nonzero state, not full programs."""
from pathlib import Path
import argparse
import gc
import json
import os
import resource
import subprocess
import sys
import time
import weakref

import numpy as np
from scipy import sparse

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from resectionlab.surgical_continuum_exterior import ExteriorOperator,ExteriorCoreFactor,ExteriorHistorySolve,source_stiffness
from resectionlab.surgical_continuum_events import SourceHistoryLedger,source_operator_identity
from profile_continuum_exterior import load_geometry,free_partition,bound,sha,write
from qualify_continuum_exterior import contact_matrix,field_metrics
from febio_runtime import supervise

PROTOCOL=ROOT/'manifests/experiments/continuum-retained-pilot-v1.json'
SOURCES=[Path(__file__),*[ROOT/'scripts'/name for name in ('check_continuum_retained.py','profile_continuum_exterior.py',
    'qualify_continuum_exterior.py','febio_runtime.py')],*[ROOT/'src/resectionlab'/name for name in
    ('surgical_continuum_exterior.py','surgical_continuum_events.py','surgical_continuum_partition.py','surgical_continuum.py')]]


def worker(output,launch):
    output.mkdir(parents=True,exist_ok=False);p=json.loads(PROTOCOL.read_text());identity=json.loads(launch.read_text())
    started=time.perf_counter();rows=[];ledger=SourceHistoryLedger()
    report={'schema':'resectionlab.continuum-retained-pilot-result.v1','status':'started','phase':'admission',
        'sourceCommit':identity['commit'],'sourceHashes':identity['sourceHashes'],'protocolSha256':sha(PROTOCOL),
        'scope':p['scope'],'physicalValidationPassed':False,'physiologicalRL':'hold','largeScaleRL':'hold',
        'completePrograms':False,'largerDomainPublicActionsAdmitted':False,'newAcquiredDatasetBytes':0,'trainingUpdates':0}
    def check():
        if time.perf_counter()-started>p['caps']['wallSeconds']-15:raise TimeoutError('Retained pilot cooperative cap')
        if sum(f.stat().st_size for f in output.rglob('*') if f.is_file())>p['caps']['outputBytes']-65536:
            raise ValueError('Retained pilot output cap')
    def check_sources():
        if identity['protocolSha256']!=sha(PROTOCOL):raise ValueError('Launch protocol changed')
        for path,digest in identity['sourceHashes'].items():
            if sha(ROOT/path)!=digest:raise ValueError('Launch source changed')
        for entry in [*p['geometry'].values(),p['partition'],p['baselineWorld'],p['topologyReport'],p['independentQuadrature']]:bound(entry)
    try:
        check_sources();initial,partition_summary,partition=load_geometry(p);initial_free,initial_core=free_partition(initial,partition)
        topology_path=bound(p['topologyReport']);topology=json.loads(topology_path.read_text())
        if not topology['exactFactorReuseCandidate'] or topology['uniqueTissueStates']!=9:raise ValueError('Missing exact source identity gate')
        with np.load(bound(p['geometry']['reference']),allow_pickle=False) as raw:
            alpha,times,spring,contacts,common=(raw[k] for k in ('alpha','times','spring','contactCells','commonCells'))
        with np.load(bound(p['baselineWorld']),allow_pickle=False) as raw:sample={key:raw[key] for key in raw.files}
        if not np.array_equal(sample['freeIndices'],initial_free):raise ValueError('Nonzero initial source world differs')
        u=np.zeros(initial['nodes'].size);u[initial_free]=sample['displacement']
        z=np.zeros((len(alpha),len(u)));z[:,initial_free]=sample['memories']
        target=sample['target'].copy();previous_target=target.copy();h=float(sample['h'])
        if h!=p['stepSeconds']:raise ValueError('Physical pilot step differs')
        solve=None;geometry=initial;old_lineage=np.arange(len(initial_free));fresh=0;retained=0
        cumulative_removed=0.;cumulative_jump=0.;cumulative_work=0.;cumulative_loss=0.;initial_energy=None
        sequence=[*topology['states'],topology['states'][-1],topology['states'][-1]]
        for index,state in enumerate(sequence):
            check();begin=time.perf_counter();old_energy=None if not rows else rows[-1]['energyNmm']
            old_contact=0. if solve is None else solve.contact_energy(u[np.flatnonzero(geometry['freeDofs'])],previous_target)
            if index==0:
                g=initial;core=initial_core;lineage=np.arange(len(initial_free));geometry_entry=p['geometry']['reference'];event=None
            elif index<9:
                geometry_entry=dict(state['geometry'])
                path=topology_path.parent/geometry_entry['path']
                if sha(path)!=geometry_entry['sha256']:raise ValueError('Frozen derived geometry changed')
                geometry_entry['path']=str(path.relative_to(ROOT))
                with np.load(path,allow_pickle=False) as raw:
                    g={key:raw[key] for key in initial};core=raw['coreFreeDofs'];lineage=raw['originalFreeDofForNew']
                transferred=ledger.transfer(geometry,g,u,z,alpha);u,z,event=transferred['displacement'],transferred['memories'],transferred['event']
            else:
                g=geometry;event=None;geometry_entry=rows[-1]['geometry']
            free=np.flatnonzero(g['freeDofs']);build_start=time.perf_counter();reset_refusal=None;old_operator_released=None
            if index<9:
                full=source_stiffness(g,source_order=initial['cells'],check=check);k=full[free][:,free].tocsr();del full
                assembly_seconds=time.perf_counter()-build_start
                inverse={int(value):i for i,value in enumerate(old_lineage)}
                parents=np.asarray([inverse[int(value)] for value in lineage])
                if state['newExteriorGroup']:
                    if solve is not None:
                        try:solve.prepared.rebuild(k,core,parents,check=check)
                        except ValueError as error:
                            reset_refusal=str(error)
                            if reset_refusal!='EXTERIOR_RETAINED_EXTERIOR_COEFFICIENTS':raise
                        else:raise ValueError('Changed exterior was silently retained')
                        reference=weakref.ref(solve.operator);solve=None;gc.collect()
                        old_operator_released=reference() is None
                        if not old_operator_released:raise ValueError('Old exterior operator remains resident during full refactor')
                    prepared=ExteriorCoreFactor(ExteriorOperator(k,core,check=check),check=check);fresh+=1
                else:
                    prepared=solve.prepared.rebuild(k,core,parents,check=check);retained+=1
                c=contact_matrix(g,contacts)[:,free]
                next_solve=ExteriorHistorySolve(prepared.operator,c,spring,alpha,times,prepared=prepared,check=check)
                # Independently apply E elimination to deterministic supported
                # core vectors; no dense Schur reconstruction/repair is allowed.
                schur_error=0.
                for phase in p['schurProbePhases']:
                    probe=np.sin(np.arange(len(core))*.013+phase)*1e-3
                    schur_error=max(schur_error,float(np.max(np.abs(prepared.schur@probe-prepared.operator.apply(probe)),initial=0.)))
                if schur_error>p['tolerances']['forceN']:raise ValueError('Current-operator Schur probe disagreement')
                solve=next_solve;del next_solve,prepared,k
                stats=dict(solve.operator.stats)
            else:
                assembly_seconds=0.;schur_error=0.;stats=None
                if index==9:solve=solve.rebind(sparse.csr_matrix((0,len(free))),[],check=check);target=np.empty(0)
            # Target-dependent contact potential jumps are separate from
            # source removal and the subsequent BE work/dissipation terms.
            jump=0. if index==0 else solve.contact_energy(u[free],target)-old_contact
            previous_for_step=previous_target if index<9 else target
            before=u[free].copy();before_z=z[:,free].copy()
            result=solve.step(before,before_z,np.zeros(len(free)),target,h,previous_target=previous_for_step)
            u=np.zeros(g['nodes'].size);u[free]=result['displacement'];z=np.zeros((len(alpha),len(u)));z[:,free]=result['memories']
            removed=0. if event is None else event['removedStoredEnergyNmm']
            if old_energy is not None and abs(result['oldEnergyNmm']-(old_energy-removed+jump))>p['tolerances']['energyNmm']:
                raise ValueError('Instantaneous source/contact event energy does not close')
            if initial_energy is None:initial_energy=result['oldEnergyNmm']
            cumulative_removed+=removed;cumulative_jump+=jump;cumulative_work+=result['workNmm']
            cumulative_loss+=result['dissipatedEnergyNmm']+result['numericalDissipationNmm']
            balance=result['energyNmm']-initial_energy+cumulative_removed-cumulative_jump+cumulative_loss-cumulative_work
            if abs(balance)>p['tolerances']['balanceNmm']:raise ValueError('Pilot cumulative event/BE energy balance failed')
            source_rows={int(cell):i for i,cell in enumerate(g['cells'])};common_rows=np.asarray([source_rows[int(cell)] for cell in common])
            rates,gradient,center=field_metrics(g,free,result['displacement'],result['memories'],common_rows,alpha)
            if gradient>p['domain']['maxGradientNorm'] or center>p['domain']['maxCenterDisplacementMm']:
                raise ValueError('Unchanged small-gradient domain exceeded')
            reaction=np.linalg.norm(result['contactReactionN'].reshape((-1,3)).sum(axis=0))
            if reaction>p['domain']['maxReactionN']:raise ValueError('Unchanged reaction domain exceeded')
            filename=f'fields-state-{index:02d}.npz'
            np.savez_compressed(output/filename,freeIndices=free,beforeDisplacement=before,beforeMemories=before_z,
                displacement=result['displacement'],memories=result['memories'],target=target,previousTarget=previous_for_step,h=h)
            row={'state':index,'tissueState':min(index,8),'geometry':geometry_entry,'fieldsFile':filename,
                'fieldsSha256':sha(output/filename),'sourceOperatorIdentity':source_operator_identity(g,alpha,times,
                    core_free_dofs=core,removed_source_ids=state['removedSourceIds']),
                'removedSourceIds':state['removedSourceIds'],'event':event,'contactModelEnergyJumpNmm':jump,
                'removedStoredEnergyNmm':removed,'cumulativeEventBalanceResidualNmm':float(balance),
                'assemblySeconds':assembly_seconds,'stateWallSeconds':time.perf_counter()-begin,
                'operator':stats,'schurProbeMaximumN':schur_error,'exteriorResetRefusal':reset_refusal,
                'oldExteriorOperatorReleased':old_operator_released,'contacts':len(target)//3,
                'rates':rates.tolist(),'maxGradientNorm':gradient,'maxCenterDisplacementMm':center,
                **{key:value.tolist() if isinstance(value,np.ndarray) else value for key,value in result.items()
                   if key not in ('displacement','memories')}}
            rows.append(row);geometry=g;old_lineage=lineage;previous_target=target.copy()
            report.update(phase='source-states',states=rows,fullExteriorBuilds=fresh,retainedCoreBuilds=retained)
            write(output/'report.json',report)
            print(json.dumps({'state':index,'freshE':fresh,'retained':retained,'wallSeconds':time.perf_counter()-started}),flush=True)
        if (fresh!=2 or retained!=7 or len(ledger.removed_source_ids)!=8 or len(rows)!=11
                or not np.max(np.abs(z))>0):raise ValueError('Frozen pilot state/removal/release coverage incomplete')
        check_sources();check();report.update(status='completed',phase='completed',sourceInputsUnchanged=True,
            initialEnergyNmm=initial_energy,cumulativeRemovedStoredEnergyNmm=cumulative_removed,
            cumulativeContactModelEnergyJumpNmm=cumulative_jump,cumulativeWorkNmm=cumulative_work,
            cumulativeLossNmm=cumulative_loss,cumulativeEventBalanceResidualNmm=float(balance),
            retainedCorrectionBytes=solve.prepared._correction.nbytes,
            initialInterpretation='Nonzero previously qualified fixed-domain step30; eleven prescribed pilot steps are not the captured public programs.')
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
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():raise ValueError('Commit prospective pilot first')
    launch=output.with_name(output.name+'-launch.json')
    if launch.exists():raise ValueError('Launch already exists')
    write(launch,{'protocolSha256':sha(PROTOCOL),'sourceHashes':{str(f.relative_to(ROOT)):sha(f) for f in SOURCES},
        'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()})
    p=json.loads(PROTOCOL.read_text());env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1')
    started=time.perf_counter()
    receipt=supervise([sys.executable,str(Path(__file__)),'--worker','--launch',str(launch),str(output/'result')],
        output,cwd=ROOT,environment=env,seconds=p['caps']['wallSeconds'],rss_bytes=p['caps']['rssBytes'])
    valid=receipt['status']=='completed';checker_receipt=None
    if valid:
        r=json.loads((output/'result/report.json').read_text());valid=r['status']=='completed' and len(r['states'])==11
    if valid:
        remaining=p['caps']['wallSeconds']-(time.perf_counter()-started)
        if remaining<1:valid=False
        else:
            checker_receipt=supervise([sys.executable,str(ROOT/'scripts/check_continuum_retained.py'),str(output/'result'),str(PROTOCOL)],
                output/'independent',cwd=ROOT,environment=env,seconds=min(remaining,45),rss_bytes=p['caps']['rssBytes'])
            valid=checker_receipt['status']=='completed'
    if sum(f.stat().st_size for f in output.rglob('*') if f.is_file())>p['caps']['outputBytes']:valid=False
    write(output/'acceptance.json',{'retainedSourcePilotComplete':valid,'completePrograms':False,
        'physicalValidationPassed':False,'physiologicalRL':'hold','supervisionStatus':receipt['status'],
        'independentSupervisionStatus':None if checker_receipt is None else checker_receipt['status'],
        'combinedWallSeconds':time.perf_counter()-started})
    if not valid:raise RuntimeError('Retained source pilot failed/incomplete; evidence retained')
    print(json.dumps({'workerSeconds':r['wallSeconds'],'combinedSeconds':time.perf_counter()-started}))


if __name__=='__main__':main()
