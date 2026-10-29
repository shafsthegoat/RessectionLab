#!/usr/bin/env python3
"""Frozen fixed8/core20 coupling witness; no live rollout or physical acceptance."""
from pathlib import Path
import argparse
import hashlib
import io
import json
import os
import resource
import subprocess
import sys
import time

import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from resectionlab.surgical_continuum import element_operators
from resectionlab.surgical_continuum_exterior import ExteriorOperator, ExteriorHistorySolve, source_stiffness
from profile_continuum_exterior import load_geometry, free_partition, bound, sha, write
from febio_runtime import supervise

PROTOCOL = ROOT/'manifests/experiments/continuum-exterior-qualification-v1.json'
SOURCES = [Path(__file__), ROOT/'scripts/profile_continuum_exterior.py', ROOT/'scripts/febio_runtime.py',
    *[ROOT/'src/resectionlab'/name for name in
      ('surgical_continuum_exterior.py', 'surgical_continuum_partition.py', 'surgical_continuum.py')]]


def full_accounting(k, c, d, alpha, times, before, old_z, u, z, target, previous_target, h):
    """Complete equations; actual target-dependent spring potential/work."""
    du = u-before; e = c@u-target; old_e = c@before-previous_target
    energy = .5*u@(k@u)+.5*e@(d*e)
    old_energy = .5*before@(k@before)+.5*old_e@(d*old_e)
    numerical = .5*du@(k@du)+.5*(e-old_e)@(d*(e-old_e)); viscous = 0.
    for a,t,memory,old_memory in zip(alpha,times,z,old_z):
        v = u-memory; old_v = before-old_memory; dz = memory-old_memory; dv = v-old_v
        energy += .5*a*v@(k@v); old_energy += .5*a*old_v@(k@old_v)
        viscous += a*t/h*dz@(k@dz); numerical += .5*a*dv@(k@dv)
    work = float((-d*e)@(target-previous_target))
    return dict(energyNmm=float(energy), oldEnergyNmm=float(old_energy),
        dissipatedEnergyNmm=float(viscous), numericalDissipationNmm=float(numerical),
        workNmm=work, energyBalanceResidualNmm=work-float(energy-old_energy)-float(viscous)-float(numerical))


def field_metrics(geometry, free, u, memories, common_rows, alpha):
    full = np.zeros(geometry['nodes'].size); full[free] = u
    en = geometry['elementNodes']; gradients = element_operators(geometry['affine'])[2]
    local = full.reshape((-1,3))[en]
    gradient = np.einsum('eai,gaj->egij', local, gradients)
    max_gradient = float(np.linalg.norm(gradient, axis=(2,3)).max())
    max_displacement = float(np.linalg.norm(local.mean(axis=1), axis=1).max())
    # The first eight points are integration quadrature; the additional eight
    # corners only guard the small-gradient domain and carry no volume weight.
    probe_gradient = gradient[common_rows, :8]
    eps = (probe_gradient+probe_gradient.transpose(0,1,3,2))/2
    mu = geometry['muPa'][common_rows]; nu = geometry['nu'][common_rows]
    lam = 2*mu*nu/(1-2*nu)
    stress = 2*mu[:,None,None,None]*eps+lam[:,None,None,None]*np.trace(eps,axis1=2,axis2=3)[:,:,None,None]*np.eye(3)
    for a, memory in zip(alpha, memories):
        field = np.zeros_like(full); field[free] = u-memory
        g = np.einsum('eai,gaj->egij', field.reshape((-1,3))[en[common_rows]], gradients[:8])
        strain = (g+g.transpose(0,1,3,2))/2
        stress += a*(2*mu[:,None,None,None]*strain+lam[:,None,None,None]*np.trace(strain,axis1=2,axis2=3)[:,:,None,None]*np.eye(3))
    weight = abs(float(np.linalg.det(geometry['affine'][:3,:3])))/8
    return np.asarray([np.sum(eps**2)*weight, np.sum(stress**2)*weight]), max_gradient, max_displacement


def contact_matrix(geometry, cells):
    indexes = {int(cell): i for i,cell in enumerate(geometry['cells'])}; rows=[]; columns=[]
    if cells.dtype.kind not in 'iu' or len(cells) != len(np.unique(cells)):
        raise ValueError('Source contacts must have unique integer owners')
    for i,cell in enumerate(cells):
        if int(cell) not in indexes: raise ValueError('Contact outside frozen reference')
        for axis in range(3):
            rows.extend([3*i+axis]*8); columns.extend(3*geometry['elementNodes'][indexes[int(cell)]]+axis)
    return sparse.coo_matrix((np.full(len(rows),1/8.),(rows,columns)),shape=(3*len(cells),geometry['nodes'].size)).tocsr()


def worker(output, launch):
    output.mkdir(parents=True, exist_ok=False); p=json.loads(PROTOCOL.read_text()); identity=json.loads(launch.read_text())
    started=time.perf_counter(); metrics=[]; samples=[]; payloads={}; branches=[]
    report={'schema':'resectionlab.continuum-exterior-qualification-result.v1','status':'started','phase':'admission',
        'sourceCommit':identity['commit'],'sourceHashes':identity['sourceHashes'],'protocolSha256':sha(PROTOCOL),
        'sourceRole':p['sourceRole'],'physicalValidationPassed':False,'patientMechanicsValidated':False,
        'physiologicalRL':'hold','largeScaleRL':'hold','trainingUpdates':0,'newAcquiredDatasetBytes':0,
        'liveModelCapsChanged':False,'completeSurgicalPlans':False,'bodyForce':'zero',
        'artificialRadiusStability':'previous16-to-20mm stress gates remain failed'}
    def check():
        if time.perf_counter()-started>p['caps']['wallSeconds']-10: raise TimeoutError('Exterior qualification wall cap')
        if sum(f.stat().st_size for f in output.rglob('*') if f.is_file())>p['caps']['outputBytes']-65536:
            raise ValueError('Exterior qualification output cap')
    try:
        if identity['protocolSha256']!=sha(PROTOCOL): raise ValueError('Launch protocol changed')
        for path,digest in identity['sourceHashes'].items():
            if sha(ROOT/path)!=digest: raise ValueError('Launch source changed')
        if p['bodyForce']!='zero': raise ValueError('This source witness has no body load')
        bound(p['independentChecker']); bound(p['independentCheckerSelfTest'])
        reference_report=json.loads(bound(p['referenceReport']).read_text())
        if reference_report['sourceArchiveSha256']!=p['sourceArchiveSha256']: raise ValueError('Reference source archive differs')
        geometry,partition_summary,partition=load_geometry(p); free,core=free_partition(geometry,partition)
        with np.load(bound(p['geometry']['reference']),allow_pickle=False) as raw:
            alpha=raw['alpha']; times=raw['times']; spring=raw['spring']; contacts=raw['contactCells']; basis=raw['basis']
            common=raw['commonCells']
        if not set(map(int,contacts))<=set(map(int,partition['coreCells'])): raise ValueError('Contact outside induced core')
        if not np.isfinite(basis).all() or np.any(basis[~geometry['freeDofs']]!=0.): raise ValueError('Reference basis invalid')
        c=contact_matrix(geometry,contacts)[:,free]; basis=basis[free]
        source_rows={int(cell):i for i,cell in enumerate(geometry['cells'])}
        common_rows=np.asarray([source_rows[int(cell)] for cell in common])
        report.update(partition=partition_summary,phase='assembly'); write(output/'report.json',report)
        begin=time.perf_counter(); full=source_stiffness(geometry,check=check); k=full[free][:,free].tocsr(); del full
        report['assemblySeconds']=time.perf_counter()-begin; report['phase']='condensation'; write(output/'report.json',report)
        op=ExteriorOperator(k,core,check=check); del k
        solve=ExteriorHistorySolve(op,c,spring,alpha,times,check=check)
        report.update(operator=op.stats,phase='histories',coreSupportCertified=True); write(output/'report.json',report)
        limits=p['tolerances']; maximum={key:0. for key in limits}
        with np.load(bound(p['loadSchedules']),allow_pickle=False) as loads:
            schedules={b:loads[b.replace('-','_')] for b in p['branches']}
        for branch in p['branches']:
            report.update(activeBranch=branch); begin=time.perf_counter(); u=np.zeros(len(free)); z=np.zeros((len(alpha),len(free)))
            reference_before=u.copy(); reference_old_z=z.copy(); previous_target=np.zeros(len(spring)); cumulative=np.zeros(2)
            with np.load(bound(p['referenceHistories'][branch]),allow_pickle=False) as raw:
                q=raw['coordinates']; w=raw['memoryCoordinates']; expected_rates=raw['fixedProbeRates']; expected_reactions=raw['reactionsN']
            meta=reference_report['loadSchedules'][branch]; h=p['stepSeconds']
            if meta['h']!=h or len(schedules[branch])!=meta['steps'] or len(q)!=meta['steps']:
                raise ValueError('Original source load clock differs')
            for index,target in enumerate(schedules[branch]):
                check(); before=u; old_z=z
                result=solve.step(before,old_z,np.zeros(len(free)),target,h,previous_target=previous_target)
                u=result['displacement']; z=result['memories']; reference_u=basis@q[index]; reference_z=w[index]@basis.T
                account=full_accounting(op.k,c,spring,alpha,times,reference_before,reference_old_z,reference_u,reference_z,target,previous_target,h)
                rates,gradient,displacement=field_metrics(geometry,free,u,z,common_rows,alpha)
                reaction=result['contactReactionN'].reshape((-1,3)).sum(axis=0)
                cumulative+=rates*h
                errors={'displacementMm':float(np.max(np.abs(u-reference_u),initial=0.)),
                    'memoryMm':float(np.max(np.abs(z-reference_z),initial=0.)),
                    'forceN':max(float(np.max(np.abs(reaction-expected_reactions[index]),initial=0.)),
                        float(np.max(np.abs(result['contactReactionN']-spring*(target-c@reference_u)),initial=0.))),
                    'energyNmm':max(abs(result[key]-account[key]) for key in account),
                    'strainSquaredMm3':abs(float(rates[0]-expected_rates[index,0])),
                    'cumulativeStrainSquaredMm3S':abs(float(cumulative[0]-expected_rates[:index+1,0].sum()*h)),
                    'stressSquaredPa2Mm3':abs(float(rates[1]-expected_rates[index,1])),
                    'cumulativeStressSquaredPa2Mm3S':abs(float(cumulative[1]-expected_rates[:index+1,1].sum()*h))}
                for key,error in errors.items():
                    maximum[key]=max(maximum[key],error)
                    if not np.isfinite(error) or error>limits[key]: raise ValueError('Exterior/reference '+key+' disagreement')
                if gradient>p['domain']['maxGradientNorm']+1e-12 or displacement>p['domain']['maxCenterDisplacementMm'] or np.linalg.norm(reaction)>p['domain']['maxReactionN']:
                    raise ValueError('Unchanged gradient/displacement/reaction domain exceeded')
                row={'branch':branch,'step':index+1,'errors':errors,'maxGradientNorm':gradient,
                    'maxCenterDisplacementMm':displacement,'rates':rates.tolist(),'cumulative':cumulative.tolist(),
                    **{key:value for key,value in result.items() if key not in ('displacement','memories','contactReactionN')},
                    'contactReactionN':result['contactReactionN'].tolist()}
                metrics.append(row)
                if index+1 in p['sampleSteps'][branch]:
                    buffer=io.BytesIO(); np.savez_compressed(buffer,freeIndices=free,displacement=u,beforeDisplacement=before,
                        memories=z,beforeMemories=old_z,target=target,previousTarget=previous_target,h=h)
                    content=buffer.getvalue(); digest=hashlib.sha256(content).hexdigest()
                    if digest not in payloads:
                        filename=f'{branch}-step-{index+1:03d}.npz'; (output/filename).write_bytes(content); payloads[digest]=filename
                    samples.append({'branch':branch,'step':index+1,'file':payloads[digest],'sha256':digest,'stats':row})
                    write(output/'samples.json',samples)
                previous_target=target.copy(); reference_before=reference_u; reference_old_z=reference_z
            branches.append({'branch':branch,'steps':len(schedules[branch]),'historySeconds':time.perf_counter()-begin,
                'sourceActionHash':meta['actionHash'],'originalNativeActions':17,'allStepFieldsCompared':True})
            report.update(branches=branches,maximumDisagreements=maximum); write(output/'report.json',report)
            write(output/'history-metrics.json',metrics)
            print(json.dumps({'branch':branch,'elapsedWallSeconds':time.perf_counter()-started}),flush=True)
        write(output/'history-metrics.json',metrics); write(output/'samples.json',samples)
        np.savez_compressed(output/'history-metrics.npz',
            branch=np.asarray([r['branch'] for r in metrics]),step=np.asarray([r['step'] for r in metrics]),
            rates=np.asarray([r['rates'] for r in metrics]),cumulative=np.asarray([r['cumulative'] for r in metrics]),
            errorKeys=np.asarray(list(limits)),errors=np.asarray([[r['errors'][key] for key in limits] for r in metrics]),
            maxFreeResidualN=np.asarray([r['maxFreeResidualN'] for r in metrics]),
            energyBalanceResidualNmm=np.asarray([r['energyBalanceResidualNmm'] for r in metrics]))
        for path,digest in identity['sourceHashes'].items():
            if sha(ROOT/path)!=digest: raise ValueError('Source changed during qualification')
        for entry in [*p['geometry'].values(),p['partition'],p['referenceReport'],p['loadSchedules'],
                      *p['referenceHistories'].values(),p['independentChecker'],p['independentCheckerSelfTest']]:bound(entry)
        report.update(status='completed',phase='completed',branches=branches,maximumDisagreements=maximum,
            sampleCount=len(samples),historyComparisonCount=len(metrics),samplesSha256=sha(output/'samples.json'),
            samplesFile='samples.json',historyMetricsFile='history-metrics.json',
            historyMetricsSha256=sha(output/'history-metrics.json'),
            historyMetricsNpzFile='history-metrics.npz',historyMetricsNpzSha256=sha(output/'history-metrics.npz'),
            sourceInputsUnchanged=True,
            setupWitnessOnly=True,outer20mmPhysicallyAccepted=False)
        check()
    except BaseException as error:
        report.update(status='invalid_or_incomplete',error=type(error).__name__+': '+str(error)); raise
    finally:
        write(output/'samples.json',samples); write(output/'history-metrics.json',metrics)
        report.update(wallSeconds=time.perf_counter()-started,
            peakRssBytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024))
        write(output/'report.json',report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('output',type=Path)
    parser.add_argument('--worker',action='store_true'); parser.add_argument('--launch',type=Path)
    args=parser.parse_args(); output=args.output.resolve()
    if not output.is_relative_to(ROOT/'testcases'): raise ValueError('Evidence stays under testcases/')
    if args.worker:
        if args.launch is None: raise ValueError('Supervised launch required')
        worker(output,args.launch); return
    if output.exists(): raise ValueError('Use a fresh output; no automatic retry')
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip(): raise ValueError('Commit prospective source/protocol first')
    launch=output.with_name(output.name+'-launch.json')
    if launch.exists(): raise ValueError('Launch already exists')
    write(launch,{'protocolSha256':sha(PROTOCOL),'sourceHashes':{str(f.relative_to(ROOT)):sha(f) for f in SOURCES},
                  'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()})
    p=json.loads(PROTOCOL.read_text()); environment=os.environ.copy()
    environment.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1')
    workflow_started=time.perf_counter()
    receipt=supervise([sys.executable,str(Path(__file__)),'--worker','--launch',str(launch),str(output/'result')],output,
        cwd=ROOT,environment=environment,seconds=p['caps']['wallSeconds'],rss_bytes=p['caps']['rssBytes'])
    valid=receipt['status']=='completed'; checker_receipt=None
    if valid:
        r=json.loads((output/'result/report.json').read_text())
        valid=r['status']=='completed' and r['historyComparisonCount']==336 and r['sampleCount']==6
    source_complete=valid
    if valid:
        # Release the worker's large factors before the independently frozen
        # export-only process. Both stages share the original wall/output cap.
        remaining=p['caps']['wallSeconds']-(time.perf_counter()-workflow_started)
        if remaining<=0:
            valid=False
        else:
            checker=bound(p['independentChecker']); review=output/'independent'; review.mkdir()
            local_checker=review/checker.name; local_checker.write_bytes(checker.read_bytes())
            checker_receipt=supervise([sys.executable,str(local_checker),str(output/'result'),
                '--protocol',str(PROTOCOL)],output/'independent-supervision',cwd=ROOT,environment=environment,
                seconds=min(60.,remaining),rss_bytes=p['caps']['rssBytes'])
            valid=checker_receipt['status']=='completed'
            if valid:
                checked=json.loads((review/'source-export-review.json').read_text())
                valid=(checked['status']=='passed-sampled-source-export-checks'
                       and checked['scriptSha256']==p['independentChecker']['sha256']
                       and checked['reportSha256']==sha(output/'result/report.json'))
    workflow_seconds=time.perf_counter()-workflow_started
    if (workflow_seconds>p['caps']['wallSeconds'] or
            sum(f.stat().st_size for f in output.rglob('*') if f.is_file())>p['caps']['outputBytes']):valid=False
    write(output/'acceptance.json',{'numericalExecutionComplete':valid,'physicalValidationPassed':False,
        'physiologicalRL':'hold','supervisionStatus':receipt['status'],'sourceWorkerComplete':source_complete,
        'independentVerificationComplete':checker_receipt is not None and checker_receipt['status']=='completed',
        'workflowWallSeconds':workflow_seconds})
    if not valid: raise RuntimeError('Exterior witness failed/incomplete; evidence retained')
    print(json.dumps({'wallSeconds':r['wallSeconds'],'maximumDisagreements':r['maximumDisagreements']}))


if __name__=='__main__':main()
