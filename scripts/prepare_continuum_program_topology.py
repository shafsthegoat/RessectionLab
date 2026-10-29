#!/usr/bin/env python3
"""Audit exact exterior-block identities across captured removal states; no solve."""
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from resectionlab.surgical_continuum_exterior import source_stiffness
from resectionlab.surgical_continuum_topology import SourceDomain
from profile_continuum_exterior import load_geometry,free_partition,bound,sha,write
from febio_runtime import supervise

PROTOCOL = ROOT/'manifests/experiments/continuum-program-topology-v1.json'
SOURCES = [Path(__file__),ROOT/'scripts/profile_continuum_exterior.py',ROOT/'scripts/febio_runtime.py',
    *[ROOT/'src/resectionlab'/name for name in ('surgical_continuum_topology.py','surgical_continuum_events.py',
        'surgical_continuum_exterior.py','surgical_continuum_partition.py','surgical_continuum.py')]]


def compare_sparse(left,right):
    a=left.tocsr(copy=True);b=right.tocsr(copy=True)
    for matrix in (a,b):matrix.sum_duplicates();matrix.eliminate_zeros();matrix.sort_indices()
    same_pattern=(a.shape==b.shape and np.array_equal(a.indptr,b.indptr) and np.array_equal(a.indices,b.indices))
    exact=same_pattern and np.array_equal(a.data.view(np.uint64),b.data.view(np.uint64))
    if a.shape!=b.shape:return {'bitExact':False,'samePattern':False,'shapeDiffers':True}
    difference=a-b
    return {'bitExact':bool(exact),'samePattern':bool(same_pattern),
        'nonzeroCoefficientDifferences':int(difference.nnz),
        'maximumCoefficientDifferenceNPerMm':float(np.max(np.abs(difference.data),initial=0.))}


def worker(output,launch):
    output.mkdir(parents=True,exist_ok=False);p=json.loads(PROTOCOL.read_text());identity=json.loads(launch.read_text())
    started=time.perf_counter();states=[]
    report={'schema':'resectionlab.continuum-program-topology-result.v1','status':'started','phase':'admission',
        'sourceCommit':identity['commit'],'sourceHashes':identity['sourceHashes'],'protocolSha256':sha(PROTOCOL),
        'scope':p['scope'],'physicalValidationPassed':False,'physiologicalRL':'hold','largeScaleRL':'hold',
        'mechanicsSolves':0,'factorizations':0,'newAcquiredDatasetBytes':0,'largerDomainPublicActionsAdmitted':False}
    def check():
        if time.perf_counter()-started>p['caps']['wallSeconds']-5:raise TimeoutError('Topology identity cooperative cap')
        if sum(f.stat().st_size for f in output.rglob('*') if f.is_file())>p['caps']['outputBytes']-65536:
            raise ValueError('Topology output cap')
    def check_sources():
        if identity['protocolSha256']!=sha(PROTOCOL):raise ValueError('Launch protocol changed')
        for path,digest in identity['sourceHashes'].items():
            if sha(ROOT/path)!=digest:raise ValueError('Launch source changed')
        for entry in [*p['geometry'].values(),p['partition'],p['captureReport'],*p['capturedPrograms'].values()]:bound(entry)
    try:
        check_sources();capture=json.loads(bound(p['captureReport']).read_text())
        if capture['status']!='completed' or capture['suffixFixedSteps']!=4218:raise ValueError('Native input admission incomplete')
        branch_states={};orders={}
        for branch,entry in p['capturedPrograms'].items():
            data=json.loads(gzip.decompress(bound(entry).read_bytes()));prefix=set(data['prefixActivation']['removedSourceIds'])
            if not data['prefixActivation']['continuumFieldsZero'] or data['prefixActivation']['nativeRemainderSeconds']!=0.:
                raise ValueError('Post-exposure zero-field activation missing')
            masks=[()];order=[];last=set()
            for tick in data['ticks']:
                if tick['actionIndex']<15:continue
                topology=data['topologies'][tick['topologyId']]
                if topology['cutFaces'] or topology['rigidSourceIds']:raise ValueError('Source audit declares no cuts/rigid changes')
                whole=set(topology['removedSourceIds'])
                if not prefix<=whole:raise ValueError('Prefix ownership lost')
                removed=whole-prefix
                if not last<=removed:raise ValueError('Nonmonotone source program')
                if removed!=last:
                    fresh=sorted(removed-last)
                    if len(fresh)!=1:raise ValueError('Frozen program event count changed')
                    order.extend(fresh);masks.append(tuple(sorted(removed)))
                last=removed
            branch_states[branch]=masks;orders[branch]=order
        if (orders['short-hold']!=p['expectedRemovalOrder'] or orders['sustained-hold']!=p['expectedRemovalOrder']
                or orders['stop-at-model-limit'] or branch_states['short-hold']!=branch_states['sustained-hold']):
            raise ValueError('Source event sequence differs')
        geometry,summary,partition=load_geometry(p);free,core=free_partition(geometry,partition)
        begin=time.perf_counter();domain=SourceDomain(geometry,core)
        report.update(phase='geometry-and-assembly',sourcePartition=summary,
            baselineGraphAdmissionSeconds=time.perf_counter()-begin,states=states);write(output/'report.json',report)
        original_exterior=set(range(len(free)))-set(map(int,core))
        group=None;groups=0;all_core_exact=True;baseline_arrays=geometry
        for index,mask in enumerate(branch_states['short-hold']):
            check();begin=time.perf_counter();derived=domain.after(mask);g=derived['geometry']
            geometry_seconds=time.perf_counter()-begin;f=np.flatnonzero(g['freeDofs']);i=derived['coreFreeDofs'];e=np.setdiff1d(np.arange(len(f)),i)
            lineage=derived['originalFreeDofForNew'];il=lineage[i];el=lineage[e]
            begin=time.perf_counter();full=source_stiffness(g,check=check);k=full[f][:,f].tocsr();del full
            assembly_seconds=time.perf_counter()-begin;ee=k[e][:,e].tocsr();ie=k[i][:,e].tocsr();del k
            old_removed=set() if group is None else group['removed'];new_owners=set(mask)-old_removed
            touches_e=[]
            for cell in sorted(new_owners):
                nodes=geometry['elementNodes'][domain.lookup[cell]];dofs=(3*nodes[:,None]+np.arange(3)).ravel()
                original_positions=np.searchsorted(free,dofs[geometry['freeDofs'][dofs]])
                if any(int(dof) in original_exterior for dof in original_positions):touches_e.append(cell)
            group_changed=group is None or bool(touches_e)
            checks=None;reset_difference=None
            if group_changed:
                if group is not None:
                    old_e={int(v):j for j,v in enumerate(group['el'])}
                    if len(old_e)==len(group['el']) and set(map(int,el))==set(old_e):
                        ei=np.array([old_e[int(v)] for v in el])
                        reset_difference=compare_sparse(ee,group['ee'][ei][:,ei])
                groups+=1;group={'removed':set(mask),'ee':ee,'ie':ie,'il':il,'el':el,'id':groups}
            else:
                old_e={int(v):j for j,v in enumerate(group['el'])};old_i={int(v):j for j,v in enumerate(group['il'])}
                if (len(old_e)!=len(group['el']) or len(old_i)!=len(group['il'])
                        or set(map(int,el))!=set(old_e) or not set(map(int,il))<=set(old_i)):
                    raise ValueError('Source lineage cannot identify an exact retained block')
                ei=np.array([old_e[int(v)] for v in el]);ii=np.array([old_i[int(v)] for v in il])
                checks={'exterior':compare_sparse(ee,group['ee'][ei][:,ei]),
                        'coupling':compare_sparse(ie,group['ie'][ii][:,ei])}
                all_core_exact=all_core_exact and checks['exterior']['bitExact'] and checks['coupling']['bitExact']
            if index==0 and all(np.array_equal(g[key],baseline_arrays[key]) for key in g):
                export=dict(p['geometry']['reference']);export['reusedExistingGeometry']=True
            else:
                filename=f'geometry-after-removal-{index}.npz'
                np.savez_compressed(output/filename,**g,coreFreeDofs=i,
                    originalNodeForNew=derived['originalNodeForNew'],originalFreeDofForNew=lineage)
                export={'path':filename,'sha256':sha(output/filename),'reusedExistingGeometry':False}
            row={'state':index,'removedSourceIds':list(mask),'newExteriorContributingOwners':touches_e,
                'exteriorGroup':group['id'],'newExteriorGroup':group_changed,'coreFreeDofs':len(i),'exteriorFreeDofs':len(e),
                'totalFreeDofs':len(f),'totalCornerDofs':g['nodes'].size,'geometrySeconds':geometry_seconds,
                'assemblySeconds':assembly_seconds,'exactBlockComparison':checks,
                'priorExteriorGroupComparison':reset_difference,'geometry':export}
            states.append(row);report.update(states=states);write(output/'report.json',report)
            print(json.dumps({'state':index,'group':group['id'],'checks':checks}),flush=True)
        check_sources();check()
        report.update(status='completed',phase='completed',uniqueTissueStates=len(states),exteriorGroups=groups,
            coreOnlyBlockIdentitiesBitExact=bool(all_core_exact),
            exactFactorReuseCandidate=bool(all_core_exact and groups==2),
            factorReuseAdmitted=False,sourceInputsUnchanged=True,
            nextGate='Implement and independently verify retained exterior factor/Schur update; this geometry/coefficient audit performs no solve.')
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
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():raise ValueError('Commit prospective topology audit first')
    launch=output.with_name(output.name+'-launch.json')
    if launch.exists():raise ValueError('Launch already exists')
    write(launch,{'protocolSha256':sha(PROTOCOL),'sourceHashes':{str(f.relative_to(ROOT)):sha(f) for f in SOURCES},
        'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()})
    p=json.loads(PROTOCOL.read_text());env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1')
    receipt=supervise([sys.executable,str(Path(__file__)),'--worker','--launch',str(launch),str(output/'result')],
        output,cwd=ROOT,environment=env,seconds=p['caps']['wallSeconds'],rss_bytes=p['caps']['rssBytes'])
    valid=receipt['status']=='completed'
    if valid:
        r=json.loads((output/'result/report.json').read_text());valid=r['status']=='completed' and r['uniqueTissueStates']==9
        if sum(f.stat().st_size for f in output.rglob('*') if f.is_file())>p['caps']['outputBytes']:valid=False
    write(output/'acceptance.json',{'topologyIdentityAuditComplete':valid,'factorReuseAdmitted':False,
        'physicalValidationPassed':False,'physiologicalRL':'hold','supervisionStatus':receipt['status']})
    if not valid:raise RuntimeError('Topology audit failed/incomplete; evidence retained')
    print(json.dumps({key:r[key] for key in ('wallSeconds','exteriorGroups','coreOnlyBlockIdentitiesBitExact')}))


if __name__=='__main__':main()
