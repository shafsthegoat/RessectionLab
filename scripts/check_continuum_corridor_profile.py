#!/usr/bin/env python3
"""Independent geometry, retained-block and Q1 checks; no app imports/solves."""
from pathlib import Path
from itertools import product
import argparse,hashlib,json,runpy,time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def check(directory,protocol):
    started=time.perf_counter();directory=Path(directory).resolve();protocol=Path(protocol).resolve()
    p=json.loads(protocol.read_text());r=json.loads((directory/'report.json').read_text())
    def need(ok,message):
        if not ok:raise ValueError(message)
    def sha(file):return hashlib.sha256(file.read_bytes()).hexdigest()
    def bound(entry):
        f=(ROOT/entry['path']).resolve();need(f.is_relative_to(ROOT) and sha(f)==entry['sha256'],'Frozen input hash');return f
    def exported(entry):
        f=(directory/entry['file']).resolve();need(f.is_relative_to(directory) and sha(f)==entry['sha256'],'Export hash');return f
    def npz(file):
        with np.load(file,allow_pickle=False) as f:return {k:f[k] for k in f.files}
    need(directory.is_relative_to(ROOT/'testcases') and r['status']=='completed' and r['sourceInputsUnchanged'],'Incomplete profile')
    need(r['protocolSha256']==sha(protocol),'Protocol changed')
    for v in (p,r):
        need(v['physicalValidationPassed'] is False and v['largerDomainPublicActionsAdmitted'] is False and v['physiologicalRL']=='hold' and v['largeScaleRL']=='hold' and v['trainingUpdates']==v['newAcquiredDatasetBytes']==0,'Scope promotion')
    need(p['caps']==dict(wallSeconds=90,rssBytes=3221225472,outputBytes=8388608,numericalThreads=1,retries=0) and p['workerWallSeconds']==80 and p['checkerWallSeconds']==10,'Changed profile caps')
    need(p['limits']==dict(forceN=1e-10,energyNmm=1e-10,balanceNmm=1e-8,maxGradientNorm=.2,maxCenterDisplacementMm=3.,maxReactionN=.08),'Changed physics checks')
    need(p['tolerances']==dict(displacementMm=1e-9) and r['numericalThreads']==dict(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1'),'Changed comparison/thread budget')
    need(p['algebraicCorePolicy']=='conservative-corridor-retained' and p['profileActualMicroticks']==1 and p['profileActualFixedSteps']==12,'Changed profile policy/scope')
    need(r['wallSeconds']<=80 and r['peakRssBytes']<=3221225472 and r['completePilot'] is False,'Worker limits/scope')
    launch=json.loads(directory.parent.with_name(directory.parent.name+'-launch.json').read_text())
    need(launch['commit']==r['sourceCommit'] and launch['sourceHashes']==r['sourceHashes'] and launch['protocolSha256']==sha(protocol),'Launch binding')
    for path,digest in {**r['sourceHashes'],**r['inputHashes']}.items():need(sha(ROOT/path)==digest,'Changed source/input')
    need(bound(p['independentChecker'])==Path(__file__).resolve(),'Checker identity')
    prior=runpy.run_path(str(bound(p['quadratureChecker'])));q=runpy.run_path(str(bound(p['independentQuadrature'])))
    baseline_path=bound(p['qualifiedPilotReport']);baseline=json.loads(baseline_path.read_text());accept=json.loads(bound(p['qualifiedPilotAcceptance']).read_text())
    need(accept['conditionalClosedLoopPilotComplete'] and baseline['status']=='completed' and baseline['allStepReplayExact'],'Baseline admission')
    root=npz(bound(p['geometry']['reference']));free=np.flatnonzero(root['freeDofs']);lookup={int(c):i for i,c in enumerate(root['cells'])}
    active=[c for c in p['commands'] if c['operation']=='apply'];need(bool(active),'Missing absolute CUSA requests');slot=active[0]['slot']
    need(all(c['slot']==slot and c['pose']==active[0]['pose'] for c in active),'Noncollinear repeated-pose intervention')
    initial=baseline['initial'];a=np.array(initial['slots'][slot]['position'],np.longdouble);b=np.array(active[0]['pose']['position'],np.longdouble)
    A=np.array(root['affine'][:3,:3],np.longdouble);corners=np.array(list(product((-.5,.5),repeat=3)),np.longdouble)
    need(np.allclose(np.asarray(A.T@A,float),np.diag(np.asarray(np.diag(A.T@A),float)),rtol=0,atol=1e-12),'Orthogonal source frame')
    half=float(max(np.sqrt(np.sum((c@A.T)**2)) for c in corners));radius=initial['parametersBySlot'][slot]['radiusMm'];U=p['limits']['maxCenterDisplacementMm']
    limit=float(np.nextafter(radius+U+2*half+1e-9,np.inf));centers=np.array(np.unravel_index(root['cells'],root['shape']),np.longdouble).T@A.T+root['affine'][:3,3]
    v=b-a;length=np.dot(v,v);t=np.minimum(1,np.maximum(0,np.sum((centers-a)*v,axis=1)/length)) if length else np.zeros(len(centers))
    owners=np.flatnonzero(np.sqrt(np.sum((centers-a-t[:,None]*v)**2,axis=1))<=limit)
    rows=np.unique(np.r_[owners,[lookup[c['sourceCell']] for c in initial['contacts']]]).astype(int)
    full=np.unique((root['elementNodes'][rows,:,None]*3+np.arange(3)).ravel());core=np.searchsorted(free,full[root['freeDofs'][full]])
    partition=npz(exported(r['partition']));need(np.array_equal(core,partition['coreFreeDofs']) and np.array_equal(root['cells'][owners],partition['corridorSourceIds']),'Independent conservative corridor/core selection')
    need(hashlib.sha256(core.astype(np.int64).tobytes()).hexdigest()==p['rootCoreFreeDofsSha256']==r['rootCoreFreeDofsSha256'],'Core hash')
    need(hashlib.sha256(root['cells'][owners].astype(np.int64).tobytes()).hexdigest()==p['corridorSourceIdsSha256']==r['corridorSourceIdsSha256'],'Owner hash')
    selection=r['selection'];need(selection['factorRetentionCertified'] is False and selection['corridorSourceCells']==len(owners) and selection['coreFreeDofs']==len(core),'Geometry-to-retention promotion')
    need(abs(selection['restCenterBoundMm']-limit)<1e-12,'Corridor margin')
    auditor=prior['Audit'].__new__(prior['Audit']);auditor.start=started;auditor.deadline=started+10;auditor.directory=directory;auditor.cache={};auditor.geometries={};auditor.local_cache={};auditor.checked=[]
    auditor.p=p;auditor.q=q;auditor.root={k:root[k] for k in prior['KEYS']};auditor.root_lookup=lookup;auditor.root_free=free;auditor.root_core=core
    auditor.labels,auditor.affine,auditor.soft,auditor.target_labels,auditor.rate=auditor.case_labels(bound(p['nativeCase']))
    auditor.alpha=root['alpha'];auditor.times=root['times'];auditor.common=root['commonCells'];auditor.gradients,auditor.weight=q['quadrature'](auditor.affine)
    auditor.force_tol=p['limits']['forceN'];auditor.energy_tol=p['limits']['energyNmm'];auditor.balance_tol=p['limits']['balanceNmm']
    g=auditor.geometry(r['geometry']);hyped=r['hypothetical'];candidate=auditor.geometry(hyped['geometry']);d=npz(exported(r['fields']))
    baseline_geometry=initial['geometry'];physical=npz(bound({'path':str((baseline_path.parent/baseline_geometry['file']).relative_to(ROOT)),'sha256':baseline_geometry['sha256']}))
    need(all(np.array_equal(g[k],physical[k]) for k in prior['KEYS']),'Changed baseline physical domain/material/support')
    need(r['seed']['previousArrayHash']==baseline['seed']['previousArrayHash'] and r['seed']['initialInterpretation']==baseline['seed']['initialInterpretation'],'Changed declared seed')
    current_free=np.flatnonzero(g['freeDofs']);need(np.array_equal(d['freeIndices'],current_free),'Physical field order')
    def fields(u,z):
        x=np.zeros(g['nodes'].size);m=np.zeros((len(auditor.alpha),len(x)));x[current_free]=u;m[:,current_free]=z
        return x.reshape(-1,3),m.reshape(len(auditor.alpha),-1,3)
    u,z=fields(d['displacement'],d['memories']);before,oldz=fields(d['beforeDisplacement'],d['beforeMemories'])
    need(np.array_equal(d['alpha'],auditor.alpha) and np.array_equal(d['times'],auditor.times),'Spectrum')
    row=r['sampleRecord'];contacts=initial['contacts'];need(row['contacts']==contacts,'Spring binding')
    auditor.sampled_step(g,u,z,before,oldz,d['target'],d['previousTarget'],contacts,row,float(d['h']),initial['removedSourceIds'])
    samples_file=bound({'path':str((baseline_path.parent/baseline['samplesFile']).relative_to(ROOT)),'sha256':baseline['samplesSha256']});samples=json.loads(samples_file.read_text());sample=next(s for s in samples if (s['tick'],s['withinTick'])==(0,11))
    reference=npz(bound({'path':str((baseline_path.parent/sample['fieldsFile']).relative_to(ROOT)),'sha256':sample['fieldsSha256']}))
    difference=max(float(np.max(np.abs(d[k]-reference[k]),initial=0.)) for k in ('displacement','memories','beforeDisplacement','beforeMemories'))
    need(difference<=p['tolerances']['displacementMm'] and abs(difference-r['completeFieldDifferenceMm'])<=1e-16,'Complete-field parity')
    for key in ('command','limiter','orderedSourceIds','cusaEvents','resolvedPose','fixedSteps','remainderSeconds','elapsedSeconds','postEndpointCovers'):
        need(r['firstMicrotick'][key]==baseline['ticks'][0][key],'First causal decision '+key)
    event=hyped['event'];owner=hyped['sourceOwner'];need(not hyped['isEpisodeEvent'] and not hyped['appliedToPhysicalState'] and owner==r['firstMicrotick']['orderedSourceIds'][0],'Hypothetical probe scope')
    need(set(candidate['cells'])==set(g['cells'])-{owner} and event['removedSourceIds']==[owner],'Candidate source mask')
    transferred_u,transferred_z=auditor.transfer(g,candidate,u,z);saved=npz(exported(hyped['fields']))
    prior['bits'](saved['displacement'].reshape(-1,3),transferred_u,'Candidate exact u transfer');prior['bits'](saved['memories'].reshape(len(z),-1,3),transferred_z,'Candidate exact memory transfer')
    _,old_energy,_,_=auditor.local(g,u,z);_,new_energy,_,_=auditor.local(candidate,transferred_u,transferred_z);R=float(old_energy[g['_lookup'][owner]])
    for key,value in [('removedStoredEnergyNmm',R),('oldTissueEnergyNmm',old_energy.sum()),('transferredTissueEnergyNmm',new_energy.sum()),('tissueEventBalanceResidualNmm',new_energy.sum()-old_energy.sum()+R)]:prior['near'](event[key],value,1e-10,'Independent '+key)
    # Whole removed elements are in I. An injective source-corner map and
    # unchanged E bijection then preserve E/E and both E/I coefficient blocks.
    old_nodes={int(parent):node for node,parent in enumerate(g['_parents'])};parents=np.array([old_nodes[int(parent)] for parent in candidate['_parents']]);old_free=np.flatnonzero(g['freeDofs']);new_free=np.flatnonzero(candidate['freeDofs'])
    lineage=np.searchsorted(old_free,(3*parents[:,None]+np.arange(3)).ravel()[new_free]);need(len(np.unique(lineage))==len(lineage),'Noninjective retained lineage')
    old_e=np.setdiff1d(np.arange(len(old_free)),g['coreFreeDofs']);new_e=np.setdiff1d(np.arange(len(new_free)),candidate['coreFreeDofs'])
    need(np.array_equal(np.sort(lineage[new_e]),old_e),'Changed E bijection')
    removed_dofs=(3*g['elementNodes'][g['_lookup'][owner],:,None]+np.arange(3)).ravel();removed_free=np.searchsorted(old_free,removed_dofs[g['freeDofs'][removed_dofs]])
    need(set(removed_free)<=set(g['coreFreeDofs']),'Removed source element touched E')
    builds=r['builds'];need(len(builds)==2 and not builds[0]['retainedExterior'] and builds[1]['retainedExterior'] and builds[1]['operator']['exactExteriorAndBothCouplings'] and hyped['exactRetainedExteriorAndBothCouplings'],'Missing exact current coefficient proof')
    need(hyped['rollbackStateHash']==r['firstMicrotick']['afterStateHash'],'Hypothetical rollback binding')
    return {'schema':'resectionlab.continuum-corridor-independent.v1','status':'passed','sourceSolves':0,'appImports':0,'mriDecoded':False,
        'sourceRole':p['sourceRole'],'initialInterpretation':r['seed']['initialInterpretation'],
        'scope':'One actual checkpoint and one separate hypothetical transfer; no original20mm trajectory or complete-pilot qualification',
        'independentCoreFreeDofs':len(core),'independentCorridorSourceCells':len(owners),'completeFieldDifferenceMm':difference,
        'sampleWorlds':1,'freeForceMaximumN':auditor.checked[0]['freeResidualN'],'energyBalanceNmm':auditor.checked[0]['energyBalanceResidualNmm'],
        'hypotheticalRemovedEnergyNmm':R,'exactTransferVerified':True,'sourceGraphRetainedBlockStructureVerified':True,
        'independentBitwiseCoefficientRecomputation':False,'workerExactCoefficientCertificateRequired':True,
        'completePilot':False,'physicalValidationPassed':False,'largerDomainPublicActionsAdmitted':False,'physiologicalRL':'hold','largeScaleRL':'hold','wallSeconds':time.perf_counter()-started}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directory',type=Path);parser.add_argument('protocol',type=Path);args=parser.parse_args()
    try:result=check(args.directory,args.protocol)
    except Exception as error:
        result={'status':'failed','reason':str(error),'completePilot':False,'physicalValidationPassed':False,'largerDomainPublicActionsAdmitted':False,'physiologicalRL':'hold','largeScaleRL':'hold'}
        (args.directory/'checker-result.json').write_text(json.dumps(result,indent=2)+'\n');raise
    (args.directory/'checker-result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
