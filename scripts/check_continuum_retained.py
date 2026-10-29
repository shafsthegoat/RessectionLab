#!/usr/bin/env python3
"""Export-only retained-source pilot check: independent Q1, no app imports/solve."""
from pathlib import Path
import argparse
import hashlib
import json
import runpy
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
GEOMETRY_KEYS=('cells','shape','nodes','elementNodes','affine','freeDofs','muPa','nu')


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def check(directory,protocol):
    started=time.perf_counter();p=json.loads(protocol.read_text());report=json.loads((directory/'report.json').read_text())
    if report['status']!='completed' or report['protocolSha256']!=sha(protocol):raise ValueError('Pilot protocol/status mismatch')
    for path,digest in report['sourceHashes'].items():
        if sha(ROOT/path)!=digest:raise ValueError('Pilot source changed')
    def bound(entry):
        path=ROOT/entry['path']
        if sha(path)!=entry['sha256']:raise ValueError('Frozen input changed')
        return path
    independent=runpy.run_path(str(bound(p['independentQuadrature'])))
    elastic=independent['elastic_fields'];quadrature=independent['quadrature'];near=independent['near']
    with np.load(bound(p['geometry']['reference']),allow_pickle=False) as raw:
        alpha,times,spring,contacts,common,affine=(raw[key] for key in ('alpha','times','spring','contactCells','commonCells','affine'))
    with np.load(bound(p['baselineWorld']),allow_pickle=False) as raw:seed={k:raw[k] for k in raw.files}
    gradients,weight=quadrature(affine)
    limits=p['tolerances'];results=[];previous=None;removed_owners=set();initial_energy=None
    accumulated_removed=0.;accumulated_jump=0.;accumulated_loss=0.;accumulated_work=0.
    def tissue(g,field,memories):
        force,energy=elastic(field[g['elementNodes']],g['muPa'],g['nu'],gradients,weight)
        for a,memory in zip(alpha,memories):
            branch,stored=elastic((field-memory)[g['elementNodes']],g['muPa'],g['nu'],gradients,weight)
            force+=a*branch;energy+=a*stored
        return force,energy
    for index,row in enumerate(report['states']):
        if row['state']!=index:raise ValueError('Pilot row order changed')
        with np.load(bound(row['geometry']),allow_pickle=False) as raw:g={key:raw[key] for key in GEOMETRY_KEYS}
        path=directory/row['fieldsFile']
        if path.parent.resolve()!=directory.resolve() or sha(path)!=row['fieldsSha256']:raise ValueError('Field export changed')
        with np.load(path,allow_pickle=False) as raw:d={key:raw[key] for key in raw.files}
        n=len(g['nodes']);en=g['elementNodes'];free=g['freeDofs'];indices=np.flatnonzero(free)
        if not np.array_equal(d['freeIndices'],indices):raise ValueError('Free support/field order mismatch')
        if len(np.unique(g['cells']))!=len(g['cells']):raise ValueError('Duplicate source owner')
        expected=(np.column_stack(np.unravel_index(g['cells'],tuple(g['shape'])))[:,None,:]+independent['CORNERS']-.5)@g['affine'][:3,:3].T+g['affine'][:3,3]
        near(g['nodes'][en],expected,1e-10,'Source affine/corner identity')
        arrays=[]
        for key,memory in [('displacement',False),('memories',True),('beforeDisplacement',False),('beforeMemories',True)]:
            shape=(len(alpha),len(indices)) if memory else (len(indices),)
            a=independent['finite'](d[key],shape,key);full=np.zeros((len(alpha),n*3) if memory else n*3)
            if memory:full[:,indices]=a;full=full.reshape((len(alpha),n,3))
            else:full[indices]=a;full=full.reshape((n,3))
            arrays.append(full)
        u,z,before,oldz=arrays;h=float(d['h'])
        if h!=p['stepSeconds']:raise ValueError('Pilot physical step mismatch')
        near(z,(times/(times+h))[:,None,None]*oldz+(h/(times+h))[:,None,None]*u,1e-9,'Full memory BE update')
        if index==0:
            near(d['beforeDisplacement'],seed['displacement'],0.,'Nonzero initial displacement')
            near(d['beforeMemories'],seed['memories'],0.,'Nonzero initial memory')
        else:
            lookup={int(cell):i for i,cell in enumerate(previous['geometry']['cells'])}
            rows=np.asarray([lookup[int(cell)] for cell in g['cells']]);old_en=previous['geometry']['elementNodes'][rows]
            if (not np.array_equal(before[en].view(np.uint64),previous['u'][old_en].view(np.uint64))
                    or not np.array_equal(oldz[:,en].view(np.uint64),previous['z'][:,old_en].view(np.uint64))):
                raise ValueError('Retained source corner displacement/history changed during event')
            if (not np.array_equal(g['muPa'],previous['geometry']['muPa'][rows])
                    or not np.array_equal(g['nu'],previous['geometry']['nu'][rows])
                    or not np.array_equal(free.reshape((-1,3))[en],previous['geometry']['freeDofs'].reshape((-1,3))[old_en])):
                raise ValueError('Retained source material/support changed')
        local,energy=tissue(g,u,z);_,oldenergy=tissue(g,before,oldz)
        _,numerical=elastic((u-before)[en],g['muPa'],g['nu'],gradients,weight);viscous=np.zeros(len(en))
        for a,tau,memory,old_memory in zip(alpha,times,z,oldz):
            _,increment=elastic(((u-memory)-(before-old_memory))[en],g['muPa'],g['nu'],gradients,weight);numerical+=a*increment
            _,memoryloss=elastic((memory-old_memory)[en],g['muPa'],g['nu'],gradients,weight);viscous+=2*a*tau/h*memoryloss
        lookup={int(cell):i for i,cell in enumerate(g['cells'])};m=len(d['target'])
        cn=en[np.asarray([lookup[int(cell)] for cell in contacts],dtype=int)] if m else np.empty((0,8),int)
        springs=spring if m else np.empty(0)
        cu=u[cn].mean(axis=1).ravel();cb=before[cn].mean(axis=1).ravel()
        e=cu-d['target'];eb=cb-d['previousTarget'];reaction=-springs*e
        contact=.5*e@(springs*e);oldcontact=.5*eb@(springs*eb)
        contactnum=.5*(e-eb)@(springs*(e-eb))
        force=np.zeros_like(u);np.add.at(force,en,local)
        for nodes,contact_force in zip(cn,(springs*e).reshape((-1,3))):np.add.at(force,nodes,contact_force/8.)
        force_residual=float(np.max(np.abs(force.ravel()[free]),initial=0.))
        if force_residual>limits['forceN']:raise ValueError('Independent full Q1 free equilibrium failed')
        total=float(energy.sum()+contact);old_total=float(oldenergy.sum()+oldcontact)
        diss=float(viscous.sum());num=float(numerical.sum()+contactnum);work=float(reaction@(d['target']-d['previousTarget']))
        balance=work-(total-old_total)-diss-num
        if abs(balance)>limits['balanceNmm'] or diss < -1e-12 or num < -1e-12:raise ValueError('Independent step energy/dissipation failed')
        event_removed=0.;jump=0.
        if previous is not None:
            gone=set(map(int,previous['geometry']['cells']))-set(map(int,g['cells']))
            if removed_owners.intersection(gone):raise ValueError('Duplicate removal energy')
            oldlookup={int(cell):i for i,cell in enumerate(previous['geometry']['cells'])}
            event_removed=float(sum(previous['energy'][oldlookup[cell]] for cell in sorted(gone)))
            if gone and (row['event'] is None or sorted(gone)!=row['event']['removedSourceIds']):raise ValueError('Source removal ledger mismatch')
            removed_owners.update(gone);jump=float(oldcontact-previous['contact'])
            near(oldenergy.sum()-previous['energy'].sum()+event_removed,0.,limits['energyNmm'],'Tissue event balance')
            near(old_total,previous['total']-event_removed+jump,limits['energyNmm'],'Full instantaneous event balance')
        else:initial_energy=old_total
        near(row['removedStoredEnergyNmm'],event_removed,limits['energyNmm'],'Removed source energy')
        near(row['contactModelEnergyJumpNmm'],jump,limits['energyNmm'],'Contact model energy jump')
        for key,value in dict(energyNmm=total,oldEnergyNmm=old_total,dissipatedEnergyNmm=diss,
            numericalDissipationNmm=num,workNmm=work,energyBalanceResidualNmm=balance).items():
            near(row[key],value,limits['energyNmm'],'Recorded '+key)
        near(row['contactReactionN'],reaction,limits['forceN'],'Full contact reaction')
        g.update(alpha=alpha,commonRows=np.asarray([lookup[int(cell)] for cell in common]))
        rates,gradient,center=independent['field_probe_metrics'](g,u,z)
        near(row['rates'][0],rates[0],1e-9,'Strain probe rate')
        near(row['rates'][1],rates[1],.001,'Stress probe rate')
        near(row['maxGradientNorm'],gradient,1e-10,'All-point gradient bound')
        near(row['maxCenterDisplacementMm'],center,1e-10,'Center displacement bound')
        if (gradient>p['domain']['maxGradientNorm'] or center>p['domain']['maxCenterDisplacementMm']
                or np.linalg.norm(reaction.reshape((-1,3)).sum(axis=0))>p['domain']['maxReactionN']):raise ValueError('Physical approximation domain exceeded')
        accumulated_removed+=event_removed;accumulated_jump+=jump;accumulated_loss+=diss+num;accumulated_work+=work
        cumulative=total-initial_energy+accumulated_removed-accumulated_jump+accumulated_loss-accumulated_work
        if abs(cumulative)>limits['balanceNmm']:raise ValueError('Independent cumulative event/step identity failed')
        results.append({'state':index,'freeResidualN':force_residual,'stepBalanceResidualNmm':balance,
            'removedStoredEnergyNmm':event_removed,'contactModelEnergyJumpNmm':jump,'cumulativeBalanceResidualNmm':cumulative})
        previous=dict(geometry=g,u=u,z=z,energy=energy,contact=contact,total=total)
    if (len(results)!=11 or sorted(removed_owners)!=sorted(p['expectedRemovalOrder']) or not np.any(z)
            or report['fullExteriorBuilds']!=2 or report['retainedCoreBuilds']!=7):
        raise ValueError('Independent pilot coverage incomplete')
    result={'schema':'resectionlab.continuum-retained-pilot-independent.v1','status':'passed','samples':len(results),
        'removedOwners':sorted(removed_owners),'states':results,'sourceSolves':0,'appImports':0,
        'physicalValidationPassed':False,'physiologicalRL':'hold','wallSeconds':time.perf_counter()-started}
    (directory/'independent-check.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directory',type=Path);parser.add_argument('protocol',type=Path)
    args=parser.parse_args();check(args.directory,args.protocol)
