#!/usr/bin/env python3
"""Independent frozen-input/scalar and sampled full-Q1 export checks; no solve."""
from pathlib import Path
import argparse
import gzip
import hashlib
import json
import runpy
import time
import zipfile

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
KEYS=('cells','shape','nodes','elementNodes','affine','freeDofs','muPa','nu')


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def arrays_hash(u,z):
    digest=hashlib.sha256(b'resectionlab.full-source-fields.v1')
    for value in (u,z):
        value=np.asarray(value,dtype=np.float64);digest.update(str(value.shape).encode());digest.update(value.tobytes())
    return digest.hexdigest()


def record_hash(record):
    return hashlib.sha256(json.dumps(record,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def check(directory,protocol):
    started=time.perf_counter();p=json.loads(protocol.read_text());report=json.loads((directory/'report.json').read_text())
    if report['status']!='completed' or report['protocolSha256']!=sha(protocol):raise ValueError('Protocol/status mismatch')
    if (report['physicalValidationPassed'] is not False or report['largerDomainPublicActionsAdmitted'] is not False
            or report['physiologicalRL']!='hold' or report['largeScaleRL']!='hold'
            or report['newAcquiredDatasetBytes']!=0 or report['trainingUpdates']!=0):raise ValueError('Scope or RL hold violated')
    if len(report['passes'])!=2:raise ValueError('Fresh replay pass coverage differs')
    for path,digest in report['sourceHashes'].items():
        if sha(ROOT/path)!=digest:raise ValueError('Experiment source changed')
    def bound(entry):
        path=ROOT/entry['path']
        if sha(path)!=entry['sha256']:raise ValueError('Frozen input changed')
        return path
    independent=runpy.run_path(str(bound(p['independentQuadrature'])))
    near=independent['near'];elastic=independent['elastic_fields'];limits=p['tolerances']
    topofile=bound(p['topologyReport']);topology=json.loads(topofile.read_text());geometries=[];entries=[]
    for index,state in enumerate(topology['states']):
        entry=p['geometry']['reference'] if index==0 else {
            'path':str((topofile.parent/state['geometry']['path']).relative_to(ROOT)),'sha256':state['geometry']['sha256']}
        with np.load(bound(entry),allow_pickle=False) as raw:g={key:raw[key] for key in KEYS}
        geometries.append(g);entries.append(entry)
    with np.load(bound(p['geometry']['reference']),allow_pickle=False) as raw:
        alpha,times,common,labels,cells=(raw[key] for key in ('alpha','times','commonCells','labels','cells'))
    gradients,weight=independent['quadrature'](geometries[0]['affine'])
    native={name:json.loads(gzip.decompress(bound(entry).read_bytes())) for name,entry in p['capturedPrograms'].items()}
    with zipfile.ZipFile(bound(p['nativeCase'])) as archive:
        mapping=json.loads(archive.read('manifest.json'))['artifacts']['surgical_toolbox']['mapping']
    if sorted(row['id'] for row in mapping if row.get('target'))!=p['targetLabels']:raise ValueError('Source target label mapping differs')
    label_map={int(cell):int(label) for cell,label in zip(cells,labels)}
    volume=abs(float(np.linalg.det(geometries[0]['affine'][:3,:3])))
    path=directory/report['metricsFile']
    if sha(path)!=report['metricsSha256']:raise ValueError('Metric SHA mismatch')
    records=json.loads(gzip.decompress(path.read_bytes()));by_branch={name:[] for name in native};by_key={}
    for row in records:
        record={key:value for key,value in row.items() if key!='recordHash'}
        if record_hash(record)!=row['recordHash']:raise ValueError('Step record hash differs')
        key=(row['branch'],row['step'])
        if key in by_key:raise ValueError('Duplicate source step')
        by_key[key]=row;by_branch[row['branch']].append(row)
    mask_stage={tuple(state['removedSourceIds']):i for i,state in enumerate(topology['states'])}
    summaries={};native_ticks={};expected_samples={}
    def springs(contacts):return np.repeat([contact['stiffnessNPerMm'] for contact in contacts],3)
    def target(contacts):return np.asarray([contact['targetDisplacementMm'] for contact in contacts],float).ravel()
    for name,data in native.items():
        rows=by_branch[name];prefix=data['prefixActivation'];base=set(prefix['removedSourceIds']);suffix=[t for t in data['ticks'] if t['actionIndex']>=15]
        native_ticks[name]=suffix;clock=prefix['nativeRemainderSeconds'];position=0;last=None;loss=0.;work=0.;removed=0.;jumps=0.;exposure=np.zeros(2)
        chain=hashlib.sha256(b'resectionlab.full-source-program.v1')
        tick_stages=[mask_stage[tuple(sorted(set(data['topologies'][tick['topologyId']]['removedSourceIds'])-base))] for tick in suffix]
        last_zero_tick=max(i for i,stage in enumerate(tick_stages) if stage==0)
        seen_owners=set();last_stage=0
        for ti,tick in enumerate(suffix):
            h=tick['fixedStepSeconds'];dt=tick['durationSeconds'];n=tick['nativeFixedSteps']
            if (tick['nativeRemainderBeforeSeconds']!=clock or n<=0 or n!=int(np.floor((clock+dt+1e-12)/h))):
                raise ValueError('Independent native fixed clock failed')
            clock=max(0.,round(clock+dt-n*h,14))
            if clock!=tick['nativeRemainderAfterSeconds']:raise ValueError('Independent remainder failed')
            top=data['topologies'][tick['topologyId']];stage=mask_stage[tuple(sorted(set(top['removedSourceIds'])-base))]
            for within in range(n):
                row=rows[position];position+=1
                if (row['step']!=position or row['suffixTick']!=ti or row['withinTick']!=within or row['tissueState']!=stage
                        or row['h']!=h or row['actionIndex']!=tick['actionIndex'] or row['operation']!=tick['operation'] or row['slot']!=tick['slot']):
                    raise ValueError('Independent step/tick/action coverage failed')
                near(row['target'],target(tick['contacts']),0.,'Frozen source target')
                old_target=target(tick['contacts']) if within else (np.asarray(last['target']) if last is not None else np.zeros(len(row['target'])))
                previous_contacts=([t for t in data['ticks'] if t['actionIndex']<15][-1]['contacts'] if last is None
                                   else suffix[last['suffixTick']]['contacts'])
                old_key=[(c['slot'],c['sourceCell'],c['stiffnessNPerMm']) for c in previous_contacts]
                new_key=[(c['slot'],c['sourceCell'],c['stiffnessNPerMm']) for c in tick['contacts']]
                reasons=[]
                if position==1:reasons.append('first-step')
                if stage==0 and ti==last_zero_tick and within==n-1:reasons.append('initial-stage-end')
                event_expected=stage!=last_stage
                if event_expected:
                    if within!=0 or stage!=last_stage+1:raise ValueError('Independent event order failed')
                    reasons.append('source-event')
                    owners=sorted(set(topology['states'][stage]['removedSourceIds'])-set(topology['states'][last_stage]['removedSourceIds']))
                    if (row['event'] is None or row['event']['removedSourceIds']!=owners or row['event']['removedElementCount']!=len(owners)
                            or seen_owners.intersection(owners)):raise ValueError('Independent source-event coverage failed')
                    seen_owners.update(owners)
                elif row['event'] is not None:raise ValueError('Unexpected source event')
                if old_key!=new_key and not tick['contacts'] and within==0:reasons.append('release')
                if ti+1==len(suffix) and within+1==n:reasons.append('final')
                if reasons:expected_samples[(name,position)]=set(reasons)
                if old_key!=new_key:old_target=target(tick['contacts'])
                near(row['previousTarget'],old_target,0.,'Moving target charged once per tick')
                reaction=np.asarray(row['contactReactionN']);expected_work=float(reaction@(np.asarray(row['target'])-row['previousTarget']))
                near(row['workNmm'],expected_work,limits['energyNmm'],'Target work')
                previous_energy=0. if last is None else last['energyNmm']
                event_removed=0. if row['event'] is None else row['event']['removedStoredEnergyNmm']
                removed+=event_removed;jumps+=row['contactModelEnergyJumpNmm'];loss+=row['dissipatedEnergyNmm']+row['numericalDissipationNmm'];work+=expected_work
                near(row['oldEnergyNmm'],previous_energy-event_removed+row['contactModelEnergyJumpNmm'],limits['energyNmm'],'Instantaneous event accounting')
                near(row['energyNmm']-row['oldEnergyNmm']+row['dissipatedEnergyNmm']+row['numericalDissipationNmm']-expected_work,0.,limits['balanceNmm'],'BE identity')
                for key,value in [('removedStoredEnergyNmm',removed),('cumulativeContactModelEnergyJumpNmm',jumps),('cumulativeLossNmm',loss),('cumulativeWorkNmm',work)]:
                    near(row[key],value,limits['energyNmm'],'Accumulated '+key)
                near(row['energyNmm']+removed-jumps+loss-work,0.,limits['balanceNmm'],'Cumulative event/BE identity')
                exposure+=np.asarray(row['rates'])*h
                near(row['cumulativeExposure'][0],exposure[0],1e-9,'Cumulative strain exposure')
                near(row['cumulativeExposure'][1],exposure[1],.001,'Cumulative stress exposure')
                if (row['maxFreeResidualN']>limits['forceN'] or row['maxGradientNorm']>p['domain']['maxGradientNorm']
                        or row['maxCenterDisplacementMm']>p['domain']['maxCenterDisplacementMm'] or row['maxHandleReactionN']>p['domain']['maxReactionN']
                        or row['dissipatedEnergyNmm'] < -1e-12 or row['numericalDissipationNmm'] < -1e-12):raise ValueError('Unchanged source domain/residual guard failed')
                if last is not None and row['previousArrayHash']!=last['arrayHash']:raise ValueError('Previous field hash chain broken')
                if last is None and row['previousArrayHash']!=arrays_hash(np.zeros(geometries[0]['nodes'].size),np.zeros((len(alpha),geometries[0]['nodes'].size))):
                    raise ValueError('Nonzero initial source history')
                core_record={key:value for key,value in row.items() if key not in ('recordHash','sourceOperatorIdentity')}
                chain.update(bytes.fromhex(record_hash(core_record)));last=row;last_stage=stage
        if position!=len(rows) or position!=data['summary']['suffixFixedSteps']:raise ValueError('Native program step coverage incomplete')
        summary=report['passes'][0]['branches'][name]
        if summary['replayChainSha256']!=chain.hexdigest() or summary['arrayHash']!=last['arrayHash']:raise ValueError('Final replay hash mismatch')
        if summary['fullSourceAccounting']!=data['summary']['fullSourceAccounting'] or summary['unmodeledPrefixAccounting']!=prefix['fullSourceAccounting']:
            raise ValueError('Whole-source/prefix accounting changed')
        target_removed=sum(volume for cell in seen_owners if label_map[cell] in p['targetLabels'])
        source=summary['fullSourceAccounting'];base_accounting=prefix['fullSourceAccounting']
        if (base_accounting['sourceCellCount']+len(seen_owners)!=source['sourceCellCount']
                or base_accounting['targetRemovedMm3']+target_removed!=source['targetRemovedMm3']
                or base_accounting['nonTargetRemovedMm3']+len(seen_owners)*volume-target_removed!=source['nonTargetRemovedMm3']):
            raise ValueError('Independent source-label conservation failed')
        if (summary['steps']!=position or summary['ticks']!=len(suffix) or summary['finalTissueState']!=last_stage
                or summary['removedSourceIds']!=sorted(seen_owners) or sorted(seen_owners)!=data['summary']['additionalRemovedSourceIds']
                or summary['publicActions']!=len(data['actions']) or not summary['nativeActionReceiptsPreserved']
                or summary['largerDomainPublicActionsAdmitted'] is not False):raise ValueError('Final branch coverage/scope differs')
        for key,value in [('energyNmm',last['energyNmm']),('removedStoredEnergyNmm',removed),('contactModelEnergyJumpNmm',jumps),
                          ('workNmm',work),('lossNmm',loss),('cumulativeEventBalanceResidualNmm',last['cumulativeEventBalanceResidualNmm'])]:
            near(summary[key],value,limits['energyNmm'],'Final branch '+key)
        near(summary['cumulativeExposure'][0],exposure[0],1e-9,'Final strain exposure')
        near(summary['cumulativeExposure'][1],exposure[1],.001,'Final stress exposure')
        summaries[name]={'steps':position,'chainSha256':chain.hexdigest(),'energyNmm':last['energyNmm'],'cumulativeExposure':exposure.tolist()}
    if len(records)!=p['expectedFixedSteps'] or report['passes'][0]['branches']!=report['passes'][1]['branches'] or not report['deterministicAllStepReplayExact']:
        raise ValueError('Complete deterministic coverage failed')
    samples_path=directory/'samples.json'
    if sha(samples_path)!=report['samplesSha256']:raise ValueError('Sample index changed')
    samples=json.loads(samples_path.read_text());checked=[]
    sample_keys=[(entry['branch'],entry['step']) for entry in samples]
    if (len(sample_keys)!=len(set(sample_keys)) or set(sample_keys)!=set(expected_samples)
            or any(set(entry['reasons'])!=expected_samples[(entry['branch'],entry['step'])] for entry in samples)):
        raise ValueError('Frozen sample coverage differs')
    def potential(g,u,contacts,t):
        lookup={int(cell):i for i,cell in enumerate(g['cells'])}
        cn=g['elementNodes'][np.asarray([lookup[c['sourceCell']] for c in contacts],int)] if contacts else np.empty((0,8),int)
        centers=u[cn].mean(axis=1).ravel();stretch=centers-t
        return float(.5*stretch@(springs(contacts)*stretch)),cn
    def tissue(g,u,z):
        force,energy=elastic(u[g['elementNodes']],g['muPa'],g['nu'],gradients,weight)
        for a,memory in zip(alpha,z):
            f,e=elastic((u-memory)[g['elementNodes']],g['muPa'],g['nu'],gradients,weight);force+=a*f;energy+=a*e
        return force,energy
    for entry in samples:
        row=by_key[(entry['branch'],entry['step'])];stage=row['tissueState'];g=geometries[stage];en=g['elementNodes'];n=len(g['nodes']);free=g['freeDofs'];indices=np.flatnonzero(free)
        if entry['geometry']!=entries[stage]:raise ValueError('Sample geometry identity differs')
        path=directory/entry['file']
        if path.parent.resolve()!=directory.resolve() or sha(path)!=entry['sha256']:raise ValueError('Sample fields changed')
        with np.load(path,allow_pickle=False) as raw:d={key:raw[key] for key in raw.files}
        if not np.array_equal(d['freeIndices'],indices):raise ValueError('Sample support order differs')
        fields=[]
        for key,memory in [('displacement',False),('memories',True),('beforeDisplacement',False),('beforeMemories',True)]:
            a=independent['finite'](d[key],(len(alpha),len(indices)) if memory else (len(indices),),key)
            full=np.zeros((len(alpha),n*3) if memory else n*3)
            if memory:full[:,indices]=a;full=full.reshape((len(alpha),n,3))
            else:full[indices]=a;full=full.reshape((n,3))
            fields.append(full)
        u,z,before,oldz=fields
        if arrays_hash(u.ravel(),z.reshape((len(alpha),-1)))!=row['arrayHash']:raise ValueError('Sample complete field hash differs')
        previous=by_key.get((entry['branch'],entry['step']-1));oldg=g;old_before=before;old_memory=oldz
        if row['event'] is not None:
            oldg=geometries[stage-1];lookup={int(cell):i for i,cell in enumerate(oldg['cells'])};rows=np.asarray([lookup[int(c)] for c in g['cells']])
            oldnodes=oldg['elementNodes'][rows].ravel();newnodes=en.ravel();old_before=np.full_like(oldg['nodes'],np.nan);old_memory=np.full((len(alpha),len(oldg['nodes']),3),np.nan)
            old_before[oldnodes]=before[newnodes];old_memory[:,oldnodes]=oldz[:,newnodes]
            orphan=np.setdiff1d(np.arange(len(oldg['nodes'])),np.unique(oldnodes))
            if not np.array_equal(orphan,d['orphanOldNodeIndices']):raise ValueError('Vanished source-node coverage differs')
            old_before[orphan]=d['orphanOldDisplacement'];old_memory[:,orphan]=d['orphanOldMemories']
            if (not np.array_equal(old_before[oldnodes].view(np.uint64),before[newnodes].view(np.uint64))
                    or not np.array_equal(old_memory[:,oldnodes].view(np.uint64),oldz[:,newnodes].view(np.uint64))):raise ValueError('Source sector field continuity differs')
        elif len(d['orphanOldNodeIndices']):raise ValueError('Unexpected old-node supplement')
        if arrays_hash(old_before.ravel(),old_memory.reshape((len(alpha),-1)))!=row['previousArrayHash']:
            raise ValueError('Lossless pre-event field reconstruction differs from prior step hash')
        h=independent['scalar'](d['h'],'h')
        if h<=0 or h!=row['h'] or h!=p['stepSeconds']:raise ValueError('Sample physical clock differs')
        near(d['target'],row['target'],0.,'Sample target');near(d['previousTarget'],row['previousTarget'],0.,'Sample previous target')
        near(z,(times/(times+h))[:,None,None]*oldz+(h/(times+h))[:,None,None]*u,1e-9,'All memory fields')
        tick=native_ticks[entry['branch']][row['suffixTick']];contacts=tick['contacts'];k=springs(contacts)
        contact,cn=potential(g,u,contacts,d['target']);oldcontact,_=potential(g,before,contacts,d['previousTarget'])
        local,energy=tissue(g,u,z);_,oldenergy=tissue(g,before,oldz)
        _,numerical=elastic((u-before)[en],g['muPa'],g['nu'],gradients,weight);viscous=np.zeros(len(en))
        for a,tau,memory,old_memory_new in zip(alpha,times,z,oldz):
            _,inc=elastic(((u-memory)-(before-old_memory_new))[en],g['muPa'],g['nu'],gradients,weight);numerical+=a*inc
            _,loss=elastic((memory-old_memory_new)[en],g['muPa'],g['nu'],gradients,weight);viscous+=2*a*tau/h*loss
        centers=u[cn].mean(axis=1).ravel();oldcenters=before[cn].mean(axis=1).ravel();reaction=k*(d['target']-centers)
        force=np.zeros_like(u);np.add.at(force,en,local)
        for nodes,f in zip(cn,(k*(centers-d['target'])).reshape((-1,3))):np.add.at(force,nodes,f/8.)
        maximum=float(np.max(np.abs(force.ravel()[free]),initial=0.))
        if maximum>limits['forceN']:raise ValueError('Independent full source free force failed')
        total=float(energy.sum()+contact);old_total=float(oldenergy.sum()+oldcontact);diss=float(viscous.sum())
        num=float(numerical.sum()+.5*((centers-d['target'])-(oldcenters-d['previousTarget']))@(k*((centers-d['target'])-(oldcenters-d['previousTarget']))))
        work=float(reaction@(d['target']-d['previousTarget']));balance=work-(total-old_total)-diss-num
        for key,value in [('energyNmm',total),('oldEnergyNmm',old_total),('dissipatedEnergyNmm',diss),('numericalDissipationNmm',num),('workNmm',work),('energyBalanceResidualNmm',balance)]:
            near(row[key],value,limits['energyNmm'],'Independent Q1 '+key)
        near(row['contactReactionN'],reaction,limits['forceN'],'Independent contact reaction')
        event_removed=0.
        if row['event'] is not None:
            _,pre_energy=tissue(oldg,old_before,old_memory);oldlookup={int(c):i for i,c in enumerate(oldg['cells'])}
            gone=sorted(set(map(int,oldg['cells']))-set(map(int,g['cells'])))
            if gone!=row['event']['removedSourceIds']:raise ValueError('Removed owner identity differs')
            event_removed=float(sum(pre_energy[oldlookup[c]] for c in gone))
            near(row['event']['removedStoredEnergyNmm'],event_removed,limits['energyNmm'],'Independent source removal energy')
            near(pre_energy.sum()-event_removed,oldenergy.sum(),limits['energyNmm'],'Independent tissue transfer energy')
        old_contacts=([t for t in native[entry['branch']]['ticks'] if t['actionIndex']<15][-1]['contacts'] if previous is None
                      else native_ticks[entry['branch']][previous['suffixTick']]['contacts'])
        old_target=np.zeros(3*len(old_contacts)) if previous is None else np.asarray(previous['target'])
        old_potential,_=potential(oldg,old_before,old_contacts,old_target)
        near(row['contactModelEnergyJumpNmm'],oldcontact-old_potential,limits['energyNmm'],'Independent binding/model jump')
        lookup={int(c):i for i,c in enumerate(g['cells'])};g=dict(g,alpha=alpha,commonRows=np.asarray([lookup[int(c)] for c in common]))
        rates,gradient,center=independent['field_probe_metrics'](g,u,z)
        near(row['rates'][0],rates[0],1e-9,'Independent strain probe');near(row['rates'][1],rates[1],.001,'Independent stress probe')
        near(row['maxGradientNorm'],gradient,1e-10,'Independent corner bound');near(row['maxCenterDisplacementMm'],center,1e-10,'Independent center bound')
        checked.append({'branch':entry['branch'],'step':entry['step'],'freeResidualN':maximum,'energyBalanceResidualNmm':balance,'removedStoredEnergyNmm':event_removed})
    if len(checked)!=p['expectedSampleWorlds']:raise ValueError('Independent source field sample coverage incomplete')
    result={'schema':'resectionlab.full-program-independent.v1','status':'passed','allStepRecords':len(records),
        'sampleWorlds':len(checked),'sourceSolves':0,'appImports':0,'branches':summaries,'samples':checked,
        'physicalValidationPassed':False,'physiologicalRL':'hold','wallSeconds':time.perf_counter()-started}
    (directory/'independent-check.json').write_text(json.dumps(result,indent=2)+'\n');return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directory',type=Path);parser.add_argument('protocol',type=Path)
    args=parser.parse_args();check(args.directory,args.protocol)
