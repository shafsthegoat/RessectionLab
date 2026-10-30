"""Complete generated programs, exact replay and lossless source-event exports."""
from pathlib import Path
import copy
import gzip
import json
import runpy
import shutil
import sys
import zipfile

import numpy as np
import pytest

from resectionlab.surgical_continuum_program import prepare_program
from test_continuum_retained_pilot import generated_fixture


def full_fixture(tmp_path):
    _,pilot,_,_=generated_fixture(tmp_path,orphan_control=True);root=Path(__file__).resolve().parents[1];scripts=root/'scripts'
    sys.path.insert(0,str(scripts))
    try:api=runpy.run_path(str(scripts/'qualify_continuum_programs.py'))
    finally:sys.path.remove(str(scripts))
    sha=api['sha'];h=1/120
    def entry(path):return {'path':str(path.relative_to(root)),'sha256':sha(path)}
    p=json.loads(api['PROTOCOL'].read_text());captures={};data_by_name={}
    topology=json.loads((root/pilot['topologyReport']['path']).read_text());masks=[state['removedSourceIds'] for state in topology['states']]
    with np.load(root/pilot['geometry']['reference']['path'],allow_pickle=False) as raw:cells=raw['contactCells']
    held=lambda target:[{'slot':'left','sourceCell':int(cell),'stiffnessNPerMm':.02/7,'targetDisplacementMm':[target,0.,0.]} for cell in cells]
    for name in ('short-hold','sustained-hold','stop-at-model-limit'):
        ticks=[];time=0.
        def append(stage,contacts,steps,action,operation):
            nonlocal time
            dt=steps*h;time+=dt
            ticks.append({'actionIndex':action,'slot':'left','operation':operation,'topologyId':stage,'bodyForce':'zero',
                'timeSeconds':time,'durationSeconds':dt,'fixedStepSeconds':h,'nativeFixedSteps':steps,
                'nativeRemainderBeforeSeconds':0.,'nativeRemainderAfterSeconds':0.,'nativeContinuumActive':True,'contacts':contacts})
        append(0,held(0.),1,14,'hold');prefix_time=time
        if name=='stop-at-model-limit':append(0,[],2,15,'release');append(0,[],2,16,'inspect')
        else:
            append(0,held(.001),2,15,'move');append(0,held(.001),4 if name=='sustained-hold' else 2,16,'inspect')
            for stage in range(1,9):append(stage,held(.001),2,16+stage,'aspirate')
            append(8,[],2,25,'release');append(8,[],2,26,'inspect')
        added=[] if name=='stop-at-model-limit' else masks[-1];suffix=ticks[1:];actions=17 if name=='stop-at-model-limit' else 29
        accounting=lambda removed:{'sourceCellCount':len(removed),'byLabelMm3':{'target':float(len(removed))},
            'targetRemovedMm3':float(len(removed)),'nonTargetRemovedMm3':0.,'initialTargetMm3':125.,'residualTargetMm3':125.-len(removed)}
        d={'schema':'resectionlab.committed-native-mechanics-inputs.v1','ticks':ticks,
            'topologies':[{'removedSourceIds':mask,'cutFaces':[],'rigidSourceIds':[]} for mask in masks],
            'actions':[{} for _ in range(actions)],'actionReceipts':[{} for _ in range(actions)],
            'prefixActivation':{'actions':15,'continuumFieldsZero':True,'nativeRemainderSeconds':0.,'timeSeconds':prefix_time,
                'removedSourceIds':[],'fullSourceAccounting':accounting([])},
            'summary':{'branch':name,'actions':actions,'canonicalEveryActionExact':True,'canonicalFullSnapshotExact':True,
                'suffixTicks':len(suffix),'suffixFixedSteps':sum(t['nativeFixedSteps'] for t in suffix),
                'suffixDurationSeconds':sum(t['durationSeconds'] for t in suffix),'additionalRemovedSourceIds':added,
                'fullSourceAccounting':accounting(added)}}
        path=tmp_path/(name+'.json.gz');path.write_bytes(gzip.compress(json.dumps(d).encode(),mtime=0))
        captures[name]=entry(path);data_by_name[name]=d
    admission=tmp_path/'admission.json';admission.write_text(json.dumps({'retainedSourcePilotComplete':True}))
    capture_report=tmp_path/'capture-report.json';capture_report.write_text(json.dumps({'status':'completed'}))
    native_case=tmp_path/'generated-native.zip'
    with zipfile.ZipFile(native_case,'w') as archive:
        archive.writestr('manifest.json',json.dumps({'artifacts':{'surgical_toolbox':{'mapping':[{'id':1,'target':True,'name':'Generated tissue'}]}}}))
    p.update(geometry=pilot['geometry'],partition=pilot['partition'],topologyReport=pilot['topologyReport'],
        independentQuadrature=entry(scripts/'continuum_quadrature.py'),retainedPilotAcceptance=entry(admission),
        nativeCaptureReport=entry(capture_report),capturedPrograms=captures,targetLabels=[1],
        nativeCase=entry(native_case),
        expectedFixedSteps=sum(d['summary']['suffixFixedSteps'] for d in data_by_name.values()),
        sourceRole='GENERATED',scope='Generated numerical full-program regression only')
    protocol=tmp_path/'full-protocol.json';protocol.write_text(json.dumps(p));launch=tmp_path/'full-launch.json'
    launch.write_text(json.dumps({'protocolSha256':sha(protocol),'commit':'generated-control','sourceHashes':{}}))
    api['worker'].__globals__['PROTOCOL']=protocol
    return api,p,protocol,launch,data_by_name,masks


def test_full_generated_programs_replay_and_independent_source_fields(tmp_path):
    api,p,protocol,launch,data,masks=full_fixture(tmp_path)
    report=api['worker'](tmp_path/'full-result',launch)
    assert report['deterministicAllStepReplayExact'] and report['sampleWorlds']==26
    assert report['distinctFieldPayloads']<=24
    branches=report['passes'][0]['branches'];stop=branches['stop-at-model-limit']
    assert stop['finalTissueState']==0 and stop['removedSourceIds']==[] and stop['energyNmm']==0.
    assert branches['short-hold']['replayChainSha256']!=branches['sustained-hold']['replayChainSha256']
    checker=runpy.run_path(str(Path(__file__).resolve().parents[1]/'scripts/check_continuum_programs.py'))
    independent=checker['check'](tmp_path/'full-result',protocol)
    assert independent['status']=='passed' and independent['allStepRecords']==p['expectedFixedSteps']
    # Reverse branch scheduling order with actual mechanics: fixed operators
    # may be shared, but no episode field/cache history may leak between them.
    stages=api['load_stages'](p)
    with np.load(api['bound'](p['geometry']['reference']),allow_pickle=False) as raw:alpha,times,common=(raw[k] for k in ('alpha','times','commonCells'))
    programs={name:prepare_program(name,data[name],masks) for name in reversed(list(data))}
    summaries,_=api['run_pass'](programs,stages,alpha,times,common,p,lambda *args:None,lambda:None,lambda *args:None)
    assert summaries==branches
    # A checksum-consistent one-ULP corruption in a vanished, nonzero old node
    # must still fail the pre-event full-field hash reconstruction.
    corrupt=tmp_path/'corrupt-result';shutil.copytree(tmp_path/'full-result',corrupt)
    samples=json.loads((corrupt/'samples.json').read_text());chosen=None
    for sample in samples:
        with np.load(corrupt/sample['file'],allow_pickle=False) as raw:fields={k:raw[k] for k in raw.files}
        if np.any(fields['orphanOldDisplacement']):chosen=sample;break
    assert chosen is not None
    fields['orphanOldDisplacement'].flat[0]=np.nextafter(fields['orphanOldDisplacement'].flat[0],np.inf)
    np.savez_compressed(corrupt/'corrupt-orphan.npz',**fields)
    chosen.update(file='corrupt-orphan.npz',sha256=api['sha'](corrupt/'corrupt-orphan.npz'))
    (corrupt/'samples.json').write_text(json.dumps(samples));damaged=json.loads((corrupt/'report.json').read_text())
    damaged['samplesSha256']=api['sha'](corrupt/'samples.json');(corrupt/'report.json').write_text(json.dumps(damaged))
    with pytest.raises(ValueError,match='Lossless pre-event field reconstruction'):
        checker['check'](corrupt,protocol)
    for fault,pattern in [('duplicate-sample','Frozen sample coverage'),('negative-h','Sample physical clock'),
                          ('scope','Scope or RL hold'),('dropped-event','source-event coverage'),
                          ('final-energy','Final branch energyNmm')]:
        damaged_dir=tmp_path/fault;shutil.copytree(tmp_path/'full-result',damaged_dir)
        damaged=json.loads((damaged_dir/'report.json').read_text())
        index=json.loads((damaged_dir/'samples.json').read_text())
        if fault=='duplicate-sample':
            victim=next(i for i,s in enumerate(index) if 'source-event' in s['reasons'])
            index[victim]=dict(index[0])
        elif fault=='negative-h':
            item=next(s for s in index if s['branch']=='stop-at-model-limit')
            with np.load(damaged_dir/item['file'],allow_pickle=False) as raw:fields={k:raw[k] for k in raw.files}
            fields['h']=np.asarray(-p['stepSeconds']);np.savez_compressed(damaged_dir/'negative-h.npz',**fields)
            item.update(file='negative-h.npz',sha256=api['sha'](damaged_dir/'negative-h.npz'))
        elif fault=='scope':damaged['physiologicalRL']='admitted'
        elif fault=='dropped-event':
            path=damaged_dir/damaged['metricsFile'];metrics=json.loads(gzip.decompress(path.read_bytes()))
            row=next(row for row in metrics if row['event'] is not None);row['event']=None
            row['recordHash']=checker['record_hash']({k:v for k,v in row.items() if k!='recordHash'})
            path.write_bytes(gzip.compress(json.dumps(metrics).encode(),mtime=0));damaged['metricsSha256']=api['sha'](path)
        else:
            for pass_result in damaged['passes']:pass_result['branches']['short-hold']['energyNmm']+=1.
        (damaged_dir/'samples.json').write_text(json.dumps(index));damaged['samplesSha256']=api['sha'](damaged_dir/'samples.json')
        (damaged_dir/'report.json').write_text(json.dumps(damaged))
        with pytest.raises(ValueError,match=pattern):checker['check'](damaged_dir,protocol)


@pytest.mark.parametrize('fault',['clock','count','target-at-event','source-resurrection'])
def test_program_parser_refuses_changed_clock_and_undeclared_event_policy(tmp_path,fault):
    _,_,_,_,data,masks=full_fixture(tmp_path);d=copy.deepcopy(data['short-hold'])
    if fault=='clock':d['ticks'][1]['nativeRemainderBeforeSeconds']=.001
    elif fault=='count':d['ticks'][1]['nativeFixedSteps']+=1
    elif fault=='target-at-event':d['ticks'][3]['contacts'][0]['targetDisplacementMm'][0]+=.001
    else:d['ticks'][5]['topologyId']=0
    with pytest.raises(ValueError,match='PROGRAM_'):prepare_program('short-hold',d,masks)
