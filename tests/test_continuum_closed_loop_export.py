"""Generated two-pass worker/export contract before a bounded source run."""
import gzip
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np
import pytest

from resectionlab.surgical_closed_loop import hold_contacts
from resectionlab.surgical_continuum_program import contact_matrix,field_hash,record_hash
from test_surgical_closed_loop import fixture,commands,LIMITS


def test_generated_stateful_worker_exports_only_committed_replayed_steps_and_samples(tmp_path,monkeypatch):
    root=Path(__file__).resolve().parents[1];scripts=str(root/'scripts');sys.path.insert(0,scripts)
    spec=importlib.util.spec_from_file_location('qualify_continuum_closed_loop',root/'scripts/qualify_continuum_closed_loop.py')
    api=importlib.util.module_from_spec(spec)
    try:spec.loader.exec_module(api)
    finally:sys.path.remove(scripts)
    scene,grid,domain,initial,a,t,h=fixture(two_tip_candidates=True)
    plan=commands(initial);plan.insert(6,json.loads(json.dumps(plan[0])))
    g=domain.geometry;free=np.flatnonzero(g['freeDofs']);core=np.unique(contact_matrix(g,hold_contacts(scene,initial['holds']))[:,free].indices)
    core_sha=api.hashlib.sha256(core.astype(np.int64).tobytes()).hexdigest()
    def entry(path):return {'path':str(path.relative_to(root)),'sha256':api.sha(path)}
    profile=tmp_path/'profile.json';profile.write_text(json.dumps({'status':'completed','sourceInputsUnchanged':True,
        'selectedFastestPolicy':'contact-core-fresh','variants':[{'policy':'contact-core-fresh','rootCoreFreeDofsSha256':core_sha}]}))
    accepted=tmp_path/'profile-acceptance.json';accepted.write_text(json.dumps({'sourceCostProfileComplete':True}))
    checker=tmp_path/'generated-checker-placeholder.py';checker.write_text('# Test worker only; source checker is separately frozen and audited.\n')
    p={'sourceRole':'GENERATED_CONTROL','scope':'Generated beam worker/export control; no patient/source qualification',
        'costProfileReport':entry(profile),'costProfileAcceptance':entry(accepted),'algebraicCorePolicy':'contact-core-fresh',
        'rootCoreFreeDofsSha256':core_sha,'independentQuadrature':entry(root/'scripts/continuum_quadrature.py'),
        'independentChecker':entry(checker),'workerWallSeconds':20,'passes':2,'expectedTicks':10,'expectedFixedSteps':120,
        'minimumNewRemovals':1,'maximumNewRemovals':2,'expectedSamples':20,'limits':LIMITS,
        'caps':{'wallSeconds':30,'rssBytes':3221225472,'outputBytes':33554432}}
    protocol=tmp_path/'protocol.json';protocol.write_text(json.dumps(p));monkeypatch.setattr(api,'PROTOCOL',protocol)
    metadata={'branch':'generated','coupledStep':0,'previousArrayHash':field_hash(initial['displacement'],initial['memories']),
        'commands':plan,'initialInterpretation':'Generated preload plus explicit accumulated CUSA work'}
    monkeypatch.setattr(api,'seed',lambda *args:(scene,grid,domain,initial,a,t,h,np.array(domain.cells),plan,metadata))
    launch=tmp_path/'launch.json';launch.write_text(json.dumps({'commit':'generated-control','sourceHashes':{},'protocolSha256':api.sha(protocol)}))
    output=tmp_path/'result';api.worker(output,launch);report=json.loads((output/'report.json').read_text())
    assert report['status']=='completed' and report['allStepReplayExact'] and report['allStepCount']==120
    assert report['sourceRole']=='GENERATED_CONTROL' and not report['largerDomainPublicActionsAdmitted']
    rows=json.loads(gzip.decompress((output/report['metricsFile']).read_bytes()));replay=json.loads((output/report['replayFile']).read_text())
    assert len(rows)==len(replay['records'])==120 and len(report['ticks'])==len(replay['ticks'])==10
    assert report['ticks']==replay['ticks']
    assert all(record_hash(row)==other['recordSha256'] for row,other in zip(rows,replay['records']))
    samples=json.loads((output/report['samplesFile']).read_text())
    assert len(samples)==20 and {(s['tick'],s['withinTick']) for s in samples}=={(i,j) for i in range(10) for j in (0,11)}
    assert report['final']['memoryNonzero'] and len(report['final']['newRemovedSourceIds'])==2
    assert report['final']['removedEnergyNmm']>0 and report['final']['contactJumpNmm']<0
    assert all(pass_['preparationReleased'] for pass_ in report['passes'])
    assert all(row['cusaEvents']==[] for row in report['ticks'][7:])
    for sample in samples:
        assert api.sha(output/sample['fieldsFile'])==sample['fieldsSha256']
        with np.load(output/sample['fieldsFile'],allow_pickle=False) as f:
            assert f['memories'].shape[0]==1 and float(f['h'])==h


def test_main_refuses_combined_budget_overrun_even_when_both_processes_exit_zero(tmp_path,monkeypatch):
    root=Path(__file__).resolve().parents[1];scripts=str(root/'scripts');sys.path.insert(0,scripts)
    spec=importlib.util.spec_from_file_location('closed_loop_budget_control',root/'scripts/qualify_continuum_closed_loop.py')
    api=importlib.util.module_from_spec(spec)
    try:spec.loader.exec_module(api)
    finally:sys.path.remove(scripts)
    protocol=tmp_path/'protocol-budget.json';protocol.write_text(json.dumps({'workerWallSeconds':70,'checkerWallSeconds':30,
        'caps':{'wallSeconds':100,'rssBytes':3221225472,'outputBytes':33554432}}));monkeypatch.setattr(api,'PROTOCOL',protocol)
    monkeypatch.setattr(api,'SOURCES',[root/'scripts/qualify_continuum_closed_loop.py'])
    monkeypatch.setattr(api.subprocess,'check_output',lambda command,**kwargs:'' if command[1]=='status' else 'generated-control')
    output=tmp_path/'budget-result';monkeypatch.setattr(sys,'argv',['qualify_continuum_closed_loop.py',str(output)])
    stamps=iter([0.,10.,100.1]);monkeypatch.setattr(api,'time',SimpleNamespace(perf_counter=lambda:next(stamps)));calls=[]
    def supervise(command,directory,**kwargs):
        calls.append(kwargs['seconds']);directory.mkdir(parents=True,exist_ok=False)
        if len(calls)==1:
            (directory/'result').mkdir();(directory/'result/report.json').write_text(json.dumps({'status':'completed','allStepReplayExact':True,'peakRssBytes':0}))
        else:(output/'result/checker-result.json').write_text(json.dumps({'status':'passed'}))
        return {'status':'completed'}
    monkeypatch.setattr(api,'supervise',supervise)
    with pytest.raises(RuntimeError,match='failed/incomplete'):api.main()
    result=json.loads((output/'acceptance.json').read_text())
    assert calls==[70,30] and not result['conditionalClosedLoopPilotComplete'] and result['combinedWallSeconds']==100.1
