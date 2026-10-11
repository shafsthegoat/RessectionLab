"""Generated-only regression for compact reports and saved-readout recovery."""
from copy import deepcopy
import io
import json
from pathlib import Path
import socket
import subprocess
import pytest
from scripts import mechanics_hbe_ogden_comparison as core
from scripts import mechanics_hbe_ogden_recovery as recovery
from scripts.mechanics_hbe_evaluation import curve,fit_scale
from scripts.mechanics_hbe_v5_comparison import _signed_pair


@pytest.fixture(autouse=True)
def no_payload_or_process(monkeypatch):
    def refuse(*a,**k):raise AssertionError('Generated fixtures only')
    monkeypatch.setattr(subprocess,'Popen',refuse)
    monkeypatch.setattr(socket.socket,'connect',refuse)
    real=io.open
    def opened(p,*a,**k):
        if isinstance(p,(str,Path)) and ('/outputs/mechanics/' in str(p) or '/data/' in str(p)):refuse()
        return real(p,*a,**k)
    monkeypatch.setattr(io,'open',opened)


def fixture():
    observed={};reference={};old={};numerical={};pairs={}
    for b,sign in [('compression',-1),('tension',1)]:
        x=[sign*(i+1)*.00002 for i in range(30)];f=[20*u for u in x]
        y=[.7*v for v in f]
        if b=='tension':y[:4]=[-.0001,-.0002,-.0003,-.0001]
        observed[b]=curve(b,x,y,source_sha256='a'*64);reference[b]=curve(b,x,f)
        views={}
        for n in (16,24):
            u=[sign*.00074*(i/60) for i in range(61)]
            probes=[[[.00000123456789123*(i/60)*(1+j/75)*(1 if n==16 else 1.000123),0.,v/2]
                     for j in range(75)] for i,v in enumerate(u)]
            r={'run_id':f'{b}:N{n}:S60:reference','steps':60,'load_coordinate_full_m':u,
               'applied_force_N':[20*v for v in u],'probe_displacements_m':probes,'numerical_passed':True}
            views[n]=core.view(r,b,n)
            if n==24:old[b]=r;numerical[core.run_name((b,24,60))]=deepcopy(r)
        pairs[b]=_signed_pair(views[16],views[24])
    fit=fit_scale(observed,reference)
    declaration={'fit':{'path':'generated-fit.json','sha256':'f'*64},'evidence':{
        'comparison':{'path':'accepted-generated-comparison.json','sha256':'c'*64},
        'manifest':{'path':'accepted-generated-manifest.json','sha256':'d'*64}}}
    return fit,declaration,numerical,old,pairs


def test_realistic_61_state_75_probe_report_fits_without_changing_decisions(tmp_path,monkeypatch):
    fit,d,numerical,old,pairs=fixture()
    monkeypatch.setattr(core,'read_binding',lambda root,binding,**k:deepcopy(fit))
    calls=[];result=core.comparison_output(tmp_path,d,numerical,old,pairs,on_arithmetic=calls.append)
    assert calls==['profile','profile','baseline_reproduction']
    bloated={**result,'baseline_pair_screen':pairs}
    assert len(core.canonical(bloated))>512*1024
    assert len(core.canonical(result))<128*1024
    core.exclusive_json(tmp_path/'comparison.json',result,maximum=512*1024)
    for b in core.AXIAL:
        compact=result['baseline_pair_screen'][b]
        for key,value in compact.items():
            if key not in ('complete_signed_arrays','reference_identity_manifest'):assert value==pairs[b][key]
        assert compact['complete_signed_arrays']['binding']==d['evidence']['comparison']
        assert compact['complete_signed_arrays']['json_path']==['all_adjacent_signed_differences',b,'N16-N24']
        assert len(pairs[b]['signed_probe_component_difference_m'])==61
        assert len(pairs[b]['signed_probe_component_difference_m'][0])==75
        assert result['candidates']['2']['fit']['modes'][b]['observed']==fit['modes'][b]['observed']
    assert result['original_alpha2_baseline']['reproduced']
    assert result['physical_validation_pass'] is None and not result['held_out_values_used']


def test_finish_retains_full_new_screen_and_uses_same_profiles(tmp_path,monkeypatch):
    fit,d,numerical,old,pairs=fixture();checks={'passed':True,'pairs':pairs}
    report_bindings={key:{'path':key,'sha256':'1'*64} for key in numerical}
    def read(root,b,**k):return deepcopy(fit if b==d['fit'] else numerical[b['path']])
    monkeypatch.setattr(core,'read_binding',read);monkeypatch.setattr(core,'screen',lambda *a,**k:deepcopy(checks))
    monkeypatch.setattr(core,'baseline',lambda *a:(old,pairs))
    written={}
    def write(path,value,**kwargs):
        raw=core.canonical(value);assert len(raw)<=kwargs['maximum']
        written[path.name]=json.loads(raw)
    monkeypatch.setattr(core,'exclusive_json',write)
    result=core.finish(tmp_path,d,report_bindings,tmp_path)
    assert written['numerical-screen.json']==checks
    assert written['comparison.json']==result


def test_false_recovery_release_does_not_create_output(tmp_path):
    with pytest.raises(ValueError,match='separate root release'):
        recovery.execute(tmp_path,core,None,{}, {'execution_released':False},{})
    assert list(tmp_path.iterdir())==[]


def test_saved_admission_never_recomputes_numerical_metrics(tmp_path,monkeypatch):
    fit,d,numerical,old,pairs=fixture()
    def forbidden(*a,**k):raise AssertionError('No numerical replay')
    monkeypatch.setattr(core,'screen',forbidden);monkeypatch.setattr(core,'baseline',forbidden)
    reports={core.run_name(row):{'path':recovery.PRIOR+'/'+core.run_name(row)+'/readout.json','sha256':'1'*64} for row in core.RUNS}
    cleanup={'contained':True,'direct_child_reaped':True,'exit_code':0,'fallback_used':False,'remaining_members':[],'errors':[]}
    stage={'status':'completed_within_caps','exit_code':0,'kill_reason':None}
    stages={}
    for name in [core.run_name(r) for r in core.RUNS]+['read-'+str(i) for i in range(6)]:
        stages[name]={'receipt':{'readout_stage' if name.startswith('read-') else 'native_stage':stage},'cleanup':cleanup}
    prior={'status':'failed_shape_comparison','native_calls':6,'positive_scale_profiles':0,
           'release':recovery.UPSTREAM['release'],'stages':stages}
    refs={b:{'path':'reference-'+b,'sha256':'2'*64} for b in core.AXIAL}
    values={recovery.UPSTREAM['terminal']['path']:prior,
        recovery.UPSTREAM['console']['path']:b"exclusive_json(out/'comparison.json'\nValueError: output cap",
        recovery.UPSTREAM['release']['path']:{'declaration':{'path':core.DECLARATION,'sha256':core.DECLARATION_SHA256}},
        recovery.UPSTREAM['screen']['path']:{'schema':'hbe-ogden-two-grid-screen-v1','passed':True,
            'pairs':{b+'/'+kind:{'pairwise_limit_passed':True} for b in core.AXIAL for kind in ('mesh','time')},'physical_validation_pass':None},
        recovery.UPSTREAM['request']['path']:{'mode':'profile','reports':reports},
        d['evidence']['comparison']['path']:{'numerical_qualification_passed':True,'physical_validation_pass':None,
            'comparison_manifest':d['evidence']['manifest'],'all_adjacent_signed_differences':{b:{'N16-N24':pairs[b]} for b in core.AXIAL}},
        d['evidence']['manifest']['path']:{'rows':{f'{b}:N24:S60:reference':{'readout':refs[b]} for b in core.AXIAL}}}
    values.update({binding['path']:{'numerical_passed':True} for binding in reports.values()})
    values.update({refs[b]['path']:old[b] for b in core.AXIAL})
    monkeypatch.setattr(core,'read_binding',lambda root,binding,**k:deepcopy(values[binding['path']]))
    actual=recovery.admitted_inputs(tmp_path,core,d)
    assert actual[3]==reports and actual[4]==refs
    values[recovery.UPSTREAM['screen']['path']]['passed']=False
    with pytest.raises(ValueError,match='retained passing numerical screen'):recovery.admitted_inputs(tmp_path,core,d)
