"""Tiny generated source/deck/stream/profile/owned-parent controls; no native/data IO.

Portable canonical test. To exercise the ignored candidate, prepend its staged
scripts directory to scripts.__path__ before collection. No fixture is a source
specimen or a claim that the six proposed native runs have passed.
"""
from copy import deepcopy
import io
import json
import math
from pathlib import Path
import shutil
import socket
import subprocess
import sys
from types import SimpleNamespace
import xml.etree.ElementTree as ET
import numpy as np
import pytest
from scripts import mechanics_hbe_ogden_comparison as core
from scripts import mechanics_hbe_ogden_comparison_experiment as runner
from scripts import mechanics_hbe_v5_frame as frame
from scripts import mechanics_hbe_v5_stream as stream
from scripts import mechanics_hbe_v5_fitted_confirmation_experiment as helper
from scripts.mechanics_hbe_evaluation import curve,fit_scale
from test_mechanics_hbe_v5_frame import contract as old_contract,generated
from test_mechanics_hbe_halfheight_readout import geometry

ROOT=Path(__file__).resolve()
while not (ROOT/'pyproject.toml').is_file():ROOT=ROOT.parent


@pytest.fixture(autouse=True)
def no_native_or_source_payload(monkeypatch):
    def refused(*a,**k):raise AssertionError('No subprocess/network/source specimen in generated controls')
    monkeypatch.setattr(subprocess,'Popen',refused)
    monkeypatch.setattr(socket.socket,'connect',refused)
    real=io.open
    def opened(path,*a,**k):
        if isinstance(path,(str,Path)):
            p=Path(path).resolve()
            if any(p.is_relative_to(ROOT/section) for section in ('data','outputs/mechanics')):refused()
        return real(path,*a,**k)
    monkeypatch.setattr(io,'open',opened)


def deck(branch='compression',domain='lower_half_reconstructed'):
    c,_,raw,_=old_contract(branch,domain)
    tree=ET.fromstring(raw);material=tree.find('Material/material')
    ET.SubElement(material,'m1').text='2'
    ET.SubElement(material,'k').text=repr(149000/3)
    return c,ET.tostring(tree)


@pytest.mark.parametrize('branch',core.AXIAL)
@pytest.mark.parametrize('domain',('full_native','lower_half_reconstructed'))
@pytest.mark.parametrize('steps',(60,120))
def test_shape_delta_exact_geometry_endpoints_and_controls(branch,domain,steps):
    c,old=deck(branch,domain);raw,new=core.shape_deck(old,c,steps)
    before,after=map(ET.fromstring,(old,raw))
    assert after.findtext('Material/material/m1')=='-8'
    assert after.findtext('Material/material/c1')==before.findtext('Material/material/c1')
    assert after.findtext('Material/material/k')==before.findtext('Material/material/k')
    assert ET.tostring(after.find('Boundary'))==ET.tostring(before.find('Boundary'))
    assert new['full_coordinates_m'][-1].hex()==c['full_coordinates_m'][-1].hex()
    assert new['full_coordinates_m'][::steps//60]==list(c['full_coordinates_m'])
    assert new['times']==[i/steps for i in range(steps+1)]
    assert new['mu_Pa']==1000. and new['ogden_alpha']==-8.
    assert not new['native_execution_released'] and not new['source_binding_checked']


@pytest.mark.parametrize('bad',('bulk','time','plot'))
def test_deck_rejects_unapproved_material_or_output(bad):
    c,raw=deck();t=ET.fromstring(raw)
    if bad=='bulk':t.find('Material/material/k').text='3'
    elif bad=='time':t.find('Control/time_steps').text='59'
    else:ET.SubElement(t.find('Output'),'plotfile',file='hidden.xplt')
    with pytest.raises(ValueError):core.shape_deck(ET.tostring(t),c,60)


@pytest.mark.parametrize('domain',('full_native','lower_half_reconstructed'))
def test_shape_frame_energy_uses_new_oracle_and_keeps_old_lane_closed(domain):
    c,m,n,e,reconstruction=generated('tension',domain,60)
    changed=dict(c,ogden_alpha=-8.,shape_declaration_sha256=core.DECLARATION_SHA256)
    prepared=frame.prepare_generated_frame(changed,m,reconstruction=reconstruction)
    result=frame.evaluate_ogden_frame(changed,prepared,n,e)
    full,_,_=geometry();H=full['geometry']['height_m'];lam=1+c['full_coordinates_m'][-1]/H
    alpha=-8.;mu=1000.;volume=(.008**2)*H
    density=2*mu/alpha**2*(2*lam**(-alpha/3)+lam**(2*alpha/3)-3)
    density+=(149*mu/3)/4*(lam**2-1-2*math.log(lam))
    assert result['energy_J']==pytest.approx(volume*density,rel=2e-12)
    with pytest.raises(ValueError):frame.evaluate_prepared_frame(changed,prepared,n,e)
    with pytest.raises(ValueError):frame.evaluate_fitted_frame(changed,prepared,n,e)
    for key,value in [('ogden_alpha',-7.),('shape_declaration_sha256','1'*64)]:
        with pytest.raises(ValueError):frame.evaluate_ogden_frame(dict(changed,**{key:value}),prepared,n,e)


def generated_stream(branch,steps):
    c,old=deck(branch);_,c=core.shape_deck(old,c,steps)
    full,half,mapping=geometry();X=np.asarray(half['rest_nodes_m']);H=full['geometry']['height_m']
    area=.008**2;alpha=-8.;mu=1000.;K=149*mu/3
    top=np.asarray(half['boundaries']['top']['node_ids'])-1;bottom=np.asarray(half['boundaries']['bottom']['node_ids'])-1
    nodes=[];elements=[];solver=[]
    for index,(time,u) in enumerate(zip(c['times'],c['full_coordinates_m'])):
        lam=1+u/H
        derivative=(2*mu/alpha**2)*(2*(-alpha/3)*lam**(-alpha/3-1)+(2*alpha/3)*lam**(2*alpha/3-1))
        force=area*(derivative+K/2*(lam-1/lam))
        current=X.copy();current[:,2]*=lam
        reactions=np.zeros_like(X);reactions[top,2]=-force/len(top);reactions[bottom,2]=force/len(bottom)
        values=np.column_stack((current,current-X,reactions));logged=np.array([[0.,0.,0.,0.,0.,0.,lam,999.]])
        for target,name,rows in [(nodes,'mechanics_nodes_si',values),(elements,'mechanics_elements_si',logged)]:
            target.extend([f'*Step = {index}\n',f'*Time = {time:.9g}\n',f'*Data = {name}\n'])
            target.extend(f'{j},'+','.join(repr(float(x)) for x in row)+'\n' for j,row in enumerate(rows,1))
        if index:solver.extend([f'Nonlinear solution status: time= {time:.6g}\n',' residual 1e-6 1e-16 1e-14\n'])
    solver.append('N O R M A L T E R M I N A T I O N\n')
    return c,half,nodes,elements,solver,(full,mapping)


@pytest.mark.parametrize('branch',core.AXIAL)
@pytest.mark.parametrize('steps',(60,120))
def test_complete_shape_stream_with_independent_analytic_force(branch,steps):
    c,m,n,e,s,r=generated_stream(branch,steps)
    it=stream.evaluated_frames(c,m,n,e,s,reconstruction=r,ogden_shape=True)
    while True:
        try:next(it)
        except StopIteration as end:result=end.value;break
    assert result['numerical_passed'] and result['frame_count']==steps+1
    assert result['full_energy_work']['passed'] and result['native_energy_work']['passed']
    assert not result['logged_sed_used_for_energy_gate'] and result['minimum_sampled_J']>0
    with pytest.raises(ValueError):next(stream.evaluated_frames(c,m,n,e,s,reconstruction=r))


def report(branch,n,steps):
    endpoint=-.00073726 if branch=='compression' else .0007360099999999
    x=[endpoint*i/steps for i in range(steps+1)]
    # Same float association as v5 schedule.
    x=[endpoint*(i/steps) for i in range(steps+1)]
    return {'run_id':f'{branch}:N{n}:S{steps}:alpha-minus8','steps':steps,
        'load_coordinate_full_m':x,'applied_force_N':[20*u for u in x],
        'probe_displacements_m':[[[0.,0.,u/2]]*75 for u in x],'numerical_passed':True}


def test_screen_retains_negative_pair_and_no_v5_promotion():
    reports={core.run_name(row):report(*row) for row in core.RUNS}
    good=core.screen(reports,temporal=True)
    assert good['passed'] and not good['old_v5_numerical_qualification_passed']
    assert not good['continuum_error_bound'] and good['physical_validation_pass'] is None
    reports[core.run_name(('compression',24,60))]['applied_force_N'][-1]+=.1
    assert not core.screen(reports,temporal=True)['passed']
    with pytest.raises(ValueError):core.screen({k:v for i,(k,v) in enumerate(reports.items()) if i},temporal=True)


def synthetic_fit():
    observed={};references={}
    for b,sign in [('compression',-1),('tension',1)]:
        x=[sign*(i+1)*.00002 for i in range(30)];g=[20*u for u in x]
        y=[.7*f for f in g]
        if b=='tension':y[:4]=[-.0001,-.0002,-.0003,-.0001]
        observed[b]=curve(b,x,y,source_sha256='a'*64);references[b]=curve(b,x,g)
    return fit_scale(observed,references)


def test_profile_preserves_all_signed_values_and_original_weights():
    fit=synthetic_fit();refs={b:report(b,24,60) for b in core.AXIAL}
    result=core.profile(fit,refs)
    assert result['fit']['mu_Pa']>0 and result['physical_validation_pass'] is None
    for b in core.AXIAL:
        assert result['fit']['modes'][b]['observed']==fit['modes'][b]['observed']
        assert result['fit']['modes'][b]['coordinate']==fit['modes'][b]['coordinate']
        assert result['fit']['modes'][b]['weights']==fit['modes'][b]['weights']
    assert len(result['metrics']['branches']['tension']['residuals'])==30
    assert result['fit']['modes']['tension']['observed'][:4]==[-.0001,-.0002,-.0003,-.0001]


@pytest.mark.parametrize('bad',('rows','weight','extrapolation','negative_scale'))
def test_profile_refusals(bad):
    fit=synthetic_fit();refs={b:report(b,24,60) for b in core.AXIAL}
    if bad=='rows':fit['modes']['compression']['observed'].pop()
    elif bad=='weight':fit['modes']['tension']['weights'][0]*=2
    elif bad=='extrapolation':fit['modes']['tension']['coordinate'][-1]=1.
    else:
        for r in refs.values():r['applied_force_N']=[-x for x in r['applied_force_N']]
    with pytest.raises(ValueError):core.profile(fit,refs)


def test_false_release_never_enters_lifecycle(tmp_path):
    with pytest.raises(ValueError,match='root execution'):
        runner.execute(tmp_path,core,helper,{}, {'execution_released':False},{})
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('initial',(False,True))
@pytest.mark.parametrize('delta',(-1,0,1))
def test_fifty_gib_floor_and_initial_output_reservation(tmp_path,monkeypatch,initial,delta):
    assert runner.FREE_DISK_FLOOR_BYTES==50*1024**3
    threshold=50*1024**3+(core.CAPS['new_output_bytes'] if initial else 0)
    monkeypatch.setattr(shutil,'disk_usage',lambda _:SimpleNamespace(free=threshold+delta))
    if delta<0:
        with pytest.raises(ValueError,match='50 GiB'):
            runner.check_free_disk(core,tmp_path,reserve_outputs=initial)
    else:runner.check_free_disk(core,tmp_path,reserve_outputs=initial)


@pytest.mark.parametrize('failure',('none','fallback','unreaped','timeout','late_output','late_source','disk_drop'))
def test_full_parent_wiring_with_generated_fake_stages(tmp_path,monkeypatch,failure):
    from scripts import mechanics_hbe_v5_remaining_one_shot as stages
    from scripts import mechanics_hbe_v5_n8_one_shot as io_helper
    from scripts import mechanics_hbe_backend as backend
    from launchers import hbe_v5_ordinal9_continuation_v1 as owner_module
    out=tmp_path/core.OUTPUT;release={'execution_released':True,'source_commit':'a'*40}
    rb={'path':'generated-release.json','sha256':'b'*64};decl={'runtime_identity':{'path':'identity','sha256':'c'*64}}
    checks={'calls':0};cleanup={'contained':True,'direct_child_reaped':True,'remaining_members':[],
                             'errors':[],'fallback_used':False,'exit_code':0}
    if failure=='fallback':cleanup['fallback_used']=True
    if failure=='unreaped':cleanup['direct_child_reaped']=False
    class Owner:
        def cleanup(self,*a,**k):return deepcopy(cleanup)
    monkeypatch.setattr(owner_module,'OwnedStage',Owner)
    monkeypatch.setattr(core,'check_release',lambda *a,**k:(release,decl))
    def sources(*a,**k):
        checks['calls']+=1
        if failure=='late_source' and (out/'comparison.json').exists():raise ValueError('late source mutation')
    monkeypatch.setattr(core,'verify_sources',sources);monkeypatch.setattr(core,'audit_loaded',lambda *a:None)
    monkeypatch.setattr(sys,'pycache_prefix',str(tmp_path/'build/unused-parent-cache'))
    monkeypatch.setattr(backend,'verify_runtime_binding',lambda *a:{'runtime':{'executable':'/generated/febio'}})
    monkeypatch.setattr(stages,'_check_backend',lambda *a:None)
    monkeypatch.setattr(io_helper,'private_environment',lambda:{})
    monkeypatch.setattr(shutil,'disk_usage',lambda _:SimpleNamespace(free=52*1024**3))
    def write(path,value):path.parent.mkdir(parents=True,exist_ok=True);core.exclusive_json(path,value,maximum=8*1024**2)
    def fake_stage(kind,command,directory,receipt,**kwargs):
        r={'status':'completed_within_caps','exit_code':0,'command':command}
        if failure=='timeout':r.update(status='failed_or_incomplete',exit_code=-9)
        receipt[kind+'_stage']=r
        write(directory/'receipt.json',receipt)
        (directory/('console.txt' if kind=='native' else 'readout-console.txt')).write_bytes(b'')
        if failure=='disk_drop':
            monkeypatch.setattr(shutil,'disk_usage',lambda _:SimpleNamespace(free=50*1024**3-1))
            # The owned observer must refuse before calling any real RSS probe.
            kwargs['rss_observer'](12345,timeout_seconds=.1)
            pytest.fail('monitored 50 GiB floor was not enforced')
        if kind=='native':
            for key in ('nodes','elements','solver'):(directory/(key+'.log')).write_text('generated only\n')
        else:
            mode=command[command.index('--worker')+1]
            if mode=='prepare':
                runs={}
                for row in core.RUNS:
                    name=core.run_name(row);p=out/name/'specimen.feb';p.parent.mkdir();p.write_bytes(b'<generated/>')
                    runs[name]={'contract':{'run_id':name},'deck':{'path':str(p.relative_to(tmp_path)),'sha256':core.sha(p.read_bytes())}}
                write(out/'prepared.json',{'runs':runs,'runtime':{'executable':'/generated/febio'}})
            elif mode.startswith('read-'):
                row=core.RUNS[int(mode[5:])];write(out/core.run_name(row)/'readout.json',{'generated_fake_parent_readout':True})
            elif mode=='screen':write(out/'spatial-screen.json',{'passed':True})
            else:
                write(out/'numerical-screen.json',{'passed':True})
                write(out/'comparison.json',{'alpha_candidates':list(core.ALPHAS),'new_shape_screen_passed':True,'physical_validation_pass':None})
        return r
    monkeypatch.setattr(stages,'supervise_stage',fake_stage)
    actual_snapshot=helper.snapshot;seen={'count':0}
    def snapshot(*args):
        seen['count']+=1
        if failure=='late_output' and seen['count']==2:(out/'comparison.json').write_text('{}')
        return actual_snapshot(*args)
    monkeypatch.setattr(helper,'snapshot',snapshot)
    result=runner.execute(tmp_path,core,helper,rb,release,decl)
    if failure=='none':
        assert result['status']==core.DONE and result['native_calls']==6 and result['positive_scale_profiles']==2
        assert json.loads((out/'publication.json').read_text())['accepted']
        assert not (out/'finalization-failure.json').exists()
    else:
        assert result['status']=='failed_shape_comparison'
        assert (out/'finalization-failure.json').is_file()
        assert not json.loads((out/'publication.json').read_text())['accepted']
        if failure=='disk_drop':assert result['native_calls']==0 and '50 GiB' in result['error']
    assert checks['calls']>0
