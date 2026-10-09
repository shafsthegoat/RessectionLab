"""Independent analytical controls; no real measurements or native execution."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import mechanics_hbe_branch_calibration as core
from scripts import mechanics_hbe_branch_calibration_readout as readout


def half_model():
    return SimpleNamespace(
        Xh=np.array([[0., 0., 0.], [.004, .004, .005]]),
        bottom=np.array([0]), midplane=np.array([1]), radius=.004,
        half=SimpleNamespace(deformation=lambda x, mu: {'energy_J': 3 * mu, 'stress_Pa': 2 * mu}))


def test_reference_half_frame_exact_legacy_arithmetic():
    m=half_model(); raw=np.array([[0.,0.,0.],[1e-12,0.,0.]])
    original=core.inherited.HalfHeightReconstruction.half_frame(m,m.Xh,raw,full_displacement_m=0.)
    actual=readout.half_frame(m,m.Xh,raw,full_displacement_m=0.,mu=1000.)
    assert actual == original, 'Exact reference reproduction must preserve original arithmetic order'


def test_actual_mu_reaches_half_physics_and_unbalanced_force_floor():
    m=half_model(); raw=np.array([[0.,0.,0.],[1e-12,0.,0.]])
    a=readout.half_frame(m,m.Xh,raw,full_displacement_m=0.,mu=1000.)
    b=readout.half_frame(m,m.Xh,raw,full_displacement_m=0.,mu=123.456)
    assert b['energy_J']==3*123.456 and b['stress_Pa']==2*123.456
    assert b['ratios']['free_dof_reaction']==pytest.approx(a['ratios']['free_dof_reaction']*1000/123.456,rel=1e-14)
    assert core.inherited.MU_PA==1000.


@pytest.mark.parametrize('mu',[0.,-1.,True,float('nan'),float('inf')])
def test_invalid_mu_stops_before_physics(mu):
    m=half_model();m.half.deformation=lambda *args:pytest.fail('physics reached invalid modulus')
    with pytest.raises(ValueError):readout.half_frame(m,m.Xh,np.zeros_like(m.Xh),full_displacement_m=0.,mu=mu)


def test_full_frame_ratios_use_actual_mu_floor():
    X=np.array([[0.,0.,0.],[.004,.004,.005]]);raw=np.zeros_like(X)
    state={'current_nodes_m':X,'net_force_N':np.array([1e-12,0.,0.]),'net_moment_Nm':np.array([1e-14,0.,0.]),'prescribed_error_m':0.,'free_node_reaction_max_N':1e-12}
    a=readout.frame_ratios(SimpleNamespace(radius_m=.004),state,raw,mu=1000.)
    b=readout.frame_ratios(SimpleNamespace(radius_m=.004),state,raw,mu=250.)
    for key in ('force_balance','moment_balance','free_node_reaction'):assert b[key]==4*a[key]


def frame(index,mu=1000.,fault=False):
    rest=np.array([[0.,0.,0.],[.001,.002,.003]])
    current=rest+index*np.array([[0.,0.,1e-6],[1e-7,0.,0.]])
    raw=np.array([[1.,-2.,3.],[-1.,2.,-3.]])*mu/1000
    if fault:raw[1,2]+=.01
    return {'index':index,'current':current,'raw':raw,'torque':index*1e-7*mu/1000},rest


@pytest.mark.parametrize('fault',[False,True])
def test_scale_reduction_all_states_and_component_corruption(fault):
    reduction=readout.ScaleReduction(123.456)
    for i in range(121):
        a,rest=frame(i);b,_=frame(i,123.456,fault=fault and i==73)
        reduction.update(a,b,rest)
    result=reduction.finish()
    assert result['motion']['actual']==0
    assert result['torque']['actual']<1e-10
    assert (result['reaction']['actual']>1) is fault
    assert result['motion']['limit']==1e-6*.004+1e-5*(120e-6)


def test_scale_reduction_missing_or_reordered_state_rejected():
    reduction=readout.ScaleReduction(2000.)
    a,rest=frame(0);b,_=frame(0,2000.);reduction.update(a,b,rest)
    with pytest.raises(ValueError,match='Complete121'):reduction.finish()
    a,rest=frame(2);b,_=frame(2,2000.)
    with pytest.raises(ValueError,match='out-of-order'):reduction.update(a,b,rest)


def binding(root,name,value):
    path=root/name;path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(value if isinstance(value,str) else json.dumps(value,sort_keys=True))
    return {'path':name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}


def test_exact_registry_caps_survive_verify_all_and_snapshot(tmp_path,monkeypatch):
    calls=[]
    monkeypatch.setattr(core.access,'verify_binding',lambda root,b,**kw:calls.append((b['path'],kw['maximum_bytes'])))
    reg=core.Registry(tmp_path)
    primitives={k:{'path':'reference/'+k,'sha256':'a'*64} for k in core.inherited.PRIMITIVE_KEYS}
    reg.primitives('compression',primitives)
    reg.verify_all(time.monotonic()+60)
    assert dict(calls)['reference/nodes']==768*1024**2
    assert dict(calls)['reference/elements']==512*1024**2
    assert {r['path']:r['maximum_bytes'] for r in reg.snapshot()}==dict(calls)
    with pytest.raises(ValueError,match='changed'):reg.read({'path':'reference/nodes','sha256':'b'*64})
    with pytest.raises(ValueError,match='Unregistered'):reg.read({'path':'invented','sha256':'a'*64})
    with pytest.raises(ValueError,match='Conflicting'):reg.primitives('tension',primitives)


def analytical_study(tmp_path):
    """Minimal direct access-method fixture, not a release/preflight substitute."""
    schemas={b:{'analytic':True} for b in core.AXIAL+core.TORSION}
    roles={'calibration':{'members':[{'path':b+'.csv'} for b in core.AXIAL]},'held_out_validation':{'members':[{'path':b+'.csv'} for b in core.TORSION]}}
    rb=binding(tmp_path,'roles.json',roles);pb=binding(tmp_path,'protocol.json',{'roles':rb})
    release={'execution':{'access_ledger_path':'attempt/access.jsonl','csv_schemas':schemas}}
    relb=binding(tmp_path,'release.json',release);reg=core.Registry(tmp_path)
    for b in (rb,pb,relb):reg.bound(b)
    refs={}
    for b in core.AXIAL+core.TORSION:
        sign=-1 if b in ('compression','torsion_neg') else 1
        refs[b]={'load_coordinate_m':[0.,sign*.001],'load_coordinate':[0.,sign*.1],
                 'applied_force_N':[0.,sign*2.], 'applied_torque_Nm':[0.,sign*.02]}
    context={'root':tmp_path,'study':{'protocol':pb},'release_binding':relb,'release':release,'registry':reg,'references':refs,'directory':tmp_path/'attempt'}
    obj=core.BranchReleasedStudy(context,time.monotonic()+60)
    obj._calibration={b:core.evaluation.curve(b,refs[b]['load_coordinate_m'],[2*y for y in refs[b]['applied_force_N']]) for b in core.AXIAL}
    obj._audit('calibration_attempt',[b+'.csv' for b in core.AXIAL])
    obj._audit('calibration_completed',[b+'.csv' for b in core.AXIAL])
    fit=core.evaluation.fit_scale(obj._calibration,{b:core.curve_from_row(b,refs[b]) for b in core.AXIAL})
    fb=binding(tmp_path,'attempt/fit.json',fit);pred=core.predictions(refs,fit['scale']);prb=binding(tmp_path,'attempt/predictions.json',pred)
    return obj,fb,prb,schemas


def test_failed_confirmation_cannot_freeze_or_reveal(tmp_path,monkeypatch):
    obj,fb,pb,schemas=analytical_study(tmp_path)
    monkeypatch.setattr(readout,'paired_evidence',lambda *args:{'passed':False,'analytical_fixture':True})
    with pytest.raises(ValueError,match='confirmation failed'):
        obj.freeze_predictions(fit_binding=fb,prediction_binding=pb,fitted_runs={},output_path='attempt/freeze.json')
    assert not (tmp_path/'attempt/freeze.json').exists()
    assert all(e['phase']!='freeze_saved' for e in obj._events())
    monkeypatch.setattr(obj,'_read_selected',lambda *a,**k:pytest.fail('withheld member opened'))
    with pytest.raises(ValueError,match='durable freeze'):
        obj.evaluate_held_out(freeze_binding={'path':'absent','sha256':'a'*64},schemas={b:schemas[b] for b in core.TORSION})


def test_freeze_binding_change_rejected_before_heldout_and_repeat_fit_sealed(tmp_path,monkeypatch):
    obj,fb,pb,schemas=analytical_study(tmp_path)
    monkeypatch.setattr(readout,'paired_evidence',lambda *args:{'passed':True,'analytical_fixture':True})
    freeze=obj.freeze_predictions(fit_binding=fb,prediction_binding=pb,fitted_runs={},output_path='attempt/freeze.json')
    with pytest.raises(ValueError,match='sealed'):obj.read_calibration({b:schemas[b] for b in core.AXIAL})
    with pytest.raises(ValueError,match='One prior'):
        obj.freeze_predictions(fit_binding=fb,prediction_binding=pb,fitted_runs={},output_path='attempt/second-freeze.json')
    (tmp_path/fb['path']).write_text('{}')
    monkeypatch.setattr(obj,'_read_selected',lambda *a,**k:pytest.fail('withheld member opened'))
    with pytest.raises(ValueError,match='hash|Hash'):
        obj.evaluate_held_out(freeze_binding=freeze,schemas={b:schemas[b] for b in core.TORSION})


def test_changed_release_rejected_before_calibration_member_read(tmp_path,monkeypatch):
    obj,fb,pb,schemas=analytical_study(tmp_path)
    (tmp_path/obj.release_binding['path']).write_text('{}')
    monkeypatch.setattr(obj,'_read_selected',lambda *a,**k:pytest.fail('axial member opened'))
    with pytest.raises(ValueError,match='hash|Hash'):obj.read_calibration({b:schemas[b] for b in core.AXIAL})


def test_actual_mu_real_half_energy_and_reflected_full_energy():
    import importlib.util
    p=Path(__file__).resolve().parents[2]/'tests/test_mechanics_hbe_halfheight_readout.py'
    spec=importlib.util.spec_from_file_location('analytical_half_fixture',p)
    fixtures=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixtures)
    model=fixtures.model();current=model.Xh.copy();current[:,2]*=1.15
    raw=np.zeros_like(current);raw[:4,2]=1.;raw[4:,2]=-1.
    a=readout.half_frame(model,current,raw,full_displacement_m=.15*model.height,mu=1000.)
    b=readout.half_frame(model,current,raw,full_displacement_m=.15*model.height,mu=123.456)
    assert b['energy_J']==pytest.approx(a['energy_J']*.123456,rel=1e-13)
    lifted=model.lift(current,raw,np.ones((1,8)),.15*model.height)
    full=model.full.read_frame(lifted['current_nodes_m'],lifted['raw_reactions_N'],mu_Pa=123.456,branch='tension',fraction=1.)
    assert full['energy_J']==pytest.approx(2*b['energy_J'],rel=1e-13)


def test_registry_metadata_to_exact_kind_can_only_be_upgraded_once(tmp_path):
    reg=core.Registry(tmp_path)
    primitives={k:binding(tmp_path,'generated/'+k,'analytical') for k in core.inherited.PRIMITIVE_KEYS}
    reg.bound(primitives['deck'],json_value=False)
    reg.primitives('compression',primitives)
    assert reg.entries['generated/deck']['maximum_bytes']==32*1024**2
    reg.bound(primitives['deck'],json_value=False)
    assert reg.entries['generated/deck']['maximum_bytes']==32*1024**2
    with pytest.raises(ValueError,match='Conflicting'):reg.primitives('tension',primitives)


def test_declared_caps_survive_freeze_and_reveal_and_mutation_stops_open(tmp_path,monkeypatch):
    obj,fb,pb,schemas=analytical_study(tmp_path)
    primitives={k:binding(tmp_path,'large-labelled-fixture/'+k,'tiny analytical stand-in') for k in core.inherited.PRIMITIVE_KEYS}
    reg=obj.context['registry'];reg.primitives('compression',primitives)
    calls=[];original=core.access.verify_binding
    def observed(root,b,**kw):
        if b['path'] in (primitives['nodes']['path'],primitives['elements']['path']):calls.append((b['path'],kw['maximum_bytes']))
        return original(root,b,**kw)
    monkeypatch.setattr(core.access,'verify_binding',observed)
    monkeypatch.setattr(readout,'paired_evidence',lambda *args:{'passed':True,'analytical_fixture':True})
    freeze=obj.freeze_predictions(fit_binding=fb,prediction_binding=pb,fitted_runs={},output_path='attempt/freeze.json')
    assert (primitives['nodes']['path'],768*1024**2) in calls
    assert (primitives['elements']['path'],512*1024**2) in calls
    calls.clear();opened=[]
    def forbidden(*args,**kwargs):opened.append(True);raise RuntimeError('analytical member-open sentinel')
    monkeypatch.setattr(obj,'_read_selected',forbidden)
    with pytest.raises(RuntimeError,match='sentinel'):obj.evaluate_held_out(freeze_binding=freeze,schemas={b:schemas[b] for b in core.TORSION})
    assert len(opened)==1 and (primitives['nodes']['path'],768*1024**2) in calls
    (tmp_path/primitives['nodes']['path']).write_text('changed')
    with pytest.raises(ValueError,match='hash|Hash'):obj.evaluate_held_out(freeze_binding=freeze,schemas={b:schemas[b] for b in core.TORSION})
    assert len(opened)==1


def test_worker_rejects_failed_qualification_before_access(tmp_path,monkeypatch):
    from scripts import mechanics_hbe_branch_calibration_experiment as runner
    (tmp_path/'study/experiment').mkdir(parents=True)  # Launcher normally creates this.
    monkeypatch.setattr(core,'declaration',lambda root:{'output_root':'study'})
    monkeypatch.setattr(core,'preflight',lambda *a,**k:(_ for _ in ()).throw(ValueError('qualification failed')))
    monkeypatch.setattr(core.BranchReleasedStudy,'read_calibration',lambda *a,**k:pytest.fail('calibration member path reached'))
    monkeypatch.setattr(core.old,'solve',lambda *a,**k:pytest.fail('native entry reached'))
    with pytest.raises(ValueError,match='qualification failed'):runner.worker(tmp_path,{},time.monotonic()+60)
    state=json.loads((tmp_path/'study/experiment/state.json').read_text())
    assert state['native_calls']==0 and state['calibration_access_attempted'] is False


@pytest.mark.parametrize('failure',['compression','tension','confirmation','none'])
def test_worker_stops_on_native_or_confirmation_failure_and_orders_reveal(tmp_path,monkeypatch,failure):
    from scripts import mechanics_hbe_branch_calibration_experiment as runner
    obj,fb,pb,schemas=analytical_study(tmp_path)
    context=obj.context;reg=context['registry'];events=[];native=[]
    context['directory']=tmp_path/'study/experiment'
    context['directory'].mkdir(parents=True)  # Launcher normally creates this.
    archive=binding(tmp_path,'archive.tar','analytic metadata only')
    context['release']['source_archive']=archive
    context['context']={'runtime':{'executable':'never-invoked-analytic-solver'}}
    mesh=binding(tmp_path,'mesh.json',{'analytic':True})
    caps={'new_total_output_bytes':2*1024**3,'active_output_bytes_by_branch':{'compression':1536*1024**2,'tension':512*1024**2},'native_seconds_by_branch':{'compression':2100,'tension':420}}
    study=context['study'];study.update(output_root='study',budgets=caps,csv_schemas=schemas,runtime_identity={'analytic':True},backend_profile={'analytic':True},axial_references={b:{'counts':{'N':36 if b=='compression' else 24},'native_primitives':{'mesh':mesh},'reconstruction':{'analytic':True}} for b in core.AXIAL})
    monkeypatch.setattr(core,'declaration',lambda root:study)
    monkeypatch.setattr(core,'preflight',lambda *a,**k:context)
    class Watch:
        active=None
        def __init__(self,*a,**k):pass
        def __enter__(self):return self
        def __exit__(self,*a):pass
    monkeypatch.setattr(runner.old,'OutputWatch',Watch)
    class Released:
        def __init__(self,*a):pass
        def read_calibration(self,schemas):events.append('calibration');return obj._calibration
        def freeze_predictions(self,**kw):
            events.append('confirm-and-freeze')
            assert set(kw['fitted_runs'])==set(core.AXIAL)
            if failure=='confirmation':raise ValueError('analytical confirmation failure')
            return {'path':'fixture-freeze','sha256':'a'*64}
        def evaluate_held_out(self,**kw):events.append('heldout');return {'analytic':True}
    monkeypatch.setattr(core,'BranchReleasedStudy',Released)
    monkeypatch.setattr(core,'fitted_contents',lambda *a:('analytic source','analytic transformed deck',{'analytic':True}))
    def child(command,prepdir,deadline,seconds):
        assert seconds==60
        fit=json.loads((context['directory']/'fit.json').read_text());cases={}
        for branch in core.AXIAL:
            base='study/experiment/runs/'+branch+'/'
            cases[branch]={'source':binding(tmp_path,base+'skyline.feb','analytic source'),'deck':binding(tmp_path,base+'specimen.feb','analytic transformed deck'),'loading':binding(tmp_path,base+'loading.json',{'analytic':True})}
        core.receipts.durable_json(tmp_path,prepdir/'prepared.json',{'schema':'hbe-branch-fitted-preparation-v1','fit':core.old.binding(tmp_path,context['directory']/'fit.json'),'mu_Pa':fit['mu_Pa'],'cases':cases,'declaration_sha256':core.DECLARATION_SHA256,'native_calls':0,'mesher_calls':0})
    monkeypatch.setattr(core.common,'pure_child',child)
    def solve(command,target,seconds):
        native.append(target.name)
        assert 0<seconds<=caps['native_seconds_by_branch'][target.name]
        if failure==target.name:raise TimeoutError('analytical native failure')
        for name in ['nodes.log','elements.log','solver.log']:(target/name).write_text('analytic dispatch fixture, no primitive data')
        return {'analytic':True}
    monkeypatch.setattr(core.old,'solve',solve)
    monkeypatch.setattr(core.receipts,'require_completed_native',lambda *a:None)
    if failure=='none':runner.worker(tmp_path,context['release_binding'],time.monotonic()+60)
    else:
        with pytest.raises((ValueError,TimeoutError),match='analytical'):runner.worker(tmp_path,context['release_binding'],time.monotonic()+60)
    state=json.loads((context['directory']/'state.json').read_text())
    assert state['native_calls']==len(native)==(1 if failure=='compression' else 2)
    assert events==(['calibration','confirm-and-freeze','heldout'] if failure=='none' else ['calibration','confirm-and-freeze'] if failure=='confirmation' else ['calibration'])
    assert state['held_out_access_attempted'] is (failure=='none')
    assert state['status']==(runner.DONE if failure=='none' else 'failed_or_incomplete')


def test_stream_threads_actual_mu_into_all_121_frames_residual_and_both_work_scales(tmp_path,monkeypatch):
    mu=123.456;m=half_model();m.X=m.Xh.copy();calls={'half':[],'full':[],'work':[],'residual':[]}
    def deformation(current,value):calls['half'].append(value);return {'energy_J':0.,'minimum_sampled_J':1.}
    m.half=SimpleNamespace(element_count=1,deformation=deformation)
    def full_frame(current,raw,*,mu_Pa,branch,fraction):
        calls['full'].append(mu_Pa)
        return {'net_force_N':np.zeros(3),'net_moment_Nm':np.zeros(3),'prescribed_error_m':0.,'free_node_reaction_max_N':0.,'applied_force_N':0.,'applied_torque_Nm':0.,'energy_J':0.,'minimum_sampled_J':1.}
    m.full=SimpleNamespace(radius_m=.004,probe_map=lambda x:None,read_frame=full_frame,interpolate_displacement=lambda x,p:np.zeros((75,3)))
    m.lift=lambda current,raw,e,d:{'current_nodes_m':current,'raw_reactions_N':raw,'maximum_shared_displacement_mismatch_m':0.}
    def records(stack,root,b,count,fields,name):
        for i in range(121):
            if fields==9:values=np.column_stack((m.Xh,np.zeros((2,6))))
            else:values=np.ones((1,8))
            yield {'step':i,'values':values}
    monkeypatch.setattr(readout,'records',records)
    def solver(stream,*,expected_times,residual_floor_N2):
        calls['residual'].append(residual_floor_N2)
        return {'states':[{'actual_N2':0.,'limit_N2':residual_floor_N2}]}
    monkeypatch.setattr(readout.inherited,'check_solver_records',solver)
    def work(*args,**kwargs):calls['work'].append(kwargs);return {'maximum_error_J':0.,'limit_J':1.}
    monkeypatch.setattr(readout.inherited,'energy_work_check',work)
    b={k:binding(tmp_path,k+'.txt','analytical dispatch marker') for k in ('nodes','elements','solver')}
    generator=readout.stream_frames(tmp_path,b,m,branch='compression',mu=mu,deadline=time.monotonic()+60)
    for i in range(121):assert next(generator)['index']==i
    with pytest.raises(StopIteration) as end:next(generator)
    assert calls['half']==calls['full']==[mu]*121
    assert calls['residual']==[(1e-10*mu*.004**2)**2]
    assert calls['work']==[{'mu_Pa':mu,'radius_m':.004,'height_m':readout.H/2},{'mu_Pa':mu,'radius_m':.004,'height_m':readout.H}]
    assert end.value.value['mu_Pa']==mu and end.value.value['frame_count']==121
    assert end.value.value['passed'] is False  # Stationary mock deliberately violates prescribed displacement.
