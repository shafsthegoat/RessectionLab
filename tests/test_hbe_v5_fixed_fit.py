"""Tiny invented primitives only. No native subprocess or specimen payload."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import time
import numpy as np
import pytest
from scripts import mechanics_hbe_v5_fitted_confirmation as core
from scripts import mechanics_hbe_v5_fitted_confirmation_experiment as launcher
from scripts import mechanics_hbe_v5_frame as frame
from scripts import mechanics_hbe_v5_stream as stream
from test_mechanics_hbe_v5_frame import generated
from test_mechanics_hbe_v5_stream import generated_run


def test_reference_entrypoint_and_output_unchanged():
    spec,mesh,node,element,reconstruction=generated('tension','lower_half_reconstructed',30)
    candidate=frame.evaluate_generated_frame(spec,mesh,node,element,reconstruction=reconstruction)
    source=Path(__file__).resolve().parents[2] # ignored stage, baseline passed by runner
    import os
    baseline=Path(os.environ.get('HBE_SOURCE_ROOT',Path(__file__).resolve().parents[1]))/'scripts/mechanics_hbe_v5_frame.py'
    import sys
    modspec=importlib.util.spec_from_file_location('oldframe_control',baseline)
    old=importlib.util.module_from_spec(modspec);sys.modules[modspec.name]=old
    exec(compile(baseline.read_bytes(),str(baseline),'exec'),old.__dict__)
    assert candidate==old.evaluate_generated_frame(spec,mesh,node,element,reconstruction=reconstruction)
    fitted=dict(spec,mu_Pa=core.MU,fixed_fit_sha256=core.FIT_SHA)
    with pytest.raises(ValueError,match='reference material'):
        frame.evaluate_prepared_frame(fitted,frame.prepare_generated_frame(fitted,mesh,reconstruction=reconstruction),node,element)


def test_fitted_frame_scales_mu_and_force_denominators():
    spec,mesh,node,element,reconstruction=generated('compression','lower_half_reconstructed',30)
    reference=frame.evaluate_generated_frame(spec,mesh,node,element,reconstruction=reconstruction)
    fitted=dict(spec,mu_Pa=core.MU,fixed_fit_sha256=core.FIT_SHA)
    node=deepcopy(node);node['values'][:,6:9]*=core.SCALE
    result=frame.evaluate_fitted_frame(fitted,frame.prepare_generated_frame(fitted,mesh,reconstruction=reconstruction),node,element)
    assert result['energy_J']==pytest.approx(reference['energy_J']*core.SCALE,rel=2e-14)
    assert result['applied_force_N']==pytest.approx(reference['applied_force_N']*core.SCALE)
    for name in ('native_force_balance','native_free_dof_reaction','full_moment_balance'):
        assert result['criteria_ratios'][name]==pytest.approx(reference['criteria_ratios'][name],abs=1e-9)
    fitted['fixed_fit_sha256']='0'*64
    with pytest.raises(ValueError,match='fixed fit'):frame.evaluate_fitted_frame(fitted,None,node,element)


def test_three_scalar_deck_delta_and_forbidden_change():
    from test_mechanics_hbe_branch_calibration_v5 import fixture_source
    from scripts.mechanics_hbe_experiment import check_fitted_deck
    source=('<febio_spec><Material><material><c1>2000.0</c1><k>'+repr(149*1000./3)+
            '</k></material></Material><Control><solver><min_residual>'+repr((1e-10*1000.*.004**2)**2)+
            '</min_residual><fixture>immutable</fixture></solver></Control></febio_spec>')
    actual=core.fitted_deck(source)
    check_fitted_deck(source,actual,core.MU)
    with pytest.raises(ValueError):check_fitted_deck(source,actual.replace(b'immutable',b'changed'),core.MU)


def scaled_lines(lines):
    result=[]
    for line in lines:
        if line[:1].isdigit():
            cells=line.strip().split(',');cells[7:10]=[repr(float(v)*core.SCALE) for v in cells[7:10]]
            result.append(','.join(cells)+'\n')
        else:result.append(line)
    return result


def test_generated_complete_paired_stream():
    spec,mesh,nodes,elements,solver,reconstruction=generated_run('compression',half=True,steps=120)
    fitted=core.fitted_contract(spec,'a'*64)
    refs=stream.evaluated_frames(spec,mesh,nodes,elements,solver,reconstruction=reconstruction)
    new=stream.evaluated_frames(fitted,mesh,scaled_lines(nodes),elements,solver,reconstruction=reconstruction,fitted=True)
    reports,metrics,passed=core.paired_streams(refs,new,deadline=time.monotonic()+10)
    assert passed and reports[0]['frame_count']==reports[1]['frame_count']==121
    assert reports[0]==stream.evaluate_stream(spec,mesh,nodes,elements,solver,reconstruction=reconstruction)
    assert metrics['native']['motion']['actual']==0
    assert metrics['reconstructed_full']['reaction']['actual']<1e-8


@pytest.mark.parametrize('change',['endpoint','steps','domain','mu'])
def test_fitted_contract_refuses_drift(change):
    spec={'mu_Pa':1000.,'steps':120,'native_domain':'lower_half_reconstructed','branch':'tension',
          'times':tuple(i/120 for i in range(121)),'full_coordinates_m':[0.,.0007360099999999],
          'run_id':'tension:N24:S120:reference'}
    if change=='endpoint':spec['full_coordinates_m'][-1]=.15*.00489159
    else:spec[{'steps':'steps','domain':'native_domain','mu':'mu_Pa'}[change]]={'steps':60,'domain':'full_native','mu':999.}[change]
    with pytest.raises(ValueError):core.fitted_contract(spec,'a'*64)


def torsion_rows():
    H=.00489159;rows={}
    for branch,sign in [('torsion_neg',-1),('torsion_pos',1)]:
        rows[branch]={'branch':branch,'mesh_N':12,'steps':120,'mu_Pa':1000.,'frame_count':121,
            'load_coordinate':[sign*.15*H/.004*i/120 for i in range(121)],
            'applied_torque_Nm':[sign*i*1e-9 for i in range(121)],
            'criteria':{'numerical':{'actual':0.,'limit':1.,'units':'ratio'}}}
    return rows


def numerical():return {'passed':True,'mu_Pa':core.MU,'fixed_fit_sha256':core.FIT_SHA,
                       'branches':{b:{'passed':True} for b in core.AXIAL}}


def test_fixed_prediction_no_refit(monkeypatch):
    from scripts import mechanics_hbe_evaluation as evaluation
    monkeypatch.setattr(evaluation,'fit_scale',lambda *a,**k:pytest.fail('refit forbidden'))
    rows=torsion_rows();result=core.checked_predictions(rows,numerical())
    assert result['fitted']['torsion_pos']['response'][-1]==rows['torsion_pos']['applied_torque_Nm'][-1]*core.SCALE
    assert result['fitted']['torsion_neg']['coordinate']==rows['torsion_neg']['load_coordinate']


@pytest.mark.parametrize('change',['failed_branch','wrong_fit','wrong_mu','one_branch','overall_failure'])
def test_failed_confirmation_blocks_prediction(change):
    n=numerical()
    if change=='failed_branch':n['branches']['tension']['passed']=False
    elif change=='wrong_fit':n['fixed_fit_sha256']='0'*64
    elif change=='wrong_mu':n['mu_Pa']=1000.
    elif change=='one_branch':del n['branches']['tension']
    else:n['passed']=False
    with pytest.raises(ValueError):core.checked_predictions(torsion_rows(),n)


@pytest.mark.parametrize('field,value',[('fallback_used',True),('exit_code',1),('contained',False),
                                      ('direct_child_reaped',False),('remaining_members',[1]),('errors',['observer'])])
def test_owned_stage_requires_clean_reaped_exit(field,value):
    clean={'contained':True,'direct_child_reaped':True,'remaining_members':[],'errors':[],
           'fallback_used':False,'exit_code':0}
    stage={'status':'completed_within_caps','exit_code':0}
    launcher.require_stage(core,stage,clean)
    clean[field]=value
    with pytest.raises(ValueError):launcher.require_stage(core,stage,clean)


def test_false_release_cannot_reserve(tmp_path):
    with pytest.raises(ValueError,match='root release'):
        launcher.execute(tmp_path,core,{}, {'execution_released':False},{})
    assert list(tmp_path.iterdir())==[]


@pytest.mark.parametrize('defect',['motion','reaction','missing','extra','order'])
def test_paired_reduction_rejects_or_fails_changed_primitives(defect):
    from types import SimpleNamespace
    rest=np.zeros((2,3));mesh=SimpleNamespace(fingerprint='same',rest_nodes_m=rest)
    prepared=SimpleNamespace(native_mesh=mesh,full_mesh=mesh,mapping_sha256='map')
    def rows(changed=False):
        count=120 if changed and defect=='missing' else 122 if changed and defect=='extra' else 121
        for i in range(count):
            x=rest.copy();raw=rest.copy()
            if changed and defect=='motion':x[0,0]=1e-3
            if changed and defect=='reaction':raw[0,0]=1.
            index=i+1 if changed and defect=='order' else i
            yield {'index':index,'current':x,'raw':raw,'native_current':x,'native_raw':raw,'torque':0.,'native_torque':0.},prepared
        return {'numerical_passed':True}
    if defect in ('motion','reaction'):
        _,_,passed=core.paired_streams(rows(),rows(True),deadline=time.monotonic()+1)
        assert passed is False
    else:
        with pytest.raises(ValueError):core.paired_streams(rows(),rows(True),deadline=time.monotonic()+1)


def test_uninspected_torsion_domain_cannot_expand():
    rows=torsion_rows();rows['torsion_pos']['load_coordinate'][-1]+=.0001
    with pytest.raises(ValueError,match='loading grid'):core.checked_predictions(rows,numerical())


def test_output_inventory_refuses_unlisted_or_mutated_file(tmp_path,monkeypatch):
    monkeypatch.setattr(core,'CAPS',{**core.CAPS,'new_output_bytes':1024})
    (tmp_path/'value').write_bytes(b'original')
    seal=launcher.snapshot(tmp_path,core,tmp_path,{'value':100})
    (tmp_path/'value').write_bytes(b'changed')
    assert launcher.snapshot(tmp_path,core,tmp_path,{'value':100})!=seal
    (tmp_path/'extra').write_bytes(b'other')
    with pytest.raises(ValueError,match='inventory'):launcher.snapshot(tmp_path,core,tmp_path,{'value':100})


def test_nonzero_native_couple_is_distinct_from_full_torque():
    spec,mesh,node,element,reconstruction=generated('tension','lower_half_reconstructed',30)
    top=np.asarray(mesh['boundaries']['top']['node_ids'])-1
    values=node['values'].copy();values[:,6:9]=0.
    values[top,6]=-values[top,1];values[top,7]=values[top,0]
    node={**node,'values':values};captured=[]
    prepared=frame.prepare_generated_frame(spec,mesh,reconstruction=reconstruction)
    frame._evaluate_prepared_frame(spec,prepared,node,element,primitive_sink=captured.append)
    primitive=captured[0]
    expected=float(-np.cross(values[top,:3],values[top,6:9])[:,2].sum())
    assert expected!=0 and primitive['native_torque']==expected
    assert primitive['torque']!=primitive['native_torque']


def test_native_scale_torque_uses_native_couple_not_full_value():
    from types import SimpleNamespace
    rest=np.zeros((2,3));mesh=SimpleNamespace(fingerprint='same',rest_nodes_m=rest)
    prepared=SimpleNamespace(native_mesh=mesh,full_mesh=mesh,mapping_sha256='map')
    def rows(changed):
        for i in range(121):
            yield {'index':i,'current':rest,'raw':rest,'native_current':rest,'native_raw':rest,
                   'torque':0.,'native_torque':2*core.SCALE if changed else 1.},prepared
        return {'numerical_passed':True}
    _,metrics,passed=core.paired_streams(rows(False),rows(True),deadline=time.monotonic()+1)
    assert passed is False and metrics['native']['torque']['actual']>1
    assert metrics['reconstructed_full']['torque']['actual']==0


@pytest.mark.parametrize('late_mutation',[False,True])
def test_full_owned_launcher_wiring_without_process(tmp_path,monkeypatch,late_mutation):
    from types import SimpleNamespace
    from scripts import mechanics_hbe_backend as backend
    from scripts import mechanics_hbe_v5_remaining_one_shot as stages
    from scripts import mechanics_hbe_v5_n8_one_shot as io
    from launchers import hbe_v5_ordinal9_continuation_v1 as owned
    import shutil,sys
    monkeypatch.setattr(core,'OUTPUT','build/fake-fixed-fit/attempt-01')
    out=tmp_path/core.OUTPUT;rb={'path':'fake-release.json','sha256':'a'*64}
    release={'execution_released':True,'source_commit':'b'*40};decl={'fixed_fit':{'fit':{'sha256':core.FIT_SHA}},'runtime_identity':{'fake':'runtime'}}
    monkeypatch.setattr(core,'check_release',lambda *a,**k:(release,decl))
    for name in ('verify_sources','audit_loaded','fixed_fit','recheck_fixed_inputs'):
        monkeypatch.setattr(core,name,lambda *a,**k:None)
    monkeypatch.setattr(launcher,'check_cache',lambda *a,**k:None)
    monkeypatch.setattr(shutil,'disk_usage',lambda *a:SimpleNamespace(free=10*1024**3))
    monkeypatch.setattr(backend,'verify_runtime_binding',lambda *a:{'runtime':{'executable':'invented-native-no-launch'}})
    monkeypatch.setattr(stages,'_check_backend',lambda *a:None)
    clean={'contained':True,'direct_child_reaped':True,'remaining_members':[],'errors':[],
           'fallback_used':False,'exit_code':0}
    monkeypatch.setattr(owned,'OwnedStage',lambda:SimpleNamespace(cleanup=lambda *a:dict(clean)))
    calls=[];write=core.exclusive_json
    def exclusive(path,value,**kwargs):
        digest=write(path,value,**kwargs)
        if late_mutation and path.name=='terminal.json':
            (out/'predictions.json').write_text('{"late":"mutation"}\n')
        return digest
    monkeypatch.setattr(core,'exclusive_json',exclusive)
    def stage(kind,command,directory,receipt,**kwargs):
        assert kwargs['popen'] is not None
        name=directory.name;calls.append(name)
        result={'status':'completed_within_caps','exit_code':0,'command':command,'kill_reason':None,
                'elapsed_seconds':.001,'wall_cap_seconds':kwargs['wall_cap'],
                'sampled_process_group_rss_cap_bytes':kwargs['rss_cap'],'active_output_cap_bytes':kwargs['output_cap']}
        receipt[kind+'_stage']=result;write(directory/'receipt.json',receipt)
        (directory/('console.txt' if kind=='native' else 'readout-console.txt')).write_text('generated fixture\n')
        if name=='prepare-stage':
            branches={}
            for branch in core.AXIAL:
                target=out/branch;target.mkdir();deck=target/'specimen.feb';deck.write_text('<invented/>')
                branches[branch]={'deck':{'path':str(deck.relative_to(tmp_path)),'sha256':core.sha(deck.read_bytes())},
                                  'contract':{'run_id':branch+':fitted'}}
            write(out/'prepared.json',{'branches':branches,'runtime':{'executable':'invented-native-no-launch'}})
        elif kind=='native':
            for leaf in ('nodes.log','elements.log','solver.log'):(directory/leaf).write_text('tiny invented bytes\n')
        elif name=='readout-stage':
            executions=json.loads((out/'executions.json').read_text())
            for leaf in ('numerical.json','predictions.json','continuation-ledger.json'):write(out/leaf,{'generated':True})
            write(out/'freeze.json',{'mu_Pa':core.MU,'fixed_fit':decl['fixed_fit'],
                'fitted_executions':executions,'held_out_access_released':False})
        return result
    monkeypatch.setattr(stages,'supervise_stage',stage)
    result=launcher.execute(tmp_path,core,rb,release,decl)
    assert calls==['prepare-stage','compression','tension','readout-stage']
    assert result['native_calls']==2 and result['fit_calls']==0 and result['measured_member_reads']==0
    if late_mutation:
        assert result['status']=='failed_fixed_fit_continuation' and (out/'finalization-failure.json').exists()
    else:
        assert result['status']==core.DONE and json.loads((out/'publication.json').read_text())['accepted'] is True
        assert not (out/'finalization-failure.json').exists()


def test_failed_numeric_is_preserved_before_prediction_or_torsion(monkeypatch,tmp_path):
    bad=numerical();bad['branches']['compression']['passed']=False
    monkeypatch.setattr(core,'fixed_fit',lambda *a:pytest.fail('failed branch cannot reach saved-fit/torque admission'))
    core.exclusive_json(tmp_path/'numerical.json',bad)
    with pytest.raises(ValueError):core.finish(tmp_path,{}, {},tmp_path,bad,{})
    assert json.loads((tmp_path/'numerical.json').read_text())==bad
    assert not (tmp_path/'freeze.json').exists()
