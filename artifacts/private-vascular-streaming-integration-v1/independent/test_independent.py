"""Independent tiny-grid scalar parity and boundary preservation controls."""
from pathlib import Path
import ast
import importlib.util
import os
import sys

import numpy as np
import pytest
from resectionlab.evaluation import segment_box_distance_sq

ROOT=Path(__file__).resolve().parents[2]
STAGE=ROOT/'build/private-vascular-streaming-integration-preparation-v1/stage/src/resectionlab'
spec=importlib.util.spec_from_file_location('independent_streaming_integration_kernel',STAGE/'vascular_contact_streaming.py')
s=importlib.util.module_from_spec(spec);sys.modules[spec.name]=s;spec.loader.exec_module(s)

def guard(event,args):
    if event in ('subprocess.Popen','os.system','os.exec','socket.connect'):
        raise AssertionError('No process/model/network work in generated integration review')
    if event=='open' and isinstance(args[0],(str,bytes)):
        assert not os.fsdecode(args[0]).lower().endswith(('.nii','.nii.gz','.mat','.tar','.pt','.ckpt','.bin','.npz','.npy','.pkl','.safetensors','.dcm','.h5'))
sys.addaudithook(guard)


def scalar_cells(capsule,shape,affine):
    spacing=np.linalg.norm(affine[:3,:3],axis=0);rotation=affine[:3,:3]/spacing
    a,b=[rotation.T@(np.asarray(point)-affine[:3,3]) for point in (capsule.start_ras_mm,capsule.end_ras_mm)]
    cells={cell for cell in np.ndindex(shape) if segment_box_distance_sq(a,b,np.asarray(cell)*spacing-spacing/2,
        np.asarray(cell)*spacing+spacing/2)<=capsule.radius_mm**2+1e-10}
    outside=bool(np.any(np.minimum(a,b)-capsule.radius_mm<-.5*spacing)
              or np.any(np.maximum(a,b)+capsule.radius_mm>(np.asarray(shape)-.5)*spacing))
    return cells,outside


def expected_counts(cells,outside,mask,coverage,affine):
    positive=sum(bool(mask[cell]) for cell in cells);unknown=sum(not bool(coverage[cell]) for cell in cells)
    volume=abs(float(np.linalg.det(affine[:3,:3])))
    return {'touched_reference_cells':len(cells),'positive_reference_cells':positive,'unknown_reference_cells':unknown,
        'outside_reference_fov':outside,'annotated_positive_encounter':True if positive else None if unknown or outside else False,
        'annotation_coverage_complete_for_sweep':not unknown and not outside,
        'positive_cell_volume_upper_bound_mm3':positive*volume,'unknown_in_grid_cell_volume_mm3':unknown*volume,
        'biological_vessel_free':None,'clinical_injury_probability':None}


@pytest.mark.parametrize('spacing,angle,reflection,translation',[
    ((.01,.023,.07),0.,False,(0.,0.,0.)),
    ((1.3,.9,2.1),.37,True,(20000.,-15000.,99999.)),
    ((100.,51.,70.),-.4,False,(-12000.,4.,3.))])
@pytest.mark.parametrize('edge',[2,4])
def test_full_grid_parts_sweeps_and_duplicates_match_scalar_oracle(spacing,angle,reflection,translation,edge):
    shape=(5,4,3);spacing=np.array(spacing)
    rotation=np.array([[np.cos(angle),-np.sin(angle),0.],[np.sin(angle),np.cos(angle),0.],[0.,0.,-1. if reflection else 1.]])
    affine=np.eye(4);affine[:3,:3]=rotation@np.diag(spacing);affine[:3,3]=translation
    def cap(action,part,a,b,r):
        return s.Capsule(action,part,rotation@(np.array(a)*spacing)+translation,
                         rotation@(np.array(b)*spacing)+translation,float(r*min(spacing)))
    capsules=(cap('0','shaft',[-.7,1.,1.],[4.2,2.4,1.],.32),cap('0','tip',[1.,1.,1.],[4.2,2.4,1.],.7),
              cap('1','shaft',[-.7,1.,1.],[4.2,2.4,1.],.32),cap('1','tip',[1.,1.,1.],[4.2,2.4,1.],.7),
              cap('2','tip',[2.,-.1,.5],[2.,3.,2.],.4))
    indices=np.indices(shape);coverage=indices[0]<3
    mask=((indices[0]+indices[1]+indices[2])%3==0)&coverage
    calls=[]
    def sampler(selected):
        assert selected.dtype==np.int64 and 0<len(selected)<=edge**3
        calls.extend(map(tuple,selected));key=tuple(selected.T)
        return mask[key],coverage[key]
    result=s.evaluate_contacts(s.Grid(shape,affine),capsules,sample_reference=sampler,budget=s.Budget(tile_edge=edge))
    cells=[scalar_cells(c,shape,affine) for c in capsules]
    for part in ('shaft','tip','whole_tool'):
        selection=[i for i,c in enumerate(capsules) if part=='whole_tool' or c.part==part]
        union=set().union(*(cells[i][0] for i in selection));outside=any(cells[i][1] for i in selection)
        assert result[part]==expected_counts(union,outside,mask,coverage,affine)
    for action in ('0','1','2'):
        selection=[i for i,c in enumerate(capsules) if c.action_id==action]
        union=set().union(*(cells[i][0] for i in selection));outside=any(cells[i][1] for i in selection)
        assert result['per_action'][action]==expected_counts(union,outside,mask,coverage,affine)
    assert result['per_action']['0']==result['per_action']['1']
    assert len(calls)==len(set(calls))==result['whole_tool']['touched_reference_cells']
    assert result['work']['reference_sampled_cells']==len(calls) and result['work']['maximum_tile_cells']<=edge**3


@pytest.mark.parametrize('bad',['list','integer','two_dimensional','wrong_length','positive_unknown'])
def test_sampler_contract_refuses_inconsistent_private_labels(bad):
    grid=s.Grid((2,2,2),np.eye(4));caps=(s.Capsule('0','tip',(0.,0.,0.),(1.,1.,1.),.5),)
    def sampler(indices):
        n=len(indices);positive=np.zeros(n,bool);known=np.ones(n,bool)
        if bad=='list':return [positive,known]
        if bad=='integer':return positive.astype(int),known
        if bad=='two_dimensional':return positive[:,None],known
        if bad=='wrong_length':return positive[:-1],known
        return np.ones(n,bool),np.zeros(n,bool)
    with pytest.raises(ValueError,match='bounded_boolean_sample|positive_outside_annotation_domain'):
        s.evaluate_contacts(grid,caps,sample_reference=sampler)


@pytest.mark.parametrize('limit',['max_tile_capsule_pairs','max_cell_capsule_pairs','max_sampled_cells'])
def test_refused_work_has_no_reference_callback_or_partial_outcomes(limit):
    calls=[];budget=s.Budget(**{limit:0})
    with pytest.raises(s.BudgetExceeded) as error:
        s.evaluate_contacts(s.Grid((2,2,2),np.eye(4)),(s.Capsule('0','tip',(0.,0.,0.),(1.,1.,1.),.5),),
            sample_reference=lambda indices:calls.append(indices),budget=budget)
    assert calls==[] and error.value.outcomes is None and error.value.completed is False
    assert error.value.rejected_charge['limit']==0 and error.value.rejected_charge['projected']>0
    assert error.value.budget[limit]==0


def test_callback_deadline_detected_before_aggregation(monkeypatch):
    now=[0.];monkeypatch.setattr(s.time,'monotonic',lambda:now[0])
    def sampler(indices):
        now[0]=2.
        return np.ones(len(indices),bool),np.ones(len(indices),bool)
    with pytest.raises(s.BudgetExceeded,match='cooperative_wall_budget') as error:
        s.evaluate_contacts(s.Grid((2,2,2),np.eye(4)),(s.Capsule('0','tip',(0.,0.,0.),(1.,1.,1.),.5),),
            sample_reference=sampler,budget=s.Budget(wall_seconds=1.))
    assert error.value.outcomes is None and error.value.work['reference_sample_calls']==1


def test_stop_and_wholly_outside_never_sample_reference():
    grid=s.Grid((3,3,3),np.eye(4))
    for capsules,outside in (((),False),((s.Capsule('0','shaft',(-10.,0.,0.),(-9.,0.,0.),0.),),True)):
        result=s.evaluate_contacts(grid,capsules,sample_reference=lambda indices:pytest.fail('unexpected sample'))
        assert result['whole_tool']['touched_reference_cells']==0
        assert result['whole_tool']['outside_reference_fov'] is outside
        assert result['whole_tool']['annotated_positive_encounter'] is (None if outside else False)
        assert result['work']['reference_sample_calls']==0


def test_zero_radius_tolerance_crosses_tile_boundary():
    # Point lies just outside cell0, but inside squared-distance tolerance.
    grid=s.Grid((3,1,1),np.eye(4));cap=s.Capsule('0','tip',(.500005,0.,0.),(.500005,0.,0.),0.)
    expected,_=scalar_cells(cap,grid.shape,np.eye(4));assert (0,0,0) in expected
    seen=[]
    def sample(indices):
        seen.extend(map(tuple,indices));return np.ones(len(indices),bool),np.ones(len(indices),bool)
    result=s.evaluate_contacts(grid,(cap,),sample_reference=sample,budget=s.Budget(tile_edge=1))
    assert set(seen)==expected and result['whole_tool']['positive_reference_cells']==len(expected)


def test_all_admission_removal_and_shared_signature_asts_unchanged():
    old=ast.parse((ROOT/'src/resectionlab/private_vascular_evaluation.py').read_text())
    new=ast.parse((STAGE/'private_vascular_evaluation.py').read_text())
    def nodes(tree):return {node.name:node for node in tree.body if isinstance(node,(ast.FunctionDef,ast.ClassDef))}
    a,b=nodes(old),nodes(new)
    assert set(a)-set(b)=={'_capsule_cells'} and set(b)-set(a)==set()
    for name in a.keys()&b.keys():
        if name!='_evaluate_preflighted_private_vessels':assert ast.dump(a[name])==ast.dump(b[name]),name
    before,after=a['_evaluate_preflighted_private_vessels'],b['_evaluate_preflighted_private_vessels']
    assert ast.dump(before.args)==ast.dump(after.args)
    first_try=next(i for i,n in enumerate(before.body) if isinstance(n,ast.Try))
    next_try=next(i for i,n in enumerate(after.body) if isinstance(n,ast.Try))
    assert ast.dump(ast.Module(body=before.body[:first_try],type_ignores=[]))==ast.dump(ast.Module(body=after.body[:next_try],type_ignores=[]))
    old_try,new_try=before.body[first_try],after.body[next_try]
    assert ast.dump(old_try.handlers[0])==ast.dump(new_try.handlers[0])
    assert ast.dump(ast.Module(body=before.body[first_try+1:],type_ignores=[]))==ast.dump(ast.Module(body=after.body[next_try+1:],type_ignores=[]))
    for name in ('ROOT','VERSION','MAX_DOCUMENT_BYTES','MAX_REFERENCE_VOXELS','MAX_PLANNING_VOXELS','MAX_MICROSTEPS'):
        def assignment(tree):return next(n for n in tree.body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id',None)==name)
        assert ast.dump(assignment(old))==ast.dump(assignment(new)),name


def test_no_models_loaded():
    assert not any(name=='torch' or name.startswith('torch.') for name in sys.modules)
