"""Tiny independent streaming contacts; no benchmark or scientific payloads."""
from dataclasses import replace
from pathlib import Path
import importlib.util
import os
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('independent_streaming_candidate', ROOT/'build/private-vascular-streaming-preparation-v1/streaming_contact.py')
s=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=s
spec.loader.exec_module(s)
from resectionlab.evaluation import segment_box_distance_sq

FORBIDDEN=[]
def guard(event,args):
    if event in ('subprocess.Popen','os.system','os.exec','socket.connect'):
        FORBIDDEN.append(event)
        raise AssertionError('External processes/network forbidden in tiny controls')
    if event=='open' and isinstance(args[0],(str,bytes)):
        name=os.fsdecode(args[0]).lower()
        if name.endswith(('.nii','.nii.gz','.mat','.tar','.pt','.ckpt','.bin','.npy','.npz','.safetensors','.dcm','.h5','.hdf5')):
            FORBIDDEN.append(name)
            raise AssertionError('Patient/model arrays forbidden in tiny controls')
sys.addaudithook(guard)


def grid(shape=(5,6,7),angle=0.,reflected=False):
    rotation=np.array([[np.cos(angle),-np.sin(angle),0.],[np.sin(angle),np.cos(angle),0.],[0.,0.,1.]])
    if reflected: rotation[:,0]*=-1
    affine=np.eye(4);affine[:3,:3]=rotation@np.diag([.7,1.3,1.9]);affine[:3,3]=[3.1,-7.2,2.4]
    return s.Grid(shape,tuple(map(tuple,affine)))


def capsule(g,action,part,start,end,radius):
    a=np.asarray(g.affine_ras_mm);spacing=np.linalg.norm(a[:3,:3],axis=0);rotation=a[:3,:3]/spacing
    points=[tuple(rotation@np.array(point)+a[:3,3]) for point in (start,end)]
    return s.Capsule(action,part,*points,radius)


def oracle(reference,capsules):
    g=reference.grid;a=np.asarray(g.affine_ras_mm)
    spacing=np.linalg.norm(a[:3,:3],axis=0);rotation=a[:3,:3]/spacing
    indices=np.array(list(np.ndindex(g.shape)),dtype=np.int64)
    mask,coverage=reference.sample(indices)
    groups={'shaft':set(),'tip':set(),'whole_tool':set()}
    per_action={c.action_id:set() for c in capsules}
    outside={key:False for key in groups}
    for c in capsules:
        p,q=[rotation.T@(np.asarray(point)-a[:3,3]) for point in (c.start_ras_mm,c.end_ras_mm)]
        touched=set()
        for row,index in enumerate(indices):
            center=index*spacing
            if segment_box_distance_sq(p,q,center-spacing/2,center+spacing/2)<=c.radius_mm**2+1e-10:
                touched.add(row)
        groups[c.part]|=touched;groups['whole_tool']|=touched;per_action[c.action_id]|=touched
        out=bool(np.any(np.minimum(p,q)-c.radius_mm<-.5*spacing) or np.any(np.maximum(p,q)+c.radius_mm>(np.asarray(g.shape)-.5)*spacing))
        outside[c.part]|=out;outside['whole_tool']|=out
    def count(rows):
        keys=np.array(sorted(rows),dtype=int)
        return (len(rows),int(mask[keys].sum()),int((~coverage[keys]).sum()))
    return {k:count(v) for k,v in groups.items()},{k:count(v) for k,v in per_action.items()},outside


def compare(reference,capsules,edge):
    expected,actions,outside=oracle(reference,capsules)
    result=s.evaluate_generated_contacts(reference,capsules,budget=s.Budget(tile_edge=edge,coarse_batch=3))
    for part,values in expected.items():
        got=result[part]
        assert (got['touched_reference_cells'],got['positive_reference_cells'],got['unknown_reference_cells'])==values
        assert got['outside_reference_fov']==outside[part]
    for name,values in actions.items():
        got=result['per_action'][name]
        assert (got['touched_reference_cells'],got['positive_reference_cells'],got['unknown_reference_cells'])==values
    assert result['work']['maximum_tile_cells']<=edge**3
    return result


@pytest.mark.parametrize('edge',[1,2,4])
def test_uneven_tiles_and_repeated_capsules_match_full_scalar_union(edge):
    g=grid();ref=s.GeneratedReference(g,'all_covered','partial_x_half')
    shaft=capsule(g,'one','shaft',[-.2,.8,.7],[3.1,6.1,10.1],.46)
    tip=capsule(g,'two','tip',[1.2,2.6,3.8],[2.8,3.3,5.7],.71)
    result=compare(ref,(shaft,shaft,tip,replace(tip,action_id='three')),edge)
    assert result['per_action']['two']==result['per_action']['three']
    assert result['whole_tool']['touched_reference_cells']<=sum(result[p]['touched_reference_cells'] for p in ('shaft','tip'))


@pytest.mark.parametrize('angle,reflected',[(.37,False),(.71,True)])
def test_rotated_anisotropic_reflected_grid_matches_scalar(angle,reflected):
    g=grid((5,4,3),angle,reflected);ref=s.GeneratedReference(g,'lattice','full')
    caps=(capsule(g,'turn','shaft',[.1,.8,.5],[2.8,3.9,3.8],.55),
          capsule(g,'turn','tip',[.3,.2,1.7],[2.1,2.6,3.8],.4))
    compare(ref,caps,2)


def test_squared_tolerance_retains_near_tangent_cell_across_tile_face():
    g=s.Grid((4,4,4),tuple(map(tuple,np.eye(4))))
    ref=s.GeneratedReference(g,'all_covered','full')
    # The lower-y tile ends at 1.5 mm. The point is just beyond radius .25
    # from that face, but inside the established squared-distance tolerance.
    cap=s.Capsule('tangent','tip',(1.,1.75000000005,1.),(1.,1.75000000005,1.),.25)
    result=compare(ref,(cap,),2)
    assert result['whole_tool']['touched_reference_cells']==2


def test_outside_reference_with_no_contact_is_unknown_and_no_sample():
    g=s.Grid((4,4,4),tuple(map(tuple,np.eye(4))))
    ref=s.GeneratedReference(g,'empty','full')
    cap=s.Capsule('outside','shaft',(20.,20.,20.),(21.,21.,21.),.1)
    result=compare(ref,(cap,),2)
    assert result['whole_tool']['annotated_positive_encounter'] is None
    assert result['work']['reference_sample_calls']==0
    assert result['work']['tiles_pruned']==result['work']['tiles_scanned']==8


@pytest.mark.parametrize('field',['max_tile_capsule_pairs','max_cell_capsule_pairs','max_sampled_cells'])
def test_zero_budget_fails_without_completed_or_partial_outcome(field):
    g=s.Grid((2,2,2),tuple(map(tuple,np.eye(4))))
    ref=s.GeneratedReference(g,'all_covered','full')
    cap=s.Capsule('budget','tip',(0.,0.,0.),(1.,1.,1.),1.)
    with pytest.raises(s.BudgetExceeded) as captured:
        s.evaluate_generated_contacts(ref,(cap,),budget=replace(s.Budget(tile_edge=2),**{field:0}))
    error=captured.value
    assert error.reason == ('reference_sampled_cells' if field=='max_sampled_cells' else field.removeprefix('max_'))
    assert error.completed is False and error.outcomes is None
    assert error.work['reference_sample_calls']==0


def test_cancellation_returns_no_completed_result():
    ref=s.GeneratedReference(grid())
    with pytest.raises(s.BudgetExceeded) as captured:
        s.evaluate_generated_contacts(ref,(),cancelled=lambda:True)
    assert captured.value.reason=='cancelled'
    assert captured.value.completed is False and captured.value.outcomes is None


def test_no_torch_external_process_or_payload_action():
    assert FORBIDDEN==[]
    assert 'torch' not in sys.modules
