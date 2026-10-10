"""Generated cell geometry only; no source patient masks or metadata opened."""
import ast,copy
from pathlib import Path
import numpy as np
import pytest
import diagnostic_core as core
import run_owned as parent

def masks():
 S=np.zeros((9,9,9),bool);S[5:7,2:7,2:7]=True
 T=np.zeros_like(S);T[6,4,4]=True
 Ds=np.zeros_like(S);Ds[1:8,1:8,1:8]=True
 return S,T,Ds

def test_full_cell_partition_includes_plane_touch_and_reverses_direction():
 i=np.array([[2,0,0],[3,0,0],[4,0,0],[5,0,0]])
 bins,_,_=core.depth_bins(i,np.eye(4),np.array([3.5,0,0]),np.array([1.,0,0]))
 assert [int(x.sum()) for x in bins.values()]==[1,2,1]
 reverse,_,_=core.depth_bins(i,np.eye(4),np.array([3.5,0,0]),np.array([-1.,0,0]))
 np.testing.assert_array_equal(bins['wholly_proximal'],reverse['wholly_inward'])
 assert np.all(sum(bins.values())==1)

def test_rotated_affine_depth_partition_is_same():
 a=np.eye(4);a[:3,:3]=[[0,-1,0],[1,0,0],[0,0,1]];a[:3,3]=[12,-4,8]
 i=np.array([[2,0,0],[3,0,0],[4,0,0],[5,0,0]]);entry=a[:3,:3]@np.array([3.5,0,0])+a[:3,3]
 bins,_,_=core.depth_bins(i,a,entry,a[:3,0])
 assert [int(x.sum()) for x in bins.values()]==[1,2,1]

def test_hypothetical_seeds_do_not_bypass_unknown_shell_or_change_masks():
 S,T,Ds=masks();before=[x.copy() for x in (S,T,Ds)]
 x=core.aperture_candidates(S,T,Ds,np.eye(4),np.array([4.5,4,4]),np.array([1.,0,0]),2.)
 assert x['source_known_zero_candidates']['count']>0
 assert x['actual_initial_connected_free_count']==0
 assert x['boundary_U_count']==x['boundary_cell_count'] and x['boundary_known_zero_seed_count']==0
 assert x['not_an_admitted_cavity'] and x['candidate_face_adjacent_material_cells']>0
 assert sum(c['source_known_zero_cells'] for c in x['candidate_connected_components'])>0
 for a,b in zip(before,(S,T,Ds)):np.testing.assert_array_equal(a,b)

@pytest.mark.parametrize('mask',['S','T','Ds'])
def test_positive_or_unknown_candidate_is_excluded(mask):
 S,T,Ds=masks();entry=np.array([4.5,4,4]);normal=np.array([1.,0,0])
 old=core.aperture_candidates(S,T,Ds,np.eye(4),entry,normal,2.)['source_known_zero_candidates']['count']
 if mask=='S':S[4,4,4]=True
 elif mask=='T':T[4,4,4]=True
 else:Ds[4,4,4]=False
 new=core.aperture_candidates(S,T,Ds,np.eye(4),entry,normal,2.)['source_known_zero_candidates']['count']
 assert new==old-1

def test_oblique_aperture_normal_refused():
 with pytest.raises(ValueError,match='axis0'):
  core.aperture_candidates(*masks(),np.eye(4),np.array([4.5,4,4]),np.array([1.,1,0])/np.sqrt(2),2.)

def test_initial_and_swept_four_queries_counted_without_native_preview():
 S,T,Ds=masks();motion={'proposal_id':'generated-one','tool_id':'generic_suction','entry_mm':[4.5,4,4],'tip_mm':[6.,4,4],'reason':'generated_unchanged_reason'}
 calls=[];r=core.diagnose_case(S,T,Ds,np.eye(4),{'centre_mm':[4.5,4,4],'normal_inward':[1.,0,0],'radius_mm':2.},{'emitted':[motion]},on_query=lambda:calls.append(1))
 assert r['geometric_queries']==len(calls)==4
 assert [q['part'] for q in r['queries']]==['initial_shaft','initial_tip','swept_shaft','swept_tip']
 assert r['queries'][0]['intersections']['derived_U']['by_full_cell_depth']['wholly_proximal']['count']>0
 assert r['native_previews']==r['transitions']==r['models']==0 and r['source_or_state_modified'] is False
 for q in r['queries']:
  for summary in q['intersections'].values():assert sum(x['count'] for x in summary['by_full_cell_depth'].values())==summary['count']
 assert r['raw_S_full_T_U_global_depth_counts']['raw_S']['count']==S.sum()
 assert r['raw_S_full_T_U_global_depth_counts']['full_T']['count']==T.sum()

def test_failed_geometry_query_still_counts_invocation(monkeypatch):
 calls=[]
 def fail(*a):raise RuntimeError('generated geometry failure')
 monkeypatch.setattr(core,'capsule_voxel_indices',fail)
 with pytest.raises(RuntimeError,match='geometry failure'):
  core.diagnose_case(*masks(),np.eye(4),{'centre_mm':[4.5,4,4],'normal_inward':[1.,0,0],'radius_mm':2.},{'emitted':[{'proposal_id':'x','tool_id':'generic_suction','entry_mm':[4.5,4,4],'tip_mm':[6,4,4],'reason':'x'}]},on_query=lambda:calls.append(1))
 assert calls==[1]

def test_parent_monitor_and_cleanup_are_reused_exactly():
 here=Path(__file__).resolve().parents[1]
 new=ast.parse((here/'run_owned.py').read_text());old=ast.parse((here.parent/'remind-partial-domain-public-preparation-v1/run_owned.py').read_text())
 for name in ('cleanup_phase','tree_bytes','final_extents'):
  get=lambda t:ast.dump(next(x for x in t.body if isinstance(x,ast.FunctionDef) and x.name==name),include_attributes=False)
  assert get(new)==get(old)
 compile((here/'mask_worker.py').read_text(),'mask_worker.py','exec')

def test_fast_exit_log_cap(tmp_path,monkeypatch):
 monkeypatch.setitem(parent.CAPS,'log_bytes',10)
 output=tmp_path/'out';supervision=tmp_path/'supervision';output.mkdir();supervision.mkdir();(supervision/'worker.log').write_bytes(b'x'*11)
 with pytest.raises(ValueError,match='final_log_cap'):parent.final_extents(output,supervision)
