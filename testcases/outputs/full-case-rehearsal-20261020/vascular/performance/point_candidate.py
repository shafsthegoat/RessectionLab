import sys,time,json
from pathlib import Path
sys.path.insert(0,'src')
import numpy as np
from resectionlab.geometry import GeometryScene,_segment_cell_distances
rng=np.random.default_rng(20)
affine=np.eye(4);affine[:3,:3]=np.diag([-1,1,1]);affine[:3,3]=[109.53500366210938,-123.558837890625,-554.25]
s=GeometryScene(np.zeros((220,220,181),bool),affine)
indices=rng.integers(0,[220,220,181],size=(5000,3));points=rng.uniform([-110,-123,-554],[109,97,-374],size=(20,3))
max_error=0;original_time=fast_time=0
for p in points:
 t=time.perf_counter();a=_segment_cell_distances(s,indices,p,p);original_time+=time.perf_counter()-t
 t=time.perf_counter();origin=(s.world_to_voxel(p)-indices)*s._orthogonal_spacing;excess=np.maximum(np.abs(origin)-s._orthogonal_spacing/2,0);b=np.sqrt(np.sum(excess**2,axis=1));fast_time+=time.perf_counter()-t
 assert np.array_equal(a,b)
 max_error=max(max_error,float(np.max(np.abs(a-b))))
print(json.dumps({'exactFloatEquality':True,'samples':100000,'originalSeconds':original_time,'pointCandidateSeconds':fast_time,'maxError':max_error}))
