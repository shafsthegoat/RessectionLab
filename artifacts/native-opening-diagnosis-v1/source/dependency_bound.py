"""Two-tool solid-slab depth bound from saved cross sections; no native calls."""
import hashlib
import json
import math
from pathlib import Path
import time

HERE=Path(__file__).resolve().parent


def distance(cell,column,spacing):
    delta=[abs(a-b)*s for a,b,s in zip(cell,column,spacing)]
    return (math.hypot(*(max(0.,d-s/2) for d,s in zip(delta,spacing))),
            math.hypot(*(d+s/2 for d,s in zip(delta,spacing))))


def main():
    started=time.perf_counter(); raw=(HERE/'result.json').read_bytes(); data=json.loads(raw)
    small,large=data['tools']; results=[]
    for report in data['original']:
        sx,sy,sz=report['spacing_xyz_mm']; columns=report['columns']
        aspiration={tuple(r['column']):r['solid_slab_tip_depth_upper_bound_mm'] for r in report['columns_by_tool'] if r['tool_id']==large['tool_id']}
        assert all(d is not None for d in aspiration.values())
        cases=[]
        for column in columns:
            margin=math.ceil(small['shaft_radius_mm']/min(sy,sz))+2
            candidates=[]
            for y in range(column[0]-margin,column[0]+margin+1):
                for z in range(column[1]-margin,column[1]+margin+1):
                    cell=(y,z); near,_=distance(cell,column,(sy,sz))
                    if near>small['shaft_radius_mm']+1e-9:continue
                    # Any small-tool cylinder could remove this cell: no
                    # dependency bound is claimed for it.
                    if any(distance(cell,c,(sy,sz))[1]<=small['tip_radius_mm']-1e-10 for c in columns):continue
                    fronts=[]
                    for c in columns:
                        far=distance(cell,c,(sy,sz))[1]
                        if far<=large['tip_radius_mm']-1e-10:
                            fronts.append(aspiration[tuple(c)]+math.sqrt(large['tip_radius_mm']**2-far**2))
                    front=max(fronts,default=0.)
                    # Layer k spans [k*sx,(k+1)*sx]. Even optimistically its
                    # far face cannot be cut if beyond all large-tool fronts.
                    retained_layer=max(0,math.floor(front/sx))
                    limit=retained_layer*sx+small['tip_length_mm']-math.sqrt(max(0.,small['shaft_radius_mm']**2-near**2))
                    candidates.append({'cell_transverse':cell,'large_tool_cut_front_upper_bound_mm':front,
                        'first_certifiably_retained_layer':retained_layer,'layer_proximal_face_depth_mm':retained_layer*sx,
                        'small_tool_tip_depth_upper_bound_mm':limit})
            assert candidates
            cases.append({'column':column,'witness':min(candidates,key=lambda r:r['small_tool_tip_depth_upper_bound_mm'])})
        record={'name':report['name'],'small_tool_columns':cases,
            'maximum_small_tool_tip_depth_upper_bound_mm':max(r['witness']['small_tool_tip_depth_upper_bound_mm'] for r in cases),
            'maximum_large_tool_tip_depth_upper_bound_mm':max(aspiration.values())}
        results.append(record)
        center=next(r for r in cases if r['column']==[0,0])
        print(report['name'],'center suction bound',center['witness']['small_tool_tip_depth_upper_bound_mm'],
            'all-column suction bound',record['maximum_small_tool_tip_depth_upper_bound_mm'],
            'all-column aspirator bound',record['maximum_large_tool_tip_depth_upper_bound_mm'])
    output={'scope':'optimistic_two_tool_dependency_bound_uniform_solid_slab_only',
        'assumptions':data['assumptions']+['slab front lies at a voxel face; all columns begin on that same plane',
            'all possible earlier cuts are credited optimistically; no sequential reachability asserted'],
        'results':results,'elapsed_seconds':time.perf_counter()-started,
        'cross_section_result_sha256':hashlib.sha256(raw).hexdigest(),
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'patient_arrays_or_native_trajectories':0}
    with (HERE/'dependency-result.json').open('x') as f:json.dump(output,f,indent=2);f.write('\n')
    print('elapsed',output['elapsed_seconds'])


if __name__=='__main__':main()
