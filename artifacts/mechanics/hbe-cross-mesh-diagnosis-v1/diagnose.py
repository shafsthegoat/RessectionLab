"""Post hoc saved-output diagnosis only. Never reads measured curves or runs a solver/mesher."""
from pathlib import Path
import hashlib,json,math,importlib.util
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
RUN=ROOT/'outputs/mechanics/hbe-01-03-poc-v1/experiment-accelerate-csc-v1'

def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def load(p):return json.loads(p.read_bytes())
def bound(b):
    p=ROOT/b['path'];assert sha(p)==b['sha256'];return p
summary=load(RUN/'reference-numerical.json');physics_path=bound(summary['source_bindings']['physics'])
spec=importlib.util.spec_from_file_location('frozen_physics_readonly',physics_path);physics=importlib.util.module_from_spec(spec);spec.loader.exec_module(physics)
meshes={};meshmeta={};inputs={str(physics_path.relative_to(ROOT)):sha(physics_path),str((RUN/'reference-numerical.json').relative_to(ROOT)):sha(RUN/'reference-numerical.json')}
for N in (4,8,12):
    receipt=load(RUN/f'runs/compression-N{N}-S60-reference/readout.json');p=bound(receipt['primitive_bindings']['mesh']);metadata=load(p);mesh=physics.HexMesh.from_manifest(metadata);meshes[N]=mesh
    points=physics.fixed_probes(mesh.radius_m,mesh.height_m);mapping=mesh.probe_map(points)
    residual=[];nearest=[];hosts=[]
    for point,(cell,w) in zip(points,mapping):
        nodes=mesh._nodes[mesh._cells[cell]];residual.append(float(np.linalg.norm(np.array(w)@nodes-point)));nearest.append(float(np.min(np.linalg.norm(mesh._nodes-point,axis=1))))
        hosts.append({'cell_id':cell+1,'weights':list(w),'point_m':point.tolist(),'rest_reconstruction_error_m':residual[-1],'nearest_node_distance_m':nearest[-1]})
    meshmeta[N]={'mesh_sha256':sha(p),'nodes':len(mesh._nodes),'elements':mesh.element_count,'radius_m':mesh.radius_m,'height_m':mesh.height_m,'axial_layers':len(np.unique(mesh._nodes[:,2]))-1,'rest_volume_m3':mesh.rest_volume_m3,'area_m2':mesh.rest_volume_m3/mesh.height_m,'area_over_exact_circle':mesh.rest_volume_m3/(math.pi*mesh.radius_m**2*mesh.height_m),'max_rest_map_error_m':max(residual),'hosts':hosts}
    inputs[str(p.relative_to(ROOT))]=sha(p)

def endpoint(path,count):
    values=None;step=None
    with path.open() as f:
        for raw in f:
            line=raw.strip()
            if line.startswith('*Step'):
                step=int(line.split('=')[1])
                if step==60:values=np.empty((count,9));seen=set()
            elif step==60 and line and not line.startswith('*'):
                row=line.split(',');i=int(row[0])-1;assert i not in seen and len(row)==10;seen.add(i);values[i]=[float(x) for x in row[1:]]
    assert values is not None and len(seen)==count and np.isfinite(values).all();return values

# Independent bilinear-XY / linear-Z inverse for the persisted extruded hex8 cells.
def alternative_hosts(mesh,point):
    answers=[]
    for i,cell in enumerate(mesh._cells):
        X=mesh._nodes[cell]
        if np.any(point<X.min(0)-1e-13) or np.any(point>X.max(0)+1e-13):continue
        bottom=X[:4];xi=np.zeros(2)
        for _ in range(30):
            a,b=xi;w=np.array([(1-a)*(1-b),(1+a)*(1-b),(1+a)*(1+b),(1-a)*(1+b)])/4
            d=np.array([[-(1-b),-(1-a)],[1-b,-(1+a)],[1+b,1+a],[-(1+b),1-a]])/4
            residual=w@bottom[:,:2]-point[:2]
            if np.linalg.norm(residual)<1e-15:break
            xi-=np.linalg.solve(bottom[:,:2].T@d,residual)
        z=(point[2]-X[0,2])/(X[4,2]-X[0,2]);weights=np.r_[w*(1-z),w*z]
        if np.max(np.abs(xi))<=1+1e-9 and -1e-9<=z<=1+1e-9 and np.linalg.norm(weights@X-point)<1e-13:answers.append((i,weights))
    assert answers;return answers

out={'schema':'saved-hbe-cross-mesh-diagnosis-v1','classification':'post_hoc_diagnostic_not_acceptance_or_new_experiment','original_failed_numerical_gate_preserved':True,'meshes':meshmeta,'branches':{},'inputs':inputs,'solver_or_mesher_or_curve_access':False}
for branch in ('compression','tension','torsion_pos','torsion_neg'):
    receipts={};current={};u={};checks={}
    for N,mesh in meshes.items():
        path=RUN/f'runs/{branch}-N{N}-S60-reference/readout.json';record=load(path);receipts[N]=record;inputs[str(path.relative_to(ROOT))]=sha(path)
        rawpath=bound(record['primitive_bindings']['nodes']);inputs[str(rawpath.relative_to(ROOT))]=sha(rawpath);v=endpoint(rawpath,len(mesh._nodes));current[N]=v[:,:3];u[N]=np.array(record['probe_displacements_m'])
        points=physics.fixed_probes(mesh.radius_m,mesh.height_m);maximum=0.;tie_spread=0.;count=0
        for j,point in enumerate(points):
            answers=alternative_hosts(mesh,point);pred=[w@(current[N][mesh._cells[i]]-mesh._nodes[mesh._cells[i]]) for i,w in answers]
            maximum=max(maximum,max(float(np.linalg.norm(x-u[N][-1,j])) for x in pred));tie_spread=max(tie_spread,max(float(np.linalg.norm(x-pred[0])) for x in pred));count+=len(answers)>1
        checks[N]={'independent_endpoint_interpolation_max_error_m':maximum,'all_host_tie_spread_max_m':tie_spread,'probes_with_multiple_valid_hosts':count,'raw_displacement_vs_current_rest_max_error_m':float(np.max(np.linalg.norm(v[:,3:6]-(v[:,:3]-mesh._nodes),axis=1)))}
    results=[]
    for a,b in ((4,8),(8,12)):
        delta=u[b]-u[a];norm=np.linalg.norm(delta,axis=-1);t,i=map(int,np.unravel_index(np.argmax(norm),norm.shape));point=points[i]
        force_a=np.array(receipts[a]['applied_force_N']);force_b=np.array(receipts[b]['applied_force_N']);diff=force_b-force_a
        nodal=[];nodal_points=[]
        for k,x in enumerate(meshes[a]._nodes):
            if not 1e-12<x[2]<meshes[a].height_m-1e-12:continue
            dist=np.linalg.norm(meshes[b]._nodes-x,axis=1);j=int(np.argmin(dist))
            if dist[j]<1e-12:
                nodal.append(float(np.linalg.norm((current[b][j]-meshes[b]._nodes[j])-(current[a][k]-x))));nodal_points.append(x.tolist())
        nodal_index=int(np.argmax(nodal))
        results.append({'coarse_N':a,'fine_N':b,'max_probe_change_m':float(norm[t,i]),'load_step':t,'load_fraction':t/60,'zero_based_probe_index':i,'point_m':point.tolist(),'point_radius_over_R':float(np.linalg.norm(point[:2])/meshes[b].radius_m),'point_z_over_H':float(point[2]/meshes[b].height_m),'max_delta_vector_m':delta[t,i].tolist(),'endpoint_probe_rms_change_m':float(np.sqrt(np.mean(norm[-1]**2))),'endpoint_probe_median_change_m':float(np.median(norm[-1])),'endpoint_probes_exceeding_original_8um_limit':int(np.sum(norm[-1]>8e-6)),'endpoint_center_plane_max_probe_change_m':float(np.max(norm[-1,25:50])),'shared_non_end_plane_node_count':len(nodal),'shared_non_end_plane_node_max_displacement_change_m':max(nodal),'shared_node_max_point_m':nodal_points[nodal_index],'force_max_difference_N':float(np.max(np.abs(diff))),'force_max_step':int(np.argmax(abs(diff))),'force_endpoint_delta_N':float(diff[-1])})
    forces={N:float(receipts[N]['applied_force_N'][-1]) for N in meshes};area_corrected={N:forces[N]/meshmeta[N]['area_over_exact_circle'] for N in meshes}
    out['branches'][branch]={'pairs':results,'endpoint_forces_N':forces,'diagnostic_only_endpoint_force_times_exact_over_polygon_area_N':area_corrected,'endpoint_interpolation_checks':checks}
for name,digest in inputs.items():assert sha(ROOT/name)==digest
with Path(__file__).with_name('diagnosis.json').open('x') as f:json.dump(out,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
for N,m in meshmeta.items():print('mesh',N,{k:v for k,v in m.items() if k!='hosts'})
for name,b in out['branches'].items():print(name,json.dumps({k:v for k,v in b.items() if k!='endpoint_interpolation_checks'}));print('interpolationchecks',name,b['endpoint_interpolation_checks'])
