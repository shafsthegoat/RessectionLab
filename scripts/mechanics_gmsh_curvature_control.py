#!/usr/bin/env python3
"""One paired analytical discrete-surface mesher capability control, no patients."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import resource
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
VERSION='gmsh-discrete-curvature-capability-v1'
ART=ROOT/'artifacts/mechanics'/VERSION
DECL=ART/'declaration.json'
CASES=('curvature_off','curvature_on')


def sha(p):
    with Path(p).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value


def write(path,value):
    raw=(json.dumps(value,indent=2,allow_nan=False)+'\n').encode()
    if len(raw)>512*1024:raise ValueError('Receipt size cap')
    tmp=Path(path).with_suffix('.tmp');tmp.write_bytes(raw);tmp.replace(path)


def ellipsoid():
    axes=np.array([.035,.020,.010]);nlat=32;nlon=64
    nodes=[[0.,0.,axes[2]]]
    for i in range(1,nlat):
        theta=np.pi*i/nlat
        for j in range(nlon):
            phi=2*np.pi*j/nlon;nodes.append((axes*np.array([np.sin(theta)*np.cos(phi),np.sin(theta)*np.sin(phi),np.cos(theta)])).tolist())
    nodes.append([0.,0.,-axes[2]]);south=len(nodes)-1
    ring=lambda i,j:1+i*nlon+j%nlon
    faces=[]
    for j in range(nlon):faces.append([0,ring(0,j),ring(0,j+1)])
    for i in range(nlat-2):
        for j in range(nlon):
            a,b,c,d=ring(i,j),ring(i,j+1),ring(i+1,j),ring(i+1,j+1);faces.extend([[a,c,b],[b,c,d]])
    for j in range(nlon):faces.append([ring(nlat-2,j),south,ring(nlat-2,j+1)])
    return np.asarray(nodes),np.asarray(faces,dtype=np.int64)


def original_options():
    return {'General.NumThreads':1,'Mesh.MaxNumThreads1D':1,'Mesh.MaxNumThreads2D':1,'Mesh.MaxNumThreads3D':1,
            'Mesh.Algorithm':6,'Mesh.Algorithm3D':1,'Mesh.ElementOrder':2,'Mesh.SecondOrderLinear':1,'Mesh.HighOrderOptimize':0,
            'Mesh.MeshSizeFromPoints':0,'Mesh.MeshSizeFromCurvature':0,'Mesh.MeshSizeExtendFromBoundary':0,
            'Mesh.Optimize':1,'Mesh.OptimizeNetgen':0,'Mesh.RandomSeed':1,'Mesh.Binary':1}


def mesh_case(gmsh,base,vertices,faces,name,preset,charge):
    gmsh.clear();gmsh.model.add(name)
    options=original_options();options['Mesh.MeshSizeFromCurvature']=preset['elements_per_2pi']
    for key,value in options.items():gmsh.option.setNumber(key,value)
    surface=gmsh.model.addDiscreteEntity(2)
    gmsh.model.mesh.addNodes(2,surface,np.arange(1,len(vertices)+1).tolist(),vertices.ravel().tolist())
    gmsh.model.mesh.addElementsByType(surface,2,[],(faces+1).ravel().tolist())
    gmsh.model.mesh.classifySurfaces(np.pi,True,True,np.pi);gmsh.model.mesh.createGeometry()
    tags=[tag for _,tag in gmsh.model.getEntities(2)]
    if not 0<len(tags)<=64:raise ValueError('Analytical patch cap')
    # Read native curvature at up to32 original classified nodes per patch.
    curvature=[]
    for tag in tags:
        _,coords,_=gmsh.model.mesh.getNodes(2,tag,False,False)
        xyz=np.asarray(coords).reshape(-1,3)[:32]
        if len(xyz):
            uv=gmsh.model.getParametrization(2,tag,xyz.ravel().tolist())
            curvature.extend(gmsh.model.getCurvature(2,tag,uv))
    if not curvature or not np.isfinite(curvature).all():raise ValueError('No finite discrete curvature evidence')
    loop=gmsh.model.geo.addSurfaceLoop(tags);gmsh.model.geo.addVolume([loop]);gmsh.model.geo.synchronize()
    field=gmsh.model.mesh.field
    distance=field.add('Distance');field.setNumbers(distance,'SurfacesList',tags);field.setNumber(distance,'Sampling',100)
    threshold=field.add('Threshold')
    for key,value in {'InField':distance,'SizeMin':.012,'SizeMax':.024,'DistMin':.002,'DistMax':.024,'Sigmoid':0,'StopAtDistMax':0}.items():field.setNumber(threshold,key,value)
    field.setAsBackgroundMesh(threshold)
    gmsh.option.setNumber('Mesh.MeshSizeMin',preset['minimum_size_m']);gmsh.option.setNumber('Mesh.MeshSizeMax',.024)
    charge();gmsh.model.mesh.generate(3)
    kinds,element_ids,connectivity=gmsh.model.mesh.getElements(3)
    if list(kinds)!=[11]:raise ValueError('Only native tet10')
    _,dimension,order,count,local,primary=gmsh.model.mesh.getElementProperties(11)
    if (dimension,order,count,primary)!=(3,2,10,4):raise ValueError('Tet10 properties changed')
    permutation=base.tet10_permutation(local);ids,coords,_=gmsh.model.mesh.getNodes()
    cells=np.asarray(connectivity[0],dtype=np.int64).reshape(-1,10)[:,permutation]
    used=np.unique(cells);order_ids=np.argsort(ids);sorted_ids=np.asarray(ids)[order_ids];positions=np.searchsorted(sorted_ids,used)
    if np.any(positions>=len(sorted_ids)) or not np.array_equal(sorted_ids[positions],used):raise ValueError('Missing native node')
    nodes=np.asarray(coords).reshape(-1,3)[order_ids[positions]]
    curvature=np.asarray(curvature)
    return nodes,np.searchsorted(used,cells).astype(np.int64),dict(patches=len(tags),samples=len(curvature),minimum_per_m=float(curvature.min()),median_per_m=float(np.median(curvature)),maximum_per_m=float(curvature.max()),nonzero=int((curvature>0).sum()))


def binding():
    config=json.loads(DECL.read_text())
    if config.get('authorized_one_analytical_pair') is not True or config['case_order']!=list(CASES):raise ValueError('One predeclared pair required')
    bound={str(DECL):sha(DECL)}
    for path,digest in config['inputs'].items():
        actual=(ROOT/path).resolve()
        if not actual.is_relative_to(ROOT) or sha(actual)!=digest:raise ValueError('Input changed:'+path)
        bound[str(actual)]=digest
    return config,bound


def worker(output):
    output=Path(output)
    if (output/'result.json').exists():raise FileExistsError('One worker attempt only')
    config,bound=binding();started=time.monotonic();gmsh=None
    result=dict(status='running',cases=[{'case':case,'status':'not_executed'} for case in CASES],native_generations=0,solver_calls=0,patient_input=False,training=False)
    def check():
        if time.monotonic()-started>config['caps']['cooperative_seconds']:raise TimeoutError('Cooperative pair budget')
    def charge():
        check()
        if result['native_generations']>=2:raise ValueError('Only two generations')
        result['native_generations']+=1;write(output/'result.json',result)
    try:
        resource.setrlimit(resource.RLIMIT_FSIZE,(4*1024**2,4*1024**2))
        base=load(ROOT/'scripts/mechanics_patient_mesh.py','curvature_original_geometry_checks')
        V,F=ellipsoid();source=base.validate_surface(V,F);axes=np.array(config['ellipsoid_axes_m'])
        analytic_volume=4*np.pi*np.prod(axes)/3
        if np.max(np.abs(np.sum((V/axes)**2,axis=1)-1))>1e-12:raise ValueError('Analytical source equation')
        result.update(source=source,analytical_volume_m3=float(analytic_volume),source_volume_relative_error=abs(source['enclosed_volume_m3']/analytic_volume-1))
        np.savez_compressed(output/'analytical-input.npz',vertices_m=V,triangles=F)
        if 'gmsh' in sys.modules:raise ValueError('Ambient Gmsh prohibited')
        runtime=config['gmsh_runtime'];gmsh=load(ROOT/runtime['module_path'],'gmsh')
        if gmsh.__version__!='4.15.2' or Path(gmsh.lib._name).resolve()!=(ROOT/runtime['library_path']).resolve():raise ValueError('Pinned Gmsh origin')
        gmsh.initialize([],readConfigFiles=False,run=False)
        from vtkmodules.vtkCommonCore import vtkSMPTools
        if not vtkSMPTools.SetBackend('Sequential') or vtkSMPTools.GetBackend()!='Sequential':raise ValueError('Sequential VTK')
        native_distance=base.vtk_distance_function(V,F)
        for row in result['cases']:
            name=row['case'];check();before=time.monotonic();row['status']='running';write(output/'result.json',result)
            X,E,curvature=mesh_case(gmsh,base,V,F,name,config['presets'][name],charge)
            row.update(nodes=len(X),elements=len(E),native_curvature=curvature)
            if len(X)>20000 or len(E)>30000:raise ValueError('Analytical count cap')
            np.savez_compressed(output/(name+'.npz'),nodes_m=X,tet10_indices=E)
            quality,Y,G=base.validate_tet10(X,E,{'maximum_nodes':20000,'maximum_elements':30000},{'quality':{'midpoint_tolerance_m':1e-12,'minimum_mean_ratio':.01,'maximum_overlap_candidate_pairs':1000000}})
            check();fidelity={'cover_radius_m':.0005,'maximum_samples_per_direction':300000}
            forward=base.directed_surface_bound(V,F,base.vtk_distance_function(Y,G),fidelity)
            check();reverse=base.directed_surface_bound(Y,G,native_distance,fidelity)
            unique_edges=np.unique(np.sort(G[:,[[0,1],[1,2],[2,0]]],axis=2).reshape(-1,2),axis=0)
            edge_lengths=np.linalg.norm(Y[unique_edges[:,0]]-Y[unique_edges[:,1]],axis=1)
            row.update(status='completed_capability_observation',quality=quality,source_to_mesh=forward,mesh_to_source=reverse,
                boundary_edge_quantiles_m=np.quantile(edge_lengths,[0,.1,.5,.9,1]).tolist(),elapsed_seconds=time.monotonic()-before)
            write(output/'result.json',result)
        result['status']='completed_one_analytical_pair'
    except BaseException as error:
        result.update(status='failed_or_incomplete',error=f'{type(error).__name__}: {error}')
        for row in result['cases']:
            if row['status']=='running':row['status']='failed'
    finally:
        if gmsh is not None:
            try:gmsh.finalize()
            except BaseException as error:result.update(status='failed_or_incomplete',finalization_error=str(error))
        result.update(elapsed_seconds=time.monotonic()-started,input_hashes=bound,inputs_unchanged={p:Path(p).is_file() and sha(p)==d for p,d in bound.items()})
        if not all(result['inputs_unchanged'].values()):result['status']='failed_or_incomplete'
        write(output/'result.json',result)
    return 0 if result['status']=='completed_one_analytical_pair' else 1


def run(output):
    config,bound=binding();output=Path(output).resolve()
    if output!=ROOT/config['attempt_directory']:raise ValueError('Declared one attempt only')
    output.mkdir(parents=True,exist_ok=False)
    runtime=load(ROOT/'scripts/febio_runtime.py','curvature_existing_supervisor')
    guard=load(ROOT/'scripts/mechanics_patient_mesh_candidate_run.py','curvature_existing_output_guard')
    guard.OUTPUT_BYTES=16*1024**2 # Explicit declared bound; original helper file unchanged.
    env=runtime.private_environment(runtime.declaration());env['PYTHONDONTWRITEBYTECODE']='1'
    supervision,disk=guard.supervised(runtime,[sys.executable,'-B',str(Path(__file__).resolve()),'worker','--output',str(output)],output,env,{'aggregate_seconds':30,'process_group_rss_bytes':1024**3})
    result=json.loads((output/'result.json').read_text()) if (output/'result.json').exists() else {}
    passed=supervision['status']=='completed' and supervision['exit_code']==0 and disk['error'] is None and result.get('status')=='completed_one_analytical_pair' and all(sha(Path(p))==d for p,d in bound.items())
    receipt=dict(status='completed' if passed else 'failed_or_incomplete',supervision=supervision,output_guard=disk,no_retry=True)
    write(output/'acceptance.json',receipt)
    try:guard.output_usage(output)
    except BaseException as error:passed=False;receipt.update(status='failed_or_incomplete',output_error=str(error));write(output/'acceptance.json',receipt)
    return 0 if passed else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['run','worker']);parser.add_argument('--output',required=True)
    args=parser.parse_args();raise SystemExit((run if args.mode=='run' else worker)(args.output))
