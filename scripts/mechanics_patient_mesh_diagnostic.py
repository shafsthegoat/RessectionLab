#!/usr/bin/env python3
"""One released saved-geometry diagnostic; never mesh, image, landmark or solve.

Pure numerical helpers are importable for analytical controls. VTK and the
saved arrays are loaded only by the separately released worker.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import resource
import re
import subprocess
import sys
import time
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

ROOT = Path(__file__).resolve().parents[1]
VERSION = 'resect-case4-saved-mesh-diagnostic-v1'
MANIFEST = 'manifests/experiments/'+VERSION+'.json'
BASE_PATH = 'scripts/mechanics_patient_mesh.py'
BASE_SHA = 'd8aae815c66ae017e925c1b4ba16f1e6a64c1d2ee314153ef9889c165d64b74d'
CLOSURE = (str(Path(__file__).relative_to(ROOT)), MANIFEST, BASE_PATH,
           'scripts/mechanics_patient_mesh_candidate_run.py', 'scripts/febio_runtime.py',
           'artifacts/febio-runtime-investigation-v1/prospective-runtime.json')
CAPS = dict(parent_seconds=120, cooperative_seconds=110, rss_bytes=2*1024**3,
            numerical_threads=1, maximum_distance_queries=1200000,
            output_bytes=8*1024**2, attempts=1, retries=0)
FACES = np.array([[1,2,3],[0,3,2],[0,1,3],[0,2,1]])
EDGES = np.array([[0,1],[1,2],[2,0],[0,3],[1,3],[2,3]])


def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream,'sha256').hexdigest()


def load(path, name):
    spec=importlib.util.spec_from_file_location(name,path)
    value=importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
    if Path(value.__file__).resolve()!=Path(path).resolve(): raise ValueError('Import origin changed')
    return value


def write(path, value):
    raw=(json.dumps(value,indent=2,allow_nan=False)+'\n').encode()
    if len(raw)>1024**2: raise ValueError('Diagnostic JSON cap')
    temporary=Path(path).with_suffix('.tmp'); temporary.write_bytes(raw); temporary.replace(path)


def triangle_geometry(vertices, faces):
    tri=np.asarray(vertices,float)[faces]
    cross=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]); twice=np.linalg.norm(cross,axis=1)
    if not np.isfinite(tri).all() or np.any(twice<=0): raise ValueError('Invalid triangles')
    return tri, twice/2, cross/twice[:,None]


def indexed_samples(vertices, faces, cover_m, maximum_points, chunk_triangles=256):
    """Exact v1 grouped lattice/order/cover, with triangle identity retained."""
    if not np.isfinite(cover_m) or cover_m<=0: raise ValueError('Positive physical cover required')
    tri,_,_=triangle_geometry(vertices,faces)
    longest=np.max(np.linalg.norm(tri-np.roll(tri,1,axis=1),axis=2),axis=1)
    divisions=np.maximum(1,np.ceil(longest/cover_m).astype(int))
    count=int(np.sum((divisions+1)*(divisions+2)//2))
    if count>maximum_points: raise ValueError('Sample cap before allocation')
    for n in np.unique(divisions):
        weights=np.array([(i/n,j/n,1-(i+j)/n) for i in range(n+1) for j in range(n+1-i)])
        group=np.flatnonzero(divisions==n)
        for offset in range(0,len(group),chunk_triangles):
            chosen=group[offset:offset+chunk_triangles]
            yield (np.einsum('pi,tij->tpj',weights,tri[chosen]).reshape(-1,3),
                   chosen,len(weights),float(np.max(longest[chosen]/n)))


def boundary(nodes, cells):
    faces=cells[:,:4][:,FACES].reshape(-1,3)
    _,inverse,counts=np.unique(np.sort(faces,axis=1),axis=0,return_inverse=True,return_counts=True)
    if np.any(counts>2): raise ValueError('Nonmanifold volume')
    exterior=faces[counts[inverse]==1]
    used=np.unique(exterior); mapping=np.full(len(nodes),-1); mapping[used]=np.arange(len(used))
    keys=np.sort(cells[:,:4][:,EDGES],axis=2).reshape(-1,2)
    unique,first=np.unique(keys,axis=0,return_index=True)
    middle=cells[:,4:].reshape(-1)[first]
    # Pair integer encoding is exact for the fixed bounded node count.
    encoded=unique[:,0]*len(nodes)+unique[:,1]
    exterior_edges=np.unique(np.sort(exterior[:,[[0,1],[1,2],[2,0]]],axis=2).reshape(-1,2),axis=0)
    lookup=np.searchsorted(encoded,exterior_edges[:,0]*len(nodes)+exterior_edges[:,1])
    if not np.array_equal(unique[lookup],exterior_edges): raise ValueError('Missing boundary midside')
    midsides=middle[lookup]
    if np.max(np.linalg.norm(nodes[midsides]-nodes[exterior_edges].mean(axis=1),axis=1))>1e-12:
        raise ValueError('Saved straight reference geometry changed')
    return nodes[used],mapping[exterior],used,midsides


def topology(vertices, faces):
    tri,areas,normals=triangle_geometry(vertices,faces)
    edges=np.sort(faces[:,[[0,1],[1,2],[2,0]]],axis=2).reshape(-1,2)
    _,inverse,counts=np.unique(edges,axis=0,return_inverse=True,return_counts=True)
    if not np.all(counts==2): raise ValueError('Closed two-face edges required')
    edge_faces=np.repeat(np.arange(len(faces)),3)[np.argsort(inverse,kind='stable')].reshape(-1,2)
    angle=np.degrees(np.arccos(np.clip(np.sum(normals[edge_faces[:,0]]*normals[edge_faces[:,1]],axis=1),-1,1)))
    maximum=np.zeros(len(faces)); np.maximum.at(maximum,edge_faces.ravel(),np.repeat(angle,2))
    return tri,areas,edge_faces,maximum


def summaries(vertices, faces, maxima, threshold=.002):
    tri,areas,pairs,normal_change=topology(vertices,faces)
    selected=np.asarray(maxima)>threshold
    linked=pairs[selected[pairs].all(axis=1)]
    graph=coo_matrix((np.ones(2*len(linked)),(np.r_[linked[:,0],linked[:,1]],np.r_[linked[:,1],linked[:,0]])),shape=(len(faces),len(faces))).tocsr()
    _,labels=connected_components(graph,directed=False)
    groups=[]
    for label in np.unique(labels[selected]):
        ids=np.flatnonzero(selected & (labels==label)); area=float(areas[ids].sum())
        groups.append(dict(first_triangle=int(ids.min()),triangles=len(ids),area_m2=area,
            maximum_sample_distance_m=float(maxima[ids].max()),
            centroid_m=np.average(tri[ids].mean(axis=1),axis=0,weights=areas[ids]).tolist(),
            bounds_m=[tri[ids].min(axis=(0,1)).tolist(),tri[ids].max(axis=(0,1)).tolist()]))
    groups.sort(key=lambda x:(-x['area_m2'],x['first_triangle']))
    order=np.argsort(maxima,kind='stable'); cumulative=np.cumsum(areas[order])/areas.sum()
    quantiles={str(p):float(maxima[order[min(len(order)-1,int(np.searchsorted(cumulative,p)))]] ) for p in (.5,.9,.95,.99,1.)}
    angle_stats={str(cut):dict(all_face_area_fraction=float(areas[normal_change>=cut].sum()/areas.sum()),
        witness_face_area_fraction=(float(areas[selected & (normal_change>=cut)].sum()/areas[selected].sum()) if selected.any() else None)) for cut in (30,60)}
    report=dict(triangles=len(faces),area_m2=float(areas.sum()),faces_with_sample_over_2mm=int(selected.sum()),
        area_of_faces_with_sample_over_2mm_m2=float(areas[selected].sum()),
        area_fraction_of_faces_with_sample_over_2mm=float(areas[selected].sum()/areas.sum()),
        area_weighted_face_maximum_quantiles_m=quantiles,connected_witness_components=len(groups),
        largest_10_components=groups[:10],edge_normal_change_association=angle_stats,
        interpretation='Area measures entire faces with a sampled witness, not exact area of the distance excursion. Edge-normal change is a discretization proxy, not smooth curvature or a saved Gmsh seam.')
    labels[~selected]=-1
    return report,normal_change,labels


class Budget:
    def __init__(self): self.started=time.monotonic(); self.queries=0
    def check(self, count=0):
        if time.monotonic()-self.started>CAPS['cooperative_seconds']: raise TimeoutError('Cooperative deadline')
        if self.queries+count>CAPS['maximum_distance_queries']: raise ValueError('Total distance-query cap')
        self.queries+=count


def directed(vertices,faces,distance,budget,cover=.001):
    maxima=np.zeros(len(faces)); total=0; bound=0.; worst=[]
    for points,chosen,n,cover_actual in indexed_samples(vertices,faces,cover,CAPS['maximum_distance_queries']):
        budget.check(len(points)); values=np.asarray(distance(points))
        if values.shape!=(len(points),) or not np.isfinite(values).all() or np.any(values<0): raise ValueError('Invalid distance')
        maxima[chosen]=values.reshape(len(chosen),n).max(axis=1)
        bound=max(bound,float(values.max())+cover_actual)
        best=np.lexsort((np.arange(len(values)),-values))[:10]
        worst.extend(dict(ordinal=total+int(i),triangle=int(chosen[i//n]),point_m=points[i].tolist(),distance_m=float(values[i])) for i in best)
        worst=sorted(worst,key=lambda row:(-row['distance_m'],row['ordinal']))[:10]
        total+=len(points)
    return dict(sample_count=total,maximum_sample_distance_m=float(maxima.max()),full_surface_upper_bound_m=bound,worst_10=worst),maxima


def witness_locator(vertices,faces):
    from vtkmodules.vtkCommonCore import vtkPoints, reference
    from vtkmodules.vtkCommonDataModel import vtkCellArray,vtkPolyData,vtkStaticCellLocator
    from vtkmodules.util.numpy_support import numpy_to_vtk,numpy_to_vtkIdTypeArray
    points=vtkPoints();points.SetData(numpy_to_vtk(np.asarray(vertices,float),deep=True))
    cells=vtkCellArray();cells.SetCells(len(faces),numpy_to_vtkIdTypeArray(np.c_[np.full(len(faces),3),faces].astype(np.int64).ravel(),deep=True))
    poly=vtkPolyData();poly.SetPoints(points);poly.SetPolys(cells)
    locator=vtkStaticCellLocator();locator.SetDataSet(poly);locator.BuildLocator()
    def nearest(point):
        closest=[0.,0.,0.]; cell=reference(0); sub=reference(0); distance2=reference(0.)
        locator.FindClosestPoint(point,closest,cell,sub,distance2)
        return dict(nearest_triangle=int(cell),nearest_point_m=closest,nearest_distance_m=float(distance2)**.5)
    return nearest


def specification(release_path,output):
    release=json.loads(Path(release_path).read_text()); config=json.loads((ROOT/MANIFEST).read_text())
    if release.get('schema')!=VERSION+'-release' or release.get('authorized') is not True: raise ValueError('Separate root release required')
    repository=Path(release['repository_directory']).resolve()
    if (repository==ROOT or not ROOT.is_relative_to(repository) or not Path(output).resolve().is_relative_to(repository)
            or not re.fullmatch('[0-9a-f]{40}',release.get('source_commit',''))
            or Path(release['source_directory']).resolve()!=ROOT or Path(release['attempt_directory']).resolve()!=Path(output).resolve()): raise ValueError('Immutable source/one attempt mismatch')
    if config['caps']!=CAPS or not release.get('root_release','').strip(): raise ValueError('Fixed caps/root release required')
    bound={str(Path(release_path).resolve()):sha(release_path)}
    for name in CLOSURE:
        raw=subprocess.run(['git','-C',str(repository),'show',release['source_commit']+':'+name],capture_output=True,check=True,timeout=5).stdout
        digest=hashlib.sha256(raw).hexdigest()
        if sha(ROOT/name)!=digest: raise ValueError('Committed source/archive mismatch')
        bound[str(ROOT/name)]=digest
    for row in config['inputs'].values():
        path=(repository/row['path']).resolve()
        if not path.is_relative_to(repository) or sha(path)!=row['sha256']: raise ValueError('Saved input identity changed')
        bound[str(path)]=row['sha256']
    if sha(ROOT/BASE_PATH)!=BASE_SHA: raise ValueError('Original numerical helper changed')
    return repository,config,bound


def configure_vtk_threads(api):
    if not api.SetBackend('Sequential') or api.GetBackend()!='Sequential':
        raise ValueError('Sequential VTK required')


def worker(release_path,output):
    output=Path(output); report_path=output/'report.json'
    if report_path.exists(): raise FileExistsError('Existing diagnostic preserved')
    budget=Budget();bound={};report=dict(status='running',mesher_calls=0,solver_calls=0,image_or_landmark_reads=0,candidate_accepted=False,seam_attribution_available=False)
    try:
        repository,config,bound=specification(release_path,output)
        resource.setrlimit(resource.RLIMIT_FSIZE,(4*1024**2,4*1024**2))
        for name,version in config['packages'].items():
            if importlib.metadata.version(name)!=version: raise ValueError('Package changed')
        from vtkmodules.vtkCommonCore import vtkSMPTools
        configure_vtk_threads(vtkSMPTools)
        base=load(ROOT/BASE_PATH,'diagnostic_original_mesh_helpers')
        inputs={key:repository/row['path'] for key,row in config['inputs'].items()}
        with np.load(inputs['native_surface'],allow_pickle=False) as saved:
            if set(saved.files)!={'vertices_m','triangles'}: raise ValueError('Unexpected saved surface payload')
            X=saved['vertices_m'];F=saved['triangles']
        arrays={name:np.load(inputs[name],allow_pickle=False) for name in ('nodes_m','tet10_indices','gmsh_node_ids','gmsh_element_ids')}
        if X.shape!=(91951,3) or F.shape!=(183902,3) or arrays['nodes_m'].shape!=(5223,3) or arrays['tet10_indices'].shape!=(2761,10): raise ValueError('Exact saved shapes required')
        nodes,cells=arrays['nodes_m'],arrays['tet10_indices'];Y,G,used,middle=boundary(nodes,cells)
        if (len(Y),len(G))!=(732,1464): raise ValueError('Saved boundary identity changed')
        original=json.loads(inputs['candidate_result'].read_text())
        report.update(source_boundary=dict(vertices=len(X),triangles=len(F)),candidate_boundary=dict(vertices=len(Y),triangles=len(G)),saved_volume_relative_error=original['relative_volume_error'],source_import_path=base.__file__)
        distances={'source_to_mesh':base.vtk_distance_function(Y,G),'mesh_to_source':base.vtk_distance_function(X,F)}
        payload={}
        for name,A,T,B,U in [('source_to_mesh',X,F,Y,G),('mesh_to_source',Y,G,X,F)]:
            result,maxima=directed(A,T,distances[name],budget)
            expected=original[name]
            if result['sample_count']!=expected['sample_count'] or any(abs(result[k]-expected[k])>1e-12 for k in ('maximum_sample_distance_m','full_surface_upper_bound_m')): raise ValueError('Original sampled fidelity replay mismatch')
            summary,angle,labels=summaries(A,T,maxima);result['summary']=summary
            nearest=witness_locator(B,U)
            for row in result['worst_10']:
                budget.check(1); row.update(nearest(row['point_m']))
                if abs(row['nearest_distance_m']-row['distance_m'])>1e-10: raise ValueError('Witness locator disagreement')
            report[name]=result
            payload[name+'_maximum_m']=maxima;payload[name+'_normal_change_degrees']=angle;payload[name+'_component']=labels
            write(report_path,report)
        probes={'boundary_vertices':nodes[used],'boundary_midsides':nodes[middle],'boundary_face_centroids':Y[G].mean(axis=1)}
        report['mesh_to_source_probe_classes']={}
        for name,points in probes.items():
            budget.check(len(points));values=distances['mesh_to_source'](points)
            report['mesh_to_source_probe_classes'][name]=dict(count=len(points),maximum_m=float(values.max()),quantiles_m=np.quantile(values,[.5,.9,.95,.99,1]).tolist(),over_2mm=int((values>.002).sum()))
        np.savez_compressed(output/'per-triangle-diagnostics.npz',**payload)
        budget.check();report['status']='completed_saved_geometry_diagnostic_only'
    except BaseException as error: report.update(status='failed_or_incomplete',error=f'{type(error).__name__}: {error}')
    finally:
        report.update(query_count=budget.queries,elapsed_seconds=time.monotonic()-budget.started,input_hashes=bound,inputs_unchanged={p:Path(p).is_file() and sha(p)==digest for p,digest in bound.items()})
        if not bound or not all(report['inputs_unchanged'].values()):report['status']='failed_or_incomplete'
        write(report_path,report)
    return 0 if report['status']=='completed_saved_geometry_diagnostic_only' else 1


def run(release_path,output):
    output=Path(output).resolve();_,_,bound=specification(release_path,output)
    output.mkdir(parents=True,exist_ok=False)
    runtime=load(ROOT/'scripts/febio_runtime.py','diagnostic_supervisor')
    guard=load(ROOT/'scripts/mechanics_patient_mesh_candidate_run.py','diagnostic_output_guard')
    environment=runtime.private_environment(runtime.declaration());environment['PYTHONDONTWRITEBYTECODE']='1'
    receipt=guard.supervised(runtime,[sys.executable,'-B',str(Path(__file__).resolve()),'worker','--release',str(Path(release_path).resolve()),'--output',str(output)],output,environment,{'aggregate_seconds':CAPS['parent_seconds'],'process_group_rss_bytes':CAPS['rss_bytes']})
    supervision,disk=receipt
    report=json.loads((output/'report.json').read_text()) if (output/'report.json').is_file() else {}
    passed=supervision['status']=='completed' and supervision['exit_code']==0 and disk['error'] is None and report.get('status')=='completed_saved_geometry_diagnostic_only' and all(Path(p).is_file() and sha(p)==d for p,d in bound.items())
    acceptance=dict(status='completed' if passed else 'failed_or_incomplete',supervision=supervision,output_guard=disk,solver_calls=0,mesher_calls=0,clinical_or_candidate_acceptance=False)
    write(output/'acceptance.json',acceptance)
    try: guard.output_usage(output)
    except BaseException as error:
        passed=False;acceptance.update(status='failed_or_incomplete',final_output_error=str(error));write(output/'acceptance.json',acceptance)
    return 0 if passed else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['run','worker']);parser.add_argument('--release',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();raise SystemExit((run if args.mode=='run' else worker)(args.release,args.output))
