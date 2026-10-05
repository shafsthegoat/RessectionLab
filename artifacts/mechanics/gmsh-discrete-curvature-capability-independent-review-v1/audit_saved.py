"""Saved analytical records/arrays only; no mesher, VTK, patient or solver calls."""
from pathlib import Path
import datetime
import hashlib
import json
import math
import tarfile

import numpy as np

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]
OWNER=ROOT/'artifacts/mechanics/gmsh-discrete-curvature-capability-v1'
RAW=ROOT/'outputs/mechanics/gmsh-discrete-curvature-capability-v1'

def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def read(path):return json.loads(Path(path).read_text())

index=read(OWNER/'artifact-index.json');raw_index=read(OWNER/'raw-output-index.json')
result=read(RAW/'result.json');sup=read(RAW/'supervision/supervision.json');accept=read(RAW/'acceptance.json')
decl=read(OWNER/'declaration.json');proof=read(OWNER/'official-source-inspection.json')
assert sha(OWNER/'artifact-index.json')=='ab6d3b396f8750eca8ea57c16eb24769ceba44b2b922d00ff778b17eae0e4fdb'
for folder,items in [(OWNER,index['files']),(RAW,raw_index['files'])]:
    for name,entry in items.items():
        assert sha(folder/name)==entry['sha256'] and (folder/name).stat().st_size==entry['bytes']
for name in ['acceptance.json','result.json','supervision/combined.log','supervision/supervision.json']:
    assert sha(RAW/name)==sha(OWNER/'saved-records'/name)
inputs={name:sha(name)==expected for name,expected in result['input_hashes'].items()}
assert len(inputs)==13 and all(inputs.values()) and all(result['inputs_unchanged'].values())
archive=ROOT/'build/validation/gmsh-discrete-curvature-capability-v1/gmsh-4.15.2-source.tgz'
assert sha(archive)==proof['archive_sha256'] and archive.stat().st_size==proof['archive_bytes']
selected={}
with tarfile.open(archive,'r:gz') as stream:
    for name,entry in proof['selected_files'].items():
        # Exact archive path, not a suffix that can match nested CMake files.
        member=stream.getmember('gmsh-4.15.2-source/'+name)
        assert member.isfile()
        content=stream.extractfile(member).read()
        local=ROOT/'build/validation/gmsh-discrete-curvature-capability-v1/source'/name
        assert hashlib.sha256(content).hexdigest()==sha(local)==entry['sha256']
        assert len(content)==entry['bytes']
        selected[name]=entry['sha256']
assert sup==accept['supervision'] and sup['status']=='completed' and sup['exit_code']==0
assert sup['elapsed_seconds']<30 and sup['sampled_peak_process_group_rss_bytes']<1024**3 and sup['kill_reason'] is None
assert result['native_generations']==2 and result['solver_calls']==0 and not result['patient_input'] and not result['training']
assert (RAW/'supervision/combined.log').read_text().count('Info    : Meshing 3D...')==2
assert [row['case'] for row in result['cases']]==['curvature_off','curvature_on']
assert result['status']=='completed_one_analytical_pair'
assert sum((RAW/name).stat().st_size for name in raw_index['files'])==145510
axes=np.array(decl['ellipsoid_axes_m']);faces_local=np.array([[1,2,3],[0,3,2],[0,1,3],[0,2,1]])
edges=np.array([[0,1],[1,2],[2,0],[0,3],[1,3],[2,3]])
with np.load(RAW/'analytical-input.npz',allow_pickle=False) as data:V=data['vertices_m'];F=data['triangles']
assert V.shape==(1986,3) and F.shape==(3968,3)
equation_error=float(np.max(np.abs(np.sum((V/axes)**2,axis=1)-1)));assert equation_error<1e-12
tri=V[F];source_volume=float(np.einsum('ij,ij->i',tri[:,0],np.cross(tri[:,1],tri[:,2])).sum()/6)
assert np.isclose(source_volume,result['source']['enclosed_volume_m3'],rtol=1e-14,atol=1e-20)
analytic_volume=4*math.pi*float(axes.prod())/3
assert np.isclose(analytic_volume,result['analytical_volume_m3'],rtol=1e-14)

def sample_count(X,G):
    T=X[G];longest=np.max(np.linalg.norm(T-np.roll(T,1,axis=1),axis=2),axis=1)
    divisions=np.maximum(1,np.ceil(longest/.0005).astype(int))
    return int(np.sum((divisions+1)*(divisions+2)//2))

assert sample_count(V,F)==120056
numeric=[]
for row in result['cases']:
    with np.load(RAW/(row['case']+'.npz'),allow_pickle=False) as data:X=data['nodes_m'];E=data['tet10_indices']
    assert X.shape==(row['nodes'],3) and E.shape==(row['elements'],10)
    assert np.isfinite(X).all() and E.dtype==np.dtype('int64') and E.min()>=0 and E.max()<len(X)
    corners=X[E[:,:4]];det=np.linalg.det((corners[:,1:]-corners[:,:1]).transpose(0,2,1))
    midpoint=float(np.max(np.linalg.norm(X[E[:,4:]]-corners[:,edges].mean(axis=2),axis=2)))
    assert det.min()>0 and midpoint==0
    volume=float(det.sum()/6);all_faces=E[:,:4][:,faces_local].reshape(-1,3)
    _,inverse,counts=np.unique(np.sort(all_faces,axis=1),axis=0,return_inverse=True,return_counts=True)
    assert counts.max()==2
    boundary=all_faces[counts[inverse]==1];boundary_nodes=np.unique(boundary)
    boundary_edges=np.unique(np.sort(boundary[:,[[0,1],[1,2],[2,0]]],axis=2).reshape(-1,2),axis=0)
    lengths=np.linalg.norm(X[boundary_edges[:,0]]-X[boundary_edges[:,1]],axis=1)
    quantiles=np.quantile(lengths,[0,.1,.5,.9,1])
    assert len(boundary_nodes)==row['quality']['boundary']['vertices'] and len(boundary)==row['quality']['boundary']['triangles']
    assert len(boundary_nodes)-len(boundary_edges)+len(boundary)==2
    np.testing.assert_allclose(quantiles,row['boundary_edge_quantiles_m'],rtol=1e-13,atol=1e-15)
    assert np.isclose(volume,row['quality']['volume_m3'],rtol=1e-13)
    assert np.isclose(det.min(),row['quality']['minimum_reference_determinant_m3'],rtol=1e-13)
    quality=12*(det/2)**(2/3)/np.sum((corners[:,edges[:,1]]-corners[:,edges[:,0]])**2,axis=(1,2))
    assert np.isclose(quality.min(),row['quality']['minimum_mean_ratio'],rtol=1e-13)
    assert sample_count(X,boundary)==row['mesh_to_source']['sample_count']
    for direction in ['source_to_mesh','mesh_to_source']:
        metric=row[direction]
        assert metric['sample_count']<=300000
        assert metric['maximum_sample_distance_m']<=metric['full_surface_upper_bound_m']<=metric['maximum_sample_distance_m']+.0005+1e-12
    numeric.append({'case':row['case'],'nodes':len(X),'tet10_elements':len(E),
        'volume_m3_recomputed':volume,'minimum_reference_determinant_m3_recomputed':float(det.min()),
        'maximum_midpoint_error_m_recomputed':midpoint,'minimum_mean_ratio_recomputed':float(quality.min()),
        'boundary_euler_recomputed':2,'boundary_edge_quantiles_m_recomputed':quantiles.tolist(),
        'source_direction_sample_count_recomputed':sample_count(V,F),'reverse_direction_sample_count_recomputed':sample_count(X,boundary)})

verification={'schema':'gmsh-discrete-curvature-capability-independent-review-v1',
    'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'passed_saved_source_and_analytical_array_review',
    'owner_index_sha256':sha(OWNER/'artifact-index.json'),'script_sha256':sha(ROOT/'scripts/mechanics_gmsh_curvature_control.py'),
    'declaration_sha256':sha(OWNER/'declaration.json'),'review_script_sha256':sha(__file__),
    'verified_owner_records':len(index['files']),'verified_raw_files':len(raw_index['files']),'verified_raw_bytes':145510,
    'source_runtime_input_checks':inputs,'official_source_archive_sha256':sha(archive),'archive_selected_file_matches':selected,
    'source_checks':{'vertices':len(V),'triangles':len(F),'maximum_ellipsoid_equation_error':equation_error,
        'polyhedral_volume_m3_recomputed':source_volume,'analytic_volume_m3_recomputed':analytic_volume,
        'relative_polyhedral_volume_error_recomputed':abs(source_volume/analytic_volume-1)},
    'analytical_saved_array_checks':numeric,
    'resources':{'supervised_seconds':sup['elapsed_seconds'],'sampled_group_peak_rss_bytes':sup['sampled_peak_process_group_rss_bytes'],
        'wall_cap_seconds':30,'rss_cap_bytes':1073741824,'sampled_output_cap_bytes':16777216,'raw_output_bytes':145510,
        'exact_generation_count_from_receipt_and_log':2,'numeric_threads_requested':1,'VTK_sequential_required':True,'retries':0},
    'interpretation':'Joint curvature0→24 and minimum-size12→3mm profile effect on one analytical input; no isolated causal claim, patient extrapolation or solver validation.',
    'limitations':['Saved nearest-triangle distances were source-audited and hash-bound, not recomputed with VTK during this review.',
        '64 curvature samples per profile demonstrate finite/nonzero availability, not full-surface coverage or curvature-estimate accuracy.',
        'Fidelity refers to input facets, not exact smooth-ellipsoid Hausdorff distance.',
        'Source pinning is not a reproducible-build equivalence proof.',
        'RSS/output are sampled; timing does not predict patient or solver cost.'],
    'reviewer_control_correction':'Initial suffix-only archive locator matched nested CMakeLists files; exact archive prefix/path fixes the reviewer assertion. Initial record preserved.',
    'review_activity':{'patient_arrays_read':0,'saved_analytical_array_files_read':3,'Gmsh_or_VTK_imports':0,
        'mesh_generations':0,'solver_calls':0,'training_updates':0,'source_edits':0}}
(OUT/'verification.json').write_text(json.dumps(verification,indent=2,allow_nan=False)+'\n')
print(json.dumps({'status':verification['status'],'verified_owner_records':len(index['files']),
    'verified_raw_files':len(raw_index['files']),'verified_inputs':len(inputs),'official_archive_selected_files':len(selected),
    'verification_sha256':sha(OUT/'verification.json'),'recomputed_case_counts':[(row['nodes'],row['tet10_elements']) for row in numeric]},indent=2))
