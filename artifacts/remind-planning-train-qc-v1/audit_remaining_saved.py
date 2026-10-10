"""Saved JSON and native-array streaming only; no original DICOM access."""
import hashlib
import itertools
import json
import math
from contextlib import ExitStack
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
BASE = ROOT/'build/remind-planning-qc-v1/remaining-train-v1'
PINS = {'headers':'77d91566a48d2b47c3ba59d5a7d05b51a0709022ad225cbe4d9410773c6312b1', 'crop':'2bb3cefce51de48345570a361c4d479ac6d51328251ae7910d7ba87ed70effdb'}

def load(path):
    raw = path.read_bytes()
    return json.loads(raw), {'path':str(path.relative_to(ROOT)), 'bytes':len(raw), 'sha256':hashlib.sha256(raw).hexdigest()}

def corners(shape):
    return np.array([(*v,1) for v in itertools.product(*[(-.5,n-.5) for n in shape])])

def fit_planes(g):
    x,y = np.asarray(g['raw_orientation'],float).reshape(2,3)
    normal = np.cross(x,y); normal /= np.linalg.norm(normal)
    positions = np.asarray(g['raw_positions'],float)
    order = sorted(range(len(positions)),key=lambda i:float(positions[i]@normal))
    assert order == g['sorted_source_indices']
    p = positions[order]; index = np.arange(len(p))-(len(p)-1)/2
    center = p.mean(axis=0); slope = (index[:,None]*(p-center)).sum(axis=0)/(index**2).sum()
    origin = center-((len(p)-1)/2)*slope
    sp = np.asarray(g['raw_pixel_spacing'],float)
    affine = np.eye(4); affine[:3,:] = np.c_[x*sp[1],y*sp[0],slope,origin]; affine[:2,:] *= -1
    error = float(np.max(np.abs(affine-np.asarray(g['affine_xyz_to_ras_mm']))))
    residual = float(np.linalg.norm(p-origin-np.arange(len(p))[:,None]*slope,axis=1).max())
    assert error < 1e-9 and abs(residual-g['slice_position_max_residual_mm']) < 1e-9
    return {'affine_max_abs_difference':error,'max_position_residual_mm':residual}

cohort, cohort_pin = load(ROOT/'manifests/experiments/remind-component-cohort-v1.json')
assert cohort_pin['sha256'] == '326b4ebb4a6e439e47fb166d8fcfeec5ff65798294820d5aac0ed240b21fdd05'
cases = []
for tag in ('010','020','025'):
    patient = 'ReMIND-'+tag; folder = BASE/(patient+'-crop-mr')
    r,rpin = load(folder/'conversion-result.json'); h,hpin = load(BASE/(patient+'-headers')/'result.json')
    s,spin = load(BASE/(patient+'-crop-supervisor.json')); hs,hspin = load(BASE/(patient+'-headers-supervisor.json'))
    case,cpin = load(ROOT/'build/real-cohort-eligibility-v1'/(patient+'-case.json'))
    assert r['patient_id'] == h['patient_id'] == case['patient_id'] == patient
    assert r['case_sha256'] == h['case_sha256'] == cpin['sha256']
    assert r['header_snapshot_sha256'] == hpin['sha256']
    assert r['executing_script_sha256'] == PINS['crop'] and h['executing_script_sha256'] == PINS['headers']
    assert r['role'] == h['role'] == case['role'] == 'TRAIN'
    member = [v for v in cohort['members'] if v['subject'] == patient]
    assert len(member) == 1 and member[0]['role'] == 'TRAIN' and member[0]['patient_group'] == case['patient_group']
    assert h['status'] == 'header_geometry_and_ancestry_projected' and h['pixel_decode_calls'] == 0
    assert r['status'] == 'public_crop_source_samples_and_label_grids_converted_anatomy_unreviewed' and r['labels_resampled']
    for sup in (s,hs):
        assert sup['exit_code'] == 0 and sup['child_reaped'] and sup['stop_reason'] is None
    rows = {v['kind']:v for v in h['series']}; source = {v['kind']:v for v in case['series']}
    geometry_checks = {}
    for kind,row in rows.items():
        assert row['source_objects'] == source[kind]['objects']
        assert row['series_instance_uid'] == source[kind]['SeriesInstanceUID']
        geometry_checks[kind] = fit_planes(row['geometry'])
    assert sum(len(v['objects']) for v in source.values()) == h['objects_verified']
    assert sum(o['bytes'] for v in source.values() for o in v['objects']) == h['source_file_returned_bytes']
    shape = tuple(r['shape_xyz']); start = np.array(r['public_crop_start_MR'])
    mr = np.array(rows['structural_t1ce']['geometry']['affine_xyz_to_ras_mm']); native = np.array(r['MR_native_crop_affine_ras_mm']); derived = np.array(r['planning_derived_affine_ras_mm'])
    expected = mr.copy(); expected[:3,3] = (mr@np.r_[start,1])[:3]
    assert np.array_equal(native,expected)
    diff = float(np.linalg.norm((corners(shape)@(native-derived).T)[:,:3],axis=1).max())
    assert diff <= .001 and abs(diff-r['reindex_policy']['maximum_crop_world_difference_mm']) < 1e-12
    spacing = np.linalg.norm(native[:3,:3],axis=0); directions = derived[:3,:3]/spacing
    assert np.max(np.abs(directions.T@directions-np.eye(3))) < 1e-12
    public = np.array(rows['cerebrum']['geometry']['affine_xyz_to_ras_mm'])
    public_corners = np.linalg.solve(public,(corners(shape)@derived.T).T).T[:,:3]
    assert np.all(public_corners >= -.5) and np.all(public_corners < np.asarray(rows['cerebrum']['geometry']['shape_xyz'])-.5)
    selected = rows['structural_t1ce']['geometry']['sorted_source_indices'][int(start[2]):int(start[2])+shape[2]]
    expected_bytes = sum(source['structural_t1ce']['objects'][i]['bytes'] for i in selected)+sum(source[k]['objects'][0]['bytes'] for k in ('cerebrum','whole_tumor','ventricles'))
    assert r['objects_verified'] == len(selected)+3 and r['source_file_returned_bytes'] == expected_bytes
    names = {'MR':'MR_native_crop.npy'}
    for kind in ('cerebrum','whole_tumor','ventricles'):
        names[kind] = kind+'_source_label.npy'; names[kind+'_domain'] = kind+'_source_grid_domain.npy'
    transforms = {}
    for kind in ('cerebrum','whole_tumor','ventricles'):
        a = r['annotations'][kind]; g = rows[kind]['geometry']
        assert a['source_native_affine_ras_mm'] == g['affine_xyz_to_ras_mm'] and a['source_native_shape'] == g['shape_xyz']
        assert a['ancestry'] == rows[kind]['ancestry'] and a['correspondence'] == rows[kind]['alignment']
        transforms[kind] = np.linalg.solve(np.array(g['affine_xyz_to_ras_mm']),derived)
        assert np.max(np.abs(transforms[kind]-np.array(a['placement']['target_index_to_source_index']))) < 1e-10
    counts = {k:0 for k in names if k != 'MR'}; overlap = {'target_outside_support':0,'target_ventricle_overlap':0,'ventricle_in_support_zero':0}
    bboxes = {k:[list(shape),[-1,-1,-1]] for k in ('cerebrum','whole_tumor','ventricles')}
    mr_min,mr_max = math.inf,-math.inf; checked = {}
    yy,zz = np.indices(shape[1:],dtype=float)
    with ExitStack() as stack:
        streams = {k:stack.enter_context((folder/n).open('rb',buffering=0)) for k,n in names.items()}
        hashes = {k:hashlib.sha256() for k in names}; dtypes = {}
        for k,f in streams.items():
            assert np.lib.format.read_magic(f) == (1,0)
            dims,fortran,dtype = np.lib.format.read_array_header_1_0(f); assert dims == shape and not fortran
            assert dtype == np.dtype('<f4' if k == 'MR' else 'u1'); dtypes[k] = dtype
            offset = f.tell(); f.seek(0); hashes[k].update(f.read(offset))
            assert (folder/names[k]).stat().st_size == offset+math.prod(shape)*dtype.itemsize
        for x in range(shape[0]):
            arrays = {}
            for k,f in streams.items():
                size = shape[1]*shape[2]*dtypes[k].itemsize
                raw = f.read(size); assert len(raw) == size; hashes[k].update(raw)
                arrays[k] = np.frombuffer(raw,dtype=dtypes[k]).reshape(shape[1:])
            image = arrays['MR']; assert np.isfinite(image).all()
            mr_min = min(mr_min,float(image.min())); mr_max = max(mr_max,float(image.max()))
            for k in counts:
                assert arrays[k].max() <= 1; counts[k] += int(arrays[k].sum())
            for kind,mapping in transforms.items():
                point = mapping[:3,0,None,None]*x+mapping[:3,1,None,None]*yy+mapping[:3,2,None,None]*zz+mapping[:3,3,None,None]
                valid = np.all((point >= -.5)&(point < np.asarray(r['annotations'][kind]['source_native_shape'])[:,None,None]-.5),axis=0)
                assert np.array_equal(arrays[kind+'_domain'],valid)
                assert not np.any(arrays[kind] & ~valid)
                if kind == 'cerebrum': assert valid.all()
                ys,zs = np.nonzero(arrays[kind])
                if len(ys):
                    lo,hi = bboxes[kind]
                    bboxes[kind] = [[min(lo[0],x),min(lo[1],int(ys.min())),min(lo[2],int(zs.min()))],[max(hi[0],x),max(hi[1],int(ys.max())),max(hi[2],int(zs.max()))]]
            c,t,v = (arrays[k] for k in ('cerebrum','whole_tumor','ventricles'))
            overlap['target_outside_support'] += int(np.count_nonzero(t & (1-c)))
            overlap['target_ventricle_overlap'] += int(np.count_nonzero(t & v))
            overlap['ventricle_in_support_zero'] += int(np.count_nonzero(v & (1-c)))
        for k,f in streams.items():
            assert f.read(1) == b''
            actual = {'bytes':f.tell(),'sha256':hashes[k].hexdigest()}; assert actual == r['artifacts'][names[k]]; checked[names[k]] = actual
    for name,expected in r['artifacts'].items():
        if name in checked: continue
        raw = (folder/name).read_bytes(); actual = {'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}; assert actual == expected; checked[name] = actual
    assert sum(v['bytes'] for v in checked.values()) == r['output_bytes_before_receipt']
    assert overlap['target_outside_support'] == r['public_target_support_relation']['whole_tumor_positive_outside_public_support']
    assert overlap['target_ventricle_overlap'] == r['evaluation_only_target_ventricle_relation']['overlap_positive_voxels']
    assert overlap['ventricle_in_support_zero'] == r['public_support_private_ventricle_relation']['ventricle_positive_in_public_support_zero']
    voxel_mm3 = abs(float(np.linalg.det(derived[:3,:3]))); volumes = {}
    for kind in ('cerebrum','whole_tumor','ventricles'):
        a = r['annotations'][kind]; p = a['placement']
        assert counts[kind] == a['placed_positive_voxels'] == p['resampled_positive_voxels']
        assert counts[kind+'_domain'] == a['placed_source_domain_voxels'] == p['resampled_source_domain_voxels']
        source_voxel_mm3 = abs(float(np.linalg.det(np.array(a['source_native_affine_ras_mm'])[:3,:3])))
        original_volume = a['source_positive_voxels']*source_voxel_mm3; placed_volume = counts[kind]*voxel_mm3
        assert abs(original_volume-p['source_positive_volume_mm3']) < 1e-8 and abs(placed_volume-p['resampled_positive_volume_mm3']) < 1e-8
        outside = p['source_positive_centres_outside_target_grid']; assert 0 <= outside <= a['source_positive_voxels']
        volumes[kind] = {'source_positive_voxels_from_bound_execution':a['source_positive_voxels'],'source_positive_mm3_from_bound_execution':original_volume,'saved_resampled_positive_voxels_verified':counts[kind],'saved_resampled_positive_mm3_verified':placed_volume,'volume_change_percent':100*(placed_volume/original_volume-1),'source_positive_centres_outside_crop_from_bound_execution':outside,'outside_crop_fraction_of_source_positives':outside/a['source_positive_voxels'],'source_centre_cropping_mm3':outside*source_voxel_mm3,'resampled_positive_bbox_inclusive':bboxes[kind],'saved_source_domain_voxels_verified':counts[kind+'_domain']}
    zeros = math.prod(shape)-counts['cerebrum']; assert zeros == r['public_support_private_ventricle_relation']['public_support_zero_voxels']
    overlap.update(target_total=counts['whole_tumor'],target_outside_support_fraction=overlap['target_outside_support']/counts['whole_tumor'],ventricle_total=counts['ventricles'],ventricle_in_support_zero_fraction=overlap['ventricle_in_support_zero']/counts['ventricles'],public_support_zero_total=zeros,ventricle_fraction_of_public_zeros=overlap['ventricle_in_support_zero']/zeros)
    cases.append({'patient_id':patient,'role':'TRAIN','inputs':[rpin,hpin,spin,hspin,cpin],'verified_artifacts':checked,'shape_xyz':list(shape),'native_MR_spacing_mm':spacing.tolist(),'source_read_accounting':{'objects':len(selected)+3,'bytes':expected_bytes},'header_refit':geometry_checks,'maximum_derived_corner_difference_mm':diff,'public_source_domain_complete':True,'MR_finite_minmax':[mr_min,mr_max],'volumes':volumes,'conflicts':overlap,'runtime':{'seconds':s['elapsed_seconds'],'sampled_peak_RSS_bytes':s['sampled_peak_group_rss_bytes'],'worker_reported_peak_RSS_bytes':r['peak_RSS_bytes'],'child_reaped':True}})

report = {'decision':'PASS_SAVED_GEOMETRY_AND_ARRAY_INTEGRITY_FOR_EXPLICIT_CROPPED_ANNOTATION_TASK','patients':cases,'cohort_pin':cohort_pin,'source_pins':PINS,'original_DICOM_opens':0,'source_pixel_redecode_calls':0,'full_volume_allocations':0,'verification':'Every saved array byte hashed; values streamed one x-plane at a time. MR finiteness and binary masks checked. All domain voxels independently solved from saved source/target world transforms; public domain is complete. Header affines refitted independently from saved raw plane fields.','limits':['Native-positive counts and outside-crop positive-centre counts are inherited from the exact bound executed converter; they cannot be independently recounted without original native label pixels, which were not reopened. Their physical volumes and saved resampled counts were independently verified.','No clinical anatomical accuracy or independent registration claim. Root reviews the saved overlays; this audit verifies their hashes only.','All three whole-tumor source-positive-centre crop-loss counts are zero, and whole-tumor volume changes are small; this does not prove NN preserves every thin label or topology.','Public-support and ventricular reference crop/resampling losses must remain in denominators. Endpoint claims apply to the declared crop, not complete source anatomy.','Source manual whole-tumor annotations are research task targets only by explicit choice; automatic Brainlab support and ventricles remain estimates.','Support-only removable tissue makes complete target removal infeasible in all three cases. Preserve target denominators and report infeasibility unless a different removable-domain condition is explicit. Do not silently clip targets or fill support using private references.','Original public support is informative about ventricular geometry; keep its zero cues visible and declared. Source-label overlap makes simultaneous perfect target-removal and zero ventricular-reference contact infeasible.','No roles or original arrays changed. These three TRAIN cases do not themselves establish cross-patient generalization or clinical safety.']}
p=OUT/'remaining-train-combined-review.json';p.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
print(json.dumps({'report_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'cases':[{'patient':v['patient_id'],'volumes':v['volumes'],'conflicts':v['conflicts']} for v in cases]},indent=2))
