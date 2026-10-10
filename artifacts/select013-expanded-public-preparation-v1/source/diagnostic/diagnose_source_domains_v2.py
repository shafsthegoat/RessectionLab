"""Unrun SELECT013 diagnostic: two public SEG objects, no MRI pixels or planner.

Use only after root release under the existing bounded owner. This does not
write a planning manifest, replace access, or change any original result.
"""
from pathlib import Path
import argparse
import gc
import hashlib
import io
import json
import resource
import time
import types

ROOT = Path(__file__).resolve().parents[2]
PINS = {
    'qc': ('src/resectionlab/remind_planning_qc.py', 'f35a28f2cdedd777e15031b5cba370c4106c0436eefe9590ca17466d8944b3ca'),
    'core': ('scripts/convert_remind_development.py', '34e6ceb9560344a88226489b2dd8ca6bfb90cb87c52d2ff3184d393293df4ef6'),
    'case': ('build/remind-select-public-bindings-v1/ReMIND-013-case.json', '9d4448328e979c856e3c8e08599b9cc2b66fc7d324f1652d7970468099267c65'),
    'headers': ('build/remind-select-public-preparation-v1/actual-public-qc-v1/ReMIND-013-headers/result.json', 'fd3a5c5cfded6e4a60ef263c03f13df0adb07ae2f77b84f61aee9295e31b8948'),
    'old_crop': ('build/remind-select-public-preparation-v1/actual-public-qc-v1/ReMIND-013-crop-mr/conversion-result.json', '9d126eaa926d7e6c4e11903d119a9446f1e38ea98993a652735cabd72ecd0fe9'),
    'old_seed': ('build/select013-empty-seed-diagnostic-v1/result.json', 'e50085f9f9ecd4aeac258bc83701c9f4e695cd36a981b247b3f59bed791c4c4f'),
}
EPS = 1e-8


def read_bound(key):
    name, expected = PINS[key]
    raw = (ROOT/name).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError('changed_input:'+key)
    return raw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute-two-public-segs', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not args.execute_two_public_segs:
        parser.error('Prepared only; root must release the two exact public SEG reads')
    if args.output.exists() or not args.output.parent.is_dir():
        raise ValueError('new_output_in_existing_owned_directory_required')
    import numpy as np
    started = time.monotonic()
    def guard():
        if time.monotonic()-started > 60:
            raise TimeoutError('diagnostic_60_second_limit')
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > 1536*1024**2:
            raise MemoryError('diagnostic_1536_MiB_limit')
    qc = types.ModuleType('exact_remind_qc'); qc.__file__ = str(ROOT/PINS['qc'][0])
    exec(compile(read_bound('qc'), qc.__file__, 'exec', dont_inherit=True), qc.__dict__)
    core = qc.source_module(ROOT/PINS['core'][0], PINS['core'][1])
    case, headers, old, failure = (json.loads(read_bound(k)) for k in ('case','headers','old_crop','old_seed'))
    qc.need(case['patient_id'] == 'ReMIND-013' and case['role'] == 'SELECT', 'exact_SELECT013')
    qc.validate_case(case, ROOT, public_only=True)
    by_kind = qc.validate_header_binding(headers, case, PINS['case'][1], public_only=True)
    series = {r['kind']: r for r in case['series']}
    mr = by_kind['structural_t1ce']['geometry']
    start, shape, native_affine, affine, crop_map = qc.mr_sampling_domain_union_crop(
        mr, by_kind['cerebrum']['geometry'], by_kind['whole_tumor']['geometry'])
    qc.need(start.tolist() == [61,70,23] and list(shape) == [131,154,117], 'predicted_metadata_grid')
    old_start = np.asarray(old['public_crop_start_MR'], int)
    old_shape = np.asarray(old['shape_xyz'], int)
    offset = old_start-start
    overlap = tuple(slice(int(k), int(k+n)) for k,n in zip(offset,old_shape))
    accounting = {'objects_verified':0,'source_file_returned_bytes':0}
    masks, domains, placements, overlap_hashes = {}, {}, {}, {}
    for kind, old_key in (('cerebrum','supplied_support'),('whole_tumor','supplied_whole_tumor')):
        guard()
        s, h = series[kind], by_kind[kind]
        qc.need(len(s['objects']) == 1 and s['Modality'] == 'SEG', 'only_two_public_SEG_objects')
        ds = qc.read_object(s['objects'][0], pixels=True, accounting=accounting)
        qc.identity(ds, case, s)
        qc.need(str(ds.SOPInstanceUID) == h['sop_instance_uids'][0], 'saved_SOP')
        qc.need(h['geometry']['frame_of_reference_uid'] == mr['frame_of_reference_uid'], 'same_saved_frame')
        ds.pixel_array_options(index=None, raw=True, decoding_plugin='', use_v2_backend=False)
        source, source_affine, _ = core.convert_seg(ds)
        expected = qc.raw_seg_samples(ds)[h['geometry']['sorted_source_indices']].transpose(2,1,0)
        qc.need(np.array_equal(source,expected), 'independent_SEG_source_samples')
        qc.need(qc.corner_error(source_affine,h['geometry']['affine_xyz_to_ras_mm'],source.shape)<1e-9,'saved_SEG_geometry')
        values, domain, placement = qc.resample_binary_nn(source,source_affine,affine,shape,guard)
        qc.need(placement['source_positive_centres_outside_target_grid']==0,'source_positive_crop_loss')
        buf = io.BytesIO(); np.save(buf,np.ascontiguousarray(values[overlap]),allow_pickle=False)
        overlap_hashes[kind] = hashlib.sha256(buf.getvalue()).hexdigest()
        qc.need(overlap_hashes[kind] == failure['inputs'][old_key]['sha256'],'old_crop_overlap_changed')
        masks[kind], domains[kind], placements[kind] = values.astype(bool), domain.astype(bool), placement
        del source,expected,values,domain,ds,buf; gc.collect()
    S,T,Ds = masks['cerebrum'],masks['whole_tumor'],domains['cerebrum']
    qc.need(int(T.sum()) == failure['full_supplied_target_voxels'] == 35260,'full_T_changed')
    O = S|T
    # Same physical side/transverse location and same closed-disc rule. No new
    # patient, alternative access, or search for a seed is attempted.
    sign = failure['inward_axis0_sign']; qc.need(sign==1,'fixed_negative_axis0_side')
    transverse = np.asarray(failure['transverse'])+old_start[1:]-start[1:]
    center = np.argwhere(T).mean(0)
    qc.need(np.array_equal(np.rint(center[1:]).astype(int),transverse),'unchanged_target_transverse')
    spacing = np.linalg.norm(affine[:3,:3],axis=0)
    yy,zz = np.ogrid[:shape[1],:shape[2]]
    dy,dz = abs((yy-transverse[0])*spacing[1]),abs((zz-transverse[1])*spacing[2])
    footprint = np.maximum(dy-spacing[1]/2,0)**2+np.maximum(dz-spacing[2]/2,0)**2 <= (6+EPS)**2
    inside = (dy+spacing[1]/2)**2+(dz+spacing[2]/2)**2 < (6-EPS)**2
    occupied = np.flatnonzero(np.any(O&footprint[None],axis=(1,2)))
    face = float(occupied[0]-.5)
    depth = (np.arange(shape[0])-face)*spacing[0]
    lo,hi = depth-spacing[0]/2,depth+spacing[0]/2
    proximal = (lo < -EPS)&(abs(hi)<=EPS)
    K = Ds&~O&proximal[:,None,None]&inside[None]
    plane62 = 62-int(start[0])
    p0_bounds=[]
    # Both fixed generic tools have g=tip_radius throughout their <=45deg
    # allowed cone. This checks the distal point, not the whole capsule/shaft.
    for name,r in (('generic_suction',1.0),('generic_aspirator',1.7)):
        x=face-(r+EPS)/spacing[0]
        # Aperture-entry radial displacement is at most6mm and a <=45deg
        # preentry shift adds at most(r+EPS)mm transversely. Report a bound,
        # not a tool-motion query or a whole-capsule clearance certificate.
        transverse_lo=transverse-(6+r+EPS)/spacing[1:]
        transverse_hi=transverse+(6+r+EPS)/spacing[1:]
        p0_bounds.append({'tool_id':name,'distal_point_full_MR_axis0':float(x+start[0]),
            'distal_point_inside_outer_crop_axis0':bool(-.5<=x<shape[0]-.5),
            'distal_point_inside_acquired_MRI_axis0':bool(-.5<=x+start[0]<mr['shape_xyz'][0]-.5),
            'all_allowed_cone_p0_transverse_bound_inside_crop':bool(np.all(transverse_lo>=-.5) and np.all(transverse_hi<np.asarray(shape[1:])-.5)),
            'p0_transverse_full_MR_bound':[(transverse_lo+start[1:]).tolist(),(transverse_hi+start[1:]).tolist()],
            'whole_tool_in_image_certified':False})
    guard()
    result = {'status':'source_domain_diagnostic_complete','patient_id':'ReMIND-013','role':'SELECT',
        'input_bindings':PINS,'source_read_accounting':accounting,'outer_crop_start_MR':start.tolist(),
        'outer_crop_shape':list(shape),'crop_map':crop_map,'native_crop_affine':native_affine.tolist(),
        'derived_affine':affine.tolist(),'source_placement':placements,'old_crop_overlap_SHA256':overlap_hashes,
        'full_target_voxels':int(T.sum()),'restored_MR_x62_closed_disc':{
            'raw_S':int((S[plane62]&footprint).sum()),'full_T':int((T[plane62]&footprint).sum()),
            'Ds_inside_whole_disc':int((Ds[plane62]&inside).sum()),'whole_disc_cells':int(inside.sum())},
        'unchanged_rule':{'side':sign,'transverse_full_MR':(transverse+start[1:]).tolist(),
            'old_outer_face_full_MR':float(failure['outer_face_source_index']+old_start[0]),
            'restored_outer_face_full_MR':float(face+start[0]),'seed_cells':int(K.sum()),
            'known_zero_component_cells':None,'known_zero_component_status':'not_measured; seed diagnostic only',
            'seed_failure_remains':not bool(K.any())},
        'preentry_distal_point_bounds':p0_bounds,'unknown_is_free':False,'known_zero_is_physical_air':False,
        'MR_pixels_read':0,'private_objects_read':0,'native_previews':0,'model_forwards':0,'optimizer_updates':0,
        'planning_manifest_created':False,'material_or_target_deleted':False,'access_admitted':False,
        'limits':'Diagnostic source recovery only; NN sampling differs from source voxel counts. E remains an explicit unassessed simulation workspace. No whole-tool clearance or accepted route is established.',
        'elapsed_seconds':time.monotonic()-started,'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    raw=(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
    qc.need(len(raw)<2*1024**2,'compact_JSON_limit')
    with args.output.open('xb') as stream: stream.write(raw)
    print(json.dumps({'status':result['status'],'seed_cells':int(K.sum()),'output_sha256':hashlib.sha256(raw).hexdigest()}))


if __name__=='__main__':
    main()
