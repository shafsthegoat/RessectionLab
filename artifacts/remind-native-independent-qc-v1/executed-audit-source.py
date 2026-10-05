#!/usr/bin/env python3
"""Independent direct DICOM-frame versus saved native-NIfTI sample/geometry audit."""
from __future__ import annotations
import argparse
import gc
import hashlib
import itertools
import json
from pathlib import Path
import resource
import sys
import time

import nibabel as nib
import numpy as np
import pydicom

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_plane(samples, orientation, spacing, position, image, saved, seen):
    """Use direct DICOM row/column geometry; do not reuse converter fitting code."""
    orientation, spacing, position = map(lambda a: np.asarray(a, dtype=np.float64),
                                         (orientation, spacing, position))
    rows, columns = samples.shape
    points_xy = np.asarray(list(itertools.product((0, columns - 1), (0, rows - 1))))
    lps = position + points_xy[:, :1] * orientation[:3] * spacing[1] + points_xy[:, 1:] * orientation[3:] * spacing[0]
    ras = lps * (-1., -1., 1.)
    voxels = np.c_[ras, np.ones(4)] @ np.linalg.inv(image.affine).T
    z = int(np.rint(voxels[:, 2].mean()))
    if not 0 <= z < saved.shape[2] or z in seen:
        raise ValueError("Missing/duplicate source-to-native slice correspondence")
    seen.add(z)
    expected_voxels = np.c_[points_xy, np.full(4, z), np.ones(4)]
    native_world = (expected_voxels @ image.affine.T)[:, :3]
    distance = float(np.linalg.norm(native_world - ras, axis=1).max())
    if distance > .001:
        raise ValueError(f"Direct source corner differs from native grid by {distance} mm")
    if not np.array_equal(saved[:, :, z].T, samples):
        raise ValueError("Source plane samples changed or axes were transposed incorrectly")
    return distance


def audit(output):
    started = time.perf_counter()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest_path = ROOT / 'manifests/experiments/remind-001-structural-development-v1.json'
    acquisition_path = ROOT / 'artifacts/remind-development-acquisition-v1/acquisition-attempt-1.json'
    native_root = ROOT / 'data/remind_source/converted-native/ReMIND-001'
    source_root = ROOT / 'data/remind_source/idc-v24/ReMIND-001'
    record_path = native_root / 'native-conversion.json'
    manifest, acquisition, conversion = [json.loads(p.read_text()) for p in (manifest_path, acquisition_path, record_path)]
    assert conversion['manifest_sha256'] == sha(manifest_path) == acquisition['manifest_sha256']
    assert conversion['acquisition_receipt_sha256'] == sha(acquisition_path)
    assert acquisition['status'] == 'complete'
    records = {r['role']: r for r in conversion['native_series']}
    source_hashes = {r['key']: r['sha256'] for r in acquisition['objects']}
    all_verified, results, visuals = [], [], {}
    for series in manifest['series']:
        role, native = series['role'], records[series['role']]
        path = native_root / native['file']
        if sha(path) != native['sha256']:
            raise ValueError('Native NIfTI hash differs from conversion record')
        image = nib.load(path)
        saved = np.asarray(image.dataobj)
        if list(saved.shape) != native['shape_xyz'] or image.header.get_xyzt_units()[0] != 'mm':
            raise ValueError('Native shape/units changed')
        corners = np.asarray(list(itertools.product(*[(0, n-1) for n in saved.shape])))
        homogeneous = np.c_[corners, np.ones(8)]
        q, qc = image.get_qform(coded=True); s, sc = image.get_sform(coded=True)
        if not (qc and sc):
            raise ValueError('Expected declared qform and sform')
        qs_error = float(np.linalg.norm((homogeneous @ (q-s).T)[:, :3], axis=1).max())
        if qs_error > .001:
            raise ValueError('qform and sform corner disagreement')
        seen, distances, instance_frames, instance_sops, references = set(), [], set(), [], []
        for obj in series['objects']:
            original = source_root / obj['key']
            raw = original.read_bytes()
            if len(raw) != obj['bytes'] or hashlib.sha256(raw).hexdigest() != source_hashes[obj['key']]:
                raise ValueError('Source DICOM checksum/byte count mismatch')
            if hashlib.md5(raw).hexdigest() != obj['etag_md5']:
                raise ValueError('Source DICOM differs from pinned singlepart ETag')
            ds = pydicom.dcmread(original)
            if any(str(getattr(ds, k)) != str(v) for k, v in {'PatientID':'ReMIND-001',
                    'SeriesInstanceUID':series['SeriesInstanceUID'],'StudyInstanceUID':series['StudyInstanceUID'],
                    'Modality':series['Modality']}.items()):
                raise ValueError('DICOM identity mismatch')
            instance_frames.add(str(ds.FrameOfReferenceUID)); instance_sops.append(str(ds.SOPInstanceUID))
            all_verified.append(obj['key'])
            if ds.Modality == 'MR':
                values = ds.pixel_array.astype(np.float32) * float(getattr(ds, 'RescaleSlope', 1)) + float(getattr(ds, 'RescaleIntercept', 0))
                distances.append(check_plane(values, ds.ImageOrientationPatient, ds.PixelSpacing,
                    ds.ImagePositionPatient, image, saved, seen))
            else:
                if str(ds.SegmentationType) != 'BINARY' or len(ds.SegmentSequence) != 1:
                    raise ValueError('Unexpected annotation semantics')
                pixels = ds.pixel_array
                shared = ds.SharedFunctionalGroupsSequence[0]
                for index, frame in enumerate(ds.PerFrameFunctionalGroupsSequence):
                    distances.append(check_plane(pixels[index], shared.PlaneOrientationSequence[0].ImageOrientationPatient,
                        shared.PixelMeasuresSequence[0].PixelSpacing, frame.PlanePositionSequence[0].ImagePositionPatient,
                        image, saved, seen))
                references = [str(e.value) for e in ds.iterall() if e.keyword == 'ReferencedSOPInstanceUID']
                del pixels
            del raw, ds
        if seen != set(range(saved.shape[2])) or len(set(instance_sops)) != len(instance_sops):
            raise ValueError('Incomplete planes or duplicate instances')
        if instance_frames != {native['frame_of_reference_uid']} or set(instance_sops) != set(native['sop_instance_uids']):
            raise ValueError('Declared instance/frame binding mismatch')
        result = {'role':role,'native_sha256':sha(path),'source_instances':len(series['objects']),
            'source_planes':len(seen),'all_source_pixels_equal':True,'direct_source_corner_max_error_mm':max(distances),
            'qform_sform_max_corner_difference_mm':qs_error,'frame_of_reference_uid':next(iter(instance_frames)),
            'shape_xyz':list(saved.shape),'spacing_mm':np.linalg.norm(image.affine[:3,:3],axis=0).tolist(),
            'explicit_source_sop_references':references,'affine_ras_mm':image.affine.tolist()}
        if series['Modality'] == 'SEG':
            count = int(np.count_nonzero(saved)); volume = count*abs(np.linalg.det(image.affine[:3,:3]))
            if count != native['positive_voxels'] or not np.isclose(volume, native['positive_volume_mm3'],rtol=1e-6):
                raise ValueError('Annotation count/volume mismatch')
            result.update(positive_voxels=count,positive_volume_mm3=volume,
                algorithm_type=native['algorithm_type'],annotation_accuracy_reviewed=False)
        else:
            # Native planes for visual inspection; keep exact array axis labels.
            for axis in range(3):
                visuals[f'{role}_axis{axis}'] = np.take(saved, saved.shape[axis]//2, axis=axis)
        results.append(result)
        del saved, image
        gc.collect()
    if len(all_verified) != manifest['expected_total_instances'] or set(all_verified) != set(source_hashes):
        raise ValueError('Source object inventory mismatch')
    by_role={r['role']:r for r in results}
    pairings = {seg:by_role[seg]['frame_of_reference_uid']==by_role[mr]['frame_of_reference_uid']
                for seg,mr in [('cerebrum_annotation','structural_t1ce'),('tumor_annotation','structural_t2')]}
    report={'schema_version':1,'status':'direct_source_samples_and_native_geometry_verified',
        'audit_script_sha256':sha(__file__),'manifest_sha256':sha(manifest_path),
        'acquisition_receipt_sha256':sha(acquisition_path),'conversion_record_sha256':sha(record_path),
        'source_instances_verified':len(all_verified),'source_bytes_verified':sum(o['bytes'] for s in manifest['series'] for o in s['objects']),
        'native_series':results,'segmentation_same_frame_as_named_mri':pairings,
        't1_t2_same_frame':by_role['structural_t1ce']['frame_of_reference_uid']==by_role['structural_t2']['frame_of_reference_uid'],
        'cross_frame_identity_accepted':False,'registration_performed':False,'expert_anatomical_review':False,
        'cortical_access_permitted':False,'outside_segmentation_native_grid':'unassessed, not verified absence',
        'limitation':'Both SEG files have no explicit referenced source SOP links. Matching frame and release description are evidence of correspondence, not expert alignment approval.',
        'elapsed_seconds':time.perf_counter()-started,'peak_rss_bytes':int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)),
        'runtime':{'python':sys.version,'numpy':np.__version__,'nibabel':nib.__version__,'pydicom':pydicom.__version__}}
    (output/'native-independent-qc.json').write_text(json.dumps(report,indent=2)+'\n')
    np.savez_compressed(output/'native-center-planes.npz',**visuals)
    print(json.dumps({'status':report['status'],'instances':len(all_verified),'seconds':report['elapsed_seconds'],'peak_mib':report['peak_rss_bytes']/1024**2,
        'roles':[{k:r[k] for k in ('role','direct_source_corner_max_error_mm','qform_sform_max_corner_difference_mm')} for r in results]}))
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    audit(parser.parse_args().output)
