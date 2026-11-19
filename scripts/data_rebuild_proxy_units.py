#!/usr/bin/env python3
"""Preserve v1 internal proxies and supply explicit, source-bound mm headers."""
import gzip
import hashlib
import json
from pathlib import Path
import shutil
from zipfile import ZipFile

import nibabel as nib
import numpy as np

from resectionlab.data_root import atomic_json,checked_root,safe_path,sha256_file
from resectionlab.imaging import inspect_nifti


def write_nifti1_mm_copy(original, destination):
    """Write a gzip NIfTI-1 copy with only the spatial-unit header bits changed."""
    image = nib.load(original)
    if type(image) is not nib.Nifti1Image or image.header.get_xyzt_units()[0] not in ('unknown', 'mm'):
        raise ValueError('Expected a single-file gzip NIfTI-1 with compatible spatial units')
    with gzip.open(original, 'rb') as source, destination.open('wb') as file:
        with gzip.GzipFile(filename='', fileobj=file, mode='wb', mtime=0) as output:
            header = bytearray(source.read(352))
            if len(header) != 352 or header[344:348] != b'n+1\0':
                raise ValueError('Expected complete single-file NIfTI-1 header')
            header[123] = (header[123] & ~7) | 2
            output.write(header)
            shutil.copyfileobj(source, output, 1024 * 1024)
    with gzip.open(original, 'rb') as source, gzip.open(destination, 'rb') as result:
        before, after = source.read(352), result.read(352)
        if before[:123] != after[:123] or before[124:] != after[124:] or after[123] != ((before[123] & ~7) | 2):
            raise ValueError('Copy changed more than the spatial-unit bits')
        while block := source.read(1024 * 1024):
            if block != result.read(len(block)):
                raise ValueError('Spatial unit declaration changed the voxel payload')
        if result.read(1):
            raise ValueError('Spatial unit declaration extended the voxel payload')


def run(root=None):
    root=checked_root(root)
    directory=root/'qc/proxy-unit-repairs';directory.mkdir(exist_ok=True)
    rows=[]
    for receipt in sorted((root/'catalog/preparation').glob('*-control-native-v5.json')):
        if receipt.name.startswith('.'):continue
        control=json.loads(receipt.read_text())
        package=safe_path(root,control['path'])
        with ZipFile(package) as archive:
            metadata=json.loads(archive.read('manifest.json'))['metadata']
        contract=metadata['data_case'];subject=contract['source_subject']
        original=root/'prepared/placement-support'/(subject+'-high-T1-support-v1.nii.gz')
        if not original.exists():continue
        destination=original.with_name(subject+'-high-T1-support-v2-mm.nii.gz')
        report=directory/(subject+'.json')
        if report.exists():
            prior=json.loads(report.read_text())
            if sha256_file(original)!=prior['original_sha256'] or sha256_file(destination)!=prior['repaired_sha256']:raise ValueError('Historical proxy repair changed')
            rows.append(prior);continue
        checked_root(root,required_bytes=32*1024**2)
        proxy=nib.load(original)
        if (proxy.shape!=tuple(control['native_shape']) or
                not np.allclose(proxy.affine,control['affine_ras_mm'],rtol=0,atol=1e-4)):
            raise ValueError('Proxy is not on its exact known-mm source case grid')
        structural=metadata['imaging_qc']['structural']
        if structural['physical_units']!='mm' or structural['unit_scale_to_mm']!=1.:
            raise ValueError('Source imaging must explicitly establish millimeter coordinates')
        temporary=destination.with_name(destination.name+'.partial')
        write_nifti1_mm_copy(original, temporary)
        temporary.replace(destination)
        qc=inspect_nifti(destination)
        if qc['physical_units']!='mm':raise ValueError('Repaired standalone proxy remains spatially ambiguous')
        record={'schema':'ressectionlab-placement-proxy-unit-repair-v1','subject':subject,'person_id':contract['person_id'],
            'original_path':str(original.relative_to(root)),'original_sha256':sha256_file(original),
            'repaired_path':str(destination.relative_to(root)),'repaired_sha256':sha256_file(destination),
            'source_case_hash':control['case_hash'],'source_package_path':control['path'],'source_package_sha256':control['sha256'],
            'source_affine_ras_mm':control['affine_ras_mm'],'source_explicit_units':'mm','old_header_units':proxy.header.get_xyzt_units()[0],
            'repaired_header_units':'mm','operation':'NIfTI1 xyzt_units spatial bits at byte123 only; no resampling or affine/mask change',
            'mask_payload_and_other_header_bytes_exact':True,'original_embedded_proxy_is_historical_internal_helper':True,
            'native_case_anatomy_modified':False,'importable_standalone_proxy':str(destination.relative_to(root)),
            'independent_basis':'qc/independent/placement-proxy-unit-audit-v1.json'}
        atomic_json(report,record);rows.append(record)
    atomic_json(directory/'index.json',{'schema':'ressectionlab-placement-proxy-unit-repairs-v1','cases':rows})
    print(json.dumps({'source_bound_header_repairs':len(rows),'native_cases_modified':0}))


if __name__=='__main__':run()
