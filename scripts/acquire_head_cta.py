#!/usr/bin/env python3
"""Preserve selected original DICOM slices and aneurysm meshes from Head CTA."""

from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
import csv
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import re
from tempfile import TemporaryDirectory
import zipfile
import zlib

from data_rebuild_acquire import DEFAULT_ROOT, UUID, request_range
from resectionlab.data_root import atomic_json, checked_root, safe_path, sha256_file


RELEASE = 'https://zenodo.org/records/15697196'
ARCHIVE_URL = 'https://zenodo.org/api/records/15697196/files/Dataset.zip/content'
ARCHIVE_SIZE = 25_578_845_008
ARCHIVE_MD5 = '264ff9ee868c022d108b7c7aa7396d32'
ARCHIVE_MEMBERS = 149_452
RANGE_BYTES = 8 * 1024**2
MAX_MEMBER_BYTES = 32 * 1024**2
DESTINATION = Path('sources/head-cta-15697196/selective')
PILOTS = {'189': (276, 0), '013': (289, 1), '001': (643, 0)}


def get_range(first, length):
    if not (0 < length <= RANGE_BYTES and 0 <= first <= ARCHIVE_SIZE - length):
        raise ValueError('Range exceeds the bounded publisher archive request')
    with request_range(ARCHIVE_URL, first, first + length - 1, ARCHIVE_SIZE) as response:
        if response.headers.get('Content-Encoding', 'identity') != 'identity':
            raise ValueError('Publisher returned a transformed ZIP range')
        data = response.read(length + 1)
    if len(data) != length:
        raise ValueError('Publisher returned an incorrect ZIP range length')
    return data


class ArchiveReader(io.RawIOBase):
    """Expose exact archive bytes to zipfile, preferring an active study cache."""

    def __init__(self):
        self.position = 0
        self.cache = None
        self.transport_bytes = 0

    def tell(self):
        return self.position

    def seekable(self):
        return True

    def seek(self, offset, whence=io.SEEK_SET):
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.position, io.SEEK_END: ARCHIVE_SIZE}[whence]
        self.position = base + offset
        if self.position < 0:
            raise ValueError('Negative seek in ZIP archive')
        return self.position

    def read(self, size=-1):
        if size < 0:
            size = ARCHIVE_SIZE - self.position
        size = max(0, min(size, ARCHIVE_SIZE - self.position))
        if size > 2 * RANGE_BYTES:
            raise ValueError('Unbounded ZIP reader request')
        start = self.position
        stop = start + size
        if self.cache and self.cache[0] <= start <= stop <= self.cache[1]:
            first, _, stream = self.cache
            stream.seek(start - first)
            data = stream.read(size)
            if len(data) != size:
                raise ValueError('Cached ZIP range is truncated')
        else:
            data = b''.join(get_range(offset, min(RANGE_BYTES, stop - offset))
                            for offset in range(start, stop, RANGE_BYTES))
            self.transport_bytes += len(data)
        self.position = stop
        return data

    @contextmanager
    def use_cache(self, file, start, stop):
        with file.open('rb') as stream:
            self.cache = (start, stop, stream)
            try:
                yield
            finally:
                self.cache = None


def download_range(root, file, start, stop):
    """Resume a selected contiguous study range after completed chunks."""
    offset = file.stat().st_size if file.exists() else 0
    if offset > stop - start:
        raise ValueError(f'Oversized retained ZIP range: {file}')
    initial = offset
    with file.open('ab') as stream:
        while start + offset < stop:
            length = min(RANGE_BYTES, stop - start - offset)
            checked_root(root, required_bytes=length, expected_uuid=UUID)
            chunk = get_range(start + offset, length)
            stream.write(chunk)
            stream.flush()
            os.fsync(stream.fileno())
            offset += length
    return offset - initial


def extract_member(root, archive, info):
    if not 0 < info.file_size <= MAX_MEMBER_BYTES:
        raise ValueError(f'Unexpected source member size: {info.filename}')
    path = safe_path(root, DESTINATION / info.filename)
    if path.exists():
        if path.stat().st_size != info.file_size:
            raise ValueError(f'Existing source member has a different size: {path}')
        payload = path.read_bytes()
        if zlib.crc32(payload) != info.CRC:
            raise ValueError(f'Existing source member has a different ZIP CRC32: {path}')
    else:
        # ZipFile validates the local name, member length and central-directory CRC.
        payload = archive.read(info)
        if len(payload) != info.file_size:
            raise ValueError(f'Unexpected decompressed source length: {info.filename}')
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = safe_path(root, path.relative_to(root).with_name(path.name + '.partial'))
        temporary.write_bytes(payload)
        os.replace(temporary, path)
    return {
        'path': str(path.relative_to(root)), 'source_member': info.filename,
        'bytes': len(payload), 'zip_crc32': f'{info.CRC:08x}',
        'sha256': hashlib.sha256(payload).hexdigest(),
    }


def verify_receipt(root, receipt, subject):
    expected_slices, expected_meshes = PILOTS[subject]
    files = receipt['files']
    if (receipt['schema'] != 'ressectionlab-head-cta-selective-source-v1'
            or receipt['source_person_family'] != f'HeadCTA15697196:{subject}'
            or receipt['publisher_archive_url'] != ARCHIVE_URL
            or receipt['publisher_archive_md5'] != ARCHIVE_MD5
            or sum(file['source_member'].startswith(f'Studies/{subject}/') for file in files) != expected_slices
            or sum(file['source_member'].startswith(f'Annotations/an_{subject}') for file in files) != expected_meshes
            or len(files) != expected_slices + expected_meshes + 1
            or len({file['path'] for file in files}) != len(files)):
        raise ValueError(f'Acquisition receipt does not match pinned study {subject}')
    for record in files:
        path = safe_path(root, record['path'])
        if (record['path'] != str(DESTINATION / record['source_member'])
                or path.stat().st_size != record['bytes'] or sha256_file(path) != record['sha256']):
            raise ValueError(f'Retained original differs from the receipt: {path}')


def acquire(root, archive, reader, subject):
    expected_slices, expected_meshes = PILOTS[subject]
    study_pattern = re.compile(rf'Studies/{subject}/[0-9]+\.dcm')
    mesh_pattern = re.compile(rf'Annotations/an_{subject}(?:_[0-9]+)?\.stl')
    entries = sorted(archive.infolist(), key=lambda info: info.header_offset)
    studies = [info for info in entries if study_pattern.fullmatch(info.filename)]
    meshes = [info for info in entries if mesh_pattern.fullmatch(info.filename)]
    if (len(entries) != ARCHIVE_MEMBERS or len(studies) != expected_slices or len(meshes) != expected_meshes
            or len({info.filename for info in studies + meshes}) != expected_slices + expected_meshes):
        raise ValueError(f'Publisher ZIP contents differ from pinned study {subject}')
    begin = entries.index(studies[0])
    end = begin + len(studies)
    if entries[begin:end] != studies:
        raise ValueError('Study ZIP members are not one contiguous archive range')
    start, stop = studies[0].header_offset, entries[end].header_offset
    expanded = sum(info.file_size for info in studies + meshes)
    if stop - start > 256 * 1024**2 or expanded > 512 * 1024**2:
        raise ValueError('Selected study exceeds pilot archive or extraction bound')
    checked_root(root, required_bytes=stop - start + expanded, expected_uuid=UUID)
    cache = safe_path(root, DESTINATION / f'.{subject}.{start}-{stop}.range-partial')
    transport_start = reader.transport_bytes
    metadata = extract_member(root, archive, archive.getinfo('Metadata.csv'))
    metadata_path = safe_path(root, metadata['path'])
    with metadata_path.open(encoding='cp1251', newline='') as stream:
        rows = [row for row in csv.DictReader(stream, delimiter=';') if row['ID'] == subject]
    if (not rows or rows[0]['Images count'] != str(expected_slices)
            or (rows[0]['IA(0/№_?)'] == '0') != (expected_meshes == 0)):
        raise ValueError(f'Publisher metadata disagrees with pinned selected study {subject}')
    print(f'Acquiring {subject} from a {stop - start:,}-byte ZIP range', flush=True)
    received = download_range(root, cache, start, stop)
    with reader.use_cache(cache, start, stop):
        records = [extract_member(root, archive, info) for info in studies]
    records.extend(extract_member(root, archive, info) for info in meshes)
    records.append(metadata)
    result = {
        'schema': 'ressectionlab-head-cta-selective-source-v1',
        'dataset': 'Open dataset of annotated DICOM images of head CTA for intracranial aneurysm detection',
        'publication_date': '2025-06-17', 'publisher_release': RELEASE,
        'publisher_archive_url': ARCHIVE_URL, 'publisher_archive_bytes': ARCHIVE_SIZE,
        'publisher_archive_md5': ARCHIVE_MD5, 'whole_archive_md5_verified': False,
        'license': 'CC-BY-4.0', 'source_person_family': f'HeadCTA15697196:{subject}',
        'source_subject': subject, 'role': 'DEVELOPMENT', 'available_modalities': ['CTA'],
        'study_status': 'aneurysm' if meshes else 'aneurysm-free control', 'defaced': True,
        'publisher_mask_scope': 'aneurysm STL surface only' if meshes else 'no supplied annotations',
        'metadata_encoding': 'cp1251', 'metadata_separator': ';',
        'publisher_slice_thickness_mm': float(rows[0]['Slice Thickness, mm'].replace(',', '.')),
        'publisher_head_and_neck': rows[0]['Head+Neck (0/1)'] == '1', 'files': records,
        'study_range': {'first': start, 'last_inclusive': stop - 1, 'bytes': stop - start},
        'integrity_scope': 'Official exact HTTPS ranges, ZIP member CRC32, local SHA-256',
    }
    checked_root(root, expected_uuid=UUID)
    receipt_path = safe_path(root, DESTINATION / 'receipts' / f'{subject}.json')
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(receipt_path, result)
    cache.unlink()
    return result, received + reader.transport_bytes - transport_start


def prepare_source_study(root, subject, receipt_path, result):
    """Preserve one archived CTA series as a native, source-linked study."""
    import numpy as np
    import pydicom
    import SimpleITK as sitk

    from resectionlab.anatomy_import import _dicom_sources, import_dicom, import_nifti_series
    from resectionlab.patient_anatomy import MAX_ARRAY, PatientImagingCollection

    source = safe_path(root, DESTINATION / 'Studies' / subject)
    header = pydicom.dcmread(source / '0.dcm', stop_before_pixels=True)
    slices, _ = PILOTS[subject]
    last = pydicom.dcmread(source / f'{slices - 1}.dcm', stop_before_pixels=True)
    shape = [int(header.Columns), int(header.Rows), slices]
    if shape[0] * shape[1] * shape[2] * 4 > MAX_ARRAY:
        raise ValueError('Native DICOM study exceeds the application scalar array budget')
    axes = np.asarray(header.ImageOrientationPatient, dtype=float).reshape(2, 3)
    start = np.asarray(header.ImagePositionPatient, dtype=float)
    stop = np.asarray(last.ImagePositionPatient, dtype=float)
    reverse = np.dot(stop - start, np.cross(*axes)) < 0
    if reverse:
        start, stop = stop, start
    expected_affine = np.eye(4)
    row_spacing, col_spacing = [float(value) for value in header.PixelSpacing]
    expected_affine[:3, :4] = np.column_stack((
        col_spacing * axes[0], row_spacing * axes[1], (stop - start) / (slices - 1), start,
    ))
    expected_affine[:2] *= -1  # DICOM patient LPS to the native viewer's RAS.
    originals = {safe_path(root, record['path']): record['sha256'] for record in result['files']
                 if record['source_member'].startswith(f'Studies/{subject}/')}
    slice_paths = [source / f'{index}.dcm' for index in range(slices)]
    if reverse:
        slice_paths.reverse()
    if set(slice_paths) != originals.keys():
        raise ValueError('Publisher filenames do not enumerate one complete DICOM series')
    for index, path in enumerate(slice_paths):
        geometry = pydicom.dcmread(path, stop_before_pixels=True, specific_tags=[
            'ImagePositionPatient', 'ImageOrientationPatient', 'PixelSpacing',
        ])
        expected_position = start + index * (stop - start) / (slices - 1)
        if (not np.allclose(geometry.ImagePositionPatient, expected_position, atol=2e-4, rtol=0)
                or not np.allclose(geometry.ImageOrientationPatient, axes.ravel(), atol=1e-6, rtol=0)
                or not np.allclose(geometry.PixelSpacing, [row_spacing, col_spacing], atol=1e-6, rtol=0)):
            raise ValueError('Original CTA contains a slice outside the expected regular physical lattice')
    directory = safe_path(root, Path('prepared/head-cta-15697196-native') / subject)
    checked_root(root, required_bytes=2 * MAX_ARRAY, expected_uuid=UUID)
    directory.mkdir(parents=True, exist_ok=True)
    manifest = directory / 'patient-anatomy.json'
    timepoint = 'source-session-unknown'
    with safe_path(root, directory.relative_to(root) / '.intake.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        family = result['source_person_family']
        if manifest.exists():
            collection = PatientImagingCollection(directory)
            if collection.document['patient_id'] != family:
                raise ValueError('Prepared study belongs to another source family')
        else:
            collection = PatientImagingCollection.create(directory, family, timepoint=timepoint)
        patient_id = str(header.get('PatientID', '')).strip()
        receipt_sha = sha256_file(receipt_path)
        if patient_id:
            rows = import_dicom(collection, source, modality='CTA', timepoint=timepoint,
                                source_patient_id=patient_id)
            if len(rows) != 1 or sorted(item['sha256'] for item in rows[0]['dicom_originals']) != sorted(originals.values()):
                raise ValueError('DICOM intake originals differ from the extraction receipt')
            row = rows[0]
            intake_method = 'Production DICOM intake'
        else:
            # The publisher erased PatientID. The normal browser-initiated DICOM
            # import cannot establish a patient in that case. This source-specific
            # adapter has the verified archive's subject directory and metadata.
            series = list(_dicom_sources(source))
            if len(series) != 1:
                raise ValueError('Publisher subject directory contains mixed DICOM series')
            details, _, files = series[0]
            series_uid = details['series_uid']
            if (not re.fullmatch(r'[0-9]+(?:\.[0-9]+)*', details['study_uid'] or '')
                    or details['patient_id'] is not None or details['projection_or_localizer']
                    or details['storage'] != 'series' or set(map(Path, files)) != originals.keys()):
                raise ValueError('DICOM series cannot be associated with the archived subject')
            intake_method = 'GDCM native decode and verified publisher subject association'
            if collection.source_records():
                row = collection.series(series_uid)
                if row.get('publisher_acquisition_receipt_sha256') != receipt_sha:
                    raise ValueError('Prepared series was not derived from the verified Publisher ZIP receipt')
            else:
                reader = sitk.ImageSeriesReader()
                reader.SetFileNames(files)
                image = reader.Execute()
                if image.GetDimension() != 3:
                    raise ValueError('GDCM did not decode a 3D CTA volume')
                with TemporaryDirectory(prefix='.head-cta-', dir=directory) as temp:
                    native = Path(temp) / 'Head CTA native.nii.gz'
                    sitk.WriteImage(image, str(native))
                    row = import_nifti_series(collection, native, series_id=series_uid, modality='CTA',
                                              timepoint=timepoint, patient_id=family)
                row.update(
                    label='Head CTA · native DICOM grid', dicom_frame_uid=str(header.FrameOfReferenceUID),
                    dicom_modality='CT', publisher_source_study_uid=details['study_uid'],
                    publisher_original_dicom_count=len(originals),
                    publisher_originals_directory=str(source.relative_to(root)),
                )
                if details.get('intensity_units') == 'HU':
                    row.update(intensity_units='HU', intensity_units_basis='DICOM rescale type declaration')
        if row['data']['shape'] != shape:
            raise ValueError('DICOM conversion did not yield the complete native CTA grid')
        if not np.allclose(row['affine'], expected_affine, atol=2e-4, rtol=0):
            raise ValueError('Native CTA affine differs from independent DICOM endpoint geometry')
        native_array = collection.array(row['data'])
        sample = native_array[:, :, shape[2] // 2]
        if (not np.isfinite(sample).all() or np.ptp(sample) <= 0
                or sha256_file(collection.path(row['original']['path'])) != row['original']['sha256']):
            raise ValueError('Native CTA scalar or its NIfTI source failed verification')
        sample_indices = [0, slices // 2, slices - 1]
        for index in sample_indices:
            source_plane = sitk.GetArrayFromImage(sitk.ReadImage(str(slice_paths[index]))).squeeze().T
            if not np.array_equal(source_plane, native_array[:, :, index]):
                raise ValueError('Native CTA intensity samples differ from individually decoded DICOM slices')
        if row.get('publisher_acquisition_receipt_sha256') != receipt_sha:
            row.update(publisher_dataset=RELEASE, source_license='CC-BY-4.0',
                       publisher_acquisition_receipt_sha256=receipt_sha)
            collection.document['missing'] = [
                'Noncontrast CT and structural MRI', 'Brain and skull reference segmentations',
                'Whole-vessel arterial and venous annotations',
                'Aneurysm STL coordinate verification' if result['study_status'] == 'aneurysm'
                else 'Supplied anatomical mesh',
            ]
            checked_root(root, expected_uuid=UUID)
            collection.save('Preserve original Head CTA DICOM and native source grid')
        verification = {
            'schema': 'ressectionlab-head-cta-native-source-verification-v1',
            'source_person_family': family, 'role': 'DEVELOPMENT',
            'source_acquisition_receipt': str(receipt_path.relative_to(root)),
            'source_acquisition_receipt_sha256': receipt_sha,
            'native_manifest': str(manifest.relative_to(root)),
            'native_manifest_sha256': sha256_file(manifest),
            'native_shape': shape, 'native_spacing_mm': row['grid_spacing_mm'],
            'native_affine_ras_mm': row['affine'], 'modality': row['modality'],
            'native_nifti_sha256': row['original']['sha256'], 'native_scalar_sha256': row['data']['sha256'],
            'center_slice_scalar_range': [float(sample.min()), float(sample.max())],
            'native_intensity_units': row.get('intensity_units', 'not declared by DICOM'),
            'exact_dicom_intensity_match_slice_indices': sample_indices,
            'clinical_timepoint': 'unspecified by publisher',
            'native_timepoint': timepoint, 'same_original_sha256_count': len(originals),
            'dicom_transfer_syntax_uid': str(header.file_meta.TransferSyntaxUID),
            'dicom_patient_id_present': bool(patient_id), 'intake_method': intake_method,
            'slice_geometry_check': 'All DICOM headers matched one regular lattice and its native RAS affine',
            'supplied_annotation_status': result['publisher_mask_scope'],
            'registered_geometry': False,
        }
        review = safe_path(root, Path('qc/head-cta-15697196-native') / subject / 'verification.json')
        checked_root(root, expected_uuid=UUID)
        review.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(review, verification)
        return {**verification, 'verification': str(review)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=Path(os.environ.get('RESSECTIONLAB_DATA_ROOT') or DEFAULT_ROOT))
    parser.add_argument('--prepare', action='store_true', help='Prepare verified originals as a native CTA source study.')
    parser.add_argument('subjects', nargs='*', help='Pinned pilot IDs, defaulting to 189 and 013.')
    args = parser.parse_args()
    subjects = args.subjects or ['189', '013']
    if len(set(subjects)) != len(subjects) or set(subjects) - PILOTS.keys():
        parser.error('Choose distinct pilot IDs from ' + ', '.join(PILOTS))
    root = checked_root(args.data_root, expected_uuid=UUID)
    directory = safe_path(root, DESTINATION)
    directory.mkdir(parents=True, exist_ok=True)
    completed = []
    with safe_path(root, DESTINATION / '.acquisition.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with ExitStack() as stack:
            reader = stack.enter_context(ArchiveReader())
            archive = None
            for subject in subjects:
                receipt_path = safe_path(root, DESTINATION / 'receipts' / f'{subject}.json')
                if receipt_path.exists():
                    result = json.loads(receipt_path.read_text())
                    verify_receipt(root, result, subject)
                    span = result['study_range']
                    cache = safe_path(root, DESTINATION / f".{subject}.{span['first']}-{span['last_inclusive'] + 1}.range-partial")
                    cache.unlink(missing_ok=True)
                    received = 0
                else:
                    overhead = 0
                    if archive is None:
                        archive = stack.enter_context(zipfile.ZipFile(reader))
                        overhead = reader.transport_bytes
                    result, received = acquire(root, archive, reader, subject)
                    received += overhead
                print(json.dumps({
                    'subject': subject, 'files': len(result['files']),
                    'verified_original_bytes': sum(record['bytes'] for record in result['files']),
                    'new_transport_bytes': received, 'receipt': str(receipt_path),
                }), flush=True)
                completed.append((subject, receipt_path, result))
    if args.prepare:
        for subject, receipt_path, result in completed:
            study = prepare_source_study(root, subject, receipt_path, result)
            print(json.dumps({'subject': subject, **study}), flush=True)


if __name__ == '__main__':
    main()
