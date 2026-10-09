"""Header-only TRAIN ReMIND identity, timing, and physical-grid checks.

Optional DICOM/array dependencies load only when a reader or MR grid check runs.
This module has no acquisition, conversion, or patient-header execution CLI.
The caller must supply a separately reviewed, frozen series/object contract.
"""
from __future__ import annotations

import hashlib
import importlib.util
import itertools
import math
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[2]
COMPLETION_SHA = '32e319a3aafb571019ee16da1072c834015f9fdcdb97563fca4c19418f64a61d'
CORE_SHA = '34e6ceb9560344a88226489b2dd8ca6bfb90cb87c52d2ff3184d393293df4ef6'
IO_SHA = '496821360b1daee79459952f970d299856baacad75fcc57ee589e55115036fd8'
CLASSIC_MR = '1.2.840.10008.5.1.4.1.1.4'
PIXEL_TAGS = (0x7FE00008, 0x7FE00009, 0x7FE00010)
TIMING = ('StudyDate', 'StudyTime', 'SeriesDate', 'SeriesTime', 'AcquisitionDate',
          'AcquisitionTime', 'AcquisitionDateTime', 'ContentDate', 'ContentTime',
          'TemporalPositionIdentifier', 'NumberOfTemporalPositions', 'AcquisitionNumber',
          'EchoNumbers', 'EchoTime', 'TriggerTime', 'FrameTime', 'FrameTimeVector')
GROUPING = ('TemporalPositionIdentifier', 'AcquisitionNumber', 'EchoNumbers', 'EchoTime',
            'ImageType', 'SequenceName', 'ComplexImageComponent', 'DiffusionBValue')
HEADER_TAGS = tuple(dict.fromkeys((
    'PatientID', 'StudyInstanceUID', 'SeriesInstanceUID', 'SOPInstanceUID', 'SOPClassUID',
    'FrameOfReferenceUID', 'Modality', 'StudyDescription', 'SeriesDescription',
    'Rows', 'Columns', 'NumberOfFrames', 'SamplesPerPixel', 'PhotometricInterpretation',
    'BitsAllocated', 'BitsStored', 'HighBit', 'PixelRepresentation', 'PlanarConfiguration',
    'ImageOrientationPatient', 'ImagePositionPatient', 'PixelSpacing', 'SliceThickness',
    'SpacingBetweenSlices', 'RescaleSlope', 'RescaleIntercept', 'RescaleType',
    'SharedFunctionalGroupsSequence', 'PerFrameFunctionalGroupsSequence',
    'DimensionOrganizationSequence', 'DimensionIndexSequence', 'SequenceOfUltrasoundRegions',
    'VolumetricProperties') + TIMING + GROUPING))


class Refusal(ValueError):
    pass


def require(value, reason):
    if not value:
        raise Refusal(reason)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_module(path, expected_sha, name):
    require(sha(path) == expected_sha, 'source_digest_changed:' + name)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def value(ds, keyword):
    """Small JSON-safe header projection; no decoder or pixel property access."""
    item = getattr(ds, keyword, None)
    if item is None:
        return None
    if isinstance(item, (list, tuple)) or item.__class__.__name__ == 'MultiValue':
        return [str(x) for x in item]
    return str(item)


def integer(ds, key, default=None):
    raw = getattr(ds, key, default)
    try:
        result = int(raw)
    except (TypeError, ValueError, OverflowError):
        raise Refusal('missing_or_invalid_integer:' + key)
    require(str(raw).strip() == str(result), 'noninteger:' + key)
    return result


def inspect_headers(series, jobs, records):
    """Inspect already-read header records against a bound TRAIN series contract.

    records come from read_verified_header in a future release or generated tests.
    A header result cannot establish anatomical adequacy, availability, or admission.
    """
    require(series['role'] == 'TRAIN', 'protected_or_development_role')
    require(series['modality'] in ('MR', 'US'), 'source_modality')
    require(len(records) == len(jobs) == series['expected_source_objects'], 'object_coverage')
    by_path = {job['path']: job for job in jobs}
    require(len(by_path) == len(jobs), 'duplicate_job_path')
    require(len({row['path'] for row in records}) == len(records), 'duplicate_header_path')
    require({row['path'] for row in records} == set(by_path), 'header_path_coverage')
    require(sum(job['bytes'] for job in jobs) == series['expected_source_bytes'], 'series_bytes')
    summaries, datasets, sop_ids, frame_ids, sop_classes = [], [], set(), set(), set()
    for record in records:
        job, ds = by_path[record['path']], record['dataset']
        require(record['sha256'] == job['sha256'] and record['bytes'] == job['bytes'], 'receipt_identity')
        require(record['pixel_decode_calls'] == 0 and record['stop_before_pixels'] is True,
                'header_only_proof')
        require(not any(int(element.tag) in PIXEL_TAGS for element in ds.iterall()),
                'pixel_payload_in_header_record')
        for key, expected in {
                'PatientID': series['patient_id'], 'StudyInstanceUID': series['study_instance_uid'],
                'SeriesInstanceUID': series['series_instance_uid'], 'Modality': series['modality'],
                'StudyDescription': series['source_study_description'],
                'SeriesDescription': series['source_series_description']}.items():
            require(str(getattr(ds, key, '')) == expected, 'header_identity:' + key)
        sop, sop_class = str(getattr(ds, 'SOPInstanceUID', '')), str(getattr(ds, 'SOPClassUID', ''))
        require(sop and sop not in sop_ids and sop_class, 'missing_or_duplicate_SOP')
        meta = getattr(ds, 'file_meta', None)
        require(meta is not None and str(getattr(meta, 'MediaStorageSOPInstanceUID', '')) == sop and
                str(getattr(meta, 'MediaStorageSOPClassUID', '')) == sop_class and
                bool(getattr(meta, 'TransferSyntaxUID', None)), 'file_meta_identity')
        rows, columns, frames = integer(ds, 'Rows'), integer(ds, 'Columns'), integer(ds, 'NumberOfFrames', 1)
        require(rows > 0 and columns > 0 and 0 < frames <= 10000, 'dimension_bounds')
        frame = str(getattr(ds, 'FrameOfReferenceUID', ''))
        sop_ids.add(sop); frame_ids.add(frame); sop_classes.add(sop_class); datasets.append(ds)
        summaries.append({
            'source_path': record['path'], 'source_sha256': record['sha256'], 'source_bytes': record['bytes'],
            'SOPInstanceUID': sop, 'SOPClassUID': sop_class,
            'TransferSyntaxUID': str(meta.TransferSyntaxUID), 'FrameOfReferenceUID': frame or None,
            'rows': rows, 'columns': columns, 'number_of_frames': frames,
            'acquisition_timing_raw': {k: value(ds, k) for k in TIMING},
            'grouping_raw': {k: value(ds, k) for k in GROUPING},
            'pixel_encoding_raw': {k: value(ds, k) for k in ('SamplesPerPixel', 'PhotometricInterpretation',
                'BitsAllocated', 'BitsStored', 'HighBit', 'PixelRepresentation', 'PlanarConfiguration',
                'RescaleSlope', 'RescaleIntercept', 'RescaleType')},
            'shared_functional_group_count': len(getattr(ds, 'SharedFunctionalGroupsSequence', [])),
            'per_frame_functional_group_count': len(getattr(ds, 'PerFrameFunctionalGroupsSequence', [])),
        })
    require(len(frame_ids) == 1, 'mixed_reference_frames')
    require(len(sop_classes) == 1, 'mixed_SOP_classes')
    frame_uid = next(iter(frame_ids)) or None
    geometry = {'status': 'unsupported', 'affine_xyz_to_ras_mm': None, 'physical_extent_ras_mm': None,
                'patient_space_geometry_established': False}
    if series['modality'] == 'US':
        # Calibrated image regions alone do not prove spatial tracking or a 3D patient grid.
        geometry['reason'] = 'US_patient_space_adapter_not_implemented'
        for ds, summary in zip(datasets, summaries):
            summary['ultrasound_regions'] = [{k: value(region, k) for k in (
                'RegionSpatialFormat', 'RegionDataType', 'RegionFlags', 'PhysicalUnitsXDirection',
                'PhysicalUnitsYDirection', 'PhysicalDeltaX', 'PhysicalDeltaY',
                'RegionLocationMinX0', 'RegionLocationMinY0', 'RegionLocationMaxX1', 'RegionLocationMaxY1')}
                for region in getattr(ds, 'SequenceOfUltrasoundRegions', [])]
        geometry['frame_interpretation'] = 'unresolved_spatial_or_temporal_frames'
    elif sop_classes != {CLASSIC_MR} or any(integer(ds, 'NumberOfFrames', 1) != 1 for ds in datasets):
        geometry['reason'] = 'unsupported_enhanced_or_multiframe_MR'
    else:
        require(frame_uid is not None, 'missing_MR_reference_frame')
        require(len(datasets) >= 2, 'single_plane_MR_not_3D_grid')
        import numpy as np
        core = load_module(ROOT / 'scripts/convert_remind_development.py', CORE_SHA, 'remind_existing_core')
        first = datasets[0]
        orientation = np.asarray(getattr(first, 'ImageOrientationPatient', []), dtype=float)
        spacing = np.asarray(getattr(first, 'PixelSpacing', []), dtype=float)
        require(orientation.shape == (6,) and spacing.shape == (2,), 'missing_MR_geometry')
        for ds in datasets:
            require(integer(ds, 'Rows') == integer(first, 'Rows') and
                    integer(ds, 'Columns') == integer(first, 'Columns'), 'mixed_MR_dimensions')
            iop = np.asarray(getattr(ds, 'ImageOrientationPatient', []), dtype=float)
            pixel_spacing = np.asarray(getattr(ds, 'PixelSpacing', []), dtype=float)
            require(iop.shape == (6,) and pixel_spacing.shape == (2,) and
                    np.allclose(iop, orientation, atol=1e-6, rtol=0) and
                    np.allclose(pixel_spacing, spacing, atol=1e-6, rtol=0), 'mixed_MR_grid')
            require(integer(ds, 'SamplesPerPixel') == 1 and
                    str(getattr(ds, 'PhotometricInterpretation', '')) in ('MONOCHROME1', 'MONOCHROME2'),
                    'unsupported_MR_samples')
            bits, stored = integer(ds, 'BitsAllocated'), integer(ds, 'BitsStored')
            require(bits in (8, 16, 32) and 0 < stored <= bits and
                    integer(ds, 'HighBit') == stored - 1 and integer(ds, 'PixelRepresentation') in (0, 1),
                    'unsupported_MR_encoding')
            for key in ('BitsAllocated', 'BitsStored', 'HighBit', 'PixelRepresentation',
                        'SamplesPerPixel', 'PhotometricInterpretation'):
                require(value(ds, key) == value(first, key), 'mixed_MR_encoding:' + key)
            require(integer(ds, 'NumberOfTemporalPositions', 1) == 1, 'multiple_temporal_positions')
            for key in GROUPING:
                require(value(ds, key) == value(first, key), 'mixed_MR_group:' + key)
            slope, intercept = float(getattr(ds, 'RescaleSlope', 1)), float(getattr(ds, 'RescaleIntercept', 0))
            require(math.isfinite(slope) and slope != 0 and math.isfinite(intercept), 'invalid_MR_scale')
        positions = np.asarray([getattr(ds, 'ImagePositionPatient', []) for ds in datasets], dtype=float)
        require(positions.shape == (len(datasets), 3), 'missing_MR_positions')
        normal = np.cross(orientation[:3], orientation[3:])
        require(np.isfinite(positions).all() and np.isfinite(normal).all() and np.linalg.norm(normal) > 0,
                'nonfinite_or_invalid_MR_geometry')
        projected = np.sort(positions @ (normal / np.linalg.norm(normal)))
        require(np.all(np.diff(projected) > 1e-6), 'duplicate_or_coincident_MR_planes')
        order, affine, residual = core.regular_affine(orientation, spacing, positions)
        shape = [integer(first, 'Columns'), integer(first, 'Rows'), len(datasets)]
        corners = np.asarray(list(itertools.product(*[(-.5, n-.5) for n in shape])))
        physical = (np.c_[corners, np.ones(8)] @ affine.T)[:, :3]
        geometry = {
            'status': 'classic_MR_header_grid_checked', 'patient_space_geometry_established': True,
            'affine_xyz_to_ras_mm': affine.tolist(), 'shape_xyz': shape,
            'physical_extent_ras_mm': {'definition': 'axis_aligned_bounds_of_voxel_cell_corners',
                'minimum': physical.min(axis=0).tolist(), 'maximum': physical.max(axis=0).tolist()},
            'slice_position_max_residual_mm': float(residual),
            'sorted_source_paths': [summaries[int(i)]['source_path'] for i in order],
            'reused_core': 'scripts/convert_remind_development.py:regular_affine',
            'reused_core_sha256': CORE_SHA,
            'source_pixel_values_checked': False, 'conversion_status': 'not_run',
            'anatomical_FOV_coverage': 'unreviewed',
        }
    return {
        'schema': 'remind-TRAIN-header-inventory-candidate-v1', 'status': 'header_inventory_complete',
        'patient_id': series['patient_id'], 'patient_group': series['patient_group'], 'role': 'TRAIN',
        'series_uuid': series['series_uuid'], 'series_instance_uid': series['series_instance_uid'],
        'modality': series['modality'], 'frame_of_reference_uid': frame_uid,
        'source_receipt_sha256': COMPLETION_SHA, 'objects': summaries, 'geometry': geometry,
        'source_stage_hint': series['acquisition_stage_hint'],
        'available_at_preoperative_decision_time': 'unverified', 'preoperative_input_admitted': False,
        'timing_interpretation': 'raw_source_labels_and_times_only; deidentification/order unverified',
        'anatomical_coverage': 'unreviewed', 'anatomy_qc': 'not_run', 'training_admitted': False,
        'volume_conversion_performed': False, 'pixel_decode_calls': 0, 'split_changed': False,
    }


class BudgetReader:
    def __init__(self, stream, limit, deadline):
        self.stream, self.limit, self.deadline, self.bytes_read = stream, limit, deadline, 0
        self.name = stream.name

    def read(self, count=-1):
        require(time.monotonic() < self.deadline, 'header_deadline')
        require(0 <= count <= self.limit - self.bytes_read, 'header_read_budget')
        data = self.stream.read(count)
        self.bytes_read += len(data)
        return data

    def seek(self, *args):
        require(time.monotonic() < self.deadline, 'header_deadline')
        return self.stream.seek(*args)

    def tell(self):
        return self.stream.tell()


def read_verified_header(source_root, job, *, deadline, maximum_header_bytes):
    """UNRELEASED reader: only a separately reviewed launcher may call on patients.

    Uses existing full-file size/SHA verifier, then checks file identity before and
    after a bounded allowlisted header read. Never evaluates a pixel decoder.
    """
    import pydicom
    io = load_module(ROOT / 'scripts/real_intake_io.py', IO_SHA, 'remind_existing_io')
    source_root = Path(source_root).resolve()
    path = source_root / job['path']
    require(path.resolve() == path and path.is_relative_to(source_root), 'unsafe_source_path')
    fields = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    before = fields(path.stat())
    io.verify_source_file(path, {'bytes': job['bytes'], 'expected_md5': job.get('expected_md5')},
                          receipt_sha=job['sha256'], deadline=deadline)
    with path.open('rb') as stream:
        require(fields(os.fstat(stream.fileno())) == before, 'source_changed_before_header')
        bounded = BudgetReader(stream, maximum_header_bytes, deadline)
        ds = pydicom.dcmread(bounded, force=False, stop_before_pixels=True, specific_tags=HEADER_TAGS)
        require(fields(os.fstat(stream.fileno())) == before, 'source_changed_during_header')
    require(fields(path.stat()) == before, 'source_changed_after_header')
    require(time.monotonic() < deadline, 'header_deadline')
    return {'path': job['path'], 'sha256': job['sha256'], 'bytes': job['bytes'], 'dataset': ds,
            'stop_before_pixels': True, 'pixel_decode_calls': 0, 'header_bytes_read': bounded.bytes_read}
