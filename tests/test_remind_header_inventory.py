"""Generated software controls only; no acquired file or patient header reads."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from resectionlab import remind_header_inventory as adapter

OPTIONAL_MISSING = [name for name in ('numpy', 'nibabel', 'pydicom')
                    if importlib.util.find_spec(name) is None]
if not OPTIONAL_MISSING:
    import numpy as np
    import pydicom
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian

ROOT = Path(__file__).resolve().parents[1]
GENERATED_TMP = ROOT / 'build/remind-header-inventory-tests'


def fixture(modality='MR'):
    count = 3 if modality == 'MR' else 1
    series = {'role': 'TRAIN', 'patient_id': 'SOFTWARE-TEST', 'patient_group': 'SOFTWARE:TEST',
        'study_instance_uid': '1.2.3', 'series_instance_uid': '1.2.3.4', 'series_uuid': 'generated-series',
        'modality': modality, 'source_study_description': 'Preop' if modality == 'MR' else 'Intraop',
        'source_series_description': 'generated-MR' if modality == 'MR' else 'US_pre_dura',
        'expected_source_objects': count, 'expected_source_bytes': count * 100,
        'acquisition_stage_hint': 'preoperative_label_only' if modality == 'MR' else 'intraoperative_label_only'}
    jobs, records = [], []
    for index in range(count):
        ds = Dataset()
        ds.PatientID = series['patient_id']; ds.StudyInstanceUID = series['study_instance_uid']
        ds.SeriesInstanceUID = series['series_instance_uid']; ds.Modality = modality
        ds.StudyDescription = series['source_study_description']; ds.SeriesDescription = series['source_series_description']
        ds.SOPClassUID = adapter.CLASSIC_MR if modality == 'MR' else '1.2.840.10008.5.1.4.1.1.6.1'
        ds.SOPInstanceUID = f'1.2.3.4.{index+1}'
        ds.file_meta = FileMetaDataset()
        ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
        ds.file_meta.MediaStorageSOPClassUID = ds.SOPClassUID
        ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds.FrameOfReferenceUID = '1.2.99'; ds.Rows = 2; ds.Columns = 3
        ds.NumberOfFrames = 1; ds.SamplesPerPixel = 1; ds.PhotometricInterpretation = 'MONOCHROME2'
        ds.BitsAllocated = 16; ds.BitsStored = 12; ds.HighBit = 11; ds.PixelRepresentation = 0
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]; ds.PixelSpacing = [2, 3]
        ds.ImagePositionPatient = [0, 0, [4, 0, 2][index]]
        ds.EchoNumbers = 1; ds.AcquisitionNumber = 1; ds.AcquisitionTime = f'12000{index}'
        path = f'SOFTWARE-TEST/generated-series/{index}.dcm'
        jobs.append({'path': path, 'bytes': 100, 'sha256': 'a' * 64})
        records.append({'path': path, 'bytes': 100, 'sha256': 'a' * 64, 'dataset': ds,
                        'stop_before_pixels': True, 'pixel_decode_calls': 0})
    return series, jobs, records


@unittest.skipIf(bool(OPTIONAL_MISSING), 'Optional DICOM controls require: ' + ', '.join(OPTIONAL_MISSING))
class Controls(unittest.TestCase):
    def test_reuses_classic_MR_core_and_reports_direct_physical_mapping(self):
        series, jobs, records = fixture()
        with patch.object(Dataset, 'pixel_array', property(lambda self: self.fail_pixel_access())):
            result = adapter.inspect_headers(series, jobs, records)
        grid = result['geometry']
        self.assertEqual(grid['status'], 'classic_MR_header_grid_checked')
        self.assertEqual(grid['sorted_source_paths'], [jobs[i]['path'] for i in (1, 2, 0)])
        affine = np.array(grid['affine_xyz_to_ras_mm'])
        np.testing.assert_allclose(affine @ [2, 1, 2, 1], [-6, -2, 4, 1], atol=1e-12)
        self.assertEqual(grid['shape_xyz'], [3, 2, 3])
        self.assertEqual(result['available_at_preoperative_decision_time'], 'unverified')
        self.assertFalse(result['preoperative_input_admitted'])
        self.assertFalse(result['training_admitted'])
        json.dumps(result, allow_nan=False)

    def test_identity_coverage_receipt_and_payload_refusals(self):
        for change in ('role', 'missing', 'duplicate_path', 'sha', 'PatientID', 'StudyInstanceUID',
                       'SeriesInstanceUID', 'Modality', 'SOPInstanceUID', 'file_meta', 'frame', 'pixels'):
            with self.subTest(change=change):
                series, jobs, records = fixture()
                ds = records[0]['dataset']
                if change == 'role': series['role'] = 'MEASUREMENT_EVAL'
                elif change == 'missing': records.pop()
                elif change == 'duplicate_path': records[0]['path'] = records[1]['path']
                elif change == 'sha': records[0]['sha256'] = 'b' * 64
                elif change in ('PatientID', 'Modality'): setattr(ds, change, 'INVALID')
                elif change in ('StudyInstanceUID', 'SeriesInstanceUID'): setattr(ds, change, '9.9')
                elif change == 'SOPInstanceUID':
                    ds.SOPInstanceUID = records[1]['dataset'].SOPInstanceUID
                    ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
                elif change == 'file_meta': ds.file_meta.MediaStorageSOPInstanceUID = '9.9'
                elif change == 'frame': ds.FrameOfReferenceUID = '9.9'
                elif change == 'pixels': ds.PixelData = b'generated'
                with self.assertRaises(ValueError): adapter.inspect_headers(series, jobs, records)

    def test_MR_grid_temporal_and_encoding_refusals(self):
        for change in ('duplicate_plane', 'near_duplicate', 'nonuniform', 'orientation', 'missing_spacing',
                       'zero_spacing', 'echo', 'temporal', 'scale', 'encoding', 'dimension', 'missing_frame'):
            with self.subTest(change=change):
                series, jobs, records = fixture()
                ds = records[0]['dataset']
                if change == 'duplicate_plane': ds.ImagePositionPatient = [0, 0, 2]
                elif change == 'near_duplicate': ds.ImagePositionPatient = [0, 0, 2.0000001]
                elif change == 'nonuniform': ds.ImagePositionPatient = [0, 0, 4.1]
                elif change == 'orientation': ds.ImageOrientationPatient = [1, 0, 0, .1, 1, 0]
                elif change == 'missing_spacing': del ds.PixelSpacing
                elif change == 'zero_spacing': ds.PixelSpacing = [0, 3]
                elif change == 'echo': ds.EchoNumbers = 2
                elif change == 'temporal': ds.NumberOfTemporalPositions = 2
                elif change == 'scale': ds.RescaleSlope = 0
                elif change == 'encoding': ds.BitsAllocated = 32
                elif change == 'dimension': ds.Rows = 3
                elif change == 'missing_frame':
                    for row in records: del row['dataset'].FrameOfReferenceUID
                with self.assertRaises(ValueError): adapter.inspect_headers(series, jobs, records)

    def test_enhanced_and_multiframe_MR_stay_unsupported(self):
        for enhanced in (False, True):
            series, jobs, records = fixture()
            for row in records:
                if enhanced:
                    row['dataset'].SOPClassUID = '1.2.840.10008.5.1.4.1.1.4.1'
                    row['dataset'].file_meta.MediaStorageSOPClassUID = row['dataset'].SOPClassUID
                else: row['dataset'].NumberOfFrames = 10
            result = adapter.inspect_headers(series, jobs, records)
            self.assertEqual(result['geometry']['reason'], 'unsupported_enhanced_or_multiframe_MR')
            self.assertIsNone(result['geometry']['affine_xyz_to_ras_mm'])

    def test_US_calibration_and_pre_dura_name_do_not_establish_patient_geometry_or_preop(self):
        series, jobs, records = fixture('US')
        ds = records[0]['dataset']; del ds.FrameOfReferenceUID; ds.NumberOfFrames = 100
        region = Dataset(); region.PhysicalDeltaX = .01; region.PhysicalDeltaY = .01
        region.PhysicalUnitsXDirection = 3; region.PhysicalUnitsYDirection = 3
        ds.SequenceOfUltrasoundRegions = [region]
        result = adapter.inspect_headers(series, jobs, records)
        self.assertEqual(result['geometry']['reason'], 'US_patient_space_adapter_not_implemented')
        self.assertIsNone(result['geometry']['affine_xyz_to_ras_mm'])
        self.assertIsNone(result['frame_of_reference_uid'])
        self.assertEqual(result['objects'][0]['number_of_frames'], 100)
        self.assertEqual(result['source_stage_hint'], 'intraoperative_label_only')
        self.assertFalse(result['preoperative_input_admitted'])
        self.assertFalse(result['training_admitted'])

    def test_generated_file_reader_is_allowlisted_and_stops_before_pixels(self):
        series, jobs, records = fixture()
        ds = records[0]['dataset']; ds.PatientName = 'GENERATED^EXCLUDED'
        ds.Rows = 64; ds.Columns = 64; ds.PixelData = bytes(8192)
        GENERATED_TMP.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=GENERATED_TMP) as directory:
            path = Path(directory) / 'generated.dcm'
            pydicom.dcmwrite(path, ds, enforce_file_format=True)
            raw = path.read_bytes()
            job = {'path': path.name, 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest(),
                   'expected_md5': hashlib.md5(raw).hexdigest()}
            with patch.object(Dataset, 'pixel_array', property(lambda self: self.fail_pixel_access())):
                result = adapter.read_verified_header(Path(directory), job, deadline=time.monotonic()+5,
                                                       maximum_header_bytes=8192)
            self.assertNotIn('PixelData', result['dataset'])
            self.assertNotIn('PatientName', result['dataset'])
            self.assertIn('PatientID', result['dataset'])
            self.assertEqual(result['pixel_decode_calls'], 0)
            self.assertLess(result['header_bytes_read'], len(raw))
            for change in ('hash', 'budget', 'deadline', 'path'):
                with self.subTest(change=change):
                    bad = dict(job)
                    if change == 'hash': bad['sha256'] = '0' * 64
                    if change == 'path': bad['path'] = '../escape.dcm'
                    with self.assertRaises((ValueError, TimeoutError)):
                        adapter.read_verified_header(Path(directory), bad,
                            deadline=time.monotonic()+5 if change != 'deadline' else time.monotonic()-1,
                            maximum_header_bytes=8192 if change != 'budget' else 10)


if __name__ == '__main__':
    unittest.main()
