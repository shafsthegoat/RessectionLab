"""Small generated controls for the reusable case boundary; no patient files."""
import copy
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

MODULE = Path(__file__).resolve().parents[1] / "src/resectionlab/remind_planning_qc.py"
qc = types.ModuleType("remind_planning_qc_under_test")
qc.__file__ = str(MODULE)
exec(compile(MODULE.read_bytes(), str(MODULE), "exec"), qc.__dict__)
try:
    import numpy as np
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.sequence import Sequence
    from pydicom.uid import ExplicitVRLittleEndian
except ImportError:
    np = None


def fixture(root, subject="ReMIND-010"):
    group = "ReMIND:" + subject.split("-")[-1]
    cohort = {"members": [{"subject": subject, "patient_group": group, "role": "TRAIN"}]}
    case = {"patient_id": subject, "patient_group": group, "role": "TRAIN", "series": []}
    flat, source = [], []
    for index, kind in enumerate(("structural_t1ce", "whole_tumor", "ventricles", "cerebrum")):
        modality = "MR" if index == 0 else "SEG"
        uid = "2.25." + str(index + 100)
        obj = {"path": str(root / f"generated-{index}.dcm"), "bytes": 10,
               "sha256": str(index)*64, "expected_md5": str(index)*32}
        series = {"kind": kind, "series_uuid": "generated-"+str(index), "SeriesInstanceUID": uid,
                  "StudyInstanceUID": "2.25.99", "Modality": modality, "source_description": kind, "objects": [obj]}
        case["series"].append(series)
        flat.append({**obj, "path": Path(obj["path"]).name, "series_uuid": series["series_uuid"]})
        source.append({"PatientID": subject, "crdc_series_uuid": series["series_uuid"], "SeriesInstanceUID": uid,
                       "StudyInstanceUID": "2.25.99", "Modality": modality, "SeriesDescription": kind, "instanceCount": 1})
    binding = {k: case[k] for k in ("patient_id", "patient_group", "role")}
    binding.update(total_objects=4, total_bytes=40, objects=flat, source_series=source)
    raw = json.dumps(binding).encode(); (root / "binding.json").write_bytes(raw)
    case["parent_bindings"] = {"source_binding": {"path": "binding.json", "sha256": qc.digest(raw)}}
    folder = root / "manifests/experiments"; folder.mkdir(parents=True)
    raw = json.dumps(cohort).encode(); (folder / "remind-component-cohort-v1.json").write_bytes(raw)
    return case, cohort, qc.digest(raw)


class CaseControls(unittest.TestCase):
    def test_parameterized_train_cases_and_frozen_family_roles(self):
        for subject in ("ReMIND-010", "ReMIND-020", "ReMIND-025"):
            with self.subTest(subject=subject), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary).resolve(); case, cohort, sha = fixture(root, subject)
                with patch.object(qc, "COHORT_SHA", sha):
                    self.assertEqual(qc.validate_case(case, root)["patient_group"], case["patient_group"])
                for role in ("SELECT", "MEASUREMENT_EVAL"):
                    altered = copy.deepcopy(cohort); altered["members"][0]["role"] = role
                    with self.assertRaisesRegex(ValueError, "immutable_TRAIN_role"): qc.validate_role(case, altered)
                changed = dict(case, patient_group="ReMIND:999")
                with self.assertRaisesRegex(ValueError, "immutable_TRAIN_role"): qc.validate_role(changed, cohort)

    def test_exact_file_fixity_and_series_assignment(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve(); case, _, sha = fixture(root)
            with patch.object(qc, "COHORT_SHA", sha):
                for change in ("fixity", "object_series", "series_uid", "duplicate"):
                    altered = copy.deepcopy(case)
                    if change == "fixity": altered["series"][0]["objects"][0]["sha256"] = "f"*64
                    elif change == "object_series":
                        altered["series"][0]["objects"], altered["series"][1]["objects"] = altered["series"][1]["objects"], altered["series"][0]["objects"]
                    elif change == "series_uid": altered["series"][0]["SeriesInstanceUID"] = "2.25.999"
                    else: altered["series"][1]["objects"] = altered["series"][0]["objects"]
                    with self.subTest(change=change), self.assertRaises(ValueError): qc.validate_case(altered, root)

    def test_cohort_hash_is_verified_not_supplied_role_alone(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve(); case, _, sha = fixture(root)
            with patch.object(qc, "COHORT_SHA", "f"*64), self.assertRaisesRegex(ValueError, "cohort_digest"):
                qc.validate_case(case, root)

    def test_absent_acquired_timing_stays_absent(self):
        result = qc.timing(types.SimpleNamespace(StudyDescription="Preop"))
        self.assertIsNone(result["AcquisitionDateTime"])
        self.assertIsNone(result["AcquisitionTime"])


@unittest.skipIf(np is None, "optional numpy/pydicom unavailable")
class GeometryControls(unittest.TestCase):
    def test_per_frame_orientation_or_spacing_conflict_refuses(self):
        for field in ("orientation", "spacing"):
            ds = Dataset(); ds.Modality = "SEG"; ds.SegmentationType = "BINARY"
            ds.SegmentSequence = Sequence([Dataset()])
            shared = Dataset(); orientation = Dataset(); orientation.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
            spacing = Dataset(); spacing.PixelSpacing = [1, 1]
            shared.PlaneOrientationSequence = Sequence([orientation]); shared.PixelMeasuresSequence = Sequence([spacing])
            ds.SharedFunctionalGroupsSequence = Sequence([shared]); frame = Dataset()
            if field == "orientation":
                changed = Dataset(); changed.ImageOrientationPatient = [0, 1, 0, 1, 0, 0]
                frame.PlaneOrientationSequence = Sequence([changed])
            else:
                changed = Dataset(); changed.PixelSpacing = [2, 2]; frame.PixelMeasuresSequence = Sequence([changed])
            ds.PerFrameFunctionalGroupsSequence = Sequence([frame])
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "SEG_per_frame"):
                qc.geometry([ds], None)

    def test_header_case_sources_and_cross_frame_refs_checked_before_decode(self):
        with tempfile.TemporaryDirectory() as temporary:
            case, _, _ = fixture(Path(temporary).resolve())
            rows = []
            for source in case["series"]:
                row = {"kind": source["kind"], "source_objects": source["objects"], "series_instance_uid": source["SeriesInstanceUID"],
                       "modality": source["Modality"], "source_description": source["source_description"], "sop_instance_uids": ["2.25.1"],
                       "geometry": {"frame_of_reference_uid": "2.25.99", "affine_xyz_to_ras_mm": np.eye(4).tolist()},
                       "ancestry": {"referenced_sop_instance_uids": []}}
                rows.append(row)
            h = {"case_sha256": "a"*64, "status": "header_geometry_and_ancestry_projected", "reused_converter_sha256": qc.CORE_SHA, "series": rows}
            self.assertEqual(len(qc.validate_header_binding(h, case, "a"*64)), 4)
            for mutation in ("case", "objects", "frame", "refs"):
                altered = copy.deepcopy(h)
                if mutation == "case": altered["case_sha256"] = "b"*64
                elif mutation == "objects": altered["series"][0]["source_objects"] = []
                elif mutation == "frame": altered["series"][1]["geometry"]["frame_of_reference_uid"] = "2.25.9"
                else: altered["series"][1]["ancestry"]["referenced_sop_instance_uids"] = ["2.25.42"]
                with self.subTest(mutation=mutation), self.assertRaises(ValueError): qc.validate_header_binding(altered, case, "a"*64)

    def test_integer_reindex_retains_error_and_unknown_domain(self):
        affine = np.eye(4); affine[:3, 3] = [1.0001, 1, 1]
        offset, record = qc.integer_map(affine, (2, 2, 2), np.eye(4))
        self.assertAlmostEqual(record["maximum_world_corner_difference_mm"], .0001)
        values, domain = qc.place(np.zeros((2, 2, 2), np.uint8), offset, (4, 4, 4))
        self.assertEqual(int(domain.sum()), 8); self.assertEqual(int(values.sum()), 0)
        self.assertEqual(domain[0, 0, 0], 0)
        affine[0, 3] += .002
        with self.assertRaises(ValueError): qc.integer_map(affine, (2, 2, 2), np.eye(4))

    def test_public_target_conflict_retained_without_editing_inputs(self):
        target = np.array([0, 1, 1, 1], np.uint8); support = np.array([0, 1, 0, 0], np.uint8)
        report = qc.public_target_support_relation(target, support)
        self.assertEqual(report["whole_tumor_positive_outside_public_support"], 2)
        self.assertEqual(report["outside_fraction"], 2/3)
        self.assertFalse(report["target_clipped"]); np.testing.assert_equal(target, [0, 1, 1, 1])
        np.testing.assert_equal(support, [0, 1, 0, 0])

    def test_raw_stored_bits_remain_independent_of_decoder_cache(self):
        ds = Dataset(); ds.file_meta = FileMetaDataset(); ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds.Rows = 1; ds.Columns = 4; ds.BitsAllocated = 16; ds.BitsStored = 12; ds.HighBit = 11
        ds.PixelRepresentation = 1; ds.SamplesPerPixel = 1; ds.PhotometricInterpretation = "MONOCHROME2"
        ds.PixelData = np.array([0, 2047, 2048, 4095], dtype="<u2").tobytes()
        np.testing.assert_equal(qc.raw_mr_samples(ds), [[0, 2047, -2048, -1]])


if __name__ == "__main__": unittest.main()
