"""Generated public inputs and a stub constructor: no patient or native runs."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
from resectionlab.patient_planning_admission import PROTOCOL_FIELDS

STAGE = Path(__file__).resolve().parents[1]
MODULE = STAGE / "src/resectionlab/public_patient_factory.py"
factory = types.ModuleType("public_factory_under_test"); factory.__file__ = str(MODULE)
exec(compile(MODULE.read_bytes(), str(MODULE), "exec"), factory.__dict__)


def save(path, obj):
    raw = json.dumps(obj).encode(); path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def fixture(root, subject):
    group = "ReMIND:"+subject.split("-")[-1]
    cohort = json.dumps({"members": [{"subject": subject, "patient_group": group, "role": "TRAIN"}]}).encode()
    cohort_sha = hashlib.sha256(cohort).hexdigest()
    image = np.arange(125, dtype=np.float32).reshape(5, 5, 5)
    support = np.zeros((5, 5, 5), np.uint8); support[1:4, 1:4, 1:4] = 1
    target = np.zeros_like(support); target[3:5, 2, 2] = 1
    domain = np.ones_like(support)
    files = {}
    for key, data in zip(factory.ARRAY_KEYS, (image, support, target, domain)):
        path = root/(key+".npy"); np.save(path, data, allow_pickle=False)
        files[key] = {"path": str(path), "sha256": factory.sha(path), "bytes": path.stat().st_size, "dtype": str(data.dtype)}
    manifest = {"patient_id": subject, "patient_group": group, "role": "TRAIN", "input_files": files,
        "private_evaluation_files_included": False, "task_condition": "PARTIAL_TARGET_PROGRESS",
        "public_support_domain_fully_covered": True, "shape_xyz": [5, 5, 5], "affine_ras_mm": np.eye(4).tolist(),
        "source_MR_crop_affine_ras_mm": np.eye(4).tolist(), "public_label_resampling": {},
        "source_bindings": {"cohort_sha256": cohort_sha, "saved_array_review_sha256": "a"*64},
        "reindex_policy": {"source_MR_samples_preserved": True, "original_MR_affine_overwritten": False},
        "public_target_support_consistency": {"whole_tumor_positive_voxels": 2, "whole_tumor_positive_outside_supplied_support": 1}}
    return manifest, cohort


class Controls(unittest.TestCase):
    def test_all_four_train_cases_bind_without_native_construction(self):
        from resectionlab.core import semantic_digest
        from resectionlab.patient_planning_learning import PREFLIGHT_PROTOCOL
        learning_hash = semantic_digest(PREFLIGHT_PROTOCOL)
        self.assertTrue(learning_hash.startswith("sha256:"))
        for subject in factory.TRAIN_SUBJECTS:
            with self.subTest(subject=subject), tempfile.TemporaryDirectory(dir=STAGE) as name:
                root = Path(name); manifest, cohort = fixture(root, subject)
                path = root/"manifest.json"; sha = save(path, manifest); output = root/"out"; output.mkdir()
                captured = {}
                def constructor(image, support, target, affine, access, tools, **kwargs):
                    captured.update(kwargs)
                    return types.SimpleNamespace(structural_intensity=image, observed_support=support, nominal_target=target,
                        affine_ras_mm=affine, source_hash="generated-stub-source", _supplied_goal_extent={}, _grid_record={}, _normalization_record={})
                limits = {"max_steps": 6, "max_optimizer_updates": 0, "max_native_previews": 1, "max_policy_forwards": 1,
                    "worker_seconds": 1, "memory_bytes": 1, "threads": 1, "search": {"max_calls": 1, "beam_width": 1, "seconds": 1}}
                with patch.object(factory, "COHORT_SHA256", hashlib.sha256(cohort).hexdigest()), patch("resectionlab.native_spatial_task.NativeSpatialCase", constructor):
                    source, binding, qc, protocol = factory.prepare_public_source(output, {"limits": limits}, "b"*64, None,
                        lambda *a, **kw: None, public_manifest_path=path, public_manifest_sha256=sha, cohort_bytes=cohort,
                        learning_protocol_hash=learning_hash)
                    replay_output = root/"reconstructed"; replay_output.mkdir()
                    _, replay_binding, replay_qc, replay_protocol = factory.prepare_public_source(replay_output,
                        {"limits": limits, "baseline_public_source_hash": source.source_hash}, "b"*64, protocol,
                        lambda *a, **kw: None, public_manifest_path=path, public_manifest_sha256=sha, cohort_bytes=cohort)
                    self.assertEqual(replay_protocol, protocol)
                    self.assertEqual(replay_binding, binding); self.assertEqual(replay_qc, qc)
                self.assertEqual(binding["subject"], subject); self.assertEqual(qc["subject"], subject)
                self.assertEqual(set(protocol), PROTOCOL_FIELDS); self.assertEqual(protocol["max_steps"], 6)
                self.assertEqual(protocol["learning_protocol_hash"], learning_hash)
                self.assertEqual(captured["crop_shape"], (64, 64, 64)); self.assertEqual(captured["proposal_mode"], "nominal_cavity_v1")
                self.assertEqual(captured["native_grid_reconciliation"], "orthogonal_roundoff_1e-6mm")
                self.assertEqual(int(source.nominal_target.sum()), 2)
                derivation = json.loads((output/"public-task-derivation.json").read_bytes())
                self.assertEqual(derivation["unsupported_target_positive_voxels"], 1)
                self.assertEqual(derivation["task_condition"], "PARTIAL_TARGET_PROGRESS")

    def test_metadata_rejects_role_extra_input_coverage_and_task_changes(self):
        with tempfile.TemporaryDirectory(dir=STAGE) as name:
            root = Path(name); manifest, cohort = fixture(root, "ReMIND-010"); path = root/"manifest.json"
            for mutation in ("role", "extra", "coverage", "condition", "frame_provenance"):
                altered = copy.deepcopy(manifest)
                if mutation == "role": altered["role"] = "SELECT"
                elif mutation == "extra": altered["input_files"]["hidden_reference"] = {}
                elif mutation == "coverage": altered["public_support_domain_fully_covered"] = False
                elif mutation == "condition": altered["task_condition"] = "COMPLETE_TARGET"
                else: altered["reindex_policy"]["original_MR_affine_overwritten"] = True
                sha = save(path, altered)
                with self.subTest(mutation=mutation), patch.object(factory, "COHORT_SHA256", hashlib.sha256(cohort).hexdigest()), self.assertRaises(ValueError):
                    factory.load_public_manifest(path, sha, cohort)

    def test_manifest_hash_and_original_cohort_bytes_required(self):
        with tempfile.TemporaryDirectory(dir=STAGE) as name:
            root = Path(name); manifest, cohort = fixture(root, "ReMIND-010"); path = root/"manifest.json"; sha = save(path, manifest)
            with self.assertRaisesRegex(ValueError, "manifest changed"): factory.load_public_manifest(path, "0"*64, cohort)
            with self.assertRaisesRegex(ValueError, "cohort bytes"): factory.load_public_manifest(path, sha, cohort)

    def test_unsupported_target_stays_in_denominator_and_domain_is_enforced(self):
        support = np.array([1, 0, 0], np.uint8); target = np.array([1, 1, 0], np.uint8); domain = np.array([1, 1, 0], np.uint8)
        manifest = {"public_target_support_consistency": {"whole_tumor_positive_voxels": 2, "whole_tumor_positive_outside_supplied_support": 1}}
        points, unsupported = factory.check_public_labels(support, target, domain, manifest)
        self.assertEqual(len(points), 2); self.assertEqual(unsupported, 1); self.assertEqual(int(target.sum()), 2)
        domain[1] = 0
        with self.assertRaisesRegex(ValueError, "domain changed"): factory.check_public_labels(support, target, domain, manifest)

    def test_empty_public_support_has_explicit_refusal(self):
        with self.assertRaisesRegex(ValueError, "support is empty"):
            factory.check_public_labels(np.zeros(3, np.uint8), np.ones(3, np.uint8), np.ones(3, np.uint8), {})


if __name__ == "__main__": unittest.main()
