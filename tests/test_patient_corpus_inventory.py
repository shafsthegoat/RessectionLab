"""Inventory safety and independent physical-volume accounting controls."""
from io import BytesIO
import importlib.util
import json
from pathlib import Path
from unittest.mock import patch
import zipfile

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location("patient_corpus_inventory", Path(__file__).resolve().parents[1] / "scripts/patient_corpus_inventory.py")
corpus = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(corpus)


def bundle(tmp_path, *, patient="BTC-sub-PAT05", frame="RAS+", synthetic=False, bad_hash=False):
    path = tmp_path / (patient + ".ressectionlab")
    first = np.zeros((3, 4, 2), dtype=bool)
    first[1, 2, 0] = first[1, 2, 1] = True
    second = first.copy()  # overlapping compartments must count each voxel once
    affine = np.array([[0, -2., 0, 10.], [3., 0, 0, 20.], [0, 0, 4., 30.], [0, 0, 0, 1.]])
    output = BytesIO()
    np.savez_compressed(output, affine=affine, source1=first, source2=second,
                        estimated=np.ones_like(first))
    payload = output.getvalue()
    manifest = {"case_id": patient, "frame": frame, "metadata": {"is_synthetic": synthetic,
                 "anatomy_layers": {"one": {"kind": "tumor", "source_role": "imported_source_annotation"},
                                    "two": {"kind": "tumor", "source_role": "imported_source_annotation"},
                                    "inferred": {"kind": "tumor", "source_role": "cached_model_inference"}}},
                "array_index": {"source_compartments": {"one": "source1", "two": "source2", "inferred": "estimated"}},
                "array_sha256": "wrong" if bad_hash else corpus.sha256_bytes(payload)}
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("manifest.json", json.dumps(manifest))
        archive.writestr("arrays.npz", payload)
    return path


def test_known_aliases_do_not_create_people():
    assert corpus.canonical_patient("UCSF-PDGM-0004") == corpus.canonical_patient("UCSF-PDGM-004")
    assert corpus.canonical_patient("BTC-ds001226-sub-PAT05-preop") == "BTC:sub-PAT05"
    assert corpus.canonical_patient("ReMIND-001") == "ReMIND:001"
    assert corpus.canonical_patient("Lausanne-sub-000-ses-20110101") == "Lausanne:sub-000"
    assert corpus.canonical_patient("anonymous-workspace") is None


def test_protected_bundle_never_opened(tmp_path):
    with patch.object(corpus.zipfile, "ZipFile", side_effect=AssertionError("opened protected bundle")):
        result = corpus.read_bundle_manifest(tmp_path / "missing.zip", "BTC:sub-PAT29", {"BTC:sub-PAT29": "UNOPENED_TRANSFER"})
        assert result["status"] == "protected_not_opened"
        with pytest.raises(ValueError, match="Protected"):
            corpus.measure_tumor_geometry(tmp_path / "missing.zip", "BTC:sub-PAT29", {"BTC:sub-PAT29": "UNOPENED_TRANSFER"})


def test_protected_source_not_even_statted(tmp_path):
    inventory = corpus.Inventory(tmp_path, tmp_path, tmp_path)
    inventory.role("ReMIND:013", "SELECT")
    with patch.object(Path, "stat", side_effect=AssertionError("protected stat")):
        inventory.source("ReMIND:013", tmp_path / "image.dcm", {"bytes": 5, "sha256": "recorded"}, evidence="receipt", reported_verified=True)
    assert inventory.sources[0]["presence"] == "protected_not_inspected"
    assert inventory.report()["materialized_source_patient_groups"] == 1


def test_role_conflict_fails_closed(tmp_path):
    inventory = corpus.Inventory(tmp_path, tmp_path, tmp_path)
    inventory.role("Lausanne:sub-000", "TRAIN")
    with pytest.raises(ValueError, match="Conflicting"):
        inventory.role("Lausanne:sub-000", "MEASUREMENT_EVAL")


def test_measurement_uses_union_affine_and_excludes_model_mask(tmp_path):
    path = bundle(tmp_path)
    result = corpus.measure_tumor_geometry(path, "BTC:sub-PAT05", {"BTC:sub-PAT05": "TRAIN"})
    assert result["positive_voxels"] == 2
    assert result["source_label_union_volume_ml"] == pytest.approx(0.048)
    assert result["centroid_ras_mm"] == pytest.approx([6., 23., 32.])
    assert result["anatomical_region"] == result["hemisphere"] == "UNKNOWN"
    assert "inferred" not in result["label_names"]


def test_lps_coordinates_are_transformed(tmp_path):
    path = bundle(tmp_path, frame="LPS+")
    result = corpus.measure_tumor_geometry(path, "BTC:sub-PAT05", {"BTC:sub-PAT05": "TRAIN"})
    assert result["centroid_ras_mm"] == pytest.approx([-6., -23., 32.])


def test_geometry_rejects_wrong_identity_hash_or_synthetic(tmp_path):
    path = bundle(tmp_path)
    with pytest.raises(ValueError, match="identity mismatch"):
        corpus.measure_tumor_geometry(path, "BTC:sub-PAT16", {"BTC:sub-PAT16": "TRAIN"})
    path = bundle(tmp_path, bad_hash=True)
    with pytest.raises(ValueError, match="hash mismatch"):
        corpus.measure_tumor_geometry(path, "BTC:sub-PAT05", {"BTC:sub-PAT05": "TRAIN"})
    path = bundle(tmp_path, synthetic=True)
    with pytest.raises(ValueError, match="Synthetic"):
        corpus.measure_tumor_geometry(path, "BTC:sub-PAT05", {"BTC:sub-PAT05": "TRAIN"})


def test_copies_and_virtual_descendants_are_not_new_people(tmp_path):
    inventory = corpus.Inventory(tmp_path, tmp_path, tmp_path)
    inventory.role("BTC:sub-PAT05", "TRAIN")
    path = bundle(tmp_path)
    row = corpus.read_bundle_manifest(path, "BTC:sub-PAT05", inventory.roles)
    inventory.bundles.extend([row, dict(row), {**row, "is_synthetic": True}])
    report = inventory.report()
    assert report["inspected_real_bundle_patient_groups"] == 1
    assert report["virtual_bundle_representations_in_inspected_roots"] == 1
    assert report["virtual_patient_count"] == 0


def test_metadata_or_size_mismatch_alone_not_acquired_person(tmp_path):
    inventory = corpus.Inventory(tmp_path, tmp_path, tmp_path)
    inventory.role("BTC:sub-PAT05", "TRAIN")
    file = tmp_path / "metadata.json"
    file.write_text("{}")
    inventory.source("BTC:sub-PAT05", file, {"bytes": 2}, evidence="metadata")
    assert inventory.report()["materialized_source_patient_groups"] == 0
    image = tmp_path / "image.nii.gz"
    image.write_bytes(b"incorrect")
    inventory.source("BTC:sub-PAT05", image, {"bytes": 20}, evidence="receipt", reported_verified=True)
    assert inventory.report()["materialized_source_patient_groups"] == 0


def test_non_tumor_compartments_do_not_imply_tumor(tmp_path):
    path = tmp_path / 'SynthRAD2023-1BA336.ressectionlab'
    manifest = {'case_id': 'SynthRAD2023-1BA336', 'metadata': {'is_synthetic': False,
                'anatomy_layers': {'skull': {'kind': 'skull'}, 'brain': {'kind': 'brain'}}},
                'array_index': {'source_compartments': {'skull': 'mask'}}}
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr('manifest.json', json.dumps(manifest))
    row = corpus.read_bundle_manifest(path, 'SynthRAD2023:1BA336', {'SynthRAD2023:1BA336': 'TRAIN'})
    assert 'tumor' in row['missing_anatomy']
    assert 'tumor_annotation_or_estimate' not in row['available_layer_kinds']
