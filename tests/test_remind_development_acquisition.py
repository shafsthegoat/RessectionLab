"""Transport and coordinate controls, not simulated patients or model training."""
from copy import deepcopy
import hashlib
import importlib.util
import io
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


acquisition = load_script("acquire_remind_development")


def test_official_manifest_is_bounded_and_development_only():
    raw = (ROOT / "manifests/experiments/remind-001-structural-development-v1.json").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == acquisition.MANIFEST_SHA256
    manifest = json.loads(raw)
    objects = acquisition.validate_manifest(manifest)
    assert len(objects) == 386 and sum(o["bytes"] for o in objects) == 59_163_552
    assert manifest["role"] == "development_annotation_assisted_geometry"
    assert not manifest["molecular_or_outcome_fields_as_inputs"]


@pytest.mark.parametrize("mutation", ["other_patient", "intraoperative", "foreign_host", "path_traversal", "duplicate_object"])
def test_modified_identity_is_rejected(mutation):
    manifest = json.loads((ROOT / "manifests/experiments/remind-001-structural-development-v1.json").read_bytes())
    series = manifest["series"][1]
    if mutation == "other_patient":
        manifest["patient_id"] = "ReMIND-002"
    elif mutation == "intraoperative":
        series["StudyDescription"] = "Intraop"
    elif mutation == "foreign_host":
        series["objects"][0]["url"] = "https://example.org/test.dcm"
    elif mutation == "path_traversal":
        series["objects"][0]["key"] = "../test.dcm"
    else:
        series["objects"][1] = deepcopy(series["objects"][0])
    with pytest.raises(ValueError):
        acquisition.validate_manifest(manifest)


def test_invalid_cached_bytes_are_not_accepted_or_replaced(tmp_path):
    obj = {"key": "uuid/file.dcm", "url": "unused", "bytes": 4, "etag_md5": hashlib.md5(b"good").hexdigest()}
    path = tmp_path / obj["key"]
    path.parent.mkdir()
    path.write_bytes(b"bad!")
    with pytest.raises(ValueError, match="MD5"):
        acquisition.acquire_one(obj, tmp_path)
    assert path.read_bytes() == b"bad!"


@pytest.mark.parametrize("failure", ["etag", "length", "url", "body"])
def test_mismatched_response_is_never_published(tmp_path, monkeypatch, failure):
    obj = {"key": "uuid/file.dcm", "url": "https://idc-open-data.s3.amazonaws.com/uuid/file.dcm",
           "bytes": 4, "etag_md5": hashlib.md5(b"good").hexdigest()}
    class Response(io.BytesIO):
        status = 200
        url = obj["url"] if failure != "url" else "https://example.org/other.dcm"
        headers = {"Content-Length": "4" if failure != "length" else "5",
                   "ETag": '"' + (obj["etag_md5"] if failure != "etag" else "0" * 32) + '"'}
    class Opener:
        def open(self, request, timeout):
            assert request.get_header("If-match") == '"' + obj["etag_md5"] + '"'
            return Response(b"good" if failure != "body" else b"bad!")
    monkeypatch.setattr(acquisition.urllib.request, "build_opener", lambda *a: Opener())
    with pytest.raises(ValueError):
        acquisition.acquire_one(obj, tmp_path)
    assert not (tmp_path / obj["key"]).exists()


@pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
def test_redirect_is_rejected_before_followup(status):
    with pytest.raises(ValueError, match="Redirect"):
        acquisition.RejectRedirects().redirect_request(None, None, status, "", {}, "https://other.invalid")


@pytest.fixture
def converter():
    pytest.importorskip("pydicom", reason="Run in the isolated IDC acquisition environment for DICOM controls")
    return load_script("convert_remind_development")


def test_dicom_row_column_lps_to_ras_mapping_with_permuted_planes(converter):
    order, affine, residual = converter.regular_affine([0, 1, 0, 0, 0, 1], [2, 3],
                                                      [[14, 20, 30], [10, 20, 30], [12, 20, 30]])
    assert order.tolist() == [1, 2, 0]
    # Column 4 moves 4*3 mm along LPS Y; row 5 moves 5*2 mm along LPS Z.
    np.testing.assert_allclose(affine @ [4, 5, 1, 1], [-12, -32, 40, 1], atol=1e-12)
    assert residual < 1e-12


def test_realistic_decimal_rounding_is_fit_without_relaxing_tolerance(converter):
    p = [[0, 0, 0.0006], [0, 0, 1], [0, 0, 2], [0, 0, 2.9994]]
    _, _, residual = converter.regular_affine([1, 0, 0, 0, 1, 0], [1, 1], p)
    assert residual < 1e-3


@pytest.mark.parametrize("positions", [[[0, 0, 0], [0, 0, 1], [0, 0, 2.02]],
                                      [[0, 0, 0], [0, 0, 0], [0, 0, 0]]])
def test_nonuniform_or_duplicate_planes_are_rejected(converter, positions):
    with pytest.raises(ValueError):
        converter.regular_affine([1, 0, 0, 0, 1, 0], [1, 1], positions)


def test_nonorthogonal_orientation_is_rejected(converter):
    with pytest.raises(ValueError, match="orientation"):
        converter.regular_affine([1, 0, 0, 0.1, 1, 0], [1, 1], [[0, 0, 0], [0, 0, 1]])
