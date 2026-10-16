"""Generated controls: no real image/model execution or protected body access."""
import gzip
from copy import deepcopy
import importlib.util
import io
import json
from pathlib import Path
import zlib
import nibabel as nib
import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("synthrad_extension", Path(__file__).resolve().parents[1]/"scripts/poc_synthrad_extension.py")
runner = importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)


def fixtures():
    cohort = json.loads((runner.ROOT/"manifests/synthrad-component-cohort-v1.json").read_text())
    pairs = json.loads((runner.ROOT/"artifacts/synthrad-first-pair-qc-preparation-v1/brain-pair-map.json").read_text())["pairs"]
    return cohort, pairs


def test_exact_two_people_follow_frozen_roles_and_hash_order():
    cohort, pairs = fixtures()
    assert [p["subject_id"] for p in runner.select_pairs(cohort, pairs)] == ["1BB033", "1BC077"]
    next(row for row in cohort["members"] if row["subject_id"] == "1BB033")["role"] = "SELECT"
    with pytest.raises(ValueError, match="role/rank"):
        runner.select_pairs(cohort, pairs)


def test_relabelled_synthetic_or_outline_member_cannot_enter_pair():
    cohort, pairs = fixtures()
    row = next(row for row in pairs if row["subject_id"] == "1BB033")
    row["candidate_acquired_members"][0]["name"] = "Task1/brain/1BB033/synthetic_ct.nii.gz"
    with pytest.raises(ValueError, match="member names"):
        runner.select_pairs(cohort, pairs)


def compressed_row(payload, name="Task1/brain/1BB033/ct.nii.gz"):
    compressor = zlib.compressobj(wbits=-15)
    data = compressor.compress(payload)+compressor.flush()
    return data, {"name": name, "compressed_bytes": len(data), "zip_entry_bytes": len(payload), "crc32": f"{zlib.crc32(payload):08x}"}


def test_member_rejected_before_body_read():
    class Forbidden:
        def seek(self, *_): pytest.fail("Disallowed body was accessed")
    _, row = compressed_row(b"generated")
    with pytest.raises(ValueError, match="allowlist"):
        list(runner.member_bytes(Forbidden(), row, 0, set(), lambda: None))


def test_strict_deflate_rejects_size_lie_trailing_data_and_bad_crc():
    data, row = compressed_row(b"generated source "*500)
    allowed = {row["name"]}
    assert b"".join(runner.member_bytes(io.BytesIO(data), row, 0, allowed, lambda: None)) == b"generated source "*500
    with pytest.raises(ValueError, match="Trailing"):
        list(runner.member_bytes(io.BytesIO(data+b"x"), {**row, "compressed_bytes": len(data)+1}, 0, allowed, lambda: None))
    with pytest.raises(ValueError, match="length"):
        list(runner.member_bytes(io.BytesIO(data), {**row, "zip_entry_bytes": 2}, 0, allowed, lambda: None))
    with pytest.raises(ValueError, match="CRC"):
        list(runner.member_bytes(io.BytesIO(data), {**row, "crc32": "00000000"}, 0, allowed, lambda: None))


def save_fixture(path, *, nonfinite=False, trailing=False):
    values = np.arange(9*10*11, dtype=np.float32).reshape((9,10,11))
    if nonfinite: values[5,6,7] = np.nan
    image = nib.Nifti1Image(values, np.diag([1.,1.2,1.4,1.]))
    image.header.set_xyzt_units("mm"); image.set_qform(image.affine,1); image.set_sform(image.affine,1)
    payload = image.to_bytes()+(b"extra" if trailing else b"")
    path.write_bytes(gzip.compress(payload))
    h = runner.helper()
    return h, {"bytes": path.stat().st_size, "sha256": runner.digest(path), "staged_stat": h.stat_record(path.stat())}


def test_streaming_source_qualification_enforces_finiteness_crc_and_exact_length(tmp_path):
    class NoOpBudget:
        def check(self): pass
    path = tmp_path/"ct.nii.gz"; h, proof = save_fixture(path)
    result = runner.qualify(path, proof, h, NoOpBudget())
    assert result["grid"]["shape"] == [9,10,11]
    assert result["scalar"]["finite_voxels"] == 990
    assert result["clinical_or_training_admission"] is False
    h, proof = save_fixture(path, nonfinite=True)
    with pytest.raises(h.Refusal, match="nonfinite"):
        runner.qualify(path, proof, h, NoOpBudget())
    h, proof = save_fixture(path, trailing=True)
    with pytest.raises(h.Refusal, match="trailing"):
        runner.qualify(path, proof, h, NoOpBudget())
    h, proof = save_fixture(path)
    corrupt = bytearray(path.read_bytes()); corrupt[-8] ^= 1; path.write_bytes(corrupt)
    proof = {**proof, "sha256": runner.digest(path), "staged_stat": h.stat_record(path.stat())}
    with pytest.raises(zlib.error):
        runner.qualify(path, proof, h, NoOpBudget())


def test_declared_expanded_size_rejected_before_voxel_read():
    h = runner.helper(); header = nib.Nifti1Header()
    header.set_data_shape((512,512,513)); header.set_data_dtype(np.float32)
    header["magic"] = b"n+1\0"; header["vox_offset"] = 352
    class HeaderOnly:
        def read(self, size):
            assert size == 348, "Must reject oversized source before any further allocation/read"
            return header.binaryblock
    with pytest.raises(h.BoundExceeded, match="payload_cap"):
        h.parse_header(HeaderOnly(), runner.BOUNDS)


@pytest.mark.parametrize("change", ["role", "hash", "path", "nested_role"])
def test_allowed_id_cannot_rebind_source_or_role_before_payload_access(tmp_path, monkeypatch, change):
    report = {"subject_id": "1BB033", "role": "TRAIN", "training_admitted": False,
        "patient_group": "SynthRAD2023:1BB033", "images": {m: {"path": str(tmp_path/"1BB033"/(m+".nii.gz")),
        "sha256": runner.SOURCE_HASHES["1BB033"][m], "gzip_crc_and_exact_payload_length": "passed"} for m in ("ct", "mr")}}
    source_result = tmp_path/"source-result.json"
    source_result.write_text(json.dumps({"pairs": [report]}))
    digest = runner.digest(source_result)
    monkeypatch.setattr(runner, "OUTPUT", tmp_path)
    monkeypatch.setattr(runner, "SOURCE_RESULT_SHA", digest)
    def no_payload(path, *args):
        assert Path(path) == source_result, "Negative source guard read a patient or model payload"
        return digest
    monkeypatch.setattr(runner, "digest", no_payload)
    supplied = deepcopy(report)
    if change == "role": supplied["role"] = "SELECT"
    elif change == "hash": supplied["images"]["ct"]["sha256"] = "0"*64
    elif change == "path": supplied["images"]["ct"]["path"] = "/protected/ct.nii.gz"
    else: supplied["images"]["ct"]["role"] = "SELECT"
    with pytest.raises(ValueError): runner.bound_source(supplied)


def test_changed_frozen_result_rejected_before_patient_payload(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "OUTPUT", tmp_path)
    (tmp_path/"source-result.json").write_text('{"pairs": []}')
    with pytest.raises(ValueError, match="source-result"):
        runner.bound_source({"subject_id": "1BB033", "role": "TRAIN", "training_admitted": False})


def test_independent_resource_guards_stop_at_each_declared_limit():
    good = {"rss": 1024**3, "available": 2*1024**3, "elapsed": 1, "free_disk": 60*1024**3, "output_bytes": 1}
    assert runner.model_resource_failure(**good) is None
    for field, value, expected in [("rss", 5*1024**3+1, "RSS"), ("available", 1024**3-1, "memory"),
        ("elapsed", 601, "wall"), ("free_disk", 50*1024**3-1, "disk"), ("output_bytes", 512*1024**2+1, "bytes")]:
        assert expected in runner.model_resource_failure(**{**good, field: value})


def test_generated_pair_roundtrip_keeps_both_originals_and_mr_separate(tmp_path, monkeypatch):
    import hashlib
    from resectionlab.poc_models import MODEL_CONFIG
    from resectionlab.workspace_bundle import load_workspace
    folder = tmp_path/"1BB033"; folder.mkdir()
    images = {}
    hashes = {}
    for modality in ("ct", "mr"):
        path = folder/(modality+".nii.gz"); h, proof = save_fixture(path)
        hashes[modality] = proof["sha256"]
        images[modality] = {"path": str(path), "sha256": proof["sha256"], "gzip_crc_and_exact_payload_length": "passed"}
    source = {"subject_id": "1BB033", "patient_group": "SynthRAD2023:1BB033", "role": "TRAIN", "training_admitted": False, "images": images}
    frozen = tmp_path/"source-result.json"; frozen.write_text(json.dumps({"pairs": [source]}))
    monkeypatch.setattr(runner, "OUTPUT", tmp_path)
    monkeypatch.setattr(runner, "SOURCE_RESULT_SHA", runner.digest(frozen))
    monkeypatch.setattr(runner, "SOURCE_HASHES", {"1BB033": hashes})
    weights = tmp_path/"generated-test-weights"; weights.write_bytes(b"generated fixture, never executed")
    monkeypatch.setattr(runner, "WEIGHT", weights); monkeypatch.setattr(runner, "WEIGHT_SHA", runner.digest(weights))
    monkeypatch.setattr(runner, "MODEL_SUPPORT", {})
    model_folder = tmp_path/"model"; model_folder.mkdir()
    request = runner.model_request(source, model_folder)
    assert request["image"] == images["ct"]["path"] and "annotation" not in request and "mr" not in request
    reference = nib.load(images["ct"]["path"])
    values = np.arange(np.prod(reference.shape)).reshape(reference.shape)
    outputs = []
    for name, mask in [("brain", values > 200), ("skull", values < 100)]:
        path = model_folder/(name+"_mask.nii.gz")
        image = nib.Nifti1Image(mask.astype(np.uint8), reference.affine, reference.header.copy())
        nib.save(image, path)
        outputs.append({"path": str(path), "name": path.name, "sha256": runner.digest(path)})
    identity = {"image_sha256": hashes["ct"], "weights": [runner.WEIGHT_SHA], "model_id": "totalsegmentator-skull",
        "parameters": request["parameters"], "device": "mps", "runner_sha256": runner.PINS["src/resectionlab/poc_models.py"]}
    report = {"schema": "resectionlab-poc-inference-v1", "status": "succeeded", "case": "SynthRAD2023-1BB033",
        "model_id": "totalsegmentator-skull", "model": MODEL_CONFIG["totalsegmentator-skull"], "weights": request["weights"],
        "identity": identity, "cache_key": hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:20], "device": "mps",
        "input": {"path": images["ct"]["path"], "sha256": hashes["ct"], "shape": list(reference.shape), "affine_ras_mm": reference.affine.tolist(), "modality": "CT"},
        "outputs": outputs}
    report_path = model_folder/"inference.json"; report_path.write_text(json.dumps(report))
    output = tmp_path/"paired.ressectionlab"
    result = runner.package_pair(source, report_path, output)
    primary, _, restored = load_workspace(output)
    assert result["fusion_permitted"] is False and set(primary.compartments) == {"brain", "skull"}
    assert not primary.critical_evidence and not primary.prior_proposals
    assert primary.metadata["split"]["training_admitted"] is False
    assert len(restored["sources"]) == 1 and restored["sources"][0].modality == "T1"
    assert restored["sources"][0].annotation_kind == "none"
    np.testing.assert_array_equal(restored["sources"][0].case.mri, reference.get_fdata(dtype=np.float32))
    archived = {row.name: row.data for row in restored["originals"]}
    for modality in ("ct", "mr"):
        assert archived[modality+".nii.gz"] == (folder/(modality+".nii.gz")).read_bytes()
