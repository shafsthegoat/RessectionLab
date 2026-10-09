"""Metadata and control tests only: no acquired image/mask is opened or decoded."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import time

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import lausanne_deferred_qc as qc
_INSPECT_ORIGINAL_CONTROL = qc.inspect_original


@pytest.fixture(autouse=True)
def forbid_scientific_io(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Control tests cannot transfer or inspect scientific payloads")
    monkeypatch.setattr(qc.gzip, "open", forbidden)
    monkeypatch.setattr(qc.annotations, "open_without_redirect", forbidden)
    monkeypatch.setattr(qc, "inspect_original", forbidden)
    monkeypatch.setattr(qc.annotations, "inspect_mask", forbidden)
    monkeypatch.setattr(qc, "verify_source_file", forbidden)
    original = Path.open
    def guarded(path, *args, **kwargs):
        if path.name.endswith((".nii", ".nii.gz", ".nii.gz.partial")):
            forbidden()
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", guarded)


@pytest.fixture(scope="module")
def metadata():
    m = json.loads(qc.MANIFEST.read_bytes())
    if not (ROOT / m["basis_queue"]["path"]).exists():
        pytest.skip("Requires preserved real receipt metadata; never create a patient substitute")
    return qc.preflight()


@pytest.mark.parametrize('route', ['legacy', 'explicit_none', 'v2'])
def test_original_recorder_keyword_is_explicit_and_preserves_default(monkeypatch, route):
    import io
    import nibabel as nib
    from resectionlab import imaging
    from resectionlab.nifti_header_records import nifti1_header_record_coded_v2
    h = nib.Nifti1Header()
    h.set_data_shape((2, 2, 2)); h.set_xyzt_units('mm')
    h.set_sform([[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,1.]], code=1)
    h['pixdim'][0] = 0; h['vox_offset'] = 352
    raw = h.binaryblock
    class PrefixOnly(io.BytesIO):
        def read(self, count=-1):
            assert 0 <= count <= 348 and self.tell()+count <= 352, 'No scalar reads permitted'
            return super().read(count)
    monkeypatch.setattr(qc.gzip, 'open', lambda *a, **k: PrefixOnly(raw+b'\0'*4))
    class StopBeforeScalars(Exception): pass
    def geometry_stop(*a, **k): raise StopBeforeScalars()
    monkeypatch.setattr(imaging, 'inspect_nifti', geometry_stop)
    calls = []
    def recorder(payload, sha):
        calls.append((payload, sha))
        return nifti1_header_record_coded_v2(payload, sha)
    bounds = {'scalar_chunk_voxels': 65536, 'max_scalar_workspace_bytes': 8388608,
              'max_voxels': 134217728, 'max_native_payload_bytes': 536870912, 'max_nifti_data_offset': 1048576}
    kwargs = {} if route == 'legacy' else {'header_recorder': None if route == 'explicit_none' else recorder}
    expected = StopBeforeScalars if route == 'v2' else nib.spatialimages.HeaderDataError
    with pytest.raises(expected):
        _INSPECT_ORIGINAL_CONTROL(ROOT/'build/header-control-not-an-image', 'a'*64, bounds, time.monotonic()+10, **kwargs)
    assert calls == ([(raw, 'a'*64)] if route == 'v2' else [])


def test_invalid_recorder_rejected_before_open(monkeypatch):
    with pytest.raises(TypeError, match='header_recorder'):
        _INSPECT_ORIGINAL_CONTROL(ROOT/'build/no-image', 'a'*64, {}, time.monotonic()+10, header_recorder=False)


def test_exact_scope_and_retained_failures(metadata):
    m, _, _ = metadata
    assert len(m["originals"]) == 26 and len(m["reference_followups"]) == 27
    assert len({(r["subject"], r["session"]) for r in m["originals"]}) == 13
    assert sum(r["reference_mode"] == "separate_original_review" for r in m["reference_followups"]) == 15
    conflicts = [r for r in m["preserved_failures"] if r["next_qc_stage"] == "known_frame_conflict_review"]
    assert {r["subject"] for r in conflicts} == {"sub-253", "sub-410", "sub-411", "sub-418", "sub-424", "sub-473"}
    grids = [r for r in m["preserved_failures"] if r["next_qc_stage"] == "known_mask_grid_conflict_review"]
    assert len(grids) == 1 and grids[0]["subject"] == "sub-454"
    assert len(m["preserved_failures"]) == 11
    assert all(r["annotation_available_at"] is None and r["review_available_at"] is None
               and r["background_semantics"] == "unknown" for r in m["reference_followups"])
    assert qc.CLAIMS["training_admitted"] is False and qc.CLAIMS["anatomy_qc"] == "not_run"


def test_final_transfer_chain_accepts_actual_metadata_only(metadata):
    m, _, _ = metadata
    index = json.loads(qc.proof_bytes(m["original_index"]))
    result = qc.transfer_contract(m["originals"][0], index, m)
    assert result["status"] == "byte_verified"
    assert result["header_qc"] == "not_run"


@pytest.mark.parametrize("change", ["failed_parent", "wrong_result_hash", "wrong_source", "wrong_role",
                                    "wrong_intent", "claim", "wrong_declaration", "wrong_bytes", "wrong_sha"])
def test_actual_transfer_contract_mutations_refused(metadata, monkeypatch, change):
    m, _, _ = metadata
    row = copy.deepcopy(m["originals"][0])
    records = {k: json.loads(qc.proof_bytes(p)) for k, p in row["transfer"].items()}
    if change == "failed_parent":
        records["outcome"]["status"] = "transport_deferred"
    elif change == "wrong_result_hash":
        records["outcome"]["result_sha256"] = "0" * 64
    elif change == "wrong_source":
        records["result"]["source"]["session"] = "wrong-session"
    elif change == "wrong_role":
        records["result"]["source"]["role"] = "SELECT"
    elif change == "wrong_intent":
        records["result"]["intent_sha256"] = "0" * 64
    elif change == "claim":
        records["outcome"]["training_admitted"] = True
    elif change == "wrong_declaration":
        records["intent"]["declaration_sha256"] = "0" * 64
    elif change == "wrong_bytes":
        records["result"]["verified_bytes"] += 1
    else:
        records["outcome"]["sha256"] = "0" * 64
    altered = {row["transfer"][k]["path"]: qc.encode(v) for k, v in records.items()}
    real = qc.proof_bytes
    monkeypatch.setattr(qc, "proof_bytes", lambda p: altered[p["path"]] if p["path"] in altered else real(p))
    with pytest.raises(qc.Refusal):
        qc.transfer_contract(row, json.loads(real(m["original_index"])), m)


def test_changed_retained_source_snapshot_refused(metadata, monkeypatch):
    m, _, _ = metadata
    read = qc.read_metadata
    name = next(iter(m["protected_live_sources"]))
    monkeypatch.setattr(qc, "read_metadata", lambda p: b"modified source" if str(p).endswith("snapshot/" + name) else read(p))
    with pytest.raises(qc.Refusal, match="snapshot changed"):
        qc.transfer_contract(m["originals"][0], json.loads(qc.proof_bytes(m["original_index"])), m)


@pytest.mark.parametrize("field,value", [("review_available_at", "invented-source-time"),
                                        ("annotation_subtype_status", "invented-contour"),
                                        ("role", "EVAL")])
def test_preserved_annotation_identity_and_timing_cannot_change(metadata, field, value):
    m, am, _ = metadata
    row = copy.deepcopy(m["reference_followups"][0])
    row[field] = value
    a = next(a for a in am["records"] if a["path"] == row["path"])
    with pytest.raises(qc.Refusal):
        qc.prior_mask_contract(row, a)


def test_raw_header_binding_rejects_changed_source_or_header(metadata):
    m, am, _ = metadata
    row = m["reference_followups"][0]
    a = next(a for a in am["records"] if a["path"] == row["path"])
    old = qc.prior_mask_contract(row, a)
    grid = old["content_qc"]["raw_grid"]
    assert len(qc.validate_raw_grid(grid, old["acquisition"]["sha256"])) == 348
    with pytest.raises(qc.Refusal):
        qc.validate_raw_grid(grid, "0" * 64)
    changed = copy.deepcopy(grid)
    changed["header_sha256"] = "0" * 64
    with pytest.raises(qc.Refusal):
        qc.validate_raw_grid(changed, old["acquisition"]["sha256"])


def test_only_explicit_stage_whitelist_allowed(metadata):
    m, _, _ = metadata
    mask = m["reference_followups"][0]["path"]
    image = m["originals"][0]["path"]
    assert len(qc.selected_rows(m, "originals", image)) == 1
    assert len(qc.selected_rows(m, "references", mask)) == 1
    with pytest.raises(qc.Refusal):
        qc.selected_rows(m, "originals", mask)
    with pytest.raises(qc.Refusal):
        qc.selected_rows(m, "references", image)


@pytest.mark.parametrize("shape,itemsize", [([0, 1, 1], 4), ([1, 1], 4), ([1, 1, 1], 16), ([True, 1, 1], 4), ([2**30, 1, 1], 8)])
def test_scalar_allocation_controls_reject_invalid_or_oversize(metadata, shape, itemsize):
    with pytest.raises(qc.Refusal):
        qc.scalar_budget(shape, itemsize, metadata[0]["bounds"])


def test_scalar_workspace_bound_is_chunked_not_image_sized(metadata):
    b = metadata[0]["bounds"]
    # Arithmetic resource control only, not generated image dimensions/evidence.
    r = qc.scalar_budget([b["max_voxels"], 1, 1], 8, b)
    assert r["scalar_workspace_bytes_bound"] <= b["max_scalar_workspace_bytes"]
    assert r["chunk_voxels"] < r["voxels"]


def test_missing_explicit_review_never_falls_back_to_transfer(metadata):
    m, _, _ = metadata
    row = m["originals"][0]
    with pytest.raises(qc.Refusal, match="separate immutable namespace"):
        qc.completed_original_review(ROOT / row["transfer"]["result"]["path"], row, m)


@pytest.mark.parametrize("change", ["role", "session", "other_modality", "original_role"])
def test_separate_reference_rejects_cross_role_session_or_modality_before_image_access(metadata, monkeypatch, change):
    m, am, sessions = metadata
    target = next(r for r in m["reference_followups"] if r["reference_mode"] == "separate_original_review")
    row = copy.deepcopy(next(r for r in am["records"] if r["path"] == target["path"]))
    original = copy.deepcopy(sessions[(row["subject"], row["session"])])
    if change == "role":
        row["role"] = "SELECT"
    elif change == "session":
        row["session"] = "ses-19000101"
    elif change == "other_modality":
        row["original_reference"] = next(f for f in original["files"] if f["path"].endswith("_T1w.nii.gz"))
    else:
        original["role"] = "EVAL"
    monkeypatch.setattr(qc, "preflight", lambda: metadata)
    with pytest.raises(qc.Refusal, match="exact same-session original TOF"):
        qc.separately_reviewed_reference(row, original, am, ROOT / "build/control-unused", time.monotonic() + 1,
                                        ROOT / "build/no-scientific-review.json")


def test_no_execution_flag_is_no_payload_operation():
    with pytest.raises(SystemExit) as error:
        qc.main(["batch", "--stage", "originals", "--run-id", "control"])
    assert error.value.code == 2


def test_claim_types_cannot_use_numeric_false_or_omit_fields():
    assert qc.claims_match(dict(qc.CLAIMS), qc.CLAIMS)
    value = dict(qc.CLAIMS)
    value["training_admitted"] = 0
    assert not qc.claims_match(value, qc.CLAIMS)
    value = dict(qc.CLAIMS)
    del value["anatomy_qc"]
    assert not qc.claims_match(value, qc.CLAIMS)


def test_reference_followup_reuses_prior_content_without_mask_decode(metadata, monkeypatch, tmp_path):
    m, am, sessions = metadata
    row = next(r for r in m["reference_followups"] if r["reference_mode"] == "legacy_session_receipt")
    calls = []
    prior = json.loads(qc.proof_bytes(row["receipt"]))
    monkeypatch.setattr(qc, "verify_source_file", lambda p, e, **kw: calls.append((str(p), kw)) or prior["acquisition"]["sha256"])
    monkeypatch.setattr(qc.annotations, "verify_reference", lambda *a, **kw: {"status": "failed_referenced_tof_qc"})
    monkeypatch.setattr(qc, "atomic_preserve", lambda *a, **kw: None)
    r = qc.followup_reference(row, m, am, sessions, tmp_path, time.monotonic() + 1, {})
    assert r["status"] == "review_failed" and r["mask_scalars_decoded"] is False
    assert r["content_qc"]["status"] == "reused_source_bound_prior_pass"
    assert r["grid_qc"]["status"] == "deferred_reference_not_passed"
    assert len(calls) == 2 and all(p.endswith(row["path"]) for p, _ in calls)


def test_preservation_never_overwrites_a_different_receipt(tmp_path):
    p = tmp_path / "control.json"
    qc.atomic_preserve(p, b"control record one\n")
    with pytest.raises(ValueError, match="Preserve existing"):
        qc.atomic_preserve(p, b"control record two\n")
    assert p.read_bytes() == b"control record one\n"


@pytest.mark.parametrize("parent_status,child_status,exit_code", [
    ("completed", "review_failed", 0), ("timeout", "review_passed", None),
])
def test_supervised_control_outcome_never_promotes_a_failure(metadata, monkeypatch, parent_status, child_status, exit_code):
    """Only process-state controls; no positive scientific QC fields are made."""
    m, am, sessions = metadata
    row = m["originals"][0]
    with tempfile.TemporaryDirectory(prefix="deferred-review-controls-", dir=ROOT / "build") as folder:
        cache = Path(folder)
        monkeypatch.setattr(qc, "CACHE", cache)
        monkeypatch.setattr(qc, "preflight", lambda: (m, am, sessions))
        monkeypatch.setattr(qc, "execution_source", lambda _: {"files": {}})
        monkeypatch.setattr(qc, "validate_execution", lambda *a, **kw: None)
        def supervisor(command, log, *, deadline, on_start):
            on_start(123)
            trial = log.parent
            intent = json.loads((trial / "intent.json").read_bytes())
            qc.save(trial / "receipt.json", {
                "schema": "lausanne-file-review-v1", "manifest_sha256": qc.MANIFEST_SHA,
                "path": row["path"], "subject": row["subject"], "session": row["session"], "role": "TRAIN",
                "intent_sha256": qc.digest(qc.encode(intent)), "declaration_sha256": intent["declaration_sha256"],
                "status": child_status, **qc.CLAIMS})
            return parent_status, exit_code
        monkeypatch.setattr(qc, "supervise", supervisor)
        report = qc.batch("process-control", "originals", 30, only_path=row["path"])
        result = next(r for r in report["outcomes"] if r["path"] == row["path"])
        assert result["status"] == parent_status and result["review_status"] == child_status
        assert len(report["outcomes"]) == 26
        assert sum(r["status"] == "deferred_not_selected" for r in report["outcomes"]) == 25
        assert report["training_admitted"] is False
        trial = cache / "runs/process-control/items" / qc.review_key(row["path"])
        if parent_status == "timeout":
            with pytest.raises(qc.Refusal, match="parent/declaration/intent"):
                qc.completed_original_review(trial / "receipt.json", row, m)
        with pytest.raises(FileExistsError):
            qc.batch("process-control", "originals", 30, only_path=row["path"])


@pytest.mark.parametrize("expire_at", ["supervision", "receipt_validation", "attempt_source", "batch_source"])
def test_parent_completion_deadlines_preserve_child_receipt(metadata, monkeypatch, expire_at):
    m, am, sessions = metadata
    row = m["originals"][0]
    clock, offset, validations = time.monotonic, [0], [0]
    with tempfile.TemporaryDirectory(prefix="deferred-deadline-controls-", dir=ROOT / "build") as folder:
        cache = Path(folder)
        monkeypatch.setattr(qc, "CACHE", cache)
        monkeypatch.setattr(qc, "preflight", lambda: metadata)
        monkeypatch.setattr(qc, "execution_source", lambda _: {"files": {}})
        monkeypatch.setattr(qc.time, "monotonic", lambda: clock() + offset[0])
        def source_validation(*args, **kwargs):
            validations[0] += 1
            if (expire_at == "attempt_source" and validations[0] == 2
                    or expire_at == "batch_source" and validations[0] == 3):
                offset[0] = 31
        monkeypatch.setattr(qc, "validate_execution", source_validation)
        retain = qc.retain_attempt_receipt
        def receipt_validation(*args):
            retain(*args)
            if expire_at == "receipt_validation":
                offset[0] = 31
        monkeypatch.setattr(qc, "retain_attempt_receipt", receipt_validation)
        def supervisor(command, log, **kwargs):
            kwargs["on_start"](123)
            intent = qc.bound_json(log.parent / "intent.json")
            # Process-state record only: no image header, scalars or positive QC.
            qc.save(log.parent / "receipt.json", {
                "schema": "lausanne-file-review-v1", "manifest_sha256": qc.MANIFEST_SHA,
                "path": row["path"], "subject": row["subject"], "session": row["session"], "role": "TRAIN",
                "intent_sha256": qc.digest(qc.encode(intent)), "declaration_sha256": intent["declaration_sha256"],
                "status": "review_failed", **qc.CLAIMS})
            if expire_at == "supervision":
                offset[0] = 31
            return "completed", 0
        monkeypatch.setattr(qc, "supervise", supervisor)
        report = qc.batch("deadline-control", "originals", 30, only_path=row["path"])
        assert report["status"] == "failed_or_incomplete" and report["elapsed_seconds"] >= 30
        item = next(r for r in report["outcomes"] if r["path"] == row["path"])
        trial = cache / "runs/deadline-control/items" / qc.review_key(row["path"])
        assert item["attempt"] == str(trial.relative_to(ROOT))
        assert item["receipt_sha256"] == qc.digest((trial / "receipt.json").read_bytes())
        assert item["review_status"] == "review_failed"
        # Earlier per-file completion stays true when only final batch closure
        # crosses the deadline; the batch must still refuse completion.
        assert item["status"] == ("completed" if expire_at == "batch_source" else "failed_or_incomplete")


@pytest.mark.parametrize("failure", [KeyboardInterrupt, OSError])
@pytest.mark.parametrize("valid_receipt", [True, False])
def test_interrupted_parent_retains_receipt_or_explicit_binding_error(metadata, monkeypatch, failure, valid_receipt):
    m, am, sessions = metadata
    row = m["originals"][0]
    with tempfile.TemporaryDirectory(prefix="deferred-interruption-controls-", dir=ROOT / "build") as folder:
        cache = Path(folder)
        monkeypatch.setattr(qc, "CACHE", cache)
        monkeypatch.setattr(qc, "preflight", lambda: metadata)
        monkeypatch.setattr(qc, "execution_source", lambda _: {"files": {}})
        monkeypatch.setattr(qc, "validate_execution", lambda *args, **kwargs: None)
        def supervisor(command, log, **kwargs):
            kwargs["on_start"](123)
            intent = qc.bound_json(log.parent / "intent.json")
            if valid_receipt:
                qc.save(log.parent / "receipt.json", {
                    "schema": "lausanne-file-review-v1", "manifest_sha256": qc.MANIFEST_SHA,
                    "path": row["path"], "subject": row["subject"], "session": row["session"], "role": "TRAIN",
                    "intent_sha256": qc.digest(qc.encode(intent)), "declaration_sha256": intent["declaration_sha256"],
                    "status": "failed_or_incomplete", **qc.CLAIMS})
            else:
                qc.atomic_preserve(log.parent / "receipt.json", b"invalid control metadata")
            raise failure("Control interruption after the child wrote a record")
        monkeypatch.setattr(qc, "supervise", supervisor)
        report = qc.batch("interruption-control", "originals", 30, only_path=row["path"])
        item = next(r for r in report["outcomes"] if r["path"] == row["path"])
        trial = cache / "runs/interruption-control/items" / qc.review_key(row["path"])
        assert report["status"] == item["status"] == "failed_or_incomplete"
        assert item["error"]["type"] == failure.__name__
        assert item["receipt_sha256"] == qc.digest((trial / "receipt.json").read_bytes())
        assert item["attempt"] == str(trial.relative_to(ROOT))
        if valid_receipt:
            assert item["review_status"] == "failed_or_incomplete" and "receipt_binding_error" not in item
        else:
            assert "review_status" not in item and item["receipt_binding_error"]["type"] == "JSONDecodeError"
