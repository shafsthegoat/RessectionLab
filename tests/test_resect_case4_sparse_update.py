"""Algebra/header grammar/access controls only; no patient files are opened.

Temporary point tables and headers exercise software identities and disclosure
ordering. They are not patient anatomy, acquired observations or benchmarks.
"""
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest

from resectionlab import mechanics_landmarks as lm
from scripts import mechanics_patient_comparison as comparison

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("sparse_update", ROOT / "scripts/run_resect_case4_sparse_update.py")
r = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(r)


def write(root, name, value):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    data = value if isinstance(value, bytes) else r.encode(value)
    path.write_bytes(data)
    return {"path": name, "sha256": r.digest(data)}


@pytest.fixture
def declaration():
    # Temporary controls bind the interpreter running this test. The archived
    # patient experiment retains its original runtime unchanged on disk.
    value = json.loads((ROOT / r.MANIFEST).read_bytes())
    value["runtime"] = r.runtime_identity()
    return value


@pytest.mark.parametrize("change", [
    {"role": "TRAIN"}, {"idw_power": 3}, {"external_field": {}},
    {"B_ids": [1, 2, 3, 4, 5, 6]}, {"phase_wall_seconds": 31},
    {"header_rule": "infer_from_landmarks"}, {"output_directory": "elsewhere"},
])
def test_declaration_refuses_scope_or_partition_changes(declaration, change):
    r.validate_declaration(declaration)
    with pytest.raises(ValueError, match="DECLARATION_POLICY_CHANGED"):
        r.validate_declaration({**declaration, **change})


def test_patient_guard_distinguishes_header_and_tag_phases(tmp_path, monkeypatch):
    monkeypatch.setattr(r, "ROOT", tmp_path)
    header = r.patient_guard("header-qc")
    fitting = r.patient_guard("fit-freeze")
    image = tmp_path / r.INPUTS["during"]["path"]
    tag = tmp_path / r.INPUTS["tag"]["path"]
    header("open", (str(image), "rb", os.O_RDONLY))
    fitting("open", (str(tag), "rb", os.O_RDONLY))
    for guard, path, mode, flags in ((header, tag, "rb", os.O_RDONLY),
            (fitting, image, "rb", os.O_RDONLY), (fitting, tag, "wb", os.O_WRONLY),
            (header, tmp_path / "data/other/person.nii.gz", "rb", os.O_RDONLY)):
        with pytest.raises(PermissionError):
            guard("open", (str(path), mode, flags))
    alias = tmp_path / "outside-alias"
    alias.parent.mkdir(exist_ok=True)
    tag.parent.mkdir(parents=True)
    tag.write_text("opaque test bytes")
    alias.symlink_to(tag)
    with pytest.raises(PermissionError):
        fitting("open", (str(alias), "rb", os.O_RDONLY))


def test_header_grammar_preserves_oblique_sform_and_refuses_unverified_frames():
    h = nib.Nifti1Header()
    h.set_data_shape((4, 5, 6))
    h.set_xyzt_units("mm")
    h.set_sform([[0, -2, 0, 10], [2, 0, 0, 20], [0, 0, 3, 30], [0, 0, 0, 1]], code=1)
    h.set_qform(None, code=0)
    record = r.header_record(h.binaryblock)
    assert record["shape"] == [4, 5, 6]
    assert record["selected_affine"] == h.get_sform().tolist()
    assert record["full_cell_bounds_ras_mm"] == {"minimum": [1., 19., 28.5], "maximum": [11., 27., 46.5]}
    assert record["image_array_accessed"] is False
    for units, qcode, scode in (("unknown", 0, 1), ("meter", 0, 1), ("mm", 1, 1), ("mm", 0, 0)):
        invalid = h.copy()
        invalid.set_xyzt_units(units)
        invalid["qform_code"], invalid["sform_code"] = qcode, scode
        with pytest.raises(ValueError, match="NATIVE_HEADER_CONVENTION_UNVERIFIED"):
            r.header_record(invalid.binaryblock)
    with pytest.raises(ValueError, match="NIFTI_HEADER_LENGTH"):
        r.header_record(h.binaryblock + b"extra")


def test_authentication_precedes_any_patient_read(tmp_path, monkeypatch, declaration):
    monkeypatch.setattr(r, "ROOT", tmp_path)
    write(tmp_path, r.MANIFEST, declaration)
    # Only source/metadata placeholders: no original patient paths exist here.
    for name in r.SOURCE_NAMES:
        write(tmp_path, name, b"# inert source binding control\n")
    hashes = r.source_hashes()
    declaration["reused_source_sha256"] = {k: v for k, v in hashes.items() if k != r.SOURCE_NAMES[0]}
    binding = write(tmp_path, r.MANIFEST, declaration)
    release = {"schema": r.SCHEMA + "-release", "phase": "header-qc", "authorized": True,
               "authorizer": "root", "source_commit": "a" * 40,
               "declaration_sha256": binding["sha256"], "source_sha256": hashes, "dependencies": {}}
    name = r.OUTPUT + "/releases/header-qc.json"
    write(tmp_path, name, release)
    monkeypatch.setattr(r.subprocess, "run", lambda command, **kwargs:
                        SimpleNamespace(stdout=(tmp_path / command[2].split(":", 1)[1]).read_bytes()))
    assert r.authenticate(name, "header-qc")[1] == release
    with monkeypatch.context() as altered:
        altered.setattr(r, "runtime_identity", lambda: {**declaration["runtime"], "numpy": "changed"})
        with pytest.raises(ValueError, match="RUNTIME_BINDING_CHANGED"):
            r.authenticate(name, "header-qc")
    actual_runtime = r.runtime_identity()
    with monkeypatch.context() as changed:
        changed.setattr(r, "runtime_identity", lambda: {**actual_runtime, "numpy": "different"})
        with pytest.raises(ValueError, match="RUNTIME_BINDING_CHANGED"):
            r.authenticate(name, "header-qc")
    assert not (tmp_path / "data").exists()
    write(tmp_path, name, {**release, "authorized": False})
    with pytest.raises(ValueError, match="EXPLICIT_ROOT_RELEASE_REQUIRED"):
        r.authenticate(name, "header-qc")
    write(tmp_path, name, release)
    write(tmp_path, r.SOURCE_NAMES[0], b"# changed\n")
    with pytest.raises(ValueError, match="SOURCE_CLOSURE_CHANGED"):
        r.authenticate(name, "header-qc")
    with pytest.raises(ValueError, match="EXACT_PHASE_RELEASE_REQUIRED"):
        r.authenticate(name, "fit-freeze")


@pytest.fixture
def algebra(tmp_path, monkeypatch, declaration):
    """An abstract nonplanar point table, never a clinical-looking test case."""
    points = np.array([[i, i * i % 13, i * i * i % 17] for i in range(19)], dtype=float)
    target = points + [.25, -.5, 1.]
    rows = [' '.join(map(str, (*a, *b))) for a, b in zip(points, target)]
    payload = ("MNI Tag Point File\nVolumes = 2;\nPoints =\n" + "\n".join(rows) + "\n;\n").encode()
    pair = lm.LandmarkPairBinding("algebra_control", lm.DISPLACEMENT_ROLE,
        r.digest(payload), "1" * 64, "2" * 64)
    partition = lm.partition_displacement_sources(lm.parse_tag_sources(payload, pair))
    # The production binding remains Case4; the runner's real fixed-partition
    # check is tested separately. This fixture supplies the already typed values.
    qc = lm.LandmarkFrameQC("1" * 64, "2" * 64, np.eye(4), np.eye(4), "3" * 64, "algebra only")
    pair = lm.LandmarkPairBinding("algebra_control", lm.DISPLACEMENT_ROLE,
        r.digest(payload), "1" * 64, "2" * 64, qc)
    monkeypatch.setattr(r, "ROOT", tmp_path)
    monkeypatch.setattr(r, "B_IDS", partition.boundary_ids)
    monkeypatch.setattr(r, "V_IDS", partition.validation_ids)
    monkeypatch.setattr(r, "prepare_landmarks", lambda *args: (lm, comparison, payload, pair, partition))
    protocol = write(tmp_path, r.MANIFEST, declaration)
    release = {"declaration_sha256": protocol["sha256"], "dependencies": {}}
    return declaration, release, pair, partition, payload


def test_fit_freezes_complete_fields_without_V_then_evaluates_once(algebra, monkeypatch, tmp_path):
    declaration, release, pair, partition, payload = algebra
    original = lm._coordinate_tokens
    converted = []
    def observed(tokens):
        converted.append(tuple(tokens))
        return original(tokens)
    monkeypatch.setattr(lm, "_coordinate_tokens", observed)
    result = r.scientific_stage("fit-freeze", declaration, release, {"frame_qc": {}}, float("inf"))
    raw_rows = lm._records(payload, pair)
    withheld = {tuple(raw_rows[i - 1][3:6]) for i in partition.validation_ids}
    assert withheld.isdisjoint(converted)
    assert (tmp_path / result["freeze"]["path"]).exists()
    assert result["V_destinations_accessed"] == []
    release["dependencies"]["freeze"] = result["freeze"]
    outcome = r.scientific_stage("evaluate", declaration, release, {"frame_qc": {}}, float("inf"))
    report = json.loads((tmp_path / outcome["evaluation"]["path"]).read_bytes())
    assert report["validation_landmarks"] == 13
    assert report["methods"]["no_shift"]["all_supported"]["rms_mm"] == pytest.approx(np.sqrt(1.3125))
    assert report["methods"]["proper_rigid"]["all_supported"]["maximum_mm"] < 1e-12
    assert outcome["physical_clearance_mm"] is None
    assert outcome["future_untouched_Case4_validation_available"] is False
    before = len(converted)
    with pytest.raises(FileExistsError):
        r.scientific_stage("evaluate", declaration, release, {"frame_qc": {}}, float("inf"))
    assert len(converted) == before


def test_changed_freeze_refuses_before_V_conversion(algebra, monkeypatch, tmp_path):
    declaration, release, _, _, _ = algebra
    result = r.scientific_stage("fit-freeze", declaration, release, {"frame_qc": {}}, float("inf"))
    release["dependencies"]["freeze"] = result["freeze"]
    (tmp_path / result["freeze"]["path"]).write_bytes(b"changed\n")
    monkeypatch.setattr(lm, "reveal_validation_landmarks", lambda *args:
                        pytest.fail("Invalid freeze reached V reveal"))
    with pytest.raises(ValueError, match="Frozen artifact bytes changed"):
        r.scientific_stage("evaluate", declaration, release, {"frame_qc": {}}, float("inf"))


def test_read_bounds_and_symlink_refusal(tmp_path, monkeypatch):
    monkeypatch.setattr(r, "ROOT", tmp_path)
    (tmp_path / "large").write_bytes(b"12345")
    with pytest.raises(ValueError, match="READ_LIMIT_EXCEEDED"):
        r.bounded_read("large", maximum=4)
    (tmp_path / "alias").symlink_to(tmp_path / "large")
    with pytest.raises(ValueError, match="PATH_ALIAS_PROHIBITED"):
        r.bounded_read("alias")
    with pytest.raises(ValueError, match="RELATIVE_FIXED_PATH_REQUIRED"):
        r.bounded_read("../outside")


def test_worker_claim_is_one_shot_even_without_scientific_result(tmp_path, monkeypatch):
    monkeypatch.setattr(r, "ROOT", tmp_path)
    write(tmp_path, r.OUTPUT + "/header-qc/attempt.json", {"phase": "header-qc", "release_sha256": "a" * 64})
    with pytest.raises(ValueError, match="SUPERVISED_ATTEMPT_REQUIRED"):
        r.claim_worker("header-qc", "b" * 64)
    r.claim_worker("header-qc", "a" * 64)
    with pytest.raises(FileExistsError):
        r.claim_worker("header-qc", "a" * 64)


def test_promoted_frame_refuses_before_original_tag_read(declaration, monkeypatch):
    monkeypatch.setattr(r, "load_helpers", lambda: (lm, comparison))
    monkeypatch.setattr(r, "bounded_read", lambda *args: pytest.fail("Invalid frame opened tag"))
    frame = {"schema": r.SCHEMA + "-frame-qc", "status": "coordinate_convention_verified",
             "source_image_sha256": r.INPUTS["before"]["sha256"],
             "destination_image_sha256": r.INPUTS["during"]["sha256"],
             "source_world_to_ras_mm": np.eye(4).tolist(), "destination_world_to_ras_mm": np.eye(4).tolist(),
             "convention": declaration["coordinate_convention"], "anatomical_alignment_accepted": False,
             "physical_clearance_mm": 5}
    with pytest.raises(ValueError, match="FRAME_QC_CHANGED_OR_PROMOTED"):
        r.prepare_landmarks(declaration, {}, frame)


@pytest.mark.parametrize('elapsed,status,accepted', [
    (29.99, 'completed', True), (30.01, 'completed', False),
    (29.99, 'worker_failed', False),
])
def test_supervision_acceptance_uses_saved_final_elapsed(tmp_path, monkeypatch, elapsed, status, accepted):
    import importlib
    monkeypatch.syspath_prepend(str(ROOT / 'scripts'))
    intake = importlib.import_module('real_intake_io')
    monkeypatch.setattr(r, 'ROOT', tmp_path)
    path = r.OUTPUT + '/releases/header-qc.json'
    binding = write(tmp_path, path, b'opaque release control')
    release = {'source_sha256': {}}
    monkeypatch.setattr(r, 'authenticate', lambda *args:
                        ({'phase_wall_seconds': 30, 'runtime': r.runtime_identity()}, release, {}, binding['sha256']))
    monkeypatch.setattr(r, 'source_hashes', lambda: {})
    ticks = iter([0., elapsed])
    monkeypatch.setattr(r.time, 'monotonic', lambda: next(ticks))
    def completed(*args, **kwargs):
        write(tmp_path, r.OUTPUT + '/header-qc/result.json', {'status': 'completed', 'runtime': r.runtime_identity()})
        return status, 0 if status == 'completed' else 1
    monkeypatch.setattr(intake, 'supervise', completed)
    receipt = r.launch(path, 'header-qc')
    saved = json.loads((tmp_path / r.OUTPUT / 'header-qc/supervision.json').read_bytes())
    assert receipt == saved
    assert saved['elapsed_seconds'] == elapsed
    assert saved['scientific_result_accepted'] is accepted


@pytest.mark.parametrize('changed', [
    {'schema': 'unrelated'}, {'phase': 'evaluate'}, {'exit_code': 9},
    {'elapsed_seconds': 30.01}, {'elapsed_seconds': -1},
    {'elapsed_seconds': True}, {'source_sha256': {}},
    {'release_sha256': '0' * 64}, {'scientific_result_accepted': False},
])
def test_prerequisite_receipt_identity_before_patient_access(tmp_path, monkeypatch, declaration, changed):
    monkeypatch.setattr(r, 'ROOT', tmp_path)
    for name in r.SOURCE_NAMES:
        write(tmp_path, name, b'# inert source binding control\n')
    hashes = r.source_hashes()
    declaration['reused_source_sha256'] = {k: v for k, v in hashes.items() if k != r.SOURCE_NAMES[0]}
    protocol = write(tmp_path, r.MANIFEST, declaration)
    header_release = {'schema': r.SCHEMA + '-release', 'phase': 'header-qc', 'authorized': True,
                      'authorizer': 'root', 'source_commit': 'a' * 40,
                      'declaration_sha256': protocol['sha256'], 'source_sha256': hashes,
                      'dependencies': {}}
    header = write(tmp_path, r.OUTPUT + '/releases/header-qc.json', header_release)
    receipt = {'schema': r.SCHEMA + '-supervision', 'phase': 'header-qc', 'status': 'completed',
               'scientific_result_accepted': True, 'source_sha256': hashes,
               'release_sha256': header['sha256'], 'exit_code': 0, 'elapsed_seconds': .5}
    frame = write(tmp_path, r.FRAME_PATH, {})
    supervision = write(tmp_path, r.DEPENDENCY_PATHS['header_supervision'], receipt)
    release = {**header_release, 'phase': 'fit-freeze',
               'dependencies': {'frame_qc': frame, 'header_supervision': supervision}}
    path = r.OUTPUT + '/releases/fit-freeze.json'
    write(tmp_path, path, release)
    monkeypatch.setattr(r.subprocess, 'run', lambda command, **kwargs:
                        SimpleNamespace(stdout=(tmp_path / command[2].split(':', 1)[1]).read_bytes()))
    assert r.authenticate(path, 'fit-freeze')[1] == release
    release['dependencies']['header_supervision'] = write(
        tmp_path, r.DEPENDENCY_PATHS['header_supervision'], {**receipt, **changed})
    write(tmp_path, path, release)
    with pytest.raises(ValueError, match='PREVIOUS_STAGE_NOT_ACCEPTED'):
        r.authenticate(path, 'fit-freeze')
    assert not (tmp_path / 'data').exists()
