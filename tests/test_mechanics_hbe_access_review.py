"""Independent disclosure-boundary controls using temporary analytical ZIPs only."""
import importlib.util
import copy
import hashlib
import io
import json
from pathlib import Path
import tarfile
import zipfile

import pytest

from scripts import mechanics_hbe_access as a


SPEC = importlib.util.spec_from_file_location("hbe_access_owner_fixture",
    Path(__file__).with_name("test_mechanics_hbe_access.py"))
f = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(f)


def forbid_member_reads(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("Rejected evidence must not open any selected archive member")
    monkeypatch.setattr(zipfile.ZipFile, "read", fail)


def test_twenty_claimed_rows_with_dummy_primitive_blobs_are_not_numerical_evidence(tmp_path):
    study = f.study_fixture(tmp_path)
    report = f.numerical_fixture(tmp_path, study, 2000)
    assert len(report["runs"]) == 20
    binding = f.save(tmp_path, "invented-numerical.json", report)
    with pytest.raises(ValueError):
        a.validate_numerical_evidence(tmp_path, binding,
            protocol_sha256=study.protocol_binding["sha256"], fitted_mu_Pa=2000)


def test_claimed_large_limits_and_wrong_units_cannot_pass_frozen_protocol(tmp_path):
    study = f.study_fixture(tmp_path)
    report = f.numerical_fixture(tmp_path, study, 2000)
    key = next(iter(report["runs"]))
    report["runs"][key]["criteria"]["force_balance"] = {
        "actual": 100., "limit": 1000., "units": "arbitrary"}
    binding = f.save(tmp_path, "loose-numerical.json", report)
    with pytest.raises(ValueError):
        a.validate_numerical_evidence(tmp_path, binding,
            protocol_sha256=study.protocol_binding["sha256"], fitted_mu_Pa=2000)


def test_rehashing_prediction_and_freeze_does_not_make_a_new_authorized_freeze(tmp_path, monkeypatch):
    f.analytical_prerequisite_stubs(monkeypatch)  # isolate lifecycle, not numerical authenticity
    study = f.study_fixture(tmp_path)
    freeze, _, prediction_binding, _ = f.frozen_fixture(tmp_path, study)
    prediction = json.loads((tmp_path / prediction_binding["path"]).read_text())
    prediction["fitted"]["torsion_pos"]["response"][-1] += 1
    replacement = f.save(tmp_path, "changed-prediction.json", prediction)
    record = json.loads((tmp_path / freeze["path"]).read_text())
    record["predictions"] = replacement
    record["calibration_member_sha256"] = {"compression": "0" * 64, "tension": "0" * 64}
    resealed = f.save(tmp_path, "hand-authored-freeze.json", record)
    forbid_member_reads(monkeypatch)
    with pytest.raises(ValueError):
        study.evaluate_held_out(freeze_binding=resealed,
            schemas=f.schemas(("torsion_neg", "torsion_pos")))


def test_holdout_attempt_permanently_closes_calibration_even_for_new_instance(tmp_path, monkeypatch):
    f.analytical_prerequisite_stubs(monkeypatch)
    study = f.study_fixture(tmp_path)
    freeze, *_ = f.frozen_fixture(tmp_path, study)
    study.evaluate_held_out(freeze_binding=freeze,
        schemas=f.schemas(("torsion_neg", "torsion_pos")))
    restarted = a.ReleasedStudy(tmp_path, study.protocol_binding, study.release_binding,
        ledger_path="access.jsonl")
    forbid_member_reads(monkeypatch)
    with pytest.raises(ValueError):
        restarted.read_calibration(f.schemas(("compression", "tension")))


def test_holdout_attempt_permanently_closes_refreeze(tmp_path, monkeypatch):
    f.analytical_prerequisite_stubs(monkeypatch)
    study = f.study_fixture(tmp_path)
    freeze, fit_binding, prediction_binding, numerical_binding = f.frozen_fixture(tmp_path, study)
    study.evaluate_held_out(freeze_binding=freeze,
        schemas=f.schemas(("torsion_neg", "torsion_pos")))
    with pytest.raises(ValueError):
        study.freeze_predictions(fit_binding=fit_binding, prediction_binding=prediction_binding,
            numerical_binding=numerical_binding, output_path="post-disclosure-freeze.json")


def test_calibration_release_needs_actual_replayable_source_and_prerequisites(tmp_path, monkeypatch):
    study = f.study_fixture(tmp_path)
    # The original fixture has only hash-shaped source labels and one dummy prerequisite blob.
    release = json.loads((tmp_path / study.release_binding["path"]).read_text())
    assert "source_archive" not in release
    forbid_member_reads(monkeypatch)
    with pytest.raises(ValueError):
        study.read_calibration(f.schemas(("compression", "tension")))


def test_wrong_archive_hash_rejects_before_any_zip_member_read(tmp_path, monkeypatch):
    f.analytical_prerequisite_stubs(monkeypatch)
    study = f.study_fixture(tmp_path)
    with (tmp_path / "analytical.zip").open("ab") as stream:
        stream.write(b"changed archive bytes")
    forbid_member_reads(monkeypatch)
    with pytest.raises(ValueError, match="Archive"):
        study.read_calibration(f.schemas(("compression", "tension")))


def test_failed_heldout_member_parse_still_seals_refitting_and_new_ledger(tmp_path, monkeypatch):
    f.analytical_prerequisite_stubs(monkeypatch)
    study = f.study_fixture(tmp_path)
    freeze, fit, prediction, numerical = f.frozen_fixture(tmp_path, study)
    original = a.parse_member_csv
    def failed(data, branch, schema):
        if branch.startswith("torsion"):
            raise ValueError("Constructed heldout parse failure")
        return original(data, branch, schema)
    monkeypatch.setattr(a, "parse_member_csv", failed)
    with pytest.raises(ValueError, match="Constructed heldout"):
        study.evaluate_held_out(freeze_binding=freeze, schemas=f.schemas(("torsion_neg", "torsion_pos")))
    events = [json.loads(line) for line in (tmp_path / "access.jsonl").read_text().splitlines()]
    assert events[-1]["phase"] == "held_out_attempt"
    assert events[-2]["phase"] == "freeze_saved"
    assert events[-2]["freeze"] == freeze
    forbid_member_reads(monkeypatch)
    with pytest.raises(ValueError, match="seals subsequent"):
        study.freeze_predictions(fit_binding=fit, prediction_binding=prediction,
            numerical_binding=numerical, output_path="after-failed-read.json")
    fresh = a.ReleasedStudy(tmp_path, study.protocol_binding, study.release_binding, ledger_path="new-ledger.jsonl")
    with pytest.raises(ValueError, match="ledger location"):
        fresh.read_calibration(f.schemas(("compression", "tension")))


def test_replayed_values_cannot_be_replaced_by_rehashed_loose_claims(tmp_path, monkeypatch):
    """Isolate exact replay comparison; authentic-source rejection is tested separately."""
    study = f.study_fixture(tmp_path)
    rebuilt = f.numerical_fixture(tmp_path, study, 2000)
    claimed = copy.deepcopy(rebuilt)
    key = next(iter(claimed["runs"]))
    claimed["runs"][key]["criteria"]["force_balance"] = {"actual": 100., "limit": 1000., "units": "wrong"}
    binding = f.save(tmp_path, "coherently-rehashed-claim.json", claimed)
    checked = a._check_report
    def metadata_only(report, *, protocol_sha256, fitted_mu_Pa, require_fitted, root=None):
        return checked(report, protocol_sha256=protocol_sha256, fitted_mu_Pa=fitted_mu_Pa,
                       require_fitted=require_fitted)
    calls = []
    def replay(*args, **kwargs):
        calls.append(True)
        return rebuilt
    monkeypatch.setattr(a, "_check_report", metadata_only)
    monkeypatch.setattr(a, "_replay_evidence", replay)
    with pytest.raises(ValueError, match="independent primitive replay"):
        a.validate_numerical_evidence(tmp_path, binding,
            protocol_sha256=study.protocol_binding["sha256"], fitted_mu_Pa=2000)
    assert calls == [True]


def test_source_tar_member_corruption_rejected_even_when_outer_tar_hash_matches(tmp_path, monkeypatch):
    """Use a constructed module-origin tree; real provenance/hash/tar code stays active."""
    source_dir = tmp_path / "scripts"
    source_dir.mkdir()
    bindings = {}
    contents = {}
    for name, filename in a.SOURCE_FILES.items():
        data = ("# constructed source marker: " + name + "\n").encode()
        path = source_dir / filename
        path.write_bytes(data)
        contents[filename] = data
        bindings[name] = {"path": "scripts/" + filename, "sha256": hashlib.sha256(data).hexdigest()}
    monkeypatch.setattr(a, "__file__", str(source_dir / "mechanics_hbe_access.py"))
    bundle = tmp_path / "source.tar"
    with tarfile.open(bundle, "w", format=tarfile.PAX_FORMAT, pax_headers={"comment": "1" * 40}) as archive:
        for filename, data in contents.items():
            if filename == "mechanics_hbe_access.py":
                data += b"# changed archive member\n"
            info = tarfile.TarInfo("scripts/" + filename)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    digest = hashlib.sha256(bundle.read_bytes()).hexdigest()
    release = {"source_commit": "1" * 40, "source_archive_sha256": digest,
               "execution": {"source_bindings": bindings,
                             "source_archive": {"path": "source.tar", "sha256": digest}}}
    with pytest.raises(ValueError, match="Archived source bytes differ"):
        a.verify_release_provenance(tmp_path, release, "2" * 64)


def test_duplicate_selected_zip_name_rejects_before_member_payload(tmp_path, monkeypatch):
    f.analytical_prerequisite_stubs(monkeypatch)
    study = f.study_fixture(tmp_path)
    archive_path = tmp_path / "analytical.zip"
    with pytest.warns(UserWarning, match="Duplicate name"):
        with zipfile.ZipFile(archive_path, "a") as archive:
            archive.writestr("fixture/compression.csv", "x,y\n0,0\n-0.0005,-0.01\n")
    protocol = json.loads((tmp_path / study.protocol_binding["path"]).read_text())
    roles = json.loads((tmp_path / protocol["roles"]["path"]).read_text())
    roles["source"]["archive_sha256"] = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    roles["source"]["archive_bytes"] = archive_path.stat().st_size
    role_binding = f.save(tmp_path, "roles.json", roles)
    protocol["roles"] = role_binding
    protocol_binding = f.save(tmp_path, "protocol.json", protocol)
    release = json.loads((tmp_path / study.release_binding["path"]).read_text())
    release.update(protocol_sha256=protocol_binding["sha256"], roles_sha256=role_binding["sha256"],
                   archive_sha256=roles["source"]["archive_sha256"])
    release_binding = f.save(tmp_path, "release.json", release)
    study = a.ReleasedStudy(tmp_path, protocol_binding, release_binding, ledger_path="access.jsonl")
    forbid_member_reads(monkeypatch)
    with pytest.raises(ValueError, match="Duplicate ZIP"):
        study.read_calibration(f.schemas(("compression", "tension")))


def test_execution_receipt_binds_working_directory_not_just_command_text(tmp_path, monkeypatch):
    """Constructed executable/runtime identity; no program is executed."""
    executable = f.save(tmp_path, "fake-executable", {"analytical_marker": True})
    runtime = f.save(tmp_path, "runtime.json", {"executable_sha256": executable["sha256"]})
    monkeypatch.setattr(a, "RUNTIME_IDENTITY_SHA256", runtime["sha256"])
    names = {"deck": "specimen.feb", "loading": "loading.json", "nodes": "nodes.log",
             "elements": "elements.log", "solver": "solver.log", "mesh": "mesh.json"}
    primitives = {key: f.save(tmp_path, "run/" + filename, {"constructed": key})
                  for key, filename in names.items()}
    record = {"schema": "hbe-run-execution-v1", "run_id": "tension:N4:S60:reference",
              "protocol_sha256": "8" * 64, "primitive_bindings": primitives,
              "execution": {"exit_code": 0, "timed_out": False, "elapsed_seconds": 1},
              "runtime_identity": runtime, "executable": executable, "cwd": "run",
              "command": [str(tmp_path / "fake-executable"), "-noconfig", "-no_title",
                          "-i", "specimen.feb", "-o", "solver.log"]}
    row = {"primitive_bindings": primitives,
           "execution_binding": f.save(tmp_path, "execution.json", record)}
    assert a.verify_run_execution(tmp_path, record["run_id"], row, "8" * 64)["cwd"] == "run"
    record["cwd"] = "different-run"
    row["execution_binding"] = f.save(tmp_path, "wrong-execution.json", record)
    with pytest.raises(ValueError, match="working directory"):
        a.verify_run_execution(tmp_path, record["run_id"], row, "8" * 64)
