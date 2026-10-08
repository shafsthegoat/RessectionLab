"""Controls on acquired source metadata; no generated patient fixtures."""
import csv
import importlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/anatomy/ds003949-v1.0.1/source-metadata/participants.tsv"
pytestmark = pytest.mark.skipif(not SOURCE.exists(), reason="Acquire pinned Lausanne source metadata first")


def test_grouped_roles_match_actual_source_sessions_and_known_overlap():
    source = {}
    for row in csv.DictReader(SOURCE.read_text().splitlines(), delimiter="\t"):
        source.setdefault(row["participant_id"], []).append((row["exam_date"], row["group"]))
    cohort = json.loads((ROOT / "manifests/lausanne-component-cohort-v1.json").read_text())
    members = cohort["members"]
    assert {person["subject"] for person in members} == set(source)
    assert len(members) == len({person["canonical_person"] for person in members}) == 284
    for person in members:
        assert person["canonical_person"] == "Lausanne:" + person["subject"]
        assert sorted((session, person["group"]) for session in person["sessions"]) == sorted(source[person["subject"]])
    assert sum(len(person["sessions"]) for person in members) == 296
    assert {role: sum(person["role"] == role for person in members) for role in
            ["TRAIN", "SELECT", "MEASUREMENT_EVAL"]} == {"TRAIN": 199, "SELECT": 43, "MEASUREMENT_EVAL": 42}
    assert next(person for person in members if person["subject"] == "sub-000")["role"] == "TRAIN"
    assert sum(person["topcow_overlap"] for person in members) == 20


def test_deadline_during_qc_escapes_instead_of_starting_second_image(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    acquisition = importlib.import_module("acquire_lausanne_pilot")
    if not acquisition.RESULT.exists():
        pytest.skip("Requires already acquired, byte-verified original pilot")
    from resectionlab import imaging
    calls = []
    def deadline(path):
        calls.append(path)
        raise TimeoutError("cooperative wall deadline")
    def existing_only(entry, directory):
        acquisition.verify_file(directory / entry["path"], entry)
        return "already_verified"
    monkeypatch.setattr(acquisition, "RESULT", tmp_path / "must-not-exist.json")
    monkeypatch.setattr(acquisition, "acquire_file", existing_only)
    monkeypatch.setattr(imaging, "inspect_nifti", deadline)
    with pytest.raises(TimeoutError, match="cooperative wall deadline"):
        acquisition.acquire()
    assert len(calls) == 1
    assert not acquisition.RESULT.exists()
