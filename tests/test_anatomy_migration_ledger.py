import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "migrate_anatomy_library.py"


def dump(path, document):
    encoded = json.dumps(document).encode()
    path.write_bytes(encoded)
    return hashlib.sha256(encoded).hexdigest()


@pytest.fixture
def migration(tmp_path):
    def ref(path):
        return {"path": str(path), "sha256": "a" * 64, "bytes": 30}

    def case(key, outcome, role):
        return {
            "key": key, "family": "fixture", "person_id": key, "patient_id": key,
            "case_id": key + "-case", "source_version": "source-v1", "timepoint": "source-session-unknown",
            "outcome": outcome, "role": role, "reason": key + " frozen reason",
            "available_scans": [], "existing_layers": [], "source_ontology": None,
            "format": "native collection", "source_locations": [str(tmp_path / "unavailable-source")],
        }

    authority = ref(tmp_path / "authority.json")
    manifest = ref(tmp_path / "unavailable-source" / "patient-anatomy.json")
    rows = [case(key, "eligible", "DEVELOPMENT") for key in ("prepared", "cancelled", "untouched", "failed")]
    for row in rows:
        row.update(route="verified-native-reuse", native_manifest=manifest["path"],
                   admission=[authority["path"]], inputs=[manifest], exposure="private",
                   prepare_options={"indication": "meningioma"})
    rows[0].update(available_scans=[{"modality": "T1CE"}], existing_layers=[{"kind": "tumor"}],
                   source_ontology={"4": "Enhancing lesion"})
    rows.extend((case("protected", "protected", "SELECT"), case("retained", "retained", "TRAIN"),
                 case("unknown", "unavailable", "UNKNOWN")))
    inventory = tmp_path / "inventory.json"
    checksum = dump(inventory, {
        "schema": "resectionlab.anatomy-migration-ledger.v1", "version": "fixture-v1",
        "cases": rows, "authorities": [authority], "external_roots": [],
    })
    receipt_path = tmp_path / "migration-receipt.json"
    receipt = {
        "schema": "resectionlab.anatomy-migration-receipt.v1",
        "ledger_sha256": checksum,
        "cases": {
            "prepared": {"outcome": "migrated_with_limitations", "reason": "prepared, optional model failed",
                         "retry_failed": True, "manifest": manifest["path"], "snapshot_id": "fixture-snapshot",
                         "manifest_sha256": "b" * 64,
                         "stages": [{"id": "lesion", "status": "cached", "reason": "Source-bound cached mask"},
                                    {"id": "vessels", "status": "failed", "reason": "Optional worker timeout"}]},
            "cancelled": {"outcome": "cancelled", "reason": "Stopped after one stage",
                          "stages": {"stage/~": {"id": "stage/~", "status": "cached", "reason": "Kept stage"}}},
            "failed": {"outcome": "failed", "reason": "Frozen file missing: " + manifest["path"], "stages": {}},
        },
    }
    dump(receipt_path, receipt)
    output = tmp_path / "joined.json"

    def cli(*options):
        return subprocess.run([sys.executable, str(SCRIPT), str(inventory), *map(str, options)],
                              text=True, capture_output=True)

    return cli, output, inventory, receipt_path, receipt


def test_join_keeps_frozen_rows_and_reports_partial_attempts(migration):
    cli, output, inventory, receipt_path, _ = migration
    original = {path: path.read_bytes() for path in (inventory, receipt_path)}
    command = cli("--results", receipt_path.parent, "--ledger-output", output, "--only", "prepared")
    assert command.returncode == 0, command.stderr
    assert json.loads(command.stdout) == {"migrated_with_limitations": 1}
    result = json.loads(output.read_text())
    assert not result["complete"]
    assert result["frozen_inventory"]["sha256"] == hashlib.sha256(original[inventory]).hexdigest()
    assert result["migration_receipt"]["sha256"] == hashlib.sha256(original[receipt_path]).hexdigest()
    assert result["summary"] == {
        "cases": 7, "pending": 2,
        "outcomes": dict.fromkeys(
            ("migrated_with_limitations", "cancelled", "unprocessed", "failed", "protected", "retained", "unavailable"), 1
        ),
        "stage_outcomes": {"cached": 2, "failed": 1},
    }
    rows = {row["key"]: row for row in result["cases"]}
    prepared = rows["prepared"]
    assert prepared["frozen_outcome"] == "eligible"
    assert prepared["role"] == "DEVELOPMENT" and prepared["source_ontology"] == {"4": "Enhancing lesion"}
    assert prepared["available_scans"] == [{"modality": "T1CE"}] and prepared["existing_layers"] == [{"kind": "tumor"}]
    assert prepared["prepare_options"] == {"indication": "meningioma"} and prepared["migration"]["retry_failed"]
    assert prepared["reason_paths"] == [
        {"path": str(inventory.resolve()), "json_pointer": "/cases/0/reason"},
        {"path": str(receipt_path.resolve()), "json_pointer": "/cases/prepared/reason"},
    ]
    stage = prepared["migration"]["stages"][1]
    assert stage["reason"] == "Optional worker timeout"
    assert stage["reason_path"]["json_pointer"] == "/cases/prepared/stages/1/reason"
    stage = rows["cancelled"]["migration"]["stages"][0]
    assert stage["reason_path"]["json_pointer"] == "/cases/cancelled/stages/stage~1~0/reason"
    assert rows["untouched"]["outcome"] == "unprocessed"
    assert rows["protected"]["role"] == "SELECT" and rows["protected"]["migration"] is None
    assert rows["retained"]["outcome"] == "retained" and rows["unknown"]["role"] == "UNKNOWN"
    assert rows["failed"]["reason"].startswith("Frozen file missing: ")
    assert all(path.read_bytes() == data for path, data in original.items())


def test_completed_ledger_can_replace_earlier_join_for_same_inventory(migration):
    cli, output, _, receipt_path, receipt = migration
    assert cli("--results", receipt_path.parent, "--ledger-output", output).returncode == 0
    for key in ("cancelled", "untouched"):
        receipt["cases"][key] = {"outcome": "migrated", "reason": "Resume completed", "stages": []}
    final_digest = dump(receipt_path, receipt)
    command = cli("--results", receipt_path.parent, "--ledger-output", output)
    assert command.returncode == 0, command.stderr
    result = json.loads(output.read_text())
    assert result["complete"] and result["summary"]["pending"] == 0
    assert result["migration_receipt"]["sha256"] == final_digest


@pytest.mark.parametrize("invalid", ("digest", "protected", "unknown", "overwrite", "mode"))
def test_rejects_invalid_join_without_writing(migration, invalid):
    cli, output, inventory, receipt_path, receipt = migration
    if invalid == "digest":
        receipt["ledger_sha256"] = "c" * 64
    if invalid in {"protected", "unknown"}:
        key = "protected" if invalid == "protected" else "not-in-inventory"
        receipt["cases"][key] = {"outcome": "migrated", "reason": "inadmissible"}
    dump(receipt_path, receipt)
    options = ("--results", receipt_path.parent)
    if invalid == "mode":
        options = ()
    if invalid == "overwrite":
        output = inventory
    before = output.read_bytes() if output.exists() else None
    command = cli(*options, "--ledger-output", output)
    assert command.returncode == 2
    assert (output.read_bytes() if output.exists() else None) == before
