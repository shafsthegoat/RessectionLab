"""Prospective frozen-case protocol tests; no patient images or network reads."""
from __future__ import annotations

import importlib.util
import hashlib
import json
import copy
from zipfile import ZipFile
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "prepare_frozen_ucsf_assessment_test",
    ROOT / "scripts/prepare_frozen_ucsf_assessment.py")
frozen = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(frozen)


def test_metadata_only_declaration_selects_first_unused_and_is_immutable(tmp_path, monkeypatch):
    inventory = tmp_path / "inventory.json"
    inventory.write_text(json.dumps([{"type": "directory", "path": f"UCSF-PDGM-{n:04d}",
                                      "oid": f"oid-{n}"}
                                     for n in (4, 5, 7, 8, 9)]))
    development = tmp_path / "development.json"
    development.write_text(json.dumps({"role": "DEVELOPMENT",
        "mirror_revision": frozen.REVISION,
        "cases": ["UCSF-PDGM-0005", "UCSF-PDGM-0007"]}))
    monkeypatch.setattr(frozen, "INVENTORY_SHA256", frozen.digest(inventory))
    path = frozen.declare(tmp_path, inventory=inventory, development=development)
    initial = path.read_bytes()
    protocol = frozen.validate_protocol(tmp_path, inventory=inventory,
                                        development=development)
    assert protocol["patientId"] == "UCSF-PDGM-0008"
    assert protocol["role"] == "FROZEN_ASSESSMENT"
    assert protocol["declarationState"] == "pending_no_pixels"
    assert protocol["directoryOid"] == "oid-8"
    assert protocol["trainingProhibited"] is True
    assert protocol["primaryCheckpoint"] == "ppo-seed17.pt"
    assert protocol["accessMethod"] == "axis0-access"
    assert len(protocol["checkpointRoster"]) == 6
    assert frozen.declare(tmp_path, inventory=inventory, development=development) == path
    assert path.read_bytes() == initial
    assert list(tmp_path.rglob("*.nii.gz")) == []
    changed = json.loads(initial)
    changed["trainingProhibited"] = False
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="role restrictions"):
        frozen.validate_protocol(tmp_path, inventory=inventory,
                                 development=development)


def test_declaration_rejects_changed_inventory_or_role(tmp_path, monkeypatch):
    inventory = tmp_path / "inventory.json"
    inventory.write_text(json.dumps([{"type": "directory", "path": "UCSF-PDGM-0008",
                                      "oid": "oid-8"}]))
    development = tmp_path / "development.json"
    development.write_text(json.dumps({"role": "DEVELOPMENT",
        "mirror_revision": frozen.REVISION,
        "cases": ["UCSF-PDGM-0005", "UCSF-PDGM-0007"]}))
    monkeypatch.setattr(frozen, "INVENTORY_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="inventory changed"):
        frozen.declare(tmp_path, inventory=inventory, development=development)
    monkeypatch.setattr(frozen, "INVENTORY_SHA256", frozen.digest(inventory))
    development.write_text(json.dumps({"role": "TRAIN", "mirror_revision": frozen.REVISION,
                                      "cases": ["UCSF-PDGM-0005", "UCSF-PDGM-0007"]}))
    with pytest.raises(ValueError, match="development roles"):
        frozen.declare(tmp_path, inventory=inventory, development=development)
    assert not frozen.protocol_path(tmp_path).exists()


def test_seal_requires_all_six_paired_dev_checkpoints_before_any_pixels(tmp_path,
                                                                          monkeypatch):
    import run_patient_sector_learning as runner
    from resectionlab import patient_sector_learning as policy
    from resectionlab.core import semantic_digest
    import torch

    protocol_path = frozen.declare(tmp_path)
    assert protocol_path.exists()
    fit_dir = tmp_path / "fit"
    fit_dir.mkdir()
    producer = {"testProducer": "current"}
    monkeypatch.setattr(runner, "producer_identity", lambda: producer)
    monkeypatch.setattr(policy, "frozen_actor", lambda path, expected_producer: object())
    monkeypatch.setattr(policy, "parameter_hash", lambda actor: "actor-hash")
    sources = {}
    for key, patient in (("train", "UCSF-PDGM-0007"),
                         ("transfer", "UCSF-PDGM-0005")):
        archive = fit_dir / (patient + ".ressectionlab")
        manifest = {"case_id": patient + "-scene", "artifacts": {"surgical_toolbox": {
            "planning": {"sourceSceneId": "scene-" + key,
                "comparisonSeal": "comparison-" + key,
                "alternatives": [{"method": "axis0-access",
                                  "planSeal": "plan-" + key}]},
            "sectorPlanning": {"sourceAccessComparisonSeal": "comparison-" + key}}}}
        with ZipFile(archive, "w") as stream:
            stream.writestr("manifest.json", json.dumps(manifest))
        sources[key] = {"path": str(archive), "fileSha256": frozen.digest(archive),
            "caseId": patient + "-scene", "sceneId": "scene-" + key,
            "accessPlanSeal": "plan-" + key, "accessMethod": "axis0-access",
            "sectorCellSha256": {"near": "a" * 64, "far": "b" * 64},
            "roleDeclaration": {"patientId": patient, "role": "DEVELOPMENT",
                                "mirrorRevision": frozen.REVISION}}
    plan = [{"update": update, "episode": episode,
             "sourceKey": "train" if episode == 0 else "transfer"}
            for update in range(2) for episode in range(2)]
    fit = {"status": "saved", "mode": "fit", "sourceRole": "DEVELOPMENT",
        "ppoSourceSchedule": "paired-development", "producerIdentity": producer,
        "seeds": [17, 29, 43], "sources": sources, "taskGate": {"fixed": True},
        "ppoEpisodePlan": plan}
    bindings = {key: frozen._source_binding_from_fit(fit, key)
                for key in ("train", "transfer")}
    assignments = [{**row,
        "sourceArchiveSha256": bindings[row["sourceKey"]]["sourceArchiveSha256"],
        "optimizationRole": "DEV_TRAINED_FOR_PPO"} for row in plan]
    bc_proof = {"sourceBinding": bindings["train"]}
    ppo_proof = {"sourceBinding": bindings["train"],
        "ppoSourceSchedule": "paired-development",
        "optimizationSourceBindings": bindings,
        "ppoEpisodePlan": plan, "sampledEpisodeAssignments": assignments}
    monkeypatch.setattr(torch, "load", lambda path, **_: {
        "stage": "bc" if Path(path).name.startswith("bc-") else "bc-initialized-ppo",
        "trainingReceipt": bc_proof if Path(path).name.startswith("bc-") else ppo_proof,
        "sourceBinding": bindings["train"]})
    runs = []
    for seed in (17, 29, 43):
        run = {"seed": seed, "ppoSampledEpisodeAssignments": assignments}
        for stage in ("bc", "ppo"):
            name = f"{stage}-seed{seed}.pt"
            file = fit_dir / name
            file.write_bytes((name + " synthetic weights fixture").encode())
            run[stage] = {"model": {"path": str(file),
                "fileSha256": frozen.digest(file), "actorSha256": "actor-hash",
                "stage": "bc" if stage == "bc" else "bc-initialized-ppo",
                "sourceBinding": bindings["train"],
                "trainingReceiptSha256": semantic_digest(
                    bc_proof if stage == "bc" else ppo_proof)}}
        runs.append(run)
    fit.update(seedRuns=runs)
    (fit_dir / "receipt.json").write_text(json.dumps(fit))
    bad_model = fit["seedRuns"][0]["bc"]["model"]
    saved_binding = bad_model["sourceBinding"]
    bad_model["sourceBinding"] = {"selfReportedOnly": True}
    (fit_dir / "receipt.json").write_text(json.dumps(fit))
    with pytest.raises(ValueError, match="stage or paired exposure"):
        frozen.seal(tmp_path, fit_dir)
    bad_model["sourceBinding"] = saved_binding
    (fit_dir / "receipt.json").write_text(json.dumps(fit))
    assert frozen._ppo_proof_matches(ppo_proof, runs[0], bindings, plan)
    wrong_transfer = copy.deepcopy(ppo_proof)
    wrong_transfer["optimizationSourceBindings"]["transfer"]["sceneId"] = "other-scene"
    assert not frozen._ppo_proof_matches(wrong_transfer, runs[0], bindings, plan)
    wrong_episode = copy.deepcopy(ppo_proof)
    wrong_episode["sampledEpisodeAssignments"][1]["sourceArchiveSha256"] = \
        bindings["train"]["sourceArchiveSha256"]
    assert not frozen._ppo_proof_matches(wrong_episode, runs[0], bindings, plan)
    path = frozen.seal(tmp_path, fit_dir)
    seal = json.loads(path.read_text())
    assert seal["primaryCheckpoint"] == "ppo-seed17.pt"
    assert set(seal["checkpoints"]) == set(frozen.ROSTER)
    assert seal["modelSelectionUsedAssessmentPixels"] is False
    assert list(tmp_path.rglob("*.nii.gz")) == []
    assert frozen.validate_freeze(tmp_path)[1] == seal
    with pytest.raises(FileExistsError, match="immutable"):
        frozen.seal(tmp_path, fit_dir)
    (fit_dir / "ppo-seed17.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Sealed checkpoint changed"):
        frozen.validate_freeze(tmp_path)


def test_acquisition_and_access_refuse_without_explicit_pixel_go(tmp_path):
    with pytest.raises(ValueError, match="PIXEL GO"):
        frozen.acquire_and_prepare(tmp_path, pixel_go=False, infer_brain=False)
    with pytest.raises(ValueError, match="PIXEL GO"):
        frozen.prepare_access_comparison(tmp_path, tmp_path / "output", pixel_go=False)
    assert not (tmp_path / "output").exists()


def test_native_counter_charges_preview_internal_step_once():
    from resectionlab.surgical_engine import SurgicalEngine

    counter = frozen.NativeCallCounter(3, 10.)
    counter.begin_execution(max_calls=2, max_seconds=10.)
    preview = SimpleNamespace(f_code=SurgicalEngine.preview.__code__, f_back=None)
    internal = SimpleNamespace(f_code=SurgicalEngine.step.__code__, f_back=preview)
    direct = SimpleNamespace(f_code=SurgicalEngine.step.__code__, f_back=None)
    counter._profile(preview, "call", None)
    counter._profile(internal, "call", None)
    counter._profile(direct, "call", None)
    assert counter.calls == 2
    assert counter.internal_steps == 1
    with pytest.raises(TimeoutError, match="execution native-call budget"):
        counter._profile(direct, "call", None)
    assert counter.end_execution()["physicalCalls"] == 2


def test_shared_importer_keeps_frozen_role_in_first_source_receipt(tmp_path,
                                                                    monkeypatch):
    script = ROOT / "scripts/prepare_fresh_ucsf_demo.py"
    spec = importlib.util.spec_from_file_location("fresh_ucsf_role_test", script)
    fresh = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fresh)
    monkeypatch.setattr(fresh, "ROOT", tmp_path)
    monkeypatch.setattr(fresh, "SOURCE_ROOT", tmp_path / "testcases/frozen")
    case = "UCSF-PDGM-0008"
    content = b"synthetic metadata fixture, not a patient image"
    filename = case + "_T1.nii.gz"
    image = fresh.SOURCE_ROOT / case / filename
    image.parent.mkdir(parents=True)
    image.write_bytes(content)
    sha = hashlib.sha256(content).hexdigest()
    monkeypatch.setattr(fresh, "source_listing", lambda *_: [{
        "path": case + "/" + filename, "size": len(content),
        "lfs": {"oid": sha, "size": len(content)}, "type": "file"}])
    monkeypatch.setattr(fresh, "reusable_files", lambda *_: {})
    receipt = fresh.acquire(case, {"role": "FROZEN_ASSESSMENT",
        "mirror_revision": frozen.REVISION, "image_fetch_cap_bytes": 60_000_000,
        "role_declaration_sha256": "protocol-sha-fixture",
        "checkpoint_freeze_sha256": "freeze-sha-fixture"})
    assert receipt["role"] == "FROZEN_ASSESSMENT"
    assert receipt["role_declaration_sha256"] == "protocol-sha-fixture"
    assert receipt["checkpoint_freeze_sha256"] == "freeze-sha-fixture"
    assert json.loads((image.parent / "source-receipt.json").read_text())["role"] == \
        "FROZEN_ASSESSMENT"
    with pytest.raises(ValueError, match="supported role"):
        fresh.prepare(case, {"role": "TRAIN"}, infer_brain=False)


def test_identical_source_bytes_cannot_be_called_distinct_anatomy(tmp_path,
                                                                   monkeypatch):
    development = tmp_path / "development-declaration.json"
    development.write_text("{}")
    monkeypatch.setattr(frozen, "DEVELOPMENT", development)
    previous = development.parent / "UCSF-PDGM-0005" / "source-receipt.json"
    previous.parent.mkdir()
    previous.write_text(json.dumps({"files": [{"sha256": str(i)} for i in range(5)]}))
    with pytest.raises(ValueError, match="duplicate a DEVELOPMENT"):
        frozen._reject_duplicate_development_bytes(
            {"files": [{"sha256": str(i)} for i in range(5)]})


def test_first_acquisition_and_preparation_anchor_freeze_hash(tmp_path,
                                                                monkeypatch):
    monkeypatch.setattr(frozen, "ROOT", tmp_path)
    development = tmp_path / "testcases/downloaded-datasets/dev/development-declaration.json"
    development.parent.mkdir(parents=True)
    development.write_text("{}")
    monkeypatch.setattr(frozen, "DEVELOPMENT", development)
    package_root = tmp_path / "testcases"
    protocol_file = frozen.protocol_path(package_root)
    freeze_file = frozen.freeze_path(package_root)
    protocol_file.parent.mkdir(parents=True)
    protocol_file.write_text("fixed protocol")
    freeze_file.write_text("first freeze")
    monkeypatch.setattr(frozen, "validate_freeze", lambda *_: (
        {"missingAnatomy": ["skull"]}, {}))
    for patient in ("UCSF-PDGM-0005", "UCSF-PDGM-0007"):
        path = development.parent / patient / "source-receipt.json"
        path.parent.mkdir()
        path.write_text(json.dumps({"files": [{"sha256": patient + str(i)}
                                               for i in range(5)]}))
    image_dir = frozen.source_root(package_root) / frozen.CASE
    image_dir.mkdir()
    files = []
    for i in range(5):
        image = image_dir / f"synthetic-{i}.nii.gz"
        image.write_bytes(f"fixture {i}".encode())
        files.append({"path": str(image.relative_to(tmp_path)),
                      "sha256": frozen.digest(image)})
    receipt = {"role": "FROZEN_ASSESSMENT", "patient_id": frozen.CASE,
        "mirror_revision": frozen.REVISION,
        "role_declaration_sha256": frozen.digest(protocol_file),
        "checkpoint_freeze_sha256": frozen.digest(freeze_file),
        "files": files}
    (image_dir / "source-receipt.json").write_text(json.dumps(receipt))
    report_path = frozen.output_root(package_root) / frozen.CASE / "preparation-receipt.json"
    report_path.parent.mkdir(parents=True)
    preparation = {"source_role": "FROZEN_ASSESSMENT",
        "patient_id": frozen.CASE, "scene_reopened": True,
        "missing_anatomy": ["skull"],
        "checkpoint_freeze_sha256": frozen.digest(freeze_file)}
    report_path.write_text(json.dumps(preparation))
    assert frozen.validate_acquired_role(package_root) == (receipt, preparation)
    freeze_file.write_text("mutated freeze")
    with pytest.raises(ValueError, match="lost frozen assessment identity"):
        frozen.validate_acquired_role(package_root)

