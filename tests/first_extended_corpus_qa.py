"""Opt-in independent ten-person release audit: metadata only.

--prepare exercises generated guard/search controls without freezing a result.
A final run requires externally supplied catalog/source pins and a separate
independent reader result. No real images, bundles, weights or solvers are opened.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import nibabel

from resectionlab import patient_corpus as pc

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT/"outputs/patient-corpus-ixi265-v1"
RELEASE = ROOT/"manifests/patient-corpus-releases/existing-nine-plus-ixi265-v1.json"
RELEASE_SHA = "2c426f82e160be5b11bb19d9d5c4d2ac804ae9d85011cf46a4dc83904adbea84"
PLAN_HASH = "sha256:a1a2623d05270881d4291a1037d303578cb27a4f63d52cce28b5f5db251d8755"
PEOPLE = {**{f"BTC:sub-PAT{x}": "TRAIN" for x in ("05", "16", "20", "22", "25", "28")},
    "UCSF-PDGM:004": "DEVELOPMENT", "SynthRAD2023:1BA336": "TRAIN", "Lausanne:sub-000": "TRAIN", "IXI:265": "TRAIN"}
FROZEN = {
    "artifacts/patient-corpus-first-tranche-v1/independent-qa.json": "5f112d987639408911b379bd72792cab2504f68599981ef833445e31bcf7e0bd",
    "artifacts/ixi265-virtual-pilot-v1/independent-qa.json": "61e449b05aaf1396e7383a23bb903238ab3e7e223899c23a0ed64e40c814eef0",
    "artifacts/ixi265-virtual-pilot-v1/gui-and-vessel-independent-qa.json": "0a1700dd15a784405d47b1d66aa7713eb6dab3cc92faf3179bc608a6da36d31b",
}
WATCHED = {}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stamp(path):
    path = Path(path)
    return {"sha256": sha(path), "bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}


def read_json(path):
    path = Path(path)
    assert path.suffix == ".json" and path.stat().st_size < 16*1024**2
    WATCHED[str(path.relative_to(ROOT))] = stamp(path)
    return json.loads(path.read_text())


def digest(value):
    return "sha256:"+hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def sealed(value, key):
    assert value[key] == digest({k: v for k, v in value.items() if k != key}), key


def forbidden(*args, **kwargs):
    raise AssertionError("Payload access attempted by metadata-only independent audit")


@contextmanager
def metadata_only():
    with ExitStack() as stack:
        for name in ("_payload_path", "_manifest_preflight", "_case_layout", "load_workspace", "inspect_nifti", "_qualification", "array_digest", "file_sha256"):
            stack.enter_context(patch.object(pc, name, forbidden))
        stack.enter_context(patch.object(nibabel, "load", forbidden))
        yield


def rejected(call):
    try:
        call()
    except pc.QualificationError as error:
        return error.code
    raise AssertionError("Invalid declaration was admitted")


def guard_controls(frozen, plan):
    candidate = deepcopy(frozen["additional_candidate"])
    blocked = [
        {"roles": {"SELECT": True}}, {"parent": {"split": {"name": "SELECT"}}},
        {"split": {"SELECT": 1}}, {"roles": {"SELECT": "false"}},
        {"parent": [{"usageRole": "MEASUREMENT_EVAL"}]}, {"partition": {"name": "held-out"}},
        {"source_collection": {"name": "UPENN-GBM"}}, {"ancestry": [{"dataset_name": "UTSW"}]},
        {"parent": {"patient_group": "BTC:sub-PAT26"}}, {"parent": {"case_id": "ReMIND-013"}},
        {"patient_group": "IXI:266"}, {"virtual_geometry": {}},
        {"virtual_glioma": {"host_patient_group": "IXI:265"}}, {"generated_descendant": True},
        {"anatomy_layers": {"target": {"kind": "tumor", "provenance": "generated"}}},
        {"anatomy_layers": {"target": {"kind": "tumor", "provenance": "simulated_geometry"}}},
    ]
    rows = []
    with metadata_only(), patch.object(pc, "_frozen_release", lambda *args: (deepcopy(frozen), {})):
        for metadata in blocked:
            manifest = {"case_id": candidate["case_id"], "metadata": {"is_synthetic": False, **metadata}}
            rows.append({"declaration": metadata, "rejection": rejected(lambda: pc.require_manifest(manifest, candidate, pc.IXI_RELEASE))})
        for provenance in ("simulated", "generated"):
            manifest = {"case_id": candidate["case_id"], "metadata": {"is_synthetic": False}, "source_refs": [{"provenance": provenance}]}
            rows.append({"source_provenance": provenance, "rejection": rejected(lambda: pc.require_manifest(manifest, candidate, pc.IXI_RELEASE))})
        # An estimate whose model learned from simulations remains an estimate.
        positive = {"case_id": candidate["case_id"], "metadata": {"is_synthetic": False,
            "roles": {"SELECT": False, "TRAIN": True}, "virtual_geometry": None,
            "anatomy_layers": {"vessels": {"provenance": "estimated", "model": {"training": {"provenance": "simulated"}}}}},
            "source_refs": [{"provenance": "observed"}, {"provenance": "estimated"}]}
        pc.require_manifest(positive, candidate, pc.IXI_RELEASE)
        changes = {"patient_group": "IXI:266", "role": "SELECT", "bundle": "outputs/other.ressectionlab",
            "bundle_sha256": "0"*64, "training_admitted": True, "fusion_permitted": True,
            "registration_acceptance": True, "accepted_constraints": True, "existing_materialized": False}
        candidate_rows = []
        for key, value in changes.items():
            changed = {**candidate, key: value}
            candidate_rows.append({"field": key, "rejection": rejected(lambda: pc.require_candidate(changed, pc.IXI_RELEASE))})
        nested = deepcopy(candidate)
        nested["source_files"][0]["sha256"] = "0"*64
        candidate_rows.append({"field": "source_files[0].sha256", "rejection": rejected(lambda: pc.require_candidate(nested, pc.IXI_RELEASE))})
        default_reject = rejected(lambda: pc.require_candidate(candidate))
        plan_rows = []
        for name in ("drop_person", "duplicate_person", "source_hash", "scope", "metadata", "unknown_release"):
            changed = deepcopy(plan)
            if name == "drop_person": changed["candidates"].pop()
            elif name == "duplicate_person": changed["candidates"][0] = deepcopy(changed["candidates"][1])
            elif name == "source_hash": changed["candidates"][0]["bundle_sha256"] = "f"*64
            elif name == "scope": changed["new_patient_acquisition_allowed"] = True
            elif name == "metadata": changed["metadata_evidence"] = []
            else: changed["release_id"] = "arbitrary-new-release"
            changed["plan_hash"] = digest({k: v for k, v in changed.items() if k != "plan_hash"})
            plan_rows.append({"change": name, "rejection": rejected(lambda: pc.validate_plan(changed))})
    return {"status": "pass", "generated_metadata_controls": rows, "candidate_equality_controls": candidate_rows,
        "self_resigned_plan_controls": plan_rows, "ixi_refused_by_default_nine_release": default_reject,
        "estimated_layer_with_simulated_model_training_allowed": True, "payload_decoders_and_source_openers_replaced_with_failure_sentinels": True}


def scoped_search_controls():
    rows = [{"patient_group": "BTC:sub-PAT05", "case_id": "BTC-sub-PAT05", "role": "TRAIN", "status": "qualified_with_limitations",
        "anatomy_present": ["brain", "tumor"], "input_binding": {"metadata_evidence": [{"path": "IXI265 shared provenance"}]}},
        {"patient_group": "IXI:265", "case_id": "IXI265-Guys-0845", "role": "TRAIN", "status": "qualified_with_limitations",
         "anatomy_present": ["brain", "vessels"], "imaging": [{"modality": "MRA"}]}]
    with tempfile.TemporaryDirectory(prefix="independent-metadata-search-") as temporary, metadata_only():
        target = Path(temporary)
        pc._write_search_index(target/"catalog.sqlite", {"entries": rows})
        assert [r["patient_group"] for r in pc.query_catalog(target, text="ixi")] == ["IXI:265"]
        assert [r["patient_group"] for r in pc.query_catalog(target, anatomy="tumor")] == ["BTC:sub-PAT05"]
        assert not pc.query_catalog(target, text="' OR 1=1 --")
        assert not pc.query_catalog(target, modality="MRA", anatomy="tumor")
        assert len(pc.query_catalog(target, modality="MRA", role="TRAIN")) == 1
    return {"status": "pass", "generated_metadata_only": True, "shared_release_evidence_excluded_from_person_search": True,
        "literal_query_and_conjunctive_filters_verified": True}


def catalog_audit(plan):
    catalog = read_json(OUTPUT/"catalog.json")
    baseline = read_json(ROOT/"outputs/patient-corpus-v1/catalog.json")
    sealed(baseline, "catalog_hash")
    assert len(baseline["entries"]) == 9
    original_entries = {row["patient_group"]: row for row in baseline["entries"]}
    sealed(catalog, "catalog_hash")
    assert catalog["plan_hash"] == PLAN_HASH and catalog["release_id"] == pc.IXI_RELEASE
    assert {r["patient_group"]: r["role"] for r in catalog["entries"]} == PEOPLE
    assert len(catalog["entries"]) == len(PEOPLE) == 10
    candidates = {r["patient_group"]: r for r in plan["candidates"]}
    summary, source_provided_overlays = [], []
    for row in catalog["entries"]:
        receipt = read_json(OUTPUT/row["receipt_path"])
        sealed(receipt, "receipt_hash")
        assert receipt == {k: v for k, v in row.items() if k != "receipt_path"}
        assert row["input_binding_hash"] == digest(row["input_binding"])
        binding = row["input_binding"]
        candidate = candidates[row["patient_group"]]
        assert binding["candidate"] == candidate and binding["plan_hash"] == PLAN_HASH
        assert row["bundle_sha256"] == binding["bundle_sha256"] == candidate["bundle_sha256"]
        assert row["bundle"] == candidate["bundle"] and row["rights"] == candidate["rights"]
        for name, expected in binding["validation_code_sha256"].items():
            assert sha(ROOT/"src/resectionlab"/name) == expected, "Receipt validator binding is stale: "+name
        declared = {r["sha256"] for r in candidate["source_files"]}
        assert all(image["source_sha256"] in declared and image["source_voxels_equal"] for image in row["imaging"])
        assert all(not row[key] for key in ("new_patient", "training_admitted", "training_admission_established", "clinical_use_permitted", "virtual_descendant"))
        assert all(row["tumor_location"][key] == "UNKNOWN" for key in ("lobe", "hemisphere", "anatomical_region"))
        assert all(not layer["accepted_anatomical_truth"] for layer in row["layers"])
        for registration in row["registrations"]:
            assert not registration["independently_validated_registration"] and not registration["anatomical_registration_accepted"]
            if registration["overlayPermitted"]:
                # Existing UCSF modalities have source-declared co-registration.
                # This display permission is distinct from independent anatomy acceptance.
                assert row["patient_group"] == "UCSF-PDGM:004" and registration["status"] == "source-provided"
                assert registration["source_frame_hash"] == registration["reference_frame_hash"]
                old = next(item for item in original_entries[row["patient_group"]]["registrations"]
                    if item["source_modality"] == registration["source_modality"])
                assert all(registration[key] == value for key, value in old.items())
                source_provided_overlays.append(registration["source_modality"])
        assert row["anatomy_present"] == sorted({layer["kind"] for layer in row["layers"] if layer["present"] and layer["kind"] in pc.ANATOMY})
        assert row["missing_anatomy"] == sorted(set(pc.ANATOMY)-set(row["anatomy_present"]))
        summary.append({"patient_group": row["patient_group"], "role": row["role"], "receipt_hash": row["receipt_hash"],
            "bundle_sha256": row["bundle_sha256"], "anatomy_present": row["anatomy_present"], "primary_anatomy": row["anatomy_present_on_primary"]})
    counts = catalog["counts"]
    assert counts["existing_materialized_people"] == counts["qualified_existing_people"] == 10
    for key in ("newly_acquired_people", "anatomical_truth_accepted_people", "virtual_descendants_counted_as_people", "pending", "rejected"):
        assert counts[key] == 0
    assert counts["newly_cataloged_existing_people"] == 1
    assert counts["verified_embedded_original_files"] == sum(row["originals"]["embedded_count"] for row in catalog["entries"]) == 56
    assert counts["people_with_portable_source_originals"] == sum(row["originals"]["portable_originals_complete"] for row in catalog["entries"]) == 4
    combinations = {name: sum(set(required) <= set(row["anatomy_present"]) for row in catalog["entries"]) for name, required in {
        "brain_plus_tumor": ["brain", "tumor"], "brain_plus_skull": ["brain", "skull"],
        "brain_plus_vessels": ["brain", "vessels"], "brain_tumor_skull_vessels": ["brain", "tumor", "skull", "vessels"]}.items()}
    assert combinations == counts["anatomical_presence_combinations"] == counts["same_person_presence_across_saved_grids"]
    assert combinations == {"brain_plus_tumor": 7, "brain_plus_skull": 1, "brain_plus_vessels": 1, "brain_tumor_skull_vessels": 0}
    assert not any(counts["accepted_fused_anatomy_combinations"].values()) and not catalog["protected_imaging_opened"]
    ixi = next(row for row in catalog["entries"] if row["patient_group"] == "IXI:265")
    assert ixi["anatomy_present_on_primary"] == ["brain"] and len(ixi["registrations"]) == 3
    assert all(not item["overlayPermitted"] for item in ixi["registrations"])
    assert sorted(source_provided_overlays) == ["FLAIR", "T1", "T2"]
    vessel_layers = [layer for layer in ixi["layers"] if layer["kind"] == "vessels"]
    assert len(vessel_layers) == 3 and all(layer["scope"] == "display-only-native-grid" and not layer["accepted_constraint"] for layer in vessel_layers)
    assert sorted(layer["provenance"] for layer in vessel_layers) == ["estimated", "source-provided", "source-provided"]
    with metadata_only():
        assert len(pc.query_catalog(OUTPUT)) == 10
        assert [r["patient_group"] for r in pc.query_catalog(OUTPUT, text="IXI")] == ["IXI:265"]
        assert len(pc.query_catalog(OUTPUT, anatomy="tumor")) == 7
        assert len(pc.query_catalog(OUTPUT, anatomy="skull")) == 1
        assert len(pc.query_catalog(OUTPUT, anatomy="vessels")) == 2
        assert not pc.query_catalog(OUTPUT, text="UPENN")
    return {"status": "pass", "catalog_hash": catalog["catalog_hash"], "counts": counts, "entries": summary,
        "receipt_seals_candidate_equality_validator_hashes_and_declared_source_lineage_verified": True,
        "unknown_locations_and_absence_of_independent_registration_acceptance_preserved": True,
        "unchanged_UCSF_source_declared_display_overlays": source_provided_overlays,
        "all_IXI_overlay_permissions_false": True, "same_person_cross_grid_presence_is_not_fusion": True,
        "scope": "Metadata consistency only; saved source equality and embedded-original claims were not rechecked against real payloads here"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--catalog-sha256")
    parser.add_argument("--corpus-source-sha256")
    parser.add_argument("--reader-audit", type=Path)
    args = parser.parse_args()
    frozen = read_json(RELEASE)
    assert sha(RELEASE) == RELEASE_SHA
    for name, expected in FROZEN.items():
        read_json(ROOT/name)
        assert sha(ROOT/name) == expected
    with metadata_only():
        plan = pc.release_plan(ROOT, pc.IXI_RELEASE)
        base = pc.release_plan(ROOT)
    sealed(plan, "plan_hash")
    assert plan["plan_hash"] == PLAN_HASH and len(base["candidates"]) == 9
    assert {row["patient_group"]: row["role"] for row in plan["candidates"]} == PEOPLE
    assert plan["candidates"] == sorted([*frozen["base_candidates"], frozen["additional_candidate"]], key=lambda r: r["patient_group"])
    controls = guard_controls(frozen, plan)
    search = scoped_search_controls()
    if args.prepare:
        print(json.dumps({"status": "controls_pass_final_pins_pending", "guard_controls": controls, "search_controls": search}, indent=2))
        return
    if not all((args.catalog_sha256, args.corpus_source_sha256, args.reader_audit)):
        parser.error("Final audit requires externally frozen catalog/source hashes and independent reader audit")
    assert sha(OUTPUT/"catalog.json") == args.catalog_sha256
    assert sha(ROOT/"src/resectionlab/patient_corpus.py") == args.corpus_source_sha256
    reader = read_json(args.reader_audit.resolve())
    assert reader["status"] == "pass" and reader["real_payloads_opened"] == 0
    for name, expected in reader["implementation_sha256"].items():
        assert sha(ROOT/name) == expected
    catalog = catalog_audit(plan)
    assert all(stamp(ROOT/name) == original for name, original in WATCHED.items())
    report = {"schema": "independent-ten-person-corpus-audit-v1", "status": "pass", "created_utc": datetime.now(timezone.utc).isoformat(),
        "release_sha256": RELEASE_SHA, "plan_hash": PLAN_HASH, "catalog_sha256": args.catalog_sha256,
        "corpus_source_sha256": args.corpus_source_sha256, "auditor_sha256": sha(Path(__file__)),
        "metadata_guards": controls, "scoped_search": search, "catalog": catalog,
        "reader_audit": {"path": str(args.reader_audit.resolve().relative_to(ROOT)), "sha256": sha(args.reader_audit), "result": reader},
        "input_stamps": WATCHED, "prior_frozen_audits_unchanged": True, "real_payloads_opened": 0,
        "auditor_adjustments": ["Initial overly broad overlay-disabled oracle was corrected to preserve the three existing UCSF source-declared, equal-grid display overlays. Their registration records exactly match the nine-person baseline and still carry no independent anatomical acceptance; all IXI overlays remain disabled."],
        "models_or_solvers_executed": False, "clinical_or_independent_accuracy_established": False}
    content = json.dumps(report, indent=2)+"\n"
    assert "/Users/" not in content
    destination = ROOT/"artifacts/patient-corpus-ten-person-v1/independent-qa.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(content)
    print(json.dumps({"status": "pass", "artifact": str(destination.relative_to(ROOT)), "sha256": sha(destination)}))


if __name__ == "__main__":
    main()
