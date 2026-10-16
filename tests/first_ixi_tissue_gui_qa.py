"""Narrow GUI-save followup to the immutable 17bb tissue bundle auditor.

Native Save adds artifacts.workspace display preferences to the primary
manifest. Validate that exact UI-only envelope; all source/archive/mask checks
continue through the unchanged base auditor. No source file is rewritten.
"""
from io import BytesIO
import importlib.util
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE_PATH = ROOT/"tests/first_ixi_tissue_bundle_qa.py"
BASE_SHA = "17bbbe128a116ae29388979e067e61ba2dd7260108331b05e7670a01e582c7b3"
EXPECTED_UI_ARTIFACT = {"case_hash": "sha256:72f6ae73183d8d5d52f30eafd90a533b0cf81a38e0070d7fa70ca954461c505c",
    "category": "pareto", "cursor": [0.8176703732460737, -1.997637267690152, 0.9734293008805253],
    "overlayOpacity": .32, "routeA": "", "routeB": "", "visibleLayers": {"brain": True}}
spec = importlib.util.spec_from_file_location("frozen_tissue_bundle_auditor", BASE_PATH)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
assert base.sha(BASE_PATH) == BASE_SHA
ADMISSION = base.admitted_records
RETAINED = {
    "tests/first_ixi_tissue_bundle_qa.py": BASE_SHA,
    "artifacts/ixi265-tissue-display-independent-v1/independent-qa.json": "817b26109d53a89930caa5c8e80572f92b218e9d9b5dab4f15fd6b0622d14155",
    "artifacts/ixi265-tissue-display-independent-v1/gui-save-initial-metadata-guard.json": "d05c8ea391932eb087eae042cb92bf9d0537a46e095add5e05bd889e0d6f7523",
    "artifacts/ixi265-tissue-display-independent-v1/gui-save-guard-followup-preparation.json": "4770a348e862c85643457cbe0458ff45997a35c717efbb18e66b7737630cd61e",
    "artifacts/ixi265-tissue-display-independent-v1/gui-save-initial-wrapper.py.txt": "37911b79fbae8a9e0a239a9e76fb8ebc2196ae5cf02e626f29133948a02007c6",
}


def admitted_records(pins):
    result = ADMISSION(pins)
    before = result[1]
    for relative, expected in RETAINED.items():
        recorded = base.stamp(ROOT/relative)
        assert recorded["sha256"] == expected
        before[relative] = recorded
    return result


def exact_ui_addition(prior, saved):
    assert prior == {}
    if saved == {}: return None
    assert saved == {"workspace": EXPECTED_UI_ARTIFACT}
    return saved["workspace"]


def check_primary_manifest(prior, saved):
    assert {k: v for k, v in prior.items() if k not in {"workspace_sha256", "artifacts"}} == {
        k: v for k, v in saved.items() if k not in {"workspace_sha256", "artifacts"}}
    return exact_ui_addition(prior.get("artifacts", {}), saved.get("artifacts", {}))


def check_cutaway(clipping):
    assert clipping["axis"] == "z" and clipping["enabled"] is True and clipping["flipped"] is False
    assert np.isfinite(clipping["value"]) and clipping["value"] == EXPECTED_UI_ARTIFACT["cursor"][2]


def generated_delta_controls():
    from copy import deepcopy
    assert exact_ui_addition({}, {}) is None
    assert exact_ui_addition({}, {"workspace": deepcopy(EXPECTED_UI_ARTIFACT)}) == EXPECTED_UI_ARTIFACT
    failures = 0
    for field, value in (("case_hash", "foreign"), ("category", "other"), ("cursor", [0, 0, 0]),
            ("overlayOpacity", .4), ("routeA", "planning-route"), ("routeB", "planning-route"),
            ("visibleLayers", {"brain": False}), ("extra", "new-state")):
        changed = deepcopy(EXPECTED_UI_ARTIFACT); changed[field] = value
        try: exact_ui_addition({}, {"workspace": changed})
        except AssertionError: failures += 1
    for previous, saved in (({"other": True}, {"workspace": EXPECTED_UI_ARTIFACT}),
            ({}, {"workspace": EXPECTED_UI_ARTIFACT, "other": True})):
        try: exact_ui_addition(previous, saved)
        except AssertionError: failures += 1
    assert failures == 10
    original = {"case_semantic_hash": EXPECTED_UI_ARTIFACT["case_hash"], "artifacts": {},
        "workspace_sha256": "old", "metadata": {"source": "observed"}, "source_refs": ["acquired"], "array_sha256": "fixed"}
    saved = deepcopy(original); saved["artifacts"] = {"workspace": deepcopy(EXPECTED_UI_ARTIFACT)}; saved["workspace_sha256"] = "new"
    assert check_primary_manifest(original, saved) == EXPECTED_UI_ARTIFACT
    source_failures = 0
    for field in ("case_semantic_hash", "metadata", "source_refs", "array_sha256"):
        changed = deepcopy(saved); changed[field] = "mutated"
        try: check_primary_manifest(original, changed)
        except AssertionError: source_failures += 1
    assert source_failures == 4
    cutaway = {"axis": "z", "enabled": True, "flipped": False, "value": EXPECTED_UI_ARTIFACT["cursor"][2]}
    check_cutaway(cutaway)
    flipped = dict(cutaway, flipped=True)
    try: check_cutaway(flipped)
    except AssertionError: pass
    else: raise AssertionError("Flipped cutaway passed")
    return {"exact_expected_delta_accepted": True, "ten_other_artifact_changes_refused": failures,
        "four_source_manifest_mutations_refused": source_failures, "exact_axial_cutaway_accepted": True,
        "flipped_cutaway_refused": True}


def preservation(reference, candidate, *, original_count=16, display_count=2, added_originals=0, added_displays=1):
    with ZipFile(reference) as a, ZipFile(candidate) as b:
        am, bm = (json.loads(z.read("manifest.json")) for z in (a, b))
        aw, bw = (json.loads(z.read("workspace.json")) for z in (a, b))
        ui_record = None
        ui = check_primary_manifest(am, bm)
        if ui is not None:
            assert ui["case_hash"] == bm["case_semantic_hash"]
            assert ui["category"] == "pareto" and ui["routeA"] == ui["routeB"] == ""
            assert ui["visibleLayers"] == {"brain": True} and ui["overlayOpacity"] == .32
            state = bw["imagingState"]["states"][bw["imagingState"]["selectedSeriesId"]]
            assert ui["cursor"] == state["cursor"] and len(ui["cursor"]) == 3 and np.isfinite(ui["cursor"]).all()
            expected_visibility = {"atropos_csf_like_estimate": False, "atropos_gray_matter_like_estimate": True,
                "atropos_white_matter_like_estimate": True, "unknown_missing_intensity": True}
            assert state["visibleLayers"] == expected_visibility
            clipping = state["viewer"]["clipping"]
            check_cutaway(clipping)
            ui_record = {"only_added_artifact_key": "workspace", "validated_workspace_preferences": ui,
                "source_or_planning_identity_change": False, "selected_visibility": expected_visibility,
                "visible_layer_count": 3, "total_layer_count": 4}
        assert aw["referenceCaseHash"] == bw["referenceCaseHash"]
        assert aw["primaryPlanningHash"] == bw["primaryPlanningHash"]
        primary = base.nested_members(a, b, "arrays.npz")
        assert len(aw["originalFiles"]) == original_count
        assert len(bw["originalFiles"]) == original_count+added_originals
        assert aw["originalFiles"] == bw["originalFiles"][:original_count]
        originals = []
        for row in aw["originalFiles"]:
            checked = base.exact_members(a, b, row["member"])
            assert checked == {"sha256": row["sha256"], "bytes": row["bytes"]}
            originals.append({"name": row["name"], "role": row["role"], **checked})
        assert len(aw["images"]) == display_count and len(bw["images"]) == display_count+added_displays
        sources = []
        for old, new in zip(aw["images"], bw["images"][:display_count], strict=True):
            assert old["source"] == new["source"]
            members = base.nested_members(a, b, old["member"], new["member"])
            assert base.digest(a.read(old["member"])) == old["sha256"]
            assert base.digest(b.read(new["member"])) == new["sha256"]
            sources.append({"source_identity": new["source"], "all_member_bytes_equal": True,
                "member_sha256": {k: v["sha256"] for k, v in members.items()}})
        return {"primary_source_manifest_exact": True, "primary_arrays": primary, "originals": originals,
            "preserved_reference_sources": sources, "source_and_planning_identity_unchanged": True,
            "validated_GUI_save_artifact_addition": ui_record, "frozen_base_auditor_sha256": BASE_SHA}


if __name__ == "__main__":
    generated_delta_controls()
    base.preservation = preservation
    base.admitted_records = admitted_records
    # The frozen base's release pin and report now bind this wrapper, which in
    # turn verifies its immutable base source before any payload operation.
    base.__file__ = __file__
    base.main()
