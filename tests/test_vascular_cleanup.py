import base64
import json
from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest

from resectionlab.anatomy_bridge import VASCULAR_REVIEW_MANIFEST, dispatch, open_anatomy_manifest
from resectionlab.anatomy_package import open_package, save_package
from resectionlab.anatomy_workspace import save_view
from resectionlab.desktop_bridge import BridgeSession, _Request
from resectionlab.patient_anatomy import PatientImagingCollection, digest, file_hash
from resectionlab.vascular_cleanup import (
    HIDDEN, RETAINED, UNCERTAIN, attached_vessel_uncertainty, classify_components, review_component_visibility,
    separate_reviewed_components,
)


def test_native_review_preserves_pial_and_bright_extracranial_candidates():
    shape = (64, 64, 48)
    axes = np.ogrid[:shape[0], :shape[1], :shape[2]]
    inside = sum(((axis - center) / radius) ** 2 for axis, center, radius in zip(
        axes, (32, 32, 24), (16, 17, 15)
    )) < 1
    signal = np.where(inside, 1000, 5).astype(np.float32)
    signal[32, 14:50, 24] = np.linspace(1500, 3000, 36)
    signal[12:14, 14:17, 24] = 1450
    mask = np.zeros(shape, dtype=bool)
    mask[29:36, 14:50, 24] = True
    mask[12:14, 14:17, 24] = True
    mask[48:54, 5:9, 24] = True
    scores = np.where(mask, 0.94, 0.1)
    frame = np.diag([0.5, 0.5, 0.65, 1.0])

    partition, report = classify_components(mask, signal, scores, frame)

    np.testing.assert_array_equal(partition > 0, mask)
    assert np.all(partition[29:36, 14:50, 24] == RETAINED)
    assert np.all(partition[12:14, 14:17, 24] == UNCERTAIN)
    assert np.all(partition[48:54, 5:9, 24] == HIDDEN)
    component = next(row for row in report["components"] if row["state"] == HIDDEN)
    restored, revised = review_component_visibility(partition, report, restore=[component["id"]])
    np.testing.assert_array_equal(restored > 0, mask)
    assert not np.any(restored == HIDDEN)
    assert report["hidden_voxels"] == component["voxels"]
    assert revised["uncertain_voxels"] == report["uncertain_voxels"] + component["voxels"]
    excluded, rereviewed = review_component_visibility(restored, revised, exclude=[component["id"]])
    np.testing.assert_array_equal(excluded, partition)
    assert rereviewed["hidden_voxels"] == report["hidden_voxels"]
    with pytest.raises(ValueError, match="Unknown source component"):
        review_component_visibility(partition, report, restore=["component-0"])


def test_no_source_support_abstains_and_preserves_all_components():
    mask = np.zeros((20, 20, 20), dtype=bool)
    mask[5:12, 5, 5] = True
    mask[12, 12, 12] = True
    partition, report = classify_components(mask, np.ones(mask.shape), None, np.eye(4))
    np.testing.assert_array_equal(partition > 0, mask)
    assert report["hidden_components"] == 0
    assert report["reference_signal_available"] is False


def _attached_fixture():
    shape = (80, 76, 44)
    mask = np.zeros(shape, dtype=bool)
    mask[14:61, 31:40, 17:27] = True
    weak = np.zeros(shape, dtype=bool)
    weak[31:40, 40:48, 18:26] = True       # Wide attached weak-signal lobe.
    weak[32, 48:58, 20] = True             # Narrow low-signal continuation.
    weak[7:14, 54:61, 17:24] = True        # Detached lobe.
    weak[68:75, 1:8, 1:8] = True           # Lobe touching the native extent guard.
    bright = np.zeros(shape, dtype=bool)
    bright[22, 39:52, 22] = True           # Small bright branch.
    bright[34:42, 25:31, 17:27] = True     # Substantial bright continuation.
    bright[67:69, 4, 4] = True
    mask |= weak | bright
    signal = np.broadcast_to(np.linspace(60, 160, shape[0], dtype=np.float32)[:, None, None], shape).copy()
    signal[weak] = 6
    signal[bright] = 300
    scores = np.where(mask, 0.93, 0.03).astype(np.float32)
    frame = np.diag([.4, .4, .5, 1.0])
    frame[:3, 3] = [9, -3, 71]
    return mask, signal, scores, frame, bright


def test_attached_screen_keeps_thin_detached_bright_and_boundary_controls_out_of_overlay():
    mask, signal, scores, frame, bright = _attached_fixture()
    originals = [array.copy() for array in (mask, signal, scores)]
    for array in (mask, signal, scores):
        array.setflags(write=False)
    cells, report = attached_vessel_uncertainty(mask, signal, scores, frame)

    assert report["status"] == "complete"
    finding, = report["findings"]
    ix, iy, iz = np.unravel_index(cells[finding["id"]], mask.shape)
    assert np.all((31 <= ix) & (ix < 40) & (40 <= iy) & (iy < 48) & (18 <= iz) & (iz < 26))
    assert np.all(mask[ix, iy, iz])
    assert not np.any(bright[ix, iy, iz])
    assert len(ix) > 300
    assert finding["volumeMm3"] == pytest.approx(len(ix) * .4 * .4 * .5)
    assert finding["uncalibratedModelScoreMedian"] == pytest.approx(.93)
    np.testing.assert_allclose(finding["centerRasMm"], frame[:3, 3] + np.diag(frame)[:3] * finding["centerIjk"])
    scaled, _ = attached_vessel_uncertainty(mask, signal * 3 + 12, scores, frame)
    np.testing.assert_array_equal(scaled[finding["id"]], cells[finding["id"]])
    for original, array in zip(originals, (mask, signal, scores)):
        np.testing.assert_array_equal(original, array)


def test_attached_review_uses_bound_native_sections_without_writes(tmp_path):
    mask, signal, scores, frame, _ = _attached_fixture()
    image = nib.Nifti1Image(signal, frame)
    image.header.set_xyzt_units("mm")
    scan_path = tmp_path / "tof.nii"
    nib.save(image, scan_path)
    collection = PatientImagingCollection.create(tmp_path / "study", "native-attached-tof")
    source = collection.add_series(scan_path, series_id="tof", modality="TOF")
    layer = collection.add_layer(
        mask.astype(np.uint8), source["affine"], layer_id="estimate", kind="vessels", source_series="tof",
        role="context", color="#b83946",
        provenance={"model": "vesselfm", "source": source["original"]["sha256"]}, probability=scores,
    )
    collection.save()
    manifest = collection.root / "patient-anatomy.json"
    initial = {file: file_hash(file) for file in collection.root.rglob("*") if file.is_file()}
    bridge = BridgeSession(tmp_path / "transfers")

    def call(op, args):
        return bridge.execute(op, args, _Request("tof-attachment"), lambda *_: None)

    opened = call("anatomyOpen", {"path": str(manifest)})
    args = {"collectionId": opened["collectionId"], "snapshotId": collection.identity(),
            "seriesId": "tof", "layerId": "estimate"}
    op = "anatomyAttachedVesselReview"
    report = call(op, args)
    finding, = report["findings"]
    assert report["sourceSha256"] == source["original"]["sha256"]
    assert report["sourceScoreSha256"] == layer["probability"]["sha256"]
    assert report["sourceDataSha256"] == layer["data"]["sha256"]
    with pytest.raises(ValueError, match="Anatomy changed"):
        call(op, {**args, "snapshotId": "stale"})
    with pytest.raises(ValueError, match="refresh the findings"):
        call(op, {**args, "reviewId": "stale", "findingId": finding["id"]})
    with pytest.raises(ValueError, match="native TOF"):
        call(op, {**args, "seriesId": "other-scan"})
    assert {**report} == call(op, args)
    sections = report["preview"]
    assert sections == call(op, {**args, "reviewId": report["reviewId"], "findingId": finding["id"]})["preview"]
    assert len(sections["sections"]) == 3
    for plane in sections["sections"]:
        assert plane["index"] == finding["centerIjk"][plane["axis"]]
        assert len(base64.b64decode(plane["image"])) == np.prod(plane["shape"])
        assert 2 in base64.b64decode(plane["overlay"])
    assert bridge.anatomy_drafts == {}
    assert initial == {file: file_hash(file) for file in collection.root.rglob("*") if file.is_file()}
    assert not list(bridge.transfers.root.iterdir())
    with collection.path(source["original"]["path"]).open("ab") as stream:
        stream.write(b"source changed")
    with pytest.raises(ValueError, match="TOF checksum differs"):
        call(op, args)


def test_native_attached_edit_is_local_and_preserves_other_score_crossings(tmp_path):
    from resectionlab.vascular_cleanup import ATTACHED_ERASE, ATTACHED_SCORE_SURFACE

    mask, signal, scores, frame, bright = _attached_fixture()
    scan = nib.Nifti1Image(signal, frame)
    scan.header.set_xyzt_units("mm")
    path = tmp_path / "signal.nii"
    nib.save(scan, path)
    collection = PatientImagingCollection.create(tmp_path / "study", "attached-erase")
    collection.add_series(path, series_id="reference", modality="T1")
    source = collection.add_series(path, series_id="tof", modality="TOF")
    source_layer = collection.add_layer(mask.astype(np.uint8), source["affine"], layer_id="estimate", kind="vessels",
        source_series="tof", role="context", color="#b83946", probability=scores,
        provenance={"model": "vesselfm", "source": source["original"]["sha256"]})
    collection.save()
    manifest = collection.root / "patient-anatomy.json"
    initial_hash = file_hash(manifest)
    bridge = BridgeSession(tmp_path / "transfers")

    def call(op, args):
        return bridge.execute(op, args, _Request("native-attached-edit"), lambda *_: None)

    def view(**extra):
        return call("anatomyView", {**args, "layers": ["estimate"], "vesselSurfaceMode": "source", **extra})["compartments"][0]

    def array(descriptor):
        return np.fromfile(descriptor["path"], dtype=descriptor["dtype"]).reshape(descriptor["shape"])

    opened = call("anatomyOpen", {"path": str(manifest)})
    args = {"collectionId": opened["collectionId"], "snapshotId": collection.identity(), "seriesId": "tof"}
    selection = {**args, "layerId": "estimate"}
    report = call("anatomyAttachedVesselReview", selection)
    finding, = report["findings"]
    cells, _ = attached_vessel_uncertainty(mask, signal, scores, frame)
    positions = np.column_stack(np.unravel_index(cells[finding["id"]], mask.shape))
    center = finding["centerIjk"]
    radius = 1.31
    delta = (positions - center) @ frame[:3, :3].T
    expected = mask.copy()
    expected[tuple(positions[np.linalg.norm(delta, axis=1) <= radius].T)] = False
    assert 0 < np.count_nonzero(mask & ~expected) < len(positions)
    assert np.all(expected[bright])

    pick = {**selection, "reviewId": report["reviewId"], "findingId": finding["id"]}
    recentered = call("anatomyAttachedVesselReview", {**pick, "centerIjk": positions[-1].tolist()})["preview"]
    assert [section["index"] for section in recentered["sections"]] == positions[-1].tolist()
    for section in recentered["sections"]:
        axis = section["axis"]
        u, v = [d for d in range(3) if d != axis]
        offset = positions[-1] - section["startIjk"]
        assert base64.b64decode(section["overlay"])[offset[v] * section["shape"][1] + offset[u]] == 2
    original = view()
    raw_positions = array(original["surface"]["positions"])
    edit = {**pick, "action": ATTACHED_ERASE, "points": [center], "radiusMm": radius}
    with pytest.raises(ValueError, match="refresh the findings"):
        call("anatomyEdit", {**edit, "reviewId": "stale"})
    with pytest.raises(ValueError, match="amber voxel"):
        call("anatomyEdit", {**edit, "points": [[22, 41, 22]]})
    with pytest.raises(ValueError, match="native TOF acquisition"):
        call("anatomyEdit", {**edit, "seriesId": "reference"})
    assert bridge.anatomy_drafts == {}
    preview = call("anatomyEdit", edit)
    assert preview["draft"]["mode"] == ATTACHED_ERASE
    assert preview["draft"]["changedVoxels"] == np.count_nonzero(mask & ~expected)
    with pytest.raises(ValueError, match="changing editing tools"):
        call("anatomyEdit", {**selection, "action": "threshold"})
    changed = view()
    assert changed["surfaceDiagnostics"]["surfaceField"] == "filtered-native-probability"
    np.testing.assert_array_equal(array(changed["mask"]), expected)
    changed_positions = array(changed["surface"]["positions"])
    assert not np.array_equal(changed_positions, raw_positions)
    np.testing.assert_array_equal(changed_positions[changed_positions[:, 0] < 26], raw_positions[raw_positions[:, 0] < 26])
    assert view(original=True)["surface"]["positions"]["sha256"] == original["surface"]["positions"]["sha256"]
    assert call("anatomyEdit", {**selection, "action": "undo"})["draft"]["changedVoxels"] == 0
    assert call("anatomyEdit", {**selection, "action": "redo"})["draft"]["changedVoxels"] == preview["draft"]["changedVoxels"]
    call("anatomyEdit", {**selection, "action": "reject"})
    assert file_hash(manifest) == initial_hash
    call("anatomyEdit", edit)
    committed = call("anatomyEdit", {**selection, "action": "commit", "author": "source reviewer"})
    assert committed["document"]["transforms"] == opened["document"]["transforms"]
    bridge.close()

    saved = PatientImagingCollection(collection.root)
    layer = saved.layer("estimate")
    assert layer["edits"][-1]["surface_policy"] == ATTACHED_SCORE_SURFACE
    assert layer["edits"][-1]["operations"][0]["reviewId"] == report["reviewId"]
    assert layer["probability"] == source_layer["probability"]
    assert layer["original_data"] == source_layer["data"]
    np.testing.assert_array_equal(saved.mask(layer), expected)
    bridge = BridgeSession(tmp_path / "cold")
    opened = call("anatomyOpen", {"path": str(manifest)})
    args = {**args, "collectionId": opened["collectionId"], "snapshotId": saved.identity()}
    cold = view()
    assert cold["surface"]["positions"]["sha256"] == changed["surface"]["positions"]["sha256"]
    assert cold["surfaceDiagnostics"]["surfaceField"] == "filtered-native-probability"
    bridge.close()


def test_native_view_hides_only_detached_exterior_air_and_restores_complete_scores(tmp_path):
    shape = (64, 64, 64)
    distance2 = np.sum((np.indices(shape) - 32) ** 2, axis=0)
    signal = ((distance2 < 15**2) & (distance2 > 10**2)).astype(np.float32) * 1000
    signal[10:12, 51, 49] = 1000
    base = np.zeros(shape, dtype=bool)
    base[14:27, 32, 32] = True     # Attached route with a dark tip.
    base[32:37, 32, 33] = True     # Dark interior thin route.
    base[10:12, 50, 49] = True     # Near a separately measured source signal.
    base[1:4, 20, 10] = True       # Close to an acquisition face.
    base[9:13, 52, 20] = True      # Isolated exterior air with a complete native neighborhood.
    image = nib.Nifti1Image(signal, np.eye(4))
    image.header.set_xyzt_units("mm")
    scan_path = tmp_path / "tof.nii"
    nib.save(image, scan_path)
    c = PatientImagingCollection.create(tmp_path / "study", "native-tof-fixture")
    source = c.add_series(scan_path, series_id="tof", modality="TOF")
    estimate = c.add_layer(
        base.astype(np.uint8), np.eye(4), layer_id="estimate", kind="vessels", source_series="tof",
        role="context", color="#b83946",
        provenance={"model": "vesselfm", "source": source["original"]["sha256"]},
        probability=np.where(base, .9, .1).astype(np.float32),
    )
    c.save()
    original_snapshot = file_hash(c.root / "patient-anatomy.json")
    expected = base.copy()
    expected[9:13, 52, 20] = False

    def check_workspace(root, saved_scope=None):
        bridge = BridgeSession(tmp_path / "transfers")

        def call(op, args):
            return bridge.execute(op, args, _Request("tof-exterior"), lambda *_: None)

        opened = call("anatomyOpen", {"path": str(root / "patient-anatomy.json")})
        if saved_scope is not None:
            assert opened["viewState"]["vesselScope"] == saved_scope
        args = {"collectionId": opened["collectionId"], "snapshotId": c.identity(), "seriesId": "tof",
                "layers": ["estimate"]}
        for scope, raw, mask in (("exterior-air", False, expected), ("all", False, base), ("exterior-air", True, base)):
            layer = call("anatomyView", {**args, "vesselScope": scope, "original": raw})["compartments"][0]
            descriptor = layer["mask"]
            actual = np.fromfile(descriptor["path"], dtype=descriptor["dtype"]).reshape(descriptor["shape"])
            np.testing.assert_array_equal(actual, mask)
            if scope == "exterior-air" and not raw:
                assert layer["displayFilter"]["hiddenVoxels"] == 4
                assert layer["displayFilter"]["excludedFragments"] == 1
                assert layer["displayFilter"]["rawDataSha256"] == estimate["data"]["sha256"]
                assert layer["displayFilter"]["sourceProbabilitySha256"] == estimate["probability"]["sha256"]
                assert layer["surfaceDiagnostics"]["surfaceField"] == "filtered-native-probability"
            else:
                assert layer["displayFilter"] is None
                assert layer["surfaceDiagnostics"]["surfaceField"] == "native-probability"

    check_workspace(c.root)
    save_view(c, {"schema": "patient-anatomy-view-v1", "patientId": c.document["patient_id"],
                  "snapshotId": c.identity(), "seriesId": "tof", "states": {},
                  "vesselDisplayVersion": 2, "vesselScope": "exterior-air"})
    package = tmp_path / "study.rsworkspace"
    save_package(c, package)
    reopened = open_package(package, tmp_path / "restored").parent
    assert file_hash(c.root / "patient-anatomy.json") == original_snapshot
    check_workspace(reopened, saved_scope="exterior-air")


def test_compartment_separation_preserves_whole_native_connected_components():
    mask = np.zeros((8, 8, 8), dtype=bool)
    mask[1:4, 2, 2] = True
    mask[4, 3, 3] = True
    mask[5:7, 6, 1] = True
    anchor = int(np.ravel_multi_index((1, 2, 2), mask.shape))
    main, compartment = separate_reviewed_components(mask, {anchor: 4})

    np.testing.assert_array_equal(main | compartment, mask)
    assert not np.any(main & compartment)
    assert compartment[4, 3, 3] and main[5, 6, 1]
    with pytest.raises(ValueError, match="differs"):
        separate_reviewed_components(mask, {anchor: 3})
    inside = int(np.ravel_multi_index((2, 2, 2), mask.shape))
    with pytest.raises(ValueError, match="differs"):
        separate_reviewed_components(mask, {inside: 4})


def test_review_splits_extend_the_active_native_priority_layer(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from separate_eden2020_vascular_compartment import CANDIDATE, FIELD_PRIORITY, MAIN, split_layers

    posterior = "vesselfm-review-posterior-superficial"
    field = "vesselfm-review-field-edge"
    skull = "vesselfm-review-skull-priority"
    source = "vesselfm-review-source-priority"
    review = {"vascular_review": {}, "display": {"initial_layers": [CANDIDATE]}}
    assert split_layers(review, "skull-exterior") == (CANDIDATE, skull, [skull])
    assert split_layers(review, "source-disagreement") == (CANDIDATE, source, [source])
    assert split_layers(review, "posterior") == (CANDIDATE, MAIN, [MAIN, posterior])
    review["vascular_review"]["posterior_compartment_review"] = {"disjoint_union_matches_selected": True}
    review["display"]["initial_layers"] = [MAIN, posterior]
    assert split_layers(review, "field-edge") == (MAIN, FIELD_PRIORITY, [FIELD_PRIORITY, field, posterior])
    review["vascular_review"]["field_edge_compartment_review"] = {"disjoint_union_matches_selected": True}
    review["display"]["initial_layers"] = [FIELD_PRIORITY, field, posterior]
    assert split_layers(review, "skull-exterior") == (FIELD_PRIORITY, skull, [skull])
    with pytest.raises(ValueError, match="already contains"):
        split_layers(review, "field-edge")
    review["vascular_review"]["skull_exterior_compartment_review"] = {"disjoint_union_matches_selected": True}
    review["display"]["initial_layers"] = [skull]
    assert split_layers(review, "source-disagreement") == (skull, source, [source])
    review["vascular_review"]["source_disagreement_compartment_review"] = {"disjoint_union_matches_selected": True}
    review["display"]["initial_layers"] = [source]
    with pytest.raises(ValueError, match="already contains"):
        split_layers(review, "source-disagreement")


def test_source_disagreement_checks_protected_paths_and_geometry_receipt(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from separate_eden2020_vascular_compartment import stage

    canonical = tmp_path / "patient-anatomy.json"
    selected = tmp_path / VASCULAR_REVIEW_MANIFEST
    geometry = tmp_path / "geometry.json"
    for path in (canonical, selected, geometry):
        path.write_text("{}")
    screen = tmp_path / "source-screen.json"
    evidence = {
        "version": "native-tof-mri-source-disagreement-v1",
        "study_sha256": file_hash(selected),
        "source_index_sha256": file_hash(canonical),
        "structural_source_manifest": {"hdd_relative_path": canonical.name, "sha256": file_hash(canonical)},
    }
    screen.write_text(json.dumps(evidence))
    with pytest.raises(ValueError, match="protected native path checks"):
        stage(tmp_path, selected, screen, [1], compartment="source-disagreement")

    binding = {"hdd_relative_path": geometry.name, "sha256": file_hash(geometry)}
    evidence.update(zero_frozen_path_overlap=True, zero_frozen_junction_overlap=True,
                    source_evidence=binding, exact_geometry_audit=binding)
    screen.write_text(json.dumps(evidence))
    geometry.write_text('{"changed":true}')
    with pytest.raises(ValueError, match="source evidence is missing or has changed"):
        stage(tmp_path, selected, screen, [1], compartment="source-disagreement")


def test_alternate_review_manifest_binds_and_preserves_canonical_anatomy(tmp_path):
    base = PatientImagingCollection.create(tmp_path, "generated")
    base.save()
    manifest = tmp_path / "patient-anatomy.json"
    checksum = file_hash(manifest)
    review = {
        **base.document,
        "vascular_review": {
            "schema": "eden2020-restricted-vascular-companion-v1",
            "linked_source": True,
            "source_manifest": {"sha256": checksum, "snapshot_id": base.identity()},
        },
    }
    review.pop("snapshot_id")
    review["snapshot_id"] = digest(review)
    path = tmp_path / VASCULAR_REVIEW_MANIFEST
    path.write_text(json.dumps(review))
    assert open_anatomy_manifest(path).identity() == review["snapshot_id"]
    assert open_anatomy_manifest(manifest).identity() == base.identity()
    session = SimpleNamespace()
    opened = dispatch(session, "anatomyOpen", {"path": str(path)}, None, None)
    assert opened["readOnlyStudy"] and opened["document"] == review
    with pytest.raises(ValueError, match="read-only"):
        dispatch(session, "anatomySaveView", {"collectionId": opened["collectionId"]}, None, None)
    assert file_hash(manifest) == checksum
    base.save("unrelated native revision")
    with pytest.raises(ValueError, match="pinned source manifest has changed"):
        dispatch(session, "anatomyOpen", {"path": str(path)}, None, None)
