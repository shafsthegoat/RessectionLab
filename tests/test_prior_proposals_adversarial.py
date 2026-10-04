"""Independent physical-sampling and capability-boundary prior regressions."""
from dataclasses import replace
import hashlib
from io import BytesIO
import json
import threading
from types import SimpleNamespace
from zipfile import ZIP_DEFLATED, ZipFile

import nibabel as nib
import numpy as np
import pytest

from resectionlab.desktop_bridge import BridgeRuntime, _Request
from resectionlab.prior_proposals import import_registered_prior_proposals

# These fixtures create synthetic source files; expected sampling and attacks
# below are independent of the production registration/resampling helpers.
from test_prior_proposals import registered_fixture
from test_prior_proposals_bridge import fixture_case, import_paths, proposal


def test_registered_translation_matches_analytic_physical_ramp_and_coverage(registered_fixture):
    case, kwargs, _preview = registered_fixture
    revised = import_registered_prior_proposals(case, **kwargs)
    item = next(iter(revised.prior_proposals.values()))
    x, y, z = np.indices(case.mri.shape)
    # Source atlas is (x+y+z)/24 on an identity RAS grid. A +1.5 mm
    # source-to-patient translation must sample source x=patient x-1.5.
    covered = x >= 2
    expected = np.where(covered, (x - 1.5 + y + z) / 24., 0.)
    np.testing.assert_array_equal(item.sampling_coverage, covered)
    np.testing.assert_allclose(item.data, expected, rtol=0, atol=6e-8)
    assert item.data[0, 0, 0] == 0 and not item.sampling_coverage[0, 0, 0]


def test_real_import_is_invariant_to_equivalent_lps_case_coordinates(registered_fixture):
    case, kwargs, _preview = registered_fixture
    ras = import_registered_prior_proposals(case, **kwargs)
    lps_case = replace(case, frame="LPS+", affine=np.diag([-1., -1., 1., 1.]) @ case.affine)
    lps = import_registered_prior_proposals(lps_case, **kwargs)
    first, second = next(iter(ras.prior_proposals.values())), next(iter(lps.prior_proposals.values()))
    np.testing.assert_array_equal(first.data, second.data)
    np.testing.assert_array_equal(first.sampling_coverage, second.sampling_coverage)
    np.testing.assert_array_equal(first.affine_ras_mm, second.affine_ras_mm)
    assert lps.frame == "LPS+"
    assert second.registration_case_hash == case.semantic_hash
    assert second.source_case_planning_hash == lps_case.planning_hash


@pytest.mark.parametrize("changes", [
    {"component": "phonology"},
    {"map_kind": "structural_mask", "data": np.zeros((24, 24, 24), np.float32)},
    {"map_id": "speech_articulation_functional_concordance"},
    {"map_id": "motor_structural_mask", "map_kind": "structural_mask",
     "data": np.zeros((24, 24, 24), np.float32)},
])
def test_typed_proposal_cannot_change_network_identity_or_map_semantics(changes):
    item = proposal(fixture_case())
    with pytest.raises(ValueError, match="(?i)(map|component|canonical|supported)"):
        replace(item, **changes)


@pytest.mark.parametrize("name", ["registration_comparison.json", "functional_overlay_inventory.json",
                                  "implementation_snapshot.py"])
def test_fixed_registration_inputs_cannot_follow_symlinks_outside_selected_directory(registered_fixture, name):
    case, kwargs, _preview = registered_fixture
    directory = kwargs["registration_directory"]
    selected = directory / name
    outside = directory.parent / f"outside-{name}"
    outside.write_bytes(selected.read_bytes())
    selected.unlink()
    selected.symlink_to(outside)
    with pytest.raises(ValueError, match="escapes"):
        import_registered_prior_proposals(case, **kwargs)
    assert not case.prior_proposals


def test_view_only_dense_prior_cannot_supply_functional_rewards_or_change_native_actions():
    from resectionlab.native_simulation import make_native_patient_simulator
    case = fixture_case()
    item = replace(proposal(case), data=np.ones(case.mri.shape, np.float32),
                   sampling_coverage=np.ones(case.mri.shape, bool))
    enriched = case.revised(prior_proposals={item.proposal_id: item})
    before = make_native_patient_simulator(case, candidate_count=2, max_steps=2, max_actions=5)
    after = make_native_patient_simulator(enriched, candidate_count=2, max_steps=2, max_actions=5)
    assert before.config.evidence_available == after.config.evidence_available == (False, False)
    first, second = before.observation(), after.observation()
    assert first.action_ids == second.action_ids
    np.testing.assert_array_equal(first.action_features, second.action_features)
    np.testing.assert_array_equal(first.action_mask, second.action_mask)
    np.testing.assert_array_equal(first.state_features, second.state_features)
    for position in np.flatnonzero(first.action_mask):
        baseline, with_prior = before.clone(), after.clone()
        a, b = baseline.step(int(position)), with_prior.step(int(position))
        assert a.reward == b.reward
        np.testing.assert_array_equal(a.observation.action_features, b.observation.action_features)
        assert with_prior.metrics()["clinical_deficit_probability"] is None
    assert enriched.planning_hash == case.planning_hash


def test_cancellation_after_prior_data_transfer_removes_partial_assets(tmp_path, monkeypatch):
    case = fixture_case()
    item = proposal(case)
    enriched = case.revised(prior_proposals={item.proposal_id: item})
    events = []
    runtime = BridgeRuntime(tmp_path / "transfer", events.append)
    entered, release = threading.Event(), threading.Event()
    try:
        runtime.session._install_case(case, {}, _Request("install"))
        existing = {path.name for path in runtime.session.transfers.root.iterdir()}
        original = runtime.session.transfers.array
        def blocked_array(array, dtype):
            descriptor = original(array, dtype)
            if array is item.data:
                entered.set()
                assert release.wait(5)
            return descriptor
        monkeypatch.setattr(runtime.session.transfers, "array", blocked_array)
        monkeypatch.setattr("resectionlab.prior_proposals.import_registered_prior_proposals",
                            lambda *args, **kwargs: enriched)
        runtime.submit({"id": "import", "op": "importPriorProposals", "args": {
            "caseHash": case.semantic_hash, **import_paths(tmp_path, 1)}})
        assert entered.wait(3)
        assert {path.name for path in runtime.session.transfers.root.iterdir()} != existing
        runtime.submit({"id": "cancel", "op": "cancel", "args": {"requestId": "import"}})
        release.set()
        assert runtime.wait_idle(5)
        assert [event for event in events if event.get("id") == "import"][-1]["event"] == "cancelled"
        assert list(runtime.session.cases) == [case.semantic_hash]
        assert {path.name for path in runtime.session.transfers.root.iterdir()} == existing
    finally:
        release.set()
        runtime.close()


def test_source_header_budget_precedes_atlas_allocation_or_resampling(registered_fixture, monkeypatch):
    case, kwargs, _preview = registered_fixture
    original = nib.load
    prior_path = kwargs["source_cache_directory"] / "analytic_prior.nii"
    def bounded_loader(path, *args, **options):
        if str(path) == str(prior_path):
            return SimpleNamespace(shape=(1024, 1024, 1024))
        return original(path, *args, **options)
    monkeypatch.setattr(nib, "load", bounded_loader)
    monkeypatch.setattr("resectionlab.anatomy.load_functional_prior",
        lambda *a, **k: pytest.fail("Oversized header reached source allocation"))
    with pytest.raises(ValueError, match="512 MiB"):
        import_registered_prior_proposals(case, **kwargs)
    assert not case.prior_proposals


def test_portable_case_rejects_fractional_coverage_even_with_resealed_payload(registered_fixture, tmp_path):
    from resectionlab.imaging import ImagingError, load_case, save_case
    case, kwargs, _preview = registered_fixture
    imported = import_registered_prior_proposals(case, **kwargs)
    path = save_case(imported, tmp_path / "fractional-coverage.ressectionlab")
    with ZipFile(path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        with np.load(BytesIO(archive.read("arrays.npz")), allow_pickle=False) as stored:
            arrays = {key: stored[key].copy() for key in stored.files}
    coverage_key = next(iter(manifest["prior_proposals"].values()))["coverage_key"]
    arrays[coverage_key] = arrays[coverage_key].astype(np.float32)
    arrays[coverage_key][0, 0, 0] = .25
    buffer = BytesIO()
    np.savez_compressed(buffer, **arrays)
    payload = buffer.getvalue()
    manifest["array_sha256"] = hashlib.sha256(payload).hexdigest()
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("arrays.npz", payload)
        archive.writestr("manifest.json", json.dumps(manifest))
    with pytest.raises(ImagingError, match="Sampling coverage must be binary"):
        load_case(path)
