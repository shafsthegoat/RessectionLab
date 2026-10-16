"""Generated evaluator controls only. No acquired tag/image payload is opened."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

PATH = Path(__file__).resolve().parents[1] / "scripts/evaluate_resect_case3_image_registration.py"
spec = importlib.util.spec_from_file_location("case3_fixed_evaluator", PATH)
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)


def generated_tag(pairs):
    return ("MNI Tag Point File\n% generated control only\nVolumes = 2;\nPoints =\n"
            + "\n".join(" ".join(map(str, row)) for row in pairs) + "\n;\n").encode()


def generated_header(affine=None, shape=(20, 21, 22)):
    return {"shape": list(shape), "selected_affine": np.eye(4).tolist() if affine is None else affine.tolist(),
            "sform_code": 1, "qform_code": 0, "spatial_units": "mm", "image_array_accessed": False,
            "affine_replaced": False, "header_sha256": "a" * 64}


def write(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.chmod(0o644)
    path.write_bytes(ev.encoded(record))
    return ev.script_hash(path)


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    """Build a complete, generated sealed chain without importing the fit runner."""
    monkeypatch.setattr(ev, "ROOT", tmp_path)
    paths = SimpleNamespace(root=tmp_path, fit=tmp_path / "outputs/generated-fit",
                            preflight=tmp_path / ev.PREFLIGHT,
                            fit_release=tmp_path / "artifacts/generated/root-fit-release.json",
                            preparation=tmp_path / ev.PREPARATION_DIR / "preparation.json",
                            release=tmp_path / "artifacts/generated/root-evaluation-release.json",
                            output=tmp_path / "outputs/generated-evaluation")
    cohort = {"members": [{"patient_group": ev.CASE, "case_id": "Case3", "role": "TRAIN",
                           "all_timepoints_and_derivatives_grouped": True}]}
    monkeypatch.setattr(ev, "COHORT_SHA", write(tmp_path / ev.COHORT, cohort))
    runner = tmp_path / ev.RUNNER
    runner.parent.mkdir(parents=True, exist_ok=True)
    runner.write_bytes(b"# Generated inert runner identity; never imported.\n")
    monkeypatch.setattr(ev, "RUNNER_SHA", ev.script_hash(runner))
    test_file = tmp_path / ev.TESTS
    test_file.parent.mkdir(parents=True, exist_ok=True)
    test_file.write_bytes(b"# Generated test identity.\n")
    values = np.array([[1, 2, 3, 2, 2, 3], [4, 5, 6, 5, 5, 6], [30, 2, 3, 31, 2, 3]], float)
    payload = generated_tag(values)
    monkeypatch.setattr(ev, "TAG", {**ev.TAG, "bytes": len(payload), "sha256": ev.sha_bytes(payload)})
    tag = tmp_path / ev.TAG["path"]
    tag.parent.mkdir(parents=True)
    tag.write_bytes(payload)
    paths.tag = tag
    matrix = np.eye(4)
    matrix[0, 3] = 1
    records = {"preflight": {"schema": ev.FIT_SCHEMA + "-preflight", "status": "headers_verified_fit_not_run",
        "case_id": ev.CASE, "role": "TRAIN", "cohort_sha256": ev.COHORT_SHA,
        "decoded_array_bytes": 0, "landmark_coordinates_read": False, "fit_run": False,
        "anatomical_alignment_accepted": False, "inputs": deepcopy(ev.IMAGES),
        "headers": {name: generated_header() for name in ev.IMAGES}, "runner_sha256": ev.RUNNER_SHA,
        "configuration": {"one_fixed_generated_configuration": True},
        "limits": {"sampled_family_rss_stop_bytes": 4096}, "runtime": {"generated": True}}}
    preflight_sha = write(paths.preflight, records["preflight"])
    monkeypatch.setattr(ev, "PREFLIGHT_SHA", preflight_sha)
    records["fit_release"] = {"schema": ev.FIT_SCHEMA + "-root-release", "decision": "release_one_fit",
        "case_id": ev.CASE, "preflight_sha256": preflight_sha, "runner_sha256": ev.RUNNER_SHA,
        "landmark_access_permitted": False, "output_directory": str(paths.fit.resolve())}
    fit_release_sha = write(paths.fit_release, records["fit_release"])
    records["candidate"] = {"schema": ev.FIT_SCHEMA + "-candidate", "case_id": ev.CASE,
        "preflight_sha256": preflight_sha, "fit_release_sha256": fit_release_sha,
        "runner_sha256": ev.RUNNER_SHA, "runtime": {"generated": True}, "sources_rechecked_after_each_decode": True,
        "source_fixity_after_fit": {key: row["sha256"] for key, row in ev.IMAGES.items()},
        "report": {ev.FORWARD: matrix.tolist(), ev.INVERSE: np.linalg.inv(matrix).tolist(),
                   "fixed": "before_resection_US", "moving": "during_resection_US", "sampling_direction": ev.DIRECTION,
                   "landmarks_used": False, "configuration": records["preflight"]["configuration"],
                   "identity_baseline_before_RAS_to_during_RAS": np.eye(4).tolist()}}
    records["supervision"] = {"schema": ev.FIT_SCHEMA + "-supervision", "status": "completed", "exit_code": 0,
        "worker_reaped": True, "process_group_cleanup_attempted": True, "samples": 2,
        "peak_family_rss_bytes_sampled": 1024, "sampled_family_rss_stop_bytes": 4096}
    records["preparation"] = ev.prepare(paths.preflight, paths.preparation)
    records["release"] = {"schema": ev.FIT_SCHEMA + "-evaluation-release", "issuer": "ROOT",
        "decision": "release_separate_evaluation", "case_id": ev.CASE, "role": "TRAIN",
        "full_tag_role": "evaluation_only_all_points", "refitting_or_parameter_selection_permitted": False,
        "landmark_access_permitted": True, "tag": ev.TAG, "cohort": {"path": ev.COHORT, "sha256": ev.COHORT_SHA},
        "fit_directory": str(paths.fit.resolve()), "output_directory": str(paths.output.resolve()),
        "evaluator_sha256": ev.script_hash(PATH), "tests_sha256": ev.script_hash(test_file),
        "runner_sha256": ev.RUNNER_SHA, "preparation_sha256": ev.script_hash(paths.preparation),
        "preflight_sha256": preflight_sha, "fit_release_sha256": fit_release_sha}

    def reseal():
        candidate_sha = write(paths.fit / "candidate.json", records["candidate"])
        supervision_sha = write(paths.fit / "supervision.json", records["supervision"])
        frozen = {**records["candidate"], "schema": ev.FIT_SCHEMA + "-frozen", "candidate_sha256": candidate_sha,
                  "supervision_sha256": supervision_sha, "evaluation_requires_separate_root_release": True,
                  "anatomical_alignment_accepted": False}
        frozen_sha = write(paths.fit / "frozen-transform.json", frozen)
        result = {"schema": ev.FIT_SCHEMA + "-result", "case_id": ev.CASE, "status": "completed", "exit_code": 0,
                  "frozen_transform_sha256": frozen_sha, "supervision_sha256": supervision_sha,
                  "preflight_sha256": preflight_sha, "fit_release_sha256": fit_release_sha,
                  "evaluation_run": False, "anatomical_alignment_accepted": False}
        records["result"] = result
        records["frozen"] = frozen
        result_sha = write(paths.fit / "result.json", result)
        records["release"].update({"fit_result_sha256": result_sha, "candidate_sha256": candidate_sha,
            "frozen_transform_sha256": frozen_sha, "supervision_sha256": supervision_sha})
        write(paths.release, records["release"])
    reseal()
    paths.records, paths.reseal = records, reseal
    paths.run = lambda: ev.evaluate(paths.fit, paths.preflight, paths.fit_release, paths.preparation, paths.release, paths.output)
    return paths


def test_preparation_is_metadata_only_and_contains_no_header_or_tag_coordinates(fixture, monkeypatch):
    original = Path.open
    def guarded(path, *args, **kwargs):
        if path.suffix == ".tag" or str(path).endswith(".nii.gz"):
            pytest.fail("preparation opened patient payload")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", guarded)
    record = ev.prepare(fixture.preflight, fixture.preparation.with_name("second-generated-preparation.json"))
    assert record["actual_evaluation_run"] is False
    assert record["acquired_landmark_coordinates_read"] is False
    assert "selected_affine" not in json.dumps(record) and "headers" not in record


def test_generated_full_chain_retains_outside_rows_and_separates_coordinate_outputs(fixture, monkeypatch):
    original = Path.open
    reads = []
    def guarded(path, *args, **kwargs):
        if str(path).endswith((".nii.gz", "-short.tag")):
            pytest.fail("evaluator opened image or short tag")
        if path.suffix == ".tag":
            assert path == fixture.tag
            reads.append(path)
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", guarded)
    result = fixture.run()
    assert reads == [fixture.tag]
    assert result["point_count"] == 3 and result["excluded_point_indices"] == []
    assert result["identity"]["RMS_mm"] == 1 and result["rigid"]["RMS_mm"] == 0
    assert result["primary_delta_RMS_mm_rigid_minus_identity"] == -1
    assert result["inclusion_failure_indices"]["before_native"] == [3]
    assert [r["point_index"] for r in result["rows"]] == [1, 2, 3]
    assert all(r["retained_in_all_statistics"] for r in result["rows"])
    assert '"before_RAS_mm":' not in json.dumps(result)
    full = json.loads((fixture.output / "all-points-with-coordinates.json").read_text())
    assert len(full["points"]) == 3 and "before_RAS_mm" in full["points"][2]
    with pytest.raises(ValueError, match="NEW_OUTPUTS"):
        fixture.run()


@pytest.mark.parametrize("mutation", ["no_release", "decision", "issuer", "role", "evaluator_hash", "tag_hash",
    "result_hash", "result_missing", "fit_failed", "nonzero_exit", "supervision_failed", "not_reaped",
    "cleanup_missing", "over_budget", "landmarks_used", "direction", "inverse", "source_hash", "runtime",
    "frozen_changed", "specification", "runner_changed", "duplicate_json"])
def test_missing_or_tampered_metadata_refused_before_tag_read(fixture, monkeypatch, mutation):
    r, p = fixture.records, fixture
    if mutation == "no_release": p.release.unlink()
    elif mutation in ("decision", "issuer", "role"):
        r["release"][mutation] = "invalid"
        write(p.release, r["release"])
    elif mutation in ("evaluator_hash", "tag_hash", "result_hash"):
        if mutation == "tag_hash": r["release"]["tag"] = {**ev.TAG, "sha256": "f" * 64}
        else: r["release"]["evaluator_sha256" if mutation == "evaluator_hash" else "fit_result_sha256"] = "f" * 64
        write(p.release, r["release"])
    elif mutation == "result_missing": (p.fit / "result.json").unlink()
    elif mutation in ("fit_failed", "nonzero_exit"):
        r["result"]["status" if mutation == "fit_failed" else "exit_code"] = "worker_failed" if mutation == "fit_failed" else 1
        r["release"]["fit_result_sha256"] = write(p.fit / "result.json", r["result"])
        write(p.release, r["release"])
    elif mutation in ("supervision_failed", "not_reaped", "cleanup_missing", "over_budget"):
        key, value = {"supervision_failed": ("status", "memory_stop"), "not_reaped": ("worker_reaped", False),
                      "cleanup_missing": ("process_group_cleanup_attempted", False), "over_budget": ("peak_family_rss_bytes_sampled", 4097)}[mutation]
        r["supervision"][key] = value
        p.reseal()
    elif mutation in ("landmarks_used", "direction", "inverse", "source_hash", "runtime"):
        if mutation == "landmarks_used": r["candidate"]["report"]["landmarks_used"] = True
        if mutation == "direction": r["candidate"]["report"]["sampling_direction"] = "during to before"
        if mutation == "inverse": r["candidate"]["report"][ev.INVERSE] = np.eye(4).tolist()
        if mutation == "source_hash": r["candidate"]["source_fixity_after_fit"]["before"] = "f" * 64
        if mutation == "runtime": r["candidate"]["runtime"] = {"changed": True}
        p.reseal()
    elif mutation == "frozen_changed":
        r["frozen"]["anatomical_alignment_accepted"] = True
        frozen_sha = write(p.fit / "frozen-transform.json", r["frozen"])
        r["result"]["frozen_transform_sha256"] = frozen_sha
        r["release"]["frozen_transform_sha256"] = frozen_sha
        r["release"]["fit_result_sha256"] = write(p.fit / "result.json", r["result"])
        write(p.release, r["release"])
    elif mutation == "specification":
        r["preparation"]["specification"] = {**ev.SPEC, "exclusions": True}
        r["release"]["preparation_sha256"] = write(p.preparation, r["preparation"])
        write(p.release, r["release"])
    elif mutation == "runner_changed": (p.root / ev.RUNNER).write_bytes(b"changed")
    elif mutation == "duplicate_json": p.release.write_text('{"issuer":"ROOT","issuer":"other"}')
    original = Path.open
    def guarded(path, *args, **kwargs):
        if path.suffix == ".tag" or str(path).endswith(".nii.gz"):
            pytest.fail("gate failure opened a payload")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", guarded)
    with pytest.raises((ValueError, FileNotFoundError)):
        p.run()
    assert not p.output.exists()


def test_ras_lps_sign_and_rotated_transform_not_inverse():
    angle = np.deg2rad(30.)
    matrix = np.array([[np.cos(angle), -np.sin(angle), 0, 2], [np.sin(angle), np.cos(angle), 0, -3],
                       [0, 0, 1, 1], [0, 0, 0, 1]], float)
    before = np.array([[1, 2, 3], [4, 3, 2], [-2, 1, 3]], float)
    during = ev.map_points(matrix, before)
    pairs = np.column_stack((before, during))
    headers = {name: generated_header() for name in ev.IMAGES}
    right, _ = ev.score_pairs(pairs, matrix, headers)
    inverse, _ = ev.score_pairs(pairs, np.linalg.inv(matrix), headers)
    ras_lps = np.diag([-1., -1., 1., 1.])
    wrong_frame, _ = ev.score_pairs(pairs, ras_lps @ matrix @ ras_lps, headers)
    assert right["rigid"]["max_mm"] < 1e-12
    assert inverse["rigid"]["RMS_mm"] > 4 and wrong_frame["rigid"]["RMS_mm"] > 4
    # Source-world points are already RAS; conversion belongs only to an LPS transform adapter.
    np.testing.assert_allclose(ras_lps @ (ras_lps @ matrix @ ras_lps) @ ras_lps, matrix)


def test_oblique_active_sform_inclusion_uses_native_cells_not_world_aabb():
    a = np.sqrt(.5)
    affine = np.array([[a, -a, 0, 20], [a, a, 0, -10], [0, 0, 2, 7], [0, 0, 0, 1.]])
    header = generated_header(affine, (2, 2, 2))
    voxel = np.array([[-.5, -.5, -.5], [1.5, 1.5, 1.5], [-.501, 0, 0], [1.501, 1, 1]])
    recovered, inside = ev.inclusion(ev.map_points(affine, voxel), header)
    np.testing.assert_allclose(recovered, voxel, atol=1e-12)
    assert inside.tolist() == [True, True, False, False]


def test_duplicates_are_reported_and_retained_with_original_indices():
    pairs = [[1, 2, 3, 2, 2, 3], [1, 2, 3, 2, 2, 3], [1, 2, 3, 4, 2, 3]]
    parsed = ev.parse_full_tag(generated_tag(pairs))
    result, _ = ev.score_pairs(parsed, np.eye(4), {name: generated_header() for name in ev.IMAGES})
    assert result["point_count"] == 3
    assert result["rows"][1]["duplicate_of_first_point_index"] == {"pair": 1, "before": 1, "during": 1}
    assert result["rows"][2]["duplicate_of_first_point_index"] == {"pair": None, "before": 1, "during": None}
    assert result["identity"]["RMS_mm"] == pytest.approx(np.sqrt(11 / 3))


@pytest.mark.parametrize("payload", [
    b"", b"not a tag", b"MNI Tag Point File\nVolumes = 1;\nPoints = 0 0 0 0 0 0;",
    b"MNI Tag Point File\nVolumes = 2;\nPoints = ;",
    generated_tag([[1, 2, 3, 4, 5]]), generated_tag([[1, 2, 3, 4, 5, 6, 7]]),
    generated_tag([[1, 2, 3, 4, 5, float('nan')]]), generated_tag([[1, 2, 3, 4, 5, float('inf')]]),
    generated_tag([[1, 2, 3, 4, 5, '1e999']]), generated_tag([[1, 2, 3, 4, 5, 6]])[:-2],
    generated_tag([[1, 2, 3, 4, 5, 6]]) + b"1 2 3 4 5 6;\n",
])
def test_malformed_or_nonfinite_tag_refuses_whole_evaluation(payload):
    with pytest.raises(ValueError):
        ev.parse_full_tag(payload)


def test_source_six_field_comment_and_final_inline_terminator():
    parsed = ev.parse_full_tag(b"MNI Tag Point File\nVolumes=2;\nPoints=\n1e0 -2 .3 4. +5 6; % end\n")
    assert parsed.tolist() == [[1, -2, .3, 4, 5, 6]]


@pytest.mark.parametrize("change", ["reflection", "scale", "nan", "last_row", "shape"])
def test_nonrigid_or_nonfinite_transforms_are_rejected(change):
    matrix = np.eye(4)
    if change == "reflection": matrix[0, 0] = -1
    if change == "scale": matrix[0, 0] = 1.1
    if change == "nan": matrix[0, 3] = np.nan
    if change == "last_row": matrix[3, 0] = .01
    if change == "shape": matrix = matrix[:3]
    with pytest.raises(ValueError, match="PROPER_RIGID"):
        ev.rigid_matrix(matrix)


def test_released_tag_hash_mismatch_refuses_scoring_and_publication(fixture, monkeypatch):
    payload = fixture.tag.read_bytes()
    fixture.tag.write_bytes(payload.replace(b"1.0 2.0", b"9.0 2.0", 1))
    monkeypatch.setattr(ev, "score_pairs", lambda *_: pytest.fail("scored mismatched tag"))
    with pytest.raises(ValueError, match="FULL_TAG_FIXITY"):
        fixture.run()
    assert not fixture.output.exists()


@pytest.mark.parametrize("change", ["qform", "units", "singular", "array", "replacement"])
def test_header_records_must_be_native_active_sform_mm(change):
    header = generated_header()
    if change == "qform": header["sform_code"] = 0
    if change == "units": header["spatial_units"] = "meter"
    if change == "singular": header["selected_affine"][0][0] = 0
    if change == "array": header["image_array_accessed"] = True
    if change == "replacement": header["affine_replaced"] = True
    with pytest.raises(ValueError, match="ACTIVE_SFORM"):
        ev.header_geometry(header)


def test_output_outside_outputs_is_refused_before_tag(fixture, monkeypatch):
    original = Path.open
    def guarded(path, *args, **kwargs):
        if path.suffix == ".tag": pytest.fail("opened tag before output boundary validation")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", guarded)
    with pytest.raises(ValueError, match="NEW_OUTPUTS"):
        ev.evaluate(fixture.fit, fixture.preflight, fixture.fit_release, fixture.preparation, fixture.release,
                    fixture.root / "artifacts/forbidden-coordinate-output")
