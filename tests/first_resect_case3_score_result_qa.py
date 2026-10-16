#!/usr/bin/env python3
"""Separate opt-in saved-score audit; actual mode needs explicit ROOT release.

Generated mode opens no acquired payload. Actual mode hashes the two compressed
source files, reads only the exact full tag, and uses the previously frozen
independent parser/stdlib arithmetic. It never imports the production scorer,
fit runner, image decoder or optimization library. No coordinates are published.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "artifacts/resect-case3-evaluation-independent-v1"
HELPER = ROOT / "tests/first_resect_case3_evaluation_qa.py"
HELPER_SHA = "5b9fdb550ccea490691f7861cb8842cfc57a93351549cce1dd6dac256c2dd240"
FIT = "outputs/poc-resect-case3-image-registration-v1/"
PREP = "artifacts/resect-case3-registration-evaluation-preparation-v2/preparation.json"
PREFLIGHT = "artifacts/resect-case3-image-registration-preparation-v1/header-preflight-rss-v2.json"
TAG = {"path": "data/anatomy/resect-v1/Case3/Landmarks/Case3-beforeUS-duringUS-full.tag",
       "bytes": 1190, "sha256": "585769a006d8e5a995f987d91cef75ca45288cf6132aa3842cd44ca19927fc08"}
FIXED = {
    "tests/first_resect_case3_evaluation_qa.py": HELPER_SHA,
    "scripts/evaluate_resect_case3_image_registration.py": "7d987eb0bd31375d924e444e1838e4255327cef7036ea2e4ed7e9d78b9e3406e",
    "tests/test_evaluate_resect_case3_image_registration.py": "eda3b282c158ceaecb1320fed0722e0e4ea8bdabad416826ca665192538d5007",
    "scripts/poc_resect_case3_image_registration.py": "e17d755eeadb6a3d1fd365b7a3b4dc31f9e27c4f93afdb6045d8f6e27ef9612f",
    "manifests/resect-component-cohort-v1.json": "4fc51f01e42253d395660d850cda2296a3e7f23e5a5ade80bbd5a3ed411b6a77",
    PREP: "0b9668d8b6a1adf319b55a6ed8aded986039bc6eed8876a4c1fabf9f50f5518b",
    PREFLIGHT: "ef9e0cd6029e869a6364de6c946f0284bc0a7ca1f700695a8bb75b52e60fd7ba",
    FIT + "result.json": "ea8e57322989e6b5f580c67181201deead9235a4829315781bc7aad0f598b12b",
    FIT + "candidate.json": "17a9599de644b9d7d4cd3fd9b356630adb3e38180496b9501f914520056949ef",
    FIT + "frozen-transform.json": "e6648339d69841985f8fa644ae5d69168422ce12ca5dadc4659b275a8b9cd8af",
    FIT + "supervision.json": "e675d52669dfef2442e31d019e98c8d14eb705a470692510f0bc37a1cf2c9fe9",
    "artifacts/poc-resect-case3-image-registration-v1/root-fit-release.json": "bc196c57fb4701c161f50bcb4b3edae03b5988b864308a0c217e9c83ac8962e0",
    "artifacts/resect-case3-registration-result-independent-v1/independent-fit-qa.json": "3277460d1529126ea04828e8785ffd0ebfebdc1ae6209b77e7d4cd92eed1a955",
    "artifacts/resect-case3-evaluation-independent-v1/prospective-qa.json": "6c9afea9647399b743174203fa10814615acf6ccbeb32708317d058ec5d0a265",
    "artifacts/poc-resect-case3-image-registration-v1/initial-evaluation-refusal.json": "49f7d947e44487c7e7a274628541a7b5a3611d389d731d4be3d9b14dbf28bd16",
    "artifacts/resect-case3-registration-evaluation-preparation-v2/diagnosis.json": "a9edc80d8129abf2fca13ac2d6b2f79d667ee5dcd7aaf1ff544c136425c772f3",
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


if sha(HELPER) != HELPER_SHA:
    raise RuntimeError("frozen independent oracle changed")
spec = importlib.util.spec_from_file_location("frozen_case3_independent_oracle", HELPER)
qa = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qa)
require, equal = qa.require, qa.equal


def verify_records(summary, coordinates, pairs, matrix, headers, common, coordinate_sha):
    numerical, expected_points = qa.oracle(pairs, matrix, headers)
    equal(coordinates, {**common, "points": expected_points})
    nonclaims = {"population_confidence_interval": None, "clinical_acceptance": None,
                "anatomical_alignment_accepted": False,
                "interpretation": "One TRAIN person; all finite full-tag pairs equally weighted. No retuning, exclusions or population inference."}
    equal(summary, {**common, **numerical, **nonclaims, "coordinate_output_sha256": coordinate_sha})
    require([row["point_index"] for row in summary["rows"]] == list(range(1, len(pairs) + 1)), "all original row indices")
    return numerical


def generated():
    matrix = [[0., -1., 0., 2.], [1., 0., 0., -3.], [0., 0., 1., 1.], [0., 0., 0., 1.]]
    frame = [[.8, -.6, 0., 20.], [.6, .8, 0., -10.], [0., 0., 2., 7.], [0., 0., 0., 1.]]
    headers = {key: {"selected_affine": frame, "shape": [3, 4, 5]} for key in ("before", "during")}
    pairs = [[float(i), 1., 2., float(i + 1), 2., 3.] for i in range(17)]
    pairs[5] = pairs[0][:]
    common = {"schema": "generated", "role": "TRAIN", "bindings": {"source": "generated-only"}}
    numerical, points = qa.oracle(pairs, matrix, headers)
    coordinates = {**common, "points": points}
    summary = {**common, **numerical, "population_confidence_interval": None, "clinical_acceptance": None,
               "anatomical_alignment_accepted": False,
               "interpretation": "One TRAIN person; all finite full-tag pairs equally weighted. No retuning, exclusions or population inference.",
               "coordinate_output_sha256": "a" * 64}
    verify_records(summary, coordinates, pairs, matrix, headers, common, "a" * 64)
    mutations = ("point omitted", "point reordered", "duplicate removed", "wrong coordinates", "wrong prediction",
                 "wrong voxel", "wrong inclusion", "wrong duplicate", "wrong maximum", "wrong median", "wrong RMS",
                 "wrong delta", "exclusion", "clinical claim", "anatomy acceptance", "changed binding", "wrong coordinate hash")
    for change in mutations:
        s, c = copy.deepcopy(summary), copy.deepcopy(coordinates)
        if change == "point omitted": c["points"].pop()
        elif change == "point reordered": c["points"][0], c["points"][1] = c["points"][1], c["points"][0]
        elif change == "duplicate removed": s["point_count"] = 16
        elif change == "wrong coordinates": c["points"][0]["before_RAS_mm"][0] += .1
        elif change == "wrong prediction": c["points"][0]["predicted_during_RAS_mm"][0] += .1
        elif change == "wrong voxel": c["points"][0]["native_voxel_coordinates"]["before_native"][0] += .1
        elif change == "wrong inclusion": s["rows"][0]["included_in_native_image"]["before_native"] ^= True
        elif change == "wrong duplicate": s["rows"][5]["duplicate_of_first_point_index"]["pair"] = None
        elif change == "wrong maximum": s["rigid"]["max_mm"] += .1
        elif change == "wrong median": s["rigid"]["median_mm"] += .1
        elif change == "wrong RMS": s["identity"]["RMS_mm"] += .1
        elif change == "wrong delta": s["primary_delta_RMS_mm_rigid_minus_identity"] += .1
        elif change == "exclusion": s["excluded_point_indices"] = [17]
        elif change == "clinical claim": s["clinical_acceptance"] = True
        elif change == "anatomy acceptance": s["anatomical_alignment_accepted"] = True
        elif change == "changed binding": s["bindings"]["source"] = "changed"
        else: s["coordinate_output_sha256"] = "b" * 64
        try:
            verify_records(s, c, pairs, matrix, headers, common, "a" * 64)
        except AssertionError:
            pass
        else:
            raise AssertionError("mutation not detected: " + change)
    return {"schema": "resect-case3-saved-score-auditor-generated-v1", "status": "pass",
            "checks_passed": len(mutations) + 1, "generated_points": 17,
            "mutations_rejected": list(mutations), "auditor_sha256": sha(__file__), "oracle_sha256": HELPER_SHA,
            "acquired_payloads_read": False, "actual_evaluation_run": False,
            "actual_score_verification_run": False}


def actual(args):
    # All metadata/output pins are checked before either compressed source or tag read.
    require(args.release and args.release_sha and args.summary_sha and args.coordinates_sha, "explicit exact release/output pins")
    release_path = args.release.resolve()
    require(release_path.is_relative_to((ROOT / "artifacts").resolve()) and release_path.suffix == ".json", "ROOT release location")
    expected = {**FIXED, str(release_path.relative_to(ROOT)): args.release_sha}
    for relative, wanted in expected.items():
        require(sha(ROOT / relative) == wanted, "metadata pin: " + relative)
    release = qa.record(release_path)
    require(release["schema"] == "resect-case3-image-registration-v1-evaluation-release"
            and release["issuer"] == "ROOT" and release["decision"] == "release_separate_evaluation"
            and release["case_id"] == "RESECT:Case3" and release["role"] == "TRAIN"
            and release["landmark_access_permitted"] is True
            and release["refitting_or_parameter_selection_permitted"] is False
            and release["full_tag_role"] == "evaluation_only_all_points" and release["tag"] == TAG,
            "exact full-tag ROOT release")
    fit_directory = Path(release["fit_directory"]).resolve()
    output = Path(release["output_directory"]).resolve()
    require(fit_directory == (ROOT / FIT).resolve() and output.is_relative_to((ROOT / "outputs").resolve()), "released directories")
    summary_path, coordinates_path = output / "summary-no-coordinates.json", output / "all-points-with-coordinates.json"
    require(sha(summary_path) == args.summary_sha and sha(coordinates_path) == args.coordinates_sha, "saved-output pins")
    expected.update({str(summary_path.relative_to(ROOT)): args.summary_sha,
                     str(coordinates_path.relative_to(ROOT)): args.coordinates_sha})
    mapping = {"preparation_sha256": PREP, "preflight_sha256": PREFLIGHT,
               "evaluator_sha256": "scripts/evaluate_resect_case3_image_registration.py",
               "tests_sha256": "tests/test_evaluate_resect_case3_image_registration.py",
               "runner_sha256": "scripts/poc_resect_case3_image_registration.py",
               "fit_result_sha256": FIT + "result.json", "candidate_sha256": FIT + "candidate.json",
               "frozen_transform_sha256": FIT + "frozen-transform.json", "supervision_sha256": FIT + "supervision.json",
               "fit_release_sha256": "artifacts/poc-resect-case3-image-registration-v1/root-fit-release.json"}
    for key, relative in mapping.items():
        require(release[key] == FIXED[relative], "release source/fit binding: " + key)
    preparation, preflight = qa.record(ROOT / PREP), qa.record(ROOT / PREFLIGHT)
    candidate = qa.record(ROOT / FIT / "candidate.json")
    frozen = qa.record(ROOT / FIT / "frozen-transform.json")
    require(frozen == {**candidate, "schema": "resect-case3-image-registration-v1-frozen",
                       "candidate_sha256": FIXED[FIT + "candidate.json"], "supervision_sha256": FIXED[FIT + "supervision.json"],
                       "evaluation_requires_separate_root_release": True, "anatomical_alignment_accepted": False},
            "exact sealed candidate")
    require(candidate["report"]["landmarks_used"] is False
            and candidate["runtime"] == preflight["runtime"]
            and preparation["scoring_runtime"]["python"] == sys.version.split()[0], "sealed runtime and image-only fit")
    matrix = candidate["report"][qa.FORWARD]
    equal(candidate["report"][qa.INVERSE], qa.invert(matrix))
    for header in preflight["headers"].values():
        require(header["sform_code"] == 1 and header["qform_code"] == 0 and header["spatial_units"] == "mm"
                and header["image_array_accessed"] is False and header["affine_replaced"] is False, "native active sform")
    source_pins = {}
    for image in preparation["images"].values():
        source = ROOT / image["path"]
        require(source.stat().st_size == image["bytes"] and sha(source) == image["sha256"], "compressed source fixity")
        source_pins[image["path"]] = image["sha256"]
    # First acquired coordinate read by this independent result audit.
    tag_path = ROOT / TAG["path"]
    require(not tag_path.is_symlink() and tag_path.stat().st_size == TAG["bytes"], "exact full-tag file")
    payload = tag_path.read_bytes()
    require(hashlib.sha256(payload).hexdigest() == TAG["sha256"], "full-tag hash")
    pairs = qa.independent_parse(payload)
    require(len(pairs) == 17, "all 17 frozen full-tag rows")
    summary, coordinates = qa.record(summary_path), qa.record(coordinates_path)
    bindings = {key: release[key] for key in mapping}
    bindings["evaluation_release_sha256"] = args.release_sha
    common = {"schema": "resect-case3-registration-evaluation-v1", "case_id": "RESECT:Case3", "role": "TRAIN",
              "status": "scored_all_full_tag_points", "bindings": bindings, "tag": TAG,
              "specification": preparation["specification"], "image_arrays_read": False,
              "models_loaded": False, "fit_or_selection_run": False}
    numerical = verify_records(summary, coordinates, pairs, matrix, preflight["headers"], common, args.coordinates_sha)
    for relative, wanted in {**expected, **source_pins, TAG["path"]: TAG["sha256"]}.items():
        require(sha(ROOT / relative) == wanted, "after-audit fixity: " + relative)
    return {"schema": "resect-case3-evaluation-independent-saved-result-v1", "status": "pass",
            "case_id": "RESECT:Case3", "role": "TRAIN", "auditor_sha256": sha(__file__),
            "oracle_sha256": HELPER_SHA, "pins": expected, "compressed_source_pins": source_pins,
            "full_tag_sha256": TAG["sha256"], "all_point_numerical_recount": numerical,
            "all_saved_coordinates_predictions_voxels_verified": True, "all_before_after_hashes_unchanged": True,
            "comparison_tolerance": {"absolute": 1e-9, "relative": 1e-11},
            "coordinate_values_in_report": False, "acquired_image_arrays_read": False,
            "full_tag_read": True, "short_tag_read": False, "model_or_optimizer_run": False,
            "denominator": {"actual_image_only_fits": 1, "initial_scoring_refusals": 1,
                            "completed_corrected_scoring_outputs": 1, "people": 1, "full_tag_rows": 17},
            "anatomical_alignment_accepted": False, "clinical_acceptance": False,
            "limits": ["One TRAIN person; source tag was exposed for syntax development before scoring.",
                       "All full-tag rows retained, including duplicates or out-of-image points.",
                       "Numerical agreement is not held-out accuracy or clinical alignment acceptance."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--generated", action="store_true")
    modes.add_argument("--actual", action="store_true")
    parser.add_argument("--release", type=Path)
    parser.add_argument("--release-sha")
    parser.add_argument("--summary-sha")
    parser.add_argument("--coordinates-sha")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(args.output.resolve().is_relative_to(DEST.resolve()) and not args.output.exists(), "new QA record only")
    result = generated() if args.generated else actual(args)
    qa.publish(args.output, result)
    print(json.dumps({"status": result["status"], "report_sha256": sha(args.output),
                      "actual_score_verification_run": args.actual}))


if __name__ == "__main__":
    main()
