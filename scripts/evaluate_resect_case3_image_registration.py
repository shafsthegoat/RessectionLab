#!/usr/bin/env python3
"""Fixed all-point TRAIN evaluation, gated separately from the image-only fit.

Preparation reads source code, cohort and preflight metadata only. Evaluation
opens the exact full tag only after the sealed fit chain and a separate ROOT
release have passed. Images, masks, models and the overlapping short tag are
never read. Coordinate-bearing results may be written only beneath outputs/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "resect-case3-registration-evaluation-v1"
FIT_SCHEMA = "resect-case3-image-registration-v1"
CASE = "RESECT:Case3"
COHORT = "manifests/resect-component-cohort-v1.json"
COHORT_SHA = "4fc51f01e42253d395660d850cda2296a3e7f23e5a5ade80bbd5a3ed411b6a77"
RUNNER = "scripts/poc_resect_case3_image_registration.py"
RUNNER_SHA = "e17d755eeadb6a3d1fd365b7a3b4dc31f9e27c4f93afdb6045d8f6e27ef9612f"
PREFLIGHT = "artifacts/resect-case3-image-registration-preparation-v1/header-preflight-rss-v2.json"
PREFLIGHT_SHA = "ef9e0cd6029e869a6364de6c946f0284bc0a7ca1f700695a8bb75b52e60fd7ba"
TESTS = "tests/test_evaluate_resect_case3_image_registration.py"
PREPARATION_DIR = "artifacts/resect-case3-registration-evaluation-preparation-v1"
TAG = {"path": "data/anatomy/resect-v1/Case3/Landmarks/Case3-beforeUS-duringUS-full.tag",
       "bytes": 1190, "sha256": "585769a006d8e5a995f987d91cef75ca45288cf6132aa3842cd44ca19927fc08"}
IMAGES = {
    "before": {"path": "data/anatomy/resect-v1/Case3/US/Case3-US-before.nii.gz",
               "bytes": 9822564, "sha256": "9fbb4d63cb26d324c7c8281a2b2b932a55a5f3abf4a8fd571a0e0518ac996192"},
    "during": {"path": "data/annotations/resect-seg-v1/originals/Case3-US-during.nii.gz",
               "bytes": 9156131, "sha256": "431ae1e23bd2984801c0b0d2720c9f4c240a404bab5fe37f5d8afdac521ddb20"},
}
MAX_RECORD = 1024 * 1024
FORWARD = "before_RAS_mm_to_during_RAS_mm"
INVERSE = "during_RAS_mm_to_before_RAS_mm"
DIRECTION = "before-RAS output physical point -> during-RAS input physical point"
SPEC = {
    "tag_format": "ASCII MNI Tag Point File; Volumes=2; exactly six numbers per point row",
    "point_order": "first three before/reference world mm; last three during world mm; original 1-based row indices",
    "coordinate_convention": "MINC and NIfTI world coordinates identical; active sform RAS mm; no LPS conversion of tag points",
    "transform_direction": FORWARD,
    "identity_baseline": "before point unchanged in the shared RAS-mm frame",
    "tre": "Euclidean norm(predicted during point minus observed during point), millimetres",
    "primary_endpoint": "delta_RMS_mm = RMS(rigid TRE) - RMS(identity TRE); negative is lower error",
    "statistics": ["all-point RMS", "all-point median", "all-point maximum", "per-row rigid-minus-identity TRE"],
    "inclusion": "inverse active sform; closed native voxel-cell bounds [-0.5, shape-0.5]",
    "inclusion_tolerance_voxels": 1e-7,
    "inclusion_policy": "report before, during, identity-predicted during and rigid-predicted during inclusion; retain every finite row",
    "duplicate_policy": "retain every row at equal weight; flag exact repeated pair/before/during indices; never deduplicate",
    "malformed_or_nonfinite_policy": "reject the complete evaluation; do not drop individual rows",
    "short_tag_policy": "never read or split; overlaps the full tag",
    "parameter_selection": False,
    "refitting": False,
    "exclusions": False,
    "population_confidence_intervals": False,
    "clinical_acceptance": False,
    "unit_of_observation": "one TRAIN person; points are paired observations within that person",
}


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def no_duplicate_keys(pairs):
    record = {}
    for key, value in pairs:
        if key in record:
            raise ValueError("DUPLICATE_JSON_KEY")
        record[key] = value
    return record


def bounded_bytes(path, maximum=MAX_RECORD):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > maximum:
        raise ValueError("MISSING_LINKED_OR_OVERSIZED_RECORD")
    with path.open("rb") as stream:
        payload = stream.read(maximum + 1)
    if len(payload) > maximum:
        raise ValueError("RECORD_SIZE_CHANGED")
    return payload


def metadata(path):
    path = Path(path)
    # No data/ file, including a .tag disguised as metadata, is allowed here.
    resolved = path.resolve()
    if path.suffix != ".json" or not any(resolved.is_relative_to((ROOT / part).resolve())
                                        for part in ("artifacts", "outputs", "manifests")):
        raise ValueError("METADATA_PATH_NOT_ALLOWED")
    payload = bounded_bytes(path)
    value = json.loads(payload, object_pairs_hook=no_duplicate_keys,
                       parse_constant=lambda _: (_ for _ in ()).throw(ValueError("NONFINITE_JSON")))
    if not isinstance(value, dict):
        raise ValueError("METADATA_OBJECT_REQUIRED")
    return value, sha_bytes(payload)


def script_hash(path):
    return sha_bytes(bounded_bytes(path))


def cohort_gate():
    cohort, actual = metadata(ROOT / COHORT)
    members = [m for m in cohort.get("members", []) if m.get("patient_group") == CASE]
    if (actual != COHORT_SHA or len(members) != 1 or members[0].get("case_id") != "Case3"
            or members[0].get("role") != "TRAIN"
            or members[0].get("all_timepoints_and_derivatives_grouped") is not True):
        raise ValueError("EXACT_CASE3_TRAIN_COHORT_REQUIRED")


def rigid_matrix(value):
    matrix = np.asarray(value, dtype=float)
    if (matrix.shape != (4, 4) or not np.isfinite(matrix).all()
            or not np.array_equal(matrix[3], [0, 0, 0, 1])
            or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-7, rtol=0)
            or not math.isclose(float(np.linalg.det(matrix[:3, :3])), 1., abs_tol=1e-7)):
        raise ValueError("FINITE_PROPER_RIGID_RAS_TRANSFORM_REQUIRED")
    return matrix


def header_geometry(header):
    affine = np.asarray(header.get("selected_affine"), dtype=float)
    shape = header.get("shape")
    if (not isinstance(shape, list) or len(shape) != 3
            or any(type(n) is not int or n < 1 for n in shape)
            or header.get("sform_code") != 1 or header.get("qform_code") != 0
            or header.get("spatial_units") != "mm" or header.get("image_array_accessed") is not False
            or header.get("affine_replaced") is not False
            or affine.shape != (4, 4) or not np.isfinite(affine).all()
            or not np.array_equal(affine[3], [0, 0, 0, 1])
            or abs(float(np.linalg.det(affine[:3, :3]))) < 1e-12
            or not re.fullmatch(r"[0-9a-f]{64}", header.get("header_sha256", ""))):
        raise ValueError("ACTIVE_SFORM_NATIVE_HEADER_REQUIRED")
    fields = ("header_sha256", "shape", "selected_affine", "sform_code", "qform_code", "spatial_units",
              "image_array_accessed", "affine_replaced")
    return {key: header[key] for key in fields}


def validate_preflight(value):
    if (value.get("schema") != FIT_SCHEMA + "-preflight"
            or value.get("status") != "headers_verified_fit_not_run" or value.get("case_id") != CASE
            or value.get("role") != "TRAIN" or value.get("cohort_sha256") != COHORT_SHA
            or value.get("decoded_array_bytes") != 0 or value.get("landmark_coordinates_read") is not False
            or value.get("fit_run") is not False or value.get("anatomical_alignment_accepted") is not False):
        raise ValueError("FIXED_PREFLIGHT_REQUIRED")
    for name, identity in IMAGES.items():
        if any(value.get("inputs", {}).get(name, {}).get(key) != expected for key, expected in identity.items()):
            raise ValueError("PREFLIGHT_IMAGE_IDENTITY_CHANGED")
        header_geometry(value["headers"][name])


def prepare(preflight_path, destination):
    """Metadata-only protocol freeze. This function cannot open a tag or image."""
    cohort_gate()
    preflight, preflight_sha = metadata(preflight_path)
    if Path(preflight_path).resolve() != (ROOT / PREFLIGHT).resolve() or preflight_sha != PREFLIGHT_SHA:
        raise ValueError("EXACT_PREFLIGHT_RECEIPT_REQUIRED")
    validate_preflight(preflight)
    runner_sha = script_hash(ROOT / RUNNER)
    if preflight.get("runner_sha256") != runner_sha or runner_sha != RUNNER_SHA:
        raise ValueError("PREFLIGHT_RUNNER_CHANGED")
    destination = Path(destination)
    if destination.resolve().parent != (ROOT / PREPARATION_DIR).resolve():
        raise ValueError("PREPARATION_DESTINATION_REQUIRED")
    record = {"schema": SCHEMA + "-preparation", "status": "prepared_actual_evaluation_withheld",
              "case_id": CASE, "role": "TRAIN", "specification": SPEC, "tag": TAG, "images": IMAGES,
              "cohort": {"path": COHORT, "sha256": COHORT_SHA},
              "evaluator_sha256": script_hash(__file__), "tests_sha256": script_hash(ROOT / TESTS),
              "runner_sha256": runner_sha, "preflight_sha256": preflight_sha,
              "scoring_runtime": {"python": sys.version.split()[0], "numpy": np.__version__},
              "header_geometry_sha256": {name: sha_bytes(encoded(header_geometry(preflight["headers"][name])))
                                         for name in IMAGES},
              "fit_configuration": preflight["configuration"], "fit_limits": preflight["limits"],
              "acquired_landmark_coordinates_read": False, "acquired_image_arrays_read": False,
              "actual_evaluation_run": False,
              "release_requirement": "Separate ROOT-authored release binding this preparation, evaluator and complete successful fit metadata chain before any tag open"}
    publish(destination, record)
    return record


def bind(record, key, actual):
    if record.get(key) != actual:
        raise ValueError("RELEASE_OR_FREEZE_BINDING_CHANGED: " + key)


def output_path_gate(output):
    output = Path(output)
    if output.exists() or output.is_symlink() or not output.resolve().is_relative_to((ROOT / "outputs").resolve()):
        raise ValueError("NEW_OUTPUTS_DIRECTORY_REQUIRED")
    return output


def evaluation_gate(fit_directory, preflight_path, fit_release_path, preparation_path, release_path, output):
    """All operations here are metadata-only; no image or landmark byte reads."""
    output = output_path_gate(output)
    fit = Path(fit_directory)
    if not fit.resolve().is_relative_to((ROOT / "outputs").resolve()):
        raise ValueError("FIT_OUTPUT_DIRECTORY_REQUIRED")
    release, release_sha = metadata(release_path)
    if (release.get("schema") != FIT_SCHEMA + "-evaluation-release" or release.get("issuer") != "ROOT"
            or release.get("decision") != "release_separate_evaluation" or release.get("case_id") != CASE
            or release.get("role") != "TRAIN" or release.get("full_tag_role") != "evaluation_only_all_points"
            or release.get("refitting_or_parameter_selection_permitted") is not False
            or release.get("landmark_access_permitted") is not True or release.get("tag") != TAG
            or release.get("cohort") != {"path": COHORT, "sha256": COHORT_SHA}
            or release.get("fit_directory") != str(fit.resolve())
            or release.get("output_directory") != str(output.resolve())):
        raise ValueError("SEPARATE_EXACT_ROOT_EVALUATION_RELEASE_REQUIRED")
    preparation, preparation_sha = metadata(preparation_path)
    if (Path(preparation_path).resolve().parent != (ROOT / PREPARATION_DIR).resolve()
            or preparation.get("schema") != SCHEMA + "-preparation"
            or preparation.get("status") != "prepared_actual_evaluation_withheld"
            or preparation.get("case_id") != CASE or preparation.get("role") != "TRAIN"
            or preparation.get("specification") != SPEC or preparation.get("tag") != TAG
            or preparation.get("images") != IMAGES
            or preparation.get("cohort") != {"path": COHORT, "sha256": COHORT_SHA}
            or preparation.get("acquired_landmark_coordinates_read") is not False
            or preparation.get("acquired_image_arrays_read") is not False
            or preparation.get("actual_evaluation_run") is not False):
        raise ValueError("FIXED_PREPARATION_REQUIRED")
    hashes = {"preparation_sha256": preparation_sha, "evaluator_sha256": script_hash(__file__),
              "tests_sha256": script_hash(ROOT / TESTS), "runner_sha256": script_hash(ROOT / RUNNER)}
    if (hashes["runner_sha256"] != RUNNER_SHA
            or preparation.get("scoring_runtime") != {"python": sys.version.split()[0], "numpy": np.__version__}):
        raise ValueError("FROZEN_RUNNER_OR_SCORING_RUNTIME_CHANGED")
    for key, actual in hashes.items():
        bind(release, key, actual)
        if key != "preparation_sha256":
            bind(preparation, key, actual)
    cohort_gate()
    preflight, hashes["preflight_sha256"] = metadata(preflight_path)
    if Path(preflight_path).resolve() != (ROOT / PREFLIGHT).resolve() or hashes["preflight_sha256"] != PREFLIGHT_SHA:
        raise ValueError("EXACT_PREFLIGHT_RECEIPT_REQUIRED")
    validate_preflight(preflight)
    bind(preparation, "preflight_sha256", hashes["preflight_sha256"])
    bind(preflight, "runner_sha256", hashes["runner_sha256"])
    bind(preflight, "configuration", preparation["fit_configuration"])
    bind(preflight, "limits", preparation["fit_limits"])
    for name in IMAGES:
        bind(preparation["header_geometry_sha256"], name, sha_bytes(encoded(header_geometry(preflight["headers"][name]))))
    fit_release, hashes["fit_release_sha256"] = metadata(fit_release_path)
    if (fit_release.get("schema") != FIT_SCHEMA + "-root-release" or fit_release.get("decision") != "release_one_fit"
            or fit_release.get("case_id") != CASE or fit_release.get("landmark_access_permitted") is not False
            or fit_release.get("output_directory") != str(fit.resolve())):
        raise ValueError("BOUND_IMAGE_ONLY_FIT_RELEASE_REQUIRED")
    for key in ("preflight_sha256", "runner_sha256"):
        bind(fit_release, key, hashes[key])
    result, hashes["fit_result_sha256"] = metadata(fit / "result.json")
    frozen, hashes["frozen_transform_sha256"] = metadata(fit / "frozen-transform.json")
    candidate, hashes["candidate_sha256"] = metadata(fit / "candidate.json")
    supervision, hashes["supervision_sha256"] = metadata(fit / "supervision.json")
    for key, actual in hashes.items():
        bind(release, key, actual)
    if (result.get("schema") != FIT_SCHEMA + "-result" or result.get("case_id") != CASE
            or result.get("status") != "completed" or result.get("exit_code") != 0
            or result.get("evaluation_run") is not False or result.get("anatomical_alignment_accepted") is not False):
        raise ValueError("SUCCESSFULLY_SUPERVISED_FIT_REQUIRED")
    for key in ("preflight_sha256", "fit_release_sha256", "frozen_transform_sha256", "supervision_sha256"):
        bind(result, key, hashes[key])
    cap = preparation["fit_limits"].get("sampled_family_rss_stop_bytes")
    if (not isinstance(cap, (int, float)) or cap <= 0
            or supervision.get("schema") != FIT_SCHEMA + "-supervision"
            or supervision.get("status") != "completed" or supervision.get("exit_code") != 0
            or supervision.get("worker_reaped") is not True or supervision.get("process_group_cleanup_attempted") is not True
            or supervision.get("samples", 0) < 1
            or supervision.get("sampled_family_rss_stop_bytes") != cap
            or not 0 <= supervision.get("peak_family_rss_bytes_sampled", math.inf) <= cap):
        raise ValueError("SUCCESSFUL_MEMORY_SUPERVISION_REQUIRED")
    if (candidate.get("schema") != FIT_SCHEMA + "-candidate" or candidate.get("case_id") != CASE
            or candidate.get("sources_rechecked_after_each_decode") is not True
            or candidate.get("source_fixity_after_fit") != {name: row["sha256"] for name, row in IMAGES.items()}
            or candidate.get("runtime") != preflight.get("runtime")):
        raise ValueError("CANDIDATE_SOURCE_OR_RUNTIME_CHANGED")
    for key in ("preflight_sha256", "fit_release_sha256", "runner_sha256"):
        bind(candidate, key, hashes[key])
    expected_freeze = {**candidate, "schema": FIT_SCHEMA + "-frozen", "candidate_sha256": hashes["candidate_sha256"],
                       "supervision_sha256": hashes["supervision_sha256"],
                       "evaluation_requires_separate_root_release": True, "anatomical_alignment_accepted": False}
    if frozen != expected_freeze:
        raise ValueError("FROZEN_CANDIDATE_CHAIN_CHANGED")
    report = frozen["report"]
    if (report.get("fixed") != "before_resection_US" or report.get("moving") != "during_resection_US"
            or report.get("sampling_direction") != DIRECTION or report.get("landmarks_used") is not False
            or report.get("configuration") != preparation["fit_configuration"]
            or report.get("identity_baseline_before_RAS_to_during_RAS") != np.eye(4).tolist()):
        raise ValueError("FIXED_IMAGE_ONLY_RAS_DIRECTION_REQUIRED")
    forward, inverse = rigid_matrix(report.get(FORWARD)), rigid_matrix(report.get(INVERSE))
    if not np.allclose(inverse @ forward, np.eye(4), atol=1e-9, rtol=0):
        raise ValueError("FORWARD_INVERSE_MISMATCH")
    return {"headers": preflight["headers"], "matrix": forward,
            "bindings": {**hashes, "evaluation_release_sha256": release_sha}}


NUMBER = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?\Z")


def parse_full_tag(payload):
    """Strict source-declared six-value rows. Indices include duplicate rows."""
    if type(payload) is not bytes or not 0 < len(payload) <= MAX_RECORD:
        raise ValueError("BOUNDED_IMMUTABLE_TAG_REQUIRED")
    try:
        lines = [line.split("%", 1)[0].strip() for line in payload.decode("ascii").splitlines()]
    except UnicodeError as error:
        raise ValueError("ASCII_TAG_REQUIRED") from error
    lines = [line for line in lines if line]
    if not lines or lines[0] != "MNI Tag Point File":
        raise ValueError("MNI_TAG_HEADER_REQUIRED")
    body = "\n".join(lines[1:])
    match = re.match(r"\AVolumes\s*=\s*2\s*;\s*Points\s*=", body)
    if match is None:
        raise ValueError("TWO_VOLUMES_POINTS_REQUIRED")
    body = body[match.end():].strip()
    if not body.endswith(";") or body.count(";") != 1:
        raise ValueError("SINGLE_FINAL_TAG_TERMINATOR_REQUIRED")
    pairs = []
    for line in body[:-1].splitlines():
        if not line.strip():
            continue
        tokens = line.split()
        if len(tokens) != 6 or any(NUMBER.fullmatch(token) is None for token in tokens):
            raise ValueError("EXACTLY_SIX_FINITE_NUMBERS_PER_ROW_REQUIRED")
        row = [float(token) for token in tokens]
        if not np.isfinite(row).all():
            raise ValueError("NONFINITE_TAG_ROW")
        pairs.append(row)
    if not pairs:
        raise ValueError("NONEMPTY_PAIRED_TAG_REQUIRED")
    return np.asarray(pairs, dtype=float)


def map_points(matrix, points):
    return np.asarray(points) @ np.asarray(matrix)[:3, :3].T + np.asarray(matrix)[:3, 3]


def inclusion(points, header):
    geometry = header_geometry(header)
    voxels = map_points(np.linalg.inv(geometry["selected_affine"]), points)
    tolerance = SPEC["inclusion_tolerance_voxels"]
    flags = ((voxels >= -.5 - tolerance) & (voxels <= np.asarray(geometry["shape"]) - .5 + tolerance)).all(axis=1)
    return voxels, flags


def metrics(values):
    return {"RMS_mm": float(np.sqrt(np.mean(np.square(values)))),
            "median_mm": float(np.median(values)), "max_mm": float(np.max(values))}


def score_pairs(pairs, matrix, headers):
    matrix = rigid_matrix(matrix)
    pairs = np.asarray(pairs, dtype=float)
    if pairs.ndim != 2 or pairs.shape[1] != 6 or not len(pairs) or not np.isfinite(pairs).all():
        raise ValueError("ALL_FINITE_SIX_VALUE_PAIRS_REQUIRED")
    before, during = pairs[:, :3], pairs[:, 3:]
    predicted = map_points(matrix, before)
    identity_tre = np.linalg.norm(before - during, axis=1)
    rigid_tre = np.linalg.norm(predicted - during, axis=1)
    if not np.isfinite(predicted).all() or not np.isfinite(identity_tre).all() or not np.isfinite(rigid_tre).all():
        raise ValueError("NONFINITE_SCORING_RESULT")
    included = {"before_native": inclusion(before, headers["before"]),
                "during_native": inclusion(during, headers["during"]),
                "identity_in_during_native": inclusion(before, headers["during"]),
                "rigid_in_during_native": inclusion(predicted, headers["during"])}
    seen = {key: {} for key in ("pair", "before", "during")}
    rows, coordinate_rows = [], []
    for offset in range(len(pairs)):
        index = offset + 1
        duplicates = {}
        for key, values in (("pair", pairs[offset]), ("before", before[offset]), ("during", during[offset])):
            token = tuple(values)
            duplicates[key] = seen[key].get(token)
            seen[key].setdefault(token, index)
        row = {"point_index": index, "identity_TRE_mm": float(identity_tre[offset]),
               "rigid_TRE_mm": float(rigid_tre[offset]),
               "paired_rigid_minus_identity_TRE_mm": float(rigid_tre[offset] - identity_tre[offset]),
               "included_in_native_image": {key: bool(value[1][offset]) for key, value in included.items()},
               "duplicate_of_first_point_index": duplicates, "retained_in_all_statistics": True}
        rows.append(row)
        coordinate_rows.append({**row, "before_RAS_mm": before[offset].tolist(), "during_RAS_mm": during[offset].tolist(),
                                "predicted_during_RAS_mm": predicted[offset].tolist(),
                                "native_voxel_coordinates": {key: value[0][offset].tolist() for key, value in included.items()}})
    identity, fitted = metrics(identity_tre), metrics(rigid_tre)
    summary = {"point_count": len(pairs), "identity": identity, "rigid": fitted,
               "primary_delta_RMS_mm_rigid_minus_identity": fitted["RMS_mm"] - identity["RMS_mm"],
               "inclusion_failure_indices": {key: [i + 1 for i, flag in enumerate(value[1]) if not flag]
                                             for key, value in included.items()},
               "excluded_point_indices": [], "rows": rows, "population_confidence_interval": None,
               "clinical_acceptance": None, "anatomical_alignment_accepted": False,
               "interpretation": "One TRAIN person; all finite full-tag pairs equally weighted. No retuning, exclusions or population inference."}
    return summary, coordinate_rows


def publish(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = encoded(record)
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    path.chmod(0o444)


def evaluate(fit_directory, preflight_path, fit_release_path, preparation_path, release_path, output):
    gate = evaluation_gate(fit_directory, preflight_path, fit_release_path, preparation_path, release_path, output)
    # First and only acquired payload read: after every metadata and release check.
    payload = bounded_bytes(ROOT / TAG["path"], TAG["bytes"])
    if len(payload) != TAG["bytes"] or sha_bytes(payload) != TAG["sha256"]:
        raise ValueError("EXACT_FULL_TAG_FIXITY_REQUIRED")
    pairs = parse_full_tag(payload)
    summary, coordinates = score_pairs(pairs, gate["matrix"], gate["headers"])
    common = {"schema": SCHEMA, "case_id": CASE, "role": "TRAIN", "status": "scored_all_full_tag_points",
              "bindings": gate["bindings"], "tag": TAG, "specification": SPEC,
              "image_arrays_read": False, "models_loaded": False, "fit_or_selection_run": False}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    publish(output / "all-points-with-coordinates.json", {**common, "points": coordinates})
    compact = {**common, **summary, "coordinate_output_sha256": script_hash(output / "all-points-with-coordinates.json")}
    publish(output / "summary-no-coordinates.json", compact)
    return compact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="stage", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--preflight", type=Path, required=True)
    prep.add_argument("--output", type=Path, required=True)
    run = commands.add_parser("evaluate")
    for argument in ("fit-directory", "preflight", "fit-release", "preparation", "release", "output"):
        run.add_argument("--" + argument, type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.stage == "prepare":
            result = prepare(args.preflight, args.output)
        else:
            result = evaluate(args.fit_directory, args.preflight, args.fit_release, args.preparation, args.release, args.output)
        print(json.dumps({"status": result["status"], "output": str(args.output),
                          "actual_evaluation_run": args.stage == "evaluate"}, allow_nan=False))
    except (ValueError, OSError, KeyError, TypeError) as error:
        # Do not print data values or tracebacks that could contain coordinates.
        print(json.dumps({"status": "refused", "error_type": type(error).__name__}), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
