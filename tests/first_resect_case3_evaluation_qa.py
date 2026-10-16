#!/usr/bin/env python3
"""Opt-in independent all-point QA. Default/generated mode never opens a tag.

Actual mode requires a separately reviewed pin file and explicit ROOT release.
No fitting library, image decoder, production parser or production arithmetic is
used by the actual-result oracle. Coordinates are retained only in memory.
"""
from __future__ import annotations

import argparse
import ast
import copy
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
from pathlib import Path
import re
import shlex
import statistics
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "artifacts/resect-case3-evaluation-independent-v1"
SOURCE = ROOT / "scripts/evaluate_resect_case3_image_registration.py"
OLD = DEST / "initial-evaluator.py.txt"
OLD_SHA = "ac821134851d4b310ae6e5f7b12187e36b90266a8105ffce0737c4876af5cab0"
TAG = "data/anatomy/resect-v1/Case3/Landmarks/Case3-beforeUS-duringUS-full.tag"
TAG_SHA = "585769a006d8e5a995f987d91cef75ca45288cf6132aa3842cd44ca19927fc08"
FIT = "outputs/poc-resect-case3-image-registration-v1/"
PREFLIGHT = "artifacts/resect-case3-image-registration-preparation-v1/header-preflight-rss-v2.json"
FORWARD = "before_RAS_mm_to_during_RAS_mm"
INVERSE = "during_RAS_mm_to_before_RAS_mm"
DOMAINS = ("before_native", "during_native", "identity_in_during_native", "rigid_in_during_native")


def require(value, message):
    if not value:
        raise AssertionError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def record(path):
    def unique(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key")
            result[key] = value
        return result
    path = Path(path)
    require(path.stat().st_size <= 1024 * 1024 and not path.is_symlink(), "bounded plain metadata")
    return json.loads(path.read_bytes(), object_pairs_hook=unique,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))


def publish(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    path.chmod(0o444)


def independent_parse(payload):
    """Independent line/statement parser; Decimal and shlex preserve quote type."""
    require(type(payload) is bytes and 0 < len(payload) <= 1024 * 1024, "bounded tag")
    text = payload.decode("ascii")
    text = "\n".join(line.partition("%")[0] for line in text.splitlines()).strip()
    header, rest = text.split("\n", 1)
    require(header.strip() == "MNI Tag Point File", "header")
    declaration, points = rest.split(";", 1)
    require(re.fullmatch(r"\s*Volumes\s*=\s*2\s*", declaration) is not None, "two volumes")
    key, body = points.split("=", 1)
    require(key.strip() == "Points" and body.rstrip().endswith(";"), "points/final terminator")
    body = body.rstrip()[:-1]
    require(";" not in body, "single terminator")
    rows = []
    for line in body.splitlines():
        if not line.strip():
            continue
        lexer = shlex.shlex(line, posix=False)
        lexer.whitespace_split = True
        lexer.commenters = ""
        fields = list(lexer)
        if len(fields) == 7:
            require(fields.pop() == '""', "only optional empty label")
        require(len(fields) == 6, "six numbers")
        values = []
        for field in fields:
            require(not any(c in field for c in '_"\''), "unquoted decimal")
            value = Decimal(field)
            require(value.is_finite(), "finite decimal")
            converted = float(value)
            require(math.isfinite(converted), "finite binary value")
            values.append(converted)
        rows.append(values)
    require(bool(rows), "nonempty full tag")
    return rows


def invert(matrix):
    """Small Gauss-Jordan inverse, independent of the NumPy scorer."""
    work = [[float(value) for value in row] + [float(i == j) for j in range(4)]
            for i, row in enumerate(matrix)]
    for column in range(4):
        pivot = max(range(column, 4), key=lambda i: abs(work[i][column]))
        require(abs(work[pivot][column]) > 1e-15, "invertible frame")
        work[column], work[pivot] = work[pivot], work[column]
        divisor = work[column][column]
        work[column] = [value / divisor for value in work[column]]
        for i in range(4):
            if i != column:
                factor = work[i][column]
                work[i] = [a - factor * b for a, b in zip(work[i], work[column])]
    return [row[4:] for row in work]


def transform(matrix, point):
    return [math.fsum(matrix[i][j] * point[j] for j in range(3)) + matrix[i][3] for i in range(3)]


def stats(values):
    return {"RMS_mm": math.hypot(*values) / math.sqrt(len(values)),
            "median_mm": statistics.median(values), "max_mm": max(values)}


def oracle(pairs, matrix, headers):
    frames = {key: invert(header["selected_affine"]) for key, header in headers.items()}
    rows, coordinates, identity_errors, rigid_errors = [], [], [], []
    seen = {key: {} for key in ("pair", "before", "during")}
    failed = {key: [] for key in DOMAINS}
    for index, values in enumerate(pairs, 1):
        before, during = values[:3], values[3:]
        predicted = transform(matrix, before)
        identity, rigid = math.dist(before, during), math.dist(predicted, during)
        identity_errors.append(identity)
        rigid_errors.append(rigid)
        duplicates = {}
        for key, point in (("pair", values), ("before", before), ("during", during)):
            token = tuple(point)
            duplicates[key] = seen[key].get(token)
            seen[key].setdefault(token, index)
        inclusion, voxels = {}, {}
        for name, image, point in zip(DOMAINS, ("before", "during", "during", "during"),
                                      (before, during, before, predicted)):
            voxels[name] = transform(frames[image], point)
            inclusion[name] = all(-.5 - 1e-7 <= value <= size - .5 + 1e-7
                                  for value, size in zip(voxels[name], headers[image]["shape"]))
            if not inclusion[name]:
                failed[name].append(index)
        row = {"point_index": index, "identity_TRE_mm": identity, "rigid_TRE_mm": rigid,
               "paired_rigid_minus_identity_TRE_mm": rigid - identity,
               "included_in_native_image": inclusion, "duplicate_of_first_point_index": duplicates,
               "retained_in_all_statistics": True}
        rows.append(row)
        coordinates.append({**row, "before_RAS_mm": before, "during_RAS_mm": during,
                            "predicted_during_RAS_mm": predicted, "native_voxel_coordinates": voxels})
    identity, fitted = stats(identity_errors), stats(rigid_errors)
    return {"point_count": len(pairs), "identity": identity, "rigid": fitted,
            "primary_delta_RMS_mm_rigid_minus_identity": fitted["RMS_mm"] - identity["RMS_mm"],
            "inclusion_failure_indices": failed, "excluded_point_indices": [], "rows": rows}, coordinates


def equal(actual, expected):
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected), "record keys differ")
        for key in expected:
            equal(actual[key], expected[key])
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), "list length differs")
        for left, right in zip(actual, expected):
            equal(left, right)
    elif type(expected) is float:
        require(type(actual) in (int, float) and math.isclose(actual, expected, rel_tol=1e-11, abs_tol=1e-9),
                "numerical value differs")
    else:
        require(type(actual) is type(expected) and actual == expected, "categorical value differs")


def module(path):
    namespace = {"__file__": str(path), "__name__": "qa_loaded_evaluator"}
    exec(compile(Path(path).read_text(), str(path), "exec"), namespace)
    return namespace


def generated():
    require(digest(OLD) == OLD_SHA, "initial evaluator snapshot")
    before, after = module(OLD), module(SOURCE)
    # Exact permitted AST change: optional empty label parser, tag_format and prep destination.
    old_ast, new_ast = ast.parse(OLD.read_text()), ast.parse(SOURCE.read_text())
    allowed = {"PREPARATION_DIR", "SPEC", "PRIOR_TAG_EXPOSURE"}
    # Only two additional disclosure fields may differ in prepare().
    for node in ast.walk(new_ast):
        if isinstance(node, ast.Dict):
            retained = [(key, value) for key, value in zip(node.keys, node.values)
                        if not (isinstance(key, ast.Constant) and key.value in
                                ("operation_scope", "prior_actual_tag_exposure"))]
            node.keys, node.values = [p[0] for p in retained], [p[1] for p in retained]
    def untouched(tree):
        return [ast.dump(node, include_attributes=False) for node in tree.body
                if not (isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in allowed for t in node.targets))
                and not (isinstance(node, ast.FunctionDef) and node.name == "parse_full_tag")]
    require(untouched(old_ast) == untouched(new_ast), "unexpected evaluator change outside approved scope")
    require({k: v for k, v in before["SPEC"].items() if k != "tag_format"}
            == {k: v for k, v in after["SPEC"].items() if k != "tag_format"}, "scientific settings changed")
    require(after["PREPARATION_DIR"].endswith("-v2"), "separate preparation required")
    old_parser = next(n for n in old_ast.body if isinstance(n, ast.FunctionDef) and n.name == "parse_full_tag")
    new_parser = next(n for n in new_ast.body if isinstance(n, ast.FunctionDef) and n.name == "parse_full_tag")
    permitted = ast.parse("if len(tokens) == 7 and tokens[-1] == '\"\"':\n    tokens = tokens[:-1]").body[0]
    removed = 0
    for node in ast.walk(new_parser):
        if isinstance(node, ast.For):
            keep = []
            for child in node.body:
                if ast.dump(child) == ast.dump(permitted):
                    removed += 1
                else:
                    keep.append(child)
            node.body = keep
    old_parser.body = old_parser.body[1:]
    new_parser.body = new_parser.body[1:]
    require(removed == 1 and ast.dump(old_parser) == ast.dump(new_parser), "parser change exceeds empty label")
    checks = ["initial source pin", "unchanged outside parser and preparation", "all scientific settings unchanged"]
    matrix = [[0., -1., 0., 2.], [1., 0., 0., -3.], [0., 0., 1., 1.], [0., 0., 0., 1.]]
    frame = [[.8, -.6, 0., 20.], [.6, .8, 0., -10.], [0., 0., 2., 7.], [0., 0., 0., 1.]]
    headers = {name: {"shape": [3, 4, 5], "selected_affine": frame, "sform_code": 1, "qform_code": 0,
                     "spatial_units": "mm", "image_array_accessed": False, "affine_replaced": False,
                     "header_sha256": "a" * 64} for name in ("before", "during")}
    pairs = [[float(i), float(i % 3), -float(i), float(i + 1), float(i % 5), float(i)] for i in range(17)]
    pairs[3] = pairs[0][:]
    pairs[4][:3] = pairs[0][:3]
    pairs[5][3:] = pairs[0][3:]
    def payload(rows, suffix=' ""', inline=True):
        return ("MNI Tag Point File\nVolumes = 2;\nPoints =\n"
                + "\n".join(" ".join(map(str, row)) + suffix for row in rows)
                + (";\n" if inline else "\n;\n")).encode()
    for suffix, inline in (("", True), (' ""', True), (' ""', False)):
        raw = payload(pairs, suffix, inline)
        equal(independent_parse(raw), pairs)
        equal(after["parse_full_tag"](raw).tolist(), pairs)
        expected, coords = oracle(pairs, matrix, headers)
        scored, saved = after["score_pairs"](after["parse_full_tag"](raw), matrix, headers)
        equal({key: scored[key] for key in expected}, expected)
        equal(saved, coords)
        checks.append("17-row independent parser/arithmetic " + repr((suffix, inline)))
    try:
        before["parse_full_tag"](payload(pairs))
    except ValueError:
        checks.append("original exact-six-token refusal reproduced without patient data")
    else:
        raise AssertionError("original parser unexpectedly accepted empty labels")
    for suffix in (' "nonempty"', ' " "', ' "', ' "" ""', ' 7 ""', ' "" trailing', " ' ' "):
        for parser in (independent_parse, after["parse_full_tag"]):
            try:
                parser(payload(pairs, suffix))
            except (AssertionError, ValueError, InvalidOperation):
                pass
            else:
                raise AssertionError("unexpected label or field accepted")
        checks.append("reject malformed or nonempty label " + repr(suffix))
    for invalid in ("NaN", "Inf", "-Inf", "1e999", "1_000"):
        altered = copy.deepcopy(pairs)
        altered[7][5] = invalid
        for parser in (independent_parse, after["parse_full_tag"]):
            try:
                parser(payload(altered))
            except (AssertionError, ValueError, InvalidOperation):
                pass
            else:
                raise AssertionError("invalid number accepted")
        checks.append("reject complete file for invalid row " + invalid)
    # Known physical-cell boundary witness under an oblique affine.
    for voxel, wanted in (([-.5, -.5, -.5], True), ([2.5, 3.5, 4.5], True),
                          ([-.500001, 0., 0.], False), ([2.500001, 0., 0.], False)):
        point = transform(frame, voxel)
        report, _ = oracle([point + point], [[float(i == j) for j in range(4)] for i in range(4)], headers)
        require(report["rows"][0]["included_in_native_image"]["before_native"] is wanted, "oblique bound")
        checks.append("oblique native boundary " + repr(voxel))
    equal(stats([5., 0., 12.]), {"RMS_mm": 13. / math.sqrt(3), "median_mm": 5., "max_mm": 12.})
    checks.append("known 5/0/12 RMS median maximum")
    # Counterexamples must expose row omission, wrong direction, de-duplication and bad accounting.
    expected, _ = oracle(pairs, matrix, headers)
    for change in ("drop", "deduplicate", "wrong inverse", "wrong RMS", "wrong inclusion", "lost duplicate"):
        mutated = copy.deepcopy(expected)
        if change == "drop": mutated["rows"].pop()
        elif change == "deduplicate": mutated["point_count"] -= 1
        elif change == "wrong inverse": mutated, _ = oracle(pairs, invert(matrix), headers)
        elif change == "wrong RMS": mutated["rigid"]["RMS_mm"] += .1
        elif change == "wrong inclusion": mutated["rows"][0]["included_in_native_image"]["before_native"] ^= True
        else: mutated["rows"][3]["duplicate_of_first_point_index"]["pair"] = None
        try:
            equal(mutated, expected)
        except AssertionError:
            checks.append("reject saved-output mutation " + change)
        else:
            raise AssertionError("saved-output mutation undetected")
    return checks


def prospective():
    original_open = Path.open
    def guarded(path, *args, **kwargs):
        require(path.suffix != ".tag" and not str(path).endswith((".nii", ".nii.gz", ".ressectionlab")),
                "prospective QA attempted acquired payload access")
        return original_open(path, *args, **kwargs)
    with patch.object(Path, "open", guarded):
        checks = generated()
        refusal_path = ROOT / "artifacts/poc-resect-case3-image-registration-v1/initial-evaluation-refusal.json"
        refusal = record(refusal_path)
        require(refusal["evaluator_sha256"] == OLD_SHA and refusal["cli_exit_code"] == 1
                and refusal["cli_output"] == {"error_type": "ValueError", "status": "refused"}
                and refusal["metadata_only_gate_status"] == "pass"
                and refusal["metrics_published"] is False and refusal["evaluation_output_directory_created"] is False,
                "initial refusal accounting")
        prep_path = ROOT / "artifacts/resect-case3-registration-evaluation-preparation-v2/preparation.json"
        prep = record(prep_path)
        require(prep["evaluator_sha256"] == digest(SOURCE) and prep["specification"] == module(SOURCE)["SPEC"],
                "corrected preparation source/specification")
        for key in ("actual_evaluation_run", "acquired_image_arrays_read", "acquired_landmark_coordinates_read"):
            require(prep[key] is False, "prospective preparation access flags")
        diagnosis_path = ROOT / "artifacts/resect-case3-registration-evaluation-preparation-v2/diagnosis.json"
        diagnosis = record(diagnosis_path)
        exposure = prep["prior_actual_tag_exposure"]
        require(exposure == module(SOURCE)["PRIOR_TAG_EXPOSURE"]
                and exposure["diagnosis_sha256"] == digest(diagnosis_path)
                and exposure["initial_refusal_sha256"] == digest(refusal_path)
                and exposure["actual_full_tag_read"] is True and exposure["accuracy_computed"] is False
                and diagnosis["syntax_diagnosis"]["point_rows"] == 17
                and diagnosis["initial_attempt"]["accuracy_computed"] is False, "prior exposure disclosure")
        pins = {str(path.relative_to(ROOT)): digest(path) for path in (
            SOURCE, Path(__file__), OLD, refusal_path, prep_path,
            diagnosis_path,
            ROOT / "tests/test_evaluate_resect_case3_image_registration.py",
            ROOT / "artifacts/resect-case3-registration-evaluation-preparation-v1/preparation.json",
            ROOT / "artifacts/poc-resect-case3-image-registration-v1/root-evaluation-release.json",
            ROOT / "artifacts/resect-case3-registration-result-independent-v1/independent-fit-qa.json")}
    return {"schema": "resect-case3-evaluation-independent-prospective-v1", "status": "pass",
            "case_id": "RESECT:Case3", "role": "TRAIN", "pins": pins,
            "generated_controls_passed": len(checks), "checks": checks,
            "initial_refusal": {"CLI_invocations": 1, "published_scores": 0,
                "original_receipt_phase": refusal["actual_landmark_read_phase"],
                "independent_generated_witness": "original parser rejects optional empty MINC label",
                "source_syntax_diagnosis": "Pinned owner diagnosis reports 17 finite six-coordinate rows with optional empty label",
                "source_coordinates_read_by_this_audit": False},
            "acquired_image_arrays_read": False, "acquired_tag_coordinates_read": False,
            "actual_score_verification_run": False, "fit_or_optimizer_run": False,
            "development_consumption": "Actual tag syntax was inspected by owner after refusal; no held-out claim.",
            "limits": ["Prospective generated checks do not verify actual scores.",
                       "A separate ROOT release and exact output pins are required for actual scoring QA.",
                       "No clinical or anatomical acceptance is conferred."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prospective", action="store_true", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(args.output.resolve().is_relative_to(DEST.resolve()), "QA output directory")
    result = prospective()
    publish(args.output, result)
    print(json.dumps({"status": result["status"], "generated_controls_passed": result["generated_controls_passed"],
                      "report_sha256": digest(args.output)}))


if __name__ == "__main__":
    main()
