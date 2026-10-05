"""Read-only schema/count audit. No fitting, associations, or role assignment.

Run from the repository root with its NumPy environment. Public matrix bytes
are retained under outputs/datasets; matrix-downloads.json pins every file.
"""
from pathlib import Path
import csv
import hashlib
import json
import math
from collections import Counter
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = ROOT / "outputs/datasets/glioma-connectivity-7578326-v1"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit():
    metadata = json.loads((HERE / "baseline-v1.json").read_text())
    file = metadata["files"][0]
    table = HERE / file["name"]
    raw = table.read_bytes()
    assert len(raw) == file["size"]
    assert hashlib.md5(raw).hexdigest() == file["computed_md5"]
    with table.open(newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        headers = reader.fieldnames
    ids = [int(row["ID"]) for row in rows]
    assert len(ids) == len(set(ids)) == 63 and set(ids) == set(range(1, 64))
    missing_tokens = {"", "na", "nan", "null", "none", "n/a", "missing"}
    missing = {key: sum(row[key].strip().lower() in missing_tokens for row in rows)
               for key in headers}
    numeric = {key: [float(row[key]) for row in rows] for key in headers if key != "sex"}
    assert all(math.isfinite(value) for values in numeric.values() for value in values)
    numeric_schema = {key: {"min": min(values), "max": max(values),
                           "unique": len(set(values))} for key, values in numeric.items()}
    labels = {}
    for timepoint in ("T0", "T3"):
        columns = [key for key in headers if key.startswith(timepoint + "_")]
        assert len(columns) == 7
        mismatches = []
        derived_count = 0
        for row in rows:
            supplied = int(row[timepoint + " 2 imp"])
            assert supplied in (0, 1)
            derived = int(sum(float(row[key]) < -1.5 for key in columns) >= 2)
            derived_count += derived
            if derived != supplied:
                mismatches.append(int(row["ID"]))
        labels[timepoint] = {"source_column": timepoint + " 2 imp",
            "supplied_positive": sum(int(row[timepoint + " 2 imp"]) for row in rows),
            "supplied_negative": sum(1 - int(row[timepoint + " 2 imp"]) for row in rows),
            "available_domain_columns": columns,
            "seven_domain_derived_positive": derived_count,
            "seven_domain_derivation": "at least two available domain Z-scores strictly below -1.5; diagnostic only, not replacement labels",
            "mismatch_ids": mismatches}
    matrix_metadata = json.loads((HERE / "matrices-v1.json").read_text())
    download = json.loads((HERE / "matrix-downloads.json").read_text())
    pinned = {item["name"]: item for item in download["files"]}
    visit_ids = {"pre": [], "post": []}
    shape_count, dtype_count = Counter(), Counter()
    checks = Counter()
    minimum, maximum = math.inf, -math.inf
    for item in matrix_metadata["files"]:
        path = DATA / item["name"]
        data = path.read_bytes()
        assert len(data) == item["size"] == pinned[path.name]["bytes"]
        assert hashlib.md5(data).hexdigest() == item["computed_md5"]
        assert sha(path) == pinned[path.name]["sha256"]
        patient, _, _, visit = path.stem.split("_")
        visit_ids[visit].append(int(patient))
        matrix = np.load(path, allow_pickle=False)
        shape_count[str(matrix.shape)] += 1
        dtype_count[str(matrix.dtype)] += 1
        checks["nonfinite_matrices"] += int(not np.isfinite(matrix).all())
        checks["negative_matrices"] += int((matrix < 0).any())
        checks["asymmetric_matrices"] += int(not np.array_equal(matrix, matrix.T))
        checks["nonzero_diagonal_matrices"] += int(np.diag(matrix).any())
        checks["allzero_matrices"] += int(not matrix.any())
        minimum, maximum = min(minimum, float(matrix.min())), max(maximum, float(matrix.max()))
    assert all(sorted(value) == sorted(ids) for value in visit_ids.values())
    assert shape_count == {"(115, 115)": 126} and dtype_count == {"float64": 126}
    assert not any(checks.values())
    return {"schema_version": 1, "purpose": "metadata_schema_label_consistency_audit_only",
        "model_fits": 0, "patient_roles_assigned": False, "patient_id_namespace": "figshare:7578326:v1",
        "rows": len(rows), "columns": len(headers), "headers": headers,
        "id_unique": True, "missing_token_search": sorted(missing_tokens),
        "missing_counts": missing, "numeric_schema": numeric_schema,
        "unrecognized_missing_sentinels": "No source data dictionary supplied; finite values are not proven valid measurements",
        "categorical_counts": {key: dict(Counter(row[key] for row in rows)) for key in
            ("sex", "tumor_grade", "radiotherapy", "chemotherapy", "left_hemi", "right_hemi")},
        "labels": labels,
        "supplied_binary_transition_counts": dict(Counter(row["T0 2 imp"] + "->" + row["T3 2 imp"] for row in rows)),
        "paper_T0_positive": 30, "paper_T3_positive": 34,
        "matrix_count": 126, "matrix_bytes": sum(item["size"] for item in matrix_metadata["files"]),
        "visit_counts": {key: len(value) for key, value in visit_ids.items()},
        "matrix_shapes": dict(shape_count), "matrix_dtypes": dict(dtype_count),
        "matrix_checks": dict(checks), "matrix_value_range": [minimum, maximum],
        "matrix_duplicate_file_hashes": len(pinned) - len({item["sha256"] for item in pinned.values()}),
        "matrix_labels_unit": "115 SLANT ROIs; SIFT2 weighted connectivity, not probability or physical injury volume",
        "source_table_sha256": sha(table), "source_matrix_manifest_sha256": sha(HERE / "matrix-downloads.json"),
        "audit_source_sha256": sha(Path(__file__))}


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2, allow_nan=False))
