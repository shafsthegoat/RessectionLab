"""Saved public support pattern only; no source DICOM/header reread."""
import hashlib
import json
from pathlib import Path
import signal
import time
import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE / "actual-public-qc-v1"


def main():
    signal.alarm(30); started = time.monotonic(); cases = []
    for subject in ("ReMIND-013", "ReMIND-037"):
        folder = BASE / (subject + "-crop-mr")
        report = json.loads((folder / "conversion-result.json").read_bytes())
        header = json.loads((BASE / (subject + "-headers/result.json")).read_bytes())
        record = next(s for s in header["series"] if s["kind"] == "cerebrum")
        path = folder / "cerebrum_source_label.npy"
        raw = path.read_bytes(); actual = hashlib.sha256(raw).hexdigest(); del raw
        assert actual == report["artifacts"][path.name]["sha256"]
        support = np.load(path, allow_pickle=False, mmap_mode="r")
        assert support.dtype == np.uint8 and np.isin(support, (0, 1)).all()
        geometry = record["geometry"]; pos = np.asarray(geometry["raw_positions"], float)[geometry["sorted_source_indices"]]
        distances = np.linalg.norm(np.diff(pos, axis=0), axis=1)
        axis_records = []
        for axis in range(3):
            counts = np.sum(support, axis=tuple(a for a in range(3) if a != axis), dtype=np.int64)
            changes = []
            for step in range(1, 9):
                a = [slice(None)]*3; b = list(a); a[axis] = slice(None, -step); b[axis] = slice(step, None)
                first, second = support[tuple(a)], support[tuple(b)]
                changed = int(np.count_nonzero(first != second)); union = int(np.count_nonzero(first | second))
                changes.append({"offset_output_voxels": step, "different_pairs": changed,
                    "pairs_with_either_positive": union, "difference_fraction_of_union": changed/union})
            axis_records.append({"axis": axis, "positive_voxels_per_plane": counts.tolist(),
                "entirely_zero_output_planes": np.flatnonzero(counts == 0).tolist(), "shift_disagreement": changes})
        placements = report["annotations"]["cerebrum"]["placement"]
        cases.append({"subject": subject, "source_shape": geometry["shape_xyz"], "output_shape": list(support.shape),
            "saved_support_sha256": actual, "source_frame_count": len(pos),
            "adjacent_source_position_step_mm_min": float(distances.min()),
            "adjacent_source_position_step_mm_max": float(distances.max()),
            "source_plane_max_residual_mm": geometry["slice_position_max_residual_mm"],
            "target_index_to_source_index": placements["target_index_to_source_index"],
            "source_positive_centres_cropped_from_execution": placements["source_positive_centres_outside_target_grid"],
            "source_positives_from_execution": placements["source_positive_voxels"], "axes": axis_records})
    result = {"schema": "saved-public-support-pattern-diagnostic-v1", "cases": cases,
        "original_source_reads": 0, "source_annotation_accuracy": "unreviewed", "elapsed_seconds": time.monotonic()-started,
        "scope": "Dense saved frame geometry and saved NN support pattern; cannot identify the clinical or source-encoding cause of stripes"}
    output = HERE / "saved-support-pattern-v1.json"
    with output.open("x") as stream: json.dump(result, stream, indent=2, allow_nan=False); stream.write("\n")
    print(json.dumps({"elapsed_seconds": result["elapsed_seconds"], "output": str(output), "cases": [
        {"subject": c["subject"], "source_step_range": [c["adjacent_source_position_step_mm_min"], c["adjacent_source_position_step_mm_max"]],
         "empty_output_planes": [a["entirely_zero_output_planes"] for a in c["axes"]],
         "shift_disagreement_fraction_by_axis": [[r["difference_fraction_of_union"] for r in a["shift_disagreement"]] for a in c["axes"]]} for c in cases]}))


if __name__ == "__main__": main()
