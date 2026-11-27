"""Render matched native TOF source and vessel overlays for restricted studies."""

from __future__ import annotations

import argparse
import colorsys
import hashlib
import io
import json
import zipfile
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from matplotlib.lines import Line2D
import nibabel as nib
import numpy as np

BASELINE_COLOR = "#70ddf4"
RETAINED_COLOR = "#58defa"
UNCERTAIN_COLOR = "#ffc65e"
REJECTION_COLOR = "#ff6786"
CHALLENGER_COLOR = "#bda9ff"
PLANES = ("sagittal", "coronal", "axial")


def checked_array(root: Path, record: dict) -> np.ndarray:
    # Read sequentially from the archive before slicing along oblique native axes.
    array = np.load(root / record["path"], allow_pickle=False)
    if tuple(record["shape"]) != array.shape or array.dtype != np.dtype(record["dtype"]):
        raise ValueError(f"Array differs from manifest: {record['path']}")
    return array


def original_tof(root: Path, source: dict) -> np.ndarray:
    path = root / source["original"]["path"]
    with path.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != source["original"]["sha256"]:
            raise ValueError("Original TOF NIfTI checksum differs")
    image = nib.load(path)
    if list(image.shape) != source["data"]["shape"] or not np.allclose(
        image.affine, source["affine"], rtol=0, atol=1e-5
    ):
        raise ValueError("Original TOF NIfTI differs from the native source lattice")
    return image.get_fdata(dtype=np.float32, caching="unchanged")


def native_arrays(path: Path) -> tuple[str, dict, dict, np.ndarray, np.ndarray]:
    manifest = json.loads(path.read_text())
    tof = [series for series in manifest["series"] if series["modality"] == "TOF"]
    if len(tof) != 1:
        raise ValueError(f"Expected one TOF source, found {len(tof)}")
    source = tof[0]
    vessels = [
        layer
        for layer in manifest["layers"]
        if layer.get("source_series") == source["id"]
        and layer.get("provenance", {}).get("model") == "vesselfm"
    ]
    if len(vessels) != 1:
        raise ValueError(f"Expected one native vesselFM layer, found {len(vessels)}")
    vessel = vessels[0]
    if vessel["data"]["shape"] != source["data"]["shape"] or not np.allclose(
        vessel["affine"], source["affine"], rtol=0, atol=1e-5
    ):
        raise ValueError("Native vesselFM layer and TOF source grids differ")
    return (
        manifest["patient_id"],
        source,
        vessel,
        original_tof(path.parent, source),
        checked_array(path.parent, vessel["data"]),
    )


def intensity_window(volume: np.ndarray) -> tuple[float, float]:
    sample = volume[::4, ::4, ::3]
    finite = sample[np.isfinite(sample) & (sample > 0)]
    if not finite.size:
        raise ValueError("No finite positive source TOF signal")
    low, high = np.percentile(finite, [2, 99.7])
    return float(low), float(max(high, low + 1e-5))


def historical_display(path: Path, subject: str, source: dict) -> tuple[dict, np.ndarray]:
    with zipfile.ZipFile(path) as archive:
        manifest = json.loads(archive.read("patient-anatomy.json"))
        source_row = next(row for row in manifest["series"] if row["id"] == source["id"])
        layer = next(row for row in manifest["layers"] if row["id"] == "eden-display-tof-vessels")
        if (
            manifest["patient_id"] != subject
            or source_row["original"]["sha256"] != source["original"]["sha256"]
            or layer["source_series"] != source["id"]
            or not np.array_equal(layer["affine"], source["affine"])
            or layer["data"]["shape"] != source["data"]["shape"]
        ):
            raise ValueError("The earlier combined display differs from the native TOF source")
        payload = archive.read(layer["data"]["path"])
    if hashlib.sha256(payload).hexdigest() != layer["data"]["sha256"]:
        raise ValueError("Historical display mask checksum differs")
    return layer, np.load(io.BytesIO(payload), allow_pickle=False)


def project(volume: np.ndarray, axis: int, bounds: tuple[slice, ...]) -> np.ndarray:
    return np.max(volume[bounds], axis=axis).T


def project_label(
    volume: np.ndarray, axis: int, bounds: tuple[slice, ...], label: int
) -> np.ndarray:
    # Work through native planes to avoid a full-size 3D temporary for a projection.
    plane = volume[bounds]
    result = np.zeros(tuple(n for dim, n in enumerate(plane.shape) if dim != axis), bool)
    chunk_axis = 2 if volume.flags.f_contiguous else 0
    for start in range(0, plane.shape[chunk_axis], 16):
        index = [slice(None)] * 3
        index[chunk_axis] = slice(start, start + 16)
        projected = (plane[tuple(index)] == label).any(axis=axis)
        if chunk_axis == axis:
            result |= projected
        else:
            output_axis = chunk_axis - (chunk_axis > axis)
            target = [slice(None)] * 2
            target[output_axis] = index[chunk_axis]
            result[tuple(target)] |= projected
    return result.T


def show_panel(
    panel: plt.Axes,
    image: np.ndarray,
    overlays: list[tuple[np.ndarray, str]],
    axis: int,
    bounds: tuple[slice, ...],
    spacing: np.ndarray,
    window: tuple[float, float],
    *,
    center: tuple[int, ...] | None = None,
    station_points: np.ndarray | None = None,
    linewidth: float = 0.55,
    markers: bool = False,
) -> None:
    x, y = [dimension for dimension in range(3) if dimension != axis]
    bx, by = bounds[x], bounds[y]
    panel.imshow(
        image,
        cmap="gray",
        vmin=window[0],
        vmax=window[1],
        extent=(bx.start - 0.5, bx.stop - 0.5, by.start - 0.5, by.stop - 0.5),
        origin="lower",
        interpolation="nearest",
        aspect=spacing[y] / spacing[x],
    )
    for mask, color in overlays:
        if mask.any() and not mask.all():
            padded = np.pad(mask, 1)
            panel.contour(
                np.arange(bx.start - 1, bx.stop + 1),
                np.arange(by.start - 1, by.stop + 1),
                padded,
                levels=[0.5],
                colors=[color],
                linewidths=linewidth,
                alpha=0.93,
            )
    panel.set_xlim(bx.start - 0.5, bx.stop - 0.5)
    panel.set_ylim(by.start - 0.5, by.stop - 0.5)
    if center is not None:
        panel.scatter(
            center[x], center[y], s=41, facecolors="none", edgecolors="#ffeab2",
            linewidths=0.65, zorder=5,
        )
    if station_points is not None:
        depth = bounds[axis]
        stations = station_points[
            (station_points[:, axis] >= depth.start - 0.5)
            & (station_points[:, axis] < depth.stop - 0.5)
        ]
        panel.scatter(
            stations[:, x], stations[:, y], s=8, c="#ffae50",
            edgecolors="#332619", linewidths=0.4, zorder=6,
        )
    if markers:
        panel.set_xlabel(f"native voxel {x}", fontsize=8)
        panel.set_ylabel(f"native voxel {y}", fontsize=8)
        panel.tick_params(labelsize=7)
    else:
        panel.set_xticks([])
        panel.set_yticks([])


def methods(
    vessel: dict,
    mask: np.ndarray,
    partition: np.ndarray | None,
    challenger: np.ndarray | None,
    challenger_name: str,
    historical: tuple[dict, np.ndarray] | None,
) -> list[tuple[str, list[tuple[np.ndarray, int, str]]]]:
    columns = [
        ("Original TOF", []),
        ("Native vesselFM", [(mask, vessel["label_value"], BASELINE_COLOR)]),
    ]
    if historical is not None:
        layer, array = historical
        columns.append(("Earlier 5 mm display", [(array, layer["label_value"], BASELINE_COLOR)]))
    if partition is not None:
        columns.extend(
            [
                (
                    "Review visibility\nCyan retained · amber uncertain",
                    [
                        (partition, 1, RETAINED_COLOR),
                        (partition, 2, UNCERTAIN_COLOR),
                    ],
                ),
                ("Rejection candidates", [(partition, 3, REJECTION_COLOR)]),
            ]
        )
    if challenger is not None:
        columns.append((challenger_name, [(challenger, 1, CHALLENGER_COLOR)]))
    return columns


def draw_views(
    output: Path,
    subject: str,
    source: np.ndarray,
    spacing: np.ndarray,
    window: tuple[float, float],
    columns: Sequence[tuple[str, list[tuple[np.ndarray, int, str]]]],
    views: Sequence[tuple[int, tuple[slice, ...], str, tuple[int, ...] | None]],
    *,
    subtitle: str,
    cell_size: float = 3.5,
    coordinates: bool = False,
    station_points: np.ndarray | None = None,
    legend: dict[str, str] | None = None,
) -> None:
    row_heights = []
    for axis, bounds, _, _ in views:
        x, y = [dimension for dimension in range(3) if dimension != axis]
        width = (bounds[x].stop - bounds[x].start) * spacing[x]
        height = (bounds[y].stop - bounds[y].start) * spacing[y]
        row_heights.append(min(1.0, max(0.45, height / width)))
    with plt.rc_context(
        {
            "figure.facecolor": "#10151d",
            "axes.facecolor": "#10151d",
            "axes.edgecolor": "#607483",
            "axes.labelcolor": "#b9c7d0",
            "xtick.color": "#b9c7d0",
            "ytick.color": "#b9c7d0",
            "text.color": "#edf2f5",
        }
    ):
        figure, axes = plt.subplots(
            len(views),
            len(columns),
            squeeze=False,
            figsize=(cell_size * len(columns), cell_size * sum(row_heights) + 0.65),
            gridspec_kw={"height_ratios": row_heights},
            layout="constrained",
        )
        figure.suptitle(f"{subject}\n{subtitle}", fontsize=11)
        for row, (axis, bounds, label, center) in enumerate(views):
            image = project(source, axis, bounds)
            for col, (method, layers) in enumerate(columns):
                panel = axes[row, col]
                overlays = [
                    (project_label(layer, axis, bounds, label_value), color)
                    for layer, label_value, color in layers
                ]
                show_panel(
                    panel, image, overlays, axis, bounds, spacing, window,
                    center=center, markers=coordinates, station_points=station_points,
                )
                if row == 0:
                    panel.set_title(method, fontsize=10, pad=8)
                if col == 0:
                    ylabel = f"\n{panel.get_ylabel()}" if coordinates else ""
                    panel.set_ylabel(f"{label}{ylabel}", fontsize=8)
        if legend:
            figure.legend(
                [Line2D([], [], color=color, linewidth=1.7) for color in legend.values()],
                list(legend),
                loc="outside lower center",
                ncols=min(6, len(legend)),
                fontsize=8,
                frameon=False,
            )
        try:
            figure.savefig(output, dpi=145, bbox_inches="tight", pad_inches=0.13)
        finally:
            plt.close(figure)


def roi_views(
    roi: dict, shape: tuple[int, ...], spacing: np.ndarray
) -> list[tuple[int, tuple[slice, ...], str, tuple[int, ...]]]:
    center = tuple(roi["voxel_ijk"])
    if len(center) != 3 or any(
        type(v) is not int or v < 0 or v >= n for v, n in zip(center, shape)
    ):
        raise ValueError(f"Out-of-bounds native ROI center: {center}")
    radius = np.ceil(float(roi["radius_mm"]) / spacing).astype(int)
    half_slab_mm = float(roi.get("half_slab_mm", 1.3))
    views = []
    for axis, plane in enumerate(PLANES):
        for half_slab in (0, max(1, round(half_slab_mm / spacing[axis]))):
            bounds = tuple(
                slice(max(v - r, 0), min(v + r + 1, n))
                for v, r, n in zip(center, radius, shape)
            )
            depth = slice(
                max(0, center[axis] - half_slab),
                min(shape[axis], center[axis] + half_slab + 1),
            )
            bounds = bounds[:axis] + (depth,) + bounds[axis + 1 :]
            label = f"Native {plane}\n{depth.start}" if not half_slab else (
                f"Native {plane} MIP\n{depth.start}–{depth.stop - 1}"
            )
            views.append((axis, bounds, label, center))
    return views


def roi_voxel_counts(
    roi: dict, spacing: np.ndarray, vessel: dict, baseline: np.ndarray,
    partition: np.ndarray | None, challenger: np.ndarray | None,
    historical: tuple[dict, np.ndarray] | None = None,
) -> dict:
    center = np.asarray(roi["voxel_ijk"])
    radius = float(roi["radius_mm"])
    limits = np.ceil(radius / spacing).astype(int)
    bounds = tuple(
        slice(max(0, v - limit), min(n, v + limit + 1))
        for v, limit, n in zip(center, limits, baseline.shape)
    )
    offsets = np.ogrid[tuple(slice(b.start - v, b.stop - v) for b, v in zip(bounds, center))]
    sphere = sum((offset * step) ** 2 for offset, step in zip(offsets, spacing)) <= radius**2
    positive = baseline[bounds][sphere] == vessel["label_value"]
    counts = {
        "radius_mm": radius,
        "vesselfm_positive": int(np.count_nonzero(positive)),
    }
    if historical is not None:
        layer, array = historical
        counts["historical_display_positive"] = int(
            np.count_nonzero(array[bounds][sphere] == layer["label_value"])
        )
    if partition is not None:
        labels = partition[bounds][sphere]
        counts.update(
            retained=int(np.count_nonzero(labels == 1)),
            uncertain=int(np.count_nonzero(labels == 2)),
            rejection_candidate=int(np.count_nonzero(labels == 3)),
        )
    if challenger is not None:
        new = challenger[bounds][sphere] == 1
        counts.update(
            challenger_positive=int(np.count_nonzero(new)),
            overlap_with_vesselfm=int(np.count_nonzero(new & positive)),
        )
    return counts


def rejection_rois(report: dict, source: dict) -> list[dict]:
    rejected = [row for row in report["analysis"]["components"] if row["state"] == 3]
    if not rejected:
        return []
    contrast = [row for row in rejected if row["source_contrast"] is not None]
    largest = max(rejected, key=lambda row: row["voxels"])
    picks = [("largest-rejection", "Largest proposed exclusion", largest)]
    if contrast:
        strongest = max(contrast, key=lambda row: row["source_contrast"]["foreground_peak"])
        if strongest["id"] != largest["id"]:
            picks.append(("brightest-rejection", "Highest-signal measured exclusion", strongest))
    shape = source["data"]["shape"]
    inverse_affine = np.linalg.inv(np.asarray(source["affine"]))
    rois = []
    for ident, label, row in picks:
        center = np.rint(nib.affines.apply_affine(inverse_affine, row["centroid_ras_mm"]))
        rois.append(
            {
                "id": ident,
                "label": label,
                "voxel_ijk": np.clip(center, 0, np.asarray(shape) - 1).astype(int).tolist(),
                "radius_mm": max(6.0, max(row["axis_extent_mm"]) / 2 + 2.0),
                "half_slab_mm": max(1.3, min(5.0, max(row["axis_extent_mm"]) / 2)),
                "component_id": row["id"],
                "component_voxels": row["voxels"],
                "component_reason": row["reason"],
            }
        )
    return rois


def review(args: argparse.Namespace) -> None:
    subject, source_record, vessel, source, baseline = native_arrays(args.manifest)
    affine = np.asarray(source_record["affine"])
    spacing = nib.affines.voxel_sizes(affine)
    orientation = nib.aff2axcodes(affine)
    historical = (
        historical_display(args.historical_workspace, subject, source_record)
        if args.historical_workspace else None
    )
    partition = (
        np.load(args.partition, allow_pickle=False) if args.partition else None
    )
    if partition is not None and (partition.shape != source.shape or partition.dtype != np.uint8):
        raise ValueError("Expected uint8 cleanup partition on the source TOF grid")
    partition_report = None
    if args.partition_report:
        if partition is None:
            raise ValueError("A cleanup report requires its native partition")
        partition_report = json.loads(args.partition_report.read_text())
        if (
            partition_report["source_tof_sha256"] != source_record["original"]["sha256"]
            or partition_report["source_signal_sha256"] != source_record["data"]["sha256"]
            or partition_report["source_mask_sha256"] != vessel["data"]["sha256"]
        ):
            raise ValueError("Cleanup report source identities differ from the manifest")
        with args.partition.open("rb") as stream:
            partition_hash = hashlib.file_digest(stream, "sha256").hexdigest()
        if partition_report["partition_sha256"] != partition_hash:
            raise ValueError("Cleanup partition does not match its report")
    challenger = None
    challenger_labels = None
    label_names = {}
    label_counts = {}
    if args.challenger:
        image = nib.load(args.challenger)
        if image.shape != source.shape or not np.allclose(image.affine, affine, atol=1e-4, rtol=0):
            raise ValueError("Challenger result differs from the source TOF grid")
        challenger_labels = np.asarray(image.dataobj)
        challenger = challenger_labels > 0
    if args.challenger_labels:
        if challenger_labels is None:
            raise ValueError("A challenger label map requires a source-grid NIfTI")
        label_map = json.loads(args.challenger_labels.read_text())
        label_names = {
            int(value): name
            for name, value in label_map.get("labels", label_map).items()
            if int(value)
        }
        present, counts = np.unique(challenger_labels[challenger], return_counts=True)
        if any(value != int(value) or int(value) not in label_names for value in present):
            raise ValueError("Challenger image contains an undeclared label")
        label_counts = {
            label_names[int(value)]: int(count)
            for value, count in zip(present, counts)
        }
    columns = methods(vessel, baseline, partition, challenger, args.challenger_name, historical)
    window = tuple(args.window) if args.window else intensity_window(source)
    if not np.isfinite(window).all() or window[0] >= window[1]:
        raise ValueError("Expected a finite, increasing source intensity window")
    args.output.mkdir(parents=True, exist_ok=True)
    bounds = tuple(slice(0, n) for n in source.shape)
    overview = [
        (axis, bounds, f"Native {PLANES[axis]} MIP", None)
        for axis in (2, 1, 0)
    ]
    if not args.roi_only:
        draw_views(
            args.output / "native-projections.png", subject, source, spacing, window,
            columns, overview, subtitle="Full-stack maximum intensity projections",
            coordinates=True,
        )
    if args.scout:
        views = []
        for z in np.linspace(24, source.shape[2] - 25, 9, dtype=int):
            depth = slice(max(0, z - 5), min(z + 6, source.shape[2]))
            label = f"Native axial MIP\n{depth.start}–{depth.stop - 1}"
            views.append((2, bounds[:2] + (depth,), label, None))
        for start in range(0, len(views), 3):
            draw_views(
                args.output / f"scout-{start // 3 + 1:02}.png", subject, source, spacing,
                window, columns, views[start : start + 3],
                subtitle="Axial scout · 11 native slices per slab", coordinates=True,
            )
    rois = json.loads(args.rois.read_text()) if args.rois else []
    if partition_report is not None and (not args.roi_only or not rois):
        rois.extend(rejection_rois(partition_report, source_record))
    for roi in rois:
        views = roi_views(roi, source.shape, spacing)
        roi["ras_mm"] = nib.affines.apply_affine(affine, roi["voxel_ijk"]).tolist()
        roi["voxel_counts"] = roi_voxel_counts(
            roi, spacing, vessel, baseline, partition, challenger, historical
        )
        for start in range(0, len(views), 2):
            plane = PLANES[start // 2]
            stations = roi.get("station_points_voxel_ijk")
            roi_columns = columns
            legend = None
            if label_names:
                labels = np.unique(challenger_labels[views[start + 1][1]])
                segments = []
                legend = {}
                for value in labels:
                    if value:
                        color = to_hex(colorsys.hsv_to_rgb((value * 0.618034) % 1, 0.75, 1))
                        legend[label_names[int(value)]] = color
                        segments.append((challenger_labels, int(value), color))
                roi_columns = [*columns[:-1], (args.challenger_name, segments)]
            draw_views(
                args.output / f"{roi['id']}-{plane}.png", subject, source, spacing, window,
                roi_columns, views[start : start + 2], subtitle=f"{roi['label']} · {plane}",
                cell_size=2.65, coordinates=True,
                station_points=np.asarray(stations) if stations is not None else None,
                legend=legend,
            )
    receipt = {
        "patient_id": subject,
        "manifest": str(args.manifest.resolve()),
        "source_tof_original_sha256": source_record["original"]["sha256"],
        "recorded_source_tof_array_sha256": source_record["data"]["sha256"],
        "rendered_source": "Original TOF NIfTI decoded on its native lattice",
        "source_vesselfm_array_sha256": vessel["data"]["sha256"],
        "source_tof_affine_ras_mm": affine.tolist(),
        "source_tof_shape": list(source.shape),
        "native_axis_directions": list(orientation),
        "intensity_window": list(window),
        "partition": str(args.partition.resolve()) if args.partition else None,
        "partition_report": str(args.partition_report.resolve()) if args.partition_report else None,
        "partition_sha256": partition_report["partition_sha256"] if partition_report else None,
        "challenger": str(args.challenger.resolve()) if args.challenger else None,
        "challenger_name": args.challenger_name if args.challenger else None,
        "challenger_labels": label_names or None,
        "challenger_named_foreground_voxels": label_counts or None,
        "historical_display": {
            "workspace": str(args.historical_workspace.resolve()),
            "layer_sha256": historical[0]["data"]["sha256"],
        } if historical else None,
        "legend": {
            "cyan": "baseline or retained vesselFM, depending on column",
            "amber": "uncertain and kept visible",
            "rose": "rejection candidate, provisional visibility only",
            "purple": "challenger foreground in combined projections",
            "named_class_colors": "individually identified in local figure legends",
        },
        "interpretation": (
            "Native slices and thin slabs share a voxel crop and source TOF window "
            "across methods. Projected overlap cannot establish 3D continuity. "
            "Source signal is evidence for review without vascular annotations."
        ),
        "rois": rois,
    }
    (args.output / "source-review.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(args.output / "source-review.json", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--partition", type=Path)
    parser.add_argument(
        "--partition-report", type=Path, help="Add focused checks of proposed exclusions"
    )
    parser.add_argument("--challenger", type=Path, help="Source-grid NIfTI label map or mask")
    parser.add_argument("--challenger-name", default="CereVessSeg")
    parser.add_argument("--challenger-labels", type=Path, help="JSON mapping of names to labels")
    parser.add_argument("--historical-workspace", type=Path, help="Earlier combined TOF workspace")
    parser.add_argument("--rois", type=Path, help="JSON list of pinned native voxel centers")
    parser.add_argument("--roi-only", action="store_true")
    parser.add_argument("--window", type=float, nargs=2)
    parser.add_argument("--scout", action="store_true")
    review(parser.parse_args())


if __name__ == "__main__":
    main()
