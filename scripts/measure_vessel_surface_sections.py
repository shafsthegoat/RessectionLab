#!/usr/bin/env python3
"""Audit native probability surfaces at matched source-centerline sections."""

import argparse
import json
from pathlib import Path

import numpy as np
from vtkmodules.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray
from vtkmodules.vtkCommonCore import reference, vtkPoints
from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData, vtkStaticCellLocator
from vtkmodules.vtkFiltersModeling import vtkSelectEnclosedPoints

from resectionlab.anatomy_geometry import mesh, vascular_cleanup_source
from resectionlab.patient_anatomy import PatientImagingCollection, world


def locate_surface(points, faces):
    vertices = vtkPoints()
    vertices.SetData(numpy_to_vtk(points, deep=True))
    cells = vtkCellArray()
    cells.SetData(
        numpy_to_vtkIdTypeArray(np.arange(0, 3 * (len(faces) + 1), 3, dtype=np.int64), deep=True),
        numpy_to_vtkIdTypeArray(faces.astype(np.int64).ravel(), deep=True),
    )
    surface = vtkPolyData()
    surface.SetPoints(vertices)
    surface.SetPolys(cells)
    locator = vtkStaticCellLocator()
    locator.SetDataSet(surface)
    locator.BuildLocator()
    enclosed = vtkSelectEnclosedPoints()
    enclosed.SetTolerance(1e-6)
    enclosed.Initialize(surface)
    return locator, enclosed


def section_area(surface, center, tangent, max_radius, rays=32):
    locator, enclosed = surface
    if not enclosed.IsInsideSurface(center):
        return None
    tangent = tangent / np.linalg.norm(tangent)
    axis = np.eye(3)[np.argmin(np.abs(tangent))]
    first = np.cross(tangent, axis)
    first /= np.linalg.norm(first)
    second = np.cross(tangent, first)
    angles = np.arange(rays) * (2 * np.pi / rays)
    directions = np.cos(angles)[:, None] * first + np.sin(angles)[:, None] * second
    t = reference(0.0)
    sub_id = reference(0)
    target, local = [0.0] * 3, [0.0] * 3
    distance = []
    for direction in directions:
        if not locator.IntersectWithLine(
                center, center + direction * max_radius, 1e-7, t, target, local, sub_id):
            return None
        distance.append(float(t) * max_radius)
    if min(distance) <= 1e-6 or max(distance) >= max_radius:
        return None
    return float(0.5 * np.sin(2 * np.pi / rays) * np.dot(distance, np.roll(distance, 1)))


def stations(branches, limit):
    """Take interior stations evenly through annotation-radius strata."""
    candidates = []
    for branch in branches:
        points = np.asarray(branch["points_ras_mm"], dtype=float)
        if len(points) < 5 or branch["length_mm"] < 1.5:
            continue
        skip = max(2, len(points) // 5)
        for station in range(skip, len(points) - skip, skip):
            tangent = points[station + 2] - points[station - 2]
            radius = branch["annotation_radii_mm"][station]
            if radius is not None and np.isfinite(radius) and radius > 0 and np.linalg.norm(tangent) > 1e-6:
                candidates.append((radius, branch["id"], station, points[station], tangent))
    candidates.sort(key=lambda row: row[:3])
    indices = np.linspace(0, len(candidates) - 1, min(len(candidates), limit), dtype=int)
    return [candidates[i] for i in indices]


def audit(collection, layer, samples):
    result = mesh(collection, layer, display_source=vascular_cleanup_source(collection, layer))
    frame = np.asarray(layer["affine"])
    raw = locate_surface(world(frame, result["source"]), result["indices"])
    smooth = locate_surface(world(frame, result["positions"]), result["indices"])
    rows = []
    try:
        for radius, branch, station, center, tangent in samples:
            span = max(1.5, 3 * radius)
            before = section_area(raw, center, tangent, span)
            after = section_area(smooth, center, tangent, span)
            rows.append({
                "branch": branch,
                "station": station,
                "center_ras_mm": center.tolist(),
                "annotation_radius_mm": radius,
                "raw_area_mm2": before,
                "display_area_mm2": after,
                "fraction_change": (after / before - 1) if before and after else None,
            })
    finally:
        raw[1].Complete()
        smooth[1].Complete()
    changes = [row["fraction_change"] for row in rows if row["fraction_change"] is not None]
    thin = [row["fraction_change"] for row in rows
            if row["annotation_radius_mm"] <= 0.6 and row["fraction_change"] is not None]
    return {
        "layer": layer["id"],
        "surface_field": str(result.get("surface_field", "unknown")),
        "maximum_surface_displacement_mm": float(result["max_displacement_mm"]),
        "restored_components": int(result.get("restored_components", 0)),
        "matched_sections": len(changes),
        "unmeasured_sections": len(rows) - len(changes),
        "fraction_change_p0_p5_p50_p95_p100": np.percentile(changes, [0, 5, 50, 95, 100]).tolist() if changes else None,
        "thin_sections": len(thin),
        "thin_fraction_change_p0_p5_p50_p95_p100": np.percentile(thin, [0, 5, 50, 95, 100]).tolist() if thin else None,
        "sections": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path, help="Native review patient-anatomy.json")
    parser.add_argument("branches", type=Path, help="Source-owned native graph JSON")
    parser.add_argument("--layer", action="append", required=True)
    parser.add_argument("--sections", type=int, default=128)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if not 1 <= args.sections <= 512:
        parser.error("Use 1–512 section stations")
    collection = PatientImagingCollection(args.study.parent)
    graph = json.loads(args.branches.read_text())
    source = collection.series(collection.layer(args.layer[0])["source_series"])
    if graph["source_scan_sha256"] != source["original"]["sha256"]:
        raise ValueError("Branch stations do not belong to the selected TOF")
    samples = stations(graph["branches"], args.sections)
    if not samples:
        raise ValueError("No eligible source-centerline sections")
    results = []
    for name in args.layer:
        layer = collection.layer(name)
        if layer["source_series"] != source["id"] or layer["affine"] != source["affine"]:
            raise ValueError("Compared layers must share the native source lattice")
        results.append(audit(collection, layer, samples))
    pairs = []
    for comparison in results[1:]:
        original = results[0]
        for field in ("raw_area_mm2", "display_area_mm2"):
            changes = [
                current[field] / first[field] - 1
                for first, current in zip(original["sections"], comparison["sections"])
                if first[field] is not None and current[field] is not None
            ]
            pairs.append({
                "baseline_layer": original["layer"],
                "comparison_layer": comparison["layer"],
                "area_field": field,
                "matched_sections": len(changes),
                "fraction_change_p0_p5_p50_p95_p100": np.percentile(changes, [0, 5, 50, 95, 100]).tolist()
                if changes else None,
            })
    args.output.write_text(json.dumps({
        "method": "32-ray first-boundary polygon perpendicular to an interior source-centerline station",
        "meaning": "Approximate local display section area. Sections with an exterior center or missing rays are unmeasured.",
        "source_scan_sha256": graph["source_scan_sha256"],
        "requested_sections": args.sections,
        "results": results,
        "matched_layer_comparisons": pairs,
    }, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
