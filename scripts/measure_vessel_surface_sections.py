#!/usr/bin/env python3
"""Audit native vessel walls at matched source-centerline sections."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from types import SimpleNamespace

import numpy as np
from vtkmodules.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray
from vtkmodules.vtkCommonCore import reference, vtkPoints
from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData, vtkStaticCellLocator
from vtkmodules.vtkFiltersModeling import vtkSelectEnclosedPoints

from resectionlab.anatomy_bridge import open_anatomy_manifest
from resectionlab.anatomy_geometry import mesh, vascular_cleanup_source
from resectionlab.patient_anatomy import file_hash, world
from resectionlab.surface_fairing import fair_surface, intersection_vertices


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


def _surface_metrics(source, variants, faces):
    """Measure indexed geometry in physical space without shader normals."""
    edge_ids = np.sort(faces[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1)
    _, mapped, counts = np.unique(edge_ids, axis=0, return_inverse=True, return_counts=True)
    sorted_faces = np.argsort(mapped) // 3
    start = (np.cumsum(counts) - counts)[counts == 2]
    first, second = sorted_faces[start], sorted_faces[start + 1]

    def normals(points):
        a, b, c = points[faces].transpose(1, 0, 2)
        n = np.cross(b - a, c - a)
        areas = np.linalg.norm(n, axis=1)
        return n / np.maximum(areas, 1e-20)[:, None], areas

    def angles(a, b):
        return np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", a, b), -1, 1)))

    baseline, _ = normals(source)
    metrics = {}
    for key, points in {"source": source, **variants}.items():
        n, areas = normals(points)
        metrics[key] = {
            "surface_area_mm2": float(areas.sum() / 2),
            "dihedral_degrees_p50_p90_p95": np.percentile(angles(n[first], n[second]), [50, 90, 95]).tolist()
            if len(first) else None,
            "normal_turn_from_source_degrees_p50_p95_p100": np.percentile(angles(n, baseline), [50, 95, 100]).tolist()
            if len(n) else None,
            "displacement_mm_p50_p95_p100": np.percentile(np.linalg.norm(points - source, axis=1), [50, 95, 100]).tolist()
            if len(points) else None,
            "introduced_nonincident_contact_vertices": len(intersection_vertices(source, points, faces)),
        }
    return {
        "common_vertices": len(source), "common_faces": len(faces), "common_edges": len(counts),
        "common_open_edges": int(np.count_nonzero(counts == 1)),
        "common_non_manifold_edges": int(np.count_nonzero(counts > 2)),
        "surfaces": metrics,
    }


def audit(collection, layer, samples, *, cache_root, conditioned=True, compare_previous=False):
    # Isolate derived arrays from the opened study and allow the reproduction
    # runner to clean up the cache without touching source or existing evidence.
    cache = SimpleNamespace(root=cache_root, mask=collection.mask, array=collection.array,
                            path=collection.path, series=collection.series)
    result = mesh(cache, layer, display_source=vascular_cleanup_source(collection, layer), conditioned=conditioned)
    frame = np.asarray(layer["affine"])
    vertices = result["source"]
    faces = result["indices"]
    if not len(faces):
        raise ValueError(f"Layer {layer['id']} has no source surface")
    ras_source = world(frame, vertices)
    ras_display = world(frame, result["positions"])
    variants = {"conditioned": ras_display} if conditioned else {}
    previous_ms = None
    if compare_previous:
        start = perf_counter()
        mask = collection.mask(layer)
        lo = np.maximum(0, np.floor(vertices.min(axis=0)).astype(int) - 2)
        hi = np.minimum(mask.shape, np.ceil(vertices.max(axis=0)).astype(int) + 3)
        cropped = mask[tuple(slice(a, b) for a, b in zip(lo, hi))]
        previous, _ = fair_surface(vertices, faces, frame, layer["kind"],
                                   mask=np.pad(cropped, 1), mask_origin=lo - 1)
        previous_ms = (perf_counter() - start) * 1000
        variants["previous_fairing"] = world(frame, previous)
    raw = locate_surface(ras_source, faces)
    smooth = locate_surface(ras_display, faces) if conditioned else None
    previous = locate_surface(variants["previous_fairing"], faces) if compare_previous else None
    rows = []
    try:
        for radius, branch, station, center, tangent in samples:
            span = max(1.5, 3 * radius)
            before = section_area(raw, center, tangent, span)
            after = section_area(smooth, center, tangent, span) if smooth is not None else before
            prior = section_area(previous, center, tangent, span) if previous is not None else None
            rows.append({
                "branch": branch,
                "station": station,
                "center_ras_mm": center.tolist(),
                "annotation_radius_mm": radius,
                "raw_area_mm2": before,
                "display_area_mm2": after,
                "previous_area_mm2": prior,
                "fraction_change": (after / before - 1) if before and after else None,
            })
    finally:
        raw[1].Complete()
        if smooth is not None:
            smooth[1].Complete()
        if previous is not None:
            previous[1].Complete()
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
        "mesh_preparation_ms": result["preparation_ms"],
        **({
            "previous_fairing_preparation_ms": previous_ms,
            "geometry": _surface_metrics(ras_source, variants, faces),
        } if compare_previous else {}),
        "sections": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path, help="Canonical or linked native review manifest")
    parser.add_argument("branches", type=Path, help="Source-owned native graph JSON")
    parser.add_argument("--layer", action="append", required=True)
    parser.add_argument("--sections", type=int, default=128)
    parser.add_argument("--source-only", action="store_true", help="Compare native sections without running fairing")
    parser.add_argument("--compare-previous", action="store_true",
                        help="Compare the previous fairing and measure physical triangle geometry")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if not 1 <= args.sections <= 512:
        parser.error("Use 1–512 section stations")
    if args.source_only and args.compare_previous:
        parser.error("Choose --source-only or --compare-previous")
    manifest = args.study.resolve()
    manifest_hash = file_hash(manifest)
    collection = open_anatomy_manifest(manifest)
    graph = json.loads(args.branches.read_text())
    source = collection.series(collection.layer(args.layer[0])["source_series"])
    if graph["source_scan_sha256"] != source["original"]["sha256"]:
        raise ValueError("Branch stations do not belong to the selected TOF")
    samples = stations(graph["branches"], args.sections)
    if not samples:
        raise ValueError("No eligible source-centerline sections")
    results = []
    with TemporaryDirectory(prefix=".vessel-mesh-audit-", dir=args.output.parent) as temporary:
        for name in args.layer:
            layer = collection.layer(name)
            if layer["source_series"] != source["id"] or layer["affine"] != source["affine"]:
                raise ValueError("Compared layers must share the native source lattice")
            if args.compare_previous and layer.get("publisher_surface"):
                raise ValueError("Prior fairing comparison requires a reconstructed vessel layer")
            results.append(audit(collection, layer, samples, cache_root=Path(temporary),
                                 conditioned=not args.source_only, compare_previous=args.compare_previous))
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
    if file_hash(manifest) != manifest_hash:
        raise ValueError("The selected study manifest changed during the audit")
    args.output.write_text(json.dumps({
        "method": "32-ray first-boundary polygon perpendicular to an interior source-centerline station",
        "meaning": "Approximate local display section area. Sections with an exterior center or missing rays are unmeasured.",
        "source_scan_sha256": graph["source_scan_sha256"],
        "study_manifest": str(manifest),
        "study_manifest_sha256": manifest_hash,
        "study_snapshot_id": collection.identity(),
        "requested_sections": args.sections,
        "display_surface": "native" if args.source_only else "conditioned",
        "results": results,
        "matched_layer_comparisons": pairs,
    }, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
