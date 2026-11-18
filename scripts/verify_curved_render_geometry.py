#!/usr/bin/env python3
"""Measure engine display meshes against the generated field's physical definitions."""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from generate_curved_render_case import (
    build_case,
    near_surface,
    branch_point,
    TISSUE,
    TARGET,
    VESSEL,
    MEMBRANE,
)
from resectionlab.surgical_scene import from_case

# Run the production extractor with the original shared material partition.
# Typed geometry buffers live only in the temporary directory.
EXTRACT = """
import fs from 'node:fs';
import {pathToFileURL} from 'node:url';
const [moduleFile,dir,mode]=process.argv.slice(1);
const {prepareSurface}=await import(pathToFileURL(moduleFile));
const {shape,affine,regions}=JSON.parse(fs.readFileSync(dir+'/scene.json'));
const owner=new Uint16Array(Uint8Array.from(fs.readFileSync(dir+'/owner')).buffer);
const protectedRegions=regions.filter(r=>['vessel','membrane'].includes(r.material)).map(r=>r.id);
for(const r of regions){
  const mask=Uint8Array.from(owner,v=>Number(v===r.id));
  const mesh=prepareSurface(mask,shape,affine,r.material,true,mode==='prior'?undefined:{owner,protectedRegions});
  for(const key of ['indices','positions','normals','smoothPositions','smoothNormals']){
    const a=mesh[key]; fs.writeFileSync(dir+'/'+r.id+'-'+key,Buffer.from(a.buffer,a.byteOffset,a.byteLength));
  }
  fs.writeFileSync(dir+'/'+r.id+'-policy',JSON.stringify(mesh.diagnostics));
}
"""


def intersections(tri, xy):
    """Sorted, deduplicated intersections of a physical +z ray with a triangle mesh."""
    x, y = xy
    selected = tri[
        (tri[:, :, 0].min(1) <= x)
        & (tri[:, :, 0].max(1) >= x)
        & (tri[:, :, 1].min(1) <= y)
        & (tri[:, :, 1].max(1) >= y)
    ]
    a, b, c = selected.transpose(1, 0, 2)
    u, v = b - a, c - a
    det = u[:, 0] * v[:, 1] - v[:, 0] * u[:, 1]
    ok = np.abs(det) > 1e-10
    a, u, v, det = a[ok], u[ok], v[ok], det[ok]
    px, py = x - a[:, 0], y - a[:, 1]
    s, t = (px * v[:, 1] - py * v[:, 0]) / det, (u[:, 0] * py - u[:, 1] * px) / det
    inside = (s >= -1e-7) & (t >= -1e-7) & (s + t <= 1.0 + 1e-7)
    return np.unique(np.round((a[:, 2] + s * u[:, 2] + t * v[:, 2])[inside], 7))


def assess(case, truth, scene, directory, mode):
    result, retained = {}, {}
    spacing = scene.spacing
    foreground = np.logical_or.reduce([case.compartments[n] for n in (TARGET, VESSEL, MEMBRANE)])
    other = cKDTree(np.argwhere(foreground) * spacing + case.affine[:3, 3])
    for r in scene.regions:
        name = r["name"]

        def read(key, dtype):
            return np.fromfile(directory / f"{r['id']}-{key}", dtype=dtype).reshape(-1, 3)

        faces = read("indices", "<u4")
        raw = read("positions", "<f4").astype(float) * spacing + scene.affine[:3, 3]
        mesh = (
            raw
            if mode == "raw"
            else read("smoothPositions", "<f4").astype(float) * spacing + scene.affine[:3, 3]
        )
        normals = read("normals" if mode == "raw" else "smoothNormals", "<f4") / spacing
        graph = coo_matrix(
            (np.ones(faces.size, np.int8), (faces.ravel(), np.roll(faces, 1, axis=1).ravel())),
            shape=(len(mesh), len(mesh)),
        )
        components, labels = connected_components(graph, directed=False)
        tri = mesh[faces]
        signed_vol = np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])) / 6.0
        row = dict(
            triangles=len(faces),
            surfaceComponents=int(components),
            componentVolumeMm3=sorted(
                np.abs(np.bincount(labels[faces[:, 0]], weights=signed_vol)).tolist(), reverse=True
            ),
            maximumVertexShiftMm=float(np.linalg.norm(mesh - raw, axis=1).max()),
        )
        assert components == truth["layers"][name]["faceConnectedComponents"], name
        if name == TISSUE:
            # The vertex set is fixed on source geometry. Exclude the equator,
            # other materials and the far side before comparing algorithms.
            x, y, z = raw.T
            sel = (
                ((x / 19.0) ** 2 + (y / 17.0) ** 2 < 0.7)
                & (np.abs(z - near_surface(x, y)) < 1.1)
                & (other.query(raw)[0] > 2.4)
            )
            x, y, z = mesh[sel].T
            eps = 0.001
            dx = (near_surface(x + eps, y) - near_surface(x - eps, y)) / (2 * eps)
            dy = (near_surface(x, y + eps) - near_surface(x, y - eps)) / (2 * eps)
            gradient = np.column_stack((dx, dy, -np.ones(len(dx))))
            projected_error = np.abs(z - near_surface(x, y)) / np.linalg.norm(gradient, axis=1)
            angles = np.degrees(
                np.arccos(
                    np.clip(
                        np.sum(gradient * normals[sel], axis=1)
                        / (np.linalg.norm(gradient, axis=1) * np.linalg.norm(normals[sel], axis=1)),
                        -1.0,
                        1.0,
                    )
                )
            )
            row.update(
                samples=int(sel.sum()),
                meanLocalPlaneErrorMm=float(projected_error.mean()),
                p95LocalPlaneErrorMm=float(np.percentile(projected_error, 95)),
                meanNormalErrorDeg=float(angles.mean()),
                p95NormalErrorDeg=float(np.percentile(angles, 95)),
            )
            profile = [(-5.93, 0.11), (-1.97, 0.11), (1.93, 0.11)]
            zobs = [intersections(tri, xy)[0] for xy in profile]
            row["grooveDepthRelativeToLipsMm"] = float(zobs[1] - (zobs[0] + zobs[2]) / 2)
            exact = [near_surface(*xy) for xy in profile]
            row["analyticGrooveDepthRelativeToLipsMm"] = float(exact[1] - (exact[0] + exact[2]) / 2)
        if name == VESSEL:
            centers = []
            for b in truth["vessel"]["branches"]:
                for xy in np.linspace(truth["vessel"]["junctionRasMm"][:2], b["tipRasMm"][:2], 29):
                    centers.append(branch_point(xy))
            hits = [np.count_nonzero(intersections(tri, (x, y)) < z) % 2 == 1 for x, y, z in centers]
            row.update(centerlineSamples=len(hits), centerlineInside=int(sum(hits)))
        if name == MEMBRANE:
            crossing = intersections(tri, (2.13, -8.14))
            assert len(crossing) == 2
            row["centralThicknessMm"] = float(np.ptp(crossing))
        # Raw coincident vertices identify the same cell edge on adjacent labels.
        ijk = np.rint((raw - scene.affine[:3, 3]) / spacing * 2).astype(int) + 2
        keys = np.ravel_multi_index(ijk.T, np.array(scene.shape) * 2 + 6)
        retained[name] = (keys, mesh)
        result[name] = row
    pairs = []
    for a, (ida, pa) in retained.items():
        for b, (idb, pb) in retained.items():
            if a >= b:
                continue
            shared, i, j = np.intersect1d(ida, idb, return_indices=True)
            if len(shared):
                mismatch = np.linalg.norm(pa[i] - pb[j], axis=1)
                pairs.append(
                    dict(
                        layers=[a, b],
                        matchedVertices=len(shared),
                        maximumSeparationMm=float(mismatch.max()),
                        meanSeparationMm=float(mismatch.mean()),
                    )
                )
    return dict(layers=result, interfaces=pairs)


def verify(output, prior):
    case, truth = build_case()
    scene = from_case(case)
    module = Path(__file__).resolve().parents[1] / "desktop/src/viewer/prepareSurface.ts"
    report = dict(caseHash=case.semantic_hash, scope=truth["scope"], sourceSpacingMm=truth["sourceSpacingMm"])
    with tempfile.TemporaryDirectory() as scratch:
        temp = Path(scratch)
        (temp / "scene.json").write_text(
            json.dumps(dict(shape=scene.shape, affine=scene.affine.tolist(), regions=scene.regions))
        )
        scene.labels.astype("<u2").tofile(temp / "owner")
        for mode, path in [("prior", prior or module), ("conditioned", module)]:
            subprocess.run(
                [
                    "node",
                    "--experimental-strip-types",
                    "--input-type=module",
                    "-e",
                    EXTRACT,
                    str(path.resolve()),
                    str(temp),
                    mode,
                ],
                check=True,
            )
            if mode == "prior":
                report["raw"] = assess(case, truth, scene, temp, "raw")
            report[mode] = assess(case, truth, scene, temp, mode)
            report[mode]["policy"] = json.loads((temp / "1-policy").read_text())["policy"]
    row = report["conditioned"]
    assert all(p["maximumSeparationMm"] < 1e-5 for p in row["interfaces"])
    assert row["layers"][MEMBRANE]["centralThicknessMm"] == 1.0
    assert row["layers"][VESSEL]["centerlineInside"] == row["layers"][VESSEL]["centerlineSamples"]
    assert all(s["maximumVertexShiftMm"] <= min(scene.spacing) * 0.5 for s in row["layers"].values())
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--prior-module", type=Path, help="Optional archived desktop/src/viewer/prepareSurface.ts"
    )
    args = parser.parse_args()
    verify(args.output, args.prior_module)
