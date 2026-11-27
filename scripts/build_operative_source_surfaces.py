#!/usr/bin/env python3
"""Build source-resolution display surfaces for an operative rehearsal case, with a preview sheet.

  build_operative_source_surfaces.py OPERATIVE_CASE --host HOST_STUDY --output DIR [--data-root ROOT]

Writes DIR/operative-source-surfaces.npz (the asset; schema
resectionlab.operative-source-surfaces.v1), a readable copy of its header and
DIR/operative-source-surfaces-preview.png: shaded off-screen renders of the
skin, the skull and the cortex with vessels from two views, a close view of
the cortex under the craniotomy, and sections of the surfaces on the native T1.
The host's sources are found from the record the operative case carries (an
EDEN2020 v2 host on the data drive). Sources are only read.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from resectionlab.imaging import load_case
from resectionlab import surgical_source_surfaces as sources

COLOURS = {'skin': (.86, .66, .57), 'skullOuter': (.91, .87, .77), 'skullInner': (.85, .80, .70),
           'dura': (.80, .72, .76), 'pial': (.83, .61, .58), 'white': (.90, .87, .78),
           'tumourRim': (.70, .52, .52), 'tumourCore': (.55, .40, .45)}
VESSEL_COLOURS = {0: (.62, .22, .30), 1: (.78, .16, .16), 2: (.25, .30, .65)}


def _basis(toward, up):
    toward = np.asarray(toward, float) / np.linalg.norm(toward)
    up = np.asarray(up, float) - toward * (np.dot(up, toward))
    up /= np.linalg.norm(up)
    return np.cross(up, toward), up, toward


def render(meshes, centre, toward, up, *, extent, size=560, light=None):
    """Orthographic z-buffered render of triangle meshes by barycentric splatting, with smooth normals."""
    right, up, toward = _basis(toward, up)
    pixel = extent / size
    depth = np.full(size * size, -np.inf)
    colour = np.ones((size * size, 3))
    light = toward if light is None else np.asarray(light, float) / np.linalg.norm(light)
    light = light + .35 * up
    light /= np.linalg.norm(light)
    for points, faces, normals, rgb in meshes:
        if not len(faces):
            continue
        rgb = np.asarray(rgb, float)
        per_face_colour = rgb.ndim == 2
        rel = points - centre
        x = rel @ right / pixel + size / 2
        y = size / 2 - rel @ up / pixel
        z = rel @ toward
        for start in range(0, len(faces), 150_000):
            f = faces[start:start + 150_000]
            px, py = x[f], y[f]
            span = np.maximum(np.ptp(px, axis=1), np.ptp(py, axis=1))
            steps = int(np.clip(np.ceil(np.percentile(span, 98) * 1.5), 1, 8))
            grid = [(a, b) for a in range(steps + 1) for b in range(steps + 1 - a)]
            for a, b in grid:
                u, v = a / steps, b / steps
                w = 1. - u - v
                sx = u * px[:, 0] + v * px[:, 1] + w * px[:, 2]
                sy = u * py[:, 0] + v * py[:, 1] + w * py[:, 2]
                sz = u * z[f[:, 0]] + v * z[f[:, 1]] + w * z[f[:, 2]]
                ix, iy = np.floor(sx).astype(int), np.floor(sy).astype(int)
                ok = (ix >= 0) & (ix < size) & (iy >= 0) & (iy < size)
                if not ok.any():
                    continue
                pix = iy[ok] * size + ix[ok]
                zz = sz[ok]
                order = np.lexsort((zz, pix))
                pix, zz = pix[order], zz[order]
                last = np.r_[pix[1:] != pix[:-1], True]
                pix, zz, chosen = pix[last], zz[last], np.flatnonzero(ok)[order][last]
                better = zz > depth[pix]
                if not better.any():
                    continue
                pix, zz, chosen = pix[better], zz[better], chosen[better]
                fc = f[chosen]
                n = u * normals[fc[:, 0]] + v * normals[fc[:, 1]] + w * normals[fc[:, 2]]
                n /= np.maximum(np.linalg.norm(n, axis=1), 1e-9)[:, None]
                facing = n @ toward
                n[facing < 0] *= -1
                diffuse = np.clip(n @ light, 0, 1)
                half = light + toward
                half /= np.linalg.norm(half)
                specular = np.clip(n @ half, 0, 1) ** 40
                base = rgb[start + chosen] if per_face_colour else rgb[None, :]
                colour[pix] = np.clip(base * (.22 + .78 * diffuse[:, None]) + .18 * specular[:, None], 0, 1)
                depth[pix] = zz
    return colour.reshape(size, size, 3)


def _project(points, centre, toward, up, extent, size):
    right, up, toward = _basis(toward, up)
    rel = np.asarray(points, float) - centre
    closed = np.vstack([rel, rel[:1]]) if len(rel) else rel
    return np.column_stack([closed @ right / (extent / size) + size / 2, size / 2 - closed @ up / (extent / size)])


def _surface(asset, name):
    arrays = asset['arrays']
    key = 'surface.' + name
    if key + '.positions' not in arrays:
        return None
    return (arrays[key + '.positions'].astype(float), arrays[key + '.triangles'].astype(np.int64),
            arrays[key + '.normals'].astype(float))


def _vessel_mesh(asset):
    arrays = asset['arrays']
    if 'vessels.wallPositions' not in arrays:
        return None
    kinds = np.r_[arrays['vessels.segmentKind'], 0]
    faces = arrays['vessels.wallTriangles'].astype(np.int64)
    owner = arrays['vessels.wallSegment'].astype(np.int64)
    colour = np.array([VESSEL_COLOURS[int(k)] for k in kinds])[owner[faces[:, 0]]]
    return arrays['vessels.wallPositions'].astype(float), faces, arrays['vessels.wallNormals'].astype(float), colour


def _plane_segments(points, faces, origin, normal, u_axis, v_axis):
    signed = (points - origin) @ normal
    s = signed[faces]
    crossing = (s.min(axis=1) < 0) & (s.max(axis=1) > 0)
    segments = []
    for tri, sv in zip(faces[crossing], s[crossing]):
        hits = []
        for i, j in ((0, 1), (1, 2), (2, 0)):
            if (sv[i] < 0) != (sv[j] < 0):
                t = sv[i] / (sv[i] - sv[j])
                p = points[tri[i]] + t * (points[tri[j]] - points[tri[i]])
                hits.append(((p - origin) @ u_axis, (p - origin) @ v_axis))
        if len(hits) == 2:
            segments.append(hits)
    return segments


def preview(asset, case, host_paths, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from scipy import ndimage

    header = asset['header']
    affine = np.asarray(header['workingGrid']['affine'], float)
    shape = header['workingGrid']['shape']
    lateral, toward_hinge, outward = (affine[:3, a] / np.linalg.norm(affine[:3, a]) for a in range(3))
    centre_ijk = np.array([(shape[0] - 1) / 2, (shape[1] - 1) / 2, 0.])
    declaration = sources.operative_context(case)['declaration']
    entry = np.asarray(declaration['entry']['entryRasMm'], float)
    superior = -toward_hinge
    oblique = outward * .55 + lateral * .45 + superior * .25
    views = (('outward', outward, superior), ('oblique', oblique, superior))
    head_centre = entry - outward * 60.

    skin, outer, pial = _surface(asset, 'skin'), _surface(asset, 'skullOuter'), _surface(asset, 'pial')
    white, rim = _surface(asset, 'white'), _surface(asset, 'tumourRim')
    vessels = _vessel_mesh(asset)
    panels = []
    arrays = asset['arrays']
    for label, name, mesh, colour, body in (('Skin', 'skin', skin, COLOURS['skin'], 'scalp'),
                                            ('Skull outer table', 'skullOuter', outer, COLOURS['skullOuter'], 'bone')):
        # Faces bound to the body (all three vertices) are tinted; the body's outline is drawn over them.
        flags = arrays['surface.' + name + '.body'][mesh[1]].min(axis=1) > 0
        tinted = np.where(flags[:, None], np.asarray(colour) * np.array([.82, .9, 1.05]), np.asarray(colour))
        lines = [arrays[f'outline.{body}.outer.points'].astype(float)]
        if body == 'bone':
            lines.append(arrays['outline.scalp.inner.points'].astype(float))
        for view, toward, up in views:
            image = render([(*mesh, np.clip(tinted, 0, 1))], head_centre, toward, up, extent=200.)
            panels.append((f'{label} · {view} (tint: {body} body)', image,
                           [_project(line, head_centre, toward, up, 200., 560) for line in lines]))
    cortex = [(*pial, COLOURS['pial'])] + ([vessels] if vessels else [])
    for view, toward, up in views:
        panels.append((f'Cortex and vessels · {view}', render(cortex, head_centre, toward, up, extent=200.), []))
    dural = arrays['outline.dura.outer.points'].astype(float)
    panels.append(('Cortex under the craniotomy · 80 mm (dural outline)',
                   render([(*pial, COLOURS['pial'])] + ([vessels] if vessels else []), entry, outward, superior,
                          extent=80., size=640), [_project(dural, entry, outward, superior, 80., 640)]))
    deep = [(*white, COLOURS['white'])] + ([(*rim, COLOURS['tumourRim'])] if rim else [])
    panels.append(('White matter and tumour · outward', render(deep, entry - outward * 25., outward, superior,
                                                               extent=120.), []))

    # Sections on the native T1 through the entry.
    t1, t1_affine, _ = sources._load_nifti(host_paths['t1'], np.float32)
    inverse = np.linalg.inv(t1_affine)
    sections = []
    for title, u_axis, v_axis, normal, extent in (
            ('Section across the hinge axis', lateral, outward, toward_hinge, (95., 115., 30.)),
            ('Section along the hinge axis', -toward_hinge, outward, lateral, (95., 115., 30.)),
            ('Section near the entry (zoom)', lateral, outward, toward_hinge, (30., 30., 15.))):
        half, below, above = extent
        uu, vv = np.meshgrid(np.arange(-half, half, .3), np.arange(-below, above, .3))
        world = entry + uu[..., None] * u_axis + vv[..., None] * v_axis
        ijk = world.reshape(-1, 3) @ inverse[:3, :3].T + inverse[:3, 3]
        image = ndimage.map_coordinates(t1, ijk.T, order=1, mode='constant').reshape(uu.shape)
        lines = {}
        for name in ('skin', 'skullOuter', 'skullInner', 'dura', 'pial', 'white', 'tumourRim'):
            mesh = _surface(asset, name)
            if mesh is not None:
                lines[name] = _plane_segments(mesh[0], mesh[1], entry, normal, u_axis, v_axis)
        sections.append((title, image, (-half, half, -below, above), lines))

    columns = 4
    rows = int(np.ceil(len(panels) / columns)) + 1
    figure = plt.figure(figsize=(columns * 4.6, rows * 4.8))
    for index, (title, image, lines) in enumerate(panels):
        axis = figure.add_subplot(rows, columns, index + 1)
        axis.imshow(image, interpolation='lanczos')
        for line in lines:
            axis.plot(*line.T, color='#1f5fa8', linewidth=.8)
        axis.set_title(title, fontsize=9)
        axis.axis('off')
    for index, (title, image, extent, lines) in enumerate(sections):
        axis = figure.add_subplot(rows, columns, (rows - 1) * columns + index + 1)
        lo, hi = np.percentile(image[image > 0], [1, 99.5]) if (image > 0).any() else (0, 1)
        axis.imshow(image, cmap='gray', vmin=lo, vmax=hi, origin='lower', extent=extent)
        for name, segments in lines.items():
            if segments:
                axis.add_collection(LineCollection(segments, colors=[COLOURS[name]], linewidths=.7))
        axis.set_xlim(extent[0], extent[1])
        axis.set_ylim(extent[2], extent[3])
        axis.set_title(title + ' (mm)', fontsize=9)
        axis.tick_params(labelsize=7)
    axis = figure.add_subplot(rows, columns, rows * columns)
    axis.axis('off')
    text = []
    for row in header['surfaces']:
        dev = row['labelDeviation']
        text.append(f"{row['name']}: {row['vertices']:,} v, {row['triangles']:,} t, edge "
                    f"{row['quality']['meanEdgeMm']} mm, |dev| {dev['meanAbsMm']} mm (p95 {dev['p95Mm']}), "
                    f"{row['provenance']['origin']}")
    if header.get('vessels'):
        coverage = header['vessels']['coverage']
        text.append(f"vessels: {coverage['centrelineMm']} mm centreline, {coverage['centrelineInsideBoxMm']} mm in box, "
                    f"{coverage['graphComponents']} components")
    axis.text(0, 1, '\n'.join(text), va='top', fontsize=7, family='monospace')
    figure.suptitle(f"{header['operativeCaseId']} · source surfaces (skull is a T1 model estimate, dura assumed, "
                    "tumour generated)", fontsize=11)
    figure.tight_layout()
    figure.savefig(path, dpi=100)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('case', type=Path, help='Operative rehearsal case (host-rehearsal.ressectionlab)')
    parser.add_argument('--host', type=Path, required=True, help='Prepared host study the case was built from')
    parser.add_argument('--output', type=Path, required=True, help='Output directory')
    parser.add_argument('--data-root', type=Path, help='Data root holding the host sources (default: from --host)')
    parser.add_argument('--no-preview', action='store_true')
    args = parser.parse_args()
    root = args.data_root or args.host.resolve().parents[2]
    if not (root / '.ressectionlab-data-root.json').exists():
        raise SystemExit(f'Data root marker missing under {root}; is the data drive mounted?')
    started = time.perf_counter()
    case = load_case(args.case)
    paths = sources.eden_host_paths(case, args.host, root)
    asset = sources.build_assets(case, paths, log=lambda message: print(message, flush=True))
    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output / sources.ASSET_NAME
    header = sources.write_asset(asset, destination)
    (args.output / 'operative-source-surfaces.header.json').write_text(json.dumps(header, indent=1))
    built = time.perf_counter() - started
    if not args.no_preview:
        preview(asset, case, paths, args.output / 'operative-source-surfaces-preview.png')
    print(json.dumps({'asset': str(destination), 'bytes': destination.stat().st_size, 'buildSeconds': round(built, 1),
                      'totalSeconds': round(time.perf_counter() - started, 1),
                      'surfaces': {row['name']: [row['vertices'], row['triangles'], row['labelDeviation']['meanAbsMm']]
                                   for row in header['surfaces']}}))


if __name__ == '__main__':
    main()
