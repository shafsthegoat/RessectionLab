#!/usr/bin/env python3
"""Thin HBE specimen mesh/deck preparation; no equilibrium solver or curve reader.

Gmsh is injected explicitly only by a separately released, bounded private-runtime
worker. Importing this module never imports Gmsh, reads curves, or launches FEBio.
Geometry uses the pinned Gmsh transfinite/extrusion API:
https://gmsh.info/doc/texinfo/gmsh.html
"""
from __future__ import annotations

from collections import Counter, defaultdict
import argparse
import gzip
import hashlib
import importlib.util
import itertools
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
import xml.etree.ElementTree as ET

import numpy as np

VERSION = "hbe-specimen-hex8-v1"
PROTOCOL_SHA256 = "ab4385f5ad315d2444ca3aedc87b455db0d36539cea4e2119114d803fba49035"
RESOLUTION_SHA256 = "23b4f5e4d8a0b5ff05fcd45fbca88463a32460793d5b0f27d1a913a5e0a2443c"
HEX_SIGNS = np.array([[-1,-1,-1], [1,-1,-1], [1,1,-1], [-1,1,-1],
                      [-1,-1,1], [1,-1,1], [1,1,1], [-1,1,1]], dtype=float)
# Outward local face cycles in FEBio's documented hex8 ordering.
HEX_FACES = np.array([[0,3,2,1], [4,5,6,7], [0,1,5,4],
                      [1,2,6,5], [2,3,7,6], [3,0,4,7]], dtype=int)
GAUSS = np.array(list(itertools.product((-1/math.sqrt(3), 1/math.sqrt(3)), repeat=3)))
SITES = np.vstack((GAUSS, HEX_SIGNS, np.zeros((1, 3))))
NODE_FIELDS = "x;y;z;ux;uy;uz;Rx;Ry;Rz"
ELEMENT_FIELDS = "sx;sy;sz;sxy;syz;sxz;J;sed"


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_protocol(path):
    path = Path(path)
    if sha256(path) != PROTOCOL_SHA256:
        raise ValueError("Specimen protocol bytes differ from the source-bound declaration")
    return json.loads(path.read_text())


def _level(protocol, N):
    if type(N) is not int:
        raise ValueError("Declared integer mesh level required")
    matches = [row for row in protocol["mesher"]["levels"] if row["N"] == N]
    if len(matches) != 1:
        raise ValueError("Undeclared mesh level")
    return matches[0]


def build_five_block_geometry(gmsh, N, protocol):
    """Create CAD and physical tags, without generating or writing a mesh.

    Five surfaces share the very same inner-edge and connector entities. A single
    extrusion call preserves those interfaces; later topology checks verify the
    actual mesh rather than assuming that CAD adjacency proves conformity.
    """
    _level(protocol, N)
    R, H = (protocol["geometry"][key] for key in ("radius_m", "height_m"))
    gmsh.model.add(f"HBE_01_03_N{N}")
    geo = gmsh.model.geo
    center = geo.addPoint(0, 0, 0)
    signs = ((1,1), (-1,1), (-1,-1), (1,-1))
    inner = [geo.addPoint(x*R/2, y*R/2, 0) for x, y in signs]
    outer = [geo.addPoint(x*R/math.sqrt(2), y*R/math.sqrt(2), 0) for x, y in signs]
    edges = [geo.addLine(inner[i], inner[(i+1)%4]) for i in range(4)]
    radial = [geo.addLine(inner[i], outer[i]) for i in range(4)]
    arcs = [geo.addCircleArc(outer[i], center, outer[(i+1)%4]) for i in range(4)]
    surfaces = [geo.addPlaneSurface([geo.addCurveLoop(edges)])]
    corners = [inner]
    for i in range(4):
        j = (i+1)%4
        surfaces.append(geo.addPlaneSurface([geo.addCurveLoop([radial[i], arcs[i], -radial[j], -edges[i]])]))
        corners.append([inner[i], outer[i], outer[j], inner[j]])
    for curve in edges+arcs:
        geo.mesh.setTransfiniteCurve(curve, N+1)
    for curve in radial:
        geo.mesh.setTransfiniteCurve(curve, N//2+1)
    for surface, points in zip(surfaces, corners):
        geo.mesh.setTransfiniteSurface(surface, cornerTags=points)
        geo.mesh.setRecombine(2, surface)
    extruded = geo.extrude([(2, tag) for tag in surfaces], 0, 0, H,
                           numElements=[N//2], recombine=True)
    geo.synchronize()
    volumes = sorted(tag for dimension, tag in extruded if dimension == 3)
    if len(volumes) != 5:
        raise ValueError("Five shared-interface volumes were not produced")
    boundary = gmsh.model.getBoundary([(3, v) for v in volumes], combined=True,
                                      oriented=False, recursive=False)
    tags = {name: [] for name in ("bottom", "top", "side")}
    for dimension, surface in boundary:
        if dimension != 2:
            raise ValueError("Unexpected volume boundary dimension")
        _, curves = gmsh.model.getAdjacencies(2, surface)
        vertices = set()
        for curve in curves:
            _, points = gmsh.model.getAdjacencies(1, int(curve))
            vertices.update(map(int, points))
        xyz = np.array([gmsh.model.getValue(0, tag, []) for tag in sorted(vertices)])
        if xyz.ndim != 2 or xyz.shape[1] != 3 or not len(xyz):
            raise ValueError("Surface has no unambiguous geometric vertices")
        if np.all(np.abs(xyz[:, 2]) <= H*1e-10):
            name = "bottom"
        elif np.all(np.abs(xyz[:, 2]-H) <= H*1e-10):
            name = "top"
        else:
            name = "side"
        tags[name].append(int(surface))
    if {key: len(value) for key, value in tags.items()} != {"bottom":5, "top":5, "side":4}:
        raise ValueError("Actual cylinder surface classification is incomplete")
    for number, name in enumerate(("bottom", "top", "side"), 1):
        tags[name].sort()
        physical = gmsh.model.addPhysicalGroup(2, tags[name], tag=number)
        gmsh.model.setPhysicalName(2, physical, name)
    physical = gmsh.model.addPhysicalGroup(3, volumes, tag=4)
    gmsh.model.setPhysicalName(3, physical, "specimen")
    return {"volumes": volumes, "surface_tags": tags}


def hex8_permutation(local_coordinates):
    """Determine, then verify, Gmsh -> canonical FEBio local node ordering."""
    local = np.asarray(local_coordinates, dtype=float)
    if local.shape != (8, 3) or not np.isfinite(local).all():
        raise ValueError("Eight finite reference hexahedron vertices required")
    permutation = []
    for sign in HEX_SIGNS:
        matches = np.flatnonzero(np.all(np.abs(local-sign) <= 1e-12, axis=1))
        if len(matches) != 1:
            raise ValueError("Unknown or repeated Gmsh reference-node coordinates")
        permutation.append(int(matches[0]))
    return permutation


def extract_mesh(gmsh, N, protocol, geometry_tags):
    """Read only actual volume nodes/cells and explicitly tagged surface quads."""
    _level(protocol, N)
    types, element_tags, element_nodes = gmsh.model.mesh.getElements(3)
    if list(types) != [5]:
        raise ValueError("Only first-order Gmsh hex8 volume cells are permitted")
    _, dimension, order, count, local, primary = gmsh.model.mesh.getElementProperties(5)
    if (dimension, order, count, primary) != (3, 1, 8, 8):
        raise ValueError("Gmsh element properties do not establish hex8 ordering")
    permutation = hex8_permutation(np.asarray(local).reshape(8, 3))
    gmsh_cells = np.asarray(element_nodes[0], dtype=np.int64).reshape(-1, 8)[:, permutation]
    cell_ids = np.asarray(element_tags[0], dtype=np.int64)
    order_cells = np.argsort(cell_ids)
    gmsh_cells, cell_ids = gmsh_cells[order_cells], cell_ids[order_cells]
    all_ids, all_xyz, _ = gmsh.model.mesh.getNodes()
    all_ids = np.asarray(all_ids, dtype=np.int64)
    all_xyz = np.asarray(all_xyz, dtype=float).reshape(-1, 3)
    if len(set(all_ids.tolist())) != len(all_ids):
        raise ValueError("Duplicate Gmsh node IDs")
    coordinates = dict(zip(all_ids.tolist(), all_xyz.tolist()))
    used = sorted(set(gmsh_cells.flat))
    mapping = {int(old): new for new, old in enumerate(used, 1)}
    try:
        X = [coordinates[old] for old in used]
    except KeyError as error:
        raise ValueError("Volume refers to an absent node") from error
    boundaries = {}
    for name, surfaces in geometry_tags["surface_tags"].items():
        faces = []
        for surface in surfaces:
            kinds, _, rows = gmsh.model.mesh.getElements(2, surface)
            if list(kinds) != [3]:
                raise ValueError("Cylinder boundary must contain only quad4 cells")
            try:
                faces.extend([[mapping[int(i)] for i in row] for row in np.asarray(rows[0]).reshape(-1, 4)])
            except KeyError as error:
                raise ValueError("Boundary node is absent from the volume") from error
        boundaries[name] = {"gmsh_surface_tags": surfaces,
            "node_ids": sorted(set(itertools.chain.from_iterable(faces))), "faces_quad4": faces}
    mesh = {"schema_version": VERSION, "indexing": "one_based", "mesh_N": N,
        "geometry": protocol["geometry"], "rest_nodes_m": X,
        "node_ids": list(range(1, len(used)+1)), "gmsh_node_ids": list(map(int, used)),
        "element_ids": list(range(1, len(cell_ids)+1)), "gmsh_element_ids": cell_ids.tolist(),
        "elements_hex8": [[mapping[int(i)] for i in row] for row in gmsh_cells],
        "boundaries": boundaries, "gmsh_to_febio_local_permutation": permutation,
        "unused_geometric_node_ids": sorted(set(all_ids.tolist())-set(used))}
    # Persist outward cycles from the positively oriented volume, not an assumed
    # Gmsh surface orientation. No coordinates or cell order are repaired.
    topology = boundary_topology(np.asarray(mesh["elements_hex8"]), boundaries)
    for name in boundaries:
        boundaries[name]["faces_quad4"] = topology["outward_faces"][name]
    return mesh


def shape_gradients(local):
    local = np.asarray(local, dtype=float)
    dN = np.empty((8, 3))
    for axis in range(3):
        other = [j for j in range(3) if j != axis]
        dN[:, axis] = HEX_SIGNS[:, axis]/8*np.prod(1+HEX_SIGNS[:, other]*local[other], axis=1)
    return dN


def rest_jacobians(nodes, cells):
    X, cells = np.asarray(nodes, dtype=float), np.asarray(cells)
    if (X.ndim != 2 or X.shape[1] != 3 or not np.isfinite(X).all()
            or cells.ndim != 2 or cells.shape[1] != 8 or not np.issubdtype(cells.dtype, np.integer)
            or not len(cells) or cells.min() < 1 or cells.max() > len(X)):
        raise ValueError("Finite nodes and nonempty one-based hex8 connectivity required")
    if any(len(set(row)) != 8 for row in cells):
        raise ValueError("Repeated vertex within a volume cell")
    maps = np.einsum("eia,sib->esab", X[cells-1], np.array([shape_gradients(p) for p in SITES]))
    determinants = np.linalg.det(maps)
    denominator = np.prod(np.linalg.norm(maps, axis=2), axis=2)
    scaled = np.divide(determinants, denominator, out=np.full_like(determinants, -np.inf), where=denominator>0)
    return determinants, scaled


def boundary_topology(cells, boundaries):
    """Require exactly paired interior faces and tagged exterior faces, one body."""
    cells = np.asarray(cells)
    faces = defaultdict(list)
    for cell_index, cell in enumerate(cells):
        for local in HEX_FACES:
            row = tuple(map(int, cell[local]))
            faces[tuple(sorted(row))].append((cell_index, row))
    if any(len(owners) > 2 for owners in faces.values()):
        raise ValueError("Nonmanifold or duplicated volume faces")
    adjacency = defaultdict(set)
    for owners in faces.values():
        if len(owners) == 2:
            first = owners[0][1]
            reversed_second = owners[1][1][::-1]
            if not any(first == reversed_second[i:]+reversed_second[:i] for i in range(4)):
                raise ValueError("Shared interior faces must have opposite cyclic orientation")
            a, b = [owner[0] for owner in owners]
            adjacency[a].add(b); adjacency[b].add(a)
    seen, pending = set(), [0]
    while pending:
        current = pending.pop()
        if current not in seen:
            seen.add(current); pending.extend(adjacency[current]-seen)
    if len(seen) != len(cells):
        raise ValueError("Volume cells do not share one conforming connected body")
    exterior = {key for key, owners in faces.items() if len(owners) == 1}
    declared = []
    outward = {}
    if set(boundaries) != {"top", "bottom", "side"}:
        raise ValueError("Three explicit cylinder boundary tags required")
    for name, row in boundaries.items():
        keys = [tuple(sorted(face)) for face in row["faces_quad4"]]
        if not keys or any(len(key) != 4 or len(set(key)) != 4 for key in keys):
            raise ValueError("Nonempty quad4 boundary faces required")
        if set(row["node_ids"]) != set(itertools.chain.from_iterable(keys)):
            raise ValueError("Boundary node and face tags disagree")
        if any(key not in exterior for key in keys):
            raise ValueError("Declared face is not an actual exterior volume face")
        declared.extend(keys)
        outward[name] = [list(faces[key][0][1]) for key in sorted(keys)]
    if set(declared) != exterior or any(value != 1 for value in Counter(declared).values()):
        raise ValueError("Unmatched, omitted or multiply tagged exterior faces")
    return {"exterior_face_count": len(exterior), "interior_face_count": len(faces)-len(exterior),
            "connected_cell_count": len(seen), "outward_faces": outward}


def mesh_quality(mesh, protocol):
    """Independent rest-map quadrature and actual polygonal cylinder checks."""
    level = _level(protocol, mesh["mesh_N"])
    X, cells = np.asarray(mesh["rest_nodes_m"], float), np.asarray(mesh["elements_hex8"])
    determinants, scaled = rest_jacobians(X, cells)
    if mesh["node_ids"] != list(range(1, len(X)+1)) or mesh["element_ids"] != list(range(1, len(cells)+1)):
        raise ValueError("Contiguous explicit FEBio IDs required")
    if len(set(mesh["gmsh_node_ids"])) != len(X) or len(set(mesh["gmsh_element_ids"])) != len(cells):
        raise ValueError("Source IDs must remain unique and complete")
    R, H = (protocol["geometry"][key] for key in ("radius_m", "height_m"))
    if len(np.unique(np.rint(X/(R*1e-12)).astype(np.int64), axis=0)) != len(X):
        raise ValueError("Coincident volume nodes conceal unshared interfaces")
    topology = boundary_topology(cells, mesh["boundaries"])
    top, bottom, side = (np.asarray(mesh["boundaries"][name]["node_ids"])-1 for name in ("top", "bottom", "side"))
    tolerance = max(R, H)*1e-10
    if (np.max(np.abs(X[top, 2]-H)) > tolerance or np.max(np.abs(X[bottom, 2])) > tolerance
            or set(top)&set(bottom) or np.max(np.abs(np.linalg.norm(X[side, :2], axis=1)-R)) > tolerance
            or X[:, 2].min() < -tolerance or X[:, 2].max() > H+tolerance
            or np.linalg.norm(X[:, :2], axis=1).max() > R+tolerance):
        raise ValueError("Actual boundary tags or node positions differ from the declared cylinder")
    # Extruded straight side edges form chords. Min distance to each projected
    # quad edge gives the actual maximum radial sag, not the ideal CAD radius.
    minimum_radius = R
    for face in mesh["boundaries"]["side"]["faces_quad4"]:
        xy = X[np.asarray(face)-1, :2]
        for a, b in zip(xy, np.roll(xy, -1, axis=0)):
            edge = b-a; length2 = float(edge@edge)
            fraction = np.clip(-float(a@edge)/length2, 0, 1) if length2 else 0
            minimum_radius = min(minimum_radius, float(np.linalg.norm(a+fraction*edge)))
    volume = float(determinants[:, :8].sum())
    exact = math.pi*R*R*H
    relative_volume_error = abs(volume-exact)/exact
    sag = (R-minimum_radius)/R
    limits = protocol["mesher"]["mesh_quality"]
    checks = {"declared_cell_count": len(cells) == level["nominal_hex8_cells"],
        "positive_rest_jacobians": bool(np.all(determinants > 0)),
        "minimum_scaled_jacobian": bool(np.min(scaled) > limits["minimum_rest_scaled_jacobian_exclusive"])}
    if mesh["mesh_N"] == max(row["N"] for row in protocol["mesher"]["levels"]):
        checks.update(finest_volume=relative_volume_error <= limits["finest_relative_volume_error_max"],
                      finest_boundary_sag=sag <= limits["finest_max_radial_sag_over_R"])
    return {"passed": all(checks.values()), "checks": checks, "node_count": len(X), "hex8_cell_count": len(cells),
        "sampled_site_count_per_cell": len(SITES), "minimum_rest_determinant_m3": float(np.min(determinants)),
        "minimum_rest_scaled_jacobian": float(np.min(scaled)), "integrated_volume_m3": volume,
        "analytic_cylinder_volume_m3": exact, "relative_volume_error": relative_volume_error,
        "maximum_radial_boundary_sag_m": R-minimum_radius, "maximum_radial_boundary_sag_over_R": sag,
        **{key:value for key,value in topology.items() if key != "outward_faces"}}


def prescribed_motion(mesh, branch, steps, protocol):
    definitions = {row["id"]: row for row in protocol["loading"]["branches"]}
    if branch not in definitions or steps not in (protocol["loading"]["base_equal_pseudotime_steps"], protocol["loading"]["fine_repeat_equal_pseudotime_steps"]):
        raise ValueError("Declared branch and fixed load schedule required")
    if type(steps) is not int:
        raise ValueError("Integer load steps required")
    definition = definitions[branch]
    X = np.asarray(mesh["rest_nodes_m"], float)
    top = np.asarray(mesh["boundaries"]["top"]["node_ids"])-1
    times = np.linspace(0., 1., steps+1)
    movement = np.zeros((steps+1, len(top), 3))
    if definition["mode"] == "axial":
        coordinate = times*definition["strain_endpoint"]*protocol["geometry"]["height_m"]
        movement[:, :, 2] = coordinate[:, None]
        units = "m"
    else:
        coordinate = times*definition["angle_endpoint_rad"]
        x, y = X[top, 0], X[top, 1]
        movement[:, :, 0] = (np.cos(coordinate)[:, None]-1)*x-np.sin(coordinate)[:, None]*y
        movement[:, :, 1] = np.sin(coordinate)[:, None]*x+(np.cos(coordinate)[:, None]-1)*y
        units = "rad"
    return {"times": times, "load_coordinate": coordinate, "load_coordinate_units": units,
            "top_node_ids": top+1, "top_displacement_m": movement}


def _child(parent, tag, value=None, **attributes):
    child = ET.SubElement(parent, tag, attributes)
    if value is not None:
        child.text = str(value)
    return child


def specimen_deck(mesh, branch, steps, mu_Pa, protocol):
    """Return deterministic FEBio XML and loading metadata; never run FEBio."""
    if not math.isfinite(mu_Pa) or mu_Pa <= 0:
        raise ValueError("Positive finite SI shear modulus required")
    X, cells = np.asarray(mesh["rest_nodes_m"], float), np.asarray(mesh["elements_hex8"])
    determinants, _ = rest_jacobians(X, cells)
    if np.any(determinants <= 0):
        raise ValueError("Deck requires positive rest Jacobians")
    boundary_topology(cells, mesh["boundaries"])
    loading = prescribed_motion(mesh, branch, steps, protocol)
    root = ET.Element("febio_spec", version="4.0")
    root.append(ET.Comment("HBE_01_03 specimen baseline; SI; no measured curves or patient properties"))
    _child(root, "Module", type="solid")
    control = _child(root, "Control")
    for key, value in {"analysis":"STATIC", "time_steps":steps, "step_size":format(1/steps, ".17g"),
                       "plot_level":"PLOT_NEVER", "plot_zero_state":0,
                       "output_level":"OUTPUT_MAJOR_ITRS", "output_stride":1}.items():
        _child(control, key, value)
    stepper = _child(control, "time_stepper", type="default")
    for key, value in {"max_retries":0, "dtmin":format(1/steps, ".17g"), "dtmax":format(1/steps, ".17g")}.items():
        _child(stepper, key, value)
    solver = _child(control, "solver", type="solid")
    settings = protocol["solver"]
    for key in ("symmetric_stiffness", "max_refs", "dtol", "etol", "rtol"):
        _child(solver, key, settings[key])
    residual = (1e-10*mu_Pa*protocol["geometry"]["radius_m"]**2)**2
    _child(solver, "min_residual", format(residual, ".17g"))
    _child(solver, "qn_method", type="BFGS"); _child(solver, "linear_solver", type="skyline")
    material = _child(_child(root, "Material"), "material", id="1", name="specimen_material", type="Ogden")
    for term in range(1, 7):
        _child(material, f"c{term}", format(2*mu_Pa if term == 1 else 0, ".17g"))
        _child(material, f"m{term}", 2)
    _child(material, "k", format(149*mu_Pa/3, ".17g")); _child(material, "pressure_model", 1)
    mesh_xml = _child(root, "Mesh"); nodes = _child(mesh_xml, "Nodes", name="specimen_nodes")
    for i, point in enumerate(X, 1):
        _child(nodes, "node", ",".join(format(value, ".17g") for value in point), id=str(i))
    elements = _child(mesh_xml, "Elements", type="hex8", name="specimen")
    for i, cell in enumerate(cells, 1):
        _child(elements, "elem", ",".join(map(str, cell)), id=str(i))
    # Preserve identical deck bytes after sorted JSON storage or caller reordering.
    for name in ("bottom", "top", "side"):
        boundary = mesh["boundaries"][name]
        _child(mesh_xml, "NodeSet", ",".join(map(str, boundary["node_ids"])), name=name)
        surface = _child(mesh_xml, "Surface", name=name)
        for i, face in enumerate(boundary["faces_quad4"], 1):
            _child(surface, "quad4", ",".join(map(str, face)), id=str(i))
    for node in loading["top_node_ids"]:
        _child(mesh_xml, "NodeSet", int(node), name=f"top_node_{node}")
    domain = _child(_child(root, "MeshDomains"), "SolidDomain", name="specimen", mat="specimen_material", type="three-field-solid")
    _child(domain, "laugon", 0)
    boundaries = _child(root, "Boundary"); data = _child(root, "LoadData")
    curves = {}
    def prescribe(node_set, axis, values):
        key = tuple(values)
        if key not in curves:
            curves[key] = len(curves)+1
            curve = _child(data, "load_controller", id=str(curves[key]), type="loadcurve")
            _child(curve, "interpolate", "LINEAR")
            points = _child(curve, "points")
            for t, value in zip(loading["times"], values):
                _child(points, "pt", f"{t:.17g},{value:.17g}")
        curve_id = curves[key]
        bc = _child(boundaries, "bc", type="prescribed displacement", node_set=node_set)
        _child(bc, "dof", axis); _child(bc, "value", 1, lc=str(curve_id)); _child(bc, "relative", 0)
    zeros = np.zeros(steps+1)
    for axis in "xyz":
        prescribe("bottom", axis, zeros)
    for column, node in enumerate(loading["top_node_ids"]):
        for axis, name in enumerate("xyz"):
            prescribe(f"top_node_{node}", name, loading["top_displacement_m"][:, column, axis])
    logfile = _child(_child(root, "Output"), "logfile", file="solver.log")
    _child(logfile, "node_data", data=NODE_FIELDS, name="mechanics_nodes_si", file="nodes.log", delim=",")
    _child(logfile, "element_data", data=ELEMENT_FIELDS, name="mechanics_elements_si", file="elements.log", delim=",")
    ET.indent(root, space="  ")
    xml = ET.tostring(root, encoding="unicode", xml_declaration=True)+"\n"
    record = {"branch":branch, "steps":steps, "mu_Pa":mu_Pa, "K_Pa":149*mu_Pa/3,
        "times":loading["times"].tolist(), "load_coordinate":loading["load_coordinate"].tolist(),
        "load_coordinate_units":loading["load_coordinate_units"], "prescribed_dofs":{"bottom":"xyz", "top":"xyz"},
        "node_fields":NODE_FIELDS, "element_fields":ELEMENT_FIELDS, "raw_reaction_convention":"body_on_constraint",
        "reference_output_state":"actual CB_INIT output required; expect steps+1 records including zero",
        "restart_output":"runtime default DUMP_NEVER; solver command must not enable dump output",
        "plot_output":"PLOT_NEVER", "load_controller_count":len(curves),
        "min_residual_N2":residual, "deck_sha256":hashlib.sha256(xml.encode()).hexdigest()}
    return xml, record


def compress_closed_log(path, *, maximum_raw_bytes, maximum_retained_bytes):
    """Losslessly archive a closed log, recording decompressed identity.

    The external supervisor must have reaped the process first. The original is
    removed only after a complete compressed roundtrip/hash check; active files
    are never modified and a failed compression does not delete the evidence.
    """
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > maximum_raw_bytes:
        raise ValueError("Closed regular log exceeds declared raw bound")
    target = path.with_name(path.name+".gz")
    size = path.stat().st_size
    before = sha256(path)
    with path.open("rb") as source, target.open("xb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            for chunk in iter(lambda: source.read(1024*1024), b""):
                compressed.write(chunk)
                if raw.tell() > maximum_retained_bytes:
                    raise ValueError("Compressed output exceeded its retained-byte allowance")
    if target.stat().st_size > maximum_retained_bytes:
        raise ValueError("Compressed output exceeded its retained-byte allowance")
    after = hashlib.sha256(); restored = 0
    with gzip.open(target, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            restored += len(chunk)
            if restored > maximum_raw_bytes:
                raise ValueError("Archive expanded beyond the original bound")
            after.update(chunk)
    if restored != size or after.hexdigest() != before or sha256(path) != before:
        raise ValueError("Closed log changed or lossless archive verification failed")
    receipt = {"raw_name":path.name, "raw_bytes":size, "raw_sha256":before,
               "archive_name":target.name, "archive_bytes":target.stat().st_size,
               "archive_sha256":sha256(target), "decompressed_hash_verified":True}
    path.unlink()
    return receipt


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name+".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+"\n")
    temporary.replace(path)


def verify_runtime(path, expected_sha256, protocol):
    """Bind private module/library/interpreter bytes without importing Gmsh."""
    path = Path(path)
    if not expected_sha256 or sha256(path) != expected_sha256:
        raise ValueError("An exact frozen private-runtime receipt is required")
    value = json.loads(path.read_text())
    if (value.get("status") != "verified_ready_for_separate_meshing_release"
            or value.get("version") != protocol["mesher"]["version"]
            or value.get("wheel", {}).get("sha256") != protocol["mesher"]["sha256"]
            or value.get("source_declaration", {}).get("manifest_sha256") != PROTOCOL_SHA256
            or value.get("probe", {}).get("child", {}).get("status") != "passed"):
        raise ValueError("Runtime identity or its successful probe does not match this protocol")
    prefix = Path(value["private_prefix"]).resolve()
    for name in ("module", "library", "license"):
        row = value[name]
        resolved = Path(row["path"]).resolve()
        if not resolved.is_relative_to(prefix) or sha256(resolved) != row["sha256"]:
            raise ValueError("Private runtime file escaped its prefix or changed: "+name)
    interpreter = Path(value["interpreter"]["resolved_path"]).resolve()
    if Path(sys.executable).resolve() != interpreter or sha256(interpreter) != value["interpreter"]["sha256"]:
        raise ValueError("Mesh worker must use the verified interpreter")
    return value


def read_resolution_protocol(protocol_path, extension_path, extension_sha256):
    """Read the separate fixed N16/N24 declaration without opening any curves.

    The original protocol remains byte-bound; only its returned copy's mesh
    levels change. The experiment orchestrator owns the independent mesh and
    solver releases, aggregate budgets, old results and co-primary comparisons.
    """
    if extension_sha256 != RESOLUTION_SHA256 or sha256(extension_path) != RESOLUTION_SHA256:
        raise ValueError("Exact source-bound axial resolution declaration required")
    extension = json.loads(Path(extension_path).read_text())
    if (extension["schema"] != "hbe-axial-mesh-resolution-v1"
            or extension["original_protocol"]["sha256"] != PROTOCOL_SHA256
            or extension["branches"] != ["compression", "tension"]
            or extension["steps"] != 60 or extension["mu_Pa"] != 1000.0
            or extension["measured_data_access"] is not False or extension["calibration"] is not False):
        raise ValueError("Fixed axial-only numerical extension required")
    levels = extension["mesh_levels"]
    if levels != [{"N":16,"expected_nodes":7209,"nominal_hex8_cells":6144},
                  {"N":24,"expected_nodes":23101,"nominal_hex8_cells":20736}]:
        raise ValueError("Only the declared N16 and N24 meshes are permitted")
    protocol = read_protocol(protocol_path)
    protocol["mesher"]["levels"] = json.loads(json.dumps(levels))
    return protocol, extension


def prepare_resolution_level(N, protocol_path, extension_path, extension_sha256,
                             runtime_path, runtime_sha256, output):
    """Prepare one separately released finer mesh and two original Skyline decks.

    The parent must supervise the process and enforce the extension's aggregate
    preparation budget. This additive worker leaves the legacy worker and CLI
    unchanged. It imports the private mesher only after exact declaration and
    runtime verification, never launches FEBio and never reads measured curves.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    record = {"status":"starting", "mesh_N":N, "gmsh_generation_calls":0,
              "solver_calls":0, "curve_values_opened":False, "decks":{},
              "source_sha256":sha256(__file__), "protocol_sha256":PROTOCOL_SHA256,
              "resolution_declaration_sha256":extension_sha256,
              "runtime_receipt_sha256":runtime_sha256}
    write_json(output/"receipt.json", record)
    gmsh = None
    initialized = False
    inserted_gmsh = False
    try:
        protocol, extension = read_resolution_protocol(protocol_path, extension_path, extension_sha256)
        level = _level(protocol, N)
        runtime = verify_runtime(runtime_path, runtime_sha256, protocol)
        if "gmsh" in sys.modules:
            raise RuntimeError("An ambient Gmsh import cannot enter the released worker")
        spec = importlib.util.spec_from_file_location("gmsh", runtime["module"]["path"])
        if spec is None or spec.loader is None:
            raise RuntimeError("Private Gmsh module could not be resolved")
        gmsh = importlib.util.module_from_spec(spec)
        sys.modules["gmsh"] = gmsh
        inserted_gmsh = True
        spec.loader.exec_module(gmsh)
        if (Path(gmsh.__file__).resolve() != Path(runtime["module"]["path"]).resolve()
                or Path(gmsh.lib._name).resolve() != Path(runtime["library"]["path"]).resolve()
                or gmsh.__version__ != runtime["version"]):
            raise RuntimeError("Loaded Gmsh module/library/version differs from the frozen runtime")
        gmsh.initialize([], readConfigFiles=False, run=False)
        initialized = True
        if gmsh.option.getString("General.Version") != runtime["version"]:
            raise RuntimeError("Initialized Gmsh reports a different version")
        for option in ("General.NumThreads", "Mesh.MaxNumThreads1D", "Mesh.MaxNumThreads2D", "Mesh.MaxNumThreads3D"):
            gmsh.option.setNumber(option, 1)
        gmsh.option.setNumber("Mesh.ElementOrder", 1)
        gmsh.option.setNumber("Mesh.Binary", 1)
        geometry = build_five_block_geometry(gmsh, N, protocol)
        record.update(gmsh_generation_calls=1, status="generating")
        write_json(output/"receipt.json", record)
        generation_started = time.monotonic()
        gmsh.model.mesh.generate(3)
        record["generation_seconds"] = time.monotonic()-generation_started
        actual = extract_mesh(gmsh, N, protocol, geometry)
        write_json(output/"mesh.json", actual)
        gmsh.write(str(output/"specimen.msh"))
        quality = mesh_quality(actual, protocol)
        limits = protocol["mesher"]["mesh_quality"]
        # Both new levels meet the unchanged original finest geometry gates.
        quality["checks"].update(
            declared_node_count=len(actual["node_ids"]) == level["expected_nodes"],
            finest_volume=quality["relative_volume_error"] <= limits["finest_relative_volume_error_max"],
            finest_boundary_sag=quality["maximum_radial_boundary_sag_over_R"] <= limits["finest_max_radial_sag_over_R"])
        quality["passed"] = all(quality["checks"].values())
        record.update(status="quality_checked", quality=quality)
        write_json(output/"receipt.json", record)
        if not quality["passed"]:
            raise ValueError("Actual resolution mesh failed unchanged specimen quality gates")
        for branch in extension["branches"]:
            name = f"{branch}-60-reference"
            directory = output/name
            directory.mkdir()
            xml, deck_record = specimen_deck(actual, branch, 60, 1000.0, protocol)
            (directory/"specimen.feb").write_text(xml)
            deck_record.update(role="reference", mesh_sha256=sha256(output/"mesh.json"),
                               protocol_sha256=PROTOCOL_SHA256, resolution_declaration_sha256=extension_sha256)
            write_json(directory/"loading.json", deck_record)
            record["decks"][name] = {"deck_sha256":deck_record["deck_sha256"], "loading_sha256":sha256(directory/"loading.json")}
            write_json(output/"receipt.json", record)
        gmsh.finalize()
        initialized = False
        verify_runtime(runtime_path, runtime_sha256, protocol)
        if (sha256(__file__) != record["source_sha256"]
                or sha256(protocol_path) != PROTOCOL_SHA256
                or sha256(extension_path) != extension_sha256):
            raise RuntimeError("Mesh/deck source or declaration changed during preparation")
        record.update(status="prepared_not_solved", mesh_sha256=sha256(output/"mesh.json"),
                      native_mesh_sha256=sha256(output/"specimen.msh"), runtime_unchanged=True,
                      elapsed_seconds=time.monotonic()-started)
    except BaseException as error:
        record.update(status="failed", error_type=type(error).__name__, reason=str(error),
                      traceback=traceback.format_exc(), elapsed_seconds=time.monotonic()-started)
        raise
    finally:
        if initialized:
            try:
                gmsh.finalize()
            except BaseException as error:
                record["finalize_error"] = str(error)
        if inserted_gmsh:
            sys.modules.pop("gmsh", None)
        write_json(output/"receipt.json", record)
    return record


def prepare_level(N, protocol, runtime_path, runtime_sha256, output):
    """One separately released worker: one actual mesh, fixed reference decks.

    A parent must supervise this call. It does not launch any solver or open any
    experimental response. Calibration-modulus decks require the later freeze.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    record = {"status":"starting", "mesh_N":N, "gmsh_generation_calls":0,
              "solver_calls":0, "curve_values_opened":False, "decks":{},
              "source_sha256":sha256(__file__), "protocol_sha256":PROTOCOL_SHA256,
              "runtime_receipt_sha256":runtime_sha256}
    write_json(output/"receipt.json", record)
    gmsh = None
    initialized = False
    try:
        _level(protocol, N)
        runtime = verify_runtime(runtime_path, runtime_sha256, protocol)
        if "gmsh" in sys.modules:
            raise RuntimeError("An ambient Gmsh import cannot enter the released worker")
        spec = importlib.util.spec_from_file_location("gmsh", runtime["module"]["path"])
        if spec is None or spec.loader is None:
            raise RuntimeError("Private Gmsh module could not be resolved")
        gmsh = importlib.util.module_from_spec(spec)
        sys.modules["gmsh"] = gmsh
        spec.loader.exec_module(gmsh)
        if (Path(gmsh.__file__).resolve() != Path(runtime["module"]["path"]).resolve()
                or Path(gmsh.lib._name).resolve() != Path(runtime["library"]["path"]).resolve()
                or gmsh.__version__ != runtime["version"]):
            raise RuntimeError("Loaded Gmsh module/library/version differs from the frozen runtime")
        gmsh.initialize([], readConfigFiles=False, run=False)
        initialized = True
        if gmsh.option.getString("General.Version") != runtime["version"]:
            raise RuntimeError("Initialized Gmsh reports a different version")
        for option in ("General.NumThreads", "Mesh.MaxNumThreads1D", "Mesh.MaxNumThreads2D", "Mesh.MaxNumThreads3D"):
            gmsh.option.setNumber(option, 1)
        gmsh.option.setNumber("Mesh.ElementOrder", 1)
        gmsh.option.setNumber("Mesh.Binary", 1)
        geometry = build_five_block_geometry(gmsh, N, protocol)
        record["gmsh_generation_calls"] = 1  # Charge the attempted native call.
        record["status"] = "generating"
        write_json(output/"receipt.json", record)
        generation_started = time.monotonic()
        gmsh.model.mesh.generate(3)
        record["generation_seconds"] = time.monotonic()-generation_started
        actual = extract_mesh(gmsh, N, protocol, geometry)
        write_json(output/"mesh.json", actual)
        gmsh.write(str(output/"specimen.msh"))
        quality = mesh_quality(actual, protocol)
        record.update(status="quality_checked", quality=quality)
        write_json(output/"receipt.json", record)
        if not quality["passed"]:
            raise ValueError("Actual specimen mesh failed prospective quality gates")
        base_steps = protocol["loading"]["base_equal_pseudotime_steps"]
        fine_steps = protocol["loading"]["fine_repeat_equal_pseudotime_steps"]
        mu = protocol["material"]["reference_mu_Pa"]
        schedules = [(branch["id"], base_steps, mu, "reference") for branch in protocol["loading"]["branches"]]
        if N == max(row["N"] for row in protocol["mesher"]["levels"]):
            schedules.extend((branch["id"], fine_steps, mu, "reference") for branch in protocol["loading"]["branches"])
            schedules.extend((branch, fine_steps, 2*mu, "double_mu") for branch in protocol["numerical_checks"]["scale_check"]["branches"])
        for branch, steps, modulus, role in schedules:
            name = f"{branch}-{steps}-{role}"
            directory = output/name
            directory.mkdir()
            xml, deck_record = specimen_deck(actual, branch, steps, modulus, protocol)
            (directory/"specimen.feb").write_text(xml)
            deck_record.update(role=role, mesh_sha256=sha256(output/"mesh.json"), protocol_sha256=PROTOCOL_SHA256)
            write_json(directory/"loading.json", deck_record)
            record["decks"][name] = {"deck_sha256":deck_record["deck_sha256"], "loading_sha256":sha256(directory/"loading.json")}
        gmsh.finalize()
        initialized = False
        verify_runtime(runtime_path, runtime_sha256, protocol)
        if sha256(__file__) != record["source_sha256"]:
            raise RuntimeError("Mesh/deck source changed during preparation")
        record.update(status="prepared_not_solved", mesh_sha256=sha256(output/"mesh.json"),
                      native_mesh_sha256=sha256(output/"specimen.msh"), runtime_unchanged=True,
                      elapsed_seconds=time.monotonic()-started)
    except BaseException as error:
        record.update(status="failed", error_type=type(error).__name__, reason=str(error),
                      traceback=traceback.format_exc(), elapsed_seconds=time.monotonic()-started)
        raise
    finally:
        if initialized:
            try:
                gmsh.finalize()
            except BaseException as error:
                record["finalize_error"] = str(error)
        sys.modules.pop("gmsh", None)
        write_json(output/"receipt.json", record)
    return record


def tree_bytes(directory):
    """Measure retained bytes; symlinks are not permitted in generated evidence."""
    total = 0
    for path in Path(directory).rglob("*"):
        if path.is_symlink():
            raise ValueError("Generated-output tree contains a symlink")
        try:
            if path.is_file():
                total += path.stat().st_size
        except FileNotFoundError:
            continue  # A cooperating worker may have atomically renamed a receipt.
    return total


def supervise_level(command, output, raw_root, protocol, seconds):
    """A narrow process-family watchdog for a single mesher worker; no retries."""
    from febio_runtime import process_group_rss
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    budgets = protocol["budgets"]
    # Keep the user's HOME/CODEX_HOME intact while isolating only this child.
    removed_keys = sorted(key for key in os.environ if key.startswith(("DYLD_", "PYTHON"))
                          or key in {"__PYVENV_LAUNCHER__", "LD_PRELOAD", "LD_LIBRARY_PATH"})
    environment = {key:value for key,value in os.environ.items() if key not in removed_keys}
    environment.update({"PYTHONDONTWRITEBYTECODE":"1", "PYTHONNOUSERSITE":"1", "OMP_NUM_THREADS":"1",
        "OPENBLAS_NUM_THREADS":"1", "MKL_NUM_THREADS":"1", "VECLIB_MAXIMUM_THREADS":"1", "NUMEXPR_NUM_THREADS":"1"})
    record = {"status":"starting", "command":command, "seconds_cap":seconds,
        "rss_cap_bytes":budgets["sampled_process_family_rss_bytes"], "active_output_cap_bytes":budgets["each_active_run_output_bytes"],
        "total_output_cap_bytes":budgets["generated_output_bytes"], "sample_interval_seconds":.1,
        "sampled_peak_process_family_rss_bytes":0, "maximum_active_output_bytes":0, "maximum_total_output_bytes":0,
        "kill_reason":None, "exit_code":None, "automatic_retry":False,
        "removed_environment_override_names":removed_keys,
        "sampling_limit":"Transient between-sample RSS/output peaks may be missed; caps stop observed excess."}
    write_json(output/"supervision.json", record)
    started = time.monotonic(); process = None
    try:
        with (output/"worker.log").open("x") as log:
            process = subprocess.Popen(command, cwd=output, env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            record["pid"] = process.pid
            while True:
                elapsed = time.monotonic()-started
                if elapsed >= seconds:
                    record["kill_reason"] = "wall_cap"; break
                code = process.poll()
                rss, members = process_group_rss(process.pid, timeout_seconds=min(1.,seconds-elapsed))
                active, total = tree_bytes(output), tree_bytes(raw_root)
                record.update(elapsed_seconds=elapsed,
                    sampled_peak_process_family_rss_bytes=max(record["sampled_peak_process_family_rss_bytes"],rss),
                    maximum_active_output_bytes=max(record["maximum_active_output_bytes"],active),
                    maximum_total_output_bytes=max(record["maximum_total_output_bytes"],total))
                if rss > budgets["sampled_process_family_rss_bytes"]:record["kill_reason"] = "process_family_rss_cap"
                elif active > budgets["each_active_run_output_bytes"]:record["kill_reason"] = "active_output_cap"
                elif total > budgets["generated_output_bytes"]:record["kill_reason"] = "total_output_cap"
                write_json(output/"supervision.json", record)
                if record["kill_reason"]:break
                if code is not None:
                    record["exit_code"] = code
                    if members:record["kill_reason"] = "descendants_outlived_worker"
                    break
                if not members and process.poll() is None:
                    raise RuntimeError("Running mesher process-family RSS unavailable")
                time.sleep(min(.1,max(0.,seconds-(time.monotonic()-started))))
    except BaseException as error:
        record.update(kill_reason=record["kill_reason"] or "supervision_error", reason=str(error))
    finally:
        if process is not None:
            try:
                if record["kill_reason"] or process.poll() is None:
                    try:os.killpg(process.pid,signal.SIGKILL)
                    except ProcessLookupError:pass
                record["exit_code"] = process.wait(timeout=2)
            except BaseException as error:
                record.update(kill_reason=record["kill_reason"] or "cleanup_error", cleanup_error=str(error))
        record["elapsed_seconds"] = time.monotonic()-started
        record["status"] = "complete" if record["exit_code"] == 0 and record["kill_reason"] is None and record["elapsed_seconds"] < seconds else "failed"
        write_json(output/"supervision.json",record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol",type=Path,required=True)
    parser.add_argument("--runtime-receipt",type=Path,required=True)
    parser.add_argument("--runtime-receipt-sha256",required=True)
    parser.add_argument("--expected-source-sha256",required=True)
    parser.add_argument("--output",type=Path)
    parser.add_argument("--execute",action="store_true")
    parser.add_argument("--worker-N",type=int,help=argparse.SUPPRESS)
    args = parser.parse_args()
    if sha256(__file__) != args.expected_source_sha256:
        raise ValueError("Mesh/deck source differs from the released source hash")
    protocol = read_protocol(args.protocol)
    verify_runtime(args.runtime_receipt,args.runtime_receipt_sha256,protocol)
    if not args.execute:
        if args.worker_N is not None:raise ValueError("Worker requires explicit execution release")
        print("Protocol/runtime/source hashes verified; no Gmsh import, curve read, mesh or solver execution.")
        return
    if args.output is None:raise ValueError("Explicit new ignored output directory required")
    if args.worker_N is not None:
        prepare_level(args.worker_N,protocol,args.runtime_receipt,args.runtime_receipt_sha256,args.output)
        return
    output = args.output.resolve()
    required_suffix = Path(protocol["storage"]["immutable_raw_output_root"]).parts+("mesh-preparation",)
    if tuple(output.parts[-len(required_suffix):]) != required_suffix:
        raise ValueError("Mesh outputs must use the declared ignored specimen output root")
    output.mkdir(parents=True,exist_ok=False)
    started = time.monotonic()
    receipt = {"status":"starting", "source_sha256":args.expected_source_sha256,
        "protocol_sha256":PROTOCOL_SHA256,"runtime_receipt_sha256":args.runtime_receipt_sha256,
        "solver_calls":0,"curve_values_opened":False,"levels":{str(row["N"]):{"status":"not_executed"} for row in protocol["mesher"]["levels"]}}
    write_json(output/"receipt.json",receipt)
    for level in protocol["mesher"]["levels"]:
        N = level["N"]
        remaining = protocol["budgets"]["gmsh_aggregate_preparation_seconds"]-(time.monotonic()-started)
        if remaining <= 0:
            receipt.update(status="failed",reason="aggregate_preparation_wall_cap")
            write_json(output/"receipt.json",receipt)
            raise RuntimeError(receipt["reason"])
        receipt["levels"][str(N)] = {"status":"starting"}
        write_json(output/"receipt.json",receipt)
        directory = output/f"N{N}"
        command = [sys.executable,str(Path(__file__).resolve()),"--protocol",str(args.protocol.resolve()),
            "--runtime-receipt",str(args.runtime_receipt.resolve()),"--runtime-receipt-sha256",args.runtime_receipt_sha256,
            "--expected-source-sha256",args.expected_source_sha256,"--execute","--worker-N",str(N),"--output",str(directory/"generated")]
        status = supervise_level(command,directory,output.parent,protocol,min(remaining,protocol["budgets"]["each_mesh_generation_seconds"]))
        receipt["levels"][str(N)] = status
        receipt["elapsed_seconds"] = time.monotonic()-started
        if status["status"] != "complete":
            receipt["status"] = "failed"
            write_json(output/"receipt.json",receipt)
            raise RuntimeError("Bounded mesh preparation failed; no replacement or retry")
        write_json(output/"receipt.json",receipt)
    receipt.update(status="prepared_not_solved",elapsed_seconds=time.monotonic()-started)
    write_json(output/"receipt.json",receipt)


if __name__ == "__main__":
    main()
