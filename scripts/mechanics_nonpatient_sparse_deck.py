"""Deterministic, in-memory FEBio deck preparation for the frozen cube cases.

This module has no native execution path. The numbers are numerical gauges, not
patient material properties. Generated XML is not an execution release.
"""
from __future__ import annotations

from dataclasses import dataclass
import xml.etree.ElementTree as ET

import numpy as np

from scripts import mechanics_nonpatient_sparse_feasibility as design
from scripts import mechanics_patient_constraints as fixture


@dataclass(frozen=True)
class CubeMesh:
    n: int
    nodes_m: np.ndarray
    tet10_indices: np.ndarray  # zero-based; FEBio 12,23,31,14,24,34 edge order
    face_node_ids: dict[str, np.ndarray]  # zero-based; intersections occur on several faces


_FACES = (("xmin", 0, 0), ("xmax", 0, -1),
          ("ymin", 1, 0), ("ymax", 1, -1),
          ("bottom", 2, 0), ("top", 2, -1))


def build_mesh(n: int, *, software_control: bool = False) -> CubeMesh:
    """Build all half-grid nodes and conforming six-tet-per-cube incidence.

    ``n=1`` exists only for a software comparison with the pinned one-cube
    fixture. The experiment itself has exactly the three declared levels.
    """
    declared = design.validate_declaration()
    levels = {row["subdivisions_per_axis"] for row in declared["mesh"]["levels"]}
    if isinstance(n, bool) or not isinstance(n, int) or (n not in levels and not (software_control and n == 1)):
        raise ValueError("Only frozen mesh levels, or explicit one-cube software control, are allowed")
    if tuple(map(tuple, declared["mesh"]["tet10_edges"])) != fixture.EDGES:
        raise ValueError("Pinned FEBio tet10 edge order changed")
    side = declared["mesh"]["side_m"]
    width = 2 * n + 1
    grid = np.indices((width, width, width), dtype=np.int32).reshape(3, -1).T
    nodes = grid.astype(np.float64) * (side / (2 * n))

    def node_id(h: np.ndarray) -> int:
        return int((int(h[0]) * width + int(h[1])) * width + int(h[2]))

    local_vertices = np.asarray(declared["mesh"]["cube_vertex_order"], dtype=np.int32)
    local_tets = declared["mesh"]["local_tet4"]
    elements = np.empty((6 * n**3, 10), dtype=np.int64)
    row = 0
    for i in range(n):
        for j in range(n):
            for k in range(n):
                vertices = local_vertices * 2 + 2 * np.array([i, j, k], dtype=np.int32)
                for local in local_tets:
                    corners = vertices[local]
                    elements[row, :4] = [node_id(h) for h in corners]
                    elements[row, 4:] = [node_id((corners[a] + corners[b]) // 2)
                                          for a, b in fixture.EDGES]
                    row += 1
    faces = {name: np.flatnonzero(grid[:, axis] == (2 * n if endpoint == -1 else 0))
             for name, axis, endpoint in _FACES}
    result = CubeMesh(n, nodes, elements, faces)
    validate_mesh(result)
    return result


def validate_mesh(mesh: CubeMesh) -> dict:
    """Check actual incidence, shared midnodes, boundary and reference volume.

    Reference det(dX/drst) has m^3 units, while the physical volume of one
    straight tetrahedron is det/6. This check includes all eight pinned
    TET10G8 points, four vertices, six edge midpoints and centroid.
    """
    declared = design.validate_declaration()
    n = mesh.n
    if n not in design.LEVELS and n != 1:
        raise ValueError("Undeclared mesh level")
    X, E = mesh.nodes_m, mesh.tet10_indices
    if X.shape != ((2*n+1)**3, 3) or E.shape != (6*n**3, 10):
        raise ValueError("Actual node/element counts do not match the structured recipe")
    if not np.isfinite(X).all() or not np.issubdtype(E.dtype, np.integer):
        raise ValueError("Nonfinite coordinates or noninteger incidence")
    if np.min(E) < 0 or np.max(E) >= len(X) or np.any(np.diff(np.sort(E, axis=1), axis=1) == 0):
        raise ValueError("Out-of-range or repeated element node")
    if len(np.unique(E)) != len(X):
        raise ValueError("Unreferenced half-grid node")
    side = declared["mesh"]["side_m"]
    axis = np.linspace(0, side, 2*n+1)
    expected_grid = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
    if not np.allclose(X, expected_grid, atol=1e-15, rtol=0):
        raise ValueError("Half-grid coordinates or ordering changed")
    for name, dimension, endpoint in _FACES:
        expected = np.flatnonzero(expected_grid[:, dimension] == (side if endpoint == -1 else 0))
        if name not in mesh.face_node_ids or not np.array_equal(mesh.face_node_ids[name], expected):
            raise ValueError("Incomplete or altered boundary tag: " + name)
    if set(mesh.face_node_ids) != {face[0] for face in _FACES}:
        raise ValueError("Unexpected boundary tag")
    if np.intersect1d(mesh.face_node_ids["bottom"], mesh.face_node_ids["top"]).size:
        raise ValueError("Bottom/top overlap")
    edge_to_mid: dict[tuple[int, int], int] = {}
    for tet in E:
        for offset, (a, b) in enumerate(fixture.EDGES, 4):
            pair = tuple(sorted((int(tet[a]), int(tet[b]))))
            midpoint = int(tet[offset])
            if pair in edge_to_mid and edge_to_mid[pair] != midpoint:
                raise ValueError("Nonconforming shared edge midpoint")
            edge_to_mid[pair] = midpoint
            if not np.allclose(X[midpoint], (X[pair[0]] + X[pair[1]])/2, atol=1e-15, rtol=0):
                raise ValueError("Curved or displaced reference midnode")
    points, _ = fixture.gauss_rule()
    extra = np.array([[0.,0.,0.], [1.,0.,0.], [0.,1.,0.], [0.,0.,1.],
                      [.25,.25,.25]] +
                     [((np.eye(4)[a] + np.eye(4)[b])/2)[1:].tolist()
                      for a,b in fixture.EDGES])
    samples = np.vstack((points, extra))
    derivatives = np.stack([fixture.shape(q)[1] for q in samples])
    minimum = np.inf
    maximum = -np.inf
    # Chunking avoids allocating the full [E,19,10,3] Jacobian tensor.
    for start in range(0, len(E), 1024):
        jac = np.einsum("eni,qnj->eqij", X[E[start:start+1024]], derivatives)
        determinants = np.linalg.det(jac)
        if not np.isfinite(determinants).all() or np.min(determinants) <= 0:
            raise ValueError("Nonpositive reference Jacobian")
        minimum = min(minimum, float(np.min(determinants)))
        maximum = max(maximum, float(np.max(determinants)))
    expected_det_m3 = (side/n)**3
    if not np.allclose([minimum, maximum], expected_det_m3, atol=1e-15, rtol=1e-9):
        raise ValueError("Reference Jacobian differs from straight six-tet cube recipe")
    return {"nodes":len(X), "tet10":len(E), "shared_edges":len(edge_to_mid),
            "reference_samples_per_element":len(samples),
            "minimum_reference_det_m3":minimum,
            "maximum_reference_det_m3":maximum,
            "total_reference_volume_m3":len(E)*expected_det_m3/6,
            "boundary_nodes":len(np.unique(np.concatenate(list(mesh.face_node_ids.values()))))}


def _child(parent: ET.Element, tag: str, value: object = None, **attributes: object) -> ET.Element:
    node = ET.SubElement(parent, tag, {k:str(v) for k,v in attributes.items()})
    if value is not None:
        node.text = str(value)
    return node


def build_deck(case_id: str) -> tuple[str, dict]:
    """Prepare exact source-only XML and geometry metrics for one frozen case."""
    declared = design.validate_declaration()
    match = [row for row in declared["cases"] if row["id"] == case_id]
    if len(match) != 1:
        raise ValueError("Undeclared case")
    case = match[0]
    mesh = build_mesh(case["n"])
    metrics = validate_mesh(mesh)
    X, E = mesh.nodes_m, mesh.tet10_indices
    gauge = declared["numerical_gauge"]
    steps = case.get("time_steps", gauge["time_steps"])
    dt = case.get("step_size", gauge["step_size"])
    root = ET.Element("febio_spec", version="4.0")
    root.append(ET.Comment("Idealized numerical cube only; source-only preparation; no patient or fitted tissue law"))
    _child(root,"Module",type="solid")
    control = _child(root,"Control")
    for key,value in {"analysis":"STATIC", "time_steps":steps, "step_size":dt,
                      "plot_level":"PLOT_NEVER", "output_level":"OUTPUT_MAJOR_ITRS",
                      "output_stride":1}.items():
        _child(control,key,value)
    stepper = _child(control,"time_stepper",type="default")
    for key,value in {"max_retries":0,"dtmin":dt,"dtmax":dt}.items():
        _child(stepper,key,value)
    solver = _child(control,"solver",type="solid")
    for key,value in {"symmetric_stiffness":1,"max_refs":20,"dtol":1e-10,
                      "etol":1e-10,"rtol":1e-10,"min_residual":1e-20}.items():
        _child(solver,key,value)
    _child(solver,"qn_method",type="BFGS")
    linear = ET.fromstring(declared["runtime"]["solver_subtree"])
    solver.append(linear)
    material = _child(_child(root,"Material"),"material",id="1",name="numerical_ogden",type="Ogden")
    for i in range(1,7):
        _child(material,f"c{i}",format(2*gauge["mu_Pa"] if i == 1 else 0,".17g"))
        _child(material,f"m{i}",2)
    _child(material,"k",format(gauge["K_Pa"],".17g"))
    _child(material,"pressure_model",1)
    geometry = _child(root,"Mesh")
    nodes = _child(geometry,"Nodes",name="cube_nodes")
    for i,x in enumerate(X,1):
        _child(nodes,"node",",".join(format(v,".17g") for v in x),id=i)
    elements = _child(geometry,"Elements",type="tet10",name="cube")
    for i,e in enumerate(E,1):
        _child(elements,"elem",",".join(str(int(v)+1) for v in e),id=i)
    _child(_child(root,"MeshDomains"),"SolidDomain",name="cube",mat="numerical_ogden",
           elem_type="TET10G8",type="elastic-solid")
    boundary = _child(root,"Boundary")
    if case["load"] == "affine":
        exterior = np.unique(np.concatenate(list(mesh.face_node_ids.values())))
        F = np.asarray(declared["loads"]["affine"]["final_F"])
        displacement = X @ (F-np.eye(3)).T
        for i in exterior:
            name = f"node_{int(i)+1}"
            _child(geometry,"NodeSet",int(i)+1,name=name)
            for axis,label in enumerate("xyz"):
                bc = _child(boundary,"bc",type="prescribed displacement",node_set=name)
                _child(bc,"dof",label)
                _child(bc,"value",format(displacement[i,axis],".17g"),lc="1")
                _child(bc,"relative",0)
    else:
        for name,ids,vector in (("bottom",mesh.face_node_ids["bottom"],
                                declared["loads"]["nonuniform"]["bottom_displacement_m"]),
                               ("top",mesh.face_node_ids["top"],
                                declared["loads"]["nonuniform"]["top_displacement_m"])):
            _child(geometry,"NodeSet",",".join(str(int(i)+1) for i in ids),name=name)
            for axis,label in enumerate("xyz"):
                bc = _child(boundary,"bc",type="prescribed displacement",node_set=name)
                _child(bc,"dof",label)
                _child(bc,"value",format(vector[axis],".17g"),lc="1")
                _child(bc,"relative",0)
    controller = _child(_child(root,"LoadData"),"load_controller",id="1",type="loadcurve")
    _child(controller,"interpolate","LINEAR")
    points = _child(controller,"points")
    _child(points,"pt","0,0")
    _child(points,"pt","1,1")
    log = _child(_child(root,"Output"),"logfile",file=f"{case_id}.log")
    _child(log,"node_data",data=fixture.patch.NODE_FIELDS,name="mechanics_nodes_si",
           file=f"{case_id}.nodes.log",delim=",")
    _child(log,"element_data",data=fixture.patch.ELEMENT_FIELDS,name="mechanics_elements_si",
           file=f"{case_id}.elements.log",delim=",")
    ET.indent(root,space="  ")
    xml = ET.tostring(root,encoding="unicode",xml_declaration=True)+"\n"
    return xml,{"case_id":case_id,"load":case["load"],"n":case["n"],
                "time_steps":steps,"step_size":dt,"geometry":metrics,
                "scope":"In-memory source-only XML; no FEBio parse, solve, or execution release."}
