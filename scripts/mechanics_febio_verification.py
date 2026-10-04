#!/usr/bin/env python3
"""Prepare/check five FEBio 4.13 patch controls; never execute a solver.

SI units throughout. Closed-form homogeneous continuum expectations are an
independent oracle, not a finite-element equilibrium solver or a tissue fit.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import numpy as np

VERSION = "mechanics-febio-patch-v1"
FEBIO_COMMIT = "32ae206ff4881dfb54f62296cd1558e58ed9fcc6"
SIDE_M = 0.01
MU_REF_PA = 1000.0  # Arbitrary numerical scale; no measured tissue parameter.
K_OVER_MU = 149.0 / 3.0
TIMES = (0.25, 0.5, 0.75, 1.0)
NODE_FIELDS = "x;y;z;ux;uy;uz;Rx;Ry;Rz"
ELEMENT_FIELDS = "sx;sy;sz;sxy;syz;sxz;J;sed"
RESIDUAL_FLOOR_N2 = 1e-20
TOLERANCES = {"position_m": 1e-10, "force_N": 1e-8, "moment_Nm": 1e-10,
              "stress_Pa": 1e-5, "energy_density_Pa": 1e-7, "jacobian": 1e-7,
              "relative": 2e-6, "residual_print_relative": 2e-5}
HEX_SIGNS = np.array([[-1,-1,-1], [1,-1,-1], [1,1,-1], [-1,1,-1],
                      [-1,-1,1], [1,-1,1], [1,1,1], [-1,1,1]], dtype=float)


def cases():
    """Fresh fixed controls; users cannot silently introduce a patient fixture."""
    identity = np.eye(3)
    shear = identity.copy(); shear[0, 1] = 0.1
    definitions = (("zero", identity, (0., 0., 0.), MU_REF_PA),
        ("translation", identity, (0.0003, -0.0002, 0.0001), MU_REF_PA),
        ("finite_stretch", np.diag((1.10, 0.95, 1.0)), (0., 0., 0.), MU_REF_PA),
        ("shear", shear, (0., 0., 0.), MU_REF_PA),
        ("shear_double_stiffness", shear, (0., 0., 0.), 2 * MU_REF_PA))
    return {name: {"F": F.tolist(), "translation_m": list(t), "mu_Pa": mu}
            for name, F, t, mu in definitions}


def cube_mesh():
    nodes = np.array([(x, y, z) for z in range(3) for y in range(3) for x in range(3)], float) * (SIDE_M / 2)
    def number(x, y, z): return 1 + x + 3*y + 9*z
    elements = []
    for z, y, x in itertools.product(range(2), repeat=3):
        elements.append([number(x,y,z), number(x+1,y,z), number(x+1,y+1,z), number(x,y+1,z),
                         number(x,y,z+1), number(x+1,y,z+1), number(x+1,y+1,z+1), number(x,y+1,z+1)])
    boundary = [i+1 for i, point in enumerate(nodes) if np.any((point == 0) | (point == SIDE_M))]
    return nodes, np.array(elements, dtype=int), boundary


def homogeneous_response(F, mu_Pa):
    """Exact uncoupled alpha=2 Ogden, pressure_model=1 continuum response."""
    F = np.asarray(F, dtype=float)
    if F.shape != (3, 3) or not np.isfinite(F).all() or not math.isfinite(mu_Pa) or mu_Pa <= 0:
        raise ValueError("Finite 3x3 deformation and positive SI modulus required")
    J = float(np.linalg.det(F))
    if J <= 0:
        raise ValueError("Deformation must preserve positive volume")
    B = F @ F.T
    k = K_OVER_MU * mu_Pa
    W = 0.5 * mu_Pa * (J**(-2/3) * np.trace(B) - 3) + 0.25 * k * (J*J - 1 - 2*math.log(J))
    sigma = mu_Pa * J**(-5/3) * (B - np.trace(B)/3 * np.eye(3)) + 0.5*k*(J - 1/J)*np.eye(3)
    P = J * sigma @ np.linalg.inv(F).T
    return {"J": J, "W_Pa": float(W), "sigma_Pa": sigma, "P_Pa": P}


def expected_state(case_id, time_value=1.0):
    if case_id not in cases() or not math.isfinite(time_value) or not 0 <= time_value <= 1:
        raise ValueError("Declared case and load fraction required")
    case = cases()[case_id]
    F = np.eye(3) + time_value * (np.array(case["F"]) - np.eye(3))
    translation = time_value * np.array(case["translation_m"])
    X, elements, boundary = cube_mesh()
    response = homogeneous_response(F, case["mu_Pa"])
    current = X @ F.T + translation
    reactions = np.zeros_like(X)
    # Integral of constant nominal traction against a bilinear boundary shape.
    # Each 1-D boundary shape integrates h/2 at an edge and h at the midpoint.
    # FEBio Rx reports body-on-constraint, the negative required applied force.
    for i, point in enumerate(X):
        for axis in range(3):
            if point[axis] in (0., SIDE_M):
                normal = -1. if point[axis] == 0 else 1.
                area = math.prod(SIDE_M / (2 if point[j] == SIDE_M/2 else 4)
                                 for j in range(3) if j != axis)
                reactions[i] -= normal * response["P_Pa"][:, axis] * area
    sigma = response["sigma_Pa"]
    element_row = [sigma[0,0], sigma[1,1], sigma[2,2], sigma[0,1], sigma[1,2], sigma[0,2], response["J"], response["W_Pa"]]
    return {"nodes": np.column_stack((current, current-X, reactions)),
            "elements": np.tile(element_row, (len(elements), 1)), "F": F,
            "energy_J": response["W_Pa"] * SIDE_M**3, "response": response}


def _child(parent, tag, value=None, **attributes):
    child = ET.SubElement(parent, tag, attributes)
    if value is not None: child.text = str(value)
    return child


def deck_xml(case_id):
    case = cases()[case_id]
    X, elements, boundary = cube_mesh()
    root = ET.Element("febio_spec", version="4.0")
    root.append(ET.Comment("Numerical patch only: SI m,N,Pa; arbitrary modulus; no patient or HBE values"))
    _child(root, "Module", type="solid")
    control = _child(root, "Control")
    _child(control, "analysis", "STATIC"); _child(control, "time_steps", 4); _child(control, "step_size", .25)
    solver = _child(control, "solver", type="solid")
    for key, value in {"symmetric_stiffness": 1, "max_refs": 15, "dtol": 1e-10,
                       "etol": 1e-10, "rtol": 1e-10, "min_residual": RESIDUAL_FLOOR_N2}.items():
        _child(solver, key, value)
    _child(solver, "qn_method", type="BFGS")
    _child(solver, "linear_solver", type="skyline")
    material = _child(_child(root, "Material"), "material", id="1", name="reference_ogden", type="Ogden")
    for term in range(1, 7):
        _child(material, f"c{term}", 2*case["mu_Pa"] if term == 1 else 0)
        _child(material, f"m{term}", 2)
    _child(material, "k", format(K_OVER_MU*case["mu_Pa"], ".17g")); _child(material, "pressure_model", 1)
    mesh = _child(root, "Mesh"); nodes = _child(mesh, "Nodes", name="cube_nodes")
    for i, point in enumerate(X, 1): _child(nodes, "node", ",".join(format(v, ".17g") for v in point), id=str(i))
    group = _child(mesh, "Elements", type="hex8", name="cube")
    for i, element in enumerate(elements, 1): _child(group, "elem", ",".join(map(str, element)), id=str(i))
    for i in boundary: _child(mesh, "NodeSet", i, name=f"node_{i}")
    domain = _child(_child(root, "MeshDomains"), "SolidDomain", name="cube", mat="reference_ogden", type="three-field-solid")
    _child(domain, "laugon", 0)
    boundaries = _child(root, "Boundary")
    displacement = expected_state(case_id)["nodes"][:, 3:6]
    for i in boundary:
        for component, axis in enumerate("xyz"):
            bc = _child(boundaries, "bc", type="prescribed displacement", node_set=f"node_{i}")
            _child(bc, "dof", axis)
            _child(bc, "value", format(displacement[i-1, component], ".17g"), lc="1")
            _child(bc, "relative", 0)
    curve = _child(_child(root, "LoadData"), "load_controller", id="1", type="loadcurve")
    _child(curve, "interpolate", "LINEAR")
    points = _child(curve, "points"); _child(points, "pt", "0,0"); _child(points, "pt", "1,1")
    log = _child(_child(root, "Output"), "logfile", file=f"{case_id}.log")
    _child(log, "node_data", data=NODE_FIELDS, name="mechanics_nodes_si", file=f"{case_id}.nodes.log", delim=",")
    _child(log, "element_data", data=ELEMENT_FIELDS, name="mechanics_elements_si", file=f"{case_id}.elements.log", delim=",")
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode", xml_declaration=True) + "\n"


def reconstructed_jacobians(current):
    """Measure returned trilinear elements at all eight Gauss points and corners.

    This kinematic check uses returned positions, not the element-averaged J log.
    It does not establish positivity everywhere in an arbitrary warped hex.
    """
    X, elements, _ = cube_mesh()
    current = np.asarray(current, float)
    if current.shape != X.shape or not np.isfinite(current).all(): raise ValueError("Complete finite positions required")
    gauss = list(itertools.product((-1/math.sqrt(3), 1/math.sqrt(3)), repeat=3))
    samples = gauss + list(map(tuple, HEX_SIGNS)) + [(0., 0., 0.)]
    jacobians, gradients = [], []
    for element in elements:
        initial = X[element-1]; moved = current[element-1]
        for local in samples:
            local = np.array(local)
            dN = np.empty((8, 3))
            for axis in range(3):
                others = [j for j in range(3) if j != axis]
                dN[:, axis] = HEX_SIGNS[:, axis] / 8 * np.prod(1 + HEX_SIGNS[:, others] * local[others], axis=1)
            rest_map = initial.T @ dN
            if np.linalg.det(rest_map) <= 0: raise ValueError("Reference element orientation invalid")
            F = moved.T @ dN @ np.linalg.inv(rest_map)
            gradients.append(F); jacobians.append(float(np.linalg.det(F)))
    return np.array(jacobians).reshape(8, 17), np.array(gradients).reshape(8, 17, 3, 3)


def parse_data_log(text, *, field_count, record_name, item_count):
    """Strict parser for pinned DataRecord default 12-digit comma-separated output."""
    if len(text.encode()) > 2*1024**2: raise ValueError("Data log exceeds tiny-control bound")
    records = []; record = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line: continue
        if line.startswith("*Step"):
            if record is not None: records.append(record)
            match = re.fullmatch(r"\*Step\s*=\s*(\d+)", line)
            if not match: raise ValueError("Malformed step header")
            record = {"step": int(match[1]), "time": None, "name": None, "rows": {}}
        elif line.startswith("*Time"):
            if record is None or record["time"] is not None: raise ValueError("Missing/duplicate time header")
            record["time"] = float(line.split("=", 1)[1])
        elif line.startswith("*Data"):
            if record is None or record["name"] is not None: raise ValueError("Missing/duplicate data header")
            record["name"] = line.split("=", 1)[1].strip()
        else:
            if record is None or record["time"] is None or record["name"] != record_name:
                raise ValueError("Unbound or unexpected data row")
            fields = line.split(",")
            if len(fields) != field_count+1 or not fields[0].isdigit(): raise ValueError("Malformed data columns")
            index = int(fields[0]); values = [float(v) for v in fields[1:]]
            if index in record["rows"] or not 1 <= index <= item_count or not all(map(math.isfinite, values)):
                raise ValueError("Duplicate, out-of-range or nonfinite output")
            record["rows"][index] = values
    if record is not None: records.append(record)
    times = [r["time"] for r in records]
    if times not in [list(TIMES), [0.]+list(TIMES)]: raise ValueError("Missing, extra or incomplete load states")
    if [r["step"] for r in records] not in [list(range(1,5)), list(range(5))]: raise ValueError("Unexpected output step sequence")
    for record in records:
        if record["name"] != record_name or len(record["rows"]) != item_count: raise ValueError("Incomplete data record")
        record["values"] = np.array([record["rows"][i] for i in range(1, item_count+1)])
        del record["rows"]
    return records


def check_solver_log(text):
    """Keep actual last residual norms for each physical load state."""
    if len(text.encode()) > 4*1024**2: raise ValueError("Solver log exceeds tiny-control bound")
    banners = [re.sub(r"[^A-Za-z]", "", line).upper() for line in text.splitlines()]
    if "NORMALTERMINATION" not in banners: raise ValueError("No normal termination evidence")
    if any(value in banners for value in ("ERRORTERMINATION", "ABNORMALTERMINATION")) or re.search(r"\bnan\b|negative jacobian", text, flags=re.I):
        raise ValueError("Solver failure evidence")
    last = {}; current_time = None
    for line in text.splitlines():
        match = re.search(r"Nonlinear solution status:\s*time=\s*(\S+)", line)
        if match: current_time = float(match[1])
        fields = line.split()
        if fields and fields[0] == "residual":
            if current_time is None or len(fields) != 4: raise ValueError("Unbound residual evidence")
            values = [float(value) for value in fields[1:]]
            if not all(math.isfinite(v) and v >= 0 for v in values): raise ValueError("Invalid residual norm")
            last[current_time] = values
    if set(last) != set(TIMES): raise ValueError("Missing final residual evidence")
    passed = all(v[1] <= max(v[2], RESIDUAL_FLOOR_N2) * (1+TOLERANCES["residual_print_relative"]) for v in last.values())
    return {"passed": passed, "last_squared_force_residual_norms_N2": {str(t): last[t] for t in TIMES},
            "criterion": "current <= max(FEBio required relative norm, declared absolute floor), allowing only log-print rounding"}


def check_outputs(case_id, node_text, element_text, solver_text):
    nodes = parse_data_log(node_text, field_count=9, record_name="mechanics_nodes_si", item_count=27)
    elements = parse_data_log(element_text, field_count=8, record_name="mechanics_elements_si", item_count=8)
    if [r["time"] for r in nodes] != [r["time"] for r in elements]: raise ValueError("Unsynchronized output records")
    solver = check_solver_log(solver_text)
    failures = [] if solver["passed"] else ["solver_residual"]
    states = []
    X, _, boundary = cube_mesh()
    def compare(actual, expected, absolute):
        return bool(np.all(np.abs(actual-expected) <= absolute + TOLERANCES["relative"] * np.abs(expected)))
    for node_record, element_record in zip(nodes, elements):
        t = node_record["time"]; actual = node_record["values"]; logged = element_record["values"]
        expected = expected_state(case_id, t)
        jacobians, gradients = reconstructed_jacobians(actual[:, :3])
        gauss_J = jacobians[:, :8]
        checks = {
            "positions_m": bool(np.max(np.abs(actual[:, :3]-expected["nodes"][:, :3])) <= TOLERANCES["position_m"]),
            "displacements_m": bool(np.max(np.abs(actual[:, 3:6]-(actual[:, :3]-X))) <= TOLERANCES["position_m"]),
            "signed_reactions_N": compare(actual[:, 6:9], expected["nodes"][:, 6:9], TOLERANCES["force_N"]),
            "stress_Pa": compare(logged[:, :6], expected["elements"][:, :6], TOLERANCES["stress_Pa"]),
            "energy_density_Pa": compare(logged[:, 7], expected["elements"][:, 7], TOLERANCES["energy_density_Pa"]),
            "positive_actual_sampled_jacobians": bool(np.all(jacobians > 0)),
            "actual_gradient": bool(np.max(np.abs(gradients-expected["F"])) <= TOLERANCES["jacobian"]),
            "logged_J_matches_actual_Gauss_average": bool(np.max(np.abs(logged[:, 6]-gauss_J.mean(axis=1))) <= TOLERANCES["jacobian"]),
            "force_balance_N": bool(np.linalg.norm(actual[:, 6:9].sum(axis=0)) <= TOLERANCES["force_N"]),
            "current_coordinate_moment_balance_Nm": bool(np.linalg.norm(np.cross(actual[:, :3], actual[:, 6:9]).sum(axis=0)) <= TOLERANCES["moment_Nm"]),
            "free_center_residual_N": bool(np.linalg.norm(actual[13, 6:9]) <= TOLERANCES["force_N"]),
        }
        failures.extend(f"t={t}:{name}" for name, passed in checks.items() if not passed)
        states.append({"time": t, "checks": checks, "minimum_actual_sampled_J": float(jacobians.min()),
                       "total_logged_energy_J": float(logged[:, 7].sum() * (SIDE_M/2)**3),
                       "expected_energy_J": expected["energy_J"],
                       "net_force_N": actual[:, 6:9].sum(axis=0).tolist(),
                       "net_current_moment_Nm": np.cross(actual[:, :3], actual[:, 6:9]).sum(axis=0).tolist(),
                       "center_displacement_m": actual[13, 3:6].tolist()})
    return {"case": case_id, "passed": not failures, "failures": failures, "states": states, "solver": solver,
            "scope": "Tiny homogeneous patch verification only; no locking, mesh convergence, specimen fit, or patient mechanics validation"}


def write_bundle(output):
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    manifest = {"version": VERSION, "status": "prepared_not_executed", "febio_commit": FEBIO_COMMIT,
        "units": {"length": "m", "force": "N", "stress": "Pa", "energy": "J"},
        "reference_modulus_role": "Arbitrary numerical scale, not a measured or fitted tissue property",
        "side_m": SIDE_M, "mu_reference_Pa": MU_REF_PA, "K_over_mu": K_OVER_MU,
        "law": "Ogden c1=2mu,m1=2,c2..6=0; pressure_model=1; laugon=0",
        "element": "three-field-solid hex8; 8 elements,27 nodes,center free in3 DOFs",
        "reaction_sign": "FEBio Rx/Ry/Rz = body-on-constraint = negative required applied nodal force",
        "time_fractions": list(TIMES), "tolerances": TOLERANCES, "cases": cases(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "decks": {}}
    for name in cases():
        content = deck_xml(name).encode(); (output/f"{name}.feb").write_bytes(content)
        manifest["decks"][f"{name}.feb"] = hashlib.sha256(content).hexdigest()
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False)+'\n')
    return manifest


def check_stiffness_scaling(base_node_text, base_element_text, doubled_node_text, doubled_element_text):
    """Cross-check actual outputs at fixed prescribed motion when all stiffness doubles."""
    base_n = parse_data_log(base_node_text, field_count=9, record_name="mechanics_nodes_si", item_count=27)
    base_e = parse_data_log(base_element_text, field_count=8, record_name="mechanics_elements_si", item_count=8)
    double_n = parse_data_log(doubled_node_text, field_count=9, record_name="mechanics_nodes_si", item_count=27)
    double_e = parse_data_log(doubled_element_text, field_count=8, record_name="mechanics_elements_si", item_count=8)
    records = (base_n, base_e, double_n, double_e)
    if any([r["time"] for r in data] != [r["time"] for r in base_n] for data in records):
        raise ValueError("Scaling pair states differ")
    checks = []
    for bn, be, dn, de in zip(*records):
        n, e, n2, e2 = (row["values"] for row in (bn, be, dn, de))
        checks.append({"time": bn["time"],
            "same_motion": bool(np.max(np.abs(n2[:, :6]-n[:, :6])) <= TOLERANCES["position_m"]),
            "double_reaction": bool(np.all(np.abs(n2[:, 6:]-2*n[:, 6:]) <= TOLERANCES["force_N"]+TOLERANCES["relative"]*np.abs(2*n[:, 6:]))),
            "same_J": bool(np.max(np.abs(e2[:, 6]-e[:, 6])) <= TOLERANCES["jacobian"]),
            "double_stress": bool(np.all(np.abs(e2[:, :6]-2*e[:, :6]) <= TOLERANCES["stress_Pa"]+TOLERANCES["relative"]*np.abs(2*e[:, :6]))),
            "double_energy": bool(np.all(np.abs(e2[:, 7]-2*e[:, 7]) <= TOLERANCES["energy_density_Pa"]+TOLERANCES["relative"]*np.abs(2*e[:, 7])))})
    return {"passed": all(all(value for key, value in row.items() if key != "time") for row in checks), "states": checks}


def read_bounded_text(path, maximum_bytes=4*1024**2):
    with Path(path).open("rb") as stream:
        data = stream.read(maximum_bytes+1)
    if len(data) > maximum_bytes: raise ValueError("Control output exceeds bounded input size")
    return data.decode("utf-8", errors="strict")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation", required=True)
    prepare = sub.add_parser("prepare"); prepare.add_argument("--output", type=Path, required=True)
    check = sub.add_parser("check"); check.add_argument("--case", choices=cases(), required=True)
    check.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    if args.operation == "prepare": write_bundle(args.output); return 0
    if read_bounded_text(args.directory/f"{args.case}.feb") != deck_xml(args.case):
        raise ValueError("Result directory deck differs from the frozen control")
    texts = [read_bounded_text(args.directory/f"{args.case}{suffix}") for suffix in ('.nodes.log', '.elements.log', '.log')]
    result = check_outputs(args.case, *texts)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
