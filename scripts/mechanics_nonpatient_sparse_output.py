"""Bounded, source-only FEBio text-output checks for one declared numerical cube.

The saved outputs have no patient observations or measured tissue properties.
This module never reads files or starts FEBio. A separate release must establish
that native outputs came from the pinned deck, executable, and solver backend.
"""
from __future__ import annotations

import math
import re

import numpy as np

from scripts import mechanics_nonpatient_sparse_deck as deck
from scripts import mechanics_nonpatient_sparse_feasibility as design
from scripts import mechanics_patient_constraints as fixture


# The declaration limits *all* active output to 512 MiB. Limiting each supplied
# ASCII log and their combined size before parsing also bounds text-row work.
MAX_TOTAL_CHARS = 536_870_912
MAX_LINE_CHARS = 2_048
RESIDUAL_PRINT_RTOL = 2e-5  # Existing verified FEBio 12-digit text-log allowance.
_STEP = re.compile(r"\*Step\s*=\s*(\d+)")
_TIME = re.compile(r"\*Time\s*=\s*(\S+)")
_DATA = re.compile(r"\*Data\s*=\s*(\S+)")
_STATUS = re.compile(r"Nonlinear solution status:\s*time=\s*(\S+)")
_NORMS = re.compile(r"convergence norms\s*:\s*INITIAL\s+CURRENT\s+REQUIRED", re.I)
_CONVERGED = re.compile(r"-{3,}\s+converged at time\s*:\s*(\S+)", re.I)
_COUNT_NODES = re.compile(r"Number of nodes\s*\.+\s*:\s*(\d+)\b")
_COUNT_ELEMENTS = re.compile(r"Number of solid elements\s*\.+\s*:\s*(\d+)\b")
_SOLVER_TOLERANCE = re.compile(r"\s*(dtol|etol|rtol|min_residual)\s*\.+\s*:\s*(\S+)\s*")


def _ascii_lines(value: str):
    if not isinstance(value, str) or not value.isascii():
        raise ValueError("FEBio text output must be ASCII")
    start = 0
    while start < len(value):
        end = value.find("\n", start)
        if end < 0:
            end = len(value)
        if end - start > MAX_LINE_CHARS:
            raise ValueError("FEBio output line exceeds bound")
        yield value[start:end].rstrip("\r")
        start = end + 1


def _number(value: str, *, positive: bool = False) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError("Malformed FEBio number") from exc
    if not math.isfinite(parsed) or (positive and parsed <= 0):
        raise ValueError("Nonfinite or nonpositive FEBio number")
    return parsed


def _expected_times(case: dict, gauge: dict) -> tuple[float, ...]:
    count = case.get("time_steps", gauge["time_steps"])
    step = case.get("step_size", gauge["step_size"])
    return tuple(k * step for k in range(count + 1))


def parse_data_log(text: str, *, expected_times: tuple[float, ...],
                   item_count: int, field_count: int, record_name: str) -> list[dict]:
    """Parse FEBio 4.13 DataRecord rows by exact 1-based IDs and full steps.

    This is deliberately stricter than the old tiny-control parser: no omitted
    initial state, duplicate or reordered row ID, omitted/extra state, or NaN.
    """
    if len(text) > MAX_TOTAL_CHARS:
        raise ValueError("Data log exceeds declared active-output bound")
    if item_count <= 0 or field_count <= 0 or not expected_times:
        raise ValueError("Invalid expected data-record shape")
    records: list[dict] = []
    current: dict | None = None
    next_id = 1

    def finish() -> None:
        nonlocal current
        if current is None:
            return
        if current["time"] is None or current["name"] != record_name or next_id != item_count + 1:
            raise ValueError("Incomplete FEBio data record")
        records.append(current)
        current = None

    for raw in _ascii_lines(text):
        line = raw.strip()
        if not line:
            continue
        if line.startswith("*Step"):
            finish()
            match = _STEP.fullmatch(line)
            if not match or len(records) >= len(expected_times) or int(match[1]) != len(records):
                raise ValueError("Missing, duplicate, or out-of-order FEBio step")
            current = {"step": len(records), "time": None, "name": None,
                       "values": np.empty((item_count, field_count), dtype=np.float64)}
            next_id = 1
        elif line.startswith("*Time"):
            match = _TIME.fullmatch(line)
            if current is None or current["time"] is not None or not match:
                raise ValueError("Missing or duplicate FEBio time")
            observed = _number(match[1])
            if abs(observed - expected_times[current["step"]]) > 1e-12:
                raise ValueError("FEBio step time differs from frozen schedule")
            current["time"] = observed
        elif line.startswith("*Data"):
            match = _DATA.fullmatch(line)
            if current is None or current["time"] is None or current["name"] is not None or not match:
                raise ValueError("Missing or duplicate FEBio data label")
            if match[1] != record_name:
                raise ValueError("Unexpected FEBio data-record label")
            current["name"] = match[1]
        else:
            if current is None or current["name"] is None or next_id > item_count:
                raise ValueError("Unbound or excess FEBio data row")
            cells = line.split(",")
            if len(cells) != field_count + 1 or cells[0] != str(next_id):
                raise ValueError("Missing, duplicate, or reordered FEBio row ID")
            current["values"][next_id - 1] = [_number(cell) for cell in cells[1:]]
            next_id += 1
    finish()
    if len(records) != len(expected_times):
        raise ValueError("Missing FEBio initial or completed load state")
    return records


def parse_solver_log(text: str, *, expected_times: tuple[float, ...],
                     node_count: int, element_count: int) -> dict:
    """Require FEBio mesh counts, every convergence token, and final residuals.

    This parser recognizes the grammar saved by the pinned FEBio 4.13 tet10
    fixture. Different native grammar is an explicit qualification dependency.
    """
    if len(text) > 16 * 1024**2:
        raise ValueError("Solver log exceeds declared active-output bound")
    lines = list(_ascii_lines(text))
    banners = [re.sub(r"[^A-Za-z]", "", line).upper() for line in lines]
    termination = [i for i, banner in enumerate(banners) if banner == "NORMALTERMINATION"]
    last_nonempty = next((i for i in range(len(lines)-1, -1, -1) if lines[i].strip()), -1)
    if termination != [last_nonempty]:
        raise ValueError("No FEBio normal-termination evidence")
    if ("ERRORTERMINATION" in banners or "ABNORMALTERMINATION" in banners
            or re.search(r"\b(?:E\s+R\s+R\s+O\s+R|ERROR|FATAL|EXCEPTION)\b|"
                         r"negative jacobian|failed to converge|not converged|\bnan\b", text, re.I)):
        raise ValueError("FEBio failure evidence")
    tolerance_headers = {name: [] for name in ("dtol", "etol", "rtol", "min_residual")}
    for line in lines:
        match = _SOLVER_TOLERANCE.fullmatch(line)
        if match:
            tolerance_headers[match[1]].append(_number(match[2]))
        elif re.search(r"\b(?:dtol|etol|rtol|min_residual)\b", line):
            raise ValueError("Ambiguous FEBio solver-tolerance header")
    if any(tolerance_headers[name] != [1e-10] for name in ("dtol", "etol", "rtol")):
        raise ValueError("FEBio dtol/etol/rtol headers differ from frozen 1e-10")
    if tolerance_headers["min_residual"] != [1e-20]:
        raise ValueError("FEBio min_residual header differs from frozen 1e-20 N2")
    nodes = [int(_COUNT_NODES.search(s)[1]) for s in lines if _COUNT_NODES.search(s)]
    elements = [int(_COUNT_ELEMENTS.search(s)[1]) for s in lines if _COUNT_ELEMENTS.search(s)]
    if nodes != [node_count] or elements != [element_count]:
        raise ValueError("FEBio log mesh counts missing or inconsistent")
    times = expected_times[1:]
    state: dict | None = None
    completed: list[dict] = []
    for raw in lines:
        line = raw.strip()
        status = _STATUS.fullmatch(line)
        if "Nonlinear solution status" in line and status is None:
            raise ValueError("Malformed nonlinear status token")
        if status:
            if state is not None and state["residual"] is None:
                raise ValueError("Unfinished nonlinear status before next status")
            time = _number(status[1])
            if len(completed) >= len(times) or abs(time - times[len(completed)]) > 1e-12:
                raise ValueError("Unexpected solver status time")
            state = {"time": time, "norms_header": False, "residual": None}
            continue
        if _NORMS.fullmatch(line):
            if state is None or state["norms_header"] or state["residual"] is not None:
                raise ValueError("Unbound or duplicate convergence-norms header")
            state["norms_header"] = True
            continue
        cells = line.split()
        if (cells and cells[0] != "residual"
                and any(token.lower() == "residual" and len(cells) - index - 1 == 3
                        for index, token in enumerate(cells))):
            raise ValueError("Ambiguous FEBio residual-like line")
        if cells and cells[0] == "residual":
            if (state is None or not state["norms_header"]
                    or state["residual"] is not None or len(cells) != 4):
                raise ValueError("Unbound, duplicate, or malformed residual norms")
            norms = tuple(_number(cell) for cell in cells[1:])
            if min(norms) < 0:
                raise ValueError("Negative residual norm")
            state["residual"] = norms
            continue
        converged = _CONVERGED.fullmatch(line)
        if re.search(r"\bconverged\s+at\s+time\b", line, re.I) and converged is None:
            raise ValueError("Malformed or ambiguous FEBio convergence token")
        if converged:
            time = _number(converged[1])
            if (state is None or state["residual"] is None or len(completed) >= len(times)
                    or abs(time - state["time"]) > 1e-12
                    or abs(time - times[len(completed)]) > 1e-12):
                raise ValueError("Unbound, duplicate, or out-of-order convergence token")
            initial, current, required = state["residual"]
            if not math.isclose(required, initial * 1e-10, rel_tol=RESIDUAL_PRINT_RTOL, abs_tol=1e-30):
                raise ValueError("Reported solver residual rule differs from frozen 1e-10")
            passed = current <= max(required, 1e-20) * (1 + RESIDUAL_PRINT_RTOL)
            completed.append({"time": time, "initial_N2": initial,
                              "current_N2": current, "required_N2": required,
                              "residual_passed": passed})
            state = None
    if state is not None or len(completed) != len(times):
        raise ValueError("Incomplete FEBio convergence evidence")
    return {"passed": all(row["residual_passed"] for row in completed),
            "steps": completed,
            "scope": "Saved FEBio log tokens only; executable/backend identity requires separate binding."}


def sampled_deformation_jacobians(mesh: deck.CubeMesh, current_m: np.ndarray) -> tuple[float, np.ndarray]:
    """Reconstruct dimensionless det(F) at G8, vertices, mid-edges, center."""
    if current_m.shape != mesh.nodes_m.shape or not np.isfinite(current_m).all():
        raise ValueError("Complete finite current node positions required")
    points, _ = fixture.gauss_rule()
    extra = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.],
                      [.25, .25, .25]] +
                     [((np.eye(4)[a] + np.eye(4)[b])/2)[1:].tolist()
                      for a, b in fixture.EDGES])
    derivatives = np.stack([fixture.shape(q)[1] for q in np.vstack((points, extra))])
    average = np.empty(len(mesh.tet10_indices), dtype=np.float64)
    minimum = math.inf
    for start in range(0, len(average), 512):
        indices = mesh.tet10_indices[start:start + 512]
        jac0 = np.einsum("eni,qnj->eqij", mesh.nodes_m[indices], derivatives)
        jac1 = np.einsum("eni,qnj->eqij", current_m[indices], derivatives)
        J = np.linalg.det(jac1) / np.linalg.det(jac0)
        if not np.isfinite(J).all():
            raise ValueError("Nonfinite sampled deformation Jacobian")
        minimum = min(minimum, float(J.min()))
        average[start:start + len(indices)] = J[:, :8].mean(axis=1)
    return minimum, average


def check_case_outputs(case_id: str, node_text: str, element_text: str,
                       solver_text: str) -> dict:
    """Check one frozen case from supplied text, without assigning provenance.

    Returns per-state numerical gates and compact metrics. Its `passed` applies
    only to these output-level checks, never to native-release or physical gates.
    """
    declared = design.validate_declaration()
    case = next((row for row in declared["cases"] if row["id"] == case_id), None)
    if case is None:
        raise ValueError("Undeclared numerical cube case")
    texts = (node_text, element_text, solver_text)
    if any(not isinstance(value, str) for value in texts) or sum(map(len, texts)) > MAX_TOTAL_CHARS:
        raise ValueError("Combined FEBio output exceeds declared active-output bound")
    mesh = deck.build_mesh(case["n"])
    times = _expected_times(case, declared["numerical_gauge"])
    nodes = parse_data_log(node_text, expected_times=times, item_count=len(mesh.nodes_m),
                           field_count=9, record_name="mechanics_nodes_si")
    elements = parse_data_log(element_text, expected_times=times,
                              item_count=len(mesh.tet10_indices), field_count=8,
                              record_name="mechanics_elements_si")
    solver = parse_solver_log(solver_text, expected_times=times,
                              node_count=len(mesh.nodes_m),
                              element_count=len(mesh.tet10_indices))
    checks = declared["checks"]
    states = []
    for node_record, element_record in zip(nodes, elements, strict=True):
        time = node_record["time"]
        if time != element_record["time"]:
            raise ValueError("Node and element state times differ")
        nodal, elemental = node_record["values"], element_record["values"]
        current, displacement, reactions = nodal[:, :3], nodal[:, 3:6], nodal[:, 6:9]
        minimum_J, mean_gauss_J = sampled_deformation_jacobians(mesh, current)
        if case["load"] == "affine":
            F = np.eye(3) + time * (np.asarray(declared["loads"]["affine"]["final_F"]) - np.eye(3))
            boundary = np.unique(np.concatenate(list(mesh.face_node_ids.values())))
            prescribed = mesh.nodes_m[boundary] @ (F - np.eye(3)).T
        else:
            boundary = np.concatenate((mesh.face_node_ids["bottom"], mesh.face_node_ids["top"]))
            base = np.asarray(declared["loads"]["nonuniform"]["bottom_displacement_m"])
            top = np.asarray(declared["loads"]["nonuniform"]["top_displacement_m"])
            prescribed = np.vstack((np.broadcast_to(time * base, (len(mesh.face_node_ids["bottom"]), 3)),
                                    np.broadcast_to(time * top, (len(mesh.face_node_ids["top"]), 3))))
        free = np.ones(len(mesh.nodes_m), dtype=bool)
        free[boundary] = False
        errors = {
            "position_displacement_m": float(np.max(np.linalg.norm(current - mesh.nodes_m - displacement, axis=1))),
            "boundary_displacement_m": float(np.max(np.linalg.norm(displacement[boundary] - prescribed, axis=1))),
            "logged_vs_sampled_J": float(np.max(np.abs(elemental[:, 6] - mean_gauss_J))),
            "net_force_N": float(np.linalg.norm(reactions.sum(axis=0))),
            "net_current_moment_Nm": float(np.linalg.norm(np.cross(current, reactions).sum(axis=0))),
            "maximum_free_node_reaction_N": float(np.max(np.linalg.norm(reactions[free], axis=1))),
        }
        gates = {
            "position_displacement_consistency": errors["position_displacement_m"] <= checks["maximum_affine_displacement_error_m"],
            "prescribed_boundary_motion": errors["boundary_displacement_m"] <= checks["maximum_affine_displacement_error_m"],
            "positive_sampled_deformation_J": minimum_J >= checks["minimum_sampled_deformation_J"],
            "positive_logged_element_J": float(elemental[:, 6].min()) >= checks["minimum_sampled_deformation_J"],
            "logged_J_matches_actual_gauss_average": errors["logged_vs_sampled_J"] <= checks["maximum_affine_J_error"],
            "force_balance": errors["net_force_N"] <= checks["maximum_independent_force_balance_N"],
            "moment_balance": errors["net_current_moment_Nm"] <= checks["maximum_independent_moment_balance_Nm"],
            "free_node_reactions": errors["maximum_free_node_reaction_N"] <= checks["maximum_independent_force_balance_N"],
        }
        if node_record["step"] == 0:
            side_m = declared["mesh"]["side_m"]
            initial = {
                "position_m": float(np.max(np.linalg.norm(current - mesh.nodes_m, axis=1))),
                "displacement_m": float(np.max(np.linalg.norm(displacement, axis=1))),
                "reaction_N": float(np.max(np.linalg.norm(reactions, axis=1))),
                "stress_Pa": float(np.max(np.abs(elemental[:, :6]))),
                "logged_J": float(np.max(np.abs(elemental[:, 6] - 1))),
                "sampled_J": float(np.max(np.abs(mean_gauss_J - 1))),
                "energy_density_Pa": float(np.max(np.abs(elemental[:, 7]))),
            }
            errors["initial_unloaded"] = initial
            gates["initial_unloaded_state"] = (
                initial["position_m"] <= checks["maximum_affine_displacement_error_m"]
                and initial["displacement_m"] <= checks["maximum_affine_displacement_error_m"]
                and initial["reaction_N"] <= checks["maximum_independent_force_balance_N"]
                and initial["stress_Pa"] <= checks["maximum_affine_stress_error_Pa"]
                and initial["logged_J"] <= checks["maximum_affine_J_error"]
                and initial["sampled_J"] <= checks["maximum_affine_J_error"]
                and initial["energy_density_Pa"] * side_m**3 <= checks["maximum_affine_energy_error_J"])
        states.append({"step": node_record["step"], "time": time,
                       "minimum_sampled_deformation_J": minimum_J,
                       "minimum_logged_element_J": float(elemental[:, 6].min()),
                       "checks": gates, "errors": errors,
                       "top_reaction_N": reactions[mesh.face_node_ids["top"]].sum(axis=0).tolist(),
                       "bottom_reaction_N": reactions[mesh.face_node_ids["bottom"]].sum(axis=0).tolist()})
    passed = solver["passed"] and all(all(row["checks"].values()) for row in states)
    return {"case_id": case_id, "passed": bool(passed), "states": states, "solver": solver,
            "scope": "Saved non-patient cube output checks only; no native provenance, numerical convergence across meshes, physical validation, or clinical evidence."}
