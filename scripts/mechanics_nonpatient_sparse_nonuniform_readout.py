"""Source-only constitutive, force, and fixed-point audit of numerical cubes.

No solver is launched here. This module consumes complete FEBio text records or
the existing strict parser's records; neither route establishes native provenance.
The Ogden constants are an idealized numerical gauge, not brain properties.
"""
from __future__ import annotations

import numpy as np

from scripts import mechanics_nonpatient_sparse_deck as deck
from scripts import mechanics_nonpatient_sparse_feasibility as design
from scripts import mechanics_nonpatient_sparse_output as output
from scripts import mechanics_patient_constraints as fixture


CASE_IDS = frozenset(("n5_nonuniform", "n9_nonuniform", "n13_nonuniform",
                      "n13_nonuniform_half_step"))
# Existing declared allowances, applied per element/node. No parent threshold
# is edited. A future stricter criterion requires a new prospective version.
SAMPLE_AGREEMENT_M = 1e-12
REFERENCE_LOCATION_M = 1e-12
CHUNK_ELEMENTS = 256
LOCAL_FORCE_TOL_N = 1e-8
EXPECTED_OWNER_COUNTS = (6, 2, 2, 2, 2, 2, 2)


def _response(F: np.ndarray, mu: float, bulk: float):
    J = np.linalg.det(F)
    if not np.isfinite(J).all() or np.any(J <= 0):
        raise ValueError("Nonpositive integration-point deformation Jacobian")
    B = F @ np.swapaxes(F, -1, -2)
    trace = np.trace(B, axis1=-2, axis2=-1)
    W = mu / 2 * (J ** (-2 / 3) * trace - 3) + bulk / 4 * (J * J - 1 - 2 * np.log(J))
    sigma = mu * J[..., None, None] ** (-5 / 3) * (
        B - trace[..., None, None] / 3 * np.eye(3))
    sigma += (bulk / 2 * (J - 1 / J))[..., None, None] * np.eye(3)
    P = J[..., None, None] * sigma @ np.swapaxes(np.linalg.inv(F), -1, -2)
    if not (np.isfinite(W).all() and np.isfinite(sigma).all() and np.isfinite(P).all()):
        raise ValueError("Nonfinite reconstructed Ogden response")
    return J, W, sigma, P


def _sample_locator(mesh: deck.CubeMesh, points: np.ndarray):
    """Return every enclosing straight-reference tetrahedron for each point."""
    X, E = mesh.nodes_m, mesh.tet10_indices
    corner = X[E[:, :4]]
    matrix = np.swapaxes(corner[:, 1:] - corner[:, :1], 1, 2)
    inverse = np.linalg.inv(matrix)
    # The physical tolerance scales to each cell's barycentric coordinates.
    bary_tolerance = np.linalg.norm(np.concatenate(
        (-inverse.sum(axis=1, keepdims=True), inverse), axis=1), axis=2) * REFERENCE_LOCATION_M
    locators = []
    for point in points:
        local = np.einsum("eij,ej->ei", inverse, point - corner[:, 0])
        bary = np.column_stack((1 - local.sum(axis=1), local))
        hits = np.flatnonzero(np.all((bary >= -bary_tolerance) &
                                     (bary <= 1 + bary_tolerance), axis=1))
        if not len(hits):
            raise ValueError("Fixed physical point outside reference mesh")
        # A tolerance-only owner must share the same face/edge/vertex as the
        # other owners; an unrelated overlap cannot become an interpolation.
        common = set(E[hits[0], :4].tolist())
        for cell in E[hits[1:], :4]:
            common.intersection_update(cell.tolist())
        if len(hits) > 1 and (not common or any(
            abs(weight) > tol for cell_id, weights in zip(hits, bary[hits], strict=True)
            for vertex, weight, tol in zip(E[cell_id, :4], weights,
                                           bary_tolerance[cell_id], strict=True)
            if int(vertex) not in common)):
            raise ValueError("Ambiguous physical point ownership")
        locators.append((hits, bary[hits]))
    if tuple(len(hits) for hits, _ in locators) != EXPECTED_OWNER_COUNTS:
        raise ValueError("Frozen sample point interface multiplicity changed")
    return locators


def _samples(mesh: deck.CubeMesh, displacement: np.ndarray, points: np.ndarray, locators):
    result, owner_counts, disagreements = [], [], []
    for point, (hits, bary) in zip(points, locators, strict=True):
        L = bary
        edges = np.asarray(fixture.EDGES)
        shape = np.concatenate((L * (2 * L - 1),
                                4 * L[:, edges[:, 0]] * L[:, edges[:, 1]]), axis=1)
        reference = np.einsum("en,eni->ei", shape, mesh.nodes_m[mesh.tet10_indices[hits]])
        if float(np.max(np.linalg.norm(reference - point, axis=1))) > REFERENCE_LOCATION_M:
            raise ValueError("Reference tet10 interpolation misses physical point")
        values = np.einsum("en,eni->ei", shape, displacement[mesh.tet10_indices[hits]])
        disagreement = float(np.max(np.linalg.norm(values - values[0], axis=1)))
        if disagreement > SAMPLE_AGREEMENT_M:
            raise ValueError("Containing tet10 interpolants disagree")
        result.append(values[0].tolist())  # lowest element ID owns exact ties
        owner_counts.append(len(hits))
        disagreements.append(disagreement)
    return result, owner_counts, disagreements


def _state(mesh: deck.CubeMesh, nodal: np.ndarray, logged: np.ndarray,
           time: float, declared: dict, points: np.ndarray, locators) -> dict:
    X, E = mesh.nodes_m, mesh.tet10_indices
    x, u, reactions = nodal[:, :3], nodal[:, 3:6], nodal[:, 6:9]
    q, w = fixture.gauss_rule()
    derivatives = np.stack([fixture.shape(point)[1] for point in q])
    checks = declared["checks"]
    gauge = declared["numerical_gauge"]
    volume = 0.
    energy = 0.
    min_J = np.inf
    max_stress = max_J = max_density = 0.
    f_int = np.zeros_like(X)
    for start in range(0, len(E), CHUNK_ELEMENTS):
        ids = E[start:start + CHUNK_ELEMENTS]
        J0 = np.einsum("eni,qnj->eqij", X[ids], derivatives)
        Jx = np.einsum("eni,qnj->eqij", x[ids], derivatives)
        det0 = np.linalg.det(J0)
        if not np.isfinite(det0).all() or np.any(det0 <= 0):
            raise ValueError("Invalid reference integration Jacobian")
        inverse = np.linalg.inv(J0)
        F = Jx @ inverse
        J, W, sigma, P = _response(F, gauge["mu_Pa"], gauge["K_Pa"])
        dV = det0 * w
        gradient = np.einsum("qnj,eqjk->eqnk", derivatives, inverse)
        local_force = np.einsum("eqij,eqnj,eq->eni", P, gradient, dV)
        np.add.at(f_int, ids.reshape(-1), local_force.reshape(-1, 3))
        reference_element_volume = dV.sum(axis=1)
        volume += float(reference_element_volume.sum())
        energy += float(np.sum(W * dV))
        stress_fields = np.stack((sigma[..., 0, 0], sigma[..., 1, 1],
                                  sigma[..., 2, 2], sigma[..., 0, 1],
                                  sigma[..., 1, 2], sigma[..., 0, 2]), axis=-1)
        # FEBio's DataRecord element output is the arithmetic G8 mean.
        max_stress = max(max_stress, float(np.max(np.abs(
            logged[start:start + len(ids), :6] - stress_fields.mean(axis=1)))))
        max_J = max(max_J, float(np.max(np.abs(
            logged[start:start + len(ids), 6] - J.mean(axis=1)))))
        max_density = max(max_density, float(np.max(np.abs(
            logged[start:start + len(ids), 7] - W.mean(axis=1)))))
        min_J = min(min_J, float(J.min()))
    side = declared["mesh"]["side_m"]
    if abs(volume - side ** 3) > 1e-15:
        raise ValueError("Reference integration volume differs from declared cube")
    bottom, top = mesh.face_node_ids["bottom"], mesh.face_node_ids["top"]
    fixed = np.zeros(len(X), dtype=bool)
    fixed[np.concatenate((bottom, top))] = True
    free = ~fixed
    # FEBio Rx is body-on-constraint: negative of applied force required to
    # sustain prescribed displacement. Verified against saved n5 affine output.
    residual = f_int + reactions
    max_free = float(np.max(np.linalg.norm(f_int[free], axis=1)))
    max_constrained = float(np.max(np.linalg.norm(residual[fixed], axis=1)))
    max_logged_free = float(np.max(np.linalg.norm(reactions[free], axis=1)))
    density_limit = checks["maximum_affine_energy_error_J"] / side ** 3
    errors = {"logged_stress_from_actual_Pa": max_stress,
              "logged_J_from_actual": max_J,
              "logged_density_from_actual_Pa": max_density,
              "maximum_free_internal_force_N": max_free,
              "maximum_constrained_force_residual_N": max_constrained,
              "maximum_logged_free_reaction_N": max_logged_free}
    gates = {"positive_actual_J": min_J >= checks["minimum_sampled_deformation_J"],
             "logged_stress_matches_actual": max_stress <= checks["maximum_affine_stress_error_Pa"],
             "logged_J_matches_actual": max_J <= checks["maximum_affine_J_error"],
             "logged_density_matches_actual": max_density <= density_limit,
             "local_free_equilibrium": max_free <= LOCAL_FORCE_TOL_N,
             "local_signed_constraint_reactions": max_constrained <= LOCAL_FORCE_TOL_N,
             "zero_logged_free_reactions": max_logged_free <= LOCAL_FORCE_TOL_N}
    if not np.isfinite(x).all() or not np.isfinite(u).all() or not np.isfinite(reactions).all():
        raise ValueError("Complete finite node fields required")
    samples, owner_counts, disagreements = _samples(mesh, u, points, locators)
    return {"time": time, "passed": bool(all(gates.values())), "checks": gates,
            "errors": errors, "minimum_actual_J": min_J,
            "reconstructed_energy_J": energy,
            "logged_density_interpretation": "Arithmetic G8 mean per element; not a physical energy integral",
            "boundary_work_J": float(np.sum(reactions[fixed] * u[fixed])),
            "top_reaction_N": reactions[top].sum(axis=0).tolist(),
            "bottom_reaction_N": reactions[bottom].sum(axis=0).tolist(),
            "sample_displacement_m": samples, "sample_owner_counts": owner_counts,
            "maximum_sample_tie_disagreement_m": max(disagreements),
            "derived_density_limit_Pa": density_limit,
            "local_force_limit_N": LOCAL_FORCE_TOL_N}


def check_parsed_nonuniform_readout(case_id: str, node_states: list[dict],
                                    element_states: list[dict]) -> dict:
    """Audit parsed records; caller separately requires solver/output gates."""
    if case_id not in CASE_IDS:
        raise ValueError("Undeclared nonuniform case")
    declared = design.validate_declaration()
    case = next(row for row in declared["cases"] if row["id"] == case_id)
    mesh = deck.build_mesh(case["n"])
    steps = case.get("time_steps", declared["numerical_gauge"]["time_steps"])
    dt = case.get("step_size", declared["numerical_gauge"]["step_size"])
    times = tuple(step * dt for step in range(steps + 1))
    if len(node_states) != len(times) or len(element_states) != len(times):
        raise ValueError("Complete nonuniform states required")
    points = np.asarray(declared["loads"]["nonuniform"]["sample_points_m"])
    locators = _sample_locator(mesh, points)
    rows = []
    for step, (node, element, time) in enumerate(zip(node_states, element_states, times, strict=True)):
        if (node.get("step") != step or element.get("step") != step
                or node.get("time") != time or element.get("time") != time
                or node.get("name") != "mechanics_nodes_si"
                or element.get("name") != "mechanics_elements_si"):
            raise ValueError("Unaligned nonuniform record step, time, or field label")
        nodal = np.asarray(node.get("values"), dtype=float)
        logged = np.asarray(element.get("values"), dtype=float)
        if (nodal.shape != (len(mesh.nodes_m), 9)
                or logged.shape != (len(mesh.tet10_indices), 8)
                or not np.isfinite(nodal).all() or not np.isfinite(logged).all()):
            raise ValueError("Complete finite nonuniform node and element fields required")
        rows.append(_state(mesh, nodal, logged, time, declared, points, locators))
    return {"case_id": case_id, "passed": bool(all(row["passed"] for row in rows)),
            "states": rows, "sample_points_m": points.tolist(),
            "scope": "Supplied nonpatient numerical cube fields only; parsed entry has no solver or native provenance, mesh convergence, physical validation, or clinical evidence."}


def check_nonuniform_readout(case_id: str, node_text: str, element_text: str,
                             solver_text: str) -> dict:
    """Apply existing strict saved-output gates plus nonuniform readout."""
    baseline = output.check_case_outputs(case_id, node_text, element_text, solver_text)
    declared = design.validate_declaration()
    case = next(row for row in declared["cases"] if row["id"] == case_id)
    mesh = deck.build_mesh(case["n"])
    steps = case.get("time_steps", declared["numerical_gauge"]["time_steps"])
    dt = case.get("step_size", declared["numerical_gauge"]["step_size"])
    times = tuple(step * dt for step in range(steps + 1))
    nodes = output.parse_data_log(node_text, expected_times=times,
                                  item_count=len(mesh.nodes_m), field_count=9,
                                  record_name="mechanics_nodes_si")
    elements = output.parse_data_log(element_text, expected_times=times,
                                     item_count=len(mesh.tet10_indices), field_count=8,
                                     record_name="mechanics_elements_si")
    result = check_parsed_nonuniform_readout(case_id, nodes, elements)
    result["existing_output_checks_passed"] = baseline["passed"]
    result["passed"] = bool(result["passed"] and baseline["passed"])
    return result
