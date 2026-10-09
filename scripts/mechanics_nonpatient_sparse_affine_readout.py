"""Independent field readout for supplied declared affine numerical cubes.

This source-only check reads pinned local declarations through existing
validators but no saved outputs or patient files; it does not start FEBio,
identify a native backend, or claim measured brain properties. Complete FEBio
text records are parsed by the separately reviewed sparse-output grammar.
"""
from __future__ import annotations

import numpy as np

from scripts import mechanics_nonpatient_sparse_deck as deck
from scripts import mechanics_nonpatient_sparse_feasibility as design
from scripts import mechanics_nonpatient_sparse_output as output
from scripts import mechanics_patient_constraints as fixture


CASE_ID = "n5_affine"
AFFINE_CASES = frozenset(("n5_affine", "n9_affine", "n13_affine"))


def _case(declared: dict, case_id: str) -> dict:
    if case_id not in AFFINE_CASES:
        raise ValueError("Undeclared affine case")
    case = next((row for row in declared["cases"] if row["id"] == case_id), None)
    if case is None or case["load"] != "affine":
        raise ValueError("Affine case differs from frozen declaration")
    return case


def _response(F: np.ndarray, *, mu_Pa: float, K_Pa: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate the declared Ogden-alpha-2 gauge at reconstructed G8 F.

    The constitutive *assumption* is necessarily shared with the oracle. The
    reconstruction of F from actual nodal positions and tet10 quadrature is
    independent of the declared affine target and the element text fields.
    """
    J = np.linalg.det(F)
    if not np.isfinite(J).all() or np.any(J <= 0):
        raise ValueError("Nonpositive actual Gauss-point deformation Jacobian")
    B = F @ np.swapaxes(F, -1, -2)
    trace_B = np.trace(B, axis1=-2, axis2=-1)
    W = mu_Pa / 2 * (J ** (-2 / 3) * trace_B - 3) + K_Pa / 4 * (J * J - 1 - 2 * np.log(J))
    stress = mu_Pa * J[..., None, None] ** (-5 / 3) * (
        B - trace_B[..., None, None] / 3 * np.eye(3))
    stress += (K_Pa / 2 * (J - 1 / J))[..., None, None] * np.eye(3)
    if not np.isfinite(W).all() or not np.isfinite(stress).all():
        raise ValueError("Nonfinite reconstructed constitutive response")
    return J, W, stress


def _readout_state(mesh: deck.CubeMesh, nodes: np.ndarray, elements: np.ndarray,
                   time: float, declared: dict) -> dict:
    X, E = mesh.nodes_m, mesh.tet10_indices
    current, displacement, reactions = nodes[:, :3], nodes[:, 3:6], nodes[:, 6:9]
    target_F = np.eye(3) + time * (
        np.asarray(declared["loads"]["affine"]["final_F"], dtype=float) - np.eye(3))
    oracle = design.affine_oracle(target_F)
    points, weights = fixture.gauss_rule()
    derivatives = np.stack([fixture.shape(q)[1] for q in points])
    gauge, thresholds = declared["numerical_gauge"], declared["checks"]
    expected_stress = np.asarray(oracle["stress_Pa"])
    total_volume_m3 = declared["mesh"]["side_m"] ** 3
    maxima = {"actual_F": 0., "actual_J": 0., "actual_stress_Pa": 0.,
              "logged_J_from_actual": 0., "logged_stress_from_actual_Pa": 0.,
              "logged_energy_density_from_actual_Pa": 0.}
    minimum_J = np.inf
    actual_energy_J = 0.
    logged_energy_J = 0.
    reference_volume_m3 = 0.
    # Bounded chunking: no full [element, point, node]
    # gradient tensor need be retained across states.
    for start in range(0, len(E), 256):
        sl = slice(start, start + 256)
        X_tet, x_tet = X[E[sl]], current[E[sl]]
        dX = np.einsum("eni,qnj->eqij", X_tet, derivatives)
        dx = np.einsum("eni,qnj->eqij", x_tet, derivatives)
        det_dX = np.linalg.det(dX)
        if not np.isfinite(det_dX).all() or np.any(det_dX <= 0):
            raise ValueError("Invalid reference integration Jacobian")
        F = dx @ np.linalg.inv(dX)
        J, W, stress = _response(F, mu_Pa=gauge["mu_Pa"], K_Pa=gauge["K_Pa"])
        reference_weights_m3 = det_dX * weights
        element_volume_m3 = reference_weights_m3.sum(axis=1)
        actual_energy_J += float(np.sum(W * reference_weights_m3))
        logged_energy_J += float(np.sum(elements[sl, 7] * element_volume_m3))
        reference_volume_m3 += float(np.sum(element_volume_m3))
        minimum_J = min(minimum_J, float(J.min()))
        reconstructed_stress = np.stack((stress[..., 0, 0], stress[..., 1, 1],
                                         stress[..., 2, 2], stress[..., 0, 1],
                                         stress[..., 1, 2], stress[..., 0, 2]), axis=-1).mean(axis=1)
        maxima["actual_F"] = max(maxima["actual_F"], float(np.max(np.abs(F - target_F))))
        maxima["actual_J"] = max(maxima["actual_J"], float(np.max(np.abs(J - oracle["J"]))))
        maxima["actual_stress_Pa"] = max(maxima["actual_stress_Pa"],
                                          float(np.max(np.abs(stress - expected_stress))))
        maxima["logged_J_from_actual"] = max(maxima["logged_J_from_actual"],
                                              float(np.max(np.abs(elements[sl, 6] - J.mean(axis=1)))))
        maxima["logged_stress_from_actual_Pa"] = max(maxima["logged_stress_from_actual_Pa"],
                                                      float(np.max(np.abs(elements[sl, :6] - reconstructed_stress))))
        maxima["logged_energy_density_from_actual_Pa"] = max(
            maxima["logged_energy_density_from_actual_Pa"],
            float(np.max(np.abs(elements[sl, 7] - W.mean(axis=1)))))
    if abs(reference_volume_m3 - total_volume_m3) > 1e-15:
        raise ValueError("Reference quadrature volume differs from declared cube")
    boundary = np.unique(np.concatenate(list(mesh.face_node_ids.values())))
    free = np.ones(len(X), dtype=bool)
    free[boundary] = False
    errors = {
        "position_displacement_m": float(np.max(np.linalg.norm(current - X - displacement, axis=1))),
        "affine_displacement_m": float(np.max(np.linalg.norm(displacement - X @ (target_F - np.eye(3)).T, axis=1))),
        "actual_F_dimensionless": maxima["actual_F"],
        "actual_J_dimensionless": maxima["actual_J"],
        "actual_stress_Pa": maxima["actual_stress_Pa"],
        "logged_J_from_actual_dimensionless": maxima["logged_J_from_actual"],
        "logged_stress_from_actual_Pa": maxima["logged_stress_from_actual_Pa"],
        "logged_energy_density_from_actual_Pa": maxima["logged_energy_density_from_actual_Pa"],
        "actual_energy_J": abs(actual_energy_J - oracle["energy_J"]),
        "logged_energy_J": abs(logged_energy_J - actual_energy_J),
        "boundary_virtual_work_J": abs(float(np.sum(reactions * displacement)) - oracle["boundary_virtual_work_J"]),
        "force_balance_N": float(np.linalg.norm(reactions.sum(axis=0))),
        "moment_balance_Nm": float(np.linalg.norm(np.cross(current, reactions).sum(axis=0))),
        "free_reaction_N": float(np.max(np.linalg.norm(reactions[free], axis=1))),
    }
    # The density limit is just the declared total-energy limit divided by
    # the declared cube volume, valid here because the oracle is homogeneous.
    density_gate_Pa = thresholds["maximum_affine_energy_error_J"] / total_volume_m3
    checks = {
        "position_displacement_consistency": errors["position_displacement_m"] <= thresholds["maximum_affine_displacement_error_m"],
        "all_nodes_affine": errors["affine_displacement_m"] <= thresholds["maximum_affine_displacement_error_m"],
        "positive_actual_J": minimum_J >= thresholds["minimum_sampled_deformation_J"],
        "actual_J_matches_oracle": errors["actual_J_dimensionless"] <= thresholds["maximum_affine_J_error"],
        "actual_stress_matches_oracle": errors["actual_stress_Pa"] <= thresholds["maximum_affine_stress_error_Pa"],
        "logged_J_matches_actual": errors["logged_J_from_actual_dimensionless"] <= thresholds["maximum_affine_J_error"],
        "logged_stress_matches_actual": errors["logged_stress_from_actual_Pa"] <= thresholds["maximum_affine_stress_error_Pa"],
        "logged_density_matches_actual": errors["logged_energy_density_from_actual_Pa"] <= density_gate_Pa,
        "integrated_actual_energy_matches_oracle": errors["actual_energy_J"] <= thresholds["maximum_affine_energy_error_J"],
        "integrated_logged_energy_matches_actual": errors["logged_energy_J"] <= thresholds["maximum_affine_energy_error_J"],
        "signed_boundary_work_matches_oracle": errors["boundary_virtual_work_J"] <= thresholds["maximum_affine_boundary_virtual_work_error_J"],
        "force_balance": errors["force_balance_N"] <= thresholds["maximum_independent_force_balance_N"],
        "current_moment_balance": errors["moment_balance_Nm"] <= thresholds["maximum_independent_moment_balance_Nm"],
        "free_node_reactions": errors["free_reaction_N"] <= thresholds["maximum_independent_force_balance_N"],
    }
    return {"time": time, "passed": bool(all(checks.values())), "checks": checks,
            "errors": errors, "minimum_actual_J": minimum_J,
            "reconstructed_energy_J": actual_energy_J,
            "integrated_logged_energy_J": logged_energy_J,
            "boundary_virtual_work_J": float(np.sum(reactions * displacement)),
            "oracle_energy_J": oracle["energy_J"],
            "oracle_boundary_virtual_work_J": oracle["boundary_virtual_work_J"],
            "derived_density_gate_Pa": density_gate_Pa}


def check_parsed_affine_readout(node_states: list[dict], element_states: list[dict],
                                case_id: str = CASE_ID) -> dict:
    """Audit complete already-parsed declared affine records, with no native claim."""
    declared = design.validate_declaration()
    case = _case(declared, case_id)
    mesh = deck.build_mesh(case["n"])
    times = tuple(k * case.get("step_size", declared["numerical_gauge"]["step_size"])
                  for k in range(case.get("time_steps", declared["numerical_gauge"]["time_steps"]) + 1))
    if len(node_states) != len(times) or len(element_states) != len(times):
        raise ValueError("Complete initial and declared affine states required")
    rows = []
    for step, (node, element, time) in enumerate(zip(node_states, element_states, times, strict=True)):
        if (node.get("step") != step or element.get("step") != step
                or node.get("name") != "mechanics_nodes_si"
                or element.get("name") != "mechanics_elements_si"
                or node.get("time") != time or element.get("time") != time):
            raise ValueError("Unaligned affine record step, time, or field label")
        nodes = np.asarray(node.get("values"), dtype=float)
        elements = np.asarray(element.get("values"), dtype=float)
        if (nodes.shape != (len(mesh.nodes_m), 9)
                or elements.shape != (len(mesh.tet10_indices), 8)
                or not np.isfinite(nodes).all() or not np.isfinite(elements).all()):
            raise ValueError("Complete finite affine node and element fields required")
        rows.append(_readout_state(mesh, nodes, elements, time, declared))
    return {"case_id": case_id, "passed": bool(all(row["passed"] for row in rows)),
            "states": rows, "scope": "Supplied complete non-patient affine fields only; no native provenance, solver residual, physical validation, or clinical evidence."}


def check_affine_readout(node_text: str, element_text: str,
                         case_id: str = CASE_ID) -> dict:
    """Parse complete saved text streams, then apply the independent readout."""
    if (not isinstance(node_text, str) or not isinstance(element_text, str)
            or len(node_text) + len(element_text) > output.MAX_TOTAL_CHARS):
        raise ValueError("Combined affine text exceeds declared active-output bound")
    declared = design.validate_declaration()
    case = _case(declared, case_id)
    mesh = deck.build_mesh(case["n"])
    times = tuple(k * case.get("step_size", declared["numerical_gauge"]["step_size"])
                  for k in range(case.get("time_steps", declared["numerical_gauge"]["time_steps"]) + 1))
    nodes = output.parse_data_log(node_text, expected_times=times,
                                  item_count=len(mesh.nodes_m), field_count=9,
                                  record_name="mechanics_nodes_si")
    elements = output.parse_data_log(element_text, expected_times=times,
                                     item_count=len(mesh.tet10_indices), field_count=8,
                                     record_name="mechanics_elements_si")
    return check_parsed_affine_readout(nodes, elements, case_id=case_id)
