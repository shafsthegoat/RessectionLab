"""Generated-only readout controls plus a committed tiny native sign witness."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from scripts import mechanics_nonpatient_sparse_deck as deck
from scripts import mechanics_nonpatient_sparse_feasibility as design
from scripts import mechanics_nonpatient_sparse_nonuniform_readout as readout
from scripts import mechanics_nonpatient_sparse_output as output
from scripts import mechanics_patient_constraints as fixture


ROOT = Path(__file__).resolve().parents[1]


def _zero_records(case_id="n5_nonuniform"):
    case = next(row for row in design.validate_declaration()["cases"] if row["id"] == case_id)
    mesh = deck.build_mesh(case["n"])
    steps = case.get("time_steps", 4)
    dt = case.get("step_size", .25)
    nodes, elements = [], []
    for step in range(steps + 1):
        nodes.append({"step": step, "time": step * dt, "name": "mechanics_nodes_si",
                      "values": np.column_stack((mesh.nodes_m, np.zeros((len(mesh.nodes_m), 6))))})
        primitive = np.array([0., 0., 0., 0., 0., 0., 1., 0.])
        elements.append({"step": step, "time": step * dt, "name": "mechanics_elements_si",
                         "values": np.tile(primitive, (len(mesh.tet10_indices), 1))})
    return mesh, nodes, elements


def _data_text(records):
    parts = []
    for row in records:
        parts.append(f"*Step  = {row['step']}\n*Time  = {row['time']:.12g}\n*Data  = {row['name']}\n")
        parts.extend(f"{index}," + ",".join(f"{value:.12g}" for value in values) + "\n"
                     for index, values in enumerate(row["values"], 1))
    return "".join(parts)


def _solver_text(mesh):
    header = ("dtol .............................................. : 1e-10\n"
              "etol .............................................. : 1e-10\n"
              "rtol .............................................. : 1e-10\n"
              "min_residual ...................................... : 1e-20\n"
              f"Number of nodes ................................ : {len(mesh.nodes_m)}\n"
              f"Number of solid elements ....................... : {len(mesh.tet10_indices)}\n")
    steps = "".join(
        f"Nonlinear solution status: time= {step/4:.12g}\n"
        "convergence norms :     INITIAL         CURRENT         REQUIRED\n"
        "residual 1e-4 1e-15 1e-14\n"
        f"------- converged at time : {step/4:.12g}\n"
        for step in range(1, 5))
    return header + steps + "N O R M A L   T E R M I N A T I O N\n"


def test_complete_generated_identity_fields_pass_parsed_constitutive_and_local_checks():
    _, nodes, elements = _zero_records()
    result = readout.check_parsed_nonuniform_readout("n5_nonuniform", nodes, elements)
    assert result["passed"]
    assert len(result["states"]) == 5
    assert result["states"][-1]["sample_owner_counts"] == [6, 2, 2, 2, 2, 2, 2]
    assert result["states"][-1]["sample_displacement_m"] == [[0., 0., 0.]] * 7
    assert result["states"][-1]["local_force_limit_N"] == 1e-8
    assert "no solver or native provenance" in result["scope"]


@pytest.mark.parametrize("case_id,states", [("n9_nonuniform", 5),
                                           ("n13_nonuniform", 5),
                                           ("n13_nonuniform_half_step", 9)])
def test_all_mesh_levels_and_half_step_complete_schedules(case_id, states):
    _, nodes, elements = _zero_records(case_id)
    result = readout.check_parsed_nonuniform_readout(case_id, nodes, elements)
    assert result["passed"]
    assert len(result["states"]) == states
    assert result["states"][-1]["time"] == 1.


def test_million_pascal_logged_stress_and_density_mutation_is_rejected():
    _, nodes, elements = _zero_records()
    elements[-1]["values"][749, 0] = 1e6
    elements[-1]["values"][749, 7] = 1e6
    result = readout.check_parsed_nonuniform_readout("n5_nonuniform", nodes, elements)
    assert not result["passed"]
    assert not result["states"][-1]["checks"]["logged_stress_matches_actual"]
    assert not result["states"][-1]["checks"]["logged_density_matches_actual"]


def test_text_entry_combines_existing_solver_boundary_and_new_material_gates():
    mesh, nodes, elements = _zero_records()
    node_text, element_text, solver = _data_text(nodes), _data_text(elements), _solver_text(mesh)
    result = readout.check_nonuniform_readout("n5_nonuniform", node_text, element_text, solver)
    assert not result["existing_output_checks_passed"]  # loaded zero field violates top motion
    assert all(row["passed"] for row in result["states"])
    bad_element = element_text.replace("750,0,0,0,0,0,0,1,0", "750,1000000,0,0,0,0,0,1,1000000", 1)
    assert bad_element != element_text
    corrupted = readout.check_nonuniform_readout("n5_nonuniform", node_text, bad_element, solver)
    assert not corrupted["states"][0]["checks"]["logged_stress_matches_actual"]
    assert not corrupted["states"][0]["checks"]["logged_density_matches_actual"]
    with pytest.raises(ValueError, match="Incomplete FEBio"):
        readout.check_nonuniform_readout("n5_nonuniform", node_text,
                                         element_text.rsplit("\n", 2)[0] + "\n", solver)


def test_balanced_wrong_local_reactions_fail_even_when_global_force_and_moment_cancel():
    mesh, nodes, elements = _zero_records()
    X = mesh.nodes_m
    updates = [((0., 0., .01), 2e-7), ((.01, 0., .01), -2e-7),
               ((0., .01, .01), -2e-7), ((.01, .01, .01), 2e-7)]
    for point, force in updates:
        index = int(np.flatnonzero(np.all(X == point, axis=1))[0])
        nodes[-1]["values"][index, 6] = force
    result = readout.check_parsed_nonuniform_readout("n5_nonuniform", nodes, elements)
    state = result["states"][-1]
    assert not result["passed"]
    assert not state["checks"]["local_signed_constraint_reactions"]
    np.testing.assert_allclose(nodes[-1]["values"][:, 6:9].sum(axis=0), 0., atol=1e-20)
    np.testing.assert_allclose(np.cross(X, nodes[-1]["values"][:, 6:9]).sum(axis=0), 0., atol=1e-20)


@pytest.mark.parametrize("n,owners", [(5, [6, 2, 2, 2, 2, 2, 2]),
                                        (9, [6, 2, 2, 2, 2, 2, 2]),
                                        (13, [6, 2, 2, 2, 2, 2, 2])])
def test_fixed_physical_samples_reproduce_quadratic_and_affine_fields_on_all_levels(n, owners):
    mesh = deck.build_mesh(n)
    points = np.asarray(design.validate_declaration()["loads"]["nonuniform"]["sample_points_m"])
    locators = readout._sample_locator(mesh, points)
    assert [len(ids) for ids, _ in locators] == owners
    def polynomial(X):
        z = X[:, 2]
        return np.column_stack((.0003*z/.01 + 4e-5*z*(.01-z)/.01,
                                3e-5*X[:, 0]*X[:, 1]/.01,
                                -.0002*z/.01 + 2e-5*z*(.01-z)/.01))
    values, observed_owners, disagreement = readout._samples(mesh, polynomial(mesh.nodes_m), points, locators)
    assert observed_owners == owners
    assert max(disagreement) < 2e-18
    np.testing.assert_allclose(values, polynomial(points), rtol=0, atol=2e-18)
    affine = mesh.nodes_m @ np.array([[.01,.02,.03],[0.,-.01,0.],[0.,0.,.02]]).T
    values, _, _ = readout._samples(mesh, affine, points, locators)
    np.testing.assert_allclose(values,
                               points @ np.array([[.01,.02,.03],[0.,-.01,0.],[0.,0.,.02]]).T,
                               rtol=0, atol=2e-18)


def test_nonuniform_quadratic_g8_primitives_and_weighted_energy_reconstruct():
    mesh, nodes, elements = _zero_records()
    X, E = mesh.nodes_m, mesh.tet10_indices
    z = X[:, 2]
    u = np.column_stack((.0003*z/.01 + 1e-3*z*(.01-z)/.01,
                         np.zeros(len(X)),
                         -.0002*z/.01 + 1e-3*z*(.01-z)/.01))
    nodes[-1]["values"][:, :3] = X + u
    nodes[-1]["values"][:, 3:6] = u
    q, weights = fixture.gauss_rule()
    derivatives = np.stack([fixture.shape(point)[1] for point in q])
    ref = np.einsum("eni,qnj->eqij", X[E], derivatives)
    xyz = np.einsum("qn,eni->eqi", np.stack([fixture.shape(point)[0] for point in q]), X[E])
    zq = xyz[..., 2]
    F = np.broadcast_to(np.eye(3), (len(E), len(q), 3, 3)).copy()
    F[..., 0, 2] += .0003/.01 + 1e-3*(.01 - 2*zq)/.01
    F[..., 2, 2] += -.0002/.01 + 1e-3*(.01 - 2*zq)/.01
    J, W, sigma = fixture.response(F)  # independently established fixture law
    element = elements[-1]["values"]
    element[:, :6] = np.stack((sigma[...,0,0], sigma[...,1,1], sigma[...,2,2],
                               sigma[...,0,1], sigma[...,1,2], sigma[...,0,2]), axis=-1).mean(axis=1)
    element[:, 6] = J.mean(axis=1)
    element[:, 7] = W.mean(axis=1)
    result = readout.check_parsed_nonuniform_readout("n5_nonuniform", nodes, elements)
    state = result["states"][-1]
    assert state["checks"]["logged_stress_matches_actual"]
    assert state["checks"]["logged_J_matches_actual"]
    assert state["checks"]["logged_density_matches_actual"]
    expected_energy = float(np.sum(W * np.linalg.det(ref) * weights))
    assert state["reconstructed_energy_J"] == pytest.approx(expected_energy, rel=0, abs=1e-14)
    naive_logged_density_times_volume = float(np.sum(element[:, 7] *
                                                     np.sum(np.linalg.det(ref)*weights, axis=1)))
    assert naive_logged_density_times_volume != pytest.approx(expected_energy, rel=0, abs=1e-14)
    assert not state["checks"]["local_free_equilibrium"]  # invented field is not a solved state
    assert not result["passed"]


def test_committed_native_tet10_affine_record_confirms_reaction_sign():
    """An older committed native tiny control verifies the sign without FEBio."""
    path = ROOT / "artifacts/mechanics-patient-constraints-run-v1/attempt-01/tet10_affine"
    native_nodes = output.parse_data_log((path / "tet10_affine.nodes.log").read_text(),
                                         expected_times=(0., .25, .5, .75, 1.),
                                         item_count=27, field_count=9,
                                         record_name="mechanics_nodes_si")[-1]["values"]
    native_elements = output.parse_data_log((path / "tet10_affine.elements.log").read_text(),
                                            expected_times=(0., .25, .5, .75, 1.),
                                            item_count=6, field_count=8,
                                            record_name="mechanics_elements_si")[-1]["values"]
    mesh = deck.build_mesh(1, software_control=True)
    rows = [int(np.flatnonzero(np.all(np.isclose(native_nodes[:, :3] - native_nodes[:, 3:6], x,
                                               atol=1e-14, rtol=0), axis=1))[0]) for x in mesh.nodes_m]
    nodal = native_nodes[rows]
    declared = design.validate_declaration()
    locators = readout._sample_locator(mesh,
                                       np.asarray(declared["loads"]["nonuniform"]["sample_points_m"]))
    points = np.asarray(declared["loads"]["nonuniform"]["sample_points_m"])
    state = readout._state(mesh, nodal, native_elements, 1., declared, points, locators)
    assert state["checks"]["local_signed_constraint_reactions"]
    assert state["errors"]["maximum_constrained_force_residual_N"] < 1e-9
    opposite = nodal.copy()
    opposite[:, 6:9] *= -1
    bad = readout._state(mesh, opposite, native_elements, 1., declared, points, locators)
    assert not bad["checks"]["local_signed_constraint_reactions"]


def test_malformed_or_unaligned_records_fail_before_readout():
    _, nodes, elements = _zero_records()
    with pytest.raises(ValueError, match="Complete nonuniform"):
        readout.check_parsed_nonuniform_readout("n5_nonuniform", nodes[:-1], elements)
    nodes = deepcopy(nodes)
    nodes[-1]["time"] = .99
    with pytest.raises(ValueError, match="Unaligned"):
        readout.check_parsed_nonuniform_readout("n5_nonuniform", nodes, elements)
    with pytest.raises(ValueError, match="Undeclared"):
        readout.check_parsed_nonuniform_readout("n5_affine", nodes, elements)


def test_nonpositive_deformation_jacobian_fails_closed():
    F = np.diag([1., 1., -1.])[None, None]
    with pytest.raises(ValueError, match="Nonpositive"):
        readout._response(F, 1000., 1000.*29/3)
