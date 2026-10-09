"""Synthetic text fixtures for the unreleased non-patient cube output parser."""
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

from scripts import mechanics_nonpatient_sparse_deck as deck
from scripts import mechanics_nonpatient_sparse_feasibility as design
from scripts import mechanics_nonpatient_sparse_output as output


ROOT = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=2)
def _synthetic_case(case_id: str) -> tuple[str, str, str]:
    """Software-only affine fields with fake logs; never claim a native solve."""
    declaration = design.validate_declaration()
    mesh = deck.build_mesh(5)
    X = mesh.nodes_m
    E = mesh.tet10_indices
    node_parts, element_parts = [], []
    for step in range(5):
        time = step / 4
        if case_id == "n5_affine":
            F = np.eye(3) + time * (np.asarray(declaration["loads"]["affine"]["final_F"]) - np.eye(3))
        else:
            top = np.asarray(declaration["loads"]["nonuniform"]["top_displacement_m"])
            F = np.eye(3) + time * np.outer(top / .01, [0., 0., 1.])
        moved = X @ F.T
        node_parts.extend((f"*Step  = {step}\n*Time  = {time:.12g}\n*Data  = mechanics_nodes_si\n",
                           "".join(f"{i}," + ",".join(f"{v:.12g}" for v in np.r_[x, x-X[i-1], [0., 0., 0.]]) + "\n"
                                   for i, x in enumerate(moved, 1))))
        element_parts.extend((f"*Step  = {step}\n*Time  = {time:.12g}\n*Data  = mechanics_elements_si\n",
                              "".join(f"{i},0,0,0,0,0,0,{np.linalg.det(F):.12g},0\n"
                                      for i in range(1, len(E)+1))))
    log = ("dtol .............................................. : 1e-10\n"
           "etol .............................................. : 1e-10\n"
           "rtol .............................................. : 1e-10\n"
           "min_residual ...................................... : 1e-20\n"
           f"Number of nodes ................................ : {len(X)}\n"
           f"Number of solid elements ....................... : {len(E)}\n")
    for step in range(1,5):
        time = step / 4
        log += (f"Nonlinear solution status: time= {time:.12g}\n"
                "convergence norms :     INITIAL         CURRENT         REQUIRED\n"
                "residual 1e-4 1e-15 1e-14\n"
                f"------- converged at time : {time:.12g}\n")
    log += "N O R M A L   T E R M I N A T I O N\n"
    return "".join(node_parts), "".join(element_parts), log


@pytest.mark.parametrize("case_id", ["n5_affine", "n5_nonuniform"])
def test_complete_synthetic_output_passes_only_saved_text_gates(case_id):
    result = output.check_case_outputs(case_id, *_synthetic_case(case_id))
    assert result["passed"]
    assert len(result["states"]) == 5
    assert [row["time"] for row in result["states"]] == [0., .25, .5, .75, 1.]
    assert result["states"][-1]["minimum_sampled_deformation_J"] > .2
    assert "no native provenance" in result["scope"]


def test_data_parser_rejects_missing_step_reordered_id_nan_and_label():
    nodes, elements, log = _synthetic_case("n5_nonuniform")
    with pytest.raises(ValueError, match="Missing FEBio"):
        output.check_case_outputs("n5_nonuniform", nodes[:nodes.index("*Step  = 4")], elements, log)
    with pytest.raises(ValueError, match="row ID"):
        output.check_case_outputs("n5_nonuniform", nodes.replace("\n2,", "\n3,", 1), elements, log)
    with pytest.raises(ValueError, match="Nonfinite"):
        output.check_case_outputs("n5_nonuniform", nodes.replace("1,0,0,0,0,0,0,0,0,0", "1,nan,0,0,0,0,0,0,0,0", 1), elements, log)
    with pytest.raises(ValueError, match="label"):
        output.check_case_outputs("n5_nonuniform", nodes.replace("mechanics_nodes_si", "other", 1), elements, log)


def test_parser_rejects_wrong_step_time_and_incomplete_element_rows():
    nodes, elements, log = _synthetic_case("n5_nonuniform")
    with pytest.raises(ValueError, match="time differs"):
        output.check_case_outputs("n5_nonuniform", nodes.replace("*Time  = 0.25", "*Time  = 0.26", 1), elements, log)
    with pytest.raises(ValueError, match="Incomplete FEBio"):
        output.check_case_outputs("n5_nonuniform", nodes, elements.replace("750,0,0,0,0,0,0,1,0\n", "", 1), log)
    with pytest.raises(ValueError, match="Undeclared"):
        output.check_case_outputs("n17_affine", nodes, elements, log)


@pytest.mark.parametrize("change,matched", [
    (lambda s: s.replace("N O R M A L   T E R M I N A T I O N", "E R R O R   T E R M I N A T I O N"), "normal-termination"),
    (lambda s: s.replace("------- converged at time : 1\n", ""), "Incomplete FEBio convergence"),
    (lambda s: s.replace("residual 1e-4 1e-15 1e-14", "residual 1e-4 1e-8 1e-14", 1), None),
    # 1e-12 required from 1e-4 initial would silently admit the wrong 1e-8 rule.
    (lambda s: s.replace("residual 1e-4 1e-15 1e-14", "residual 1e-4 1e-15 1e-12", 1), "rule differs"),
    (lambda s: s.replace("residual 1e-4 1e-15 1e-14", "residual 1e-4 1e-15 1e-14\nresidual 1e-4 1e-15 1e-14", 1), "duplicate"),
    (lambda s: s.replace("------- converged at time : 0.25", "Nonlinear solution status: time= 0.25\n------- converged at time : 0.25", 1), "Unbound"),
    (lambda s: s.replace("Number of nodes ................................ : 1331", "Number of nodes ................................ : 1330"), "mesh counts"),
    (lambda s: s.replace("dtol .............................................. : 1e-10", "dtol .............................................. : 1e-8"), "dtol/etol/rtol"),
    (lambda s: s.replace("etol .............................................. : 1e-10", "etol .............................................. : 1e-8"), "dtol/etol/rtol"),
    (lambda s: s.replace("rtol .............................................. : 1e-10", "rtol .............................................. : 1e-8"), "dtol/etol/rtol"),
    (lambda s: s.replace("min_residual ...................................... : 1e-20", "min_residual ...................................... : 1e-10"), "min_residual"),
    (lambda s: s.replace("residual 1e-4 1e-15 1e-14", "Nonlinear solution status: time= 0.25\nresidual 1e-4 1e-15 1e-14", 1), "Unfinished"),
    (lambda s: s.replace("N O R M A L   T E R M I N A T I O N\n", "", 1).replace("Nonlinear solution status: time= 0.25", "N O R M A L   T E R M I N A T I O N\nNonlinear solution status: time= 0.25", 1), "normal-termination"),
    (lambda s: s.replace("N O R M A L   T E R M I N A T I O N", "E R R O R: sparse matrix factorization failed\nN O R M A L   T E R M I N A T I O N"), "failure evidence"),
    (lambda s: s.replace("------- converged at time : 1", "not converged at time : 1"), "failure evidence"),
    (lambda s: s.replace("------- converged at time : 1", "WARNING: already converged at time : 1"), "ambiguous FEBio convergence"),
    (lambda s: s.replace("convergence norms :     INITIAL         CURRENT         REQUIRED\n", "", 1), "residual norms"),
    (lambda s: s.replace("------- converged at time : 0.25", "converged at time : 0.25\n------- converged at time : 0.25", 1), "ambiguous FEBio convergence"),
    (lambda s: s.replace("rtol .............................................. : 1e-10", "WARNING rtol .............................................. : 1e-8\nrtol .............................................. : 1e-10", 1), "Ambiguous FEBio solver-tolerance"),
    (lambda s: s.replace("residual 1e-4 1e-15 1e-14", "WARNING residual 1e-4 1e-15 1e-14\nresidual 1e-4 1e-15 1e-14", 1), "Ambiguous FEBio residual-like"),
])
def test_solver_tokens_fail_closed(change, matched):
    nodes, elements, log = _synthetic_case("n5_nonuniform")
    if matched:
        with pytest.raises(ValueError, match=matched):
            output.check_case_outputs("n5_nonuniform", nodes, elements, change(log))
    else:
        assert not output.check_case_outputs("n5_nonuniform", nodes, elements, change(log))["passed"]


def test_mechanics_checks_catch_boundary_jacobian_and_equilibrium_failures():
    nodes, elements, log = _synthetic_case("n5_nonuniform")
    # A top-face displacement is declared, so an altered saved motion fails.
    moved = nodes.replace("1331,0.0103,0.01,0.0098,0.0003,0,-0.0002,0,0,0",
                          "1331,0.0103,0.01,0.0098,0.0001,0,-0.0002,0,0,0")
    assert moved != nodes
    assert not output.check_case_outputs("n5_nonuniform", moved, elements, log)["passed"]
    altered_J = elements.replace("750,0,0,0,0,0,0,0.98,0", "750,0,0,0,0,0,0,0.1,0", 1)
    assert altered_J != elements
    assert not output.check_case_outputs("n5_nonuniform", nodes, altered_J, log)["passed"]
    altered_force = nodes.replace("1331,0.0103,0.01,0.0098,0.0003,0,-0.0002,0,0,0",
                                  "1331,0.0103,0.01,0.0098,0.0003,0,-0.0002,0.1,0,0")
    assert altered_force != nodes
    result = output.check_case_outputs("n5_nonuniform", altered_force, elements, log)
    assert not result["passed"]
    assert not result["states"][-1]["checks"]["force_balance"]


def test_balanced_self_canceling_free_reactions_still_fail():
    nodes, elements, log = _synthetic_case("n5_nonuniform")
    mesh = deck.build_mesh(5)
    first = int(np.flatnonzero(np.all(np.isclose(mesh.nodes_m, [.004, .004, .004]), axis=1))[0]) + 1
    second = int(np.flatnonzero(np.all(np.isclose(mesh.nodes_m, [.006, .004, .004]), axis=1))[0]) + 1
    lines = nodes.splitlines()
    for index, line in enumerate(lines):
        if line.startswith(f"{first},"):
            parts = line.split(",")
            parts[-3] = "0.001"
            lines[index] = ",".join(parts)
        if line.startswith(f"{second},"):
            parts = line.split(",")
            parts[-3] = "-0.001"
            lines[index] = ",".join(parts)
    result = output.check_case_outputs("n5_nonuniform", "\n".join(lines) + "\n", elements, log)
    assert not result["passed"]
    assert result["states"][0]["checks"]["force_balance"]
    assert result["states"][0]["checks"]["moment_balance"]
    assert not result["states"][0]["checks"]["free_node_reactions"]


def test_balanced_initial_constrained_reactions_still_fail_no_preload_gate():
    nodes, elements, log = _synthetic_case("n5_nonuniform")
    mesh = deck.build_mesh(5)
    corners = [((0., 0., 0.), .001), ((.01, 0., 0.), -.001),
               ((0., .01, 0.), .001), ((.01, .01, 0.), -.001)]
    edits = {int(np.flatnonzero(np.all(np.isclose(mesh.nodes_m, p), axis=1))[0])+1: force
             for p, force in corners}
    lines = nodes.splitlines()
    end_initial = lines.index("*Step  = 1")
    for index in range(3, end_initial):
        parts = lines[index].split(",")
        item_id = int(parts[0])
        if item_id in edits:
            parts[-3] = str(edits[item_id])
            lines[index] = ",".join(parts)
    result = output.check_case_outputs("n5_nonuniform", "\n".join(lines)+"\n", elements, log)
    initial = result["states"][0]
    assert not result["passed"]
    assert initial["checks"]["force_balance"]
    assert initial["checks"]["moment_balance"]
    assert not initial["checks"]["initial_unloaded_state"]
    assert initial["errors"]["initial_unloaded"]["reaction_N"] == .001


@pytest.mark.parametrize("replacement", [
    "1,1,0,0,0,0,0,1,0",      # fabricated initial stress (Pa)
    "1,0,0,0,0,0,0,1,100",    # fabricated initial energy density (Pa)
    "1,0,0,0,0,0,0,0.99,0",   # fabricated initial deformation J
])
def test_initial_element_must_be_unloaded(replacement):
    nodes, elements, log = _synthetic_case("n5_nonuniform")
    changed = elements.replace("1,0,0,0,0,0,0,1,0", replacement, 1)
    assert changed != elements
    result = output.check_case_outputs("n5_nonuniform", nodes, changed, log)
    assert not result["passed"]
    assert not result["states"][0]["checks"]["initial_unloaded_state"]


def test_initial_internal_motion_fails_even_when_x_equals_X_plus_u():
    nodes, elements, log = _synthetic_case("n5_nonuniform")
    mesh = deck.build_mesh(5)
    item_id = int(np.flatnonzero(np.all(np.isclose(mesh.nodes_m, [.005,.005,.005]), axis=1))[0]) + 1
    lines = nodes.splitlines()
    for index in range(3, lines.index("*Step  = 1")):
        if lines[index].startswith(f"{item_id},"):
            parts = lines[index].split(",")
            parts[1] = str(float(parts[1]) + 1e-5)
            parts[4] = "1e-5"
            lines[index] = ",".join(parts)
            break
    result = output.check_case_outputs("n5_nonuniform", "\n".join(lines)+"\n", elements, log)
    assert not result["passed"]
    assert not result["states"][0]["checks"]["initial_unloaded_state"]


def test_parser_bounds_are_explicit():
    with pytest.raises(ValueError, match="line exceeds bound"):
        output.parse_data_log("*Step = 0\n" + "x" * (output.MAX_LINE_CHARS + 1),
                              expected_times=(0.,), item_count=1, field_count=1,
                              record_name="mechanics_nodes_si")
    with pytest.raises(ValueError, match="ASCII"):
        output.parse_data_log("é", expected_times=(0.,), item_count=1,
                              field_count=1, record_name="mechanics_nodes_si")


def test_previously_saved_native_tet10_control_matches_declared_text_grammar():
    directory = ROOT / "artifacts/mechanics-patient-constraints-run-v1/attempt-01/tet10_affine"
    expected = (0., .25, .5, .75, 1.)
    nodes = output.parse_data_log((directory / "tet10_affine.nodes.log").read_text(),
                                  expected_times=expected, item_count=27, field_count=9,
                                  record_name="mechanics_nodes_si")
    elements = output.parse_data_log((directory / "tet10_affine.elements.log").read_text(),
                                     expected_times=expected, item_count=6, field_count=8,
                                     record_name="mechanics_elements_si")
    solver = output.parse_solver_log((directory / "tet10_affine.log").read_text(),
                                     expected_times=expected, node_count=27, element_count=6)
    assert len(nodes) == len(elements) == 5
    assert solver["passed"] and len(solver["steps"]) == 4
