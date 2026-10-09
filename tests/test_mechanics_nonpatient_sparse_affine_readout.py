"""Synthetic exact-field and adversarial controls; never a native cube result."""
from copy import deepcopy

import numpy as np
import pytest

from scripts import mechanics_nonpatient_sparse_affine_readout as readout
from scripts import mechanics_nonpatient_sparse_deck as deck
from scripts import mechanics_nonpatient_sparse_feasibility as design


def _exact_records(case_id: str = "n5_affine") -> tuple[list[dict], list[dict]]:
    declared = design.validate_declaration()
    case = next(row for row in declared["cases"] if row["id"] == case_id)
    mesh = deck.build_mesh(case["n"])
    X = mesh.nodes_m
    final_F = np.asarray(declared["loads"]["affine"]["final_F"])
    nodes, elements = [], []
    for step in range(5):
        time = step / 4
        F = np.eye(3) + time * (final_F - np.eye(3))
        oracle = design.affine_oracle(F)
        u = X @ (F - np.eye(3)).T
        reaction = np.zeros_like(X)
        P = np.asarray(oracle["first_Piola_Pa"])
        # Synthetic exact boundary traction integration for a constant P:
        # each planar square face gives one quarter of its force to each
        # corner. This is a software control, not a simulated FEBio reaction.
        for axis in range(3):
            for bound, normal_sign in ((0., -1.), (.01, 1.)):
                on_face = np.flatnonzero(np.isclose(X[:, axis], bound))
                other_axes = [i for i in range(3) if i != axis]
                candidate = X[on_face][:, other_axes]
                corners = on_face[np.all(np.isclose(candidate, 0.) |
                                         np.isclose(candidate, .01), axis=1)]
                assert len(corners) == 4
                normal = np.eye(3)[axis] * normal_sign
                reaction[corners] -= .01**2 / 4 * (P @ normal)
        nodes.append({"step": step, "time": time, "name": "mechanics_nodes_si",
                      "values": np.column_stack((X + u, u, reaction))})
        sigma = np.asarray(oracle["stress_Pa"])
        primitive = np.array([sigma[0, 0], sigma[1, 1], sigma[2, 2],
                              sigma[0, 1], sigma[1, 2], sigma[0, 2],
                              oracle["J"], oracle["energy_J"] / .01**3])
        elements.append({"step": step, "time": time, "name": "mechanics_elements_si",
                         "values": np.tile(primitive, (len(mesh.tet10_indices), 1))})
    return nodes, elements


def _text(records: list[dict]) -> str:
    pieces = []
    for row in records:
        pieces.append(f"*Step  = {row['step']}\n*Time  = {row['time']:.12g}\n*Data  = {row['name']}\n")
        for number, values in enumerate(row["values"], 1):
            pieces.append(str(number) + "," + ",".join(format(v, ".12g") for v in values) + "\n")
    return "".join(pieces)


def test_exact_synthetic_affine_fields_pass_each_state_and_text_roundtrip():
    nodes, elements = _exact_records()
    direct = readout.check_parsed_affine_readout(nodes, elements)
    text = readout.check_affine_readout(_text(nodes), _text(elements))
    for result in (direct, text):
        assert result["passed"], result
        assert [row["time"] for row in result["states"]] == [0., .25, .5, .75, 1.]
        assert all(row["passed"] for row in result["states"])
        assert result["states"][-1]["reconstructed_energy_J"] == pytest.approx(
            result["states"][-1]["oracle_energy_J"], abs=1e-12)
        assert result["states"][-1]["boundary_virtual_work_J"] == pytest.approx(
            result["states"][-1]["oracle_boundary_virtual_work_J"], abs=1e-12)
        assert "no native provenance" in result["scope"]


def test_n9_generated_affine_fields_and_text_roundtrip():
    nodes, elements = _exact_records("n9_affine")
    direct = readout.check_parsed_affine_readout(nodes, elements, case_id="n9_affine")
    text = readout.check_affine_readout(_text(nodes), _text(elements), case_id="n9_affine")
    for result in (direct, text):
        assert result["case_id"] == "n9_affine"
        assert result["passed"], result["states"][-1]["errors"]
        assert [state["time"] for state in result["states"]] == [0., .25, .5, .75, 1.]
    with pytest.raises(ValueError, match="Complete finite"):
        readout.check_parsed_affine_readout(*_exact_records(), case_id="n9_affine")
    with pytest.raises(ValueError, match="Incomplete FEBio"):
        readout.check_affine_readout(_text(nodes), _text(elements).rsplit("\n", 2)[0] + "\n",
                                     case_id="n9_affine")


def test_n13_generated_affine_fields_pass_without_native_provenance():
    nodes, elements = _exact_records("n13_affine")
    result = readout.check_parsed_affine_readout(nodes, elements, case_id="n13_affine")
    assert result["case_id"] == "n13_affine" and result["passed"]
    assert len(result["states"]) == 5


@pytest.mark.parametrize("bad", ["n5_nonuniform", "n9_nonuniform", "n13_nonuniform",
                                    "n13_nonuniform_half_step", "n11_affine", "n9_Affine"])
def test_readout_rejects_wrong_or_undeclared_case(bad):
    with pytest.raises(ValueError, match="Undeclared affine"):
        readout.check_parsed_affine_readout([], [], case_id=bad)
    with pytest.raises(ValueError, match="Undeclared affine"):
        readout.check_affine_readout("", "", case_id=bad)


@pytest.mark.parametrize("target,column,increment,check", [
    ("nodes", 3, 1e-6, "all_nodes_affine"),
    ("elements", 0, 1., "logged_stress_matches_actual"),
    ("elements", 6, .01, "logged_J_matches_actual"),
    ("elements", 7, .1, "logged_density_matches_actual"),
])
def test_corrupted_field_or_logged_primitive_fails(target, column, increment, check):
    nodes, elements = deepcopy(_exact_records())
    selected = nodes if target == "nodes" else elements
    selected[-1]["values"][0, column] += increment
    result = readout.check_parsed_affine_readout(nodes, elements)
    assert not result["passed"]
    assert not result["states"][-1]["checks"][check]


def test_self_consistent_wrong_interior_deformation_fails_actual_oracle():
    nodes, elements = deepcopy(_exact_records())
    mesh = deck.build_mesh(5)
    interior = int(np.flatnonzero(np.all(np.isclose(mesh.nodes_m, [.005, .005, .005]), axis=1))[0])
    nodes[-1]["values"][interior, 0] += 1e-5
    nodes[-1]["values"][interior, 3] += 1e-5
    result = readout.check_parsed_affine_readout(nodes, elements)
    assert not result["passed"]
    assert result["states"][-1]["checks"]["position_displacement_consistency"]
    assert not result["states"][-1]["checks"]["all_nodes_affine"]
    assert not result["states"][-1]["checks"]["actual_stress_matches_oracle"]


def test_wrong_reaction_sign_and_zero_reactions_fail_boundary_work():
    nodes, elements = deepcopy(_exact_records())
    nodes[-1]["values"][:, 6:] *= -1
    result = readout.check_parsed_affine_readout(nodes, elements)
    assert not result["states"][-1]["checks"]["signed_boundary_work_matches_oracle"]
    nodes[-1]["values"][:, 6:] = 0
    result = readout.check_parsed_affine_readout(nodes, elements)
    assert not result["states"][-1]["checks"]["signed_boundary_work_matches_oracle"]


def test_canceling_logged_density_errors_do_not_hide_in_total_energy():
    nodes, elements = deepcopy(_exact_records())
    elements[-1]["values"][0, 7] += 1.
    elements[-1]["values"][1, 7] -= 1.
    result = readout.check_parsed_affine_readout(nodes, elements)
    final = result["states"][-1]
    assert final["checks"]["integrated_logged_energy_matches_actual"]
    assert not final["checks"]["logged_density_matches_actual"]
    assert not result["passed"]


def test_incomplete_or_unaligned_parsed_records_fail_closed():
    nodes, elements = deepcopy(_exact_records())
    with pytest.raises(ValueError, match="Complete initial"):
        readout.check_parsed_affine_readout(nodes[:-1], elements)
    nodes[-1]["time"] = .99
    with pytest.raises(ValueError, match="Unaligned"):
        readout.check_parsed_affine_readout(nodes, elements)
    nodes, elements = deepcopy(_exact_records())
    elements[-1]["values"] = elements[-1]["values"][:-1]
    with pytest.raises(ValueError, match="Complete finite"):
        readout.check_parsed_affine_readout(nodes, elements)


def test_text_parser_rejects_missing_element_and_nan():
    nodes, elements = _exact_records()
    node_text, element_text = _text(nodes), _text(elements)
    with pytest.raises(ValueError, match="Incomplete FEBio"):
        readout.check_affine_readout(node_text, element_text.rsplit("\n", 2)[0] + "\n")
    with pytest.raises(ValueError, match="Nonfinite"):
        readout.check_affine_readout(node_text.replace("1,0,0,0,0,0,0,0,0,0", "1,nan,0,0,0,0,0,0,0,0", 1), element_text)
