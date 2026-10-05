"""Numerical/XML/parser controls only; no FEBio process or patient data."""
import importlib.util
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location("mechanics_febio_verification", Path(__file__).resolve().parents[1]/"scripts/mechanics_febio_verification.py")
check = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(check)


def synthetic_logs(case_id, mutate=None):
    """Oracle-generated parser fixture, explicitly not a measured solver output."""
    node_text, elem_text = [], []
    solver_text = ["N O R M A L   T E R M I N A T I O N"]
    for step, time in enumerate(check.TIMES, 1):
        state = check.expected_state(case_id, time)
        if mutate: mutate(state, time)
        for output, key, name in ((node_text, "nodes", "mechanics_nodes_si"), (elem_text, "elements", "mechanics_elements_si")):
            output.extend((f"*Step = {step}", f"*Time = {time}", f"*Data = {name}"))
            output.extend(str(i)+","+",".join(format(float(v), ".12g") for v in row)
                          for i, row in enumerate(state[key], 1))
        solver_text.extend((f"Nonlinear solution status: time= {time}", " residual 1e-8 1e-26 1e-18"))
    return "\n".join(node_text), "\n".join(elem_text), "\n".join(solver_text)


def test_zero_and_translation_have_no_strain_energy_or_reactions():
    zero = check.expected_state("zero")
    translated = check.expected_state("translation")
    assert zero["response"]["J"] == 1
    assert zero["energy_J"] == translated["energy_J"] == 0
    np.testing.assert_array_equal(translated["nodes"][:, 6:], np.zeros((27, 3)))
    np.testing.assert_allclose(translated["nodes"][:, :3]-zero["nodes"][:, :3],
                               np.tile([.0003, -.0002, .0001], (27, 1)), atol=1e-18, rtol=0)


def test_simple_shear_closed_form_stress_energy_and_signed_face_center_force():
    state = check.expected_state("shear")
    mu, gamma = 1000., .1
    sigma = np.array([[2*mu*gamma**2/3, mu*gamma, 0],
                      [mu*gamma, -mu*gamma**2/3, 0], [0, 0, -mu*gamma**2/3]])
    np.testing.assert_allclose(state["response"]["sigma_Pa"], sigma, atol=1e-10, rtol=0)
    assert state["energy_J"] == pytest.approx(mu*gamma**2/2 * .01**3, abs=1e-18)
    # Node17 is the center of y=10mm face, integrated shape area=L²/4.
    assert state["nodes"][16, 6] == pytest.approx(-.0025, abs=1e-15)
    assert -state["nodes"][16, 6] > 0  # applied shear force is opposite raw Rx
    np.testing.assert_allclose(state["nodes"][:, 6:].sum(axis=0), 0, atol=1e-17)
    np.testing.assert_allclose(np.cross(state["nodes"][:, :3], state["nodes"][:, 6:]).sum(axis=0), 0, atol=1e-18)


def test_uniform_dilatation_is_pure_selected_volumetric_law_not_default_log_penalty():
    stretch = 1.02; J = stretch**3; K = 149000/3
    state = check.homogeneous_response(np.eye(3)*stretch, 1000.)
    expected_pressure = K/2*(J-1/J)
    np.testing.assert_allclose(state["sigma_Pa"], np.eye(3)*expected_pressure, rtol=1e-13, atol=1e-10)
    assert state["W_Pa"] == pytest.approx(K/4*(J**2-1-2*math.log(J)), abs=1e-10)
    assert abs(expected_pressure-K*math.log(J)/J) > 1


@pytest.mark.parametrize("case_id", list(check.cases()))
def test_decks_use_explicit_formulation_and_reaction_preserving_constraints(case_id):
    root = ET.fromstring(check.deck_xml(case_id))
    assert root.attrib == {"version": "4.0"}
    assert root.find("Module").get("type") == "solid"
    assert root.find("Control/solver/linear_solver").get("type") == "skyline"
    assert root.findtext("Control/solver/symmetric_stiffness") == "1"
    assert root.findtext("Control/solver/rtol") == "1e-10"
    assert root.find("MeshDomains/SolidDomain").get("type") == "three-field-solid"
    assert root.findtext("MeshDomains/SolidDomain/laugon") == "0"
    assert root.findtext("Material/material/pressure_model") == "1"
    assert float(root.findtext("Material/material/c1")) == 2*check.cases()[case_id]["mu_Pa"]
    assert root.find("Material/material").get("type") == "Ogden"
    assert len(root.findall("Mesh/Nodes/node")) == 27
    assert len(root.findall("Mesh/Elements/elem")) == 8
    boundaries = root.findall("Boundary/bc")
    assert len(boundaries) == 78
    assert all(bc.get("type") == "prescribed displacement" for bc in boundaries)
    assert all(bc.get("node_set") != "node_14" for bc in boundaries)
    assert all(bc.findtext("relative") == "0" for bc in boundaries)
    if case_id != "translation": assert any(float(bc.findtext("value")) == 0 for bc in boundaries)
    assert root.find("Output/logfile/node_data").get("data") == "x;y;z;ux;uy;uz;Rx;Ry;Rz"
    assert root.find("Output/logfile/element_data").get("data") == "sx;sy;sz;sxy;syz;sxz;J;sed"
    assert root.find("Output/logfile/node_data").get("format") is None
    assert not any(root.findall(tag) for tag in ("Contact", "Rigid", "Loads", "Initial"))


@pytest.mark.parametrize("case_id", list(check.cases()))
def test_oracle_parser_fixture_passes_all_physical_checks(case_id):
    result = check.check_outputs(case_id, *synthetic_logs(case_id))
    assert result["passed"], result["failures"]
    assert len(result["states"]) == 4


@pytest.mark.parametrize("fault", ["reaction_sign", "length_mm_as_m", "stress_kPa_as_Pa", "sed_per_current_volume", "wrong_J", "hidden_center_warp", "force_unbalance", "couple", "prescribed_zero_reactions_missing"])
def test_corrupted_physical_outputs_are_rejected(fault):
    def mutate(state, time):
        nodes, elements = state["nodes"], state["elements"]
        if fault == "reaction_sign": nodes[:, 6:] *= -1
        elif fault == "length_mm_as_m": nodes[:, :6] *= 1000
        elif fault == "stress_kPa_as_Pa": elements[:, :6] /= 1000
        elif fault == "sed_per_current_volume": elements[:, 7] /= elements[:, 6]
        elif fault == "wrong_J": elements[:, 6] = 1
        elif fault == "hidden_center_warp": nodes[13, 2] += .001; nodes[13, 5] += .001
        elif fault == "force_unbalance": nodes[0, 6] += .001
        elif fault == "couple": nodes[0, 6] += .001; nodes[-1, 6] -= .001
        else: nodes[:9, 6:] = 0
    result = check.check_outputs("finite_stretch", *synthetic_logs("finite_stretch", mutate))
    assert not result["passed"]


def test_actual_jacobian_detects_inversion_even_when_element_log_claims_valid_J():
    def mutate(state, time):
        state["nodes"][13, 2] += .02
    result = check.check_outputs("shear", *synthetic_logs("shear", mutate))
    assert min(row["minimum_actual_sampled_J"] for row in result["states"]) < 0
    assert any("positive_actual_sampled_jacobians" in value for value in result["failures"])


@pytest.mark.parametrize("fault", ["duplicate", "missing", "nan", "wrong_name", "wrong_columns", "missing_state", "duplicate_state", "bad_time", "wrong_step"])
def test_data_extraction_refuses_ambiguous_or_incomplete_records(fault):
    nodes, _, _ = synthetic_logs("zero")
    lines = nodes.splitlines()
    if fault == "duplicate": lines.insert(4, lines[3])
    elif fault == "missing": lines.pop(3)
    elif fault == "nan": lines[3] = lines[3].replace(",0,", ",nan,", 1)
    elif fault == "wrong_name": lines[2] = "*Data = unknown"
    elif fault == "wrong_columns": lines[3] += ",1"
    elif fault == "missing_state": lines = lines[:90]
    elif fault == "duplicate_state": lines += lines[:30]
    elif fault == "bad_time": lines[1] = "*Time = nan"
    else: lines[0] = "*Step = 2"
    with pytest.raises(ValueError):
        check.parse_data_log("\n".join(lines), field_count=9, record_name="mechanics_nodes_si", item_count=27)


@pytest.mark.parametrize("fault", ["abnormal", "missing", "nan", "unconverged", "negative_J"])
def test_termination_alone_is_not_residual_acceptance(fault):
    text = synthetic_logs("zero")[2]
    if fault == "abnormal": text = text.replace("N O R M A L", "A B N O R M A L")
    elif fault == "missing": text = text.replace(" residual 1e-8 1e-26 1e-18", "", 1)
    elif fault == "nan": text = text.replace("1e-26", "nan", 1)
    elif fault == "negative_J": text += "\nnegative jacobian detected"
    else:
        assert not check.check_solver_log(text.replace("1e-26", "1e-3"))["passed"]
        return
    with pytest.raises(ValueError): check.check_solver_log(text)


def test_actual_stiffness_scaling_pair_compares_all_states_and_keeps_sign():
    base = synthetic_logs("shear"); doubled = synthetic_logs("shear_double_stiffness")
    assert check.check_stiffness_scaling(base[0], base[1], doubled[0], doubled[1])["passed"]
    assert not check.check_stiffness_scaling(base[0], base[1], base[0], base[1])["passed"]


def test_bundle_is_fresh_deterministic_and_declares_arbitrary_SI_scale(tmp_path):
    manifest = check.write_bundle(tmp_path/"controls")
    assert manifest["units"] == {"length": "m", "force": "N", "stress": "Pa", "energy": "J"}
    assert manifest["status"] == "prepared_not_executed"
    assert "not a measured" in manifest["reference_modulus_role"]
    assert len(manifest["decks"]) == 5
    for name in check.cases(): assert (tmp_path/"controls"/(name+".feb")).read_text() == check.deck_xml(name)
    with pytest.raises(FileExistsError): check.write_bundle(tmp_path/"controls")


def test_output_reader_has_actual_byte_bound_and_strict_encoding(tmp_path):
    path = tmp_path/"output.log"
    path.write_bytes(b"12345")
    assert check.read_bounded_text(path, 5) == "12345"
    with pytest.raises(ValueError, match="bounded"): check.read_bounded_text(path, 4)
    path.write_bytes(b"\xff")
    with pytest.raises(UnicodeDecodeError): check.read_bounded_text(path)
