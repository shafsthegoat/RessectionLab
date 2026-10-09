"""Fixture-only checks. These never generate a native mesh or invoke FEBio."""
from copy import deepcopy

import numpy as np
import pytest

from scripts import mechanics_nonpatient_sparse_feasibility as design
from scripts import mechanics_patient_constraints as frozen_cube


def test_declaration_binds_real_repaired_runtime_and_fixed_schedule():
    declaration = design.validate_declaration()
    assert [row["id"] for row in declaration["cases"]] == list(design.CASE_ORDER)
    assert [(row["expected_nodes"], row["expected_tet10"])
            for row in declaration["mesh"]["levels"]] == [
                (1331, 750), (6859, 4374), (19683, 13182)]
    assert declaration["execution_release"] is None
    assert declaration["source_commit_for_execution"] is None


@pytest.mark.parametrize("change", [
    lambda x: x["runtime"]["runtime_identity"].update(sha256="0" * 64),
    lambda x: x["mesh"]["levels"][2].update(expected_nodes=19070),
    lambda x: x["cases"][-1].update(time_steps=4),
    lambda x: x["caps"].update(sampled_process_group_rss_bytes=4 * 1024**3),
    lambda x: x["checks"].update(maximum_n9_to_n13_field_difference_m=1e-4),
    lambda x: x["loads"]["nonuniform"].update(bottom_z_m=.0001),
    lambda x: x["loads"]["nonuniform"].update(top_z_m=.009),
    lambda x: x["loads"]["nonuniform"].update(lateral_faces="fixed"),
    lambda x: x["loads"]["affine"].update(boundary="prescribe every node"),
    lambda x: x["loads"]["nonuniform"].update(reaction_quantity="unsigned top only"),
    lambda x: x["checks"].update(solver_residual="skip if unavailable"),
    lambda x: x["cases"][0].update(time_steps=8),
    lambda x: x["cases"][5].update(step_size=.125),
    lambda x: x["mesh"].update(recipe="tetrahedralize automatically"),
    lambda x: x["mesh"].update(acceptance="count only"),
    lambda x: x["runtime"].update(runtime_source_binding_policy="use ambient FEBio"),
    lambda x: x["numerical_gauge"].update(unreviewed_material_override=42),
    lambda x: x.update(execution_release="premature"),
])
def test_declaration_rejects_unreviewed_drift(change):
    declaration = deepcopy(design.validate_declaration())
    change(declaration)
    with pytest.raises(ValueError):
        design.validate_declaration(declaration)


def test_affine_oracle_identity_and_finite_stretch():
    identity = design.affine_oracle(np.eye(3))
    assert identity["J"] == pytest.approx(1)
    assert np.allclose(identity["stress_Pa"], np.zeros((3, 3)), atol=1e-12)
    assert identity["energy_J"] == pytest.approx(0, abs=1e-18)
    declared = design.validate_declaration()
    result = design.affine_oracle(declared["loads"]["affine"]["final_F"])
    F = np.array(declared["loads"]["affine"]["final_F"])
    sigma = np.array(result["stress_Pa"])
    P = np.array(result["first_Piola_Pa"])
    assert result["J"] > 0
    assert np.allclose(P, result["J"] * sigma @ np.linalg.inv(F).T)
    assert result["energy_J"] > 0
    assert result["boundary_virtual_work_J"] < 0
    with pytest.raises(ValueError):
        design.affine_oracle(np.diag([-1., 1., 1.]))


def test_affine_check_accepts_analytic_cube_and_detects_reaction_sign_error():
    declared = design.validate_declaration()
    F = np.array(declared["loads"]["affine"]["final_F"])
    oracle = design.affine_oracle(F)
    X, elements, _ = frozen_cube.fixture_mesh()
    observation = {
        "reference_nodes_m":X.tolist(),
        "displacement_m":(X @ (F-np.eye(3)).T).tolist(),
        "reaction_N":frozen_cube.affine_reactions(1.).tolist(),
        "element_stress_Pa":[oracle["stress_Pa"]] * len(elements),
        "element_deformation_J":[oracle["J"]] * len(elements),
        "total_energy_J":oracle["energy_J"],
    }
    passed = design.affine_check(observation)
    assert passed["passed"], passed
    observation["reaction_N"] = (-np.asarray(observation["reaction_N"])).tolist()
    failed = design.affine_check(observation)
    assert not failed["passed"]
    assert not failed["checks"]["reaction_work"]


def _rows():
    def row(displacement, reaction):
        return {"displacement_m": [[displacement, 0., 0.]] * 7,
                "top_reaction_N": [reaction, 0., 0.]}
    return {"n5_nonuniform": row(0., 0.),
            "n9_nonuniform": row(4e-6, 8e-6),
            "n13_nonuniform": row(6e-6, 12e-6),
            "n13_nonuniform_half_step": row(6.005e-6, 12.05e-6)}


def test_predeclared_convergence_accepts_only_bounded_decreasing_differences():
    outcome = design.convergence_check(_rows())
    assert outcome["passed"]
    assert outcome["metrics"]["field_n9_n13_m"] == pytest.approx(2e-6)
    rows = _rows()
    rows["n13_nonuniform"]["displacement_m"] = [[9e-6, 0., 0.]] * 7
    failed = design.convergence_check(rows)
    assert not failed["passed"]
    assert not failed["checks"]["field_refinement_decreases"]


def test_convergence_rejects_missing_or_nonfinite_saved_values():
    rows = _rows()
    rows.pop("n13_nonuniform")
    with pytest.raises(ValueError):
        design.convergence_check(rows)
    rows = _rows()
    rows["n13_nonuniform"]["top_reaction_N"][0] = float("nan")
    with pytest.raises(ValueError):
        design.convergence_check(rows)
