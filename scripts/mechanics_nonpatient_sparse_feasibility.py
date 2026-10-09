"""Pure declaration and numerical-oracle checks for an unreleased cube benchmark.

This module does not create FEBio decks, run a mesher/solver, or read patients.
Native execution needs a separate reviewed implementation and root release.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DECLARATION = ROOT / "manifests/experiments/nonpatient-tet10-sparse-feasibility-v1.json"
CASE_ORDER = (
    "n5_affine", "n5_nonuniform", "n9_affine", "n9_nonuniform",
    "n13_affine", "n13_nonuniform", "n13_nonuniform_half_step",
)
LEVELS = (5, 9, 13)
BOUND_FILES = (
    "backend_profile", "runtime_identity", "verified_tet10_fixture", "tet10_fixture_source",
)
BOUND_PATHS = {
    "backend_profile":"artifacts/mechanics/hbe-accelerate-experiment-preparation-v1/backend-profile.json",
    "runtime_identity":"artifacts/febio-accelerate-csc-runtime-v3/runtime-identity.json",
    "verified_tet10_fixture":"artifacts/mechanics-patient-constraints-runtime-v1/decks/manifest.json",
    "tet10_fixture_source":"scripts/mechanics_patient_constraints.py",
}
SOLVER_SUBTREE = ('<linear_solver type="accelerate"><iterative>0</iterative>'
                  '<factorization>4</factorization><order_method>0</order_method>'
                  '<print_condition_number>0</print_condition_number></linear_solver>')
RUNTIME_BINDING_POLICY = ("Separate root release must pin exact committed benchmark source/decks "
                          "and reverify profile, runtime identity, executable, linkage, patch "
                          "and tet10 controls before and after each supervised call. No fallback backend.")
MESH_RECIPE = ("Structured n-by-n-by-n cubes; each split into the same six tetrahedra along "
               "local 0-to-6 body diagonal as the pinned one-cube tet10 control. Straight "
               "reference mid-edge nodes shared by integer half-grid identity; FEBio TET10G8 elastic-solid.")
MESH_ACCEPTANCE = ("Actual generated incidence, positive reference Jacobians at all eight integration "
                   "points and additional nodes/edge midpoints/centroids, conforming shared midnodes, "
                   "and boundary tags must be checked before any solve. Expected counts are predictions, not observations.")
NUMERICAL_THRESHOLDS = {
    "positive_reference_element_jacobian": True,
    "minimum_sampled_deformation_J": .2,
    "maximum_affine_displacement_error_m": 1e-9,
    "maximum_affine_stress_error_Pa": .01,
    "maximum_affine_J_error": 1e-6,
    "maximum_affine_energy_error_J": 1e-8,
    "maximum_affine_boundary_virtual_work_error_J": 1e-7,
    "maximum_independent_force_balance_N": 1e-5,
    "maximum_independent_moment_balance_Nm": 1e-7,
    "maximum_n9_to_n13_field_difference_m": 5e-6,
    "maximum_n9_to_n13_reaction_difference_N": 1e-5,
    "maximum_n13_half_step_field_difference_m": 1e-7,
    "maximum_n13_half_step_reaction_difference_N": 1e-6,
    "refinement_difference_ratio_max": .8,
}
SOLVER_RESIDUAL_RULE = ("Actual reported final norms must satisfy frozen FEBio tolerances "
                        "on each completed step; missing or ambiguous residual evidence "
                        "is inconclusive, not pass.")
EXPECTED_CASES = [
    {"id":"n5_affine","n":5,"load":"affine"},
    {"id":"n5_nonuniform","n":5,"load":"nonuniform"},
    {"id":"n9_affine","n":9,"load":"affine"},
    {"id":"n9_nonuniform","n":9,"load":"nonuniform"},
    {"id":"n13_affine","n":13,"load":"affine"},
    {"id":"n13_nonuniform","n":13,"load":"nonuniform"},
    {"id":"n13_nonuniform_half_step","n":13,"load":"nonuniform",
     "time_steps":8,"step_size":.125},
]
EXPECTED_LOADS = {
    "affine": {
        "final_F":[[1.03,.02,0.],[0.,.99,.01],[0.,0.,1.01]],
        "boundary":"Prescribe u=(F-I)X on all six exterior faces; leave all interior nodes free. Compare displacement, stress, J, energy, reaction virtual work with independent analytic homogeneous solution.",
    },
    "nonuniform": {
        "bottom_z_m":0.0,
        "top_z_m":.01,
        "bottom_displacement_m":[0.,0.,0.],
        "top_displacement_m":[.0003,0.,-.0002],
        "lateral_faces":"traction_free_except_top_and_bottom_intersections",
        "sample_points_m":[[.005,.005,.005],[.0025,.005,.005],[.0075,.005,.005],
                           [.005,.0025,.005],[.005,.0075,.005],
                           [.005,.005,.0025],[.005,.005,.0075]],
        "reaction_quantity":"Net bottom and top reaction vectors in N, independently summed from signed FEBio nodal reactions.",
    },
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _finite_array(value: object, shape: tuple[int, ...]) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"Expected finite array with shape {shape}")
    return result


def validate_declaration(value: dict | None = None, *, root: Path = ROOT) -> dict:
    """Reject a changed fixture, schedule, gate or runtime pin before design use."""
    if value is None:
        value = json.loads((root / DECLARATION.relative_to(ROOT)).read_text())
    if value.get("schema") != "nonpatient-tet10-sparse-feasibility-v1":
        raise ValueError("Wrong declaration schema")
    if set(value) != {"schema","status","purpose","execution_release",
                      "source_commit_for_execution","runtime","mesh","numerical_gauge",
                      "cases","loads","caps","checks","stop_rule","admission"}:
        raise ValueError("Unexpected or missing declaration section")
    if (value.get("status") != "prepared_not_released_or_executed"
            or value.get("execution_release") is not None
            or value.get("source_commit_for_execution") is not None):
        raise ValueError("This preparation cannot authorize native execution")
    runtime = value["runtime"]
    if set(runtime) != {"framework","upstream_commit","backend_profile_id",
                        "backend_profile","runtime_identity","verified_tet10_fixture",
                        "tet10_fixture_source","solver_subtree","runtime_source_binding_policy"}:
        raise ValueError("Unexpected or missing runtime field")
    if (runtime["framework"] != "FEBio 4.13.0"
            or runtime["upstream_commit"] != "32ae206ff4881dfb54f62296cd1558e58ed9fcc6"
            or runtime["backend_profile_id"] != "accelerate_csc_v1"
            or runtime["solver_subtree"] != SOLVER_SUBTREE
            or runtime["runtime_source_binding_policy"] != RUNTIME_BINDING_POLICY):
        raise ValueError("Pinned direct-solver profile changed")
    for key in BOUND_FILES:
        binding = runtime[key]
        if set(binding) != {"path","sha256"}:
            raise ValueError("Unexpected runtime/source binding field")
        path = Path(binding["path"])
        if (binding["path"] != BOUND_PATHS[key] or path.is_absolute()
                or ".." in path.parts or not (root / path).is_file()):
            raise ValueError("Invalid runtime/source binding path")
        if _sha256(root / path) != binding["sha256"]:
            raise ValueError(f"Changed runtime/source binding: {key}")
    profile = json.loads((root / runtime["backend_profile"]["path"]).read_text())
    if (profile.get("profile_id") != runtime["backend_profile_id"]
            or profile.get("runtime_identity") != runtime["runtime_identity"]):
        raise ValueError("Pinned backend profile does not identify runtime")
    mesh = value["mesh"]
    if set(mesh) != {"geometry","side_m","recipe","cube_vertex_order",
                     "local_tet4","tet10_edges","levels","acceptance"}:
        raise ValueError("Unexpected or missing mesh field")
    if (mesh["side_m"] != 0.01 or mesh["geometry"] != "10 mm numerical cube; no anatomy"
            or mesh["recipe"] != MESH_RECIPE or mesh["acceptance"] != MESH_ACCEPTANCE
            or mesh["cube_vertex_order"] != [[0,0,0],[1,0,0],[1,1,0],[0,1,0],
                                             [0,0,1],[1,0,1],[1,1,1],[0,1,1]]):
        raise ValueError("Numerical cube changed")
    if (mesh["local_tet4"] != [[0,1,2,6],[0,2,3,6],[0,3,7,6],
                               [0,7,4,6],[0,4,5,6],[0,5,1,6]]
            or mesh["tet10_edges"] != [[0,1],[1,2],[2,0],[0,3],[1,3],[2,3]]):
        raise ValueError("Reference topology changed")
    rows = mesh["levels"]
    if [row["subdivisions_per_axis"] for row in rows] != list(LEVELS):
        raise ValueError("Frozen mesh levels changed")
    for row in rows:
        n = row["subdivisions_per_axis"]
        if row != {"subdivisions_per_axis":n,"expected_nodes":(2*n+1)**3,
                   "expected_tet10":6*n**3}:
            raise ValueError("Nominal structured-tet10 incidence count changed")
    gauge = value["numerical_gauge"]
    if gauge != {"units":"m,N,Pa,J", "mu_Pa":1000.0, "K_Pa":1000*29/3,
                  "reference_quadrature":"FEBio TET10G8", "time_steps":4,
                  "step_size":.25,
                  "material":"Uncoupled Ogden c1=2*mu, m1=2, c2..c6=0, pressure_model=1; numerical gauge only",
                  "solver":"FEBio solid, symmetric stiffness, BFGS; dtol=etol=rtol=1e-10, min_residual=1e-20, max_refs=20, no time-step retries"}:
        raise ValueError("Numerical gauge changed")
    cases = value["cases"]
    if cases != EXPECTED_CASES or [row["id"] for row in cases] != list(CASE_ORDER):
        raise ValueError("Frozen seven-call schedule changed")
    if value["loads"] != EXPECTED_LOADS:
        raise ValueError("Frozen affine/nonuniform load design changed")
    nonuniform = value["loads"]["nonuniform"]
    samples = _finite_array(nonuniform["sample_points_m"], (7, 3))
    if np.any(samples <= 0) or np.any(samples >= mesh["side_m"]):
        raise ValueError("Fixed samples must lie strictly inside the cube")
    caps = value["caps"]
    if caps != {"maximum_native_calls":7,"attempts_per_case":1,
                "automatic_retry_or_fallback":False,"numerical_threads":1,
                "sampled_process_group_rss_bytes":3221225472,
                "per_call_wall_seconds":600,"aggregate_native_wall_seconds":1800,
                "active_output_bytes":536870912}:
        raise ValueError("Frozen resource caps changed")
    checks = value["checks"]
    if checks != {**NUMERICAL_THRESHOLDS,"complete_states":True,
                  "finite_displacement_stress_reaction":True,
                  "solver_residual":SOLVER_RESIDUAL_RULE}:
        raise ValueError("Frozen numerical/native-output checks changed")
    return value


def affine_oracle(final_F: object, *, mu_Pa: float = 1000., K_Pa: float = 1000*29/3,
                  side_m: float = .01) -> dict:
    """Independent homogeneous Ogden alpha=2 continuum identity, SI units."""
    F = _finite_array(final_F, (3, 3))
    J = float(np.linalg.det(F))
    if J <= 0 or mu_Pa <= 0 or K_Pa <= 0 or side_m <= 0:
        raise ValueError("Positive Jacobian, gauges and side length required")
    B = F @ F.T
    tr = float(np.trace(B))
    W = mu_Pa/2 * (J**(-2/3)*tr-3) + K_Pa/4*(J*J-1-2*np.log(J))
    sigma = mu_Pa*J**(-5/3)*(B-tr/3*np.eye(3)) + K_Pa/2*(J-1/J)*np.eye(3)
    P = J*sigma@np.linalg.inv(F).T
    # The existing FEBio fixture reports reactions exerted by the body on its
    # displacement constraints: -P n, hence the negative work convention.
    virtual_work = float(-np.sum(P*(F-np.eye(3)))*side_m**3)
    return {"J":J,"stress_Pa":sigma.tolist(),"energy_J":float(W*side_m**3),
            "first_Piola_Pa":P.tolist(),"boundary_virtual_work_J":virtual_work}


def affine_check(observed: dict, *, declaration: dict | None = None) -> dict:
    """Independent homogeneous field/reaction audit on supplied native outputs.

    `observed` must contain all reference nodes, nodal displacements and
    reactions, element stresses and deformation Jacobians, and total energy.
    The caller must separately establish exact backend, mesh and output origin.
    """
    declared = validate_declaration() if declaration is None else validate_declaration(declaration)
    F = _finite_array(declared["loads"]["affine"]["final_F"], (3, 3))
    oracle = affine_oracle(F)
    X = np.asarray(observed["reference_nodes_m"], dtype=float)
    if X.ndim != 2 or X.shape[1] != 3 or not len(X) or not np.isfinite(X).all():
        raise ValueError("Complete finite reference nodes required")
    u = _finite_array(observed["displacement_m"], X.shape)
    r = _finite_array(observed["reaction_N"], X.shape)
    stress = np.asarray(observed["element_stress_Pa"], dtype=float)
    if stress.ndim != 3 or stress.shape[1:] != (3, 3) or not len(stress) or not np.isfinite(stress).all():
        raise ValueError("Complete finite element stress required")
    J = _finite_array(observed["element_deformation_J"], (len(stress),))
    energy = float(observed["total_energy_J"])
    if not np.isfinite(energy):
        raise ValueError("Finite total energy required")
    checks = declared["checks"]
    target_u = X @ (F-np.eye(3)).T
    errors = {
        "maximum_displacement_m":float(np.max(np.linalg.norm(u-target_u,axis=1))),
        "maximum_stress_Pa":float(np.max(np.abs(stress-np.asarray(oracle["stress_Pa"])))),
        "maximum_deformation_J":float(np.max(np.abs(J-oracle["J"]))),
        "energy_J":abs(energy-oracle["energy_J"]),
        "boundary_virtual_work_J":abs(float(np.sum(r*u))-oracle["boundary_virtual_work_J"]),
        "force_balance_N":float(np.linalg.norm(np.sum(r,axis=0))),
        "moment_balance_Nm":float(np.linalg.norm(np.sum(np.cross(X+u,r),axis=0))),
    }
    gates = {
        "displacement":errors["maximum_displacement_m"] <= checks["maximum_affine_displacement_error_m"],
        "stress":errors["maximum_stress_Pa"] <= checks["maximum_affine_stress_error_Pa"],
        "deformation_J":errors["maximum_deformation_J"] <= checks["maximum_affine_J_error"]
            and float(np.min(J)) >= checks["minimum_sampled_deformation_J"],
        "energy":errors["energy_J"] <= checks["maximum_affine_energy_error_J"],
        "reaction_work":errors["boundary_virtual_work_J"] <= checks["maximum_affine_boundary_virtual_work_error_J"],
        "force_balance":errors["force_balance_N"] <= checks["maximum_independent_force_balance_N"],
        "moment_balance":errors["moment_balance_Nm"] <= checks["maximum_independent_moment_balance_Nm"],
    }
    return {"passed":all(gates.values()),"checks":gates,"errors":errors,
            "scope":"Supplied-array analytic numerical audit only; no mesh, backend or output provenance."}


def convergence_check(rows: dict, *, declaration: dict | None = None) -> dict:
    """Check fixed-point/aggregate-reaction trends on supplied saved outputs only.

    Each row has displacement_m[7][3] and top_reaction_N[3]. This does not
    verify solver provenance, interpolation, or mechanical equilibrium; those
    are separately required native-output audits before any decision.
    """
    declared = validate_declaration() if declaration is None else validate_declaration(declaration)
    expected = ("n5_nonuniform", "n9_nonuniform", "n13_nonuniform",
                "n13_nonuniform_half_step")
    if set(rows) != set(expected):
        raise ValueError("Missing or unexpected convergence case")
    fields, forces = {}, {}
    for key in expected:
        fields[key] = _finite_array(rows[key]["displacement_m"], (7, 3))
        forces[key] = _finite_array(rows[key]["top_reaction_N"], (3,))
    def maximum_distance(a, b):
        return float(np.max(np.linalg.norm(a-b, axis=-1)))
    a, b, c, d = expected
    field_59 = maximum_distance(fields[a], fields[b])
    field_913 = maximum_distance(fields[b], fields[c])
    reaction_59 = maximum_distance(forces[a], forces[b])
    reaction_913 = maximum_distance(forces[b], forces[c])
    field_step = maximum_distance(fields[c], fields[d])
    reaction_step = maximum_distance(forces[c], forces[d])
    def decreasing(coarse, fine):
        return fine <= declared["checks"]["refinement_difference_ratio_max"]*coarse
    checks = declared["checks"]
    gates = {
        "field_refinement_decreases":decreasing(field_59,field_913),
        "reaction_refinement_decreases":decreasing(reaction_59,reaction_913),
        "field_n9_n13_absolute":field_913 <= checks["maximum_n9_to_n13_field_difference_m"],
        "reaction_n9_n13_absolute":reaction_913 <= checks["maximum_n9_to_n13_reaction_difference_N"],
        "field_half_step":field_step <= checks["maximum_n13_half_step_field_difference_m"],
        "reaction_half_step":reaction_step <= checks["maximum_n13_half_step_reaction_difference_N"],
    }
    return {"passed":all(gates.values()), "checks":gates,
            "metrics":{"field_n5_n9_m":field_59,"field_n9_n13_m":field_913,
                       "top_reaction_n5_n9_N":reaction_59,"top_reaction_n9_n13_N":reaction_913,
                       "field_n13_step_repeat_m":field_step,
                       "top_reaction_n13_step_repeat_N":reaction_step},
            "scope":"Pure supplied-array numerical trend only; no provenance, solve or physical validation."}
