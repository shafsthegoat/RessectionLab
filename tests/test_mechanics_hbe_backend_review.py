"""Independent constructed-file controls; no solver or measured data access."""
import hashlib
import copy
import json
from pathlib import Path
import pytest

from scripts import mechanics_hbe_access as access
from scripts import mechanics_hbe_backend as backend


LIBRARY_NAMES = {
    "install/bin/febio4",
    *(f"install/lib/lib{name}.dylib" for name in (
        "feamr", "febiofluid", "febiolib", "febiomech", "febiomix",
        "febioopt", "febioplot", "febiorve", "febioxml", "fecore",
        "feimglib", "numcore")),
}


def save(root, name, value):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    data = value if isinstance(value, bytes) else access.canonical_json(value)
    path.write_bytes(data)
    return {"path": name, "sha256": hashlib.sha256(data).hexdigest()}


def runtime_fixture(root, monkeypatch, *, library_names=LIBRARY_NAMES):
    """Toy non-executable bytes isolate manifest contracts, not runtime validity."""
    libraries = {name: save(root, backend.PREFIX + "/" + name,
                           ("constructed " + name).encode())["sha256"]
                 for name in library_names}
    patched = save(root, backend.PREFIX + "/source/NumCore/AccelerateSparseSolver.cpp",
                   b"constructed source, not a solver")
    monkeypatch.setattr(backend, "PATCHED_SOURCE_SHA", patched["sha256"])
    omp = save(root, backend.PREFIX + "/openmp/lib/libomp.dylib", b"constructed OpenMP")
    identity = {
        "source_commit": backend.UPSTREAM_COMMIT, "runtime_version": "4.13.0",
        "architecture": "arm64", "source_patch_sha256": backend.PATCH_SHA,
        "patched_source_sha256": patched["sha256"],
        "executable": str(root / backend.PREFIX / "install/bin/febio4"),
        "executable_sha256": libraries["install/bin/febio4"],
        "libraries": libraries,
        "private_openmp": {"path": str(root / omp["path"]), "sha256": omp["sha256"]},
    }
    return identity, save(root, "runtime.json", identity)


def test_control_pass_labels_without_execution_and_raw_evidence_are_rejected(tmp_path, monkeypatch):
    runtime_binding = {"path": "runtime.json", "sha256": "1" * 64}
    # The runtime layer is intentionally isolated: this must fail at controls.
    monkeypatch.setattr(backend, "verify_runtime_binding", lambda *args: {
        "runtime": {}, "prefix": str(tmp_path / backend.PREFIX), "inputs": {},
        "runtime_identity": runtime_binding,
    })
    profile = {"schema": "hbe-solver-backend-v1", "profile_id": backend.PROFILE_ID,
               "runtime_identity": runtime_binding}
    # Traverse the genuine committed nested repair schema before challenging
    # controls; an obsolete flat patch fixture must not mask this regression.
    repository = Path(__file__).resolve().parents[1]
    patch = json.loads((repository / backend.PATCH_IDENTITY["path"]).read_bytes())
    for record in (backend.PATCH_IDENTITY, patch["patch"], patch["patched_source"]):
        copied = save(tmp_path, record["path"], (repository / record["path"]).read_bytes())
        assert copied["sha256"] == record["sha256"]
    profile["patch_identity"] = backend.PATCH_IDENTITY.copy()
    for name, status, field, cases in (
        ("hex8_controls", "passed_all_five_fixed_patch_controls", "rows", backend.HEX_CASES),
        ("tet10_mpc_controls", "three_actual_fixed_software_controls_passed", "case_rows", backend.TET_CASES),
    ):
        profile[name] = save(tmp_path, name + ".json", {
            "status": status, "solver_backend": "accelerate",
            "runtime_identity_sha256": runtime_binding["sha256"],
            "solver_invocations": len(cases),
            field: [{"case": name, "passed": True} for name in sorted(cases)],
            "stiffness_scaling": {"passed": True},
        })
    binding = save(tmp_path, "profile.json", profile)
    with pytest.raises(ValueError, match="actual eight-case attempt"):
        backend.verify_profile(tmp_path, binding)


def test_runtime_cannot_omit_twelve_linked_library_bindings(tmp_path, monkeypatch):
    identity, binding = runtime_fixture(tmp_path, monkeypatch,
                                        library_names={"install/bin/febio4"})
    identity["status"] = "isolated_patched_runtime_built_pending_numerical_controls"
    binding = save(tmp_path, binding["path"], identity)
    with pytest.raises(ValueError):
        backend.verify_runtime_binding(tmp_path, binding)


def test_worker_candidate_is_not_parent_accepted_runtime(tmp_path, monkeypatch):
    identity, binding = runtime_fixture(tmp_path, monkeypatch)
    identity["status"] = "candidate_pending_parent_build_acceptance"
    binding = save(tmp_path, binding["path"], identity)
    with pytest.raises(ValueError):
        backend.verify_runtime_binding(tmp_path, binding)


@pytest.mark.parametrize("change", [None, "sibling_path", "different_declared_hash", "changed_bytes"])
def test_original_openmp_exception_requires_exact_path_hash_and_bytes(tmp_path, monkeypatch, change):
    assert backend.REUSED_OPENMP == "data/optional-runtimes/febio-4.13/openmp/lib/libomp.dylib"
    assert backend.REUSED_OPENMP_SHA == "38f6afed27bf1d3bd52779547c7ab53aeda9c06c3473263b5cd11ed4f04b41f8"
    identity, binding = runtime_fixture(tmp_path, monkeypatch)
    identity["status"] = "isolated_patched_runtime_built_pending_numerical_controls"
    # Isolate the narrow OpenMP exception, not acceptance or build evidence.
    monkeypatch.setattr(backend, "_accepted_build", lambda *args: None)
    omp = save(tmp_path, backend.REUSED_OPENMP, b"constructed declared OpenMP")
    monkeypatch.setattr(backend, "REUSED_OPENMP_SHA", omp["sha256"])
    if change == "sibling_path":
        omp = save(tmp_path, backend.REUSED_OPENMP.replace("libomp.dylib", "other.dylib"),
                   b"constructed declared OpenMP")
    elif change == "different_declared_hash":
        omp = save(tmp_path, backend.REUSED_OPENMP, b"coherently rehashed different OpenMP")
    elif change == "changed_bytes":
        (tmp_path / omp["path"]).write_bytes(b"changed after binding")
    identity["private_openmp"] = {"path": str(tmp_path / omp["path"]), "sha256": omp["sha256"]}
    binding = save(tmp_path, binding["path"], identity)
    if change is None:
        result = backend.verify_runtime_binding(tmp_path, binding)
        assert result["inputs"][str(tmp_path / omp["path"])] == omp["sha256"]
    else:
        with pytest.raises(ValueError):
            backend.verify_runtime_binding(tmp_path, binding)


def constructed_control_attempt(root, monkeypatch):
    """Exercise receipt wiring with explicit checker doubles, never physics proof."""
    source = root / "source"
    attempt = root / "attempt"
    runtime_binding = {"path": "runtime.json", "sha256": "1" * 64}
    executable = str(root / "constructed-runtime/bin/never-executed")
    checker = b'''import json
def check_outputs(case, nodes, elements, solver):
    row = json.loads(nodes)
    return {"passed": row["case"] == case and row["valid"] is True, "case": case}
def check_stiffness_scaling(nodes, elements, doubled_nodes, doubled_elements):
    ratio = json.loads(doubled_nodes)["scale"] / json.loads(nodes)["scale"]
    return {"passed": ratio == 2, "ratio": ratio}
'''
    checker_records = {
        group: save(root, "source/scripts/" + filename, checker)
        for group, filename in (("hex8", "mechanics_febio_verification.py"),
                                ("tet10", "mechanics_patient_constraints.py"))
    }
    monkeypatch.setattr(backend, "CHECKER_PINS",
                        {group: record["sha256"] for group, record in checker_records.items()})
    inputs = {str(root / record["path"]): record["sha256"] for record in checker_records.values()}
    declared = {"case_order": list(backend.CASE_ORDER), "cases": {}}
    rows = []
    for name in backend.CASE_ORDER:
        original_text = '<febio_spec><Control><solver>' + backend.SKYLINE_XML + '</solver></Control><Material>' + name + '</Material></febio_spec>'
        original = save(root, "source/original/" + name + ".feb", original_text.encode())
        adapted = backend.transform_deck(original_text).encode()
        template = save(root, "source/adapted/" + name + ".feb", adapted)
        directory = "attempt/" + name + "/"
        records = {
            "original_deck": original,
            "executed_deck": save(root, directory + name + ".feb", adapted),
            "console": save(root, directory + "console.txt", b"selecting linear solver accelerate\n"),
            "nodes": save(root, directory + name + ".nodes.log", {
                "case": name, "valid": True, "scale": 2 if name == "shear_double_stiffness" else 1}),
            "elements": save(root, directory + name + ".elements.log", b"constructed element record"),
            "solver_log": save(root, directory + name + ".log", b"constructed solver record; never executed"),
            "checked": save(root, directory + "checked.json", {"passed": True, "case": name}),
            "checker_source": checker_records["hex8" if name in backend.HEX_CASES else "tet10"],
        }
        declared["cases"][name] = {
            "original_path": "original/" + name + ".feb", "original_sha256": original["sha256"],
            "adapted_path": "adapted/" + name + ".feb", "adapted_sha256": template["sha256"],
        }
        inputs.update({str(root / record["path"]): record["sha256"] for record in (original, template)})
        rows.append({
            "case": name, "status": "passed", "passed": True, "checker_passed": True,
            "solver_exit_code": 0, "deck_sha256": template["sha256"], "evidence": records,
            "command": [executable, "-noconfig", "-no_title", "-i", name + ".feb", "-o", name + ".log"],
            "backend_evidence": {
                "solver_backend": "accelerate", "actual_selection_lines": ["selecting linear solver accelerate"],
                "solver_xml": backend.ACCELERATE_XML, "solver_only_change_verified": True,
                "runtime_identity_sha256": runtime_binding["sha256"],
                "executed_deck_sha256": template["sha256"],
            },
        })
    declaration_binding = save(root, "source/artifacts/mechanics-accelerate-controls-v1/declaration.json", declared)
    inputs[str(root / declaration_binding["path"])] = declaration_binding["sha256"]
    audit = {name: {"unchanged": True} for name in inputs}
    caps = {"aggregate_seconds": 60, "process_group_rss_bytes": 3 * 1024**3,
            "numerical_threads": 1, "maximum_cases": 8, "time_steps": 4, "retries": 0}
    baseline = {
        "attempt_directory": str(attempt), "source_directory": str(source),
        "source_commit": "3" * 40, "runtime_identity_sha256": runtime_binding["sha256"],
        "caps": caps, "case_order": list(backend.CASE_ORDER), "release": {"runtime_identity": runtime_binding},
        "input_hashes": inputs, "declaration": declared,
    }
    scaling = {"passed": True, "ratio": 2.0}
    results = {
        "status": "completed", "solver_invocations": 8, "no_retry": True,
        "solver_backend": "accelerate", "runtime_identity_sha256": runtime_binding["sha256"],
        "cases": [{key: value for key, value in row.items() if key not in ("evidence", "passed")} for row in rows],
        "stiffness_scaling": scaling, "inputs_after": audit,
    }
    execution = {
        "status": "completed", "worker_started": True, "no_retry": True,
        "supervision": {"status": "completed", "exit_code": 0, "kill_reason": None,
            "wall_cap_seconds": 60, "rss_cap_bytes": 3 * 1024**3,
            "elapsed_seconds": 1.0, "sampled_peak_process_group_rss_bytes": 1024},
        "thread_environment": {"OMP_NUM_THREADS": "1", "OMP_DYNAMIC": "FALSE",
            "VECLIB_MAXIMUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
        "inputs_after": audit, "final_inputs_after": audit,
    }
    evidence = {
        "execution": save(root, "attempt/execution.json", execution),
        "results": save(root, "attempt/results.json", results),
        "baseline": save(root, "attempt/execution-baseline.json", baseline),
    }
    common = {"source_commit": baseline["source_commit"], "execution_accepted": True, "evidence": evidence}
    groups = [{**common, "rows": rows[:5], "stiffness_scaling": copy.deepcopy(scaling)},
              {**common, "case_rows": rows[5:]}]
    def bound(record, *, json_value=True):
        return access.verify_binding(root, record, maximum_bytes=1024**2, read_json=json_value)
    return {"profile": {"runtime_identity": runtime_binding}, "groups": groups,
            "runtime": {"runtime": {"executable": executable}}, "bound": bound,
            "baseline": baseline, "execution": execution, "evidence": evidence}


@pytest.mark.parametrize("change", [None, "stale_parent", "different_attempt", "relocated_original", "relocated_checker", "raw_disagreement", "scaling_disagreement"])
def test_eight_control_replay_binds_attempt_inputs_and_checker_results(tmp_path, monkeypatch, change):
    fixture = constructed_control_attempt(tmp_path, monkeypatch)
    if change == "stale_parent":
        fixture["execution"]["status"] = "failed_or_incomplete"
        save(tmp_path, fixture["evidence"]["execution"]["path"], fixture["execution"])
    elif change == "different_attempt":
        fixture["baseline"]["attempt_directory"] = str(tmp_path / "another-attempt")
        fixture["evidence"]["baseline"] = save(tmp_path, fixture["evidence"]["baseline"]["path"], fixture["baseline"])
    elif change == "relocated_original":
        row = fixture["groups"][0]["rows"][0]
        original = tmp_path / row["evidence"]["original_deck"]["path"]
        row["evidence"]["original_deck"] = save(tmp_path, "unrelated/identical.feb", original.read_bytes())
    elif change == "relocated_checker":
        row = fixture["groups"][1]["case_rows"][0]
        original = tmp_path / row["evidence"]["checker_source"]["path"]
        row["evidence"]["checker_source"] = save(tmp_path, "unrelated/copied_checker.py", original.read_bytes())
    elif change == "raw_disagreement":
        row = fixture["groups"][0]["rows"][0]
        record = row["evidence"]["nodes"]
        value = json.loads((tmp_path / record["path"]).read_bytes())
        value["valid"] = False
        row["evidence"]["nodes"] = save(tmp_path, record["path"], value)
    elif change == "scaling_disagreement":
        fixture["groups"][0]["stiffness_scaling"]["ratio"] = 2.1
    def verify():
        backend._verify_controls(tmp_path, fixture["profile"], fixture["groups"], fixture["runtime"], fixture["bound"])
    if change is None:
        verify()
    else:
        with pytest.raises(ValueError):
            verify()
