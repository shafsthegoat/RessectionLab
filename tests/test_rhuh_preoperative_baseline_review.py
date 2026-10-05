"""Independent analytical runner controls; never reads or fits clinical CSV rows."""
import json
import csv
import io
import sys
import tarfile
from pathlib import Path
import time
import warnings

import numpy as np
import pytest

from scripts import run_rhuh_preoperative_baseline as runner
from resectionlab import rhuh_outcomes as adapter

ROOT = Path(__file__).resolve().parents[1]


def study():
    return json.loads((ROOT / runner.DECLARATION_PATH).read_text())


def test_convergence_warning_subclass_cannot_be_marked_converged():
    pytest.importorskip("sklearn", reason="Requires the isolated RHUH research runtime")
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.preprocessing import StandardScaler

    class DerivedConvergenceWarning(ConvergenceWarning):
        pass

    class Model:
        def fit(self, X, y):
            self.coef_ = np.zeros((1, X.shape[1]))
            self.intercept_ = np.zeros(1)
            self.n_iter_ = np.ones(1, dtype=int)
            self.classes_ = np.array([0, 1])
            warnings.warn("constructed convergence failure", DerivedConvergenceWarning)

        def predict_proba(self, X):
            return np.tile([.5, .5], (len(X), 1))

    rows = []
    counts = {"attempted": 0, "completed": 0}
    with pytest.raises(RuntimeError, match="convergence"):
        runner.loo(np.arange(12.).reshape(6, 2), np.array([0, 1, 0, 1, 0, 1]),
                   [f"analytic-{i}" for i in range(6)], study(),
                   deadline=time.monotonic() + 10, counts=counts,
                   on_fold=rows.append, job="constructed-warning",
                   factory=lambda: (StandardScaler(), Model()))
    assert counts == {"attempted": 1, "completed": 0}
    assert len(rows) == 1 and rows[0]["status"] == "failed"


def test_runtime_refuses_numpy_from_outside_the_declared_interpreter(tmp_path, monkeypatch):
    pytest.importorskip("sklearn", reason="Requires the isolated RHUH research runtime")
    dependency = json.loads((ROOT / "artifacts/rhuh-baseline-runtime-v1/receipt.json").read_text())
    foreign = tmp_path / "foreign_numpy_init.py"
    foreign.write_text("# constructed unrelated numerical source\n")
    monkeypatch.setattr(np, "__file__", str(foreign))
    for key in runner.THREADS:
        monkeypatch.setenv(key, "1")
    with pytest.raises(ValueError):
        runner.runtime_record({"dependency": dependency})


def test_constant_features_reduce_to_training_prevalence_without_penalized_intercept():
    pytest.importorskip("sklearn", reason="Requires the isolated RHUH research runtime")
    X = np.tile([3., 9.], (6, 1))
    y = np.array([0, 0, 0, 1, 1, 1])
    rows = []
    counts = {"attempted": 0, "completed": 0}
    predictions = runner.loo(X, y, [f"analytic-{i}" for i in range(6)], study(),
        deadline=time.monotonic() + 10, counts=counts, on_fold=rows.append,
        job="constant-analytical-features", shadow_X=X.copy())
    expected = (3 - y) / 5
    for model in runner.MODELS:
        np.testing.assert_allclose(predictions[model], expected, rtol=0, atol=5e-7)
    assert counts == {"attempted": 12, "completed": 12}
    for row in rows:
        assert row["holdout_patient_id"] not in row["training_patient_ids"]
        for model, width in (("KPS", 1), ("KPS_CE_VOLUME", 2)):
            assert row["models"][model]["scaler_scale"] == [1.] * width
            assert row["models"][model]["coefficients"] == [[0.] * width]
            assert row["models"][model]["shadow_prediction_identical"]


def test_metrics_match_closed_form_and_declared_paired_sign():
    result = runner.metrics([0, 1, 0, 1], {
        "FOLD_PREVALENCE": [.5] * 4,
        "KPS": [.5] * 4,
        "KPS_CE_VOLUME": [.1, .2, .8, .9],
    })
    assert result["models"]["KPS_CE_VOLUME"]["brier"] == pytest.approx(.325)
    assert result["models"]["KPS_CE_VOLUME"]["roc_auc"] == .75
    assert result["paired_brier"]["KPS_CE_VOLUME_minus_KPS"] == pytest.approx(.075)
    assert result["models"]["KPS_CE_VOLUME"]["confusion"] == {"TN": 1, "FP": 1, "FN": 1, "TP": 1}
    assert result["models"]["KPS"]["negative_predictive_value"]["value"] is None
    assert result["confidence_intervals"] is None
    assert result["clinical_calibration_validated"] is False


def test_final_influence_failure_never_publishes_a_13_case_summary(tmp_path, monkeypatch):
    y = np.array([0] * 26 + [1] * 14)
    ids = [f"analytic-{i:02d}" for i in range(40)]
    data = {"X": np.ones((40, 2)), "y": y, "ids": ids,
            "categories": ["constructed"] * 40, "projection_hashes": ["constructed"] * 40}
    jobs = []

    def fake_loo(X, labels, kept_ids, config, *, counts, job, **kwargs):
        jobs.append(job)
        if job == "delete:" + ids[-1]:
            counts["attempted"] += 1
            raise RuntimeError("constructed final influence fit failure")
        counts["attempted"] += 2 * len(labels)
        counts["completed"] += 2 * len(labels)
        return {model: [.35] * len(labels) for model in runner.MODELS}

    monkeypatch.setattr(runner, "loo", fake_loo)
    with pytest.raises(RuntimeError, match="final influence"):
        runner.evaluate(data, data, study(), tmp_path, time.monotonic() + 15)
    state = json.loads((tmp_path / "state.json").read_text())
    assert state["status"] == "failed_or_incomplete"
    assert state["permutation"]["planned"] == state["permutation"]["completed"] == 99
    assert state["permutation"]["p_value"] == 1
    assert state["influence"]["planned"] == 14
    assert state["influence"]["completed"] == 13
    assert state["influence"]["summary"] is None
    assert state["fits"] == {"attempted": 9015, "completed": 9014}
    assert len(jobs) == 114
    lines = (tmp_path / "influence.jsonl").read_text().splitlines()
    assert len(lines) == 14
    assert json.loads(lines[-1])["status"] == "failed"


def test_shadow_rotates_every_excluded_column_and_never_moves_the_target():
    categories = ["Transient"] * 6 + ["Minor Persistent"] * 6 + ["Major Persistent"] * 2 + ["No"] * 26
    original = []
    for i, category in enumerate(categories):
        record = {field: f"constructed row {i} / {field}" for field in adapter.RHUH_HEADERS}
        record.update({"Patient ID": f"RHUH-{i+1:04d}", "Postoperative Neurological Deficit": category,
                       "Preoperative KPS": str(60 + 10 * (i % 4)),
                       "Postoperative KPS": str(50 + 10 * (i % 5)),
                       adapter.PREOPERATIVE_CE_VOLUME_FIELD: str(1 + i)})
        original.append(record)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=adapter.RHUH_HEADERS)
    writer.writeheader()
    writer.writerows(original)
    raw = stream.getvalue().encode()
    changed_bytes = runner.prohibited_shadow(raw)
    changed = list(csv.DictReader(io.StringIO(changed_bytes.decode())))
    for i in range(40):
        for key in adapter.RHUH_HEADERS:
            assert changed[i][key] == original[i if key in runner.PROTECTED_COLUMNS else (i+1) % 40][key]
    baseline = runner.projection(adapter, raw, runner.digest(raw), study())
    shadow = runner.projection(adapter, changed_bytes, runner.digest(changed_bytes), study())
    runner.validate_invariance(baseline, shadow)
    assert runner.digest(raw) != runner.digest(changed_bytes)


def test_exact_fit_cap_prevents_constructing_an_additional_estimator():
    pytest.importorskip("sklearn", reason="Requires the isolated RHUH research runtime")
    counts = {"attempted": 9092, "completed": 9092}
    rows = []

    def forbidden_factory():
        pytest.fail("No estimator may be constructed after the declared fit cap")

    with pytest.raises(RuntimeError, match="allocation exhausted"):
        runner.loo(np.ones((6, 2)), np.array([0, 0, 0, 1, 1, 1]),
                   [f"analytic-{i}" for i in range(6)], study(),
                   deadline=time.monotonic() + 10, counts=counts, on_fold=rows.append,
                   job="constructed-fit-cap", factory=forbidden_factory)
    assert counts == {"attempted": 9092, "completed": 9092}
    assert len(rows) == 1 and rows[0]["status"] == "failed"


@pytest.mark.parametrize("supervised_status,worker_elapsed,expected", [
    ("completed", 179.0, "completed_observational_baseline"),
    ("failed_or_incomplete", 179.0, "failed_or_incomplete"),
    ("completed", 180.0, "failed_or_incomplete"),
])
def test_parent_authority_keeps_sampled_resource_failure_and_worker_deadline_terminal(
        tmp_path, monkeypatch, supervised_status, worker_elapsed, expected):
    output = tmp_path / "outputs" / "constructed-attempt"
    config = study()
    plan = {"directory": str(output), "study": config, "inputs": {},
            "dependency": {"directory": str(tmp_path / "constructed-dependencies")}}
    monkeypatch.setattr(runner, "preflight", lambda *args: plan)

    def supervise(command, log_path, *, seconds, rss_bytes, environment, **kwargs):
        assert seconds == 195
        assert rss_bytes == 1024 ** 3
        assert environment["PYTHONNOUSERSITE"] == "1"
        assert all(environment[key] == "1" for key in runner.THREADS)
        assert "--worker-baseline-sha256" in command
        runner.supervisor.write_json(output / "worker-result.json", {
            "status": "completed_observational_baseline", "worker_elapsed_seconds": worker_elapsed})
        return {"status": supervised_status, "kill_reason": "process_group_rss_cap"
                if supervised_status != "completed" else None}

    monkeypatch.setattr(runner.supervisor, "supervise", supervise)
    result = runner.launch(tmp_path, {}, {})
    assert result["status"] == expected
    assert result["no_retry"] is True
    assert result["worker"]["sha256"] == runner.supervisor.sha(output / "worker-result.json")


@pytest.fixture
def constructed_preflight(tmp_path, monkeypatch):
    """Real byte/closure validators, constructed release and no clinical source."""
    import resectionlab

    def write(relative, content):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return {"path": relative, "sha256": runner.digest(content)}

    def write_json(relative, value):
        return write(relative, runner.canonical(value))

    modules = {"runner": runner, "supervisor": runner.supervisor,
               "adapter": adapter, "package_init": resectionlab}
    names = {"runner": "scripts/run_rhuh_preoperative_baseline.py",
             "supervisor": "scripts/febio_runtime.py",
             "adapter": "src/resectionlab/rhuh_outcomes.py",
             "package_init": "src/resectionlab/__init__.py"}
    sources = {}
    for key, module in modules.items():
        sources[key] = write(names[key], Path(module.__file__).read_bytes())
        monkeypatch.setattr(module, "__file__", str(tmp_path / names[key]))
    config = study()
    fake_data = write("analytical-source.txt", b"Constructed byte binding, not clinical CSV.\n")
    config["source"]["local_path"] = fake_data["path"]
    config["source"]["sha256"] = fake_data["sha256"]
    manifest = write_json(runner.DECLARATION_PATH, config)
    monkeypatch.setattr(runner, "DECLARATION_SHA", manifest["sha256"])
    dependencies = {"directory": str(tmp_path / "private-dependencies"),
                    "packages": {"numpy": "constructed", "scipy": "constructed"},
                    "files": {}, "shared_packages": {}}
    record = write("private-dependencies/sklearn/__init__.py", b"# constructed private dependency\n")
    dependencies["files"]["sklearn/__init__.py"] = record["sha256"]
    for name in ("numpy", "scipy"):
        folder = tmp_path / "shared" / name
        files = {}
        for filename in ("__init__.py", "kernel.so"):
            record = write(str(folder.relative_to(tmp_path) / filename),
                           f"constructed {name} {filename}\n".encode())
            files[filename] = record["sha256"]
        dependencies["shared_packages"][name] = {"directory": str(folder),
                                                  "version": "constructed", "files": files}
    dependency_binding = write_json("dependencies.json", dependencies)
    archive = tmp_path / "source.tar"
    with tarfile.open(archive, "w", format=tarfile.PAX_FORMAT, pax_headers={"comment": "1" * 40}) as tar:
        for relative in [*names.values(), runner.DECLARATION_PATH]:
            data = (tmp_path / relative).read_bytes()
            member = tarfile.TarInfo(relative)
            member.size = len(data)
            tar.addfile(member, io.BytesIO(data))
    release_value = {"schema": "rhuh-baseline-release-v1", "authorized": True,
        "study": manifest, "source_bindings": sources, "source_commit": "1" * 40,
        "source_archive": {"path": "source.tar", "sha256": runner.supervisor.sha(archive)},
        "interpreter": {"path": sys.executable, "sha256": runner.supervisor.sha(Path(sys.executable).resolve())},
        "dependency_manifest": dependency_binding, "output_directory": "outputs/constructed-attempt"}
    release = write_json("release.json", release_value)
    return tmp_path, manifest, release


def test_shared_numerical_kernels_are_bound_before_and_after_execution(constructed_preflight):
    root, manifest, release = constructed_preflight
    plan = runner.preflight(root, manifest, release)
    kernel = root / "shared/numpy/kernel.so"
    assert str(kernel) in plan["inputs"]
    kernel.write_bytes(b"constructed changed kernel\n")
    with pytest.raises(ValueError, match="changed"):
        runner.recheck(plan["inputs"])
    with pytest.raises(ValueError, match="runtime file changed"):
        runner.preflight(root, manifest, release)


def test_extra_numerical_file_is_rejected_before_source_data_use(constructed_preflight):
    root, manifest, release = constructed_preflight
    (root / "shared/scipy/extra.py").write_text("# unexpected source\n")
    with pytest.raises(ValueError, match="inventory changed"):
        runner.preflight(root, manifest, release)


def test_missing_archive_member_is_not_repaired_by_rehashing_release(constructed_preflight):
    root, manifest, release_binding = constructed_preflight
    release_file = root / release_binding["path"]
    release = json.loads(release_file.read_text())
    archive = root / "source.tar"
    with tarfile.open(archive, "r") as tar:
        members = [(m.name, tar.extractfile(m).read()) for m in tar if m.isfile()]
    with tarfile.open(archive, "w", format=tarfile.PAX_FORMAT, pax_headers={"comment": "1" * 40}) as tar:
        for name, data in members[:-1]:
            member = tarfile.TarInfo(name)
            member.size = len(data)
            tar.addfile(member, io.BytesIO(data))
    release["source_archive"]["sha256"] = runner.supervisor.sha(archive)
    release_file.write_bytes(runner.canonical(release))
    release_binding["sha256"] = runner.supervisor.sha(release_file)
    with pytest.raises(ValueError, match="Incomplete exact source archive"):
        runner.preflight(root, manifest, release_binding)
