"""Independent analytic/read-boundary checks; never load patient records."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


SCRIPT = Path(__file__).parents[1] / "artifacts/mechanics/resect-case4-baseline-alignment-v1/align.py"


@pytest.fixture
def alignment():
    spec = importlib.util.spec_from_file_location("resect_alignment_independent", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def isolated(alignment, tmp_path, monkeypatch):
    """A fake research tree made only of constructed text and tiny byte files."""
    root = tmp_path / "fake-repository"
    here = root / "artifacts/mechanics/alignment"
    here.mkdir(parents=True)
    script = here / "align.py"
    script.write_text("# constructed source placeholder\n")
    monkeypatch.setattr(alignment, "ROOT", root)
    monkeypatch.setattr(alignment, "HERE", here)
    monkeypatch.setattr(alignment, "__file__", str(script))
    monkeypatch.setattr(alignment, "DATA", root / "data/mechanics/resect-case4-v1/RESECT/NIFTI/Case4")
    allowed = alignment.DATA / "baseline.bin"
    allowed.parent.mkdir(parents=True)
    allowed.write_bytes(b"constructed baseline only")
    hidden = alignment.DATA / "later-sealed.bin"
    hidden.write_bytes(b"constructed withheld file")
    justification = here / "coordinate-justification.json"
    write_json(justification, {"fixture": True})
    header = root / "header.json"
    write_json(header, {"fixture": True})
    snapshot = root / "snapshot.json"
    write_json(snapshot, {"files": []})
    declaration = {
        "baseline_header_receipt": str(header.relative_to(root)),
        "baseline_header_receipt_sha256": digest(header),
        "helper_snapshot_manifest": str(snapshot.relative_to(root)),
        "helper_snapshot_manifest_sha256": digest(snapshot),
        "files": [{"local_path": str(allowed.relative_to(root)), "sha256": digest(allowed),
                   "md5": hashlib.md5(allowed.read_bytes()).hexdigest(), "bytes": allowed.stat().st_size}],
        "budget": {"wall_seconds": 1, "RSS_bytes": 2**30},
    }
    write_json(here / "declaration.json", declaration)
    release = {"declaration_sha256": digest(here / "declaration.json"),
               "coordinate_justification_sha256": digest(justification), "script_sha256": digest(script)}
    write_json(here / "root-release.json", release)
    return alignment, declaration, hidden, snapshot


@pytest.mark.parametrize("edit_disk", [True, False])
def test_tampered_declaration_is_rejected_before_any_new_path_read(isolated, monkeypatch, edit_disk):
    module, declaration, hidden, _ = isolated
    declaration["files"].append({"local_path": str(hidden.relative_to(module.ROOT)),
        "sha256": digest(hidden), "md5": hashlib.md5(hidden.read_bytes()).hexdigest(),
        "bytes": hidden.stat().st_size})
    if edit_disk:
        write_json(module.HERE / "declaration.json", declaration)
    opened = []
    original = module.sha

    def watched(path):
        opened.append(Path(path))
        return original(path)

    monkeypatch.setattr(module, "sha", watched)
    with pytest.raises(RuntimeError):
        module.verify(declaration)
    assert hidden not in opened, "An unverified declaration opened a withheld path before rejection"


def test_tampered_snapshot_is_rejected_before_following_its_paths(isolated, monkeypatch):
    module, declaration, hidden, snapshot = isolated
    write_json(snapshot, {"files": [{"path": str(hidden.relative_to(module.ROOT)), "sha256": digest(hidden)}]})
    opened = []
    original = module.sha

    def watched(path):
        opened.append(Path(path))
        return original(path)

    monkeypatch.setattr(module, "sha", watched)
    with pytest.raises(RuntimeError):
        module.verify(declaration)
    assert hidden not in opened, "An unverified snapshot manifest opened a withheld path before rejection"


def test_patient_access_hook_denies_other_cases_and_source_writes(isolated):
    module, declaration, hidden, _ = isolated
    allowed = module.ROOT / declaration["files"][0]["local_path"]
    guard = module.make_patient_guard({allowed})
    guard("open", (str(allowed), "rb", os.O_RDONLY))
    guard("open", (str(module.ROOT / "build/fixture-module.py"), "rb", os.O_RDONLY))
    for path in (hidden, module.ROOT / "data/other-patient/image.nii.gz"):
        with pytest.raises(RuntimeError):
            guard("open", (str(path), "rb", os.O_RDONLY))
    for mode, flags in (("wb", os.O_WRONLY | os.O_TRUNC), ("rb", os.O_RDWR), ("r+b", os.O_RDONLY)):
        with pytest.raises(RuntimeError):
            guard("open", (str(allowed), mode, flags))


def test_patient_access_hook_rejects_allowlisted_symlink_escape(isolated):
    module, _, _, _ = isolated
    outside = module.ROOT / "unrelated-file.bin"
    outside.write_bytes(b"constructed unrelated bytes")
    link = module.DATA / "declared-but-symlink.bin"
    link.symlink_to(outside)
    with pytest.raises(RuntimeError):
        guard = module.make_patient_guard({link})
        guard("open", (str(link), "rb", os.O_RDONLY))


class FakeProcess:
    pid = 987654

    def __init__(self):
        self.killed = False
        self.polls = 0

    def poll(self):
        self.polls += 1
        return -9 if self.killed else None

    def kill(self):
        self.killed = True

    def wait(self, timeout=None):
        return -9 if self.killed else 0


@pytest.mark.parametrize("query_failure", ["timeout", "failed_query", "empty_query", "incomplete_query", "invalid_query"])
def test_failed_resource_monitor_stops_child_and_preserves_failure(isolated, monkeypatch, query_failure):
    module, _, _, _ = isolated
    child = FakeProcess()
    monkeypatch.setattr(module.sys, "argv", ["align.py", "--execute"])
    monkeypatch.setattr(module.sys, "addaudithook", lambda _: None)
    monkeypatch.setattr(module, "verify", lambda _: {})
    monkeypatch.setattr(module.subprocess, "Popen", lambda *a, **k: child)
    monkeypatch.setattr(module.time, "sleep", lambda _: None)

    queries = []

    def query(*args, **kwargs):
        queries.append(args)
        if query_failure == "timeout":
            raise module.subprocess.TimeoutExpired("ps", 2)
        return SimpleNamespace(returncode=1 if query_failure == "failed_query" else 0,
                               stdout={"incomplete_query": "1234\n", "invalid_query": "oops\n"}.get(query_failure, ""),
                               stderr="constructed query failure")

    monkeypatch.setattr(module.subprocess, "run", query)
    try:
        module.main()
    except (SystemExit, module.subprocess.TimeoutExpired):
        pass
    assert child.killed, "Unmonitored child continued after RSS query failure"
    assert len(queries) == 1, "RSS failure was retried without a declared monitoring fallback"
    receipt = json.loads((module.HERE / "resource-receipt.json").read_text())
    assert receipt["status"] == "failed"
    assert receipt["failure"]


def test_rotation_equivariance_and_no_reflection_scaling_solution(alignment):
    p = np.array([[0, 0, 0], [3, 0, 0], [0, 4, 0], [0, 0, 2], [2, -1, 1], [-2, 3, 5.]])
    truth = np.array([[0, -1, 0, 4], [1, 0, 0, -2], [0, 0, 1, 8], [0, 0, 0, 1.]])
    world = np.array([[0, 0, -1, 19], [1, 0, 0, 3], [0, -1, 0, -11], [0, 0, 0, 1.]])
    q = alignment.apply(p, truth)
    fitted, _, residual, loo = alignment.errors(alignment.apply(p, world), alignment.apply(q, world))
    np.testing.assert_allclose(fitted, world @ truth @ np.linalg.inv(world), atol=2e-14)
    assert max(residual.max(), loo.max()) < 1e-13
    scaled = alignment.rigid(p, 1.1 * p)
    np.testing.assert_allclose(np.linalg.svd(scaled[:3, :3])[1], 1, atol=1e-14)
    assert np.linalg.norm(alignment.apply(p, scaled) - 1.1 * p) > 0.1


def test_every_loo_fit_matches_independent_held_pair_omission(alignment):
    p = np.array([[0, 0, 0], [1, 0, 0], [0, 2, 0], [0, 0, 3], [1, 2, 3], [2, -1, 1.]])
    q = p + [0.4, -0.3, 1.1]
    q[1] += [0.5, -0.2, 0.4]
    _, _, _, loo = alignment.errors(p, q)
    for index in range(len(p)):
        retained = [i for i in range(len(p)) if i != index]
        prediction = alignment.apply(p[index:index + 1], alignment.rigid(p[retained], q[retained]))[0]
        assert loo[index] == pytest.approx(np.linalg.norm(prediction - q[index]), abs=1e-14)


def test_native_plane_uses_full_affine_and_inverse_fit_without_mutation(alignment):
    shape = (9, 8, 7)
    index = np.indices(shape, dtype=float)
    source = index[0] + 2 * index[1] + 3 * index[2]
    affine = np.array([[0, -2, 0.2, 30], [1.5, 0, 0, -17], [0.1, 0.3, -3, 4], [0, 0, 0, 1.]])
    original = affine.copy()
    fit = np.eye(4)
    fit[:3, 3] = [6, -2, 1]
    before_us_frame = fit @ affine
    sampled = alignment.sample_plane(source, affine, before_us_frame, shape, 1, 3, np.linalg.inv(fit))
    np.testing.assert_allclose(sampled[1:-1, 1:-1], source[:, 3, :][1:-1, 1:-1], atol=1e-12)
    np.testing.assert_array_equal(affine, original)
    out = alignment.sample_plane(source, affine, before_us_frame, shape, 1, 100, np.linalg.inv(fit))
    assert np.isnan(out).all()
