"""Exercise the real launcher and process-group cleanup using generated workers."""

import json
import os
from pathlib import Path
import subprocess
import sys
import time

import nibabel as nib
import numpy as np
import pytest

from resectionlab import anatomy_model_identity, anatomy_preparation, poc_ct_recovery, poc_models, poc_worker_supervision


@pytest.fixture
def fake_runtime(tmp_path, monkeypatch):
    repository = tmp_path / "repo"
    output = repository / "testcases/model-runs"
    image = tmp_path / "image.nii.gz"
    nib.save(nib.Nifti1Image(np.ones((4, 4, 4), np.uint8), np.eye(4)), image)
    weights = repository / "data/models/synthstrip-v1/synthstrip.1.pt"
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"fixture; no model execution")
    ct_weights = repository / "data/models/poc-totalsegmentator/weights/fixture/checkpoint_final.pth"
    ct_weights.parent.mkdir(parents=True)
    ct_weights.write_bytes(b"fixture CT weights")
    runtime = tmp_path / "fake-runtime"
    runtime.write_text(f"#!{sys.executable}\n" + r'''
import json
import os
import shutil
import sys
from pathlib import Path

request = json.loads(Path(sys.argv[-1]).read_text())
if request["model_id"] == "totalsegmentator-skull":
    output = Path(request["output"])
    for name in ("brain_mask.nii.gz", "skull_mask.nii.gz"):
        shutil.copyfile(request["image"], output / name)
    (output / "worker.json").write_text(json.dumps({"compute_seconds": 0.001, "runtime_versions": {}}))
    raise SystemExit(0)
os.write(1, b"first worker diagnostic\n")
for _ in range(144):
    os.write(1, b"x" * (64 * 1024))
os.write(2, b"\nRESECTIONLAB_MPS_MEMORY {not json}\n")
os.write(2, b'RESECTIONLAB_MPS_MEMORY {"tensor_bytes_max_sampled": 11, "driver_bytes_max_sampled": 22}\n')
os.write(2, b"last worker diagnostic\n")
if request["case"] == "failed":
    raise SystemExit(19)
shutil.copyfile(request["image"], Path(request["output"]) / "brain_mask.nii.gz")
''')
    runtime.chmod(0o700)
    monkeypatch.setattr(poc_models, "ROOT", repository)
    monkeypatch.setattr(poc_models, "RUNTIME", runtime)
    for key in ("RESSECTIONLAB_MODEL_ROOT", "RESSECTIONLAB_DATA_ROOT", "RESSECTIONLAB_POC_OUTPUT_ROOT"):
        monkeypatch.delenv(key, raising=False)
    return image, output


def _alive(pid):
    result = subprocess.run(["ps", "-p", str(pid), "-o", "stat="], text=True, capture_output=True)
    state = result.stdout.strip()
    return bool(state) and not state.startswith("Z")


@pytest.mark.parametrize("exit_code,inherit_pipe", [(0, True), (19, True), (0, False)])
def test_exited_worker_reaps_helpers_without_reclassifying_exit(tmp_path, exit_code, inherit_pipe):
    child_pid = tmp_path / "child.pid"
    child = '''
import os, signal, sys, time
from pathlib import Path
signal.signal(signal.SIGTERM, signal.SIG_IGN)
Path(sys.argv[1]).write_text(str(os.getpid()))
time.sleep(60)
'''
    parent = '''
import subprocess, sys, time
from pathlib import Path
path, code, status, inherit = sys.argv[1:]
redirect = None if inherit == "True" else subprocess.DEVNULL
subprocess.Popen([sys.executable, "-c", code, path], stdout=redirect, stderr=redirect)
while not Path(path).exists():
    time.sleep(.01)
print("leader-completed", flush=True)
raise SystemExit(int(status))
'''
    log = tmp_path / "worker.log"
    result = poc_worker_supervision.supervise_worker(
        [sys.executable, "-c", parent, str(child_pid), child, str(exit_code), str(inherit_pipe)],
        os.environ, log, timeout=5, cancelled=lambda: False)
    assert result.exit_code == exit_code and result.failure is None
    assert "leader-completed" in log.read_text()
    assert ("inherited log pipe" in log.read_text()) is inherit_pipe
    pid = int(child_pid.read_text())
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline and _alive(pid):
        time.sleep(.02)
    assert not _alive(pid)


def _run_ct(image, output):
    return poc_models._run(image, "ct-case", "totalsegmentator-skull", "CT", device="cpu",
                           timeout=10, output_root=output)


def test_historical_ct_pipe_failure_recovers_exact_source_copy(fake_runtime, monkeypatch):
    image, output = fake_runtime
    original = _run_ct(image, output)
    receipt = next(output.rglob("inference.json"))
    original.update(status="failed", failure="worker_output_not_closed", outputs=[],
                    launcher_sha256=anatomy_model_identity.PRE_RECOVERY_LAUNCHER_SHA)
    receipt.write_text(json.dumps(original))
    previous = receipt.read_bytes()
    copied = image.with_name("relocated-exact-source.nii.gz")
    copied.write_bytes(image.read_bytes())
    monkeypatch.setattr(poc_worker_supervision.subprocess, "Popen",
                        lambda *a, **k: pytest.fail("Historical zero-exit attempt should be reused"))
    observed = _run_ct(copied, output)
    assert observed["status"] == "succeeded" and observed["failure"] is None
    assert observed["recovery"]["status"] == "validated_after_log_pipe_cleanup_failure"
    assert observed["recovery"]["original_receipt_sha256"] == poc_models.sha256(receipt)
    assert [row["foreground_voxels"] for row in observed["outputs"]] == [64, 64]
    assert receipt.read_bytes() == previous


@pytest.mark.parametrize("empty", (False, True))
def test_completed_ct_checkpoint_survives_interrupted_validation(fake_runtime, monkeypatch, empty):
    image, output = fake_runtime
    if empty:
        nib.save(nib.Nifti1Image(np.zeros((4, 4, 4), np.uint8), np.eye(4)), image)

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    with monkeypatch.context() as scoped:
        scoped.setattr(poc_ct_recovery, "ct_masks", interrupt)
        with pytest.raises(KeyboardInterrupt):
            _run_ct(image, output)
    receipt = next(output.rglob("inference.json"))
    assert json.loads(receipt.read_text())["status"] == "pending_validation"
    monkeypatch.setattr(poc_worker_supervision.subprocess, "Popen",
                        lambda *a, **k: pytest.fail("Completed attempt must not launch twice"))
    recovered = _run_ct(image, output)
    assert recovered["recovery"]["status"] == "validated_from_completed_worker_checkpoint"
    assert [row["foreground_voxels"] for row in recovered["outputs"]] == ([0, 0] if empty else [64, 64])


@pytest.mark.parametrize("problem", ("empty", "different-grid", "units", "missing", "exit", "unknown-launcher"))
def test_historical_ct_recovery_refuses_incomplete_or_unbound_results(fake_runtime, problem):
    image, output = fake_runtime
    record = _run_ct(image, output)
    receipt = next(output.rglob("inference.json"))
    record.update(status="failed", failure="worker_output_not_closed", outputs=[],
                  launcher_sha256=anatomy_model_identity.PRE_RECOVERY_LAUNCHER_SHA)
    mask = receipt.with_name("brain_mask.nii.gz")
    if problem == "empty":
        nib.save(nib.Nifti1Image(np.zeros((4, 4, 4), np.uint8), np.eye(4)), mask)
    elif problem == "different-grid":
        affine = np.eye(4)
        affine[0, 3] = 1
        nib.save(nib.Nifti1Image(np.ones((4, 4, 4), np.uint8), affine), mask)
    elif problem == "units":
        different = nib.Nifti1Image(np.ones((4, 4, 4), np.uint8), np.eye(4))
        different.header.set_xyzt_units("meter")
        nib.save(different, mask)
    elif problem == "missing":
        mask.unlink()
    elif problem == "exit":
        record["exit_code"] = 1
    else:
        record["launcher_sha256"] = "0" * 64
    receipt.write_text(json.dumps(record))
    with pytest.raises(ValueError):
        poc_ct_recovery.recover_ct_worker(
            receipt, cache_root=output, image=image, source_hash=poc_models.sha256(image),
            modality="CT", model=poc_models.MODEL_CONFIG["totalsegmentator-skull"],
            expected_identity=record["identity"], expected_weights=record["weights"])


@pytest.mark.parametrize("case", ["finished", "failed"])
def test_worker_attempt_caps_retained_log_and_keeps_last_diagnostic(fake_runtime, monkeypatch, case):
    image, output = fake_runtime
    if case == "failed":
        with pytest.raises(RuntimeError, match="worker_exit_19"):
            poc_models._run(image, case, "synthstrip-main", "T1", device="cpu", timeout=10, output_root=output)
    else:
        poc_models._run(image, case, "synthstrip-main", "T1", device="cpu", timeout=10, output_root=output)
    receipt = next((output / case).rglob("inference.json"))
    record = json.loads(receipt.read_text())
    log = receipt.with_name("inference.log")
    assert log.stat().st_size == poc_worker_supervision.MAX_LOG_BYTES
    with log.open("rb") as stream:
        assert stream.read(32).startswith(b"first worker diagnostic\n")
        stream.seek(poc_worker_supervision.LOG_HEAD_BYTES)
        assert stream.read(len(poc_worker_supervision.TRUNCATION)) == poc_worker_supervision.TRUNCATION
        stream.seek(-64, os.SEEK_END)
        assert stream.read().endswith(b"last worker diagnostic\n")
    assert record["log_truncated"]
    assert record["synthstrip_mps_allocator_samples"] == {
        "tensor_bytes_max_sampled": 11, "driver_bytes_max_sampled": 22,
    }
    assert record["identity"]["runner_sha256"] == anatomy_model_identity.LEGACY_POC_LAUNCHER_SHA
    assert record["launcher_sha256"] == anatomy_model_identity.AUDITED_LAUNCHER_SHA
    if case == "finished":
        monkeypatch.setattr(poc_worker_supervision.subprocess, "Popen",
                            lambda *args, **kwargs: pytest.fail("Compatible result should be cached"))
        assert poc_models._run(image, case, "synthstrip-main", "T1", device="cpu", timeout=10,
                               output_root=output) == record
    else:
        assert record["status"] == "failed" and record["exit_code"] == 19


@pytest.mark.parametrize("termination", ["cancelled", "monitor_error"])
def test_supervisor_cleans_up_worker_and_stubborn_descendant(tmp_path, monkeypatch, termination):
    child_pid = tmp_path / "child.pid"
    leader_pid = tmp_path / "leader.pid"
    child_code = '''
import os, signal, sys, time
from pathlib import Path
signal.signal(signal.SIGTERM, signal.SIG_IGN)
Path(sys.argv[1]).write_text(str(os.getpid()))
time.sleep(60)
'''
    leader_code = '''
import os, subprocess, sys, time
from pathlib import Path
Path(sys.argv[1]).write_text(str(os.getpid()))
print("starting child", flush=True)
subprocess.Popen([sys.executable, "-c", sys.argv[3], sys.argv[2]])
time.sleep(60)
'''

    def memory(_, measure_footprint):
        if termination == "monitor_error" and child_pid.exists():
            raise subprocess.CalledProcessError(1, "ps")
        return 0, None

    monkeypatch.setattr(poc_worker_supervision, "_family_memory", memory)
    log = tmp_path / "worker.log"

    def run():
        return poc_worker_supervision.supervise_worker(
            [sys.executable, "-c", leader_code, str(leader_pid), str(child_pid), child_code],
            os.environ, log, timeout=5, cancelled=lambda: termination == "cancelled" and child_pid.exists())

    if termination == "monitor_error":
        with pytest.raises(RuntimeError, match="Could not monitor model worker"):
            run()
    else:
        assert run().failure == "cancelled"
    assert log.read_text() == "starting child\n"
    pids = [int(path.read_text()) for path in (leader_pid, child_pid)]
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline and any(_alive(pid) for pid in pids):
        time.sleep(0.02)
    assert not any(_alive(pid) for pid in pids)


def test_supervision_only_compatibility_is_bound_to_exact_audited_sources(monkeypatch):
    legacy = anatomy_model_identity.LEGACY_POC_LAUNCHER_SHA
    installed = poc_models.sha256(Path(poc_models.__file__))
    assert installed == anatomy_model_identity.AUDITED_LAUNCHER_SHA
    before = {"files": {"poc_worker": legacy, "adapter": "other assets"}}
    after = {"files": {"poc_worker": installed, "adapter": "other assets"}}
    assert anatomy_preparation._environment_identity(before) == anatomy_preparation._environment_identity(after)
    assert anatomy_model_identity.poc_worker_identity("0" * 64) != legacy
    monkeypatch.setattr(anatomy_model_identity, "AUDITED_SUPERVISOR_SHA", "0" * 64)
    assert anatomy_preparation._environment_identity(after) != anatomy_preparation._environment_identity(before)


@pytest.mark.parametrize("rss_gib, footprint_gib", [(2, 9), (9, 2)])
def test_cpu_synthstrip_records_either_memory_limit(fake_runtime, monkeypatch, rss_gib, footprint_gib):
    image, output = fake_runtime
    gib = 1024**3
    monkeypatch.setattr(poc_models.platform, "system", lambda: "Darwin")

    def sample(_, measure_footprint):
        assert measure_footprint
        return rss_gib * gib, footprint_gib * gib

    monkeypatch.setattr(poc_worker_supervision, "_family_memory", sample)
    with pytest.raises(RuntimeError, match="memory_budget_exceeded"):
        poc_models._run(image, "bounded", "synthstrip-main", "T1", device="cpu", timeout=10, output_root=output)
    record = json.loads(next(output.rglob("inference.json")).read_text())
    assert record["memory_guard"] == {"metric": record["identity"]["memory_guard"], "limit_bytes": 8 * gib,
                                      "peak_family_phys_footprint_bytes_sampled": footprint_gib * gib,
                                      "peak_family_rss_bytes_sampled": rss_gib * gib}
    assert record["status"] == "failed" and record["failure"] == "memory_budget_exceeded"


@pytest.mark.skipif(sys.platform != "darwin", reason="Darwin libproc memory accounting")
def test_darwin_worker_footprint_measures_touched_pages(tmp_path, monkeypatch):
    # Exercise the real family and libproc sampler with a small controlled
    # allocation. The production budget remains 8 GiB.
    monkeypatch.setattr(poc_worker_supervision, "MEMORY_LIMIT_BYTES", 64 * 1024**2)
    code = "import time; data=bytearray(b'x' * (80 * 1024**2)); time.sleep(5)"
    result = poc_worker_supervision.supervise_worker(
        [sys.executable, "-c", code], os.environ, tmp_path / "worker.log",
        timeout=4, cancelled=lambda: False, measure_footprint=True)
    assert result.failure == "memory_budget_exceeded" and result.peak_footprint > 64 * 1024**2


def test_mps_synthstrip_keeps_existing_cache_identity(fake_runtime):
    image, output = fake_runtime
    record = poc_models._run(image, "finished", "synthstrip-main", "T1", device="mps", timeout=10, output_root=output)
    assert {"image_sha256", "weights", "model_id", "parameters", "device", "runner_sha256"} == set(record["identity"])
    assert record["identity"]["runner_sha256"] == anatomy_model_identity.LEGACY_POC_LAUNCHER_SHA
    assert "memory_guard" not in record



def test_crop_admission_preserves_only_audited_model_asset_keys(monkeypatch):
    from resectionlab import anatomy_models

    adapter = poc_models.sha256(Path(anatomy_models.__file__))
    assert adapter == anatomy_model_identity.SIAM_DISPATCH_SHA
    before = {"files": {"adapter": anatomy_model_identity.LEGACY_ADAPTER_SHA}}
    after = {"files": {"adapter": adapter}}
    assert anatomy_preparation._environment_identity(before) == anatomy_preparation._environment_identity(after)
    monkeypatch.setattr(anatomy_model_identity, "AUDITED_CROP_HELPER_SHA", "0" * 64)
    assert anatomy_preparation._environment_identity(after) != anatomy_preparation._environment_identity(before)
