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

from resectionlab import anatomy_model_identity, anatomy_preparation, poc_models, poc_worker_supervision


@pytest.fixture
def fake_runtime(tmp_path, monkeypatch):
    repository = tmp_path / "repo"
    output = repository / "testcases/model-runs"
    image = tmp_path / "image.nii.gz"
    nib.save(nib.Nifti1Image(np.ones((4, 4, 4), np.uint8), np.eye(4)), image)
    weights = repository / "data/models/synthstrip-v1/synthstrip.1.pt"
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"fixture; no model execution")
    runtime = tmp_path / "fake-runtime"
    runtime.write_text(f"#!{sys.executable}\n" + r'''
import json
import os
import shutil
import sys
from pathlib import Path

request = json.loads(Path(sys.argv[-1]).read_text())
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

    def rss(_):
        if termination == "monitor_error" and child_pid.exists():
            raise subprocess.CalledProcessError(1, "ps")
        return 0

    def alive(pid):
        result = subprocess.run(["ps", "-p", str(pid), "-o", "stat="], text=True, capture_output=True)
        state = result.stdout.strip()
        return bool(state) and not state.startswith("Z")

    monkeypatch.setattr(poc_worker_supervision, "_family_rss", rss)
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
    while time.monotonic() < deadline and any(alive(pid) for pid in pids):
        time.sleep(0.02)
    assert not any(alive(pid) for pid in pids)


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



def test_crop_admission_preserves_only_audited_model_asset_keys(monkeypatch):
    from resectionlab import anatomy_models

    adapter = poc_models.sha256(Path(anatomy_models.__file__))
    assert adapter == anatomy_model_identity.AUDITED_CROP_ADAPTER_SHA
    before = {"files": {"adapter": anatomy_model_identity.LEGACY_ADAPTER_SHA}}
    after = {"files": {"adapter": adapter}}
    assert anatomy_preparation._environment_identity(before) == anatomy_preparation._environment_identity(after)
    monkeypatch.setattr(anatomy_model_identity, "AUDITED_CROP_HELPER_SHA", "0" * 64)
    assert anatomy_preparation._environment_identity(after) != anatomy_preparation._environment_identity(before)
