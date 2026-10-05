"""Analytical and mocked integration only; no patient arrays or native runs."""
import hashlib
import io
import json
from types import SimpleNamespace
from pathlib import Path
import subprocess
import sys
import tarfile

import numpy as np
import pytest

from scripts import diagnose_training_observation_coverage as diagnostic
from resectionlab.native_resection import NativeResectionEngine


def test_preview_cap_violation_aborts_worker_before_any_next_case(tmp_path, monkeypatch):
    # A profiler exception disables profiling in CPython. The worker must stop,
    # not continue subsequent cases with a silently absent observer.
    calls, bodies = [], []
    def preview(self=None):
        bodies.append("entered")
    monkeypatch.setattr(NativeResectionEngine, "preview_stroke", preview)
    originals = {subject:{"coverage":{"historical":"constructed"}} for subject in diagnostic.SUBJECTS}
    monkeypatch.setattr(diagnostic,"validate_release",lambda *a:None)
    monkeypatch.setattr(diagnostic,"validate",lambda record:originals)
    monkeypatch.setattr(diagnostic,"peak_rss_bytes",lambda:0)
    monkeypatch.setattr(diagnostic,"original_records",lambda:originals)
    monkeypatch.setattr(diagnostic,"source_inventory",lambda:{})
    bundle = tmp_path/"analytical.fixture"; bundle.write_bytes(b"no patient arrays")
    member = {"case_bundle":str(bundle),"case_bundle_sha256":hashlib.sha256(bundle.read_bytes()).hexdigest()}
    record = {"members":{subject:{"member":member} for subject in diagnostic.SUBJECTS},"source_sha256":{}}
    def load(subject,*args):
        calls.append(subject)
        for _ in range(diagnostic.SETTINGS["max_initial_previews_per_case"]+1):
            preview()
        return object()
    monkeypatch.setattr(diagnostic,"load_initial",load)
    monkeypatch.setattr(diagnostic,"inspect_task",lambda *args:{"status":"complete"})
    with pytest.raises(RuntimeError):
        diagnostic.worker(record,tmp_path,manifest_sha256="0"*64,release={})
    assert calls == ["sub-PAT05"]
    assert len(bodies) == 78
    report = json.loads((tmp_path/"receipt.json").read_text())
    assert report["status"] == "incomplete"
    assert report["cohort_denominator"] == 6 and report["new_task_attempts"] == 1
    assert report["subjects"]["sub-PAT22"]["status"] == "not_attempted"
    assert all(report["subjects"][subject]["status"] == "historical_support_block" for subject in diagnostic.BLOCKED)


def test_release_commit_and_manifest_must_match_archived_bytes_before_data_access(tmp_path, monkeypatch):
    source, manifest = b"# tiny analytical source\n", b'{"analytical":true}\n'
    archive = tmp_path / "source.tar"
    with tarfile.open(archive, "w", format=tarfile.PAX_FORMAT, pax_headers={"comment":"a"*40}) as stream:
        for name,data in (("source.py",source),(diagnostic.MANIFEST,manifest)):
            member = tarfile.TarInfo(name); member.size = len(data)
            stream.addfile(member, io.BytesIO(data))
    monkeypatch.setattr(diagnostic, "ROOT", tmp_path)
    monkeypatch.setattr(diagnostic, "source_inventory", lambda:{"source.py":hashlib.sha256(source).hexdigest()})
    manifest_sha = hashlib.sha256(manifest).hexdigest()
    release = {"version":diagnostic.RELEASE_VERSION,"authorized":True,"manifest_sha256":manifest_sha,
               "runtime_root":str(tmp_path),"source_commit":"a"*40,"source_archive":str(archive),
               "source_archive_sha256":hashlib.sha256(archive.read_bytes()).hexdigest()}
    diagnostic.validate_release(release,manifest_sha)
    with pytest.raises(ValueError,match="commit"):
        diagnostic.validate_release({**release,"source_commit":"b"*40},manifest_sha)
    with pytest.raises(ValueError,match="manifest"):
        diagnostic.validate_release({**release,"manifest_sha256":"0"*64},"0"*64)


def test_cold_geometry_helper_import_is_not_optimizer_execution():
    code = '''from scripts import diagnose_training_observation_coverage as diagnostic
import sys
assert 'torch' not in sys.modules
with diagnostic.InitialInventoryGuard(lambda: None):
    from resectionlab.spatial_policy import world_to_sample_grid
'''
    result = subprocess.run([sys.executable,"-c",code],cwd=Path(__file__).resolve().parents[1],
                            text=True,capture_output=True,timeout=30)
    assert result.returncode == 0, result.stderr


def test_fractional_sample_coverage_and_center_cell_intervals_remain_distinct():
    affine = np.array([[0.,-3.,0.,17.],[-2.,0.,0.,-21.],[0.,0.,4.,6.],[0.,0.,0.,1.]])
    first = (affine @ [-1.,1.,1.,1.])[:3]
    last = (affine @ [3.,1.,1.,1.])[:3]
    coverage = np.full((6,3,3,3),.25)
    coverage[4:] = 0
    result = diagnostic.segment_visibility(first,last,affine,(3,3,3),coverage)
    assert result["sample_inside_center_domain"] == [False,True,True,True,False]
    assert result["continuous_center_domain_fraction"] == pytest.approx(.5,abs=1e-12)
    assert result["continuous_fullcell_extent_fraction"] == pytest.approx(.75,abs=1e-12)
    samples = np.asarray(result["sample_channel_coverage_fraction"])
    np.testing.assert_array_equal(samples[[0,4]],0)
    np.testing.assert_allclose(samples[1:4,:4],.25,rtol=0,atol=1e-12)
    np.testing.assert_array_equal(samples[:,4:],0)


@pytest.mark.parametrize("subject", ["sub-PAT16","sub-PAT20","sub-PAT26","sub-PAT27","sub-PAT29","sub-PAT31"])
def test_blocked_or_nontraining_case_rejected_before_check_or_decode(subject):
    def forbidden():
        pytest.fail("Rejected member must not reach loader/check callback")
    with pytest.raises(ValueError):
        diagnostic.load_initial(subject,{}, {},forbidden)


def test_union_intervals_merge_overlap_without_double_counting():
    assert diagnostic._union_length([[0.,.7],[.2,.8],None]) == pytest.approx(.8)
    assert diagnostic._union_length([[0.,.2],[.8,1.]]) == pytest.approx(.4)
    assert diagnostic._union_length([None,None]) == 0


def test_full_source_projection_does_not_read_reference_rewards_or_crop_geometry():
    from resectionlab.spatial_multiscale_views import prepare_multiscale_views
    from resectionlab.spatial_observations import CHANNEL_NAMES
    from test_spatial_multiscale_views_review import source
    permitted = source()
    class Case:
        structural_intensity = permitted.channels["structural_intensity"].data
        observed_support = permitted.channels["nominal_tissue"].data
        nominal_target = permitted.channels["nominal_target"].data
        intensity_normalization = "raw"
        _native_affine_ras_mm = permitted.affine_ras_mm
        track = permitted.track
        source_hash = "mixed-case-provenance-only"
        @property
        def reference_target(self):
            pytest.fail("Hidden reference must never enter view preparation")
        @property
        def _crop_origin(self):
            pytest.fail("Alternative view location must not use original access crop")
    observation = SimpleNamespace(channel_provenance={name:dict(permitted.channels[name].provenance) for name in CHANNEL_NAMES})
    task = SimpleNamespace(case=Case(),_engine=SimpleNamespace(removed_mask=permitted.channels["observed_cavity"].data))
    extracted = diagnostic.full_source_from_task(task,observation)
    # Full native data have complete coverage in this adapter; the separate
    # generic source constructor handles unknown-coverage masks.
    assert extracted.channels["nominal_target"].data.shape == (5,3,4)
    result = prepare_multiscale_views(extracted,local_shape=(3,2,2),coarse_shape=(2,2,3))
    task.case.source_hash = "different-hidden-audit-label"
    other = prepare_multiscale_views(diagnostic.full_source_from_task(task,observation),local_shape=(3,2,2),coarse_shape=(2,2,3))
    assert result.target_local.fingerprint == other.target_local.fingerprint
    assert result.whole_source.fingerprint == other.whole_source.fingerprint


@pytest.mark.parametrize("drift", ["before_bundle", "after_source"])
def test_source_and_bundle_drift_never_publish_complete(tmp_path, monkeypatch, drift):
    originals = {subject:{"coverage":{"historical":"constructed"}} for subject in diagnostic.BLOCKED}
    bundle = tmp_path/"analytical.fixture"; bundle.write_bytes(b"not patient arrays")
    member = {"case_bundle":str(bundle),"case_bundle_sha256":hashlib.sha256(bundle.read_bytes()).hexdigest()}
    record = {"members":{subject:{"member":member} for subject in diagnostic.SUBJECTS},"source_sha256":{"source":"old"}}
    monkeypatch.setattr(diagnostic,"validate_release",lambda *a:None)
    monkeypatch.setattr(diagnostic,"validate",lambda *a:originals)
    monkeypatch.setattr(diagnostic,"original_records",lambda:originals)
    monkeypatch.setattr(diagnostic,"peak_rss_bytes",lambda:0)
    monkeypatch.setattr(diagnostic,"source_inventory",lambda:{"source":"changed"})
    calls=[]
    monkeypatch.setattr(diagnostic,"load_initial",lambda subject,*a:calls.append(subject))
    monkeypatch.setattr(diagnostic,"inspect_task",lambda *a:{"status":"complete","zero_updates":True})
    if drift == "before_bundle":
        bundle.write_bytes(b"changed analytical bytes")
    with pytest.raises(ValueError,match="changed"):
        diagnostic.worker(record,tmp_path,manifest_sha256="0"*64,release={})
    report=json.loads((tmp_path/"receipt.json").read_text())
    assert report["status"] == "incomplete" and report["cohort_denominator"] == 6
    assert report["new_task_attempts"] == (0 if drift == "before_bundle" else 4)
    assert calls == ([] if drift == "before_bundle" else [s for s in diagnostic.SUBJECTS if s not in diagnostic.BLOCKED])
    if drift == "after_source":
        assert report["sources_unchanged"] is False
