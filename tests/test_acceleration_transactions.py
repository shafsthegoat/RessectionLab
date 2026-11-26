"""``transaction-checks``: one file-identity scan per stroke preview, same results."""
import numpy as np
import pytest

from resectionlab import accel_transactions, acceleration, native_backend
from resectionlab.geometry import AccessWindow
from resectionlab.native_resection import NATIVE_GENERIC_TOOLS, NativeResectionConfig, NativeResectionEngine


def _config():
    tissue = np.zeros((21, 21, 30), bool)
    tissue[2:19, 2:19, 5:25] = True
    labels = np.zeros(tissue.shape, np.int16)
    labels[9:12, 9:12, 12:20] = 1
    entry = np.array([10, 10, 4.5])
    return NativeResectionConfig(tissue, labels, np.eye(4), AccessWindow(entry, np.array([0, 0, 1.]), 4),
                                 NATIVE_GENERIC_TOOLS, "analytic-source", "explicit synthetic box")


def _episode(*features):
    with acceleration.override(*features):
        engine = NativeResectionEngine(_config())
        records = []
        for tip in ([10, 10, 12], [10, 10, 15], [10, 10, 18], [10, 10, 21], [11, 10, 21]):
            preview = engine.preview_stroke(NATIVE_GENERIC_TOOLS[0].tool_id, tip)
            records.append((preview.feasible, preview.reason, preview.removed_volume_mm3))
            if preview.feasible:
                engine.commit_preview(preview)
        probe = engine.preview_stroke(NATIVE_GENERIC_TOOLS[0].tool_id, [10, 10, 16], interaction_mode="probe")
        records.append((probe.feasible, probe.reason))
        return records, engine.state_hash, engine.metrics()


@pytest.mark.parametrize('features', [('transaction-checks',), ('transaction-checks', 'stroke')])
def test_guarded_previews_and_commits_are_identical(features):
    assert _episode(*features) == _episode()


def test_guard_scans_files_once_on_entry_and_once_on_exit(monkeypatch):
    seen = []
    real = native_backend._check_live

    def counted(**kwargs):
        seen.append(getattr(native_backend._execution, 'depth', 0))
        return real(**kwargs)

    monkeypatch.setattr(native_backend, '_check_live', counted)
    depth = []
    wrapped = accel_transactions.guarded(lambda: depth.append(native_backend._execution.depth) or 7)
    with acceleration.override('transaction-checks'):
        assert wrapped() == 7
    assert seen == [0, 0] and depth == [1]  # Outer scans; inner calls see depth 1.
    seen.clear(); depth.clear()
    with acceleration.override():
        assert wrapped() == 7
    assert seen == [] and depth == [getattr(native_backend._execution, 'depth', 0)]


def test_guard_restores_depth_when_the_preview_raises():
    def failing():
        raise ValueError('refused')

    with acceleration.override('transaction-checks'):
        with pytest.raises(ValueError, match='refused'):
            accel_transactions.guarded(failing)()
    assert getattr(native_backend._execution, 'depth', 0) == 0
