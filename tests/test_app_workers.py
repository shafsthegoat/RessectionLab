"""A cancelled worker must not publish a result over the inspected case."""

import pytest

pytest.importorskip("PySide6.QtCore", reason="Historical Qt workers require the optional legacy Qt dependencies")

from resectionlab.app.workers import BackgroundJob


def test_cancellation_before_operation_prevents_execution():
    effects, finished, cancelled = [], [], []
    job = BackgroundJob(lambda event, progress: effects.append("executed"))
    job.signals.finished.connect(finished.append)
    job.signals.cancelled.connect(lambda: cancelled.append(True))
    job.cancel()
    job.run()
    assert not effects and not finished and cancelled == [True]


def test_cancellation_during_operation_discards_result():
    finished, cancelled = [], []
    def operation(event, progress):
        event.set()
        return "stale result"
    job = BackgroundJob(operation)
    job.signals.finished.connect(finished.append)
    job.signals.cancelled.connect(lambda: cancelled.append(True))
    job.run()
    assert not finished and cancelled == [True]


def test_failed_operation_emits_named_error():
    failures = []
    def operation(event, progress):
        raise ValueError("mask and MRI frames disagree")
    job = BackgroundJob(operation)
    job.signals.failed.connect(failures.append)
    job.run()
    assert failures == ["ValueError: mask and MRI frames disagree"]
