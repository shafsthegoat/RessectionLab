"""Cancellable background operations with GUI-thread signal delivery."""

from threading import Event
from typing import Callable

from PySide6.QtCore import QObject, QRunnable, Signal


class JobSignals(QObject):
    progress = Signal(int, str)
    finished = Signal(object)
    failed = Signal(str)
    cancelled = Signal()


class BackgroundJob(QRunnable):
    """An operation receives (cancel_event, progress_callback).

    Cooperative cancellation is checked before publication as well as by the
    operation itself. Results from cancelled work never replace the open case.
    """

    def __init__(self, operation: Callable):
        super().__init__()
        self.operation = operation
        self.signals = JobSignals()
        self.cancel_event = Event()

    def cancel(self):
        self.cancel_event.set()

    def run(self):
        try:
            if self.cancel_event.is_set():
                self.signals.cancelled.emit()
                return
            result = self.operation(self.cancel_event, self.signals.progress.emit)
            if self.cancel_event.is_set():
                self.signals.cancelled.emit()
            else:
                self.signals.finished.emit(result)
        except Exception as error:
            if self.cancel_event.is_set():
                self.signals.cancelled.emit()
            else:
                self.signals.failed.emit(f"{type(error).__name__}: {error}")
