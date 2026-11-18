"""Bounded process supervision and diagnostic collection for POC model attempts."""

import json
import os
import selectors
import signal
import subprocess
import time
from typing import NamedTuple


MAX_LOG_BYTES = 8 * 1024**2
LOG_HEAD_BYTES = MAX_LOG_BYTES // 2
LOG_TAIL_BYTES = MAX_LOG_BYTES - LOG_HEAD_BYTES
TRUNCATION = b"\n[Worker log truncated: keeping the beginning and end.]\n"
MPS_PREFIX = b"RESECTIONLAB_MPS_MEMORY "


class _AttemptLog:
    def __init__(self, stream):
        self.stream = stream
        self.head = 0
        self.tail = bytearray()
        self.bytes_seen = 0

    def append(self, chunk):
        self.bytes_seen += len(chunk)
        head = chunk[:LOG_HEAD_BYTES - self.head]
        self.stream.write(head)
        self.head += len(head)
        self.tail.extend(chunk[len(head):])
        if len(self.tail) > LOG_TAIL_BYTES:
            del self.tail[:-LOG_TAIL_BYTES]

    def finish(self):
        if self.bytes_seen > MAX_LOG_BYTES:
            self.stream.write(TRUNCATION)
            del self.tail[:len(TRUNCATION)]
        self.stream.write(self.tail)


class WorkerResult(NamedTuple):
    exit_code: int
    failure: str | None
    peak_rss: int
    log_truncated: bool


def _family_rss(worker_pid):
    # Include child SynthStrip processes. Shared Metal allocations are separate.
    listing = subprocess.run(["ps", "-axo", "pid=,ppid=,rss="], capture_output=True,
                             text=True, check=True, timeout=2)
    rows = [list(map(int, line.split())) for line in listing.stdout.splitlines() if line.strip()]
    family = {worker_pid}
    for _ in range(5):
        family |= {pid for pid, parent, _ in rows if parent in family}
    return sum(rss * 1024 for pid, _, rss in rows if pid in family)


def _terminate_group(process):
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    # A descendant can ignore TERM after the group leader has already exited.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def _drain(selector, attempt_log, timeout):
    for event, _ in selector.select(timeout):
        data = os.read(event.fd, 64 * 1024)
        if data:
            attempt_log.append(data)
        else:
            selector.unregister(event.fd)


def supervise_worker(command, environment, log_path, *, timeout, cancelled):
    deadline = time.monotonic() + timeout
    next_sample = 0.0
    peak_rss = 0
    failure = None
    with log_path.open("wb") as stream:
        attempt_log = _AttemptLog(stream)
        process = subprocess.Popen(command, env=environment, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            with process.stdout as pipe, selectors.DefaultSelector() as log_ready:
                os.set_blocking(pipe.fileno(), False)
                log_ready.register(pipe, selectors.EVENT_READ)
                while process.poll() is None:
                    now = time.monotonic()
                    if now >= next_sample:
                        if cancelled():
                            failure = "cancelled"
                            break
                        peak_rss = max(peak_rss, _family_rss(process.pid))
                        if peak_rss > 8 * 1024**3:
                            failure = "memory_budget_exceeded"
                            break
                        next_sample = time.monotonic() + 0.5
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        failure = "time_budget_exceeded"
                        break
                    _drain(log_ready, attempt_log, max(0, min(remaining, next_sample - time.monotonic())))
                if failure:
                    _terminate_group(process)
                # Collect the last diagnostics without waiting forever on an
                # inherited pipe that a stray descendant has kept open.
                drain_until = time.monotonic() + 1
                while log_ready.get_map() and (remaining := drain_until - time.monotonic()) > 0:
                    _drain(log_ready, attempt_log, remaining)
                if log_ready.get_map() and not failure:
                    _terminate_group(process)
                    failure = "worker_output_not_closed"
        except BaseException as error:
            _terminate_group(process)
            if isinstance(error, subprocess.SubprocessError):
                raise RuntimeError(f"Could not monitor model worker; inspect {log_path}") from error
            raise
        finally:
            attempt_log.finish()
    return WorkerResult(process.returncode, failure, peak_rss, attempt_log.bytes_seen > MAX_LOG_BYTES)


def mps_samples(log_path):
    """Read small complete diagnostic lines, even if an ordinary line is huge."""
    latest = None
    remaining = MAX_LOG_BYTES
    starts_line = True
    with log_path.open("rb") as log:
        while remaining and (chunk := log.readline(min(remaining, 4096))):
            remaining -= len(chunk)
            complete = chunk.endswith(b"\n")
            if starts_line and complete and chunk.startswith(MPS_PREFIX):
                try:
                    value = json.loads(chunk[len(MPS_PREFIX):])
                except (ValueError, RecursionError):
                    value = None
                if isinstance(value, dict) and set(value) == {
                    "tensor_bytes_max_sampled", "driver_bytes_max_sampled",
                } and all(type(count) is int and count >= 0 for count in value.values()):
                    latest = value
            starts_line = complete
    return latest
