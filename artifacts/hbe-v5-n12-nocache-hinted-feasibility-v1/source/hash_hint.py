"""Diagnostic-only per-descriptor no-cache hint around the unchanged hash guard.

No global cache command or eviction is used. The supplied original file_hash
still owns both lstat calls, regular/size guards, 1 MiB SHA stream and
post-read identity check. F_NOCACHE failure aborts before reading that file.
"""
from __future__ import annotations

import fcntl
from pathlib import Path
import stat
import sys

MIN_HINT_BYTES = 16 * 1024**2


class NoCachePath:
    """Present the same existing file to the original hash guard with one hint."""

    def __init__(self, path: Path, audit: dict):
        self.path = path
        self.audit = audit

    def lstat(self):
        return self.path.lstat()

    def open(self, mode="r", *args, **kwargs):
        if mode != "rb":
            raise ValueError("No-cache hint is restricted to read-only binary hashing")
        stream = self.path.open(mode, *args, **kwargs)
        try:
            fcntl.fcntl(stream.fileno(), fcntl.F_NOCACHE, 1)
            self.audit["hinted_file_opens"] += 1
            self.audit["hinted_file_paths"].append(str(self.path))
            return stream
        except BaseException:
            stream.close()
            raise


def hinted_hash(path, *, maximum: int, original_hash, allowed_directories,
                audit: dict, minimum_bytes: int = MIN_HINT_BYTES) -> str:
    """Add only the per-fd hint for large, exact predecessor output files."""
    if sys.platform != "darwin" or getattr(fcntl, "F_NOCACHE", None) != 48:
        raise RuntimeError("Reviewed macOS F_NOCACHE=48 unavailable")
    actual = Path(path).absolute()
    allowed = {Path(directory).absolute() for directory in allowed_directories}
    before = actual.lstat()
    if (not stat.S_ISREG(before.st_mode) or before.st_size < minimum_bytes
            or before.st_size > maximum or actual.parent not in allowed):
        return original_hash(path, maximum=maximum)
    audit["eligible_bytes"] += before.st_size
    audit["eligible_files"] += 1
    return original_hash(NoCachePath(actual, audit), maximum=maximum)
