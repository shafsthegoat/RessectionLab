"""Retain diagnostics when Finder starts the app without a terminal."""

from pathlib import Path
import sys


if sys.stdout is None or sys.stderr is None:
    log_dir = Path.home() / "Library" / "Logs" / "RessectionLab"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "desktop.log"
    if log_path.exists() and log_path.stat().st_size > 5_000_000:
        log_path.replace(log_dir / "desktop.previous.log")
    stream = log_path.open("a", buffering=1, encoding="utf-8")
    if sys.stdout is None:
        sys.stdout = stream
    if sys.stderr is None:
        sys.stderr = stream
