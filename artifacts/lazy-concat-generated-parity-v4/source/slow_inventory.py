"""Background-only project-process and VM diagnostic snapshots for Gate A v3."""

import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import threading
import time


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def run(args):
    return subprocess.run(args, capture_output=True, text=True, check=True, timeout=0.5).stdout


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_acquisition_allowlist(path):
    """Load a single, exact, predeclared acquisition process; never wildcard Python."""
    if path is None:
        return None
    raw = Path(path).read_bytes()
    manifest = json.loads(raw)
    required = {"pid", "pgid", "process_start_lstart", "ucomm", "command_sha256",
                "source_path", "source_command_token", "source_sha256",
                "declaration_path", "declaration_command_token", "declaration_sha256",
                "max_rss_kib"}
    if set(manifest) != required:
        raise ValueError("acquisition allowlist fields differ from frozen schema")
    if any(not isinstance(manifest[k], int) or manifest[k] <= 0
           for k in ("pid", "pgid", "max_rss_kib")):
        raise ValueError("acquisition allowlist PID/PGID/RSS invalid")
    if not re.fullmatch(r"python(?:\d+(?:\.\d+)?)?", manifest["ucomm"].lower()):
        raise ValueError("only a single exact Python acquisition wrapper may be allowlisted")
    if any(not isinstance(manifest[k], str) or not manifest[k]
           for k in ("process_start_lstart", "command_sha256", "source_path", "source_command_token",
                     "source_sha256", "declaration_path", "declaration_command_token",
                     "declaration_sha256")):
        raise ValueError("acquisition allowlist identity field missing")
    for key in ("command_sha256", "source_sha256", "declaration_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", manifest[key]):
            raise ValueError("acquisition allowlist digest invalid")
    for kind in ("source", "declaration"):
        path = Path(manifest[kind + "_path"])
        if not path.is_absolute() or not path.is_file() or path.is_symlink():
            raise ValueError("acquisition %s must be an absolute regular non-symlink file" % kind)
        if sha256_file(path) != manifest[kind + "_sha256"]:
            raise ValueError("acquisition %s digest changed" % kind)
    manifest["manifest_sha256"] = hashlib.sha256(raw).hexdigest()
    return manifest


def acquisition_identity(process, manifest, start_time, source_hash, declaration_hash):
    """Return a receipt-safe identity result; command bytes are never persisted."""
    pid, pgid, rss, executable, arguments = process
    matches = (pid == manifest["pid"] and pgid == manifest["pgid"] and
               executable == manifest["ucomm"].lower() and
               start_time == manifest["process_start_lstart"] and
               hashlib.sha256(arguments.encode("utf-8")).hexdigest() == manifest["command_sha256"] and
               manifest["source_command_token"] in arguments.split() and
               manifest["declaration_command_token"] in arguments.split() and
               manifest["declaration_sha256"] in arguments.split() and
               source_hash == manifest["source_sha256"] and
               declaration_hash == manifest["declaration_sha256"])
    return {"pid": pid, "pgid": pgid, "rss_kib": rss,
            "command_class": "acquisition_python", "manifest_sha256": manifest["manifest_sha256"],
            "identity_matched": bool(matches), "rss_within_cap": rss <= manifest["max_rss_kib"],
            "max_rss_kib": manifest["max_rss_kib"],
            "blocking_overlap": not (matches and rss <= manifest["max_rss_kib"])}


def collect(supervisor_pid, child_pgid=None, acquisition_allowlist=None):
    started = time.monotonic()
    parent = {}
    processes = []
    for line in run(["ps", "-axww", "-o", "pid=,ppid=,pgid=,rss=,ucomm=,command="]).splitlines():
        fields = line.strip().split(maxsplit=5)
        if len(fields) != 6:
            continue
        pid, ppid, pgid, rss, executable, arguments = fields
        pid, ppid, pgid, rss = int(pid), int(ppid), int(pgid), int(rss)
        parent[pid] = ppid
        processes.append((pid, pgid, rss, executable.lower(), arguments))
    ancestors = []
    cursor = supervisor_pid
    while cursor in parent and cursor not in ancestors:
        ancestors.append(cursor)
        cursor = parent[cursor]
    if cursor and cursor not in ancestors:
        ancestors.append(cursor)
    rows = []
    observed_acquisition = None
    acquisition_start = None
    acquisition_source_hash = None
    acquisition_declaration_hash = None
    if acquisition_allowlist is not None:
        acquisition_start = run(["ps", "-p", str(acquisition_allowlist["pid"]),
                                 "-o", "lstart="]).strip()
        source = Path(acquisition_allowlist["source_path"])
        acquisition_source_hash = sha256_file(source)
        acquisition_declaration_hash = sha256_file(acquisition_allowlist["declaration_path"])
    for pid, pgid, rss, executable, arguments in processes:
        if pid in ancestors or (child_pgid is not None and pgid == child_pgid):
            continue
        if "/applications/chatgpt.app/" in arguments.lower():
            continue
        if acquisition_allowlist is not None and pid == acquisition_allowlist["pid"]:
            observed_acquisition = acquisition_identity(
                (pid, pgid, rss, executable, arguments), acquisition_allowlist,
                acquisition_start, acquisition_source_hash, acquisition_declaration_hash)
            rows.append(observed_acquisition)
            continue
        arguments = arguments.lower()
        if "pytest" in executable or ("python" in executable and "pytest" in arguments):
            kind = "pytest"
        elif "febio" in executable:
            kind = "febio"
        elif re.fullmatch(r"python(?:\d+(?:\.\d+)?)?", executable):
            kind = "python"
        elif executable == "node":
            kind = "node"
        elif executable == "curl":
            kind = "curl"
        elif executable in ("zsh", "bash", "sh") and ("ressectionlab" in arguments or
                                                       any(t in arguments for t in ("scripts/", "tests/", "build/", ".tools/"))):
            kind = "project_shell"
        else:
            continue
        project = "ressectionlab" in arguments or any(t in arguments for t in (
            "scripts/", "tests/", "build/", "src/resectionlab", ".tools/"))
        rows.append({"pid": pid, "pgid": pgid, "rss_kib": rss,
                     "command_class": kind, "project_context": bool(project),
                     "blocking_overlap": bool(kind in ("pytest", "febio") or
                                              (project and kind != "curl"))})
    if acquisition_allowlist is not None and observed_acquisition is None:
        rows.append({"pid": acquisition_allowlist["pid"],
                     "pgid": acquisition_allowlist["pgid"], "command_class": "acquisition_python",
                     "manifest_sha256": acquisition_allowlist["manifest_sha256"],
                     "identity_matched": False, "rss_within_cap": False,
                     "max_rss_kib": acquisition_allowlist["max_rss_kib"],
                     "blocking_overlap": True, "reason": "allowlisted_pid_absent"})
    vm = run(["vm_stat"])
    page_size = re.search(r"page size of (\d+) bytes", vm)
    compressor = re.search(r"Pages occupied by compressor:\s*(\d+)\.", vm)
    pageouts = re.search(r"Pageouts:\s*(\d+)\.", vm)
    swapouts = re.search(r"Swapouts:\s*(\d+)\.", vm)
    if not all((page_size, compressor, pageouts, swapouts)):
        raise ValueError("vm_stat diagnostics parse failed")
    return {"timestamp_utc": utc_now(), "sample_monotonic": time.monotonic(),
            "collection_seconds": time.monotonic() - started,
            "excluded_ancestor_pids": ancestors,
            "project_process_inventory": rows,
            "acquisition_allowlist_manifest_sha256": acquisition_allowlist["manifest_sha256"] if acquisition_allowlist else None,
            "vm_page_bytes": int(page_size.group(1)),
            "compressor_occupied_pages": int(compressor.group(1)),
            "pageouts_counter": int(pageouts.group(1)),
            "swapouts_counter": int(swapouts.group(1))}


class BackgroundInventory:
    def __init__(self, supervisor_pid, interval_seconds=0.5, acquisition_allowlist=None):
        self.supervisor_pid = supervisor_pid
        self.interval_seconds = interval_seconds
        self.acquisition_allowlist = acquisition_allowlist
        self.child_pgid = None
        self.latest = None
        self.errors = []
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()

    def set_child_pgid(self, pgid):
        with self.lock:
            self.child_pgid = pgid

    def _run(self):
        while not self.stop_event.is_set():
            began = time.monotonic()
            with self.lock:
                pgid = self.child_pgid
            try:
                snapshot = collect(self.supervisor_pid, pgid, self.acquisition_allowlist)
                with self.lock:
                    self.latest = snapshot
            except Exception as error:
                with self.lock:
                    self.errors.append({"timestamp_utc": utc_now(),
                                        "type": type(error).__name__, "message": str(error)})
                    self.errors = self.errors[-20:]
            self.stop_event.wait(max(0, self.interval_seconds - (time.monotonic() - began)))

    def status(self):
        with self.lock:
            latest = self.latest
            errors = list(self.errors)
        age = time.monotonic() - latest["sample_monotonic"] if latest else None
        return latest, age, errors

    def close(self):
        self.stop_event.set()
        self.thread.join(timeout=1.5)
        if self.thread.is_alive():
            raise TimeoutError("background process inventory did not stop")
