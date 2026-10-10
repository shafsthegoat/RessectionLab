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

from darwin_fast_sampler import FastDarwinSampler


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


def _load_one_exact_process(path, expected_sha256):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("exact acquisition process manifest changed")
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
    manifest["manifest_sha256"] = expected_sha256
    return manifest


def _load_exact_json(path, expected_sha256):
    candidate = Path(path)
    if not candidate.is_absolute() or not candidate.is_file() or candidate.is_symlink():
        raise ValueError("terminal evidence must be an absolute regular non-symlink file")
    if not isinstance(expected_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ValueError("terminal evidence digest invalid")
    raw = candidate.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("terminal evidence changed")
    return json.loads(raw)


def verify_ixi_terminal(document):
    """Bind the completed IXI session to its exact byte and terminal receipts."""
    terminal = document["ixi_terminal_receipts"]
    if set(terminal) != {"canonical", "session", "status"}:
        raise ValueError("IXI terminal receipt set changed")
    receipts = {}
    for name in ("canonical", "session", "status"):
        entry = terminal[name]
        if set(entry) != {"path", "sha256"}:
            raise ValueError("IXI terminal receipt binding changed")
        receipts[name] = _load_exact_json(entry["path"], entry["sha256"])
    canonical, session, status = (receipts[name] for name in ("canonical", "session", "status"))
    expected_bytes = 17225109877
    if (canonical.get("status") != "all_bytes_verified" or
            canonical.get("verified_files") != 3 or canonical.get("verified_bytes") != expected_bytes or
            canonical.get("all_payloads_unreviewed") is not True or
            canonical.get("decoded_array_bytes") != 0 or canonical.get("training_admitted") is not False or
            canonical.get("labels_admitted") is not False or
            {item.get("path") for item in canonical.get("files", [])} !=
            {"IXI-MRA.tar", "IXI-T1.tar", "vessel_dataset.zip"} or
            any(item.get("status") != "byte_verified" for item in canonical.get("files", []))):
        raise ValueError("IXI canonical completion is not the exact byte-only result")
    if (session.get("schema") != "ixi-owned-session-terminal-observation-v1" or
            session.get("canonical_completion_sha256") != terminal["canonical"]["sha256"] or
            session.get("tool_exit_code") != 0 or
            session.get("original_process_absent_verified_by_ps") is not True or
            session.get("stdout_terminal_status") != canonical["status"] or
            session.get("stdout_verified_files") != canonical["verified_files"] or
            session.get("stdout_verified_bytes") != canonical["verified_bytes"] or
            session.get("pid") != document["ixi_original_pid"] or
            session.get("pgid") != document["ixi_original_pgid"] or
            session.get("session_id") != document["ixi_session_id"]):
        raise ValueError("IXI session terminal receipt does not match completed intake")
    if (status.get("schema") != "ixi-live-status-v1" or
            status.get("terminal_receipt_sha256") != terminal["canonical"]["sha256"] or
            status.get("process_alive") is not False or
            status.get("terminal_status") != canonical["status"] or
            status.get("verified_files") != canonical["verified_files"] or
            status.get("verified_bytes") != canonical["verified_bytes"] or
            status.get("all_payloads_unreviewed") is not True or
            status.get("pid") != session["pid"] or status.get("pgid") != session["pgid"] or
            status.get("session_id") != session["session_id"]):
        raise ValueError("IXI terminal status does not match verified session")
    return {"status": canonical["status"], "verified_files": canonical["verified_files"],
            "verified_bytes": canonical["verified_bytes"],
            "receipt_sha256": {name: terminal[name]["sha256"] for name in terminal}}


def load_acquisition_allowlist(path):
    """Require one exact live transfer and exact receipts for the completed IXI intake."""
    if path is None:
        return None
    raw = Path(path).read_bytes()
    document = json.loads(raw)
    required = {"schema", "tracto_manifest_path", "tracto_manifest_sha256",
                "tracto_kernel_start_identity", "ixi_terminal_receipts", "ixi_original_pid",
                "ixi_original_pgid", "ixi_session_id"}
    if set(document) != required or document["schema"] != "tracto-live-ixi-terminal-v1":
        raise ValueError("live/terminal acquisition manifest schema changed")
    digest = document["tracto_manifest_sha256"]
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("acquisition manifest digest invalid")
    candidate = Path(document["tracto_manifest_path"])
    if not candidate.is_absolute() or not candidate.is_file() or candidate.is_symlink():
        raise ValueError("acquisition manifest must be an absolute regular non-symlink file")
    tracto = _load_one_exact_process(candidate, digest)
    identity = document["tracto_kernel_start_identity"]
    if (not isinstance(identity, list) or len(identity) != 2 or
            any(type(value) is not int or value < 0 for value in identity) or identity[0] == 0):
        raise ValueError("acquisition kernel start identity invalid")
    if (document["ixi_original_pid"] != 4631 or document["ixi_original_pgid"] != 4631 or
            document["ixi_session_id"] != 95570):
        raise ValueError("IXI terminal process/session identity changed")
    terminal_summary = verify_ixi_terminal(document)
    if tracto["pid"] == document["ixi_original_pid"] or tracto["pgid"] == document["ixi_original_pgid"]:
        raise ValueError("acquisition identities must be distinct")
    return {"manifest_sha256": hashlib.sha256(raw).hexdigest(), "entries": {"tracto": tracto},
            "kernel_start_identities": {"tracto": tuple(identity)},
            "ixi_terminal_document": document, "ixi_terminal_summary": terminal_summary}


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
    observed_roles = set()
    evidence = {}
    if acquisition_allowlist is not None:
        # The completed IXI intake is proven by immutable receipts, not by PID absence.
        verify_ixi_terminal(acquisition_allowlist["ixi_terminal_document"])
        sampler = FastDarwinSampler()
        for role, manifest in acquisition_allowlist["entries"].items():
            try:
                start = run(["ps", "-p", str(manifest["pid"]), "-o", "lstart="]).strip()
            except subprocess.CalledProcessError:
                start = None
            evidence[role] = (
                start, sha256_file(manifest["source_path"]),
                sha256_file(manifest["declaration_path"]),
                sampler.process_start_identity(manifest["pid"]))
    for pid, pgid, rss, executable, arguments in processes:
        if pid in ancestors or (child_pgid is not None and pgid == child_pgid):
            continue
        if "/applications/chatgpt.app/" in arguments.lower():
            continue
        if acquisition_allowlist is not None:
            role = next((name for name, manifest in acquisition_allowlist["entries"].items()
                         if pid == manifest["pid"]), None)
            if role is not None:
                manifest = acquisition_allowlist["entries"][role]
                start, source_hash, declaration_hash, kernel_start = evidence[role]
                row = acquisition_identity(
                    (pid, pgid, rss, executable, arguments), manifest,
                    start, source_hash, declaration_hash)
                kernel_match = (kernel_start == acquisition_allowlist["kernel_start_identities"][role])
                row["kernel_start_identity_matched"] = kernel_match
                row["identity_matched"] = row["identity_matched"] and kernel_match
                row["blocking_overlap"] = not (row["identity_matched"] and row["rss_within_cap"])
                row["acquisition_role"] = role
                rows.append(row)
                observed_roles.add(role)
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
    if acquisition_allowlist is not None:
        for role, manifest in acquisition_allowlist["entries"].items():
            if role in observed_roles:
                continue
            rows.append({"pid": manifest["pid"], "pgid": manifest["pgid"],
                         "command_class": "acquisition_python", "acquisition_role": role,
                         "manifest_sha256": manifest["manifest_sha256"],
                         "identity_matched": False, "kernel_start_identity_matched": False,
                         "rss_within_cap": False, "max_rss_kib": manifest["max_rss_kib"],
                         "blocking_overlap": True, "reason": "allowlisted_live_pid_absent"})
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
            "ixi_terminal_summary": acquisition_allowlist["ixi_terminal_summary"] if acquisition_allowlist else None,
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
