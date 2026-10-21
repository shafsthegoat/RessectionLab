"""Reproduce two initial v3 supervisor defects using frozen source AST only."""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import signal
import subprocess
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
INITIAL = ROOT/"artifacts/poc-ixi265-atropos-typed-mask-runner-guard-followup-v3/initial-runner.py.txt"
INITIAL_SHA = "dd7d71631c6724f519dd0af76f4f8338e260de9056f0c3f6c5fddae585085d85"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reproduce():
    assert sha(INITIAL) == INITIAL_SHA
    tree = ast.parse(INITIAL.read_text())
    run = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run")
    tries = [n for n in run.body if isinstance(n, ast.Try)]
    class Child:
        pid = 901
        def poll(self): return None
        def wait(self, timeout=None): return 0
    def gone(*args): raise ProcessLookupError("generated worker exited before killpg")
    record = {"status": "running"}
    scope = {"process": Child(), "os": SimpleNamespace(killpg=gone), "signal": signal, "subprocess": subprocess,
             "record": record, "error": RuntimeError("original resource failure")}
    escaped = False
    try:
        exec(compile(ast.fix_missing_locations(ast.Module(body=tries[0].handlers[0].body, type_ignores=[])), "<frozen-cleanup>", "exec"), scope)
    except ProcessLookupError: escaped = True
    assert escaped and record == {"status": "running"}
    cleanup = {"trigger": "Worker exits after poll reports running but before killpg; killpg raises ProcessLookupError.",
        "exception_escaped": escaped, "record_after_exception": record,
        "effect": "Original failure was not assigned; subsequent source/runtime/history postchecks and final receipt are skipped."}
    record = {"status": "fitted_candidate_unaccepted"}
    scope = {"record": record, "retained_markers": {"oldmarker": "hash"}, "sha": lambda _: "hash", "RUNTIME": Path("/generated-runtime"),
        "verify_runtime": lambda _: {"prefix": "alias-b"}, "base": None, "runtime_receipt": {"prefix": "alias-a"},
        "lineage": lambda: {"lineage": "hash"}, "LINEAGE": {"lineage": "hash"}}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[tries[1]], type_ignores=[])), "<frozen-continuity>", "exec"), scope)
    assert record["runtime_unchanged"] is False and record["status"] == "fitted_candidate_unaccepted"
    continuity = {"trigger": "Post-run verified runtime receipt differs from initial receipt while historical markers and lineage compare equal.",
        "record_after_postcheck": record, "effect": "False runtime continuity flag does not force failed status."}
    return [cleanup, continuity]


if __name__ == "__main__":
    destination = ROOT/"artifacts/ixi265-typed-mask-independent-v3/supervisor-initial-findings.json"
    assert not destination.exists()
    findings = reproduce()
    report = {"schema": "ixi265-typed-supervisor-initial-independent-findings-v3", "status": "two_prefit_defects_independently_reproduced",
        "created_utc": datetime.now(timezone.utc).isoformat(), "frozen_initial_source_sha256": INITIAL_SHA,
        "source_snapshot": str(INITIAL.relative_to(ROOT)), "findings": findings,
        "patient_or_native_calls": False, "method": "Execute exact AST-extracted frozen supervisor blocks with generated process/metadata doubles.",
        "original_preparation_sha256": sha(ROOT/"artifacts/poc-ixi265-atropos-typed-mask-runner-preparation-v3/preparation.json"),
        "auditor_sha256": sha(__file__), "supersession": "Requires separate guard followup and independent final review before any patient release; this finding record remains immutable."}
    content = json.dumps(report, indent=2)+"\n"; assert "/Users/" not in content
    with destination.open("x") as out: out.write(content)
    print(json.dumps({"artifact": str(destination.relative_to(ROOT)), "sha256": sha(destination), "findings": len(findings)}))
