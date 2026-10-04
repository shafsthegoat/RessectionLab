"""Bind the already completed synthetic review to local source and saved logs."""
from pathlib import Path
import hashlib
import json
import subprocess

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).resolve().parent
PROTECTED = ["src/resectionlab/" + name + ".py" for name in (
    "experimental_capsule_cache", "geometry", "native_resection", "native_simulation",
    "native_axis_simulation", "native_proposals", "evaluation")]
FILES = PROTECTED + [
    "scripts/probe_public_native_capsule_cache.py",
    "scripts/preflight_native_axis.py",
    "manifests/experiments/native-axis-capsule-cache-v1.json",
    "tests/test_public_capsule_probe_review.py",
    "artifacts/public-capsule-probe-review-v1/pre-review-script.py",
    "artifacts/public-capsule-probe-independent-v1/reproduce_initial_failures.py",
    "artifacts/public-capsule-probe-independent-v1/initial-negative-tests.txt",
    "artifacts/public-capsule-probe-independent-v1/repaired-tests.txt",
    "artifacts/public-capsule-probe-independent-v1/repaired-initial-harness-tests.txt",
]


if __name__ == "__main__":
    initial = (OUTPUT / "initial-negative-tests.txt").read_text()
    repaired = (OUTPUT / "repaired-tests.txt").read_text()
    assert "5 failed in 0.33s" in initial
    assert "25 passed in 1.47s" in repaired
    hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in FILES}
    unchanged = {name: subprocess.check_output(["git", "show", "39fc208:" + name], cwd=ROOT)
                 == (ROOT / name).read_bytes() for name in PROTECTED}
    assert all(unchanged.values()), unchanged
    declaration = json.loads((ROOT / FILES[len(PROTECTED) + 2]).read_text())
    receipt = {
        "status": "passed_synthetic_review_only",
        "head_at_receipt": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "protected_baseline": "39fc208", "protected_source_byte_identity": unchanged,
        "file_sha256": hashes,
        "declaration_content_hash": declaration["declaration_content_hash"],
        "observed_initial_regressions": {
            "count": 5, "result": "5 failed in 0.33s", "expected_nonzero_exit": 1,
            "findings": [
                "Common uncached template preview count did not enforce the declared denominator.",
                "Measurement-hook exit failure restored the hook but omitted the per-phase failure receipt.",
                "Launcher accepted a failed result payload when worker completion/hash otherwise matched.",
                "Launcher accepted a mismatching worker runtime when the result runtime matched.",
                "Malformed result JSON escaped parsing and omitted explicit failed launcher publication.",
            ],
        },
        "repaired_independent_tests": {"count": 25, "result": "25 passed in 1.47s", "exit_code": 0},
        "retained_harness_error": {
            "result": "1 failed, 24 passed in 1.52s",
            "cause": "Review assertion compared serialized JSON lists with in-memory tuple history directly.",
            "repair": "Compare the complete histories through the existing exact canonical serializer.",
            "production_change": False,
        },
        "verified": [
            "Four synthetic native replays agree exactly on scientific traces, certificates, observations, masks, cavity identities and work.",
            "Cold cache is created empty after the first reference; warm begins with cold's retained state; final reference does not increment it.",
            "Changed raw masks, certificates, action features, inventory reasons or preview count reject before result publication.",
            "Both hooks are restored before every independent-audit callback; rejection retains its audit receipt and withholds completion.",
            "Source and bundle identities are checked again after audits and before publication.",
            "A failure after a real native commit preserves the paid history and reports committed interruption.",
            "STOP cannot substitute for a declared cutting action.",
            "Timeout requests termination then hard kill, and cannot publish success even with a nominally complete worker payload.",
            "Worker/result statuses and both runtime identities must match; nonzero exit, bad hash, malformed or non-object JSON fail closed.",
            "Existing public-launch output directories reject reuse without modifying previous files.",
        ],
        "scope": {
            "patient_bundles_opened": 0, "public_benchmarks_executed": 0, "gradient_updates": 0,
            "final_or_stress_worlds_opened": 0, "production_or_prototype_edits": False,
            "physical_fixture": "7x7x7 synthetic source grid; one declared paid native cut per phase",
            "audit_fixture": "Audit callbacks are stubs in orchestration tests; these validate restoration and publication, not the independent checker's geometry.",
            "launcher_fixture": "Mocked subprocess lifecycle; no child process or public command is launched.",
        },
        "limitations": [
            "No patient timing, speedup, learning throughput or clinical-performance result exists in this review.",
            "Six GiB process RSS is cooperative and cumulative; 32 MiB limits cached array payload, with separate entry cap and Python bookkeeping.",
            "Phase reset/transition/verification/export times, shared construction costs and total worker/launcher wall must remain separately reported.",
            "The 590-second parent wait plus 10-second termination grace bounds the child wait; source preparation and OS scheduling are separate.",
            "The normal fresh-output launcher is reviewed; this is not a hostile filesystem or arbitrary Python monkeypatch security boundary.",
            "Public execution still requires the parent's separate release on committed frozen source.",
        ],
    }
    (OUTPUT / "review.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
