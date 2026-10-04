"""Saved-primitive readout only; no execution or measured-curve access."""
from pathlib import Path
import gzip
import hashlib
import json
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.mechanics_hbe_readout import read_run

ART = Path(__file__).resolve().parent
RUN = ROOT / "outputs/mechanics/hbe-01-03-poc-v1/experiment/runs/compression-N4-S60-reference"
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
execution = json.loads((RUN / "execution.json").read_text())
assert execution["run_id"] == "compression:N4:S60:reference"
assert execution["protocol_sha256"] == "ab4385f5ad315d2444ca3aedc87b455db0d36539cea4e2119114d803fba49035"
sources = ["scripts/mechanics_hbe_outputs.py", "scripts/mechanics_hbe_readout.py",
           "scripts/mechanics_hbe_access.py", "scripts/mechanics_hbe_physics.py",
           "scripts/mechanics_hbe_evaluation.py"]
bound = [ROOT / p for p in sources] + [RUN / "execution.json"]
bound += [ROOT / v["path"] for v in execution["primitive_bindings"].values()]
before = {str(p.relative_to(ROOT)): sha(p) for p in bound}
assert before[sources[0]] == "a372cb89ea9515186d433c74ec650dc3b5cae289619e8b35d481280f6662f473"
assert before[sources[1]] == "21700558e5cb614a11a7cc62dbd5ec8118fabb11b4fbfc88f89cb8aa02012963"
started = time.perf_counter()
result, cache = read_run(ROOT, execution["primitive_bindings"],
    protocol_sha256=execution["protocol_sha256"], expected_branch="compression",
    expected_mesh_N=4, expected_steps=60, expected_mu_Pa=1000.)
elapsed = time.perf_counter() - started
after = {str(p.relative_to(ROOT)): sha(p) for p in bound}
assert before == after
assert cache is None
payload = json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
with (ART / "saved-coarse-readout.json.gz").open("xb") as f:
    f.write(gzip.compress(payload, mtime=0))
summary = {
    "schema": 1, "operation": "read_saved_primitives_only", "solver_runs": 0,
    "measured_curve_reads": 0, "source_and_input_sha256": before,
    "all_bound_files_unchanged": True, "readout_seconds": elapsed,
    "process_peak_rss_bytes_macos": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    "run_id": execution["run_id"], "frame_count": result["frame_count"],
    "passed": result["passed"], "criteria": result["criteria"],
    "energy_work": result["energy_work"],
    "normal_termination": result["solver"]["normal_termination"],
    "nonzero_residual_states": len(result["solver"]["states"]),
    "logged_J_minimum": min(r["logged_J_min"] for r in result["diagnostics"]),
    "complete_readout_uncompressed_sha256": hashlib.sha256(payload).hexdigest(),
    "complete_readout_gzip_sha256": sha(ART / "saved-coarse-readout.json.gz"),
    "interpretation": "One existing coarse reference solve; no refinement, scale, calibration, held-out or physical validation conclusion.",
}
with (ART / "saved-coarse-summary.json").open("x") as f:
    json.dump(summary, f, indent=2)
    f.write("\n")
print(json.dumps(summary, indent=2))
