"""Package existing generated outputs; stdlib only, no model/native imports."""
from pathlib import Path
import gzip
import hashlib
import io
import json
import math
import shutil
import tarfile

BASE = Path(__file__).resolve().parent
if BASE.name == "portable-result-v1":
    BASE = BASE.parent
ROOT = BASE.parents[1]
RUN = BASE / "attempt-01"
OUT = BASE / "portable-result-v1"
RELEASE = ROOT / "build/matched-estimate-methods-independent-v1"
AUDIT = ROOT / "build/matched-estimate-methods-postrun-independent-v1"

def sha(payload):
    return hashlib.sha256(payload).hexdigest()

def read(path):
    return json.loads(path.read_text())

def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")

def semantic(value):
    return "sha256:" + sha(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())

def softmax_stop(logits):
    peak = max(logits)
    return math.exp(logits[0] - peak) / sum(math.exp(x - peak) for x in logits)

OUT.mkdir(parents=True, exist_ok=True)
files = {str(p.relative_to(RUN)): p for p in sorted(RUN.rglob("*")) if p.is_file()}
index = read(RUN / "output-sha256.json")
assert all(sha(files[name].read_bytes()) == digest for name, digest in index.items())
declaration = read(RUN / "declaration-input.json")
assert sha((RUN / "declaration-input.json").read_bytes()) == "24c9f9bc22ebf56b1567cd27f3c8777a55897ccfbb58606a5f4ae205d771dd60"
for name, digest in {**declaration["source_sha256"], **declaration["metadata_sha256"]}.items():
    assert sha(files["source-and-metadata-snapshot/" + name].read_bytes()) == digest

# Preserve every original byte, including the two nested metadata inventory files
# omitted by the original index's basename exclusion. Archive metadata is normalized.
members = {}
with (OUT / "attempt-01.tar.gz").open("wb") as raw:
    with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w", format=tarfile.USTAR_FORMAT) as archive:
            for relative, path in files.items():
                payload = path.read_bytes()
                name = "attempt-01/" + relative
                info = tarfile.TarInfo(name)
                info.size, info.mode = len(payload), 0o644
                archive.addfile(info, io.BytesIO(payload))
                members[name] = {"sha256": sha(payload), "bytes": len(payload)}
with tarfile.open(OUT / "attempt-01.tar.gz", "r:gz") as archive:
    assert {m.name: {"sha256": sha(archive.extractfile(m).read()), "bytes": m.size}
            for m in archive.getmembers()} == members
write(OUT / "archive-inventory.json", {
    "archive_sha256": sha((OUT / "attempt-01.tar.gz").read_bytes()),
    "members": members,
    "original_index_sha256": sha((RUN / "output-sha256.json").read_bytes()),
    "original_indexed_files": len(index),
    "actual_original_files": len(files),
    "actual_original_bytes": sum(p.stat().st_size for p in files.values()),
    "not_in_original_output_index": sorted(set(files) - set(index)),
    "note": "All archive members match exact original bytes. The original output index excludes all files named output-sha256.json; its two nested prior metadata inventories are separately bound by declaration.metadata_sha256.",
})

copy_sources = {
    "declaration.json": BASE / "declaration.json",
    "prospective-fixed-experiment.json": BASE / "prospective-fixed-experiment.json",
    "candidate.patch": BASE / "candidate.patch",
    "supervisor-cleanup.patch": BASE / "supervisor-cleanup.patch",
    "tests/test_matched_estimate_methods.py": BASE / "candidate/test_matched_estimate_methods.py",
    "tests/test_supervisor_cleanup.py": BASE / "candidate/test_supervisor_cleanup.py",
    "tests/test_independent_release.py": RELEASE / "test_independent_release.py",
    "reviews/release/REPORT.md": RELEASE / "REPORT.md",
    "reviews/release/review.json": RELEASE / "review.json",
    "reviews/release/tests.txt": RELEASE / "tests.txt",
    "reviews/release/metadata-preflight.json": RELEASE / "metadata-preflight.json",
    "reviews/postrun/REPORT.md": AUDIT / "REPORT.md",
    "reviews/postrun/audit.json": AUDIT / "audit.json",
    "reviews/postrun/audit_saved.py": AUDIT / "audit_saved.py",
    "reviews/postrun/file-accounting-correction.json": AUDIT / "file-accounting-correction.json",
    "reviews/postrun/package-review.json": AUDIT / "package-review.json",
    "reviews/postrun/review_package.py": AUDIT / "review_package.py",
    "reviews/postrun/original_tools-replay.json": AUDIT / "original_tools-replay.json",
    "reviews/postrun/actual_120mm_tools-replay.json": AUDIT / "actual_120mm_tools-replay.json",
    "package_result.py": Path(__file__),
}
for name, source in copy_sources.items():
    destination = OUT / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)

suites = {condition: read(RUN / condition / "planning-suite.json")
          for condition in ("original_tools", "actual_120mm_tools")}
audited = read(AUDIT / "audit.json")
assert audited["status"].startswith("PASS")

# Pure saved-logit decomposition: reconstruct only action IDs from the frozen
# generated recipe and hash algorithm. No native task, preview, array or actor.
shape = (9, 9, 7)
cavity_hash = "sha256:" + sha(json.dumps({"shape": shape, "dtype": "|b1"}, sort_keys=True).encode() + bytes(math.prod(shape)))
voxels = [(4, 4, z) for z in range(1, 6)] + [(5, 5, 1)]
tools = ("short-wide-opener", "long-narrow-cutter")
maps = {}
for condition, suite in suites.items():
    first = suite["methods"]["SEARCH"]["strategy"]["physical_history"][0]
    source_hash = first["source_hash"]
    mapping = {}
    for voxel in voxels:
        for tool in tools:
            identity = semantic({"source": source_hash, "cavity": cavity_hash,
                                 "voxel": list(voxel), "tool_id": tool})
            action_id = "NATIVE-SPATIAL:" + identity.split(":")[1][:24]
            mapping[action_id] = (tool, voxel)
    for method in ("IL", "RL"):
        decision = suite["methods"][method]["details"]["decisions"][0]
        assert decision["action_ids"][0] == "STOP"
        assert set(decision["action_ids"][1:]) <= set(mapping)
    assert mapping[first["action_id"]] == (first["tool_id"], tuple(map(int, first["tip_mm"])))
    maps[condition] = mapping

decomposition = {"scope": "saved initial logits and pure action-identity arithmetic only",
    "new_checkpoint_loads": 0, "new_model_forwards": 0, "new_native_previews": 0,
    "optimizer_updates": 0, "patient_files_opened": 0,
    "initial_empty_cavity_hash": cavity_hash,
    "matching_basis": "Frozen native_spatial_task action-ID recipe: source hash, empty bool cavity digest, voxel and tool ID; every saved legal ID maps exactly. Physical matching uses tool ID plus voxel, not list position.",
    "methods": {}}
for method in ("IL", "RL"):
    condition_rows = {}
    for condition, suite in suites.items():
        decision = suite["methods"][method]["details"]["decisions"][0]
        condition_rows[condition] = {
            "decision": decision,
            "movement": {maps[condition][action]: {"action_id": action, "logit": logit}
                         for action, logit in zip(decision["action_ids"][1:], decision["legal_logits"][1:])}}
    original, changed = condition_rows["original_tools"], condition_rows["actual_120mm_tools"]
    common = sorted(set(original["movement"]) & set(changed["movement"]))
    added = sorted(set(changed["movement"]) - set(original["movement"]))
    assert len(common) == 4 and len(added) == 3
    common_rows = [{"tool_id": key[0], "voxel": list(key[1]),
                    "original": original["movement"][key], "actual_120mm": changed["movement"][key],
                    "logit_change": changed["movement"][key]["logit"] - original["movement"][key]["logit"]}
                   for key in common]
    changed_common_logits = [changed["decision"]["legal_logits"][0]] + [changed["movement"][key]["logit"] for key in common]
    decomposition["methods"][method] = {
        "original_stop_logit": original["decision"]["legal_logits"][0],
        "actual_120mm_stop_logit": changed["decision"]["legal_logits"][0],
        "stop_logit_change": changed["decision"]["legal_logits"][0] - original["decision"]["legal_logits"][0],
        "original_stop_softmax": softmax_stop(original["decision"]["legal_logits"]),
        "actual_120mm_stop_softmax": softmax_stop(changed["decision"]["legal_logits"]),
        "actual_120mm_common_actions_only_stop_softmax": softmax_stop(changed_common_logits),
        "common_actions": common_rows,
        "newly_legal_actions": [{"tool_id": key[0], "voxel": list(key[1]), **changed["movement"][key]} for key in added],
        "removed_legal_actions": [list(key) for key in sorted(set(original["movement"]) - set(changed["movement"]))],
        "actual_120mm_stop_minus_best_movement": changed["decision"]["stop_minus_best_movement"],
    }
decomposition["interpretation"] = [
    "Both STOP logits are unchanged. The action representation/ranking changes, rather than a larger STOP scalar, explain the observed argmax change algebraically.",
    "RL common movement scores fall; three newly certified legal actions reduce the STOP preference substantially. Full actual-inventory STOP softmax must not be replaced by the prior descriptor-only figure.",
    "Saved score decomposition does not identify which training feature or architecture change will fix transfer; no thresholds, normalization or policy outputs were changed.",
    "The frozen source uses image/state/spacing context for STOP and context plus per-action geometry/ray features for movement. Candidate critic context does not feed the actor. Source inspection is an architectural explanation, not another learned experiment.",
]
write(OUT / "logit-decomposition.json", decomposition)

summary = {
    "status": "independent_postrun_PASS",
    "scope": "full-information six-cell generated development tool-shift diagnostic",
    "primary_patient_generalization_result": None,
    "clinical_validity": None,
    "declaration_sha256": audited["declaration_sha256"],
    "rows": audited["rows"],
    "saved_supervisor": audited["saved_attempt_supervisor"],
    "saved_worker_elapsed_seconds": audited["saved_worker_elapsed_seconds"],
    "online_totals": audited["totals"],
    "common_preparation": {c: suite["shared_preparation"] for c, suite in suites.items()},
    "original_evaluation_costs": {c: {k: v for k, v in read(RUN / (c + "-evaluation.json")).items()
                                      if k in ("evaluation_total_seconds", "pre_reference_validation_seconds", "pre_reference_nominal_replay_transition_calls")}
                                  for c in suites},
    "original_preview_accounting_limit": "Online756 and common24 previews are recorded phases; original fixture and evaluation preview entries were not separately instrumented. Do not interpret these as whole-run preview totals.",
    "independent_review_additional_work": audited["additional_generated_replay"],
    "training_accounting": suites["original_tools"]["training_accounting"],
    "fixed_method_budget": suites["original_tools"]["budget"],
    "fixed_worker_budget": declaration["settings"],
    "optimizer_updates": 0, "patient_files_opened": 0,
    "recorded_strategy_count": audited["reviewed_strategy_count"],
    "recorded_transition_count": audited["recorded_transition_count"],
    "recorded_microstep_count": audited["recorded_microstep_count"],
    "limits": ["one familiar generated source, not a patient or unseen-anatomy comparison", "no physical tissue or clinical validation", "outside-source-FOV tool feasibility unassessed", "sampled RSS can miss transient peaks", "fixed-order single-run timings do not establish deployment efficiency", "old checkpoints have distinct prior training and teacher costs; equal inference inputs do not erase those costs"],
}
write(OUT / "summary.json", summary)
write(OUT / "next-diagnostic.json", {
    "status": "proposal_only_no_additional_checkpoint_call_authorized_or_run",
    "smallest_step_completed": "Saved-only physical action matching and logit decomposition in logit-decomposition.json; zero native/model calls.",
    "optional_next_question": "Does the actual-long root movement-score change reside in the working-length geometry feature for the seven certified moves?",
    "bounded_intervention": "One frozen RL forward on a diagnostic copy of the already generated actual-long initial observation: replace only the seven movement working-length features with each tool's original 2.2/12 mm value. Keep actual-long legal inventory, order, masks, other geometry/rays, state, image, weights and normalization fixed. Compare all seven logits with saved actual-long logits; STOP must remain identical. Do not execute these descriptor-modified actions or score a resulting strategy.",
    "budget_if_root_approves": {"new_RL_forwards": 1, "new_IL_forwards": 0, "optimizer_updates": 0, "search_calls": 0, "patient_files_opened": 0, "supervised_wall_seconds": 20, "sampled_worker_rss_bytes": 1073741824, "cpu_threads": 1, "automatic_retry": False},
    "preparation_limit": "Reusing a certified generated actual-long observation may require bounded reconstruction previews; those must be declared/counted separately before release. This proposal is not yet a runnable approved declaration.",
    "interpretation": "Feature intervention diagnoses representation sensitivity only. It cannot validate fake short descriptors for actual long tools, serve as a policy repair, establish calibrated abstention, or replace limited-input unseen-patient evaluation.",
    "priority": "Primary progress remains real scan-derived limited-input planning with protected splits and independent full-history scoring. This tiny diagnostic need not delay patient-model preprocessing or mechanics validation.",
})
(OUT / "SUMMARY.txt").write_text(
    "Independent audit PASS: actual tool-geometry shift on one generated six-cell fixture.\n\n"
    "                     SEARCH     IL       RL       HYBRID\n"
    "Original tools        1.100    1.100    1.098       1.100\n"
    "Both lengths 120 mm    1.163    1.161    0.000 STOP  1.163\n\n"
    "Values are independently checked geometric returns. All moving methods remove 2 mm^3 target and 4 mm^3 normal tissue. Longer-tool RL chooses immediate STOP despite seven legal movement alternatives; SEARCH/HYBRID find a one-movement solution. Search finished without pruning or budget truncation.\n\n"
    "Worker 2.9182 s; supervisor 4.1885 s; sampled peak 313311232 B; zero updates/patient reads. Eight full strategies preserve 15 transitions/141 microsteps. Postrun committed-only replay is separate review work: 0.8714 s/372 previews, no checkpoint decode, policy forward or search.\n\n"
    "Saved RL STOP score stays -1.8111782074; best long-tool movement is -2.2781767845, margin +0.4669985771. Its saved softmax STOP preference is 60.9844%, compared with 6.7249% originally. The old descriptor-only 99.39% figure is a different inventory. Matched action details are in logit-decomposition.json. These are model scores, not clinical probabilities.\n\n"
    "attempt-01.tar.gz preserves every original output/source/metadata byte. archive-inventory.json adds an exact hash for all members, including prior metadata inventories omitted by the original basename-filtered output index. Both original condition suites, eight full strategies, endpoint evaluations, budgets and lineage remain inside. Frozen checkpoint files are referenced by hash, not copied. Independent reports, tests and candidate patches are separate.\n\n"
    "This supports a negative frozen-RL tool-transfer finding on a fully observed tiny generated task. It does not establish the primary rich-training/limited-input unseen-patient hypothesis or physical/clinical validity. Outside-source-FOV feasibility remains unassessed. The next optional diagnostic is proposed separately; no new forward or retraining was performed to package or decompose the result.\n")

inventory = {str(p.relative_to(OUT)): {"bytes": p.stat().st_size, "sha256": sha(p.read_bytes())}
             for p in sorted(OUT.rglob("*")) if p.is_file() and p.name != "package-sha256.json"}
write(OUT / "package-sha256.json", inventory)
print(json.dumps({"package": str(OUT), "files": len(inventory) + 1,
    "bytes": sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file()),
    "archive_files": len(members), "archive_bytes": (OUT / "attempt-01.tar.gz").stat().st_size,
    "package_inventory_sha256": sha((OUT / "package-sha256.json").read_bytes()),
    "logit_decomposition_sha256": sha((OUT / "logit-decomposition.json").read_bytes())}, sort_keys=True))
