"""Preserve existing generated diagnostic evidence, with stdlib arithmetic only."""
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
RELEASE = ROOT / "build/working-length-reverse-independent-v1"
AUDIT = ROOT / "build/working-length-reverse-postrun-independent-v1"

def sha(data):
    return hashlib.sha256(data).hexdigest()

def read(path):
    return json.loads(path.read_text())

def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")

def softmax_stop(values):
    peak = max(values)
    return math.exp(values[0] - peak) / sum(math.exp(v - peak) for v in values)

audit = read(AUDIT / "audit.json")
assert audit["status"].startswith("PASS")
OUT.mkdir(parents=True, exist_ok=True)
files = {str(p.relative_to(RUN)): p for p in sorted(RUN.rglob("*")) if p.is_file()}
index = read(RUN / "output-sha256.json")
assert set(files) - {"output-sha256.json"} == set(index)
assert all(sha(files[name].read_bytes()) == digest for name, digest in index.items())
declaration = read(RUN / "declaration-input.json")
assert sha((RUN / "declaration-input.json").read_bytes()) == "8463bbf4e244498cc1df6ca2a92df2ed981f64c9622f158aaf7f62e50bd01204"
for name, digest in {**declaration["source_sha256"], **declaration["input_sha256"]}.items():
    assert sha(files["source-and-input-snapshot/" + name].read_bytes()) == digest

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
    "members": members, "original_index_sha256": sha((RUN / "output-sha256.json").read_bytes()),
    "original_indexed_files": len(index), "actual_original_files": len(files),
    "actual_original_bytes": sum(p.stat().st_size for p in files.values()),
    "note": "Every original file is retained byte-for-byte. Frozen checkpoint bytes are not copied; exact checkpoint lineage is in the declaration and independent audit.",
})
copies = {name: RUN / name for name in ("result.json", "supervisor.json", "declaration-input.json",
    "input-identity.json", "generated-permitted-inputs.npz")}
copies.update({"package_result.py": Path(__file__), "initial-control-failure.txt": BASE / "initial-control-failure.txt"})
for category, folder in (("release", RELEASE), ("postrun", AUDIT)):
    for path in sorted(folder.iterdir()):
        if path.is_file() and path.suffix in (".json", ".md", ".py", ".txt"):
            copies["reviews/" + category + "/" + path.name] = path
for relative, source in copies.items():
    destination = OUT / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)

result, supervisor = read(RUN / "result.json"), read(RUN / "supervisor.json")
assert result["status"] == supervisor["status"] == "complete"
assert result["forward_calls"] == {"attempted": 1, "completed": 1}
assert result["diagnostic_logits"][0] == result["saved_actual_logits"][0]
changed, saved = result["diagnostic_logits"], result["saved_actual_logits"]
summary = {"status": "independent_saved_only_audit_PASS", "scope": result["scope"],
    "declaration_sha256": sha((RUN / "declaration-input.json").read_bytes()),
    "checkpoint_parameter_hash": result["checkpoint_parameter_hash"],
    "stop_logit_before_and_after": changed[0],
    "saved_actual_stop_minus_best_movement": saved[0]-max(saved[1:]),
    "diagnostic_stop_minus_best_movement": changed[0]-max(changed[1:]),
    "saved_actual_stop_softmax": softmax_stop(saved), "diagnostic_stop_softmax": softmax_stop(changed),
    "highest_ranked_id_not_executed": result["highest_ranked_id_not_executed"],
    "movement_logit_changes": result["logit_changes"][1:],
    "recorded_diagnostic_body_seconds": result["elapsed_seconds"],
    "clock_note": "Recorded diagnostic body starts after imports; supervisor wall time includes worker startup/imports. Parent snapshots and postflight hashing are outside the worker timer.",
    "reconstruction_seconds": result["reconstruction_seconds"],
    "reconstruction_preview_entries": sum(row["started"] for row in result["reconstruction_native_previews"]["phases"].values()),
    "reconstruction_native_previews": result["reconstruction_native_previews"],
    "supervisor": supervisor, "forward_calls": result["forward_calls"],
    "executed_actions": 0, "search_calls": 0, "optimizer_updates": 0, "patient_files_opened": 0,
    "generated_input_bytes": (RUN / "generated-permitted-inputs.npz").stat().st_size,
    "original_observation_hash": result["original_observation_hash"],
    "intervention_observation_hash": result["intervention_observation_hash"],
    "claim": "On this fixed generated input, reverting only working-length features is sufficient to reverse the frozen policy's STOP ranking. The STOP scalar is unchanged; movement scores account for the flip.",
    "limits": ["modified descriptors are uncertified and no resulting action is executed", "not a repaired policy or demonstration of strategy efficacy", "softmax values are model scores, not clinical probabilities", "one fully observed familiar generated source, not limited-input unseen-patient evidence", "does not justify changed normalization, STOP threshold, checkpoint selection or retraining", "physical tissue and outside-source-FOV validity remain unestablished"],
}
write(OUT / "summary.json", summary)
(OUT / "SUMMARY.txt").write_text(
    "Independent saved-output audit PASS: one working-length input intervention.\n\n"
    "At the exact generated actual-120-mm root, changing only seven working-length descriptors back to 2.2/12 mm moved the frozen RL STOP-minus-best-movement score from +0.4669986 to -6.3032722. The STOP scalar stayed exactly -1.811178207397461. Its softmax preference fell from 60.984445% to 0.173134%; these are model scores, not clinical probabilities.\n\n"
    "The unchanged action set contains seven movements. Image, coverage, affine, state, masks, IDs/order, other geometry and policy parameters/buffers remained fixed. The intervention has its own observation fingerprint and is explicitly uncertified. No action or strategy was executed.\n\n"
    "Exactly one RL forward completed. Reconstruction used 24 previews under the fixed 64-preview cap. Recorded diagnostic body time was 0.1165 s; supervisor wall time was 1.8840 s, including worker startup/imports. Sampled peak worker RSS was 311083008 B, within the fixed 20 s/1 GiB envelope. There were zero updates, searches, patient reads or retries.\n\n"
    "This isolates working-length input sensitivity on one familiar generated root. It does not establish a repaired policy, calibrated abstention, physical validity, or the primary rich-training/limited-input unseen-patient hypothesis. No normalization, STOP threshold or checkpoint was changed.\n\n"
    "The 2309-byte generated-permitted-inputs.npz and input-identity.json preserve the original input arrays and changed geometry. attempt-01.tar.gz preserves every original source/input/output/receipt byte; archive-inventory.json verifies each member. Frozen checkpoint bytes are referenced by exact hashes and are not copied. Both independent reviews and their checks are included.\n")
inventory = {str(p.relative_to(OUT)): {"bytes": p.stat().st_size, "sha256": sha(p.read_bytes())}
             for p in sorted(OUT.rglob("*")) if p.is_file() and p.name != "package-sha256.json"}
write(OUT / "package-sha256.json", inventory)
print(json.dumps({"package": str(OUT), "files": len(inventory)+1,
    "bytes": sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file()),
    "archive_files": len(members), "archive_bytes": (OUT / "attempt-01.tar.gz").stat().st_size,
    "package_inventory_sha256": sha((OUT / "package-sha256.json").read_bytes())}, sort_keys=True))
