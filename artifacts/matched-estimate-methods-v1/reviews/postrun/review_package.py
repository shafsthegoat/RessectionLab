"""Independent stdlib-only archive, identity and saved-logit verification."""
import hashlib
import json
import math
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "build/matched-estimate-methods-v1"
RUN = BASE / "attempt-01"
PACKAGE = BASE / "portable-result-v1"
OUT = Path(__file__).resolve().parent
def digest(data): return hashlib.sha256(data).hexdigest()
def sha(path): return digest(path.read_bytes())
def read(path): return json.loads(path.read_text())
def close(a, b): assert math.isclose(a, b, rel_tol=1e-13, abs_tol=1e-13), (a,b)
def probability(values):
    # Log-sum-exp via fsum independently checks the writer's plain sum.
    high = max(values)
    return math.exp(values[0] - high - math.log(math.fsum(math.exp(x-high) for x in values)))

files = {str(p.relative_to(RUN)): p for p in RUN.rglob("*") if p.is_file()}
archive = PACKAGE / "attempt-01.tar.gz"
archived = read(PACKAGE / "archive-inventory.json")
assert len(files) == archived["actual_original_files"] == 111
assert sum(p.stat().st_size for p in files.values()) == archived["actual_original_bytes"] == 2714476
assert sha(archive) == archived["archive_sha256"]
with tarfile.open(archive, "r:gz") as handle:
    members = handle.getmembers()
    assert len(members) == len(files)
    assert {m.name for m in members} == {"attempt-01/" + name for name in files}
    for member in members:
        assert member.isfile() and member.mode == 0o644
        content = handle.extractfile(member).read()
        path = files[member.name.removeprefix("attempt-01/")]
        assert content == path.read_bytes()
        assert archived["members"][member.name] == {"sha256": digest(content), "bytes": len(content)}

decomp_path = PACKAGE / "logit-decomposition.json"
assert sha(decomp_path) == "9f342b65637cae46d52d37012e6f723ea3a26b7d28861025f19081dcf5f9422b"
decomp = read(decomp_path)
# Exact frozen core.array_digest recipe for a contiguous false bool[9,9,7].
cavity = "sha256:" + digest(b'{"dtype": "|b1", "shape": [9, 9, 7]}' + b'\0' * 567)
assert cavity == decomp["initial_empty_cavity_hash"]
support = [(4,4,z) for z in range(1,6)] + [(5,5,1)]
tools = ["short-wide-opener", "long-narrow-cutter"]
matches = {}
for condition in ("original_tools", "actual_120mm_tools"):
    suite = read(RUN / condition / "planning-suite.json")
    first = suite["methods"]["SEARCH"]["strategy"]["physical_history"][0]
    source = first["source_hash"]
    table = {}
    for voxel in support:
        for tool in tools:
            fields = {"source": source, "cavity": cavity, "voxel": list(voxel), "tool_id": tool}
            encoded = json.dumps(fields, separators=(",",":"), sort_keys=True).encode()
            action = "NATIVE-SPATIAL:" + digest(encoded)[:24]
            assert action not in table
            table[action] = (tool, tuple(voxel))
    assert len(table) == 12
    assert table[first["action_id"]] == (first["tool_id"], tuple(first["tip_mm"]))
    matches[condition] = {}
    for method in ("IL", "RL"):
        decision = suite["methods"][method]["details"]["decisions"][0]
        assert decision["action_ids"][0] == "STOP"
        assert all(isinstance(x, float) and math.isfinite(x) for x in decision["legal_logits"])
        scores = {table[action]: (action, score) for action, score in
            zip(decision["action_ids"][1:], decision["legal_logits"][1:])}
        matches[condition][method] = (decision["legal_logits"][0], scores)

numbers = {}
for method in ("IL", "RL"):
    original_stop, old = matches["original_tools"][method]
    new_stop, new = matches["actual_120mm_tools"][method]
    common = set(old) & set(new); added = set(new) - set(old)
    assert len(common) == 4 and len(added) == 3 and not set(old) - set(new)
    assert added == {("short-wide-opener", (4,4,z)) for z in (3,4,5)}
    reported = decomp["methods"][method]
    assert reported["original_stop_logit"] == original_stop == new_stop == reported["actual_120mm_stop_logit"]
    assert reported["stop_logit_change"] == 0. and reported["removed_legal_actions"] == []
    assert len(reported["common_actions"]) == 4 and len(reported["newly_legal_actions"]) == 3
    seen = set()
    for row in reported["common_actions"]:
        key = (row["tool_id"], tuple(row["voxel"]))
        assert key in common and key not in seen; seen.add(key)
        assert row["original"] == {"action_id": old[key][0], "logit": old[key][1]}
        assert row["actual_120mm"] == {"action_id": new[key][0], "logit": new[key][1]}
        close(row["logit_change"], new[key][1] - old[key][1])
    seen = set()
    for row in reported["newly_legal_actions"]:
        key = (row["tool_id"], tuple(row["voxel"]))
        assert key in added and key not in seen; seen.add(key)
        assert row["action_id"] == new[key][0] and row["logit"] == new[key][1]
    calculated = {
        "original_stop_softmax": probability([original_stop] + [old[key][1] for key in sorted(old)]),
        "actual_120mm_stop_softmax": probability([new_stop] + [new[key][1] for key in sorted(new)]),
        "actual_120mm_common_actions_only_stop_softmax": probability([new_stop] + [new[key][1] for key in sorted(common)])}
    for key, value in calculated.items(): close(reported[key], value)
    numbers[method] = calculated

review = {"status": "PASS_archive_and_saved_logit_decomposition", "actual_attempt_files": 111,
    "actual_attempt_bytes": 2714476, "original_indexed_files": 108,
    "archive_sha256": sha(archive), "archive_bytes": archive.stat().st_size,
    "archive_inventory_sha256": sha(PACKAGE / "archive-inventory.json"),
    "logit_decomposition_sha256": sha(decomp_path),
    "source_recipe_reviewed": ["frozen core.array_digest bool/shape/C-order bytes", "frozen native_spatial_task fixed-lattice action-ID digest", "frozen spatial_policy.forward context/STOP/movement/critic separation"],
    "reconstructed_candidate_action_ids_per_condition": 12,
    "common_legal_movements": 4, "newly_legal_movements": 3, "removed_legal_movements": 0,
    "newly_legal_move_endpoints": [[4,4,3],[4,4,4],[4,4,5]], "newly_legal_tool": "short-wide-opener",
    "both_stop_logits_unchanged": True, "independent_softmax_values": numbers,
    "new_checkpoint_loads": 0, "new_model_forwards": 0, "new_native_previews": 0,
    "interpretation": "Common means matching tool ID and endpoint across altered working lengths, not identical full physical tool geometry. Restricted-four-action softmax is a saved-score subset calculation, not a new policy run or deployable action inventory. Movement scores and added feasible candidates explain the ranking arithmetic; which learned feature/training change would repair transfer is untested.",
    "package_scope": "All111 archived original bytes and saved-logit decomposition checked. Final package inventory must be rebound after copying corrected reviews."}
(OUT / "package-review.json").write_text(json.dumps(review,indent=2,sort_keys=True,allow_nan=False)+'\n')
print(json.dumps({"status":review["status"],"receipt_sha256":sha(OUT / "package-review.json"),"probabilities":numbers},sort_keys=True))
