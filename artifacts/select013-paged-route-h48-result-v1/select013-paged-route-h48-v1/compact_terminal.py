"""Stdlib-only compact projection of the authenticated, completed H48 extraction."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent


def pin(path):
    data = path.read_bytes()
    return {"path": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def compact(value):
    if isinstance(value, list):
        return [compact(v) for v in value]
    if isinstance(value, dict):
        result = {k: compact(v) for k, v in value.items() if k != "actions"}
        if "actions" in value:
            actions = value["actions"]
            result["action_count"] = len(actions)
            result["last_action"] = actions[-1] if actions else None
            result["action_sequence_sha256"] = hashlib.sha256(json.dumps(actions, separators=(",", ":")).encode()).hexdigest()
        return result
    return value


def main():
    projection_path = BASE / "postrun/terminal-projection.json"
    projection = json.loads(projection_path.read_text())
    result_path = BASE / "attempt-01/result.json"
    result = json.loads(result_path.read_text())
    assert pin(result_path)["sha256"] == projection["result_sha256"]
    for record in projection["lean_archive"]["files"] + projection["lean_archive"]["inventory_manifest"]:
        assert pin(ROOT / record["path"]) == record, record["path"]
    scalars = {k: v for k, v in projection.items() if k not in ("arms", "lean_archive")}
    scalars["full_projection"] = pin(projection_path)
    scalars["arms"] = {}
    for name, arm in projection["arms"].items():
        out = {k: v for k, v in arm.items() if k not in ("catalogs", "trace")}
        out["layers"] = compact(result["arms"][name]["selector_accounting"]["layers"])
        scalars["arms"][name] = out
    scalar_path = BASE / "postrun/compact-scalars.json"
    scalar_path.write_text(json.dumps(scalars, indent=2, sort_keys=True) + "\n")
    manifest_path = BASE / "postrun/inventory-manifest.json"
    manifest_path.write_text(json.dumps({"scope": "Exact saved inventory/page pins; raw files stay local, not duplicated in lean archive", "files": projection["lean_archive"]["inventory_manifest"]}, indent=2, sort_keys=True) + "\n")
    archive = {k: v for k, v in projection["lean_archive"].items() if k != "inventory_manifest"}
    archive["files"] = list(archive["files"])
    for path in (Path(__file__).resolve(), scalar_path, manifest_path, BASE / "postrun/REPORT.txt"):
        archive["files"].append(pin(path))
    archive["file_count"] = len(archive["files"])
    archive["total_bytes"] = sum(row["bytes"] for row in archive["files"])
    archive["independent_review"] = "Add the independent reviewer's compact H48 report/witness/index after its final handoff; no duplicate audit executed here."
    archive_path = BASE / "postrun/lean-archive-list.json"
    archive_path.write_text(json.dumps(archive, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"archive": pin(archive_path), "files": archive["file_count"], "bytes": archive["total_bytes"], "scalars": pin(scalar_path), "manifest": pin(manifest_path)}))


if __name__ == "__main__":
    main()
