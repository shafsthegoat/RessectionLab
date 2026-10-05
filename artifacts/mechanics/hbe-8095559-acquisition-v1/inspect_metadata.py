"""Inspect geometry and specimen lookup only; never open a force/torque CSV."""
import collections
import gzip
import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "data/mechanics/zenodo-8095559"
OUT = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    metadata = json.loads((BASE / "sample-lookup-metadata.json").read_text())
    rows = next(item["values"] for item in metadata if item["sheet"] == "Sample Overview")
    assert rows[0][:4] == ["Brain", "Sample", "Governing Region", "Region"]
    lookup = {
        f"HBE_{int(row[0]):02d}_{int(row[1]):02d}": {
            "donor_id": f"HBE_{int(row[0]):02d}", "donor_numeric_id": int(row[0]),
            "specimen_numeric_id": int(row[1]), "governing_region": row[2], "region": row[3],
        } for row in rows[1:] if row[0] is not None
    }
    assert len(lookup) == len(rows) - 1
    filenames = sorted(
        [f"{mode}_{cycle}.csv" for mode in ("compression", "tension") for cycle in ("c1", "c3")]
        + [f"torsion_{level}_{cycle}_{sign}.csv" for level in ("l1", "l2")
           for cycle in ("c1", "c3") for sign in ("neg", "pos")]
    )
    c3_filenames = [name for name in filenames if "_c3" in name]
    specimens = []
    with zipfile.ZipFile(BASE / "HBE_Data.zip") as archive:
        names = archive.namelist()
        assert len(set(names)) == len(names)
        directory = [{"name": item.filename, "bytes": item.file_size,
                      "compressed_bytes": item.compress_size, "crc32": f"{item.CRC:08x}",
                      "directory": item.is_dir()} for item in archive.infolist()]
        for specimen, details in sorted(lookup.items(), key=lambda item: (
                item[1]["donor_numeric_id"], item[1]["specimen_numeric_id"])):
            prefix = details["donor_id"] + "/" + specimen + "/"
            geometry_name = prefix + "geometry.yaml"
            raw = archive.read(geometry_name)
            geometry = {}
            for line in raw.decode().splitlines():
                key, value = line.split(":", 1)
                geometry[key.strip()] = value.strip() if key.strip() == "type" else float(value)
            # Source capitalization varies; preserve the exact source spelling.
            assert geometry["type"].lower() == "cylinder"
            assert geometry["height"] > 0 and geometry["radius"] > 0
            observed = {name[len(prefix):] for name in names if name.startswith(prefix)}
            specimens.append({
                "specimen_id": specimen, **details, "geometry_member": geometry_name,
                "geometry_sha256": hashlib.sha256(raw).hexdigest(), "geometry_m": geometry,
                "curve_members": [{"name": prefix + name,
                                   "bytes": archive.getinfo(prefix + name).file_size,
                                   "crc32": f"{archive.getinfo(prefix + name).CRC:08x}"}
                                  for name in filenames if name in observed],
                "missing_expected_curve_members": sorted(set(filenames) - observed),
                "metadata_eligible_for_c3_poc": set(c3_filenames).issubset(observed),
            })
        expected = {item["geometry_member"] for item in specimens} | {
            curve["name"] for item in specimens for curve in item["curve_members"]}
        extras = [name for name in names if not name.endswith("/") and name not in expected]
        geometry_members = {name for name in names if name.endswith("/geometry.yaml")}
        assert geometry_members == {item["geometry_member"] for item in specimens}
    result = {
        "schema_version": 1, "source_manifest": "manifests/hbe_8095559_acquisition.json",
        "source_manifest_sha256": digest(ROOT / "manifests/hbe_8095559_acquisition.json"),
        "archive_sha256": digest(BASE / "HBE_Data.zip"),
        "lookup_sha256": digest(BASE / "sample_lookup.xlsx"),
        "inspection_scope": "ZIP names/lengths/CRC, geometry and lookup metadata only; no curve members opened",
        "lookup_ranges": ["Sample Overview!A1:D183", "List of Abbreviations!A1:C33"],
        "curve_content_schema_status": "Creator specifies two columns and units; CSV headers, delimiters and row counts unopened pending split.",
        "geometry_height_provenance": "Creator says height determined from test data, not independently measured geometry or an uncertainty estimate.",
        "geometry_schema": {"height": "metres, positive number", "radius": "metres, positive number",
                            "type": "Source has cylinder and Cylinder; original spelling retained."},
        "geometry_type_spellings": dict(collections.Counter(item["geometry_m"]["type"] for item in specimens)),
        "donor_specimen_counts": dict(sorted(collections.Counter(item["donor_id"] for item in specimens).items())),
        "specimen_count": len(specimens), "curve_member_count": sum(len(item["curve_members"]) for item in specimens),
        "all_12_curve_members_present": all(not item["missing_expected_curve_members"] for item in specimens),
        "extra_members": extras,
        "numeric_sorted_first_metadata_eligible": next(item["specimen_id"] for item in specimens if item["metadata_eligible_for_c3_poc"]),
        "required_c3_filenames": c3_filenames, "specimens": specimens,
        "force_values_opened": False, "role_assignments": "Pending validation declaration; this is an inventory, not a split.",
        "metadata_inspection_failure_retained": "Initial exploratory assertion required literal lowercase cylinder and stopped at donor04. Source also uses Cylinder. No curve data opened; parser now preserves spelling and verifies case-insensitive shape only.",
    }
    (OUT / "metadata-inventory.json").write_text(json.dumps(result, indent=2) + "\n")
    raw_directory = json.dumps({"archive_sha256": result["archive_sha256"], "entries": directory,
                                "curve_members_opened": False}, indent=2).encode() + b"\n"
    (OUT / "zip-central-directory.json.gz").write_bytes(gzip.compress(raw_directory, mtime=0))
    print(json.dumps({key: value for key, value in result.items() if key != "specimens"}, indent=2))
    print("FIRST_METADATA_ELIGIBLE", json.dumps(specimens[0], indent=2))


if __name__ == "__main__":
    main()
