"""Released Case4 byte acquisition only: no NIfTI or landmark parsing."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
MANIFEST_SHA = "2babb675a77d48bfd7f7df2748956b0de7e268a6a4ee731df508b3376401830d"
RELEASE_SHA = "ffa5fe23292f94068f20136d5923160054e49fd89129323212cd8bf6077538ed"
S3_PREFIX = "https://s3.nird.sigma2.no/archive-ro/5686d8fa-2003-4837-8e66-8e887fabe21e/"


def hashes(path):
    md5, sha256 = hashlib.md5(), hashlib.sha256()
    length = 0
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            length += len(block)
            md5.update(block)
            sha256.update(block)
    return {"bytes": length, "md5": md5.hexdigest(), "sha256": sha256.hexdigest()}


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def main():
    assert hashes(OUT / "root-release.json")["sha256"] == RELEASE_SHA
    assert hashes(OUT / "frozen-manifest.json")["sha256"] == MANIFEST_SHA
    manifest = json.loads((OUT / "frozen-manifest.json").read_text())
    release = json.loads((OUT / "root-release.json").read_text())
    originals = ROOT / release["originals_directory"]
    receipt = OUT / "acquisition-receipt.json"
    assert not receipt.exists()
    assert len(manifest["files"]) == 6
    assert sum(source["bytes"] for source in manifest["files"]) == 35068010
    assert len({source["source_path"] for source in manifest["files"]}) == 6
    for source in manifest["files"]:
        relative = Path(source["source_path"])
        assert not relative.is_absolute() and ".." not in relative.parts
        assert source["source_path"].startswith("RESECT/NIFTI/Case4/")
        assert source["anonymous_https_url"] == S3_PREFIX + source["source_path"]
        target = originals / relative
        assert not target.exists() and not target.with_name(target.name + ".partial").exists()
    record = {"schema_version": 1, "started_at_utc": datetime.now(timezone.utc).isoformat(),
              "status": "running", "root_release_sha256": RELEASE_SHA,
              "manifest_sha256": MANIFEST_SHA, "driver_sha256": hashes(Path(__file__))["sha256"],
              "git_commit": release["git_commit"], "files": [], "raw_payload_inspection": False,
              "gzip_decompression": False, "image_header_or_array_loading": False,
              "landmark_text_or_coordinate_loading": False, "existing_patient_splits_changed": False,
              "curl_version": subprocess.check_output(["curl", "--version"], text=True).splitlines()[0]}
    save(receipt, record)
    started = time.monotonic()
    try:
        for source in manifest["files"]:
            target = originals / source["source_path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_name(target.name + ".partial")
            headers = OUT / (target.name + ".http-headers.txt")
            if headers.exists():
                raise FileExistsError(headers)
            command = ["curl", "--silent", "--show-error", "--fail", "--proto", "=https",
                       "--max-time", "45", "--max-filesize", str(source["bytes"]),
                       "--dump-header", str(headers), "--output", str(partial),
                       "--write-out", "%{http_code}\t%{size_download}\t%{url_effective}\n",
                       source["anonymous_https_url"]]
            entry = {"source_path": source["source_path"], "argv": command,
                     "expected_bytes": source["bytes"], "expected_md5": source["provider_md5"],
                     "started_at_utc": datetime.now(timezone.utc).isoformat()}
            record["files"].append(entry)
            save(receipt, record)
            file_started = time.monotonic()
            result = subprocess.run(command, capture_output=True, text=True, timeout=50, check=False)
            entry.update(curl_exit_code=result.returncode, transport_summary=result.stdout.strip(),
                         stderr=result.stderr, elapsed_seconds=time.monotonic() - file_started)
            save(receipt, record)
            if result.returncode:
                raise RuntimeError(f"Transfer failed; partial retained, no retry: {source['source_path']}")
            status, count, final_url = result.stdout.strip().split("\t")
            if status != "200" or int(count) != source["bytes"] or final_url != source["anonymous_https_url"]:
                raise ValueError("Transfer response identity or length differs from declaration")
            actual = hashes(partial)
            entry.update(actual)
            if actual["bytes"] != source["bytes"] or actual["md5"] != source["provider_md5"]:
                raise ValueError("Transferred bytes fail provider size/MD5 check")
            # Exclusive same-volume hard link gives no-overwrite promotion; source stays byte-identical.
            os.link(partial, target)
            partial.unlink()
            entry.update(local_path=str(target.relative_to(ROOT)), verified=True,
                         http_headers_sha256=hashes(headers)["sha256"])
            save(receipt, record)
            print(json.dumps({key: entry[key] for key in ("source_path", "bytes", "md5", "sha256", "verified")}), flush=True)
        for entry in record["files"]:
            assert hashes(ROOT / entry["local_path"]) == {key: entry[key] for key in ("bytes", "md5", "sha256")}
        record.update(status="completed", downloaded_verified_bytes=sum(entry["bytes"] for entry in record["files"]),
                      all_files_independently_rehashed=True)
    except BaseException as error:
        record.update(status="failed", failure_type=type(error).__name__, failure=str(error))
        raise
    finally:
        record.update(elapsed_seconds=time.monotonic() - started,
                      finished_at_utc=datetime.now(timezone.utc).isoformat(),
                      frozen_manifest_unchanged=hashes(OUT / "frozen-manifest.json")["sha256"] == MANIFEST_SHA,
                      original_manifest_unchanged=hashes(ROOT / release["manifest_path"])["sha256"] == MANIFEST_SHA)
        save(receipt, record)


if __name__ == "__main__":
    main()
