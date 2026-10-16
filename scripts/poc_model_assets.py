#!/usr/bin/env python3
"""Acquire the two small, public POC model releases without login or telemetry."""
from pathlib import Path
import hashlib
import json
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
HF_REVISION = "dd70bbdd2b9de06b1316b4d1105bda9766c24fd7"


def download(url, path, maximum_bytes, expected_sha256=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        temporary = path.with_name(path.name + ".partial")
        count = 0
        with urllib.request.urlopen(url, timeout=60) as response, temporary.open("wb") as stream:
            while block := response.read(1024 * 1024):
                count += len(block)
                if count > maximum_bytes:
                    raise ValueError(f"Download exceeds declared budget: {url}")
                stream.write(block)
        temporary.replace(path)
    digest = hashlib.file_digest(path.open("rb"), "sha256").hexdigest()
    if expected_sha256 and digest != expected_sha256:
        raise ValueError(f"Publisher SHA256 mismatch: {path}")
    return {"path": str(path.relative_to(ROOT)), "url": url, "sha256": digest,
            "bytes": path.stat().st_size}


def main():
    vessel = ROOT / "data/models/poc-vesselfm-base"
    base = f"https://huggingface.co/bwittmann/vesselFM/resolve/{HF_REVISION}"
    files = [download(f"{base}/vesselFM_base.pt", vessel / "vesselFM_base.pt", 126_000_000,
                      "6da4fbd06dd9c3b597e5d50cc628deb3026919ff126d01afbd13ac8fe22ee10c")]
    for name in ("LICENSE", "README.md", "meta.yaml"):
        files.append(download(f"{base}/{name}", vessel / name, 100_000))
    (vessel / "acquisition.json").write_text(json.dumps({"revision": HF_REVISION, "files": files}, indent=2)+"\n")
    total = ROOT / "data/models/poc-totalsegmentator"
    name = "Dataset297_TotalSegmentator_total_3mm_1559subj.zip"
    url = f"https://github.com/wasserth/TotalSegmentator/releases/download/v2.0.0-weights/{name}"
    receipt = download(url, total / name, 136_000_000,
                       "0baa2c8de2975600eb31801dd5c1825cd2b356f794498659cf3348714c073394")
    with zipfile.ZipFile(total / name) as archive:
        for member in archive.infolist():
            target = (total / "weights" / member.filename).resolve()
            if not target.is_relative_to((total / "weights").resolve()):
                raise ValueError("Unsafe archive member")
        archive.extractall(total / "weights")
    (total / "acquisition.json").write_text(json.dumps({"release": "v2.0.0-weights", "files": [receipt]}, indent=2)+"\n")
    synthstrip = ROOT / "data/models/synthstrip-v1"
    synth_assets = {
        "mri_synthstrip": ("https://raw.githubusercontent.com/freesurfer/freesurfer/cf4bccf24875a47245b4df9fc9372d0f1c3d784f/mri_synthstrip/mri_synthstrip", "291c253ab2f7c0cbcb84afa769ae10909549537e727db5b92d268754b7cc7871"),
        "LICENSE_FreeSurfer.txt": ("https://raw.githubusercontent.com/freesurfer/freesurfer/cf4bccf24875a47245b4df9fc9372d0f1c3d784f/LICENSE.txt", "632d404ea17b9101d9ac87cf8f09e651d18eb20d2cc0ba8dbee77e85b8516307"),
        "synthstrip.1.pt": ("https://surfer.nmr.mgh.harvard.edu/docs/synthstrip/requirements/synthstrip.1.pt", "37417f802196186441aae3e7f385d94f8a98c64a88acaeaa2723af995c653e33"),
    }
    records = [download(url, synthstrip/name, 40*1024**2, digest) for name, (url,digest) in synth_assets.items()]
    (synthstrip / "poc-acquisition.json").write_text(json.dumps({"files": records}, indent=2)+"\n")
    print(json.dumps({"vessel": str(vessel), "total": str(total), "synthstrip": str(synthstrip)}, indent=2))


if __name__ == "__main__":
    main()
