#!/usr/bin/env python3
"""Explicitly install the pinned SIAM v2 single-fold assets for local preparation."""
import argparse
import hashlib
from pathlib import Path
import shutil
import sys
import tarfile
import tempfile
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from resectionlab.anatomy_siam import assets
from resectionlab.siam_worker import ARCHIVE_MD5, ASSETS, validate_assets

URL = "https://zenodo.org/records/15780983/files/DS715_NODA.tar.gz?download=1"


def install(archive, destination):
    with archive.open("rb") as stream:
        if hashlib.file_digest(stream, "md5").hexdigest() != ARCHIVE_MD5:
            raise ValueError("SIAM archive differs from Zenodo record 15780983")
    with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
        stage = Path(temporary) / destination.name
        stage.mkdir()
        wanted = {"DS715_NODA/" + name: name for name in ASSETS}
        with tarfile.open(archive, "r|gz") as bundle:
            for member in bundle:
                name = wanted.pop(member.name, None)
                if name is None:
                    continue
                if not member.isfile() or member.size > 1200_000_000:
                    raise ValueError("Invalid SIAM archive member")
                target = stage / name
                target.parent.mkdir(exist_ok=True)
                with bundle.extractfile(member) as original, target.open("xb") as out:
                    shutil.copyfileobj(original, out)
                if not wanted:
                    break
        validate_assets(stage)
        stage.rename(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="Reuse the original DS715_NODA.tar.gz archive")
    args = parser.parse_args()
    _, destination = assets()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        validate_assets(destination)
    elif args.archive:
        install(args.archive, destination)
    else:
        with tempfile.TemporaryDirectory(dir=destination.parent) as folder:
            archive = Path(folder) / "DS715_NODA.tar.gz"
            print("Downloading 5.31 GB from Zenodo", flush=True)
            with urlopen(URL, timeout=120) as response, archive.open("xb") as out:
                shutil.copyfileobj(response, out)
            install(archive, destination)
    print(f"Verified SIAM v2 fold 1 in {destination}")


if __name__ == "__main__":
    main()
