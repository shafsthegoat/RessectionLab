"""Package existing frozen native histories without changing their raw bytes."""
import gzip
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path


ROOT = Path(__file__).parent / "experiment-2"
NAMES = ("frontier-solid-candidate.json", "frontier-barrier-candidate.json",
         "patient-expanded-candidate.json")
rows = []
for name in NAMES:
    source = ROOT / name
    raw = source.read_bytes()
    before = sha256(raw).hexdigest()
    buffer = BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buffer, compresslevel=9, mtime=0) as output:
        output.write(raw)
    compressed = buffer.getvalue()
    assert gzip.decompress(compressed) == raw
    destination = source.with_suffix(source.suffix + ".gz")
    destination.write_bytes(compressed)
    assert source.read_bytes() == raw
    rows.append({"raw_file": name, "raw_bytes": len(raw), "raw_sha256": before,
                 "gzip_file": destination.name, "gzip_bytes": len(compressed),
                 "gzip_sha256": sha256(compressed).hexdigest(),
                 "roundtrip_byte_identical": True, "raw_source_unchanged": True})
receipt = {"format": "gzip", "compresslevel": 9, "mtime": 0, "header_filename": "",
           "candidate_declarations_and_script_unchanged": True, "files": rows}
(ROOT / "candidate-compression.json").write_text(json.dumps(receipt, indent=2) + "\n")
