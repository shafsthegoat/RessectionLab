"""Metadata-only diagnostic; never decompress past the source vox_offset."""
from datetime import datetime, timezone
import hashlib
import inspect
import json
import math
from pathlib import Path
import struct
import zlib

import nibabel
import nibabel.nifti1

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
RELATIVE = "data/anatomy/ds003949-v1.0.1/derivatives/manual_masks/sub-022/ses-20101011/anat/sub-022_ses-20101011_desc-Lesion_1_mask.nii.gz"
SOURCE_SHA = "ba83116ae5fa9744bb83eb338cd491b431691b3a54a5a5b3c8d6111a1d2eec11"
RECEIPT = "data/anatomy/ds003949-v1.0.1/train-annotation-intake-v1/attempts/pilot-sub022-transfer-01/sub-022_ses-20101011_desc-Lesion_1_mask.nii.gz/receipt.json"
RECEIPT_SHA = "1a8ffdf58847b6468f884905e5101313d6799f2addeeaa0f68350dce705f1c09"


def sha(payload):
    return hashlib.sha256(payload).hexdigest()


source = ROOT / RELATIVE
assert source.is_file() and not source.is_symlink()
compressed = source.read_bytes()
assert len(compressed) == 36551 and sha(compressed) == SOURCE_SHA
receipt_bytes = (ROOT / RECEIPT).read_bytes()
assert sha(receipt_bytes) == RECEIPT_SHA
decoder = zlib.decompressobj(16 + zlib.MAX_WBITS)
header = decoder.decompress(compressed, max_length=348)
assert len(header) == 348
endian = "<" if struct.unpack_from("<i", header)[0] == 348 else ">"
assert struct.unpack_from(endian + "i", header)[0] == 348
assert 1 <= struct.unpack_from(endian + "h", header, 40)[0] <= 7
assert header[344:348] == b"n+1\0"
offset = struct.unpack_from(endian + "f", header, 108)[0]
assert math.isfinite(offset) and offset.is_integer() and 352 <= offset <= 65536
offset = int(offset)
tail = decoder.decompress(decoder.unconsumed_tail, max_length=offset - 348)
assert len(tail) == offset - 348
prefix = header + tail
assert prefix[348:352] == b"\x01\0\0\0"
cursor = 352
extensions = []
while cursor < offset:
    assert offset - cursor >= 8
    esize, ecode = struct.unpack_from(endian + "ii", prefix, cursor)
    assert esize >= 16 and esize % 16 == 0 and ecode >= 0 and cursor + esize <= offset
    block = prefix[cursor:cursor + esize]
    data = block[8:]
    extensions.append({
        "offset": cursor, "end_exclusive": cursor + esize, "esize": esize, "ecode": ecode,
        "official_classification": "unknown private format" if ecode == 0 else "uninterpreted",
        "nibabel_code_label": nibabel.nifti1.extension_codes.label.get(ecode, "unregistered"),
        "block_sha256": sha(block), "payload_bytes": len(data), "payload_sha256": sha(data),
        "payload_terminal_zero_bytes": len(data) - len(data.rstrip(b"\0")),
        "payload_interpreted_or_copied_to_output": False,
    })
    cursor += esize
assert cursor == offset
official = OUT / "official-nifti1.h"
official_receipt = json.loads((OUT / "official-source-receipt.json").read_bytes())
assert sha(official.read_bytes()) == official_receipt["sha256"]
implementation = Path(inspect.getfile(nibabel.nifti1))
crc, isize = struct.unpack("<II", compressed[-8:])
report = {
    "schema": "lausanne-sub022-extension-diagnostic-v1",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "source": {"path": RELATIVE, "bytes": len(compressed), "sha256": sha(compressed)},
    "original_failed_receipt": {"path": RECEIPT, "sha256": sha(receipt_bytes)},
    "operation": {
        "scientific_transfers": 0, "documentation_requests_in_diagnostic_task": 1,
        "header_prefix_bytes_this_run": len(prefix),
        "task_total_prefix_bytes_including_initial_probe": 1184,
        "authorized_decompressed_ceiling": 65536,
        "scalar_bytes_decompressed": 0, "scalar_values_decoded_or_counted": 0,
        "method": "zlib gzip decoder max_length=348, then max_length=vox_offset-348; no flush or unbounded decompression",
        "no_payload_content_output": True, "source_files_modified": False,
    },
    "header": {"format": "single-file NIfTI-1", "endianness": "little" if endian == "<" else "big",
               "vox_offset": offset, "header_sha256": sha(header), "prefix_sha256": sha(prefix),
               "extension_indicator": list(prefix[348:352])},
    "extension_region": {"offset": 352, "bytes": offset - 352, "sha256": sha(prefix[352:]),
                         "extension_count": len(extensions), "extensions": extensions,
                         "unframed_trailing_bytes": offset - cursor,
                         "interpretation": "Structurally valid extension chain ending exactly at vox_offset; zero suffix is inside the declared extension payload, not unframed padding."},
    "gzip_trailer": {"stored_crc32_hex": f"{crc:08x}", "stored_isize_mod_2pow32": isize,
                     "crc_or_total_uncompressed_size_validated": False,
                     "limitation": "Trailer fields are read from compressed bytes only. Full gzip integrity remains for authorized scalar intake; no scalar count is derived."},
    "primary_specification": {"url": official_receipt["url"], "retrieved_at": official_receipt["requested_at"],
                              "path": str(official.relative_to(ROOT)), "sha256": official_receipt["sha256"],
                              "sections": "nifti1.h lines 213-301: HEADER EXTENSIONS, nifti1_extender, nifti1_extension",
                              "version_limit": "Upstream master URL was mutable; exact returned document bytes and hash are retained."},
    "local_implementation": {"path": str(implementation.relative_to(ROOT)), "sha256": sha(implementation.read_bytes()),
                             "nibabel_version": nibabel.__version__,
                             "sections": "nifti1.py lines 648-676 extension_codes; 728-790 Nifti1Extensions.from_fileobj; 856-877 Nifti1Header.from_fileobj",
                             "observations": ["Code 0 maps to ignore.", "Extension presence follows the first indicator byte; extent is bounded by vox_offset.",
                                              "Nibabel swaps esize/ecode with header endianness and strips terminal NULLs from interpreted extension content.",
                                              "Nibabel warns and continues for non-16-aligned esize. The proposed intake should retain its stricter framing refusal, not copy this permissive behavior."]},
    "next_code_contract": [
        "Retain existing immutable object identity, header/data-offset bounds and untouched source file. No redownload or repair is needed for this format finding.",
        "Separate bytes 348-351 from extension records. For this source [1,0,0,0] is present; do not treat it as required zero padding.",
        "For each record before vox_offset, decode only the two signed int32 framing fields using header byte order. Require positive esize >=16, divisibility by16, nonnegative ecode, forward progress and end<=vox_offset.",
        "The observed single code0 block occupies bytes352..591. Preserve its exact block/payload hash, length, range and code; leave its contents uninterpreted. Do not instantiate extension content handlers, execute payloads, expose comments, or use extensions as annotation/coordinate evidence.",
        "Keep original payload bytes unchanged, including the 16 terminal zero bytes inside this declared record. Hash before any stripping; no reserialization is needed.",
        "Require exact chain exhaustion at vox_offset for the new supported-extension branch. Unframed nonzero bytes, invalid sizes, negative codes, overrun, or ambiguous trailer remain failures. The existing extension-absent bounded all-zero-padding branch can remain separate.",
        "This is standard extension preservation/skip support, not arbitrary acceptance of any bytes before vox_offset. The parser can report unknown nonnegative codes without interpreting them; acceptance policy must stay explicit.",
        "After the parser change is independently reviewed, explicitly rerun this same cached immutable mask in existing-only mode. Only then perform content/scalar, same-TOF reference and grid QC; preserve the first failed receipt.",
        "Passing extension framing changes no binary-content, regional-label, contour, scanner-frame, planning or training admission. Those checks remain outstanding for this mask."
    ],
    "assumptions_and_limits": [
        "The private format itself has not been interpreted or validated; that is unnecessary for safely skipping a structurally valid NIfTI extension.",
        "Terminal zero bytes are observed inside edata; their application-specific purpose is not inferred.",
        "No scalar data, mask quality, anatomy, label location or counts were inspected in this diagnostic.",
        "Compressed source SHA256 was checked before and after the diagnostic. Existing failed receipt bytes were unchanged."
    ],
    "diagnostic_source_sha256": sha(Path(__file__).read_bytes()),
}
assert sha(source.read_bytes()) == SOURCE_SHA
assert sha((ROOT / RECEIPT).read_bytes()) == RECEIPT_SHA
with (OUT / "report.json").open("xb") as handle:
    handle.write((json.dumps(report, indent=2, allow_nan=False) + "\n").encode())
print(json.dumps({"report": str((OUT / "report.json").relative_to(ROOT)), "report_sha256": sha((OUT / "report.json").read_bytes()),
                  "vox_offset": offset, "extension_count": len(extensions), "extensions": extensions,
                  "scalar_bytes_decompressed": 0}, indent=2))
