# Real annotation pilots and strict NIfTI extension support

The cache-only sub476 pilot passed: the untouched 82,420-byte source mask
contains 193 positive voxels, with the previously declared source-reference
grid proof. Independent bounded streaming and affine checks agree. This is
one reused source annotation, not an additional patient or training example.

The first new transfer acquired the exact 36,551-byte sub022 mask in one GET,
matching published MD5 and retained SHA256. Its content QC stopped before
scalar counting because the initial parser rejected all nonzero extension
bytes. Source acquisition passed; content and grid admission did not. All
148 inventory outcomes and the original failed receipt remain preserved in
`transfer-pilot/`. No automatic retry occurred.

A bounded metadata diagnosis found a valid extension: little-endian NIfTI-1,
data offset 592, one 240-byte code-0 record starting at byte 352. The official
NIfTI specification permits readers to skip unknown extension contents. The
new parser checks byte order, signed sizes/codes, 16-byte record alignment,
bounds and exact chain exhaustion. It retains lengths and hashes without
interpreting or publishing private payload contents. Arbitrary nonzero padding,
reserved indicator bytes, malformed records and unframed trailers still fail.
The unchanged original is the source of truth; no header or label is repaired.

All **56 owner and 32 independent controls pass**. The reviewer independently
read exactly the 592-byte source prefix, confirmed the original payload hashes,
and checked that scalar/binary, grid, rights, split and admission gates are
unchanged. Its initial framing-control length typo failed before production was
called; the original eight parameterized failures and correction are retained
separately. The largest allowed metadata chain fits the existing 2 MiB receipt
bound. These are parser controls, not invented patient data.

Next: a separately recorded existing-only sub022 QC repeat after this source
freeze. No sub022 scalar values, reference-grid result, full-inventory content
result, fitting or planning admission follows from the metadata diagnosis.
Background remains unknown. Scanner-frame and surgical-use validation remain
open even when a source-grid component check passes.

Historical scripts/receipts retain their original local paths and exact hashes.
Raw image/mask files and execution snapshots stay in ignored local storage;
this compact record is not a claim that copying receipts reproduces acquisition.
