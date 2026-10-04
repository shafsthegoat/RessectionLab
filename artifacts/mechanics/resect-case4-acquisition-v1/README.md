# RESECT Case4 byte acquisition

Root released exactly six transfers after commit
`4d1d5123572370e8af06730f17ea993813053dfc`, using prospective manifest SHA256
`2babb675a77d48bfd7f7df2748956b0de7e268a6a4ee731df508b3376401830d`.
The root release was saved before the first payload request. Its authorization
permits transfer and integrity checking only; measurement access remains closed.

All six files completed in **10.232 seconds**, totaling **35,068,010 bytes**.
Every provider MD5 and byte count matched, and local SHA256 values are recorded
in `acquisition-receipt.json`. All six originals were rehashed after completion.
There were no transfer failures, retries, redirects, overwrites or substitutions.
An earlier mistyped working-directory setup could not start a process; that
failure is retained in the root release and occurred before any payload request.

The ignored originals are under
`data/mechanics/resect-case4-v1/RESECT/NIFTI/Case4/`. The four compressed image
files and two landmark files were processed only as opaque byte streams.
No gzip decompression, image/header loading, tag-text reading, coordinate
inspection or plots occurred. No other patient or cohort was accessed, and
existing cohort roles remain unchanged.

`acquire.py` is the exact transfer driver. It binds the committed manifest and
release hashes, allows only the six Case4 paths, uses verified HTTPS with a
45-second per-file wall limit, checks the exact response URL/status/byte count,
and promotes verified temporary files without overwriting existing files.
Original and frozen manifest hashes remained unchanged. HTTP response headers
and byte-only logs are retained; ETags are not treated as checksums.

The next step requires the committed source-only landmark partition/access
contract and a separate root release. This acquisition provides no anatomical,
registration, mechanics or clinical approval.
