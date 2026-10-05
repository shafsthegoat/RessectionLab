# Saved diagnostic evidence

Read `RESULT.md` for findings, limits and the recorded launch deviation.
`run-records.tar.gz` preserves the five original execution files byte-for-byte;
`run-records-receipt.json` records their hashes. Originals remain locally in the
ignored `run-01` directory. On a fresh checkout, extract this archive into this
artifact directory to restore those five files before using `report.py` or the
saved-record checker. No image decoding or simulation is needed for reporting.

`post-run-source.tar.gz` separately preserves the 61 source files and prospective
manifest. It was created after execution, as its receipt explicitly records.
It must not be represented as a pre-run immutable execution archive. The independent
saved-record check, source hashes and original selected-access reproduction bound
the narrower working-tree development diagnostic claim.
