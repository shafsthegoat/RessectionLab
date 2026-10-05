# First specimen experiment: stopped at output decoding

The single released attempt from `add7e8b9c73a8a4fe28badd335bd59fcfbeb91de`
stopped after its first case, `compression:N4:S60:reference`. FEBio exited zero
with normal termination in 0.7349 s, but the frozen reader rejected its time
header. The numerical case evaluation did not finish. Nineteen cases remain
unexecuted, no calibration or held-out CSV member was opened, and no material
fit, prediction freeze, or physical validation result exists. There was no retry.

The pinned FEBio v4.13 `FECore/DataRecord.cpp` writes `*Time` using `%.9lg`
(line 222), while numerical primitive rows use 12 significant digits (line 114).
The reader incorrectly applied a 12-digit assumption and an absolute `5e-12`
time tolerance to the header. Its first reported fraction, `0.0166666667`, differs
from `1/60` by `3.333333262189875e-11`. Forty of the 61 node headers and forty of
the 61 element headers exceed that assumption. All saved headers match the
source's nine-significant-digit rendering. This diagnosis uses saved headers
and the hash-verified upstream source; it changes no solver output or mechanical
tolerance. See [failure-diagnostic.json](failure-diagnostic.json).

Before execution, one focused suite from the isolated, read-only Git archive
passed **219 tests in 3.34 s**. The release bound the exact runtime, original
meshes and decks, current helpers, and prospective CSV interpretation. CSV format
remained an explicitly unverified assumption; the run never reached that stage.
An earlier metadata assembly reference error is retained separately: final review
metadata was committed after the execution snapshot and was subsequently bound
to its genuine `d5640f2` commit without changing the executed archive or repeating
the test suite.

The experiment worker ended with exit 1 after 2.7135 s; the complete launcher
took 4.2656 s. Sampled worker-group peak RSS was 67,977,216 bytes. No resource cap
triggered. These observations describe this failed attempt and are not a speed or
accuracy benchmark.

All 4,232 archived source files, 78 driver inputs, 52 original mesh-preparation
files, the compressed measurement archive, and runtime bindings were verified
unchanged afterward. Raw outputs remain read-only under the ignored directory
`outputs/mechanics/hbe-01-03-poc-v1/experiment`; the tracked
[raw-output-index.json](raw-output-index.json) binds every retained file.
[outcome.json](outcome.json) and [post-execution-integrity.json](post-execution-integrity.json)
record the terminal state and integrity checks.

Root separately authorized a source-format regression and independent evaluation
of the already saved coarse outputs. That work must retain this failed attempt;
it does not authorize another solver run, measured-data access, or continuation.
