# Bounded RESECT cavity intake

October 8, 2026. The Case3 TRAIN pilot is fixed before scientific payload
access: one original during-resection ultrasound and its human-reviewed
visible-cavity mask, 9,184,649 bytes combined. Patient roles, original source
MD5, mask revision 2 and its published hashes are bound in the
[manifest](../../manifests/resect-case3-cavity-pilot-v1.json). The official
[revision endpoint](https://api.osf.io/v2/files/64cd21fe2fe4962f1561b1e2/versions/2/)
resolved the explicit revision-2 download route. The original image's fixity
matches the archived NIRD source inventory.

The runner separates rights acquisition, original-byte acquisition and QC.
Acquisition/QC require the exact rights README first. Only reviewed HTTPS
destinations are accepted, with a five-request redirect limit, fixed source
size/hash checks and no rate-limit retry. Each invocation has a 300-second
supervisor and worker watchdog, immutable source snapshots and closure records.
QC checks source headers, units, physical grids, finite values and binary label
vocabulary. It does not resample, register, threshold, fit a model or open Case4.

The actual rights attempt failed with HTTP 429 after one official OSF redirect.
It exited in 0.7040433 supervised seconds; no Retry-After header was supplied.
No rights file, new patient image or mask was published.
[Original worker result](rights-attempt-01/worker-result.json),
[supervision](rights-attempt-01/supervision.json) and
[executed source binding](rights-attempt-01/source.json) preserve the failure.
The exact executed source snapshot stays with the local attempt; its subsequent
watchdog/setup-closure refinements are not retroactively attributed to that run.
There has been no automatic or alternate-backend retry.

Earlier source inspection read CC BY-NC-SA 4.0 plus an explicit financial-benefit
restriction. This pilot is noncommercial research. Archiving the byte-preserved
rights record remains a prerequisite to scientific acquisition; neither missing
text nor a downloaded MRI establishes commercial rights.

Independent read-only review found no remaining concrete blocker for the bounded
runner after corrections to checkout import binding, standalone worker lifetime
and setup-failure receipts. The ten focused controls cover real source metadata
and refusal/process behavior; they generate no patient image or clinical event.
One initial setup-closure test failed because its temporary DATA directory was
outside the unchanged test ROOT; the test fixture was corrected. This was not an
observed production acquisition failure.

The mask protocol uses manual contours, morphological interpolation and expert
review; it is not all hand-drawn voxels. Visible dark cavity may omit blood-filled
regions and cannot establish complete removed tissue, cutting force or surgical
reward. Byte and grid checks do not establish anatomical correctness.
Training/spatial admission remain false and optimizer/RL-transition counts zero.
Next: obtain the exact rights record when source service permits, acquire the
frozen Case3 pair, then perform structural and anatomical review before any
component fitting. Frozen SELECT/EVAL and Case4 remain closed to this pilot.

Reproducible commands (separate invocations; stop on failure):

```sh
.venv/bin/python scripts/acquire_resect_cavity.py rights
.venv/bin/python scripts/acquire_resect_cavity.py acquire
.venv/bin/python scripts/acquire_resect_cavity.py qc
```

## Exact rights archival and current OSF route, October 8

A deliberate new bounded follow-up acquired the exact revision2 README through
OSF's official redirect chain in2.889seconds. All1656bytes, published MD5 and
SHA256 match. [Original rights text](rights-review-02/README.txt),
[HTTP receipt](rights-review-02/retrieval.json) and
[independent source review](rights-review-02/review.json) are retained.
The earlier429 and an intermediate local redirect refusal remain failures.

The downloader now permits only the exact OSF storage bucket/object identified
by the frozen rights/mask SHA256. Thirteen controls pass; independent review
verified the source and repaired malformed-query exception logging before
scientific acquisition. Source roles, manifests, declared bytes/hashes and
no-retry behavior are unchanged. The authenticated README is installed at the
runtime prerequisite path. The license is CC BY-NC-SA4.0, with the stated
financial-benefit restriction; this is noncommercial research, not commercial
clearance. Scientific acquisition and QC remain the next separate invocations.
