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

The first separately invoked scientific acquisition stopped before any payload
bytes in1.827supervised seconds: Python's certificate-chain verification failed
for the original NIRD image endpoint. [Worker result](acquire-attempt-01/worker-result.json)
and [supervision](acquire-attempt-01/supervision.json) preserve the failure.
No mask, original image, QC or training result was created. A single subsequent
system-curl HEAD diagnosis timed out during connection; it did not establish a
working alternate trust path. TLS verification remains enabled.

## Native Mac trust transfer, October 8

The reviewed native trust fix acquired the exact frozen Case3 original image:
**9,156,131 bytes**, published MD5 matched, SHA256
`431ae1e23bd2984801c0b0d2720c9f4c240a404bab5fe37f5d8afdac521ddb20`.
It used existing macOS certificate trust without adding roots or disabling TLS.
The annotation request then returned HTTP429 after the official OSF redirect.
No mask or partial was published and no automatic retry ran.

[Worker result](acquisition-native-trust-01/worker-result.json),
[supervision](acquisition-native-trust-01/supervision.json) and
[independent verification](acquisition-native-trust-01/independent-review.json)
retain the partial acquisition and 17.266-second failed pair attempt. Original
byte fixity, exact source binding and child cleanup passed independent review.
No image arrays were opened for this review. Complete pairs, QC, component
fitting and recorded RL transitions remain zero. The next step is a deliberate
bounded mask resume after source service permits it, followed by separate QC.

## Complete Case3 pair and independently checked QC, October 8

One later deliberate bounded attempt acquired the frozen 28,518-byte mask in
3.957 seconds, reusing and rechecking the verified original image. The actual
HTTP chain was 302 → 302 → 200. Source bytes, rights and role were unchanged;
no retry ran inside the attempt. Both historical rate-limit failures remain.

The separate 0.803-second QC run and independent calculations agree: both
volumes are 338 × 303 × 245 (25,091,430 finite voxels), with **20,005 binary
positive cavity voxels**. The maximum original grid-corner difference is
**0.00003068 mm**, below the frozen 0.01 mm criterion. The raw headers differ
(image qform0/sform1; mask qform1/sform0); they have not been replaced.

[Acquisition](acquisition-complete-01/worker-result.json),
[QC](pair-qc-01/worker-result.json),
[independent review](pair-qc-01/independent-review.json),
[fixed-slice visual QC](visual-qc-01/fixed-native-cavity-panel.png).

Visualization used a prospectively declared single decode per volume, three
full-field native slices through the source-mask bounding-box midpoints, and a
fixed percentile display rule. It took 1.769 seconds and decoded 200,731,440
float32 bytes; this is not a hard RSS measurement/limit. The original arrays and
rights records remain unchanged. The byte-preserved plotting script retains its
original ignored-build execution layout; it is historical evidence, not a new
anatomical acceptance command.

All three panel outlines are legible and remain source annotations. This is
engineering inspection, not independent clinical/expert review. Visible cavity
is not complete removed tissue or a force/action measurement. Noncommercial
rights apply to labels. Training, spatial planning, clinical accuracy and RL
admission remain false; next qualify the remaining eligible TRAIN annotations
and define a source-faithful cavity component task with independent evaluation.
