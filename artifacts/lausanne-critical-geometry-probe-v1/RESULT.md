# Real annotation reaches the geometry checker

One existing-API probe on TRAIN sub-476/ses-20140519 produced `FORBIDDEN_COLLISION`.
Its 27 queried cells intersect three actual source-positive aneurysm cells; the
other 24 remain unknown. The same mathematical probe with exclusions disabled
produced no collision. That second result is a software ablation, never evidence
of safe tissue.

The query uses the canonical evidence resolver and unchanged source-coded frame.
It is an intentionally positioned, generic 2 mm probe, not a recorded maneuver
or measured commercial tool. The source image, annotation, case/evidence identity,
patient role and both false scanner/spatial admission flags remained unchanged.
No target, access support, route, training episode or clinical clearance was created.

One worker completed in 2.159907 seconds (2.332141 seconds supervised), with
621,543,424 bytes maximum sampled RSS, within 60 seconds / 1 GiB. Independent
saved-output review verified 68 source files and 151 input/evidence files, and
recomputed the exact three-cell intersection. It did not repeat image decoding
or geometry execution; global first-positive ordering remains source-authenticated,
not independently reconstructed from a dense mask.

This closes the shared-kernel consumption check for this query. A bound desktop
probe operation remains absent; full route planning still requires genuine target
and access evidence. The result does not establish complete vessel coverage,
clinical safety, registration accuracy or glioma-planning benefit.

`run_probe.py` is the unchanged executable used in the ignored build directory;
its frozen source dependencies are identified by `attempt-001/intent.json` and
Git commit `2911d31`. Original local snapshots remain retained. No raw image or
mask payload is included in this publication.
