# Prepared intake for 144 real TRAIN annotation masks

The frozen manifest retains all 148 source outcomes:144 qualified masks across 106 people / 117 sessions and four metadata failures. Subtype provenance remains111unresolved manual regions, 31 person/date crosswalk mismatches and two exact voxelwise crosswalk matches. These are distinct supervision candidates. Background remains unknown, source-missing labels are not negatives, and no mask is automatically admitted to training or planning.

The runner verifies exact object versions/bytes/hashes, preserves source/receipt snapshots, streams binary-value checks with bounded chunks, and checks each mask against its own original TOF grid. Originals absent at intake leave reference/grid QC explicitly pending. Source T1 and TOF failures remain separate: sub253’s T1 conflict does not erase its passing TOF record. No label repair, resampling or scanner-world alignment is inferred.

Initial independent review passed 48 controls but reproduced an interruption-record defect with a real software-control child: cleanup succeeded, while the started attempt was mislabeled unattempted. Initial source and the negative verdict remain archived. Root repaired pre-start/post-start bookkeeping and receipt retention, adding four process controls. Final independent review passed **54 checks**, including fresh-root metadata preflight using only 14 copied metadata files and no ignored build/data cache. Root’s owner-plus-independent subset passed 39 checks in 0.60 seconds. These verify software and source metadata, not acquired annotation quality.

Eleven public source metadata/proof files are preserved byte-for-byte under `source-metadata/`; the manifest points there. Original historical qualification records retain their historical paths. The methods XML contains the authors’ CC BY 4.0 attribution/license; the crosswalk’s source Apache 2.0 notice is retained. The pickle crosswalk is parsed as metadata opcodes during review, never executed. No scientific images or masks are committed.

Before payload execution, default preflight is:

```sh
.venv/bin/python scripts/lausanne_annotation_intake.py preflight
```

An explicit payload run requires `batch --execute`, manifest SHA
`f779066f5cb784f623446f12565eb1405994ac9e3b41c1fef8210546598de624`,
a fresh `--run-id`, and declared `--max-seconds`/`--max-bytes`.
`--mask` limits the invocation to one exact qualified path; `--existing-only`
forbids transfer. Worker time is at most 120 seconds, serial batch time at most 600.
The logical mask limit is 1 GiB and actual decoding uses at most 262,144 voxels per
chunk under a 16 MiB working-chunk envelope; this is not a whole-process RSS bound.
Unsupported extensions, scalar labels or grids remain explicit failures.

At this preparation checkpoint there have been no new annotation transfers,
payload decodes, training updates, planning admissions or patient evaluations.
The next action is one cache-only source-mask QC pilot, then bounded full-inventory
acquisition/QC with failures retained.
