# Native initial-geometry cache development profile

This bounded local experiment profiles preparation, repeated episode reset,
two prescribed native strokes, and cloning on UCSF-PDGM-0004. It never opens
selection or final-evaluation worlds and does not select an improved policy.

The two source snapshots differ only in `resectionlab/native_simulation.py`.
Their SHA-256 manifests accompany them. The input case hash is recorded in the
full results; the source bundle remains in the local outputs directory.

`comparison.json` contains the measured reset improvement and memory cost.
`before/` and `after/` contain per-phase profiles, summaries, and compressed
complete output records. Decompressed output bytes match exactly, including
observations, model and world hashes, rewards, metrics, and native microstep
histories. `result-file-sha256.json` records those file hashes.

The independent source-grid audit in `independent_audit.json` passes for the
captured two-stroke sequence: 266 mm³ claimed and fully contained tissue,
zero unsupported removal, and complete-tool plus frontier checks. This certifies
the declared discrete geometric primitive, not clinical safety or tissue forces.

To repeat a profile from either preserved source snapshot, choose a new output
directory so the original measurements remain unchanged:

```sh
PYTHONPATH=artifacts/performance/native-reset-cache-v1/baseline-source \
  .venv/bin/python artifacts/performance/native-reset-cache-v1/profile_native.py \
  --case outputs/cases/UCSF-PDGM-0004.ressectionlab --output /tmp/native-baseline-repeat

PYTHONPATH=artifacts/performance/native-reset-cache-v1/after-source \
  .venv/bin/python artifacts/performance/native-reset-cache-v1/profile_native.py \
  --case outputs/cases/UCSF-PDGM-0004.ressectionlab --output /tmp/native-cached-repeat
```

The profile includes instrumentation overhead. The Mac has 16 GiB unified
memory; process peak RSS changed from 1.168 to 1.202 GiB. Concurrent system load
was uncontrolled. Five reset samples measure this one workload, not universal
training throughput. Cold preparation still costs about 2.5 seconds and remains
charged separately for every fresh experimental arm. Further cavity proposals
and history copies are unchanged and remain measured bottlenecks.
