# Continuous source acquisition preparation

October 8, 2026. The download-only queue covers 890 frozen TRAIN files:
840 Lausanne original images/embedded sidecars and 50 RESECT image/mask files,
10,578,593,091 bytes in total. It reuses verified complete files, retains failed
attempts and partials, and continues eligible files without per-batch review gates.
The separately completed 144-mask Lausanne inventory is outside this queue.

Validation passed: 126 owner/existing transport tests and 17 independent provider
controls, plus real-process deadline/descendant and startup-callback cleanup checks.
Initial review found provider Retry-After gaps on terminal HTTP errors and delayed
redirects. Those failures remain preserved; final verification confirms repair.
Unresolved terminal files retain their denominator and produce exit code 2 rather
than successful dataset completion. Source/TLS/checksum/redirect rules stay fixed.

New immutable receipts live under `data/acquisition/continuous-source-v1` and report
only byte verification. Header, geometry and anatomical QC are explicitly `not_run`;
training and spatial admission are false. No source images were decoded or fetched
by these preparation tests. Detailed QC is a separate subsequent phase.

Lausanne supports validated byte-range resume. RESECT supports queue resumption and
verified complete-file reuse; its interrupted partials are preserved, but byte-range
resume is not yet implemented. Hard-killed attempts without a closeout require
explicit recovery rather than being silently treated as complete. Provider cooldowns
honor numeric/date Retry-After, and scientific transfers remain serial.

Run `.venv/bin/python scripts/continuous_source_acquisition.py run` from the repository;
`status` is metadata-only. Freeze the eight bound acquisition/helper scripts while
the queue runs. RL/desktop edits are outside this closure and can proceed concurrently.
The source stores original patient roles and missing-label exclusions unchanged.
