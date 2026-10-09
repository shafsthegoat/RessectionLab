# Staged MR conversion preparation

The revised pipeline separates full-cohort metadata validation from selected-series conversion, reaping the first worker before the second starts. Immutable input reads and persistent read-intent accounting avoid the earlier mutable pixel-cache issue. The first supervisor-accounting failure remains preserved; v2 passes 14 generated controls and independent checks of both injected failure paths.

The separately reviewed next control generates 59,520 metadata records and 64 signed-16 MR planes of 512 × 512 pixels. It has two independent launch steps: at most 60 seconds for generation, then 90 seconds for the unchanged staged pipeline; each worker has a sampled 512 MiB RSS stop condition. No full fixture or patient conversion has run at this preparation commit. Matching synthetic cardinality does not reproduce every allocation of the actual clinical metadata.

The adjacent evidence indexes bind exact source, tests, prior negative evidence, tiny execution receipts, protocol and independent reviews. Archives contain source and small receipts only; no image arrays, patient payloads or models. The existing geometry, acquisition-time and patient-admission restrictions remain unchanged.
