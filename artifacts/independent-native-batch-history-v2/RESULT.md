# V2: complete-history parity with a faster batch audit on one numerical control

The single declared scalar → batch → scalar attempt completed. All three certificates exactly match the historical certificate after canonical JSON serialization. Their complete uncompressed trace payloads are byte-identical: 97 active-contact sets, 679 ordered cell queries and first witnesses, and 97 remaining-tissue/connected-cavity prefixes per phase. All trace hooks were restored. The batch phase used 97 batch shaft queries and retained 582 scalar cell queries; each scalar phase used 679 scalar queries. Full-tool hard-exclusion checks and causal prior-tissue masks remain part of the audit.

| Phase | Complete audit including trace capture | Nested trace capture | Separate gzip export |
| --- | ---: | ---: | ---: |
| Scalar before | 43.007 s | 0.588 s | 0.922 s |
| Batch | 4.016 s | 0.581 s | 0.916 s |
| Scalar after | 43.176 s | 0.683 s | 0.916 s |

For this fixed analytic history and runtime, the scalar-to-batch audit-time ratio was **10.71–10.75**; the ratio using the scalar arithmetic midpoint was 10.73. The scalar observations differ by 0.39% of their midpoint. These are three ordered observations, not a repeated-sample estimate or confidence interval. Audit times include common instrumentation; no overhead was subtracted to claim a speedup.

Setup before the first audit took 6.594 s, including 3.350 s for native reconstruction and binding. Worker time was 100.226 s, parent launcher time 101.100 s, and outer wrapper time 101.147 s. Sampled and worker peak RSS were both 303,218,688 bytes. The unchanged 140 s cooperative, 150 s parent and 3 GiB limits were respected, with one worker and numerical thread settings of one. There was no kill, retry, cap change, patient load or learning update. Memory sampling can miss brief excursions; numerical thread settings do not assert CPU affinity.

The original 20-degree, 0.125 mm numerical-control scene was reconstructed once. Its source, configuration and full 97-microstep history hashes matched the previously certified row before timing. The physical tool, anatomy, entry, target, 0.0625 mm motion steps and removal semantics were unchanged. The certificate retains 20.539062499999986 mm³ of fully contained source-cell tissue; this performance comparison does not resolve the earlier source-grid discretization deficit or validate tissue mechanics.

Execution used the immutable full Git archive of `ba437632417b2df7a42481f6fc4314c03fbd0d6b`. All 3,804 original files and the archive tar hash remained unchanged. The saved source receipt agrees with the archive; runtime additions were Python bytecode only. The archived worker asserts numerical-module import origins before construction. Post-run verification checked saved bytes and hashes without executing geometry again.

The failed [V1 attempt](../independent-native-batch-history-v1/RESULT.md) remains preserved. V2 changed only harness certificate representation equality and its declaration/provenance; it did not relax numerical equality. This closes the declared one-history comparison. The production default remains scalar. No patient, learning, universal performance, convergence or clinical-safety claim follows from this result.

Evidence: [original report](attempt-01/report.json), [launcher acceptance](attempt-01/launcher.json), [archive and exact parity verification](archive-and-parity-verification.json), [timing arithmetic](timing-summary.json), [prospective execution baseline](execution-baseline.json), and [file index](artifact-index.json). Compressed source history and all three full traces remain under [attempt-01](attempt-01/).
