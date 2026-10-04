# Fixed-trace native cache probe

Both cached phases had zero hits and took longer than both reference phases. This declared capacity did not improve the fixed trace.

Descriptive paired fixed trace on one development case. Reference bracket is order/OS variation, not a confidence interval; no whole-learning throughput or clinical claim.

| Phase | Reset (s) | Three transitions (s) | Other phase work (s) | Export (s) | Whole phase (s) |
|---|---:|---:|---:|---:|---:|
| reference_before | 7.147 | 12.115 | 1.467 | 0.261 | 21.182 |
| cached_cold | 7.767 | 12.969 | 1.677 | 0.261 | 22.896 |
| cached_warm | 7.815 | 13.050 | 1.720 | 0.261 | 23.050 |
| reference_after | 7.168 | 12.389 | 1.633 | 0.262 | 21.670 |

Nested proposer, adapter preview/integrity and cache query/compute times overlap; do not sum. Certificate capture is unisolated within phase residual, alongside accounting, guards and progress writes.

| Phase | Complete proposer calls / s | Native previews / s | Cache calls | Hits | Misses | Evictions |
|---|---:|---:|---:|---:|---:|---:|
| reference_before | 10 / 3.064 | 66 / 14.939 | — | — | — | — |
| cached_cold | 10 / 3.124 | 66 / 16.334 | 22364 | 0 | 22364 | 16902 |
| cached_warm | 10 / 3.097 | 66 / 16.478 | 22364 | 0 | 22364 | 22364 |
| reference_after | 10 / 3.073 | 66 / 15.158 | 0 | 0 | 0 | 0 |

Warm replay ended with 5,462 entries and 33,553,224 array-buffer bytes, within its 16,384-entry / 33,554,432-byte limits. Keys and bookkeeping are outside that payload counter. Eviction plus zero warm hits is consistent with the replay evicting covers before reuse; this is an interpretation of the recorded counters, not an access-distance trace.

Source preparation took 0.641 s; the common uncached template took 8.147 s; empty-cache construction took 0.069 s. The four passing native audits took 59.559 s in total. Worker wall time was 157.790 s and launcher wall time was 158.288 s.

Process peak RSS was 2,289,860,608 bytes. Cumulative RSS includes all phase snapshots, histories and other process allocations; it does not isolate cache memory.

Inference from completed source-bound runner checks, not a separately persisted or independently observed callable-restoration receipt.

This reader performs only artifact checks and arithmetic; it runs no simulation, geometry replay, policy or gradient update.
