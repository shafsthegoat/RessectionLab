# Experimental exact capsule-cover cache

`experimental_capsule_cache.py` is an opt-in prototype. Production native
engines, the axis adapter and learners do not import it. Its context manager
temporarily replaces the native module's pure capsule-to-cell query and restores
the original even after an exception. It rejects nesting, a foreign replacement,
cross-thread calls and query reentrancy. Use it only in one isolated worker with
the intended native configuration, without another native workload in that
process. No public benchmark or gradient run is authorized by this helper.

`ExactCapsuleCoverCache(native_config, max_payload_bytes=32*1024**2,
max_entries=16384)` validates the source configuration, then guards its immutable
identity on every query. Keys include the configuration/source namespace,
geometry source/version/mode, exact source frame and exact float64 physical
endpoint/radius bytes. A scene's inverse, cell radius and orthogonal spacing must
match that frame. Native tool dimensions already use float64; this prototype
rejects alternate floating radius precisions. Different physical keys never use
rounding or approximate matching.

Guards run after numeric input conversion, which can execute a caller's array
conversion method. The declared runtime boundary includes the cover's direct
geometry helpers and scene coordinate methods, version, epsilon and derived
descriptors. It rejects replacement within that boundary; it does not claim to
defend arbitrary monkeypatching of Python, NumPy or every process global.

Cached arrays retain the original immutable bytes-backed representation. Hits
also verify the saved descriptor so a base-class metadata bypass cannot silently
change their interpretation. The LRU has both a payload and entry cap, including
empty arrays. Zero capacities disable storage; oversized results bypass storage.
Payload statistics exclude key/bookkeeping overhead, which must be reported
separately from cumulative process peak RSS.

Only geometric cell coverage is reused. The existing full-tool hard/access
check, prior-cavity swept-shaft collision test, contained-cell credit,
connected-frontier update, partial-contact costs, certificate ancestry and
independent native replay still execute. No feasibility, cavity, integrity,
removal or audit result is cached. The helper does not repair or conceal mutated
engine state; the axis proposer's normal full integrity checks remain required.

The synthetic launcher is deliberately unable to load a patient case:

```sh
.venv/bin/python scripts/probe_native_capsule_cache.py \
  --output artifacts/capsule-cache-prototype-v1/synthetic-probe.json \
  --repetitions 5
```

It refuses to overwrite an existing receipt and runs reference → cached →
reference with exact preview, history, four-mask and ancestry comparisons. Two
separate independent history audits must pass, and a hard barrier must remain
rejected. The initial receipt recorded 46 cold misses and 184 warm hits, retaining
105,720 array bytes. Its cold cached stroke took 25.68 ms, warm repetitions
12.44–12.49 ms, and reference repetitions 23.28–30.54 ms. These tiny local
observations had no quiet-window guarantee and do not establish a patient or
complete-episode speedup. Source hashes and phase times are retained in the
receipt; no final/stress worlds or gradients were used.

The repaired prototype passed 78 focused tests. Initial input-conversion and
direct-helper guard failures, their source snapshots, and repaired receipts are
retained under `artifacts/capsule-cache-prototype-v1`; the separate independent
review receipt is stored there too. `synthetic-probe-repaired.json` repeats the
full reference/cache/reversal and two-audit check with the repaired helper and
includes the imported source-identity guard's file hash. The original receipt
remains a historical observation of its recorded source.

Before any public paired run, review a prospective declaration with exact source/runtime identities, fixed
action/episode trace, cold/warm scopes, capacity, equal work counts, full
integrity costs, setup cost and memory reporting. Cache throughput can change
which work fits a wall budget; earlier scientific results must retain their
original runtime identity.
