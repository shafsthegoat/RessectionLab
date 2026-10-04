# Native axis performance review

Status: read-only review; no engine or adapter change, benchmark execution or
speedup claim. Choose an implementation after the public full-episode receipt.

The [single UCSF observation](../artifacts/native-proposer-integrity-v1/actual-case-observation.json)
measured 0.285–0.302 seconds per complete proposal/validation call, including
history reconstruction and full-grid exterior connectivity. Its one native
preview took 0.281 seconds. These are individual observations, not distributions
or controlled method comparisons. The adapter already validates once during an
observation's three synchronous feature passes. Subsequent public inventory,
clone and metrics accesses intentionally verify again; skipping those checks
would reopen the retained source/cavity mutation failures.

## One candidate: bounded exact capsule-coverage memoization

`NativeResectionEngine.preview_stroke` computes active-region and whole-shaft
capsule-to-cell intersections at every microstep. Those geometry queries depend
on source shape/affine and exact physical endpoints/radius, independently of the
remaining cavity. Repeated rays across resets can repeat the queries. Propose a
worker-local, configuration-bound LRU cache of only these immutable index arrays,
initially limited to 32 MiB of retained array payload and 16,384 entries, including
empty results; measure key/bookkeeping overhead and process peak RSS separately.
Share that cache across same-worker
clones only after lifecycle tests; never share mutable cavity state.

Keys must bind the geometry implementation/version, native configuration,
source shape/affine, and exact float64 endpoint/radius bytes. No rounded spatial
key or approximate lookup. On a miss call the existing geometry function;
preserve its cell ordering, tolerance and immutable array representation. Do not
precompute later microsteps after a rejected prefix. Oversized values bypass the
cache, and eviction must only cause recomputation.

Keep the top-level full-tool hard-exclusion/access check, cell containment,
prior-cavity shaft collision test, connected-frontier update, partial-contact
accounting, certificate checks and independent final audit unchanged. A cached
shaft cover still indexes **the current remaining tissue before that step's
removal**. Never cache feasibility, removed cells, frontier membership, policy
values or a successful integrity check. This preserves the full-tool check;
it only avoids recomputing the same set of potentially intersected cells.

This candidate does not reduce full-grid integrity work. Cold calls may be
slower; changing distal depths may give few exact cache hits. Full-episode
receipts must establish repeated geometry work before implementation is chosen.
Record capacity, hit/miss/eviction counts, query time, integrity time, full preview
time and retained payload separately. Changed wall-time throughput can alter
budget-limited search/RL trajectories even when each geometric result is exact;
retain a new runtime identity and do not relabel earlier study results.

## Executable probe design, pending authorization

The following synthetic-only probe can be run from the repository root after
review. It patches only the native module's pure query reference within the
process, leaving full-tool hard/access checking and all dynamic guards active.
It compares reference → cached → reference to check reversal and reports each
reset/preview/commit duration. It is a design, not an executed result.

```sh
.venv/bin/python - <<'PY'
from collections import OrderedDict
from time import perf_counter
from unittest.mock import patch
import json
import numpy as np
import resectionlab.native_resection as native
from resectionlab.geometry import AccessWindow

tissue = np.ones((9, 9, 10), bool)
labels = np.zeros(tissue.shape, np.int16)
labels[:, :, 2:9] = 1
config = native.NativeResectionConfig(
    tissue, labels, np.eye(4), AccessWindow((4, 4, -.5), (0, 0, 1), 6),
    native.NATIVE_GENERIC_TOOLS, 'synthetic-cache-probe', 'synthetic solid grid')
original = native.capsule_voxel_indices
cache, payload, calls, hits, evictions = OrderedDict(), 0, 0, 0, 0
cap, entry_cap = 32 * 1024 * 1024, 16384

def memoized(scene, start, end, radius):
    global payload, calls, hits, evictions
    calls += 1
    key = (config.fingerprint, scene.shape, scene.affine.tobytes(),
           np.asarray(start, dtype=np.float64).tobytes(),
           np.asarray(end, dtype=np.float64).tobytes(),
           np.asarray(radius, dtype=np.float64).tobytes())
    if key in cache:
        hits += 1
        cache.move_to_end(key)
        return cache[key]
    result = original(scene, start, end, radius)
    if result.nbytes <= cap:
        while cache and (payload + result.nbytes > cap or len(cache) >= entry_cap):
            payload -= cache.popitem(last=False)[1].nbytes
            evictions += 1
        cache[key] = result
        payload += result.nbytes
    return result

expected, rows = None, []
for name, query in [('reference', original), ('cached', memoized),
                    ('reference_reversal', original)]:
    engine = native.NativeResectionEngine(config)
    with patch.object(native, 'capsule_voxel_indices', query):
        for repetition in range(5):
            started = perf_counter()
            engine.reset()
            result = engine.execute_stroke('native-wide-aspiration', (4, 4, 8))
            elapsed = perf_counter() - started
            assert result.feasible
            signature = (json.dumps(result.to_history_record(), sort_keys=True),
                         engine.state_hash, engine.remaining_mask.tobytes(),
                         engine.removed_mask.tobytes(), engine.contact_mask.tobytes(),
                         engine.connected_free_mask.tobytes())
            if expected is None:
                expected = signature
            assert signature == expected
            rows.append(dict(mode=name, repetition=repetition, seconds=elapsed))
print(json.dumps(dict(rows=rows, query_calls=calls, cache_hits=hits,
                     cache_misses=calls-hits, cache_evictions=evictions,
                     retained_array_bytes=payload, all_results_identical=True)))
PY
```

The probe's cache namespace is one fixed process/configuration; a production
cache also needs an explicit implementation-version namespace. Before adoption,
extend paired reference/cached checks to zero capacity, forced eviction, clone
and reset isolation, changed source/affine/tool/radius, mirrored/oblique grids,
tangencies, the existing short-tip future-cut-borrowing adversary, and hard
barriers. Compare complete accepted **and rejected** preview records, first
failure positions, every microstep, cavity/contact masks, certificate ancestry,
proposal ledgers/IDs, observations, rewards and termination. Run the existing
independent native-history audit on both accepted histories; do not reuse its
result as a cache entry. Preserve all source/cavity mutation regressions.

Only after those checks and explicit approval, use the declared development
case and fixed episode/action trace in an isolated paired run. Alternate order,
separate cold/warm measurements, keep preview and transition counts identical,
and report full episode/setup/integrity/preview time plus peak RSS. Then disable
the cache and reproduce the reference receipt again. No final/stress worlds,
new targets, altered tool dimensions or speculative clinical claims belong in
this performance slice.
