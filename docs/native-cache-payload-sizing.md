# Prospective raw capsule payload sizing

This diagnostic is declared and awaiting independent review and a separate
execution release. It does not change the experimental cache or any application
default. The completed public cache probe had zero hits at 32 MiB; its exact
argument reconstruction found 8,812 distinct queries in each 22,364-call trace.
Neither result measured the complete cover arrays' storage requirements.

`scripts/measure_native_cache_payloads.py` will read the verified saved argument
artifact, then call the original frozen `capsule_voxel_indices` function once per
distinct tuple of seven float64 arguments. Deduplication uses all argument bytes,
including signed zero and one-ULP differences. Projection hashes provide
provenance; they do not replace those bytes as identity.

The only scene is an empty, source-shaped boolean grid and the recorded native
RAS affine. Pure coverage depends on its dimensions and physical frame, without
using the forbidden-mask occupancy. The diagnostic never loads a patient image,
labels or tissue masks and never constructs a simulator, samples a world, selects
a route or updates a policy. Only the frozen standalone geometry module is loaded
after its hash, the original run's authorities and numerical dependency versions
are checked.

The original inverse-affine, spacing and cell-radius bytes were not saved.
The diagnostic therefore records **newly computed** derived frame values using
the exact original source and NumPy/SciPy versions. It does not claim that those
values were read back from the original cache or prove equality to unsaved bytes.
The strict original-counter reproduction gate is additional empirical evidence
for this reconstruction, rather than a replacement for that limitation.

Each size record retains the complete unfiltered cover's shape, int64 dtype,
array byte count, result-byte hash and exact query bytes. The return value must be
an immutable contiguous N-by-3 array, whose payload is 24 times the number of
cells. Saved active contacts are tissue-filtered and shaft covers were not saved;
neither is used as a payload-size surrogate. Full cover arrays are released after
their size/hash record is written. Empty covers have zero payload but still
occupy an entry when admitted.

Before interpreting any other capacity, integer-only LRU replay must reproduce
every cold and warm counter, every counter increment, retained entries and
retained payload bytes from the original 32 MiB / 16,384-entry experiment. The
replay uses the frozen cache's admission order: zero-capacity and oversized
queries bypass without evicting existing entries. A failed match retains its
comparison and failed authority, with no eligible capacity result. Only a passed
gate permits the declared 32, 48, 64, 96 and 128 MiB comparisons at the same entry
limit. Those are hypothetical hit counts, not measured speedups or recommended
defaults. Raw array payload excludes keys, Python objects and bookkeeping; the
process peak RSS is reported separately.

The completed original run must have a zero worker exit code and no timeout or
kill, as well as matching source, worker, result and launcher authorities. The
payload, derived-frame and baseline-comparison file hashes are retained when
each file is written, then checked again before a completed summary can be
published. An altered comparison cannot be relabeled as a passed experiment.

The prospective limits are 60 seconds, 512 MiB cumulative process peak RSS,
8,812 pure-cover calls, 22,364 ordered queries and 32 MiB decompressed input.
Worker checks are cooperative and occur around each call; an atomic call may
overshoot before the next check. A future release must use a fresh interpreter,
an immutable source archive and a separately recorded external 60-second timeout.
Timing includes import, frame preparation, sizing and metadata replay. The result
separates pure-cover compute time from worker time before summary publication;
external wall time includes the complete process. A cumulative RSS value from a
previous in-process workload is not a new measurement, so execution must use a
fresh process.

Without `--execute`, the script writes only the declaration and does not read
bound inputs or import numerical geometry. Existing outputs are always refused.
A cooperative stop or cancellation retains completed size rows and attempted /
completed counts in a failed status. A hard external kill may leave an incomplete
gzip stream and requires an external failure receipt; it cannot create completed
authority. No failed output is overwritten or silently retried.

The prospective manifest is
`manifests/experiments/native-cache-payload-v1.json`. The parent must commit the
script, declaration and synthetic review evidence before authorizing any saved
public-query sizing. Its reviewed scope is a frozen pure-cover diagnostic; it
does not certify a new planning model or expose final or stress evaluation worlds.
