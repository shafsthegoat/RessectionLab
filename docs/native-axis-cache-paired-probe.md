# Prospective fixed-trace public cache probe

Status: prepared, not executed. The [declaration](../manifests/experiments/native-axis-capsule-cache-v1.json)
and `scripts/probe_public_native_capsule_cache.py` require review and a separate
execution release. This opt-in probe does not integrate the cache into the app,
production adapter, learner or another pilot.

The source is the same previously studied UCSF-PDGM-0004 development bundle and
V2 direct native configuration. The exact three V2 greedy action IDs, tools,
access, reward, source cells, complete axis inventory rule and first optimization
seed are pinned. No new action choice, optimizer, gradients, selection panel,
final world or stress world is used. Existing anatomical and clinical limits
remain unchanged.

One common uncached template constructs the complete initial inventory; that
cost is recorded separately. Four reset-and-replay phases then run on the same
simulator: reference before, cached cold, cached warm, reference after. The cache
is constructed empty only after the first reference phase. Its 32 MiB array
payload and 16,384-entry limits remain fixed. Cold starts with an empty cache and
builds the full initial inventory and all postcut inventories; warm reuses only pure covers
retained during cold. Injection is restored before the final reference phase.

Each phase must execute three cuts and four complete inventories, with exactly
26, 22, 18 and zero native preview attempts respectively. Every accepted native
certificate, ordered ledger, proposal/action ID, action/state observation,
committed history, reward, five tissue/contact masks and cavity ancestry is
compared exactly. Full scientific traces receive the same deterministic gzip
export in all phases. Only explicitly enumerated timing paths are excluded;
new fields remain comparable. The original V2 model, action IDs, selected native
history, reward and full ordered inventory ledgers must also match.

All four histories undergo separate independent native audits after injection
has been restored. A new runtime hash records the opt-in helper and complete
frozen numerical source closure; it is distinct from the unchanged scientific
model hash and historical V2 runtime hash.

Preparation, template construction, cache construction, reset, transitions,
verification, export, native previews, audits and total wall time are reported.
A restored, per-instance measurement wrapper counts complete proposer calls,
including the pre-reset verification that the adapter's resettable counters
otherwise omit. The same wrapper is used in all four phases. Mask/certificate
capture occurs immediately after the existing guarded reset or step returns;
it adds no extra action-selection call. Cache counters and array payload are
separate from process cumulative peak RSS and key/bookkeeping overhead.

The worker has a 590-second cooperative wall budget and a 6 GiB cooperative peak
RSS limit. The parent requests termination at 590 seconds and kills after at
most ten further seconds. The memory limit is not an OS reservation. Source
copy/import startup and parent/export overhead are identified separately; normal
OS activity remains uncontrolled even in a coordinated quiet window.

The default CLI writes only a declaration. `--execute` creates a fresh isolated
worker and fresh output; existing output is never overwritten. Failed work,
partial inventory receipts and committed transitions are retained, with no
automatic retry or parameter change. Success requires both worker and launcher
completion, a completed result payload, the bound result hash and matching
worker/result runtime identities. Malformed result receipts are recorded as
launcher failures. Public execution must wait for release.

Any timing conclusion is limited to this paired fixed trace on one development
case. It does not establish whole-learning throughput or clinical performance,
and must not be compared directly with the earlier concurrently scheduled
preflight timings.
