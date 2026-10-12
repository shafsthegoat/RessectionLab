# Focused policy runtime recovery and retained negative result

The guard optimization completes GREEDY's twelve additional movements within the original 120-second planning limit. The latest complete planning phase takes 95.397676 seconds. It preserves the model, reward, catalog, input boundary and per-action certificate checks. Thirty generated controls and actual isolated worker imports pass.

The matched learned comparison is still **incomplete**: frozen TAIL completes ten additional movements and eleven scoring calls before the same 120-second cap. Neither paired plan sealing nor fresh authoritative final replay occurs. A saved complete GREEDY plan is not an independently accepted final endpoint. There is no learned quality gain, held-out improvement, physical validation or completed resection claim.

## What changed and what failed

The first recovery hashes the exact prefix of already authenticated history instead of recomputing full-volume outcome statistics, and validates each page context once. Its GREEDY rollout reaches the 24-action horizon, then result validation fails because the JSON serializer receives immutable packet mappings. Both that failure and the original timing failure remain preserved.

The second recovery normalizes immutable records through the existing `freeze_json`/`thaw_json` contract, persists each completed rollout before later validation, and exercises actual generated frozen packets through two-plan sealing and report writing. Three added controls cover that successful path and durable evidence after a later failure. It does not change scientific decisions or limits. GREEDY completes and is saved; TAIL then reaches its time limit.

## Measured cost and next bottleneck

| Attempt | Result | Native previews | Completed transitions | Frozen policy forwards | Parent seconds |
|---|---|---:|---:|---:|---:|
| Original | GREEDY time cap after eleven suffix moves | 4,733 | 35 | 0 | 170.162175 |
| Recovery 1 | GREEDY reaches horizon; immutable-record serialization fails | 4,850 | 36 | 0 | 136.908503 |
| Recovery 2 | GREEDY complete; TAIL time cap after ten suffix moves | 7,120 | 46 | 11 | 258.085528 |
| All three | No accepted paired endpoint | 16,703 | 117 | 11 | 565.156207 |

All three workers are reaped, sources remain unchanged, and no owned process remains. Two frozen checkpoints are loaded per attempt; no weights are fitted, no feature cache is reconstructed, and no SELECT/EVAL patient is opened. Peak sampled memory across these attempts is 1,253,769,216 bytes. Phase timers overlap and must not be added to parent wall time.

In Recovery 2, TAIL's eleven scoring calls consume 34.4575 seconds, including 2.2837 seconds in feature forwards (the separately instrumented policy-forward timer is 2.2831 seconds). Frozen-parameter checks total about 0.0066 seconds. The remaining 32.1738 seconds is not independently apportioned among catalog authentication, the small head, and surrounding work.

Source inspection identifies a concrete candidate: TAIL performs 32 full catalog validation passes, with 797 candidate checks. Each candidate's `export_action()` and `_certificate()` repeat task/history authentication inside a page that was already authenticated. The next bounded change should authenticate each page once **within each validation pass**, retaining complete packet-field/seal checks, certificate registry identity and digest, nominal reward, both pre/post-scoring passes, and canonical commit validation. No authentication result may be cached across mutations. The speed attribution and expected benefit are hypotheses until profiled and equivalence-checked; another larger time allowance is not the proposed fix.

After runtime is usable, the retained value-head fit exposes a separate model issue: logged future-return targets vary widely while predictions are compressed. A single residual readout from the existing 262 actor features is a prospective test, with the encoder and prior head frozen. Route decisions and full matched outcomes, not fit loss alone, would decide whether to retain it. No such fit has run in this iteration.

## Evidence and limits

[The inventory](INVENTORY.json) binds exact source snapshots, control receipts, failure reports and omitted local gzip evidence. The original integration archive remains unchanged. The recovered controls passed 27/27, followed by 30/30 after the serialization fix; complete supervised suites and isolated imports took 7.077526 and 7.183184 seconds respectively.

- Recovery 2 source index: `9112bf00953b168bf36ba409538dcbb10a0ac018272e4d65af921a8f09c342a6`.
- Terminal report: `079cfc74ae0499430b9a93438a94d7f3154a7559a972a40eee71f17fb8ab1af1`.
- Parent/cleanup receipt: `abad11ff2dbf4f9655b9ea11d6949017abdccfd7350996286e776bf14928284d`.
- [Independent saved-evidence audit](snapshot/independent-result2/failure-audit-03-corrected.json): 6,978 checks pass in 0.3813 seconds; receipt `f23479ad80689b6e87bee333ebcfef8c4613d311137cde8b5f6bed9781f84a8f`. No final replay is implied by scalar checks.

The first audit refused a changed original-context hash. Inspection found that each source release legitimately changes its admission identity. The corrected independent reader authenticates both context hashes against their own saved admissions and matches every remaining scientific context field. All earlier 23 packets, 23 native histories and eleven complete decision records match exactly. Both audit attempts remain preserved. The runner's post-seal parity checker has the same latent identity defect and still needs that correction before another full comparison; it was not reached in this time-capped attempt.

The archived source and parent command reproduce the bounded attempt after restoring required local inputs to their indexed paths; attempt output directories must be fresh. Bulk traces, patient arrays and checkpoints remain out of Git. This is annotation-assisted simulated planning on one exposed TRAIN patient; repeated records are not independent patients or observed surgery.

The [final dated storage record](STORAGE.json) records 1,817,538,560 bytes of additional allocated growth (1.693 GiB) and 350,382,301,184 bytes free (326.319 GiB), after the audit/archive but before final documentation and Git writes. No new dataset, dependency or weights were acquired. The shared stage closes at 7.121 MiB within its 8 MiB reservation; overall 10 GiB/50 GiB limits remain satisfied.
