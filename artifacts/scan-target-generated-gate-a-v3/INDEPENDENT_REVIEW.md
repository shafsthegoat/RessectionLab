# Independent audit: Gate A v3 generated CPU control

**Decision: HOLD paired MPS comparison.** The sole reviewed 64³ generated-input CPU attempt did not produce logits. Its controller stopped the worker during the forward pass after macOS reported warning-level memory pressure. `accepted_for_pair=false`; no MPS process was launched. This is a host-resource negative, not a numerical or clinical model result.

## Receipt and trace

- The 30-second host preflight was accepted. Seven direct pressure readings were mask `1` (normal); available memory readings were 48–53%; swap used stayed at 7,574,585,344 bytes. The exact ReMIND acquisition allowlist matched, so that transfer was not interrupted.
- The supervisor recorded 37 direct fast samples over 2.015 seconds. Samples 0–35 had pressure mask `1`; sample 36 at elapsed 2.015055 seconds had mask `2`, available 41%, and process-group resident memory 1,428,946,944 bytes. The post-child direct host sample also had mask `2`. A later independent read-only `sysctl` query returned mask `1`, consistent with a transient warning and recovery. Mask `2` is Apple's warning/urgent memory-pressure signal; the sampled warning is not inferred from compressor or pageout counters.
- The final worker marker was `before_generated_forward` at approximately 1.717 seconds. No after-forward marker, logits, result file, comparison, or decoded labels exist. The worker ended with signal 9; `watchdog_reason` and `post_guard_reason` both equal `kernel_pressure_warning_or_critical`. Elapsed time was 2.063718 seconds. Peak sampled process-group resident memory was 1.331 GiB, below the 3 GiB process cap; swap used did not grow in the fast series.
- `monitor_error=null`, `post_error=null`, `slow_inventory_errors=[]`, and there were no detached descendants. Slow inventory was fresh (last fast-sample age 0.491 seconds), and no other blocking project process was recorded. A subsequent process listing found no members of process group 69178. These checks support a real, directly sampled host-pressure transition rather than the v2 subprocess-timeout failure. They do not establish that the model alone caused whole-host pressure; unrecorded concurrent host activity is still possible.

## Frozen inputs and source

Post-run SHA-256 checks matched the reviewed bytes:

| Item | SHA-256 |
|---|---|
| `darwin_fast_sampler.py` | `2ff8f3e4d30ed0a8976fb01da99403ff5b2b2ffd6e750791fe7a954946d0a00f` |
| `slow_inventory.py` | `1b3915453af9eee16094bf47963a7078435a390f112765266328d30160c2eef1` |
| `supervise_gate_a.py` | `5985d7d9aa1ec5ad59a8779a71a2964da61140dcbbc4e45f61442aee1bfb5d0d` |
| `gate_a_worker.py` | `27f64a3f681b63dfccfde0fbf54be3f897e134fd716edb41bc57d49328f325d3` |
| `compare_gate_a.py` | `88b485b824bd1c9bcfe284530b6e7cd9947cbec94a4e3cf52c8d263d7aacb21b` |
| `gate-a-contract.json` | `2a1d7ed500e8fca712808b8ae21b2b946921a11e669785020f0bd71fd57d1270` |
| `acquisition-allowlist.json` | `1151663817c835dc913cd1a8af69eff25fc0f9890b9a63766a13c7f38458b52f` |
| `generated-input.npy` | `6a7435f18ebd1e95ce8561e07481f554bfd562f5564bbaaee7da0069123d30b7` |
| GlioMODA checkpoint | `0e29f882310fe8cb076d6cadb982067ef53c6a32231f40ae17d9c173aa4307b3` |

Evidence: `build/limited-input-guard-design/gliomoda-mps-gate-a-v3/cpu/host-preflight.json`, `supervision.json`, and `worker.log`. Preserve this v3 negative alongside v1/v2. The paired-comparison protocol requires a completed, accepted CPU output before conditional MPS, so MPS remains on hold. Any later experiment needs a separately reviewed protocol; this audit does not authorize a retry, altered pressure threshold, patient inference, or full 128³ inference.
