# Independent Gate A CPU negative audit

This read-only audit covers the frozen, generated-only GlioMODA Gate A CPU control in `build/limited-input-guard-design/gliomoda-mps-gate-a-v1/`. It is not patient inference, segmentation validation, or an MPS result.

## Static contract and provenance

- Frozen worker SHA-256 `27f64a3f681b63dfccfde0fbf54be3f897e134fd716edb41bc57d49328f325d3`; supervisor `2e44144bb3049cef476d5e972b9b6953fc4017677ec7053e059968c7cc591ff9`; comparator `4c3e389194c3e48323cd9846d5d919d9726d2482e89482dc9710e82cbf1b2a1e`; contract `06c3951e6628d63db765c385ea8b8066e8080511916ae9a50b89c3ed80f887e9`; generated input `.npy` `6a7435f18ebd1e95ce8561e07481f554bfd562f5564bbaaee7da0069123d30b7`. These matched the independently approved version after the run.
- The pinned checkpoint SHA-256 remained `0e29f882310fe8cb076d6cadb982067ef53c6a32231f40ae17d9c173aa4307b3`; plans `e85abbe6a41f4e5e0164d53dd3a297baf10e06cd5f05c1d46a22d2d869d64ad5`; dataset `3a7c7c1fd5eb420243a25496bae2c2fb9f210d7910363b9361aaebc69e615a3f`.

## Observed sequence

- The first 30-second host preflight deferred before launching a child: its system-wide free-memory samples included 44%, below the prospective 45% minimum. `PREFLIGHT-DEFERRED-1.md` records the operator's summary. The first preflight's raw tool output was not saved as a machine-readable file, so the exact sample series is not independently reproducible from the artifact alone.
- The next saved preflight passed: seven samples reported 50-51% free, no swap-used growth from 7279.69 MiB, and no matching large compute process. Its final baseline reported 5,023,629,312 compressor-occupied bytes and 51% free.
- The CPU worker logged `before_generated_forward` at 1.525 seconds after successful scoped weights-only load, architecture construction, alias checks, strict state copy, and cleanup. No `after_synchronized_forward` phase, logits file, or result receipt exists.
- The supervisor sampled 5,959,303,168 compressor-occupied bytes at 1.874 seconds. Increase from its saved baseline was **935,673,856 bytes = 892.33 MiB**, above the prospectively specified 256 MiB threshold (268,435,456 bytes). It killed the child process group: exit code -9, reason `host_compressor_growth_over_256_mib`, 1.898 seconds elapsed. Free was 45% at the stop sample; swap used remained 7279.69 MiB. Sampled peak child-group RSS was 1,177,968 KiB, below the independent 3,145,728 KiB RSS cap.
- Read-only `ps` group check returned no processes for PGID 59598. No MPS output directory exists, so no MPS worker was launched and no parity comparison can be made.

## Interpretation

This is a **prospective host-pressure guard stop during the generated CPU forward**, not a numerical parity failure or a model prediction. Compressor occupancy is system-wide and cannot be attributed wholly to this worker; its abrupt rise coincided with the forward. The sampled RSS stayed below its cap. The existing negative receipt and both preflight outcomes should remain unchanged. Do not retry this version or change thresholds after observing the outcome. Any new memory investigation should be a separately versioned prospective diagnostic with the same distinction between process RSS, host compression, and MPS allocator usage.
