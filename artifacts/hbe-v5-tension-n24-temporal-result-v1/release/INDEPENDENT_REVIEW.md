# Independent row-11 derivation source review

Verdict: **GO for one source-only derivation at HEAD `47fa6d225cfe6fa150aa2c76f8b355f60e03a616`; no native execution admitted by this review.**

Reviewed ignored source `build/hbe-v5-ordinal11-pending-release-v1/derive_pending.py` SHA-256 `77e0a2c6be8f782486d13e779de541011e1da3e09e629c639336c808c9847e5d` and generated tests SHA-256 `453af530e937cfdeb79aff980ea0788d523f7ff76e78f67c87b1eee5d54da0be`. The reused row-10 helper SHA-256 is exactly `985c300269ad03d3748735291ebe00b78a0b2dc47e52641a27c44a4d8c97d2a0`. Three independent tiny tests passed in 0.001 seconds; the helper itself was not executed.

The earlier defect—executing mutable ignored row-10 helper code before byte authentication—is closed. The loader rejects symlinked or oversized helper files, checks pre/post file identity, authenticates the exact SHA-256, and only then compiles and executes those bytes. The source-only derivation binds actual row-10 native and sidecar receipts, exact inner/outer releases, independent metadata and three prior extension charges through the reviewed ancestry verifier. It retains the failed original N12 receipt, the 20 frozen plus 9 extension source closure, and the frozen N24:S120 source-deck/mesh/adaptation/caps path. Its pending release files and native/sidecar target paths are one-use, and stdout is compact metadata.

Remaining gates: independently inspect the derived inner and outer bytes against the source, deck, schedule, cap and ancestry bindings; require the live host policy and the launcher's full predecessor/source validation before a separately authorized bounded native call. The derivation supplies no physical validation or clinical evidence.

## Exact pending-byte follow-up

The one source-only derivation produced inner release SHA-256 `7f1211088385245676ece1c880860afe69d29654c809cf8a4c95c1a3adcb2711`, outer envelope SHA-256 `2fd42ed10c9e7d300cb6ce1d69e307107e9dfa400f374921dd74dc709169f10a`, and derivation metadata SHA-256 `702c238a2014b2c3afa3ae058357f06532247bdaa404afe1cc6aa12cfa4b03b2`. HEAD remained `47fa6d225cfe6fa150aa2c76f8b355f60e03a616` throughout this read-only audit.

All 20 old and nine extension source bindings match both working bytes and their HEAD Git blobs. The 2,458,144-byte source deck and 3,108,096-byte native mesh match their declared SHA-256s. Relative to the exact row-10 inner release, only ordinal, run ID, source commit, adapted deck, adapter receipt, output directory, frozen caps, and appended prior receipt differ. The original N12 failed receipt remains at ordinal 1; the new final prior receipt matches the actual row-10 native receipt. The adapter receipt specifies N24 tension, 120 steps, 121 frames, and `lower_half_reconstructed`; its source and adapted hashes match the inner release. Inner caps equal `remaining.caps(11)`, outer policy equals `runtime.policy(row11.SPEC)`, and the row-10 ancestry descriptor equals the tracked constant. The derivation records prior charges in exact order 8, 9, 10 with 3 MiB reserved sidecar bytes. Both row-11 attempt directories were absent.

**GO for root's one-use launcher only if the fresh host gate and complete launcher preflight pass.** No solver or predecessor bulk replay was run by this independent review; physical validation remains null.
