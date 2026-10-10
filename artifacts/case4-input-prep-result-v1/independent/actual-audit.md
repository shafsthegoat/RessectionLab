# Independent audit: Case4 DEVELOPMENT input preparation

**Verdict:** PASS for the one released, scan-derived input-preparation slice only. It produced a finite saved two-channel tensor and complete matching receipts under the prospective resource guard. No trained-network forward, surgical planning, evaluation, or clinical validation occurred. The Case4 mask's inferior omission remains unreviewed; model-training overlap is unknown.

## Fixed source and receipt chain

- The actual contract SHA-256 is `3e856902bd477e3d7072522d86532fe585357f66d1f1fde4b735124c735dc664`; the launch, worker result, and supervision bind that value. I rehashed its seven runner files, five tracked preparation-source files, and isolated runtime lock: all 13 bytes match the contract. The worker source is `76b33835228d40ff6c5808be24219d9e0e82d95146ca0d4041db2058ec935e3a`; the supervisor is `0dc5abd229a219564a2364aa4eeeeaa579607e2fd255014dde87ce3416f94514`.
- Canonical metadata-only verification of five pinned Case4 manifest/preparation/support receipts passed. The saved input receipt's five evidence hashes and its T1c, FLAIR, support-map, plans, and dataset hashes exactly match those bindings. This audit did not reopen any source patient image, label, or checkpoint. The source worker had verified their bytes during the released run; I independently verified the saved metadata chain.
- The input receipt SHA-256 is `63269eec6d8772272d06c44b147c149a76f1ecf0b41ee0c88b05bb8463e751a0`; `result.json` binds it, the input SHA pair, and geometry SHA `1b372a9accbf1cf9060cd15b36e16d969a553725b1caf3927fbd87576ff018bb`. Geometry records atlas RAS coded sform, atlas dimensions `[240,240,155]`, and preprocessed ZYX patch start `[-9,14,6]`. The negative start denotes padded patch extent in the preprocessing design; it is not independent anatomical QC.

## Independent saved tensor check

I opened only `case4/input/input.npy` as a read-only NumPy memory map with pickle disabled and streamed bounded slabs; no model or checkpoint was loaded. The tensor is C-contiguous FP32, shape `[1,2,128,128,128]`, 16,777,344 NPY bytes, contains nonzero signal, and has zero nonfinite values. Its full NPY SHA-256 `d95383a2c60690f7c4ccbc7f73b37a210b9f6cefef93bf9d3516cc7430ec3f3f` and raw C-order SHA-256 `8d1c0ee902c365056c8b089b1c4c9d0efdae184941397ab9731eeffde7274a17` both equal the receipt and worker result. The bounded audit took 0.024 seconds, observed 66,535,424-byte peak auditor RSS below its 256 MiB observed cap, and used a 30-second alarm. A hard address-space cap was unavailable; no hard-cap claim is made. Machine-readable metadata is `case4-input-prep-saved-array-audit.json`.

## Guard and process outcome

`supervision.json` SHA-256 `cea9644f04a8304e38e8eb5ae4d8336ee8603add556556c4c379cceae950f5c0` reports `accepted_for_feasibility=true`, worker exit 0, elapsed 3.62022125 seconds and sampled peak descendant-aware RSS 1,055,850,496 bytes under the 3 GiB/120-second caps. Seven preflight and 63 fast run samples remained kernel mask 1; run available memory ranged 70–75%, with zero swap-used growth and maximum sample-start gap 0.0603 seconds. Eight slow inventories had zero blocking overlaps and each matched live Tracto PID 92434 to the exact allowlist identity under its 512 MiB cap. Post-run kernel mask was 1 with 71% available. There was no watchdog, monitor, post-guard, or slow-inventory error.

The clean finalizer (`finalization.json` SHA-256 `4a4c5d0fba3a0958517e2e5892f53614e812e32c988b5a166575b88291920a95`) records exit 0, no residual group/detached PIDs, and no cleanup errors. A subsequent PID check found no worker group PID 49467, while the exact Tracto downloader PID 92434 remained alive. Only the expected one input NPY and eight metadata/log files exist in this output directory; there are no logits.

## Scope of next decision

This establishes a software handoff for one unreviewed Case4 DEVELOPMENT scan input. It does **not** establish modality registration anatomy, segmentation accuracy, training independence, or route quality. A separate exact patient-forward release and anatomical QC remain prerequisites before any model or planning claim. Keep the NPY and all patient payloads local and out of Git; portable evidence may include only this audit and redacted metadata receipts.
