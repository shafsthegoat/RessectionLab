# HBE v5 N8 first-canary source preparation

**Disposition:** prepared source-only; native execution **HOLD**. The frozen row is `compression:N8:S60:reference` from HBE_01_03. No FEBio call, measured CSV, patient file, fit or physical comparison was used.

The old full-native N8 source deck SHA256 is `b0191042df6fccc4c4dc317bb9145f0a077880a7e5589cd471b9ac93452a2506`; the bound mesh SHA256 is `3a6b5bb720bd3998f035abecf15d5c603d46c5ed281ce1d08c4aadec69c1dab9`. The pure adapter generated a new-endpoint Skyline deck SHA256 `6b43b2a7b5acc46862c576416368a41d926e2214f738036f20491dc81f388cff` and Accelerate deck SHA256 `3cf4156b19191918841498490bc448f88a61b656b566d5d6a3d36db817d0a724`. Both exact deck files are in ignored `outputs/mechanics/hbe-v5-n8-canary-preparation-v1/`; the tracked manifest SHA256 is `2d74b090e9cef22a74d4daaf74309a26391f66f12bd7f872c6615c72e974f0b7`.

The source-only verifier checked the 1,045-node/768-hex8 topology and 630 prescribed-displacement BCs, the full-native top-z factor `1.0`, all 61 generated time/load points ending at `-0.00073726 m`, exact old/new deck delta and backend transform, and the existing hash-bound repaired Accelerate runtime profile with its earlier controls. The closed manifest has `release: null` and `native_execution: false`; the execution gate raises unconditionally. No other v5 source deck or specimen mesh was opened by this preparation.

Validation: `python -B -m pytest -q tests/test_mechanics_hbe_v5_n8_canary.py tests/test_mechanics_hbe_v5_source_bindings.py tests/test_mechanics_hbe_branch_calibration_v5.py` returned **57 passed**. Local tests exercised changed endpoint/schedule/source/mesh/runtime/deck/BC/release fields and prohibited subprocess launch.

This verifies deck preparation only. It does not establish solver convergence, complete saved-frame readout, HBE force agreement, tissue retraction fidelity, patient-specific properties or clinical utility. The independent native gate review requires a separately bounded one-shot supervisor and release, complete saved-stream negative controls and independent replay before any new-endpoint N8 solve. Twelve-run comparison and physical fitting remain later gates.

The old source/mesh, repaired runtime/control files and generated decks are ignored local dependencies. A clean checkout without their exact hash-bound bytes cannot replay this preparation and must remain on HOLD.
