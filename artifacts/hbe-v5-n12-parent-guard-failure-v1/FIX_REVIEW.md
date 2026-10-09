# Independent review: zero-byte HBE v5 readout console

Decision: **GO for the narrow inspector/rehash/source fix.** This does not promote the original N12 failed receipt, authorize another native call, or release a saved-attempt supplement. Read-only review; no FEBio run, patient record, measured response, or tracked edit.

Frozen reviewed SHA256:
- Runner `scripts/mechanics_hbe_v5_remaining_one_shot.py`: `c6cc54e91340394f432aa83623b9dd60c8b0a1b981670f16da9bfb7099807b9b`
- Tests `tests/test_mechanics_hbe_v5_remaining_one_shot.py`: `8a960f69f24e4ccfa8b13db93e0061a0e94c423748035a3ddfb41f4a64afea49`
- Preparation manifest unchanged: `42a66a35bb9aaa70d994189b4f7f7459510c360d3c6c3c1e6f407b6083edbbda`
- Documentation: `3d01caec88f55b29e5804f911601782873328f27cf7938f9ab2de2bc47e7f0ff`

The change admits zero bytes only for `readout-console.txt` in saved-readout inspection, later output rehash, and predecessor closed-output accounting. `_regular_output_binding` defaults to rejecting empty files; native inspection uses the default. Tests mutate `specimen.feb`, `solver.log`, `readout-work-order.json`, and `readout.json` to empty and confirm rejection. The empty console retains exact path, size zero and SHA256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`. Worker exit and numerical readout gates remain required by `execute`; predecessor admission still requires a passing receipt and both completed stages.

I reran 81 focused/adjacent tests successfully. I also called the corrected `inspect_readout` read-only on the actual saved N12 attempt: all eight non-receipt output bindings checked, 61-frame numerical summary returned true, and the original receipt remained `failed_or_incomplete` with failure `Native output absent or above bound`. Both native and readout stage receipts recorded completion within caps. That demonstrates the original parent failure was the zero-byte-console policy, not a solver or stream failure; it does **not** retrospectively certify skipped post-run source/runtime checks.

I reviewed `build/hbe-v5-n12-saved-attempt-recovery-contract/CONTRACT.md` as a **draft dependency**, not an implemented release. It correctly requires exact original inventory/release/source/runtime binding, one new replay-only source commit and one supervised saved-output replay, unchanged original failed receipt, exact readout equality and independent supplement review before any row-two release. Generic predecessor validation still rejects the N12 failed receipt. The contract's numerical result would remain specimen software evidence only, not measured-force, patient, physical, or clinical validation.
