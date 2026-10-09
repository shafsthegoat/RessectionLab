# Header inventory promotion delta

**GO for exact-source promotion.** Reviewed patch `build/remind-qc-adapter-preparation-v1/promote-header-inventory.patch`, SHA-256 `24171962f56df0f82cd14eea6e7b7daca2194175b7100a3a52b459d70a39a9a0`.

The patch exactly reproduces the staged files:

- `src/resectionlab/remind_header_inventory.py`: `cee078ff4e0822a6c613aaa0ec51aa62bc5b4c3c5095169682c44d9cb5b58271`
- `tests/test_remind_header_inventory.py`: `4b925332dca844044ddde777d604cfc59c1484d8e661336eeb86f3d2f32600a9`

Independent AST comparison confirms all nine retained functions/classes are unchanged: `Refusal`, `require`, `sha`, `load_module`, `value`, `integer`, `inspect_headers`, `BudgetReader`, and `read_verified_header`. Shared source hashes, allowlists, and constants are unchanged. Removed functions are exclusively `load_bound_context`, `pilot_plan`, and the preparation CLI `main`. The new root expression correctly resolves the repository from `src/resectionlab/`; no ignored handoff dependency remains. Optional numerical/DICOM imports remain inside execution paths, and no patient-header CLI is introduced.

Test changes are limited to package import, explicit skips when optional dependencies are absent, and generated temporary files under ignored `build/`. The fixture and existing assertions are unchanged. `git apply --check` succeeds; stage-only helper symlinks are absent from the patch.

No new broad audit or patient read was performed. The prior independent candidate review remains applicable. Root will apply and run the six promoted tests using the established IDC runtime before commit. A separately reviewed bounded launcher is still required for real headers.
