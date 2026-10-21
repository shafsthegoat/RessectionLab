# Exact full-case vascular verification

Source: `testcases/cases/meningioma-sub-06-standard-v2.ressectionlab`; SHA-256 `ff5a49b8706a8ccda08d8c3e82dad3e079f85dea98cb6bd40783d2262eec9838`.

Final receipts use `load_environment`, exact saved mapping/profile and strict source-bound protection. They prove deterministic software mechanisms, not clinical validation or patient blood loss. The legal vessel route is peripheral and separate from the tumor approach.

- `integrated-report.json`: 16 common skull/access/removal actions followed by 28 rescue actions; phase metrics and public preview/commit receipts.
- `patient-integrated-vascular-rescue.ressectionlab`: full source-preserving native workspace with that 44-action branch; reopen proof in `integrated-reopen.json`.
- `vascular-report.json`: initially intact control, force-free actual contact, injury, untreated bleeding, sealing, field clearance and conservation.
- `patient-untreated-vascular.ressectionlab` and `patient-vascular-rescue.ressectionlab`: standalone exact-scene untreated/treated native branches. `rescue-reopen.json` verifies the treated branch in a fresh process.
- `mechanical-contact.json`: low cumulative contact dose remains uninjured, then increasing force exposure produces progressive damage without a geometric scissors transection.
- `tool-family-inventory.json`: fifteen existing synthetic-fixture public tool interaction checks; explicitly distinguishes missing patient membrane, attachment and mapping evidence.
- `targeted-tests.log`: 69 existing vessel, integration and surgical-tool tests passed. `tool-family-fixture.log`: fifteen tool-family fixture cases passed.
- `check_vascular.py`, `check_mechanical_contact.py`, `run_integrated.py`, `reopen_exact.py`, `make_tool_inventory.py`: reproducible local generators/checkers.
- `wrong-adapter-diagnostic/README.md`: excluded first adapter error and lost initial receipt provenance. No initial wrong-adapter result is counted in final proof.

Production action plan: `src/resectionlab/full_case_vascular.py`. It reads the current engine state to generate hand cleanup, checks the exact scene and initially uninjured vessels, and returns public actions without changing the engine.
