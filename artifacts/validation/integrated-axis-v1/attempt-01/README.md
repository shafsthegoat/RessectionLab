# Immutable integrated check — retained failed attempt

Exact source `100865f037b37708f4f0d0d5a48950de22ba4e5f` passed 914 tests and failed one analytic orchestration fixture in 171.74 seconds (four existing DIPY warnings). All 2,601 baseline files, including the copied public-case bundle, stayed byte-identical. The launcher took 173.15 seconds.

The failure is `test_all_arm_orchestration_uses_explicit_nonpatient_test_scope`: historical transfer declaration inputs correctly reject the current cohort registry after its prospectively selected PAT16/PAT20 extension. The fixture still used current registry bytes. A separate test-only repair is being reviewed; this result is not replaced by a later successful run.

`before.json` records all archived source and case digests; `receipt.json` binds the Git archive, exact copied case, source manifest, and unmodified full output. This uses the existing local development environment, not a fresh dependency installation. Source imaging is not redistributed here.
