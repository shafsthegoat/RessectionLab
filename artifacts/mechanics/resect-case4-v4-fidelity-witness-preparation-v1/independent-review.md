# Independent Case4 v4 saved-geometry fidelity-witness source review

Final verdict: **GO to commit the non-executable preparation only** for the reviewed hashes below. This does not authorize opening saved patient geometry or running the worker. No patient arrays or v4 geometry were loaded, and no mesher or solver was run.

## Evidence reviewed

- The frozen manifest declares the rejected v4 source/result identities, exact 1 mm/256-triangle bidirectional replay, unchanged 2 mm gate, 30 mm / 5 mm whole-face-centroid localization rule, one attempt, and resource caps. I checked the five JSON receipt hashes without reading geometry arrays; they match the declaration. The receipt counts and failed directional maxima/upper bounds also match.
- `directed` uses the original grouped barycentric sample order and original chunk cover. The analytical isolated-source control passes, including a 262-triangle case crossing the 256-triangle chunk boundary. `cube_decision` implements max-min capture with lexicographic first tie and both area/capture thresholds. The 1,193,462 query count is 1,178,874 lattice + 14,568 probes + 20 nearest-locator checks.
- An isolated source copy (witness, original diagnostic, original mesh helper, manifest, test; no patient inputs) passed `6/6` focused tests with `.venv/bin/python -B -m pytest -q -p no:cacheprovider`.
- Seven-member `CLOSURE` covers the witness, manifest, numerical helper, original mesh helper, output guard, runtime, and runtime declaration. `guard.supervised(..., spec=spec)` matches the guard signature. Source has no Gmsh, solver, B/V, during-US, MRI or mask reads. Saved geometry is only decoded in the worker after release checks. Patient coordinates would appear only in private ignored outputs.

## Issues found and repaired

1. Direct `worker` CLI mode can run with a valid root release and a manually created attempt directory without entering `run` or `guard.supervised`. It thus bypasses the 120-second outer cap, sampled 2-GiB process-group RSS limit and aggregate output watchdog. The worker has a cooperative deadline and hard per-file size cap, but those are not equivalent. Require a parent-supervisor-only child path or equivalent proof of supervision before reading saved geometry.
2. `run` creates the attempt directory and then can throw before `acceptance.json` is written. A no-patient synthetic monkeypatch of `module` confirmed `{orphan_attempt_dir: True, terminal_acceptance: False}`. Capture post-creation exceptions into a terminal failed/incomplete acceptance receipt, preserving the fresh attempt directory and no-retry rule.

Both findings were repaired in the final source. The worker now requires a parent-generated token bound to the release hash and canonical attempt path before saved-input access. The parent now writes a failed/incomplete terminal acceptance after a post-mkdir setup/supervision exception. The source and declaration remain non-executable preparation until a separate exact-commit archive and root release. Token enforcement is an accidental-bypass safeguard, not protection against deliberate local code or environment modification.

## Repair recheck and final controls

The writer added a parent-generated 256-bit token bound to the release hash and canonical attempt path, checked by `worker` before specification or saved-geometry access. `run` now catches post-creation failures and writes `acceptance.json`. I reproduced the earlier synthetic orphan failure after the fix and found a terminal failed/incomplete acceptance. The refreshed isolated-source suite passes `9/9`. Direct worker invocation without token fails before any saved-input read. Synthetic token checks reject absent/wrong token, changed release and mismatched attempt path; the valid token/context is accepted. This is accidental-bypass protection, not a security boundary against a local operator who can modify code or capture environment variables.

An independent 100-case randomized synthetic comparison against a brute-force max-min grid implementation agrees **when candidate cubes are restricted to those containing at least one witness face centroid in either direction**. That eligibility rule is now explicit in the frozen manifest and document, with a disjoint-set zero-score tie control. The final isolated-source suite passes `10/10` controls. A valid token/context is accepted; absent, wrong, changed-release, and changed-attempt variants are rejected.

Final reviewed SHA-256: source `7dfb31ab6826aebf7e0b20528157b97ee1b4164b16afee3dbe4a0877d2a03d0d`, manifest `3a0790914b33159a3a09c275716f04f6deec6c9bd19de497b1b56dd22798a4d4`, tests `83e2d4b08d3f12e3503480702a2f0e93dee350ef184b6d2c3beabb6ad593a432`, document `677e9b0e5a48bca06799e4b3f0fe8f1333821b670e4c574e29e2a35f532571bf`.
