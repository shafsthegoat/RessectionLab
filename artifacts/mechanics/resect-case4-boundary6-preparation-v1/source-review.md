# Case4 boundary-6 v4 independent source review

Scope: exact uncommitted source-only candidate supplied to this reviewer. No
retained Case4 surface, patient image, Gmsh runtime, solver, B/V landmark, or
during-US data was opened. The isolated source export contains no `data/` or
`outputs/` payloads.

## Exact candidate reviewed

| File | SHA-256 |
| --- | --- |
| `manifests/experiments/resect-case4-patient-mesh-boundary6-v4.json` | `0cc7058ce0fbb58a8000bfa69d94dbf2d1f158c91b7a303c7943248eb0ec771a` |
| `scripts/mechanics_patient_mesh_boundary6.py` | `d4ee218653edb0a44e865c39699e3fc6e182a81947c43cc99d2bf2ab030ddf2e` |
| `scripts/mechanics_patient_mesh_boundary6_run.py` | `961075afa4eff04075d90394383b3379eefba29452c14baa60deceb4fdbf42c0` |
| `tests/test_mechanics_patient_mesh_boundary6.py` | `7e3ba587dd20d147e4d4122c43aa90e6800c5ee9cf0549d2ba6e617ef9f082cc` |
| `docs/mechanics-patient-mesh-boundary6.md` | `1d8cbe06b067c5d234dcc0e66d3b5913bac7a8a902c21ede19e28d3adc39aef5` |

## Findings

- **Preparation commit: GO for these exact five files.** The helper constructs
  v4 from the SHA-bound v2 declaration and rejects any additional change.
  Source surface hash, mask hash, native T1 RAS metre frame, surface and volume
  validators, and Gmsh options remain bound. The only sizing change is a 6 mm
  boundary request with the 24 mm interior, 2–24 mm distance transition, and
  curvature off. The requested sizes are not accuracy measurements.
- The shared path charges before one native `generate(3)` call; no generation
  loop or retry appears. The unchanged validators check source and returned
  topology, tet10 midpoint conformity/Jacobian witnesses/mean ratio, convex
  tetrahedral overlap, both surface directions against the estimated source
  envelope at 2 mm, and relative enclosed-volume error at 3%. The output
  count/byte ceilings are geometry diagnostic controls, not solver admission.
- The launcher uses a ten-file exact SHA/commit closure, prior-failure summary
  hashes, fixed context roles, source/runtime checks, an isolated attempt path,
  sampled process-group RSS and disk supervision, 180 seconds, 3 GiB, one
  numerical thread, 32 MiB aggregate output, 8 MiB per file, 32 files, and a
  hard child file-size limit. Earlier v1/v2/v3 failures remain immutable and
  are not reclassified. The manifest explicitly denies solver and clinical
  acceptance.
- **Actual source-only meshing release: NO-GO now.** These five files are not
  yet in the immutable source commit. A separate root release must bind a
  committed ten-file closure and review the exact context paths and hashes,
  source-surface and runtime receipts, archive, and fresh output directory.
  The current repository HEAD cannot satisfy the candidate closure; the
  adversarial release test fails before any attempt output or patient read.
  No run result, geometric fidelity, anatomical accuracy, displacement, or
  force validation can be inferred from this preparation.

## Verification

- `pytest -q -p no:cacheprovider tests/test_mechanics_patient_mesh*.py`:
  **176 passed** in the primary checkout, including v1–v4 controls.
- Exact minimal Git archive of the committed dependencies plus these five
  candidate files, without patient payload: **20 v4 controls passed**.
- Fake-Gmsh fixture checked the exact field values, curvature option, and
  one charged generation. Mutated source/config/gates/caps/release controls
  fail before native output. A constructed authorized release against current
  HEAD rejects missing committed candidate source with exit 128 and creates no
  attempt directory.
- Final five source hashes match the values at review start.

Review limit: this is source/fixture validation. It does not authorize a
patient meshing attempt or a mechanics solve.
