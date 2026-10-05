# Independent saved-artifact audit

The saved development comparison reconciles. Expanded proposals removed **1,113 mm³ target + 62 mm³ normal tissue**, incurred **177 mm³ cumulative partial normal contact**, and scored **1098.77**. The fixed inventory removed **249 + 17 mm³**, incurred **32 mm³ partial normal contact**, and scored **245.24**. The increased removal follows a newly declared proposal model, discovered on the same patient, rather than a policy improvement or independent generalization result.

`audit.py` verifies the frozen numerical runtime, declaration/model identity, unchanged experiment script, and all three compressed histories against their raw-byte and candidate hashes. It reconstructs source-cell target/normal accounting, unique partial-contact charges, microstep containment of all removed-cell corners, original greedy choices, dynamic endpoints and proposal omissions without executing a simulator, training, or repeating expensive geometry. The fixed two-stroke entries, targets and tools match the earlier study's independently certified source-cell history and engine identity; this launcher did not repeat that audit.

Both patient inventories used the same tools, access, reward and native engine, with limits of three cuts, 128 previews and 45 search seconds. Fixed search checked 24 previews and completed two cuts. Expanded search checked all 66 generated rays across rounds of 26, 22 and 18; empty residual columns explain every omission. Its selected three-stroke history has an independent certificate supporting all 1,175 removed tissue cells with zero unsupported cells. These certificates apply to committed histories; unchosen previews were not individually re-certified by this audit.

The barrier negative result is retained: 64 forbidden-collision rejections and 64 accepted proximal fallbacks exhaust 128 previews after two committed cuts. Its third round checks only 12 of 26 rays, leaving **14 untested rays**. No cell on or beyond the barrier was removed. It does not establish exhausted feasible search. The solid phantom's 378 target + 150 normal cells and the barrier's 36 target + 60 normal cells also reconcile with their certificates and scores.

Review of the frozen independent checker confirms that swept-shaft collision is tested against tissue remaining **before** each microstep's new removal, preventing use of future clearance. All compared search times remained below 45 seconds. Shared setup, engine preparation and independent audit time are separately reported; the previous study's beam-search timing is not a matched comparator. The failed first harness attempt stopped on a metadata-key typo before comparisons and remains recorded.

No optimization, selection, final or stress panel was opened by this probe; factory construction may initialize a deterministic nominal world object. No policy was trained. Patient anatomy remains an estimated envelope with a hypothetical opening and unvalidated functional evidence/mechanics. The expanded result still leaves 40,806 mm³ (97.345%) of annotated target.

Two prototype limitations must be corrected before reuse, without rewriting the executed script:

- The intended source-axis/integer-origin restrictions use default relative tolerances. For example, `np.isclose(cos(0.004), 1, atol=1e-8)` accepts about 0.229° of tilt, and `np.allclose([152.001], [152], atol=1e-7)` accepts a 0.001-voxel offset. Reusable validation needs `rtol=0` and near-boundary adversaries. The saved inputs are exactly aligned and integral.
- Out-of-image columns are silently skipped. All thirteen offsets are in bounds in these runs, but reusable output should give explicit exclusion reasons and denominators.

Run the lightweight audit from the repository root with `.venv/bin/python artifacts/native-frontier-expansion-v1/independent-audit/audit.py`. It reads compressed histories so it works with the versioned clean-clone files. The accompanying `report.json` is its machine-readable receipt. Production files and original results were not modified.

## Reproducibility correction — October 4, 2026

The preceding clean-clone claim is superseded. The original audit reads its
three expanded-candidate gzip files correctly, but it reads the earlier fixed
SEARCH replay only from
`artifacts/learning/procedural-native-to-ucsf-v2/comparison/development-geometry-e8a2cc46-0d88-4fc3-93be-176711ee19aa/native-history-replay.json`.
Git tracks that replay as `.json.gz`; the raw JSON is ignored. This audit also
requires the ignored `artifacts/learning/procedural-native-to-ucsf-v2/frozen-source/`
runtime and `source-case.ressectionlab` bundle. A fresh clone alone does not
supply those inputs.

The [current reconstruction guide](../../../docs/artifact-reproducibility.md)
identifies recorded source commits, checksum requirements and the separate
checkpoint prerequisites of the earlier learning audit. Its report generator
already supports tracked gzip evidence. This correction comes from static
inspection; the audit was not rerun, and its original `audit.py`, `report.json`
and experiment records remain unchanged.
