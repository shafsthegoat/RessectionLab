# Native action-coverage experiment

The wider declared proposal inventory increased simulated target removal from 249 to 1,113 mm³ under the same three-cut cap. Normal removal also increased, from 17 to 62 mm³, and partial normal contact from 32 to 177 mm³. The selected expanded history passed the independent full-tool, prior-cavity, contained-cell and connectivity audit.

This is post hoc development on the same previously explored patient simulation. The new candidate rule is an explicit new action model; the completed learning study and its frozen proposal set are unchanged. No policy was trained, and no world partition was opened.

| Measurement | Fixed four-point inventory | Expanded residual-column inventory |
| --- | ---: | ---: |
| Contained target removal | 249 mm³ | 1113 mm³ |
| Contained normal removal | 17 mm³ | 62 mm³ |
| Partial normal contact | 32 mm³ | 177 mm³ |
| Target fraction | 0.594% | 2.655% |
| Remaining target | 41,670 mm³ | 40,806 mm³ |
| Nominal physical reward | 245.24 | 1098.77 |
| Completed cuts | 2 | 3 |
| Successful / all checked previews | 5 / 24 | 66 / 66 |
| Search seconds | 1.008 | 14.951 |
| Engine preparation seconds | 0.344 | 0.347 |

Both use deterministic nominal greedy selection, at most three cuts, 128 previews and 45 search seconds. The expanded run terminated at its action cap; the fixed run had no positive legal remaining proposal after two cuts. This is a candidate-inventory comparison, not an RL improvement or a globally optimal search. The earlier completed beam-search timing uses a different algorithm and must not be treated as a matched timing result here.

## Cause and retained failures

The old inventory ends the central ray at `[-162,85,105]`, although the same annotated source column extends to `[-139,85,105]`. Only two of its four entry/target pairs initially yield legal cutting actions. After the saved two cuts, four tool/target combinations contain no new cells, three fail aperture constraints and one fails shaft clearance. The horizon has one unused action. An early separate preview demonstrated that extending the central ray can legally add 202 mm³ of target and 5 mm³ of normal tissue; that discovery informed this explicitly new development rule.

The expanded generator uses thirteen fixed transverse source-grid offsets and the distal remaining labeled target on each column, with a proximal fallback only after a rejected distal insertion. It respects the existing 6-mm-radius access disk and unchanged tool catalog. All new full-cell removals are paid for along the actual insertion. The selected three strokes use the wide tool with distinct entries and distal targets; their exact coordinates and full native microstep histories are retained.

The solid phantom removed 378 mm³ target and 150 mm³ normal tissue, with 212 mm³ partial normal contact; its audit passed. The barrier phantom produced 64 forbidden-collision rejections among 128 previews, removed only 36 mm³ target and 60 mm³ normal tissue, and retained every cell on and beyond the barrier. Its independent audit passed. It stopped at the preview cap after two committed cuts; this is not a claim that its full feasible frontier was exhausted.

## Costs, provenance and limits

Shared source/simulator setup took 2.626 seconds. The expanded patient audit took 14.750 seconds and independently supported all 1,175 mm³ of claimed tissue removal, with zero unsupported cells. The whole launcher, including both phantoms, both patient inventories, source setup and audits, took 44.353 seconds. The fixed patient sequence exactly reproduced the independently checked completed-study result; its full audit was not repeated in this launcher. Seven additional proposal-rule metamorphic tests passed in 0.07 seconds. All preserved runtime checksums still match after execution.

The first launcher stopped on a metadata-key typo before any comparison; its failure note is retained. The successful attempt's declaration and exact script were saved before comparisons. `report.json` includes every attempted/rejected proposal, physical costs and timings. The patient history was frozen before its independent audit and is versioned as `patient-expanded-candidate.json.gz`; the two phantom histories use the same `.json.gz` format. Uncompressed originals remain unchanged locally. `candidate-compression.json` records raw and compressed byte counts and SHA-256 checksums with byte-identical decompression verification. A clean clone can load each history with `json.load(gzip.open(path, "rt"))`. This lossless packaging changes no declaration, candidate or experiment-script bytes.

Only 2.655% of the annotated target was removed. The remaining 40,806 mm³ and the restricted parallel-column action family prevent any claim of a complete resection plan. The native whole-cell rule remains conservative at oblique or partial-cell boundaries. The source brain envelope is estimated, the opening is hypothetical, tools are generic research geometries, and functional evidence and tissue mechanics remain unvalidated. No clinical deficit probability is produced.

Post-run review found two prototype boundary limitations. Its NumPy alignment checks leave the default relative tolerance enabled, so sufficiently small tilts or fractional source coordinates can pass the proposal precheck despite the stated strict axis restriction. It also silently skips transverse columns outside the image instead of recording a named omission. The executed phantom and patient cases have exact axis alignment, integer transverse origins and all thirteen columns inside the image, so these findings do not change their recorded candidates or audits. The frozen script is retained unchanged. A reusable module must use zero relative tolerance, check off-axis components explicitly, validate finite nonsingular transforms and record out-of-image omissions; regression cases must include a 0.001-radian tilt and a 0.0005-voxel fractional origin.
