# Research planning from scan-derived estimates

`resectionlab.research_estimate_planning` is a **generated-only development interface** for testing a limited-information route comparison. It has not admitted a patient, verified an anatomical model, measured physical fidelity, or established a clinical route benefit. Production `require_admitted_model` remains unchanged and rejects these provisional model hashes.

The input separates preoperative scan sources, a candidate brain envelope, a candidate whole-tumor target, a hypothetical access window, tools, reward and a bounded planning horizon. Every source and estimate retains its model/run/asset/output/coverage and physical-frame hashes. An exact acquisition timestamp is used when available; otherwise the timestamp remains `None` with a hash of source evidence attesting preoperative ordering. That claim requires independent verification before any patient use. Estimate `available_at` is its actual generation time, and `decision_cutoff` is the retrospective research planning cutoff; neither implies the estimate existed at the historical operation. The `DevelopmentUseDeclaration` binds these identities to `DEVELOPMENT` nominal comparison but is caller-declared, not an authenticated release or cohort audit.

`research_planning_from_estimates` gives SEARCH, IL, RL or HYBRID callbacks a fresh task whose first `NativeSpatialCase` uses the **nominal estimate as both reference and nominal target**. No withheld reference enters planning. A completed STOP-or-horizon action sequence is independently replayed and sealed with its source, initial observation, decision model and physical history. The method names and accounting are caller claims; this interface alone does not run or compare trained policies. The sealed sequence is open-loop and does not validate adaptive intraoperative decisions.

A bounded native-grid ROI preserves the original physical affine. Planning abstains when target/support QC fails, the target extends outside the ROI, either estimate lacks coverage within it, or any declared fixed-lattice tool path touches unqualified voxels. The footprint audit uses a conservative in-image full shaft/tip capsule, including rays that might become feasible after an opening. Unknown voxels outside a qualified ROI may remain unknown. Geometry outside the source image is **unassessed**; the plan seal and evaluation result explicitly record `outside_source_fov_tool_feasibility_assessed=false` and `clinical_use_permitted=false`. The candidate brain envelope is not a cortical surface, the target is not prescribed removal, and neither model supplies validated motor, language, vascular or clinical-harm probabilities.

`evaluate_sealed_research_plan` repeats the same input/coverage/footprint preflight and validates nominal replay **before** calling a zero-argument private target loader. It then replays the fixed route for a generated target-only diagnostic. A forged STOP seal cannot bypass an abstention. The boundary is an accidental-misuse interface within one Python process, not file/process isolation: a future evaluator needs its own authenticated private-reference binding and isolation. A future patient loader must verify actual source-file bytes, source-backed preoperative status, modality registration, model and adapter rights/lineage, person-grouped split, estimate QC and the ROI selection rule before issuing any scoped research release. No patient inference is authorized by this module.

Generated-only controls: `.venv/bin/python -B -m pytest -q tests/test_research_estimate_planning.py -p no:cacheprovider`. The historical 9×9×7 limited-observation fixture remains separate regression evidence and is not a patient admission path.

## Complete strategy records

`research_strategy_to_record(plan, spec)` returns detached JSON-compatible data
for the complete nominal strategy. It includes tool changes, physical microsteps,
removed/contact cells, the planning grid/affine and resulting STOP-or-horizon
state. `research_strategy_from_record(record, spec)` verifies the existing seal
and replays the same permitted-input world before returning `FrozenResearchPlan`.
Neither helper receives a private-reference loader. Sparse state changes require
the exact bound input specification for reconstruction; they are simulated
consequences, not observed operative anatomy. Export/import replay is recorded as
overhead separately from the caller-declared planner costs.

Focused regression: `.venv/bin/python -B -m pytest -q tests/test_research_estimate_strategy_record.py -p no:cacheprovider`.
The generated example and independent review are retained in
`artifacts/research-estimate-strategy-record-v1/`.
