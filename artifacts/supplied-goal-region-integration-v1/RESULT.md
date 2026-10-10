# Preserve supplied targets when estimated tissue support disagrees

The first qualified ReMIND-008 crop exposes 6,394 of 18,509 supplied tumor
voxels outside the automatic tissue-support estimate (34.55%). The new explicit
`supplied_goal_region_may_exceed_estimated_support` condition preserves both
arrays and the full target denominator. Only occupied target cells can be
proposed for removal, removed or rewarded. Metrics label this condition
`PARTIAL_TARGET_PROGRESS` and report the unsupported extent separately.
No patient material, filled tissue, clipped target or full-resection claim is added.

The existing strict subset condition remains the default, with unchanged
source/model identities. The new semantics bind the case and proposer identity.
Root canonical validation: **102 tests passed in 5.23 seconds**, including six
new generated controls and existing proposal, native task and patient integration
checks. Independent source review passed before execution. These are software
controls; no patient search or gradient was run in this slice.
