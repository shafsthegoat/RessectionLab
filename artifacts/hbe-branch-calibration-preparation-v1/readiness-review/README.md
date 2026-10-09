# Branch-calibration readiness review

The two completed temporal qualifications support the proposed bounded implementation. This review found no additional study to run first. It does **not** clear a fit or solver launch: the new branch-aware code does not yet exist in the inspected tree.

Four concrete handoff barriers remain, all explicitly covered by the owner’s `implementation-ready.json`:

1. **Access and evidence.** The old access layer fixes 18/20 runs, N12 axial references and 90-second execution. A separate source-bound release must name compression N36/S120, tension N24/S120 and the two N12/S120 torsion references. Check qualifications before opening the two axial members. Require both actual fitted axial confirmations, then freeze the recomputed common fit and torque predictions before any held-out member read. Preserve the existing durable attempt/freeze/reveal chronology and exact member allowlists.
2. **Actual modulus.** `half_frame`, `_frame_ratios` and temporal `read_frames` calculate physics at 1000 Pa. Reuse their geometry/reflection, but pass actual fitted μ through half/full stress, energy, force/moment and residual scales, work and consistency checks. Changing report metadata alone is invalid. Keep the existing scalar-only deck guard.
3. **N36 caps throughout.** The recorded node log is 774,836,975 bytes and the element log 517,916,077 bytes. The default 256 MiB verifier rejects both. Exact path/hash/kind limits must apply before parsing, afterward, at freeze and again before reveal—not only in the large-log parser. Retain the declared 768/512 MiB N36 caps and exact item counts; stream reference/fitted frames without retaining 121 full states.
4. **Inclusive resource accounting.** The new 3600-second aggregate and 2100/420-second native allowances must replace legacy limits only within the new release. Include input hashing, fitting, both actual runs, paired replay, freeze/reveal and publication. The proposed 3 GiB sampled RSS and 2 GiB new-output caps are failure limits, not predictions. Fitted logs have little headroom under the per-file caps; any excess remains a recorded terminal failure.

The existing positive common-scale fit and held-out metrics are reusable unchanged. No new optimizer, physical engine, sign correction or extrapolation is needed. The cheapest decisive controls are member-open spies for missing qualification/confirmation, a non-1000-Pa analytical physics case, exact large-log cap dispatch across freeze/reveal, and a corrupted state/reaction or exhausted clock.

Both saved qualifications retain `spatial_convergence_accepted=false` and `calibration_released=false`. New conditional branch eligibility must be declared explicitly and must preserve the historical failed aggregate. Withheld torque tests predictive agreement for other loading branches of this one specimen; it does not establish independent-subject or surgical validation.

`verification.json` binds the inspected source, both compact qualification records and current raw file sizes. No measured curves, raw primitive contents, solver, fit or test were run, and no tracked file was edited.
