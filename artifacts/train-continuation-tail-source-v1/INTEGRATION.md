# Exact immediate reward with a learned continuation tail

The previous calibrated model recovered greedy target volume but required 42
movements versus 25 and 30 tool changes versus zero. A saved scalar diagnostic
finds lower immediate benefit than an available alternative at 36 of its 42
productive decisions. Those local gaps are not executed counterfactual outcomes.

This single change learns the best recorded prefix **after** the current action.
The planner then adds its already-computed immediate nominal reward exactly
once, clipping the predicted future tail at zero. Task rewards, geometry,
actual-positive guard, candidate inventory, STOP and patient roles stay fixed.
The distinct `FittedContinuationTail` type exposes `score_tails`; direct-Q use
is rejected. The shared planner preserves its existing direct-Q path.

The fit reuses the authenticated 497-row cache, original weights and calibrated
hidden layer. Original manifest rewards produce float64 tail labels, with an
exact single-rounding join back to the original mandatory-action Q targets.
One fixed minimum-change SVD solve changes only the final readout. No new
training trajectory, encoder update, hyperparameter sweep or acquisition occurs.

Validation is complete before release:

- 28 generated model/planner controls pass in 5.40 s, including the unchanged
  direct-Q regressions, exact decomposition, type boundary, frozen inputs,
  complete native replay, costs and failure handling. Parent: 6.304 s;
  peak group RSS: 342,687,744 bytes.
- The independent oracle passes 414 checks in 3.541 s, using direct prefix
  enumeration and scalar normal equations. Parent: 3.946 s; peak RSS:
  338,345,984 bytes.
- 19 runner controls pass in 6.58 s, including isolated startup, real exclusive
  output writing, original cache/source joins and unchanged comparison gates.
  Parent: 7.179 s; peak RSS: 373,047,296 bytes.

All three supervised batches exit zero, are reaped and retain unchanged source.
Root reviewed the model, shared planner and complete runner; source-only
preflight passes. The exact released source index is
`b5f70a83311adb2915e81dac2747a0315e704e2141d214016dba9245c7fccb04`.

The next execution compares twelve fresh GREEDY/DIRECT_CALIBRATED/TAIL routes on
the same four TRAIN patients. It must reproduce the original head and direct
calibration, plus all retained greedy/direct routes. A useful gain requires
unchanged casewise target/return and outside/contact constraints versus greedy,
then a strict reduction in previews, transitions, travel or movements. Ties and
improvement only over the weaker direct head remain negative. Existing failures
and all collection/training costs remain charged.

The shared stage is reserved for at most 32 MiB, with 12 MiB per acquired run,
one CPU thread and 3 GiB RSS. The recent comparison supports a 400–700-second
forecast, with 3,600 seconds as a ceiling. Raw anatomy, cached features and
weights remain in their original ignored locations. The precise commands and
budgets are in `runner/README.md`; this source milestone claims no acquired
route improvement. It remains an annotation-assisted simulator study, with
held-out patient and independent physical validation still outstanding.
