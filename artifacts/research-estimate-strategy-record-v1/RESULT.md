# Complete nominal strategy records

The existing scan-estimate planning interface now exports and imports complete JSON strategy records using the existing FrozenResearchPlan type. It preserves action ordering, tool changes, physical microsteps, removed and contacted cells, STOP/horizon status and resulting modeled state. Import replays against the same permitted-input specification before accepting the record. It does not open evaluator truth.

Root integration validation: 99 tests passed in 6.32 seconds (85 existing and 14 new). Independent review passed those plus six fresh controls, 105 total. A discovered float/Boolean horizon issue was corrected in the new replay helper. Corrupted history, state and accounting records are rejected; imported data are detached from mutable callers. Integrity hashes are not authentication or process isolation.

The saved explicitly generated DEVELOPMENT example keeps the same two-action strategy and six removed cells while a target-only evaluator changes scoring between two generated reference worlds (2 versus 0 mm³). Immediate STOP remains zero. This demonstrates recording and evaluation separation, not learning, patient transfer, withheld real anatomy, physical mechanics or clinical accuracy. No policy training or checkpoint loading occurred. Original planner costs remain caller-declared; recording replay calls are accounted separately.
