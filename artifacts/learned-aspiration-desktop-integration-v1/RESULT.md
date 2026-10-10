# Fixed trained aspiration selector: source integration

The Electron episode selector now exposes `Trained RL256 transfer · aspiration
only`. Its backend owns the checkpoint path, verifies the original trained
identity, and runs the actor and width-4/24-call search on the same projected
STOP/aspiration inventory. Both use the existing generated native state and
replay geometry. No renderer-provided model path is accepted.

Root validation passes **84 backend tests** and **372 desktop tests**. Both the
working desktop and exact staged desktop build successfully; the latter excludes
the unrelated App and neighboring-path edits. Independent V2 review passed 17
controls; V3 portability passed seven, including an installed-source layout with
no ignored build directory. These are software controls with no real checkpoint
load. Actual trained execution and Mac inspection are the next result slice.

The fixed checkpoint was trained on the earlier generated 9×9×7, two-step task.
The desktop task is 13×13×12 with six decisions. The adapter preserves that
distribution change instead of relabeling training ancestry. No new training
occurs; probe actions are excluded from both comparison arms. A current source
checkout and the independently bound local RL256 asset are required; packaged
model distribution remains open. The checkpoint is excluded from Git.

The owned child has a 20-second work window, 25-second operation/cleanup envelope,
1 GiB sampled direct-child RSS limit, 2 MiB result and 1 MiB log limits. It uses
one numeric thread, a fresh cache path, parent-disappearance termination and no
automatic retry. Final source/filesystem checks are parent finalization, not a
hard real-time watchdog. Sampled RSS can miss transient peaks.

Save/reopen checks native geometry without executing the actor. Imported
authorship and all recorded computational counters/timings remain explicitly
unverified. Live attribution is issued only after a successful owned checkpoint
run. Post-seal vascular scoring retains its existing separate evidence boundary.

[Preserved reviews and test receipts](source-index.json) include the V1 failures
and their repairs. The broader mixed-instrument SEARCH selector is unchanged;
the matched aspiration search is retained inside the paired backend result.
Displaying its companion route is a separate integration follow-up. These
changes establish neither patient generalization nor physical/clinical validity.
