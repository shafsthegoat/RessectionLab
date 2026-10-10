# Trained transfer executes and reopens in the Mac app

One live Electron invocation of `Trained RL256 transfer · aspiration only`
loaded the existing fixed checkpoint, ran the actor and matched search, and
published native replay. It used the generated 13×13×12, six-decision desktop
task; the checkpoint was trained on the earlier 9×9×7, two-decision task. No
weights changed. This is one generated transfer result, not patient validation.

| Saved result | Trained actor | Matched aspiration search |
| --- | --- | --- |
| Complete strategy | Aspirate, then STOP | Same exact action IDs |
| Modeled return | 3.353 | 3.353 |
| Target / other tissue removed | 4 / 3 mm³ | 4 / 3 mm³ |
| Target remaining | 32 mm³ | 32 mm³ |
| Full insertion and return path | 17 mm | 17 mm |
| Online planning work | 2 actor forwards | 14 of 24 allowed branch transitions |
| New optimizer updates | 0 | 0 |

The width-4 search completed without hitting its call/time cap, but pruned five
prefixes; this is not an exhaustive optimum certificate. The actor matches this
search result. The run does not demonstrate superiority, reliable generalization
or a speed advantage. Search reported 0.7574 seconds; isolated actor latency was
not recorded. The complete owned worker, including loading, comparison and
verification, took 5.0663 seconds at 266,321,920 bytes sampled peak RSS and exited
zero. All 24 memory samples and child cleanup are retained. Training costs remain
part of the original RL256 experiment, not zero-cost pretraining.

The UI displayed 72 recorded frames, including the full aspirator during
withdrawal at frame 37 and STOP at frame 71. Save succeeded. A fresh app process
reopened the same episode at frame 71 with identical removal quantities and
explicitly unverified imported authorship and computational provenance. There
was still only one model-attempt directory after reopen. Reopening did not run
the actor. The independent generated vascular annotation report showed one
positive and one unknown in-grid encounter cell for the aspiration; STOP had no
tool sweep. These are geometric encounters, not injury probabilities.

The [live replay capture](live-mac/trained-withdrawal.png) and
[fresh reopen capture](live-reopen/imported-provenance.png) are unmodified images
of the running app. The live UI session exited normally after 136.232 seconds
with 1,106,575,360 bytes sampled tree peak; reopen exited normally after 21.693
seconds with 784,449,536 bytes. Neither reached its declared resource limit.
The Save dialog retained its previous local directory despite the first folder
navigation attempt; the new unique filename was saved there without overwriting
prior work. Its exact local path and hash are in the [evidence index](source-index.json).

To reproduce from the local checkout with the fixed model asset present, open
the Electron app, select the Episode tab, choose the trained aspiration option,
and execute the generated episode. Scrub recorded frames, evaluate annotated
encounters, then Save and reopen. Each explicit execution creates a fresh owned
attempt; there is no automatic retry. The source release is commit
`ed87de4b8931819aac583ecd6f11efdbcf0c5361`. Packaged model distribution is not yet
implemented. The working App includes preserved unrelated local edits; the
exact staged desktop separately passed its build.

The paired search is present in the durable backend result. A desktop companion
route view is the next integration slice. Mixed-mode learned actions, unseen
patients, tissue force/deformation and clinical usefulness remain unvalidated.

The [independent saved-result audit](independent/REPORT.txt) reproduced the entire
episode and all 72 cumulative frame masks once without actor, search or weights.
It verified both recorded decision chains, permitted inventories, source bindings,
outcomes and accounting. After that replay, the reviewer corrected two mistakes
in its own workspace/episode hash recipes using saved JSON only. The original
failures and source versions remain preserved; no native replay was retried.
Workspace seals and saved frame 71 then passed. Image arrays were not decoded,
and primary-image identity remains a saved claim in this separate audit.
