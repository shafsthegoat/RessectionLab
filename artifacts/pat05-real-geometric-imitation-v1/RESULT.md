# Eight-update real PAT05 imitation diagnostic

Imitation changed the chosen path and improved this same-patient geometric
return from 12.7030 to 139.8975. It remained substantially below greedy search
at 410.3124. This is an imitation-learning gain on one TRAIN anatomy, not an RL
gain, patient generalization, or evidence of surgical effectiveness.

Starting tensors were exactly the original seed 11 pre-RL weights. The unchanged
30,827-parameter spatial CNN received the same annotation-assisted T1/nominal
maps/cavity and certified tool geometry. A fresh Adam optimizer performed eight
fixed cross-entropy updates at 0.001, global clip 5, on the three decisions from
the completed greedy plan. The teacher was reexecuted first: all three native
history records matched the prior independently accepted history exactly and
passed a new independent check. No unfinished search prefix was used. Only the
fixed latest weights were evaluated; no outcome-selected checkpoint or sweep.

| Fixed argmax readout | Target mm³ | Normal mm³ | Geometric return |
|---|---:|---:|---:|
| Original initial policy | 17.0001 | 20.0001 | 12.7030 |
| After two prior RL updates | 17.0001 | 20.0001 | 12.7030 |
| After eight BC updates | 173.0006 | 164.0006 | 139.8975 |
| Greedy three-decision reference | 493.0017 | 412.0014 | 410.3124 |

All three BC-selected actions changed. Its fresh complete rollout passed
independent full-tool geometry and source-cell reward/accounting checks,
removing 1.51% of the supplied target. All eight updates changed actual tensors;
encoder and actor gradients were nonzero, and the unsupervised critic had zero
gradient. Teacher cross-entropy fell 4.140585→4.097359. Teacher-action ranks on the
three fixed teacher observations improved 27/13/13→3/6/2, so even the training
demonstration was not exactly reproduced.

The useful and failed parts are distinct. BC's first cut earned 147.3035
(173 target/128 normal), close to the teacher's150.7035 (175/121). Its next two
cuts each removed0 target and18 normal, earning−3.7010 and−3.7050. Teacher steps 2/3
earned 132.1055 and127.5034. Thus 267.0149 of the 270.4149 return gap arose after
the first cut. The BC path creates a different cavity than its three teacher
states. This is consistent with insufficient visited-state supervision; it does
not prove that state coverage is the only cause or that the encoder cannot
represent the needed distinction.

Actual supervised run63.94 s, worker61.88 s, peak RSS1,897,152,512 bytes. Costs were
7.42 s source/initial inventory preparation,19.36 s teacher replay/audit,10.14 s
across eight loss/backward/Adam updates, and20.75 s latest rollout/audit. Latest
actor decisions totaled 0.616 s, native transitions plus successor inventories
13.921 s, and independent audit3.997 s. The reused teacher's original planning
cost 17.40 s remains additional historical cost; teacher generation was not free.
Random/search/initial anchors are reused from the prior same-task run, not new
independent samples. No amortized runtime advantage has been demonstrated.

Only PAT05 TRAIN was used; SELECT and unopened patients remain untouched.
The same three-decision horizon includes STOP. The CNN ranks already certified
proposals from a 64³ crop, while search reads full permitted nominal fields.
This is rigid-cell geometric partial removal under an unreviewed support/access
assumption. It does not establish full resection, tissue mechanics, functional
injury prediction, or clinical safety.

Initial parameter hash: `sha256:2fa97c0e730db09371786d5ba903ffdf467b3cc7149a111c182d6a7499cdd9b8`.
Latest: `sha256:0d89cd47ea87caad844e20598816d1122ad135c84365fd855b049fabe2827c9e`.
Exact source/configuration, eight-update curve, fixed teacher ranks, real action
logs, replay histories and latest checkpoint are preserved with verified lossless
archive and unchanged execution-time output hashes.

A narrowly justified next test would query the same permitted greedy teacher on
the actual BC-visited cavities, then test a fixed imitation update on those
additional states. That proposal has not run; no additional configuration or
patient has been opened on the basis of this result.
