# Controlled visited-state imitation on real PAT05

The visited-state branch improved independently checked geometric return to
290.5060, compared with 239.1058 after the same number of additional updates on
the original demonstration. This single deterministic pair supports a benefit
from the added state examples under the tested conditions. Repeatability,
patient transfer and an advantage over greedy search are not established.
This is imitation learning, not RL.

Both branches separately reloaded the exact BC8 policy and saved Adam state,
including its momentum tensors. Control used the original three demonstrated
observations twice. Augmentation used the original three plus the three live
observations encountered by the frozen BC8 policy. Both therefore processed
six examples per update, eight updates, and 48 gradient-loss forwards. The
initial observation is shared: control has three distinct observation hashes,
augmentation five. It was deliberately retained twice in both datasets.

Only existing PAT05 TRAIN, annotation-assisted inputs, the unchanged 30,827-
parameter CNN, native proposals/tools/aperture/reward and three-decision horizon
were used. Both final checkpoints were fixed in advance. No SELECT or unopened
patient, new architecture, favorable checkpoint choice or configuration sweep.

The new labels came from complete immediate scoring of each exact current legal
inventory in a permitted nominal planning clone. Before and after each teacher
query, the actual state/observation hashes were identical. The first action of
that current decision was the label; an appended STOP was never treated as a
second sample. The actual frozen BC8 rollout then completed, matched its earlier
history exactly, and passed independent geometry/reward checks before training.

| Frozen BC8 state | Its chosen immediate return | Teacher best immediate return |
|---|---:|---:|
| Before first cut | 147.3035 | 150.7035 |
| After first cut | −3.7010 | 132.1055 |
| After second cut | −3.7050 | 132.1055 |

These measurements establish that useful alternatives existed after the first
cut. The negative continuations were not forced by action availability.

| Fixed latest rollout | Target mm³ | Normal mm³ | Return | Change from BC8 |
|---|---:|---:|---:|---:|
| BC8 starting policy | 173.0006 | 164.0006 | 139.8975 | — |
| Original-only continuation | 286.0010 | 233.0008 | 239.1058 | +99.2084 |
| Visited-state continuation | 345.0012 | 271.0009 | 290.5060 | +150.6085 |
| Earlier greedy reference | 493.0017 | 412.0014 | 410.3124 | +270.4149 |

The matched branches selected the same first cut (return 144.1035). Control then
earned 98.7033 and −3.7010; augmentation earned 118.1034 and 28.2991. The augmented
branch removed 59 additional target cells and 38 additional normal cells, yielding
51.4002 more return. Its three cuts each had positive geometric utility, but it
still trailed greedy by 119.8064. Target fractions were 2.50% and 3.02% respectively;
this limited proposal catalog is not a whole-resection task. Contact and path
length remain reported in `summary.json`, without converting them into clinical
injury probabilities.

All four complete histories (original teacher, frozen visited collection,
control latest, augmented latest) passed the independent native geometry and
source-cell accounting checks. Invalid attempted actions were 0. Each branch
made eight actual parameter updates, with equal initial tensor and Adam hashes.
Training losses decreased 4.097359→4.024712 for control and 4.115049→4.038064 for
augmentation. These losses concern different sample sets and are not a direct
quality comparison; fresh rollout outcomes above are the comparison.

Supervised wall time 160.40 s (worker 158.02 s), peak RSS1,860,206,592 bytes, below
360 s/6GiB. Shared preparation 7.77 s, original teacher replay 19.56 s, and frozen
visited collection 32.29 s were explicit costs. The latter includes 7.85 s of new
label searches and must not be added twice. Eight-update compute cost 22.71 s
control/22.92 s augmented. Final rollout costs 20.68 s/21.39 s include native steps,
reporting and independent checks; actor decisions alone took 0.90 s/1.00 s. The
original teacher's17.40 s planning and BC8's earlier 63.94 s experiment are separate
historical costs. No free-preprocessing or amortized runtime advantage is claimed.

Shared starting policy: `sha256:0d89cd47ea87caad844e20598816d1122ad135c84365fd855b049fabe2827c9e`.
Shared Adam: `sha256:d29d96fc401d698bd788b1ac199b51981db08388c29f510ddd48b7df7c86ae2a`.
Control latest: `sha256:6aa1fda3f0f044b9c26a04380da24d86f02bbe45ccd24eaab2d43ecfb6a44f8a`.
Augmented latest: `sha256:74e90d9487dff6fe9db83c92e3887f15533e0499abaac386a73bbd7d4d566b4e`.
Exact source, configuration, label inventories, update curves, full histories
and both checkpoints are retained with a lossless verified archive. The original
execution-time hash inventory is unchanged.

This does not justify indefinite tuning on PAT05. The next evidence should
address the already assigned real training/selection patients under frozen
input/support contracts, preserving existing support conflicts and unopened
roles, rather than presenting another same-case improvement as generalization.
