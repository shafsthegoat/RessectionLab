# Earlier wide preparation: local benefit, no completed route

On the same exposed SELECT013 anatomy, both 48-move explorations still selected
STOP with zero target removal and zero selected return. Changing the fifth
physical preparation did not pass the unchanged useful-route criterion. This is
simulator experience on annotation-assisted anatomy, not observed surgery or an
RL improvement.

| Measured exploration | Original narrow choice | Earlier wide choice |
|---|---:|---:|
| Moves | 48 | 48 |
| Non-target cells removed | 83 | 86 |
| Target cells removed | 0 | 0 |
| Cumulative nominal return | −19.025398 | −19.658198 |
| Maximum removed-cell center depth | 5.5 mm | 5.5 mm |
| Native previews, including construction and selected replay | 9,894 | 9,876 |

Both branches recomputed the same first four actions and reached the identical
fifth state and certified catalog. The treatment changes tool and lateral ray,
so it does not isolate width alone. Every subsequent decision uses the same
focused recursive selector, world, reward and total horizon. The narrow branch's
entire history and action packets reproduce the retained baseline exactly.

The wider cut has a real delayed effect: its seven removed cells equal the union
of three later narrow-branch cuts. That advantage shrinks under the unchanged
continuation. Wide move 47 reaches the same 83-cell removed set as narrow move
48; this is set equality, not equal trajectories or complete states. The extra
wide move removes three more non-target cells without target progress. Its final
return loss of 0.632800 consists of 0.600000 extra removal penalty, 0.03 extra
tool switching and 0.0028 extra travel. [The saved diagnosis](postrun/REPORT.txt)
records the intervening preparation chains.

The [independent saved audit](independent/result-review.txt) passes: all 96
exploratory packet/state joins, both full-tool exploration geometry certificates,
and both selected STOP plans' authoritative and fresh replay bindings agree.
Both plans seal before evaluation. Full negative explorations receive independent
geometry checks; they are not themselves replayed as selected policies. The audit
does not reconstruct absent full catalogs or establish physical fidelity.

One fixed-order pair costs **351.643 s**, **19,770 previews**, **100 native
transitions**, **550,289,408 B** sampled peak memory and **19,697,184 B** output.
It exits cleanly within every cap. Seven generated helper controls and fifteen
serialized completion checks pass; the final helper differs from its tested
predecessor only in the recorded provenance wording. Failed fixture attempts
remain available. No model fitting, weights, new data, held SELECT037 or final
EVAL inputs were used. One pair does not establish a speed improvement.

The current first blockers are allocated in every focused state, yet both routes
spend their horizon on shallow preparation. Reject the particular early-wide-cut
fix. Next, estimate the full preparation burden of existing same-tool goal rays
from current public inputs, before deciding whether an alternative corridor
warrants one full comparison. A geometric burden estimate is not an action-count
bound; no global unreachability or guaranteed longer-horizon success follows.

The dated storage scan at 12:19:35 UTC measures **247.461 MiB** extension growth
and **332.632 GiB** free, before this approximately 0.88 MiB compact archive and
later Git/source writes. No dataset download or dependency was added. Full
generated traces remain local and ignored; [the evidence index](evidence-index.json)
pins their hashes and the files needed for reproduction. Runtime HEAD is
`a19b72993ef3ded6d147b364f45b7afc671975a1`.
