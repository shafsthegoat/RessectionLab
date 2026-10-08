# Native opening learning: fixed pilot

The released pilot completed, but **neither learned policy mastered the two-tool
opening task**. Complete nominal search achieved 1.1. After 16 updates, imitation
learned to STOP at zero; scratch REINFORCE produced −0.896, below the untrained
policy's −0.464. This is a retained negative learning result on one generated
six-cell fixture, with no patient data or generalization claim.

| Method | Geometric return | Target removed (mm³) | Normal removed (mm³) | Online seconds / native previews |
|---|---:|---:|---:|---:|
| STOP | 0 | 0 | 0 | 0.0071 / 0 |
| Untrained | −0.464 | 0 | 2 | 0.0712 / 12 |
| Random legal, seed 1011 | 1.098 | 2 | 4 | 0.0606 / 12 |
| Random legal, seed 1012 | −0.692 | 0 | 3 | 0.0433 / 12 |
| Random legal, seed 1013 | −0.631 | 0 | 3 | 0.0471 / 12 |
| Immediate greedy | 0 | 0 | 0 | 0.0091 / 0 |
| Complete depth-two search | 1.1 | 2 | 4 | 0.2745 / 60 |
| BC, 16 updates | 0 | 0 | 0 | 0.0732 / 0 |
| Scratch RL, 16 updates | −0.896 | 0 | 4 | 0.0500 / 12 |

Both fits began with identical seed-11 parameters (`2fa97c0e…`) and separate
Adam optimizers. All 32 updates changed parameters with nonzero actor and image
encoder gradients. BC used five distinct reachable teacher states, each checked
by an actual complete replay and independent native audit: 80 supervised loss
forwards across 16 updates. Its pre-update cross-entropy declined from 1.40251 to
1.29210, but its final initial-state argmax became STOP. This improves the
geometric objective by abstaining; it does not demonstrate useful resection.

RL used 64 complete on-policy episodes, 112 actual transitions and 112 repeated
loss forwards. Ten sampled episodes removed target tissue; mean sampled return
was −0.290234. Final argmax behavior removed four normal cells and no target.
The probability mass on the exact optimal first action changed from 0.20746 to
0.19858; BC changed it to 0.17781. These last figures are read from saved decision
probabilities and the complete nominal action-value tree, without new forwards.
There is no evidence here that the failed final behavior was caused by missing
all positive experiences, and no isolated diagnosis of representation versus
optimization or limited training budget.

Search enumerated 20 nominal transitions covering 16 terminal sequences. Unlike
myopic greedy, it retained the initially negative opening and switched tools to
reach both targets. Its teacher labels covered all five distinct nonterminal
observations. Offline validation replayed five complete demonstrations (10
transitions) in 0.3857 seconds; those costs are additional to the shared search
and reported as such. No partial or unreplayed search prefix was used as a label.

All **78 actual episodes** passed the independent complete-tool, native-cell and
reward audit: nine evaluation episodes, five teacher validations and 64 RL
episodes. They contain 137 committed transitions and zero invalid attempts.
Geometric acceptance does not validate tissue mechanics, clinical injury or
deformation. Contact remains separate from removal; clinical, motor and language
outcomes remain null. The synthetic image and exact committed cavity are
simulator observations, never recorded operative evidence.

Shared preparation took 0.03864 seconds and 12 previews. BC fitting took 0.2416
seconds; RL fitting took 4.0232 seconds, including collection and its audits.
The online table includes planning, clone, replay, successor inventory and
terminal export under identical 10-second/2,048-preview guards. Audits are
separate: 0.0372 seconds for search and 0.0267 seconds for final RL. Actor latency
alone is not the online cost. Timings are single-run measurements in fixed order,
not a randomized performance benchmark.

The worker completed in 6.5466 seconds; the whole supervised call took 8.6398
seconds with 332,644,352 bytes peak sampled RSS, below the declared 300-second/
2-GiB envelope. No retry, parameter sweep, checkpoint selection, patient opening
or cap extension occurred. The preceding cost profile used discarded weights
and a separate source snapshot.

Execution used committed source `0467ad7`, declaration SHA-256
`963238d34dce36f4a9128c782f0d3ca667640123a58a2eb2498abb01d68c5212`.
[summary.json](summary.json) contains unrounded outcomes, costs and identities.
The original `output-sha256.json`, raw initial/latest checkpoints, all receipts
and 31-file source snapshot remain unchanged for independent review. This report
and summary are post-run products outside that original execution index.

Independent saved-output review subsequently passed 10,726 checks, including all
indexed output/source hashes, reward/cell accounting, tensor/optimizer identities,
complete method denominator and resource limits. See
[verification](independent-review/verification.json), SHA-256
`faed3f45068a513272ac1927e2ce696563b346320f400c2733d5d1990bd75223`.
No new model forward, update, replay or patient access was used for that review.
