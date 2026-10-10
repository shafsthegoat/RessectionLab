# Final N24 tension timestep check

Ordinal 11 completed one native solve and one official readout at source
`47fa6d2`. Independent replay reproduces the entire saved readout exactly:
121 frames, 120 steps. All twelve scheduled numerical rows are now available
for the frozen cross-resolution comparison. This does not yet establish its
convergence verdict or physical tissue accuracy.

| Stage | Seconds | Sampled peak RSS, bytes | Result |
| --- | ---: | ---: | --- |
| Native solve | 128.8728 | 495,583,232 | Exit 0, reaped |
| Official readout | 24.1293 | 307,331,072 | Exit 0, reaped |
| Independent checksum and replay | 25.0326 | 324,190,208 | Exact equality, reaped |

Final modeled reaction is 0.0266433059681 N at 0.00073601 m. Maximum normalized
numerical criterion ratio is 0.000219597; minimum sampled Jacobian is 0.878487.
The lower-half specimen is reflected into the full representation. Sampled
Jacobian positivity is not an everywhere proof. These are simulated values,
not measured tissue forces or patient outcomes.

The [independent audit](independent/REPORT.md) checks all eight native output
files, release/source bindings, original numerical criteria and owned cleanup.
The full native-chain ledger records 12 calls, 2,394.444 seconds and
3,524,119,165 bytes, including declared continuation reserves. Historical N8
preparation/readout times remain missing. Independent audit costs above are
separate. The [source index](source-index.json) preserves compact exact evidence;
bulk numerical streams remain local and outside Git. Measured response,
calibration and patient validation remain separate unreleased work.
