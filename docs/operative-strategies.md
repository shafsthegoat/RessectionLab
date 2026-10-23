# Operative playback and a narrow rescue comparison

The desktop now offers source-interval flow effects, active drill/jaw animation,
and an optional **Follow accepted tool** camera. Tool root poses, tissue states,
quantitative ledgers and timeline endpoints remain recorded engine values.
Droplets retrospectively illustrate recorded operations once during playback,
with a labeled 3× radius display gain. They are an evidence overlay, not a fluid
surface or a measured velocity field. Pause and scrub show static evidence.

**Close tool view → Follow accepted tool** moves the camera and local cutaway
only when a new accepted state arrives. It preserves the current orbit/zoom for
the same instrument. Source slices follow only tips inside the acquired image;
otherwise they retain their last position. Restore returns the previous view.

Prepare a comparison using the existing patient trajectory and a new ignored
output directory. Select the same optional native builds used for your saved
cache, if available:

```sh
export RESSECTIONLAB_IDLE_THERMAL_BUILD="$PWD/testcases/outputs/rl-readiness-20261022/idle-integer-native-production-build/build-receipt.json"
export RESSECTIONLAB_PREFIX_JSON_BUILD="$PWD/testcases/outputs/rl-readiness-20261022/prefix-json-canonical-native-production-build/build-receipt.json"
export RESSECTIONLAB_VERIFIED_CACHE_MAX_DISK_BYTES=8589934592
.venv/bin/python scripts/compare_vascular_rescue.py \
  --case testcases/outputs/full-case-rehearsal-20261020/final/rehearsal.ressectionlab \
  --reference testcases/outputs/full-case-rehearsal-20261020/final/vascular-canonical.actions.jsonl \
  --output testcases/outputs/product-first-20261023/rescue-v2
./scripts/open_operative_demo.sh
```

For a watchable replay after a code change, prepare the authenticated observations
once with the same native-build environment shown above:

```sh
.venv/bin/python scripts/prepare_operative_playback.py \
  --case testcases/outputs/product-first-20261023/rescue-v2/rescue-strategies.ressectionlab \
  --branch strategy-recorded-delay --actions 16:23,47:73 \
  --report testcases/outputs/product-first-20261023/playback-preparation-v1.json
```

These zero-based ranges cover a short tissue-removal sequence and the separate
vessel injury/intervention. The command verifies the complete native archive,
then prepares only those actions. It preserves the workspace and reports actual
frame counts and recorded wound deposits. Use a new report path for a new run.
Preparation is separate from playback speed and is not a fresh-run benchmark.
When a later action is missing, the player now uses the preceding authenticated
observed boundary before reconstructing a prefix. This matters on this patient:
one snapshot occupies about 10.8 MB of Python objects, so two demanded states
exceed the existing 16 MiB replay-memory allowance. Cache budgets stay unchanged.

The runner canonically verifies the shared 48-action prefix, then executes and
separately replays three continuations: prompt coagulation, suction before
coagulation, and the recorded delayed response. The first two use identical
intervention doses and 23 commands in different orders. The recorded response
adds two inspection commands and seven simulated seconds before intervention.
This compares response to one deliberately recorded vessel injury; it does not
generate alternative tumor access corridors or claim completion of resection.

In **Rehearse**, verify the saved branch, then open **Research Planning** to
compare outcomes and select an alternative. The comparison reports additional
target/non-target removal, residual target, face-damage area, modeled wound fluid,
retained fluid including outlet reservoirs, travel and elapsed time. Complete
protected-structure encounters and clinical risk remain unknown. Coagulation
does not restore severed vessel patency. The command slider aligns actual
accepted commands; rejected commands do not advance the native timeline.
At a boundary, Branch creates an editable continuation while retaining the plan.

The same comparison can be computed in the app at the verified injury boundary
with **Execute three vascular rescue strategies**. All three endpoints enter the
existing bounded session cache after their replay checks, so selecting a newly
computed plan does not repeat its physics. Saved files retain the complete real
action histories, source anatomy and exclusion contract. Cold native admission
remains expensive and must finish before planning or edits are enabled.

This workflow trains no policy. The common access and removal prefix is shared,
and all geometry/flow/damage parameters remain uncalibrated research assumptions.
Local case files, failed attempts, execution measurements and visual QA belong
under `testcases/outputs/product-first-20261023/`.

## First patient execution

The complete preparation and save/reopen check took 269.095 s; executing and
checking the three continuations took 107.579 s. All 71 proposed commands and
71 separate replay commands completed without rejection. Each final full state
and measured endpoint vector matched its separate replay. The recorded-delay
endpoint also matches the original 73-action vascular archive exactly.

Actual Electron recordings are saved as `vascular-continuous.mp4` and
`cavity-pinned-continuous.mp4` under the local product-first output directory.
They show the recorded injury/intervention and progressive CUSA cavity changes.
These sampled screen recordings establish visible behavior, not compositor
throughput. Long historical CUSA strokes leave the acquired image; pinning the
camera to the cavity is more useful than following the tip through empty space.

The separately saved `rescue-v2/rescue-explorable-v3.ressectionlab` retains the
three strategies computed in Electron (112.679 s), plus an editable branch from
boundary 48 with a real additional inspection. All original native branches and
source anatomy remain identical. Open that version with:

```sh
./scripts/open_operative_demo.sh --case "$PWD/testcases/outputs/product-first-20261023/rescue-v2/rescue-explorable-v3.ressectionlab"
```

| Strategy | Commands | Additional modeled wound fluid | Retained fluid incl. reservoirs | Additional simulated time |
| --- | ---: | ---: | ---: | ---: |
| Prompt coagulation | 23 | 1.339 mm³ | 1.270 mm³ | 23 s |
| Suction before coagulation | 23 | 3.063 mm³ | 7.187 mm³ | 23 s |
| Recorded delayed response | 25 | 4.394 mm³ | 4.324 mm³ | 30 s |

All endpoints have zero active modeled outlets and the same 43,261 mm³ residual
target. The shared prefix removed 312 mm³; the rescue continuations remove no
additional target. Suction-first moves the same suction dose ahead of coagulation
and irrigation, so later fluid remains in the field. These are model predictions,
not clinical comparisons. The saved workspace preserves every historical branch,
source array, image and full-case director record byte-for-byte in decoded form.
The failed initial 100-second comparison attempt remains in `rescue-v1.log`;
the successful bounded retry uses the four-minute comparison budget.

## Catalog-sized skills: measured task limitation

The patient archival radii and inherited time/damage budgets do not satisfy the
raw RL contract. `scripts/probe_patient_rescue_skills.py` therefore makes an
explicitly privileged native injury setup, then checks catalog-sized nominal
skills through real public actions; it does not transplant a checkpoint into an
RL episode or run a learner. Run with `--case` pointing to the comparison above
and `--output` naming a new directory under `testcases/`.

The bounded probe took 219.382 s, including 162.971 s of prefix admission.
Observation, proposal and public execution together took 10.851 s for nine prompt
skills/ten primitives, and 12.775 s for eleven delayed skills/twelve primitives.
Both suffixes replay exactly and are saved as native branches. The 1 mm bipolar
tip sealed three of five wounded faces; two faces outside its fixed coverage
continued emitting. The 1.5 mm suction tip removed little fluid at the recorded
endpoint. During the final five-second observation, prompt-arm fluid rose from
2.335 to 3.119 mm³. Both arms still had one active outlet.

That initial catalog-sized probe was unsolved and launched no fit. The later
[prepared patient rescue task](prepared-patient-rescue.md) supplies validated
surface contacts, effective suction and explicit suffix observation/budget/
exposure semantics before permitting a bounded learning comparison.
The evidence, unchanged source hashes and coverage diagnosis are in
`testcases/outputs/product-first-20261023/catalog-skill-readiness-v1/`.
