# Bimanual rehearsal telemetry

`src/resectionlab/surgical_telemetry.py` exports descriptive time-resolved telemetry from the same `SurgicalEngine` transitions used for rehearsal. It supplies JSON, CSV, and a standalone interactive HTML timeline. Labels `manual`, `scripted`, `search`, and `RL` identify the action source; they do not classify expertise. No model is trained by this exporter.

We referenced **Yilmaz R et al. (2022), “Continuous monitoring of surgical bimanual expertise using deep neural networks in virtual reality simulation,” npj Digital Medicine 5, 54**, [doi:10.1038/s41746-022-00596-8](https://doi.org/10.1038/s41746-022-00596-8). The study extracted bimanual, tissue-removal, and bleeding features at 0.2-second intervals from simulator recordings and used an LSTM. Here, the paper motivates feature selection and reporting cadence. Its expertise model, coaching thresholds, and clinical implications are not transferred.

## Capture and compare

```python
from resectionlab.surgical_telemetry import capture_episode, export_telemetry

# final_engine is the existing, completed SurgicalEngine. Supply every attempted
# action, including rejected actions, in chronological order.
episode = capture_episode(
    final_engine,
    attempted_actions,
    label='scripted',
    episode_id='patient-rehearsal',
    interval_seconds=0.2,
)
paths = export_telemetry([episode], 'outputs/rehearsal-telemetry')

# Collect other action sources separately, then pass all episode dictionaries:
# export_telemetry([manual_episode, search_episode, rl_episode], output_dir)
```

Passing a final engine preserves its bound protection annotations and guard configuration, and verifies that replay ends at its exact state identity. `capture_episode(scene, attempted_actions, snapshot=final_snapshot, ...)` restores and validates a saved engine snapshot first, including its protection contract. `capture_episode(scene, attempted_actions, protection=..., guard_config=..., expected_state_id=...)` supports explicit construction. Use the exact initial scene, mapping, and physics profile of the episode.

Omitting `actions` when passing a final engine or snapshot replays the accepted history only. In that case `attemptHistory` is `accepted_history_only_rejections_unknown` and rejected-action totals are `null`, because the engine's persisted accepted history cannot recover discarded attempts. The exporter cannot recover actions rejected outside the engine, such as an RL wrapper's episode-duration check, unless those outcomes are recorded separately; passing such an action to the engine is not equivalent to that wrapper rejection.

The command-line wrapper accepts the same saved case and a final engine snapshot:

```sh
.venv/bin/python scripts/export_surgical_telemetry.py \
  --case path/to/case.ressectionlab \
  --actions path/to/attempted-actions.json \
  --snapshot path/to/final-engine-snapshot.json \
  --label scripted --episode-id patient-rehearsal \
  --output outputs/rehearsal-telemetry
```

An actions file can be an array, an object with `attemptedActions` or `actions`, or an array of records containing an `action` field. `--snapshot` is optional for a fresh, unconfigured engine; include it whenever a saved episode used a bound guard contract. `load_environment` selects the saved workspace's mapping and physics profile. A snapshot from a different scene fails verification.

## What the numbers mean

| Fields | Units and provenance |
| --- | --- |
| `timeSeconds`, `observedTimeSeconds`, `observationAgeSeconds` | Simulated seconds. Reporting time, actual source-state time, and their difference. No wall-clock performance measurement. |
| Both `positionMm`, `quaternionXyzw` | Committed simulated tip poses, RAS+ mm and unit xyzw quaternion. |
| Both `active`, `latchedActive` | Effective working activation during that microstep, and persistent activation switch. A one-shot `apply` can be active without latching the switch. |
| `tipSeparationMm` | Euclidean distance between the two recorded tips, mm. |
| Both `pathLengthMm`, `repositionDistanceMm` | Sum of actual committed microstep displacement, and separate placement-jump distance, mm. No commanded-distance substitution when a drill stalls or motion stops. |
| Both `velocityMmPerSecond`, `speedMmPerSecond` | Vector finite difference over the actual microstep `dt`, and its magnitude, mm/s. |
| Both `accelerationMmPerSecond2` and magnitude | Difference of adjacent interval velocities divided by the separation of interval midpoints, mm/s². Initial or discontinuous derivatives are `null`; no smoothing. |
| Both `forceAssumedN`, `measuredForceN` | Declared simulator force parameter/default, N, and `null` for unavailable measured force. The assumed parameter does not assert actual contact or a measured tissue force. |
| `targetRemovedMm3`, `nonTargetRemovedMm3` | Removed source-cell counts × affine voxel volume, mm³, using explicit target annotations. Non-target includes access bone and other materials; it is not a healthy-tissue injury diagnosis. |
| `woundCount`, `activeWoundCount`, `modeledWoundEmittedMm3` | Model-generated wounds and output under declared research physics. `null` when a wound model is unavailable, rather than a claim of zero injury. |
| `fluidPresentMm3`, `fluidAddedMm3`, `fluidRemovedMm3`, `fluidEscapedMm3` | Simulator fluid accounting, mm³. Present volume includes blocked-outlet reservoirs. Fluid may include irrigation and assumed wound emission; it is not measured patient blood loss. |
| `contactCellSeconds`, `exposureByMaterialCellSeconds`, `vesselEncounterCells` | Engine contact/exposure accounting. Cell-seconds accumulate over cells and can exceed elapsed seconds; material exposure is not complete per-hand force sensing. |
| `invalidActionsCumulative`, `toolExchangesCumulative`, `contactStopsCumulative` | Attempted-engine-action rejections, successful exchanges, and contact-stop events. A partial action retains its completed microsteps. |

The observer subclasses the engine only to record **successful** `_substep` results. It copies its own recording state when the engine creates a transactional candidate. A rejected candidate therefore contributes no uncommitted poses, removal, fluid, or motion. Final replay identity is checked independently when a final engine/snapshot or expected identity is supplied.

The 0.2-second reporting grid uses the latest observed state at or before each tick. It never interpolates a force, pose, wound, or tissue state. Actual microstep times and action-boundary observations remain in JSON; each reporting row includes its source observation index and age. The exact final time is appended when it falls between grid ticks. Placement is a discrete free-space reset in this engine: its jump is recorded separately, and the affected velocity/acceleration is unavailable. Action-boundary events are explicitly distinguished from events observed within successful microsteps. The final boundary may repeat the last microstep's effective activation and derivatives; it does not represent a new motion interval.

The HTML comparison is descriptive. Equal labels or matching output columns do not make episodes comparable: scene identity, annotations, guard contract, starting state, physics, duration, and task objective must be matched for a controlled experiment. Clinical injury, patient risk, and operator expertise remain unknown. Learned expertise assessment would require a separately designed, consented, labeled dataset, validation protocol, and appropriate licensing.

## Source and reuse check

Checked on October 18, 2026:

* [Paper full text](https://europepmc.org/articles/PMC9042967) / [full-text XML](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC9042967/fullTextXML): the paper reports 0.02-second raw simulator acquisition and 0.2-second feature extraction, with interpolation in its preprocessing. The article is CC BY 4.0. Our reporting uses observed simulator microsteps with an explicit hold, rather than reproducing that preprocessing.
* [Author repository `recaiyilmaz/icems`](https://github.com/recaiyilmaz/icems): the released files `1-sequenceallmetrics` and `2-LSTM training` contain MATLAB-style preprocessing and training code. Repository inspection found no license file, and GitHub's API returned `license: null`. Code was consulted for feature/context understanding; no external code or trained weights were copied into this project.
* [Yilmaz (2021), sample raw CSV, version 1](https://doi.org/10.6084/m9.figshare.15132507.v1): Figshare metadata declares CC BY 4.0 and one 3,372,166-byte CSV, MD5 `12bc4ce3450140dbae2fbd16d1f62efc`. The inspected prefix has 10 metadata lines before a 55-column header, whitespace around fields, and an additional trailing whitespace field in data rows. Fields cover time, left/right instruments, switches, knobs, force feedback, contact/threshold counters, translations, quaternions, emitted/current blood, fiber cuts, and tissue volumes. Early timestamps are near 0.02-second spacing. These format checks do not establish unit compatibility or provide enough labeled trials to train an expertise model. No raw sample is redistributed or used as patient telemetry here.

## Verification

`tests/test_surgical_telemetry.py` covers real microstep velocities and acceleration, stale reporting rows without interpolation, placement discontinuities, one-shot and simultaneous activation, rejection rollback, partial contact stops, removal and fluid accounting, missing history, bound-protection snapshot replay, JSON/CSV export, and HTML script-delimiter escaping.

```sh
.venv/bin/python -m pytest tests/test_surgical_telemetry.py -q
```
