# Working-length input sensitivity

Changing only four non-STOP working-length descriptors from 2.2/12 mm to 120 mm
flips the trained checkpoint from movement to STOP on the same generated input.
The independent saved-output audit passes all 311 checks.

| Checkpoint | Original STOP preference | Changed descriptors | Change in STOP-minus-best-movement score |
|---|---:|---:|---:|
| Initial | 16.4203% | 15.3756% | −0.069647 |
| RL after 256 updates | 6.7249% | 99.3915% | +7.956279 |

The trained STOP score stays exactly unchanged while the four movement scores
fall by 7.18–8.05. The image, state, five action IDs/order/masks, rays and all other
geometry values are unchanged, as are both checkpoint identities. The changed
input has its own fingerprint; its inherited source identity records ancestry
rather than certifying modified physical actions.

Exactly two new frozen forwards completed, with no new baseline calls, previews,
actions, optimizer updates or patient reads. Worker time was 2.364120 seconds;
supervised time was 3.836943 seconds; total launch time was 4.674622 seconds.
These clocks are nested. Sampled peak RSS was 249,380,864 bytes, within the fixed
20-second / 1-GiB envelope. All 44 indexed outputs, 29 source files, seven metadata
inputs and checkpoint bytes were independently checked. The arithmetic audit
ran no model and did not decode patient or checkpoint arrays.

This isolates descriptor sensitivity on one familiar generated root. It does not
establish that STOP is wrong for an actual long tool or explain every PAT05 input
change. The longer tools were not physically recertified; their true legal actions
and search optimum may differ. These probabilities are model preferences, not
clinical risk or success probabilities.

The next comparison must create actual tool geometries and recertify each action
inventory before testing search, imitation and RL. A future normalization ablation
requires a new architecture version and fresh fitting; it cannot reinterpret this
checkpoint. Substantial training remains behind the validated-mechanics priority.

Preparation passed 21 focused controls and eight independent controls. Two genuine
terminal-reporting failures are retained from before the fix, alongside the owner’s
initial test-fixture failure. No failed experiment or unreported retry occurred.
