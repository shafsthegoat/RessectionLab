# Independent native acceptance

The 73-action `vascular-v2/vascular.ressectionlab` branch `full-vascular`
passed one independent forced canonical replay at frozen production commit
`eeeed4ea`, recorded in `vascular-v2-independent.json` and its action JSONL.
It published one authenticated cache entry. `vascular-cache-fresh-process.json`
then passed nine cache-only seeks in a fresh process, including rewind to zero,
the saved prefix, injury, untreated bleeding, hemostasis and the endpoint; no
public-action replay was repeated. Primary episode verification remains pending.
GUI review remains pending because the second monitor is unavailable; none of
these checks establishes GUI behavior.

After production code is frozen and the final path and branch are selected:

```sh
PYTHONPATH=src .venv/bin/python testcases/outputs/full-case-rehearsal-20261020/review/accept_native.py FINAL_NATIVE BRANCH --report testcases/outputs/full-case-rehearsal-20261020/review/final-acceptance.json
```

The default source is the exact standard sub-06 file with expected SHA256
`ff5a49b8706a8ccda08d8c3e82dad3e079f85dea98cb6bd40783d2262eec9838`.
The script uses the unchanged canonical `SurgicalEngine` with forced public
replay and cache reads disabled. A read-only proxy around the canonical cache
writer audits each action before allowing its genuine writer to capture it.
Successful canonical endpoint and checkpoint verification can therefore seal
the local cache in this same run. No second patient replay is necessary to
populate the cache. Native files remain independently replayable.

The receipt reports source/MRI/labels/annotation/affine/acquired originals,
per-action state transitions, partial physical outcomes and their exact
reasons, recorded uncommitted proposal outcomes, source-cell accounting,
six-connected residual regions, fluid conservation, wound-face geometry and
clustering, and severed nonpatency. A passed integrity audit does not mean the
episode removed all target: `episodeCompletionStatus` and retained target
volume are reported separately. Residual connectivity uses original source
ownership and does not establish that all further legal strategies were tried.
The patient audit also requires exact cells/provenance for the existing eleven
source-bound protection masks, 2 mm clearance and zero protected displacement.
It checks protected removal/displacement at every boundary, nonnegative fluid
volumes and valid committed physical fractions. Vascular acceptance observes
the actual time-ordered wound states: passive bleeding, increases of actual
face coagulation dose, idle persistence of full-dose sealing, irrigation and
suction that preserves damage, patency and historical blood loss. Initialized
source `coagulationExposure` is not treated as delivered bipolar energy.
The final audit also independently balances drill input/cutting/unused energy,
cleared-voxel work plus pending work, thermal generation/storage/cooling and
event-generated debris against the spatial/removed/escaped debris ledger.
Owned target and nonexclusive raw tumor-annotation percentages are separate.

After final canonical admission, `export_observed_telemetry.py FINAL_NATIVE
BRANCH NEW_OUTPUT --prepare-frames` uses an authenticated-cache-only baseline
and one observed public replay to export 0.2-second descriptive telemetry and
publish the same genuine observed frames. It fails if any action manifest is
missing at the end. `verify_every_cached_frame.py FINAL_NATIVE BRANCH
CANONICAL_RECEIPT REPORT` then uses a fresh process to hash every unique field
blob and seek every cached frame. Its `sys.setprofile` callback counts and
refuses any public physics step without monkeypatching the implementation.
This backend coverage check reports profiled seek timings and does not verify
the Electron renderer or GUI.

Engineering-only validation:

```sh
PYTHONPATH=src .venv/bin/python -m pytest -q testcases/outputs/full-case-rehearsal-20261020/review/test_accept_native.py --basetemp=testcases/outputs/full-case-rehearsal-20261020/review/engineering-pytest
```

Six engineering tests pass: actual native replay plus cache publication;
tampered checkpoint refusal with no sealed entry; wrong source hash refusal;
actual injury, persistent leak, bipolar sealing and wound geometry checks; and
overlapping brain/target masks with zero owned non-target removal but positive
raw brain-annotation membership among actual removed target source cells.
The wound fixture now also exercises actual irrigation/suction and rejects a
falsified endpoint with insufficient actual face coagulation dose.
The sixth fixture counts unique direct target/protection-mask membership while
retaining overlapping per-structure counts and the original target denominator.
`source-protected-target-membership-baseline.json` finds zero original owned
target cells directly in the eleven exact patient protection masks. This is
separate from the 2 mm swept-tool guard clearance and does not explain or
certify the remaining target's reachability.
The tracked director test adds an actual-engine `vascular_setup` handoff check
with clean withdrawn hands before a generic engineering observation replaces
the patient-specific rescue plan. It makes no patient-route claim.

The receipt calls the exclusive healthy/unknown count OWNED non-target material
removal, and separately reports removed source-cell membership in every raw
original compartment, source compartment, and brain mask. These nonexclusive
annotation overlaps do not change the owned target denominator and do not
establish clinical healthy-tissue truth. `raw-source-overlap-baseline.json`
records source-only counts from the unchanged exact source without replay.

`cache-field-measurement.json` reads the then-current 100-action native
checkpoint without replay: 254,712 full-state gzip bytes, of which 188,703 are
the flapComponents field and 41,200 the drilling field. Cache v2 stores repeated
identical top-level fields once within each authenticated entry. These endpoint
sizes are observations, not predictions that unobserved future states stay
unchanged.
