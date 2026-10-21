Use the repository virtual environment. This script only reads saved source/native/receipt files, derives selected branch counts and writes a new JSON inventory. It never constructs a surgical engine, replays physics or operates the GUI. Existing output files are never overwritten.

Default current archives and their selected `branchId`:

```sh
.venv/bin/python testcases/outputs/full-case-rehearsal-20261020/review/refresh_tool_inventory.py
```

Explicit separate archives:

```sh
.venv/bin/python testcases/outputs/full-case-rehearsal-20261020/review/refresh_tool_inventory.py \
  --primary-native PATH/primary.ressectionlab --primary-branch full-primary \
  --vascular-native PATH/vascular.ressectionlab --vascular-branch full-vascular
```

For one combined archive, pass its same native path to both flags with the two exact branch IDs. It is read once so both selections use the same captured archive version. The names `primary` and `vascular` identify report roles; completion status, stage, stop reason, tool actions and field/material ledgers come from each actual selected branch.

Optional `--primary-telemetry-receipt` and `--vascular-telemetry-receipt` explicitly identify existing observed-replay receipts. Exact source hash, scene ID, endpoint and complete action sequence must match. An explicit receipt from a separate pre-combination archive may support an identical branch inside the combined file; the report records that its whole-file checksum differs. Automatic adjacent receipt discovery requires the current native checksum. Missing/mismatched receipts do not confer observed effect proof. Events without a unique compatible effective instrument family, including stationary active hands, remain unattributed.

`--output NEW_PATH.json` chooses an unused destination. Default outputs have fresh UTC timestamps under this ignored review directory. Previous inventory receipts and the pre-CLI script copy are preserved. Source count/overlap and synthetic fixture qualifications remain explicit; no clinical force, safety, complete clearance or outcome claim is generated.

Validation on 2026-10-20: one default inventory run captured primary1220 actions/running and vascular73 actions/complete in `tool-family-inventory-current-20261020T163201Z.json`. CLI/default parsing, two distinct branches selected from one actual saved primary archive (`full-primary`, `exposure-prefix`), missing-branch refusal, and stationary active other-hand attribution were checked read-only. No second inventory variant or physics replay was run. Further saves will naturally change primary counts/status/ledgers.
