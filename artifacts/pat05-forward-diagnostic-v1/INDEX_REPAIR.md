# Future output-index repair

The completed run and its source snapshot remain unchanged. Its outer index excluded
all files named `output-sha256.json`, unintentionally omitting two inherited input
indices. Independent review checked both against the frozen declaration; scientific
outputs are unaffected.

The current runner now excludes only its own root index and retains nested indices.
A pure filesystem regression verifies inclusion, content digests and idempotent
rewriting. The prospective manifest is refreshed for that source change; the
completed run retains its original `declaration-input.json`. No patient or model
execution was repeated for this repair.
