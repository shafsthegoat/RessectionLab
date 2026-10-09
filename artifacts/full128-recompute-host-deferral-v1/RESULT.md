# Generated full128 recomputation: host preflight deferral

The reviewed one-shot generated 128³ recomputation release (contract SHA-256 `30ac21c7cf67af6f652453bb070ec10e55d88c806267af5f6ac6521fc9a61359`) **did not start a model child**. Its 30.038-second host preflight refused release before any checkpoint load or forward pass. The receipt SHA-256 is `be990511d339373c706c212c489f4cf9533846a6c4ebf0ac59b63398eedbea47`.

Seven host samples had kernel pressure masks `2,2,1,1,1,2,2`; the lowest available-memory reading was 38%, with an 11-point spread. The fixed preflight required mask 1 throughout, at least 45% available and no more than a 5-point spread. Swap use stayed flat and the two exact live acquisition processes matched their allowlist with no project-compute blocker. The actual refusal reasons were `kernel_pressure_not_normal`, `available_percent_below_45` and `available_percent_unstable`. There is no `recompute/` output directory or generated logit result.

This is **host deferral, not a model failure**. The earlier full128 pressure stops during forward remain separate negative evidence. The prospective rule to stop further full128 attempts after another during-forward pressure stop was not triggered here. No caps were changed, no retry or extra host probe was run, and this result does not speak to full128 numerical parity, patient performance or clinical use.

This compact package contains the frozen contract and 16 source/test snapshots, generated-input receipt, exact acquisition declarations, root release, preflight receipt and prospective/actual independent reviews. It excludes the generated NPY input, weights, logits, imaging and patient records; source and input hashes remain in the contract and receipts.
