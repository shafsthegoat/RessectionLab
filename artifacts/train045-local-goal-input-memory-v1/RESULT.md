# Same policy math with lower input-assembly memory

The interrupted PPO campaign crossed its 4 GiB limit. Its exact peak allocation
site is unmeasured, but the policy constructed three large float operands before
concatenating them into the CNN input. The new implementation allocates the same
contiguous eighteen-channel tensor once, fills its slices, and releases dense
local references after their last use. Batch size, architecture, parameters,
masking, observations, rewards and checkpoint format are unchanged.

Four generated controls pass in 0.682 seconds. At batch4, the old and new paths
produce bitwise-equal constructed inputs, logits, values, loss, parameter and
raw-image gradients, and complete model/Adam state after one optimizer update.
Both frozen and trainable encoders are covered, together with mask/refusal and
input-integrity checks. Weak-reference checks confirm that the frozen path no
longer retains the dense input/output until the heads finish.

A separate fresh-process comparison at the actual batch4/64³ input shape gives:

| Assembly implementation | Peak self RSS |
| --- | ---: |
| Original |393,986,048 bytes|
| New |324,534,272 bytes|
| Reduction |69,451,776 bytes (66.234375 MiB)|

Both assembled outputs have the same SHA256. These two measurements cover input
assembly only; they do not include CNN workspace, the rollout buffer or the
native task. No full-training memory reduction or speedup is claimed yet.
The three bounded child jobs finish and are reaped in 2.720 seconds total;
no patient data, acquired checkpoints or native simulation are opened.

The actual receipt is saved with the source and test logs. Original policy
bytes remain in [the interruption archive](../train045-local-goal-curriculum-interruption-v1/original-policy.py).
The new policy SHA256 is
`029b97b545f4bec300ae5a8f9b038e9efe6f89dcb81334db6aa91c4233a5cc8a`;
the unchanged original is
`fce66f1de108591e57aadd55b37082930fafa4c2311bc4dfed3235f0b36d8ec8`.
The archived scripts expect the existing runtime stage layout. The next test is
the reverse512 checkpoint recovery under the unchanged 4 GiB limit; its actual
RSS and full-task outcome must be measured independently.
