# Generated in-place InstanceNorm candidate: numerical HOLD

The full128 lazy-concat run stopped during the encoder, where full-map
InstanceNorm contributed large observed RSS increases. A generated-only CPU
FP32 prototype therefore normalized a private tensor in place using full-map
per-case/channel biased variance and the same affine and epsilon settings as
the pinned network. This is a numerical experiment, not a model adapter.

Five tiny tests pass, including a deliberately negative counterexample. On
ordinary zero-centered generated maps, maximum difference from native
InstanceNorm was at most `9.54e-7`; on low-variance maps with a large positive
offset, the native and prototype FP32 reduction/operation orders diverged.
At offset `1e6` with noise scale `0.1`, maximum absolute output difference was
`1.7628903`, with `43` activation sign disagreements. A detached tensor alias
also evades the prototype's view check, so exclusive input ownership has not
been established for a network attachment.

Independent review issued HOLD for model use. There was no checkpoint, patient
input, trained-network forward, resource-benefit measurement or clinical
claim. The prototype remains ignored and unintegrated; the negative result is
retained to prevent a mathematically plausible but unvalidated memory change.
