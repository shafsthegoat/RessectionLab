# Generated stage-zero skip recomputation prototype

The installed `PlainConvEncoder.forward` appends every stage result to its
returned `skips` list; the decoder requests stage zero at its last stage. This
small inference-only prototype runs stage zero a second time from the unchanged
input when that skip is requested. It removes the **Python tensor reference**
after stage one has consumed the first result. It does not claim that the CPU
allocator returns resident pages to macOS or that full128 pressure improves.

On a generated, untrained three-stage CPU FP32 PlainConvUNet with native
Conv3d/InstanceNorm3d/LeakyReLU, one forward produced logits byte-identical to
the unmodified network. The wrapper runs stage zero once for the encoder and
once on the decoder's single skip request. It asserts that a weak reference to
the first stage-zero tensor is dead before decoder entry. Input, state tensor
values, parameter/module identities and state-dict aliases remained unchanged.

The prior v1 prototype is preserved as a negative: it accepted a custom stage
one that mutated its input, yielding different logits. V2 added exact checks
for every encoder stage's operation/forward composition and forbids
module/global hooks at attachment and runtime. V3 also pins the encoder and
decoder objects, stage counts, and decoder bound forward at runtime. Five tiny
controls reject stage-one mutation, stateful normalization, dropout, hooks,
custom forwards and post-attachment graph drift before output.

A separate **untrained tiny-channel six-stage** control used generated input
`[1,2,32,32,64]`, channels `[4,4,4,4,4,4]`, two Conv blocks per encoder and
decoder stage, and bottleneck `[1,1,2]`. Both arms used the same tiled Conv3d
and lazy final decoder. Baseline and recomputation outputs `[1,3,32,32,64]`
were byte-identical (raw SHA-256
`542fc82ef143c34eb9fc4a3776b5a92f8e86c56c6a17f0737cf7968635360f34`).
Input, state values, parameter/module identities and aliases were unchanged.
The one generated child exited 0 in 0.881 seconds, with 34 sampled RSS points
peaking at 226,410,496 bytes under the predeclared 512 MiB / 20-second bounds;
the host preflight had normal pressure and 51% available memory. These are
software graph and resource-guard results, not trained-model performance. Raw
logit arrays were not retained, so exact parity is a worker-enforced assertion
from the reviewed source rather than an independent array reload. The one-off
tiny supervisor's success path reaped the child, but its failure paths do not
guarantee cleanup after uncertain launch identity or wait timeout; it must not
be reused for trained-model controls.

The pinned six-stage **trained** architecture has not been loaded or run with
this prototype. No full128 or patient inference has occurred. Repeated stage
zero adds compute; the single-run tiny forward timings are not speed evidence.
Module hooks are disallowed, so any later resource experiment needs separately
reviewed instrumentation that preserves this hook-free condition. No physical
memory savings are claimed from the weak-reference result or sampled tiny RSS.
