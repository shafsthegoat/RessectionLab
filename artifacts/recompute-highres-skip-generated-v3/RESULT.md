# Generated stage-zero skip recomputation: scoped software evidence

V1 demonstrated the core idea on a tiny pure graph but allowed a custom
stage-one forward that mutated the retained skip; the independent
counterexample changed logits and put trained use on HOLD. V2 added exact
encoder operation/forward and hook guards. V3 added runtime encoder/decoder
identity and stage-count checks and one bounded untrained six-stage control.

On generated input `[1,2,32,32,64]` with six encoder stages of four channels
each, two Conv blocks per stage, bottleneck `[1,1,2]`, tiled Conv3d and the
reviewed lazy final decoder in both arms, the V3 worker asserted byte-identical
baseline and recomputed logits `[1,3,32,32,64]`. It checked unchanged input,
state values, parameter/module identities and aliases. The worker exited 0;
34 RSS samples peaked at 226,410,496 bytes under the predeclared 512 MiB and
20-second bounds. Five additional tiny generated tests passed. The full source,
worker log, supervision receipt and independent audit are copied here.

Raw logits were not retained, so the exact comparison is worker-enforced, not
an independent array reload. The one-off tiny supervisor's failure paths are
not suitable for a trained-model run. No measured memory saving, speed result,
patient inference, trained policy result or clinical claim follows. A later
trained 64³ comparison would need the reviewed durable controller, retained
arrays, matched host guards and a separate independent prospective review.
