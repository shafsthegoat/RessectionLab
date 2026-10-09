# Generated CPU control v2: monitoring failure

The prospective quiet-host preflight passed, then the CPU control stopped before producing logits. Its full process-inventory command exceeded the fixed **0.1-second timeout** during the forward. The supervisor killed the worker after **1.456 seconds**, with sampled peak RSS **1,038,096 KiB**, below its 3 GiB limit. No MPS call or parity comparison ran.

Separately, the system pageout counter increased by **58 pages (0.90625 MiB)**, triggering the frozen any-increment hold rule. Recorded kernel pressure stayed normal; the available-memory indicator fell from 53–54% in preflight to 48% afterward; swap usage and swapout count stayed unchanged. These global counters cannot be attributed solely to the model. This is a monitor failure and host-trend hold, not a numerical model result or evidence that the CPU prediction was wrong.

The [independent audit](INDEPENDENT_REVIEW.md) confirms the child group is gone and frozen sources/input/checkpoint are unchanged. [Supervision](supervision.json), [preflight](host-preflight.json), [worker phases](worker.log), [prospective contract](contract.json), and [pre-execution review](PREFLIGHT_REVIEW.md) preserve the outcome. The original v1 compressor-growth failure remains separate.

Next: benchmark a lower-overhead monitor before another model call, separating fast process/OS-pressure checks from slower process inventory and using evidence-based counter semantics. No v2 retry or retrospective threshold change is permitted. This remains a generated-input test of a pretrained segmentation component, not RL training, patient inference or clinical validation.
