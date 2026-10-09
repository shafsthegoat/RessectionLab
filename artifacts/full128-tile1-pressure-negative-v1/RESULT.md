# Full128³ generated tile1 feasibility: resource negative

One exact-reviewed full128³ tile1 CPU call ran under contract SHA
`1efe105ec62eac950b66641de707a7b020ec8b5d2569d1ca83dbdf162eaaa764`.
Its saved `tiled/supervision.json` reports **not accepted**, child exit −9,
watchdog and post-run reason `kernel_pressure_warning_or_critical`, elapsed
6.3273 s, 113 fast and 13 slow samples, and sampled peak group RSS
**2,489,253,888 bytes**. The last fast sample is the first mask 2 warning
(available 41%); recorded swap usage stayed unchanged. There is no
`result.json` or output logits. The original full128 tile4 attempt remains a
separate pressure negative at 2,394,734,592 bytes; neither attempt establishes
full-patch feasibility.

The finalizer signaled the identity-bound root and recorded same-PGID PID
95220 still present after its bounded wait, so its receipt correctly says
`residual_possible=true` and `manual_attention_required=true`. That PID was
absent from preceding fast samples; the independent auditor later found it
gone. Its type and relationship cannot be inferred from these records. The
download process was separately identity-matched in all 21 saved inventory
rows (30,672–40,640 KiB RSS) and remained running; the controller did not
signal it.

The worker log shows completed input/model hash checks, safe checkpoint load,
alias checks, strict weight copy and tile1 attachment. Immediately before
forward it recorded self RSS 591,072 KiB, near the old tile4 attempt's
591,312 KiB. Both logs end at `before_generated_forward`; the tile1 attempt
crossed 1.5 GB group RSS at 2.899 s and 2.0 GB at 4.082 s, then stopped at
6.327 s. These samples establish a forward-stage memory bottleneck, but do
not locate the exact layer. The later stop versus tile4 is confounded by host
conditions and does not prove a resource improvement at full128.

The earlier generated 64³ layer trace measured a 67.1 MB RSS increase around
the final decoder concatenation, consistent with its 64-channel full tensor.
At 128³ that tensor alone would contain 134,217,728 FP32 values (~537 MB).
Avoiding that materialized final concatenation is a targeted next hypothesis,
not an observed fix. It requires generated operator equivalence and matched
network parity/resource controls before another full128 call. No cap increase,
unchanged-run retry, patient inference, or clinical interpretation follows
from this negative.

Independent post-run trace audit `INDEPENDENT_REVIEW.md` (SHA-256
`0fcaf953235a0c02fd8814a69c1aad3df71b9902cb74d23e55bd7edd7f2eeba5`)
confirmed these negative findings.
