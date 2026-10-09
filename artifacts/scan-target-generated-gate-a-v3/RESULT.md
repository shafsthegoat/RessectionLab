# Generated CPU Gate A v3: memory-pressure stop

The revised monitor worked without the v2 timeout defect, but the one declared CPU control did not finish. Seven preflight samples were normal (kernel mask 1, available 48–53%, no swap-used increase). During the first forward pass, the direct kernel reading changed to **mask 2, warning**, at 2.015 seconds; the post-child check also reported warning. A later independent sample returned normal. This is observed global host pressure during this workload, not proof the model alone caused it.

The controller stopped its model child (exit −9) after **2.063718 s**, with **1,428,946,944 B** peak sampled resident memory, below the separate 3 GiB RSS cap. The worker reached `before_generated_forward`; no completed logits/result exist. There were no monitor, slow-inventory or overlap errors. The exact allowed ReMIND downloader continued unchanged. Source, checkpoint and generated input hashes remained fixed.

CPU was not accepted for the paired comparison, so **no MPS model call ran**. This does not measure prediction accuracy, numerical disagreement or patient transfer. Prior v1/v2 failures remain unchanged. The next bounded engineering investigation reduces Conv3d temporary workspace with full-context output-depth tiles, first comparing small generated kernels; it changes neither this outcome nor the declared model criteria.

The exact executed controller/worker/comparator source bytes are preserved under `source/` with original paths and SHA-256 in `source-manifest.json`. These are historical snapshots, not a newly installed runner; restore the recorded original layout and pinned optional runtime to reproduce the declaration. Arrays, checkpoint weights and patient data remain outside Git.
