# Working-length sensitivity: one frozen-policy diagnostic

Changing only seven working-length descriptors reversed the frozen RL policy's STOP preference on the saved generated task. All other inputs, legal movements, weights and normalization stayed fixed. Independent saved-array checks verified both observation fingerprints and exactly seven changed values.

The actual-long-tool STOP-minus-best-movement margin was +0.466999. Replacing only the length descriptors with the original tool lengths changed it to −6.303272, while the STOP score itself stayed unchanged. The four movements common to the earlier original-tool experiment recovered exactly their original scores. STOP softmax changed from 60.9844% to 0.1731338%; these are model preferences, not clinical probabilities.

One forward completed after 24 counted geometric previews reconstructed the exact saved observation. Supervision took 1.8840 seconds, including imports; the diagnostic body took 0.1165 seconds. No strategy was executed, no training occurred, and no patient data was accessed. The altered descriptors do not represent certified physical tools and must never be used to claim an improved surgical plan.

This isolates working-length sensitivity at one fixed generated state. It does not establish a policy repair, calibrated abstention, robust tool generalization, physical fidelity, or limited-input patient transfer. The next implementation should preserve this regression evidence and test any proposed representation or training change under a newly frozen comparison.

[Independent audit](reviews/postrun/REPORT.md), [exact result](result.json), and [package inventory](package-sha256.json) preserve the original source and execution evidence. The 2,309-byte input fixture is explicitly simulator-generated; model weights and patient records are excluded.
