# Saved decision diagnostic, final revision

[DIAGNOSTIC.md](DIAGNOSTIC.md) and [diagnostic.json](diagnostic.json) cover all six prescribed selection pairs, representing three unique observed states. Weights, saved logits, ranks and value predictions changed; the winning action stayed the same. No exact 15-feature float32 aliases were observed among the 26, 24 and 22 legal non-STOP rows. This is distinct from separate synthetic examples where aliasing can occur.

This revision adds a strict nonempty/unique decision-ID gate and makes the absence finding explicit in prose. All scientific JSON fields are identical to [v1](../native-axis-decision-diagnostics-v1/diagnostic.json), apart from the reporting-script hash. The original v1 outputs and its exact executed script remain preserved. Existing pilot reports, source, checkpoints and receipts were not edited.

[verification.json](verification.json) records 26 passing owner/independent constructed-record tests, unchanged scientific fields, and byte-identical reproduction when only the compressed payload copies are present. [The exact standalone reporting script](source/report_native_axis_decision_diagnostics.py) is retained here. The [independent review](../native-axis-decision-diagnostics-independent-v1/audit-v2.json) recomputed every saved action rank, margin, logit/value delta and exact row identity, and verified source/checkpoint/authority bindings.

No model forward, simulator, geometry check, optimizer, random draw, gradient update, new world or performance experiment was executed. This post hoc descriptive result neither proves a cause of the unchanged selection score nor establishes clinical efficacy, representation sufficiency or future outcome equality. It chooses no new settings.
