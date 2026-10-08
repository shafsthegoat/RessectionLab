# PAT05: two frozen checkpoint forwards

The trained RL256 checkpoint ranks **immediate STOP first out of 71 actions** on the unchanged historical PAT05 initial observation. STOP probability rises from **0.9884% to 81.3695%**. The initial checkpoint ranks STOP last. Both checkpoints execute successfully, but no action or planning endpoint was executed and no useful patient-planning performance was established.

| Saved output | Initial checkpoint | RL256 checkpoint |
|---|---:|---:|
| STOP rank | 71 / 71 | 1 / 71 |
| STOP softmax probability | 0.00988366734 | 0.81369459629 |
| STOP minus best tool logit | −0.370341435 | +5.368700624 |
| Entropy, nats | 4.261954308 | 1.271064997 |
| Uncalibrated critic value | 0.321926385 | 1.686328650 |
| Forward time, seconds | 0.381639750 | 0.190825875 |

These probabilities describe the model's preferences within this supplied inventory. They are **not clinical risk or success probabilities**. A STOP preference alone establishes neither safe avoidance nor poor clinical judgment; neither was tested.

The input is one previously consulted TRAIN observation, not a held-out case. Its full fingerprint matches the historical record: `sha256:645770594d7988980781df8324fb76ed347f47ff8d7452ca7d032d31efc93f90`. It contains a 64³ crop at source index `(3,106,119)`, a supplied tumor annotation and an unreviewed synthetic-trained SynthStrip support estimate. The reconstruction records all 11,437 source annotation cells in the crop. Independent scalar calculations from saved proposal coordinates place all 350 entry-to-tip sample points inside it; this says nothing about whole-tool safety.

The 70 non-STOP proposals were already filtered by the historical simulator. Access and the empty initial cavity are hypothetical. Annotation availability and review are unknown; motor, language, vascular and mechanical evidence remains absent or unassessed. The input retains horizon 3, compared with horizon 2 during generated training. These conditions prevent a claim of scan-only deployment or real-patient transfer.

Exactly **two forward attempts completed**, with unchanged checkpoint identities and zero native previews, executed actions or optimizer updates. The worker took **3.353384 s**; its supervisor took **5.180709 s**, with sampled peak worker RSS **1,549,369,344 bytes**, inside the declared 60 s / 2 GiB envelope. Reconstruction took 1.441588 s. The two forward clocks total 0.572466 s and are contained within the 1.110539 s load/forward/validation clocks. These clocks overlap the worker/supervisor totals and must not be added. Historical preparation cost **8.175236 s** was previously paid; acquisition and support estimation are also outside these forward timings. No end-to-end planning latency was measured.

The independent saved-output audit passed **830 checks** without decoding patient or checkpoint arrays or running a model. It verified 28 source files, ten original/snapshotted metadata authorities, the exact source-bundle and checkpoint bytes, historical reconstruction metadata, and all 71-action softmax/rank/entropy calculations. The outer output index directly lists 45 files. Two copied input files also named `output-sha256.json` were omitted by its basename filter; both remain pinned in the declaration and were separately verified. The completed run is preserved unchanged.

Bindings are recorded in `summary.json`. Audit receipt SHA256: `7fb0cdbc4f7263ec1a2b12557c2ba92f668aa4f5264e05d783da49a531a654cf`. The first audit-script attempt used the wrong historical episode-container shape; that tooling error and its correction are preserved separately and are not a diagnostic failure.
