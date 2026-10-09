# Generated 64³ baseline startup negative

The first baseline under the frozen v2 matched-control contract exited 1 before loading the GlioMODA checkpoint or running a model forward. `pair_worker.main()` reached its early sampler binding after one `before_imports` marker and raised `UnboundLocalError`: an in-function `from darwin_fast_sampler import FastDarwinSampler` later in `main` shadowed the top-level import. No `result.json` or `logits.npy` exists for this arm; the lazy arm was not launched.

The host preflight passed. The saved supervisor has no watchdog or post-guard reason, reports sampled peak process-group RSS 318,652,416 bytes and a clean finalizer with no residual. This is a worker startup defect, not a model or numerical outcome. The prior v1 monitor failure is separate and preserved in its original location.

The exact v2 contract and all source files named by `runner_source_sha256` are copied, along with direct baseline receipts and the independent negative audit. The separate erratum distinguishes the extracted checkpoint (250,184,062 bytes) from the approximately 3.3 GB published ZIP. This package contains no checkpoint, generated logits, patient image, patient annotation or bulk dataset.
