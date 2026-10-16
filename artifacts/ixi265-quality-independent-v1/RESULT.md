# IXI source-quality checks and retained tissue failure

The [independent audit](independent-qa.json), SHA256
`7a51ca88a898b11eb7fc132180d5f77da4a0bffaf49ceb6e5b8eb229d6d162a6`,
reproduces the fixed vessel set comparison and explains the first N4/Atropos
attempt's rejection. It does not accept vessel accuracy, tissue anatomy or a
virtual-glioma host.

## Fixed vessel comparison

The [prospective protocol](../poc-ixi265-vessel-agreement-v1/protocol.json) uses
the existing native vesselFM score threshold, strictly greater than.5, without
tuning, pruning, clipping or another inference. Within the declared native
coverage domain, the supplied reference has76,744 positive cells, prediction
182,108 and intersection71,379. Provided-positive-reference coverage is93.0092%;
descriptive Dice is.551504 and Jaccard.380743. Prediction in110,729 unlabelled
reference cells remains unknown, without a false-positive interpretation.

The public derived reference has model-assisted/manual-refinement provenance
and emphasizes major connected vessels/Circle of Willis. Complete negative
labels and whole-vasculature target equivalence are unestablished. The vesselFM
checkpoint includes IXI through CSD; individual training overlap is unresolved.
These observations cannot support independent, held-out or clinical accuracy.

An independent eight-bin set recount and sparse-neighbor traversal reproduce
all counts and the573/195 prediction/reference connected components. Component
counts are descriptive, without pruning or anatomical topology acceptance.
Native selected affines agree within.00000165mm; a source MRA qform/sform
discrepancy of about.00023mm is retained without rewriting either frame. The
[execution receipt](../poc-ixi265-vessel-agreement-v1/receipt.json) records1.93s
and.877GiB sampled supervisor/family RSS. All five inputs remain unchanged.

## Tissue failure and resulting dependency

After the three retained SynthSeg memory failures, the alternative uses the
official ANTsPy0.6.3 wheel in a separate Python3.12 runtime. Thirty-one hashed
binary wheels occupy95,770,415 bytes; installation occupies498,739,631 bytes.
The app's environment is unchanged. The
[setup record](../poc-ixi265-atropos-preparation-v1/asset-setup.json) and
[official/API provenance](../poc-ixi265-atropos-preparation-v1/official-provenance.json)
bind the runtime and source. N4 plus three-class Atropos is configured to fit this image; it is
not pretrained tissue inference or a ventricular segmentation.

The [first actual fitting outcome](../poc-ixi265-atropos-run-v1/outcome.json)
remains failed before N4 or Atropos ran. One declared attempt costs46.05s and
about.504GiB sampled combined RSS; no resource cap was breached, no candidate
was produced, and no retry occurred. Source images, brain mask and host bundle
retain their exact original bytes.

The independent source probe finds exactly eight zero-valued T1 cells among
1,451,509 estimated brain-mask cells. Every other input-validity predicate
passes. The eight cells are separated and surrounded by positive intensity;
their anatomical meaning remains unknown. They are not labelled CSF or repaired
by changing source values or the source mask.

A separately versioned prospective protocol may fit only the original brain
support with strictly positive acquired intensities while retaining the eight
excluded cells as UNKNOWN/ABSTAINED. That changes fitting support explicitly;
it cannot erase this failed attempt. Any later class posteriors remain fitted
membership probabilities, not physical tissue-volume fractions. Numerical
success alone will not accept anatomy or release tissue-conditioned growth.
