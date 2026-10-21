# Bounded BTC source recovery workspace

`scripts/prepare_btc_recovery_workspace.py` packages the exact recovered PAT22
source and a successful cached SynthStrip result for anatomical review. It
performs no network requests, inference, planning or optimizer updates.
Downloaded/generated app-testing data and detailed receipts belong under
Git-ignored `testcases/`; see that directory's local README.

The original acquisition manifest and seven source files retain their exact
release checksums. Preparation reuses the existing fractional-annotation importer,
including its physical-coordinate integer axis reversal. It independently
compares MRI, affine and thresholded annotation against those verified originals;
matching metadata alone cannot substitute different arrays. The cached model
must match the input, native grid, pinned weights and successful execution.
The brain estimate must be binary, nonempty, one connected component and contain
the declared target. These are engineering QC checks, not anatomical validation.

Use the existing source-preparation command before this packaging step. The
optional inference `--output-root` keeps new model outputs in `testcases/`:

```sh
PYTHONPATH=src .venv/bin/python -m resectionlab.poc_models \
  --image <recovered-T1.nii.gz> --case BTC-PAT22 \
  --model synthstrip-main --modality T1 --device mps --timeout 240 \
  --output-root testcases/<recovery-root>/inference
.venv/bin/python scripts/prepare_btc_recovery_workspace.py \
  --root testcases/<recovery-root> --inference <saved-inference.json> \
  --output testcases/<recovery-root>/PAT22-anatomy-review.ressectionlab
```

An existing workspace is never replaced. The packager checks a conservative
write reservation and total resident growth against the specifically approved
250 MiB recovery cap. This exception does not extend to other acquisitions.
Portable workspaces retain the original files, inference provenance and exact
source arrays and must pass native reopen.

PAT22 keeps its prospective population-training role. Its old scan-only contract
still withholds the source tumor from actors. The brain remains an unreviewed
model estimate; skull, vessels and function are not invented. The review
workspace has no surgical-task admission. A separate openly annotation-assisted
post-exposure contract must enforce the observed domain and declared exposure
corridor before this source can drive an episode. The current desktop shows a
planning refusal rather than treating missing evidence as safe.
