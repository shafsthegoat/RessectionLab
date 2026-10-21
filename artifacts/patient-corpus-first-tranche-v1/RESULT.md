# First corpus qualification and FK numerical adapter

This is a source/persistence and numerical software milestone after the actual
Electron rehearsal gate. It does not establish anatomical accuracy, tumor-growth
accuracy, clinical validity or publication readiness.

The resumable pipeline qualified all nine already named development/TRAIN people
with limitations and retained the same catalog hash on a nine-of-nine cached
resume. No additional people were acquired by that run. The searchable JSON and
SQLite catalog preserves seven brain/tumor cases, one brain/skull case, zero
accepted registered brain/vessel cases and zero complete same-person cases.
Location strata remain UNKNOWN. The [qualification receipt](../patient-corpus-v1/qualification-audit.json)
records the exact people, files, source hashes, masks, missing anatomy and counts.

Six BTC bundles reference verified external originals rather than embedding them.
Their brain estimates remain unaccepted; source-target/support conflicts are
recorded without clipping source labels. A source-bound estimate is not accepted
anatomical truth. Protected people and nested role/collection declarations are
rejected before image decoding or growth.

The FK adapter executes unchanged `get_D` and `FK_update` kernels from
[TumorGrowthToolkit 0.2 at the pinned commit](https://github.com/m1balcerak/TumorGrowthToolkit/tree/162c2cbbab5882fff7c3c05d9fedd87453298a0f),
with its MIT license and three exact source hashes. The wrapper supplies explicit
physical grids, source qualification, a masked seed, no-flux padding, bounded
positive steps, reaction/integral balance and correct time-zero/stopping records.
It retains continuous concentration and declared threshold masks. It does not
use private annotations to shape the lesion or claim that a contour is edema,
enhancement, necrosis or measured tumor volume.

Strict tissue mode requires attached source-bound GM/WM classifications.
`UNIFORM_DIFFUSION_DOMAIN` is a separate, explicit homogeneous geometry control;
it has no GM/WM classification or CSF barriers. Missing tissue maps do not pass
the planned tissue-conditioned host gate. A publisher-described healthy cohort
also does not certify an individual scan free of pathology.

Run or resume the frozen tranche from this checkout:

```sh
PYTHONPATH=src .venv/bin/python scripts/qualify_patient_corpus.py
PYTHONPATH=src .venv/bin/python scripts/qualify_patient_corpus.py --anatomy tumor
```

The default outputs stay local under `outputs/patient-corpus-v1/`; no model runs
or downloads occur. The source-model runs and IXI host construction have separate
receipts and are not silently added to this frozen nine-person catalog.

Independent review is recorded in [independent-qa.json](independent-qa.json).
Its initial failures identified same-grid boundary loss, floating-point fraction
rejection and protected-metadata bypasses; the final status is authoritative.
The numerical controls include an independent face-flux calculation, conservation,
boundary isolation and an analytic Neumann diffusion refinement check. These
establish numerical behavior only. A real-host overlay, GUI review and
tissue-conditioned/physiological validation remain separate work.

Validation: 139 relevant corpus, inventory, FK, persistent-episode and pilot
coordinate controls pass; three renderer provenance controls pass. The broader
legacy bridge files still report 27 failures and eight setup errors because
they expect closed synthetic/implicit-training operations. A read-only archive
of the preceding `78fc4f5` commit reproduces the exact same per-test statuses;
the [matched baseline record](legacy-baseline-comparison.json) retains that
limitation. This milestone does not claim a green repository-wide test suite.
