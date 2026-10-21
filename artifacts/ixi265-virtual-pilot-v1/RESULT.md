# IXI265 source-preserving anatomy and virtual-overlay integration

The actual Electron app opens the IXI265 T1 brain estimate and native MRA vessel
estimate as separate same-person sources. A generated FK lesion on the unchanged
T1 also executes the existing two-tool rehearsal, branches from a nonempty saved
state, plays/scrubs and reopens at the selected STOP frame in a fresh process.
This closes a source/persistence integration check, not the tissue-conditioned
growth, anatomical accuracy or physiological validation gates.

## Genuine imaging and model runs

IXI:265 retains its frozen TRAIN role with `training_admitted=false`. The
publisher describes the source cohort as healthy; no individual pathology review
or tumor-absence certificate is claimed. T1, MRA and original source vessel
reference bytes are retained. The reference is display-only and never supplied
to the vessel model or used to shape the virtual lesion.

| New run | Actual result | Qualification |
| --- | --- | --- |
| SynthStrip on native T1 | Successful; about 4.0 s worker/monitor, 0.85 GiB sampled RSS | Brain envelope estimate; local anatomical review pending. Metal allocator/driver measurements overlap RSS and are not summed. Earlier failed device/CPU attempts remain recorded. |
| Rigid T1–MRA candidate | Both directed transforms and acquired sampling coverage retained | Unreviewed. About 78.4% of the estimated brain has mapped MRA sampling; this is not vessel completeness or registration accuracy. Fusion stays disabled. |
| vesselFM on native MRA | 98 tiles; 58.21 s; 1.99 GiB sampled RSS;185,231 positive native cells | Research/noncommercial checkpoint; IXI occurs in its training lineage and subject overlap is unknown. Partial coverage and extracranial/noise-like predictions remain visible. No held-out accuracy claim. |
| SynthSeg1.0 CPU | Three measured trials stopped at the 4 GiB sampled-memory guard | No tissue/ventricle output accepted. Full-FOV, brain-preserving ROI and official fast-ROI attempts remain in the denominator. |

The vascular candidate preserves the primary MRI/brain/frame, both reference
sources and all 16 original files; it adds a separate native model-estimate source
and 8 archived model/coverage records. The mapped T1 estimate remains an archived,
unreviewed candidate rather than an accepted policy constraint. A fourth large
display source would exceed the existing attachment budget and was not added.

## Generated lesion and executable episode

The pinned TumorGrowthToolkit FK kernels run 60 one-day steps on a 55×62×56,
3 mm physical grid. The seed at RAS (−21, 21, 36) mm is a declared, unreviewed parameter
placement; its lobe and hemisphere labels remain UNKNOWN. Diffusion and growth
parameters are uncalibrated. The homogeneous domain has no GM/WM classification
or CSF barriers, deformation, mass effect, edema, vessel response or functional
risk. Acquired MRI signal and anatomy stay unchanged; this is a generated overlay.

The 0.1 contour contains 33 coarse cells / 891 mm³ and maps to 841 native cells /
886.9917 mm³, with no outside-brain exclusions. The 0.5 contour is empty. Those are
chosen scalar contours, not histological or radiological tumor compartments.
The [independent numerical/source audit](independent-qa.json) reproduces the
saved concentration within 5.55×10⁻¹⁷ and verifies every original/source array.

After one actual 1 mm aspiration step, both continuations start from that same
nonempty 1 s state:

| Generated episode outcome | Short pass + field management | Deeper pass + reassess |
| --- | ---: | ---: |
| Target removed, mm³ | 14.8 | 85.4 |
| Non-target removed, mm³ | 44.3 | 68.6 |
| Residual target, mm³ | 872.2 | 801.6 |
| Dimensionless field burden | 0 | 0.292 |

Initial exposure 16,968.9 mm³ is assumed and separately accounted. The rounded
oblique source affine uses the runtime's conservative cell cover; these counts
are not exact physical instrument/tissue intersection measurements. Vessels and
motor/language function remain unknown in this episode. Its source is one real
host with one generated lesion, adding zero patients.

## Actual desktop verification and remaining work

Native saves keep separate reviewed copies. The vascular view reopens with the
native model estimate selected and its saved opacity; the generated case reopens
on deeper branch frame 21/21, STOP 7 s, with playback paused. The file persists
branch/frame/tool selection; pausing is the frontend's reopen behavior.
Linked MPR target centering now prefers tumor layers over a larger brain envelope.
Generated labels stay distinct from source annotations, and scan choices
distinguish source labels from model estimates before selection.

The [pilot receipt](pilot-receipt.json), separate final audits and GUI evidence
inventory bind the exact local bundles, model/code hashes, screenshots and saved
state. Bulk imaging and weights remain outside Git. Relevant validation includes
139 source/runtime/pilot controls, 14 model/geometry controls, 361 renderer tests
and the production build. The retained legacy-test limitation is documented in
the [preceding corpus result](../patient-corpus-first-tranche-v1/RESULT.md).

The searchable frozen catalog still contains nine people; IXI265 is a separate
newly constructed candidate, making ten named source identities across these
outputs, not ten validated or statistically independent patients. Reviewed
location coverage, portable BTC originals, individual host review, tissue maps,
registration landmarks, vessel accuracy and the tissue-conditioned pilot remain
open. The [14-phase operative workflow](../../docs/operative-workflow-model.md)
guides later development; this implementation begins after assumed exposure.
The full-operation, high-accuracy and publication goals are not complete.
