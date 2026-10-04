# BTC PAT28 acquisition-physics audit

Checked October 4, 2026 against the source files pinned in `manifests/btc_acquisition.json`. This was a read-only source/code audit: no original headers changed, no derived correction parameters were written, and no FSL command ran.

## Readout-time finding

The reported `TotalReadoutTime=0.0266003` seconds is inconsistent with the BIDS/FSL effective-readout definition for the released images. The available metadata support a **separately recorded derived value of approximately 0.03610037 seconds**, with the original value retained and the discrepancy explained. This resolves the arithmetic/definition conflict; it does not validate distortion correction or prove every underlying scanner tag correct.

Both source JSON files report these values:

| Field | AP | PA |
|---|---:|---:|
| NIfTI reconstructed spatial shape | 96 × 96 × 60 | 96 × 96 × 60 |
| `PhaseEncodingDirection` | `j-` | `j` |
| `PhaseEncodingLines` | 96 | 96 |
| `BandwidthPerPixelPhaseEncode` | 27.412 | 27.412 |
| `EffectiveEchoSpacing` | 0.000380004 s | 0.000380004 s |
| `TotalReadoutTime` | 0.0266003 s | 0.0266003 s |
| `AccelFactPE` | 2 | 2 |
| `TrueEchoSpacing` | 0.000760008 s | 0.000760008 s |
| Converter version | `v1.0.20170724 GCC4.8.5` | same |

The released second image axis has 96 reconstructed samples; no guessed acquisition-matrix substitution is needed. The [BIDS MRI specification, stable 1.11.2](https://bids-specification.readthedocs.io/en/stable/modality-specific-files/magnetic-resonance-imaging-data.html#in-plane-spatial-encoding) defines the effective total readout using the reconstructed PE dimension. It also gives the Siemens bandwidth route to effective echo spacing. Applying those definitions:

```text
EES from bandwidth = 1 / (27.412 × 96)
                   = 0.00038000389123984637 seconds
effective TRT      = EES × (96 − 1)
                   = 0.036100369667785404 seconds

Using the rounded source EES directly gives 0.03610038 seconds.
Original TRT differs by approximately 0.00950008 seconds.
```

Do not divide the effective echo spacing by the acceleration factor again. The bandwidth-derived value already agrees with the supplied effective spacing; the source's `TrueEchoSpacing` is twice that quantity.

## Why partial Fourier is a plausible explanation, but not a missing override to guess

The [exact dcm2niix release tag](https://github.com/rordenlab/dcm2niix/tree/87d2142fff79f5d956f69f1b9f865cab2314c89b) corresponding to the reported converter version predates the September 2017 correction discussed in [the maintainers' issue 130](https://github.com/rordenlab/dcm2niix/issues/130). Its [`nii_dicom_batch.cpp` timing code](https://github.com/rordenlab/dcm2niix/blob/87d2142fff79f5d956f69f1b9f865cab2314c89b/console/nii_dicom_batch.cpp#L645) calculates total readout from effective spacing and `phaseEncodingSteps`, subtracting a fencepost equal to the rounded acceleration factor when accelerated. Thus:

```text
0.000380004 × (72 − 2) = 0.02660028 seconds
```

This reproduces the published JSON after rounding. Seventy-two acquired PE steps out of a 96-line reconstruction would be consistent with 6/8 partial Fourier. **The 72-step count and 6/8 fraction are inferred, not present as verified fields in the released sidecars.** The known legacy calculation is a strong explanation, not proof of the exact scanner protocol or absence of local converter modifications. Partial Fourier does not justify retaining the shorter value as the BIDS effective TRT.

Original DICOM headers/Siemens CSA information or the scanner protocol would establish `NumberOfPhaseEncodingSteps`, partial-Fourier fraction, phase resolution/oversampling/interpolation settings and the conversion lineage. Those inputs were not found in the acquired release subset. They would confirm the historical cause and permit reconversion; they are not required to calculate the BIDS-consistent value from the supplied bandwidth and actual reconstructed image dimension. Do not invent these missing fields.

The [FSL TOPUP guide](https://fsl.fmrib.ox.ac.uk/fsl/docs/diffusion/topup/users_guide/index.html#acqpdatain) explicitly distinguishes effective readout from a partial acquisition's physical echo train. The [TOPUP FAQ](https://fsl.fmrib.ox.ac.uk/fsl/docs/diffusion/topup/FAQ/index.html#what-if-i-have-some-old-data-where-i-am-unable-to-find-the-total-readout-time) explains cancellation of common scaling errors within its correction chain. That observation is not a reason to fabricate a number or label a field physically calibrated in hertz. A derived timing record should store source hashes, both source values, formula, dimensions, precision and citations.

## AP/PA geometry: mostly an in-plane prescription rotation

The files have matching shape and approximately 2.5 mm isotropic spacing, and both have left-handed LAS image axes. Their qform/sform pairs agree within numerical precision. They nevertheless sample **different grids**. Directly measured from their sforms:

```text
PA voxel coordinate -> AP voxel coordinate

 0.9999025161  -0.0139621790   0.0000000003   0.6609139882
 0.0139621782   0.9999025605   0.0000000013  -0.6656068951
-0.0000000007  -0.0000000023   0.9999990373   0.0000516657
 0             0             0              1
```

The relative orientation angle is approximately **0.800000 degrees**. The image-center displacement is **0.0246643 mm**; the largest corresponding voxel-center corner displacement is **2.3694961 mm** (2.3941786 mm using outside voxel-boundary corners). The corner value therefore does not establish a 2.37 mm patient translation. Headers specify sampling geometry, not measured head motion. Both files identify this same participant/preoperative session, but residual motion and image correspondence still require evaluation.

Overwriting the PA affine with the AP affine would lose this distinction. A physical-coordinate interpolation of the PA b0 images onto AP's native grid can account for the sampling difference, but it also changes the representation of the PE direction. A rigid image-to-image fit that absorbs susceptibility distortion would be a different operation and must not masquerade as header-based resampling.

## A concrete candidate treatment, requiring execution validation

Keep all 102 AP volumes on their acquired grid. Use an AP b0 from that series as the reference and first TOPUP image. Resample only the two PA b0 scalar volumes once into that reference grid through the recorded native affines, keeping the originals. Record interpolation and support loss. Do not simply declare the resampled PA direction to be `j`.

The PA PE unit vector expressed in the AP image basis is approximately:

```text
AP: [ 0,           -1,            0 ]
PA: [-0.01396218,   0.99990252,   -0.0000000023 ]
```

These are not perfectly opposite in that common grid. The out-of-plane component is at header floating-point precision; any projection to zero must be explicitly justified and recorded, rather than hidden by a broad tolerance. Both original affines are left-handed, so this case does not introduce FSL's extra first-axis flip when expressing the directions. An axis permutation/reflection on future inputs requires fresh direction bookkeeping. MRtrix's [phase-encoding documentation](https://mrtrix.readthedocs.io/en/3.0.8/concepts/pe_scheme.html) describes why image reorientation and PE metadata must move together, and why a sidecar belongs to its particular image.

An important implementation distinction comes from the current official source:

- **TOPUP** commit `25c18e445cbd602d464e80b2c8610e60b34fe74c`: its [table reader](https://git.fmrib.ox.ac.uk/fsl/topup/-/blob/25c18e445cbd602d464e80b2c8610e60b34fe74c/topup_file_io.h#L137) accepts unit vectors, and its [scan model](https://git.fmrib.ox.ac.uk/fsl/topup/-/blob/25c18e445cbd602d464e80b2c8610e60b34fe74c/topup_costfunctions.cpp#L55) supports both in-plane components while rejecting a nonzero third component. A general statement that TOPUP only accepts one nonzero component would be inaccurate for this inspected source.
- **EDDY** commit `c406fdcd66c01e6cf21c6e0501ce3c80bddda417`: its [acquisition-parameter class](https://git.fmrib.ox.ac.uk/fsl/eddy/-/blob/c406fdcd66c01e6cf21c6e0501ce3c80bddda417/EddyHelperClasses.cpp#L173) rejects oblique PE vectors for its input scans. Its [scan constructor](https://git.fmrib.ox.ac.uk/fsl/eddy/-/blob/c406fdcd66c01e6cf21c6e0501ce3c80bddda417/ECScanClasses.cpp#L1038) uses the acquisition-table row selected by each scan's index. Consequently, keep PA b0s confined to TOPUP; EDDY should process only native-grid AP data and index the AP acquisition row. Do not add resampled PA images as EDDY observations without addressing that restriction.

This is a source-supported candidate workflow, **not a tested result or authorization to execute/install FSL**. The actual installed versions, licensing gate, direction conventions and the TOPUP-to-EDDY handoff must be checked. First use a synthetic known-distortion/rotated-grid test to verify recovered displacement direction and scaling, then inspect the real correction and its residuals. Do not silently round away 0.8 degrees to pass the current preflight.

## Diffusion gradients and remaining uncertainty

PA contains only b0 images with zero diffusion gradients. Resampling those scalar references does not rotate any AP diffusion gradient. Keeping AP on its native grid also avoids an unnecessary pre-correction gradient transformation. Motion correction must subsequently provide and retain the correctly rotated AP gradients.

If weighted images are later reoriented or rigidly registered, transform gradient directions consistently with the image-basis change and physical rotation, including the FSL handedness convention; do not interpolate their three components as unrelated scalar intensities. [BIDS' gradient-orientation definition](https://bids-specification.readthedocs.io/en/stable/modality-specific-files/magnetic-resonance-imaging-data.html#required-gradient-orientation-information) specifies the image-axis relationship and first-component sign rule. The current AP native determinant is negative, so the additional right-handed first-axis inversion is absent here. Avoid applying both a manually rotated gradient table and the correction software's same rotation again.

Remaining unverified items are the original scanner/acquisition tags, inter-acquisition motion, image-correction quality, supported execution versions and downstream tract acceptance. None is resolved by a visually plausible FA image or a successful command exit. The acquisition's motor/language evidence remains unaccepted until the reconstruction and its independent QC gates pass.
