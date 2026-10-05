# RHUH fixed-plane display: preparation only

`scripts/render_rhuh_fixed_planes.py` prepares the previously declared three-plane, two-window display. **No patient rendering or image decoding has run in this preparation.** Tests and layout inspection use asymmetric numerical fixtures, not simulated anatomy or training data. A separate root release and one supervised execution are required.

Production input is fixed to the original RHUH-0001 visit-0 source-labeled T1, SHA256 `b3b9fa69b87221062261b8b16fbddfdac2c678104b354ba371b069fd784e3625`. The renderer pins the accepted saved inspection, parent/terminal records, independent saved-result audit and exact reviewed decoder source. Its default CLI reads metadata only. It checks the existing input request through the reviewed inspector, exact 240 × 240 × 155 float32 grid, 1 mm spacing and saved qform; no scanner-native/template identity is inferred.

The display is prospectively fixed:

- Stored-index planes **i=120, j=120, k=77**, full field of view.
- Upper row window **−8.976038932800293 to 10.492466926574707**, the saved global range; lower row **−3 to 3**.
- Original values remain unchanged. Window clipping is display-only; no normalization, masking, crop, anatomical overlay, segmentation or MRI volume regridding occurs.

Each image is a native plane transposed into plot row/column order. Horizontal/vertical voxel indices increase right/up; `origin='lower'`. Physical aspect is vertical spacing divided by horizontal spacing, exactly one for this image. Cardinal labels come from the qform axis directions. The actual L/P/S grid yields:

| Fixed plane | Display axes | Left / right | Bottom / top |
|---|---|---|---|
| i=120 | j / k | A / P | I / S |
| j=120 | i / k | R / L | I / S |
| k=77 | i / j | R / L | A / P |

Pixel-center indices and half-voxel field edges are explicit. Matplotlib uses nearest display rasterization; that does not mean PNG pixels are one-to-one MRI voxels. Annotation labels and the figure state sampled planes/display only/no anatomical acceptance. Pure helpers reject oblique affines rather than mislabeling cardinal edges.

APIs: `plane_views(volumeXYZ, affine, indices)` returns unchanged plane copies and orientation descriptors; `build_figure(...)` returns the scientific figure and descriptors. Execution rehashes the original bounded compressed snapshot before decoding. It reuses the frozen inspector, compares the complete new inspection with the saved report, then uses the same bounded reader to recover stored values. These are two decompressions of the same in-memory bytes, avoiding a new parser or edits to the accepted inspector. Original file identity and both implementation sources are checked afterward.

A root-created `resectionlab.rhuh-fixed-plane-release.v1` binds the renderer, decoder, inspection, independent audit, original compressed SHA, exact source, action `fixed_three_planes_two_windows_once`, `released=true` and a fresh output directory. Future argv:

```text
python scripts/render_rhuh_fixed_planes.py --execute --release <relative-release.json> --release-sha256 <exact-sha256>
```

Outputs are `fixed-planes.png`, `fixed-planes.svg`, and `receipt.json` under one new `outputs/rhuh-fixed-plane-qc-v1/run-*` directory. Existing directories fail before payload access. Failures retain their attempted outputs as unaccepted; no retry exists. Root supplies the existing 60-second/2 GiB sampled-worker supervisor and source/output hash checks; no additional launch framework was added.

Thirty-two owner analytical controls pass: exact asymmetric voxel mapping, full-field extents, signed/permuted axes, anisotropic aspect, raw values versus display clipping, source/release/grid refusals and bounded decoder reuse. An analytical PNG was visually inspected for legible labels/layout. A separate independent run passed seven scientific-display controls in 0.42 seconds; its frozen receipt is `artifacts/rhuh-fixed-plane-review-v1/verification.json`, SHA256 `de8469057049f5c27467fdd897fef0aa89587c88ba37a69171b8598e88aa0f48`. That receipt binds the earlier documentation hash; this status-only update followed review. Neither software tests nor the future three-plane view establish whole-volume anatomical validity, a working support mask, patient function or neurological-risk calibration.
