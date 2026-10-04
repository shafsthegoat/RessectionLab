# Renderer contracts

The React shell owns case review, source visibility, route comparison, and local optimization controls. Numerical computation runs in the Python sidecar through the named `window.resectionApi` methods; the renderer never receives an arbitrary file path or executable operation.

- `App.tsx`: workspace state, operation status, native menu actions, case lifecycle and A/B route choices. A newer case load invalidates older hydration; route results must match the active source fingerprint.
- `case-data.ts`: validates array encoding, shape, physical affine, source volumes and finite values before exposing typed arrays to the viewer. It converts LPS+ physical coordinates to RAS+ without resampling source voxels.
- `route-selection.ts`: preserves A/B comparison identity when one slot is empty; compacting visible routes never changes their labels or colors.
- `case-support.ts` and `StructuralEvidenceInventory.tsx`: keep estimated whole-brain proposals separate from usable research tissue support. Only the sidecar's explicit, source-bound support status can unlock hypothetical route generation; cortical access is never inferred.
- `viewer/`: linked WebGL MRI planes, source-derived annotation surfaces, complete instrument geometry and physical camera framing. Display meshes approximate voxel boundaries; quantitative volume uses original source cells. The viewer has its own geometry and surface tests.
- `RefinementPanel.tsx`: actual optimization/selection counters, selection-return history, local cancellation/resume and independently accepted replay. Search choices stay in parent state throughout refinement.
- `StructuralImportDialog.tsx`: imports the original image, native-grid proposed mask and extraction report through three main-owned file dialogs. Cancellation at any stage leaves the case unchanged. Import does not accept a proposal or grant cortical access.
- `refinement-readiness.ts`: checks engine preflight identity against the exact selected route entry, target, access window and complete tool, including LPS-to-RAS conversion. No legal initial cutting actions disables new optimization; the original route remains inspectable.
- `training-data.ts`: checks replay identity, independent certificate, source frame, binary mask and volume accounting before allowing a modeled removal overlay. It does not approve imported simulation artifacts.
- `preview-api.ts`: explicitly read-only development adapter for an ignored local export of a real public case. It cannot train, search, save or substitute invented results. Patient preview assets are excluded from production packaging.

For fractional source annotations, the binary target is labeled threshold-derived and the recorded threshold is shown in the evidence inspector. Structural proposal inventories display review status and annotation-exclusion flags without presenting those metrics as segmentation accuracy.

The current route-conditioned prototype learns STOP versus the declared fixed native stroke. It does not optimize a free-form trajectory or entry. Native-action alternatives are generated only through an explicit action, with their new hypothetical geometry visible and their search models recorded separately. Mixed-model retained sets do not constitute one shared Pareto front.

Source annotations and modeled removal remain separate. Static route accessibility is not removal. A replay is selection evidence, never final evaluation. Clinical deficit probability is unavailable. Unknown functional and vascular anatomy stays unknown.

Saved view settings can reopen. Scientific route results are restored only when the sidecar marks them as matching its current-session evaluation. Results reopened in a new engine session are withheld until a fresh search. Saved native runs require the sidecar's integrity and independent replay checks before a removal mask is returned.

Run renderer boundary checks with `npm run test:renderer`; run `npm run build` for the strict TypeScript and production bundle checks. Native Electron workflow checks are documented in the repository's packaging and experiment records.

## Hydration microbenchmark

`node --experimental-strip-types desktop/tests/benchmark-hydration.mjs <report.json>` (from repository root) requires the optional local public-case preview export. It checks all recorded source checksums outside the timed renderer work and reports three sequential warm-cache Node/V8 runs. This is a CPU/filesystem microbenchmark, not GPU or application responsiveness evidence.

For the 62,496,000-byte UCSF-PDGM-0004 export, replacing typed-array iterator/callback validation with indexed loops reduced median hydration-plus-cursor time from 351.65 ms to 134.23 ms in the first comparison (62% reduction). A subsequent source-checksum-verified run measured 113.36 ms. Both retain identical finite-MRI, binary-mask, affine, identity, byte-count and physical-volume gates; the cursor remained identical. Raw reports are under `artifacts/desktop-renderer/`. These development measurements are specific to this local machine and warm-cache workload.
