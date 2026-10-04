# Renderer contracts

The React shell owns case review, source visibility, route comparison, and local optimization controls. Numerical computation runs in the Python sidecar through the named `window.resectionApi` methods; the renderer never receives an arbitrary file path or executable operation.

- `App.tsx`: workspace state, operation status, native menu actions, case lifecycle and A/B route choices. A newer case load invalidates older hydration; route results must match the active source fingerprint.
- `case-data.ts`: validates array encoding, shape, physical affine, source volumes and finite values before exposing typed arrays to the viewer. It converts LPS+ physical coordinates to RAS+ without resampling source voxels.
- `viewer/`: linked WebGL MRI planes, exact source-cell annotation surfaces, complete instrument geometry and physical camera framing. The viewer has its own geometry and surface tests.
- `RefinementPanel.tsx`: actual optimization/selection counters, selection-return history, local cancellation/resume and independently accepted replay. Search choices stay in parent state throughout refinement.
- `training-data.ts`: checks replay identity, independent certificate, source frame, binary mask and volume accounting before allowing a modeled removal overlay. It does not approve imported simulation artifacts.
- `preview-api.ts`: explicitly read-only development adapter for an ignored local export of a real public case. It cannot train, search, save or substitute invented results. Patient preview assets are excluded from production packaging.

Source annotations and modeled removal remain separate. Static route accessibility is not removal. A replay is selection evidence, never final evaluation. Clinical deficit probability is unavailable. Unknown functional and vascular anatomy stays unknown.

Saved view settings can reopen. Scientific route results are restored only when the sidecar marks them as matching its current-session evaluation. Results reopened in a new engine session are withheld until a fresh search. Saved native runs require the sidecar's integrity and independent replay checks before a removal mask is returned.

Run renderer boundary checks with `npm run test:renderer`; run `npm run build` for the strict TypeScript and production bundle checks. Native Electron workflow checks are documented in the repository's packaging and experiment records.
