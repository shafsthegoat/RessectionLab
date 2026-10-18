# Standard surgical research environment

New desktop environments use `resectionlab.surgical-environment.v1`. This contract is shared by the general instrument scene, native-study preparation, segmentation/mesh imports, and scene descriptors consumed by the RL wrapper. It declares research mechanics and source coverage; it does not establish anatomical accuracy or clinical suitability.

## Required declarations

| Field | Contract |
| --- | --- |
| Physical frame | RAS+ coordinates in millimeters; an explicit affine retains handedness, obliquity, and voxel dimensions. LPS source data convert once. |
| Grid | Shape, affine, spacing, and determinant-based cell volume. Native-study preparation uses a declared 1 mm grid; directly imported segmentation retains its native grid. |
| Anatomy | Brain, tumor, bone, and vessel availability each read `supplied` or `unknown`. Supplied does not mean complete or reviewed. No anatomy is synthesized to fill a missing role. |
| Regions | Stable IDs, names, declared materials, target flags, colors, and explicit ownership precedence. Original overlapping memberships remain source annotations. |
| Target | Selected region IDs and initial owned target volume. The default task selector is shared with bounded rehearsal. Explicit `rehearsal_target` metadata or user mapping overrides defaults. Historical source adapters retain their established enhancing/core target selection; FLAIR abnormality is not silently added to that task. |
| Physics | Explicit V2 research profile, schema, and content hash. Full profile parameters are stored in each new scene archive, so restoring history does not depend on later defaults. |
| Provenance | Source hashes, source grids, selected layer IDs, registration status, sampling assumptions, and unknown coverage. |
| History | Source-bound scene identity, complete branch snapshots, branch ancestry, current position, and full action timeline. Previous mappings remain recoverable through Previous environment. |

`SurgicalScene.descriptor().environment` exposes the contract. `surgical_environment.standardize` creates a separate scene, leaving the input scene and source arrays unchanged. Existing saved scenes retain their original mechanics and replay identity until the user explicitly creates a standard environment.

## Preparation

The CLI `scripts/prepare_surgical_patient.py` uses the same preparation path and writes `patient.ressectionlab`, `report.json`, and a versioned `environment.json` containing the physical grid, mapping, complete physics profile, coverage, and source-bound identities. Only supplied role flags are required.

For a native study, open **Rehearse**, choose existing brain/tumor/bone/vessel layers, and use **Prepare standard environment**. Unavailable roles remain unknown. The reference scan determines the grid independently of the currently displayed CT/MRI scan. Estimated alignment requires an explicit research choice; preparation copies the unreviewed status rather than approving registration. Source manifests and originals are not rewritten.

For a saved research case or imported segmentation, use **Prepare instrument scene**. A segmentation-only case has no acquired intensity image; its zero-valued placeholder is retained without inventing MRI signal. Assign arbitrary imported labels through Material mapping before expecting tissue-specific behavior.

The reference display preset uses translucent gray brain, purple target, pale transparent bone, and opaque magenta vessels. Colors and opacity are display choices derived from declared material, independent of labels, patient identity, and simulation measurements. **Anatomy visibility and appearance → Reference appearance** restores the preset.

The general instrument environment and legacy bounded episode are exclusive viewer modes. Their histories remain separate. Legacy research has additional source-admission requirements and does not gain anatomy or capabilities from the general environment.

## Verification

`tests/test_surgical_environment.py` checks nonorthogonal/anisotropic physical frames, target selection, missing anatomy, no-op exchanges, unchanged mappings, full timeline recovery, and partial native preparation. The integration tests verify save/reopen in a fresh process. Renderer checks cover segmentation-only display windows and material-driven appearance. `SurgicalRLEnv.from_patient` and the protection-manifest preparation script use `load_environment`, so the saved desktop mapping and physics are identical in RL; reset starts with intact anatomy rather than a saved action prefix. RL contract/guard tests continue to exercise the existing wrapper and protection behavior.

Missing vascular/function evidence remains unknown. The standard format makes cases interoperable; it does not make incomplete cases equivalent to the four-material meningioma reference or grant them clinical/RL training eligibility.

## Standardize a local library

`scripts/standardize_case_library.py PLAN.json` executes an explicit
`resectionlab.local-standardization-plan.v1` plan. Each case declares its patient
ID, native collection, output folder, role mapping, and optional model jobs.
For new collections, `series` binds each original NIfTI to a declared modality;
`layers` binds categorical masks, label values, source series and provenance.
Any cross-series registration must be declared explicitly. Existing collections
and original saved rehearsals are preserved. Distinct saved versions of the same
person do not become new patients, and unavailable catalog members are listed
separately rather than reported as tested.

Model jobs use the existing bounded, hash-pinned reconstruction adapters. A
failed model does not erase its logs or prevent other anatomy from being
exported. Empty tumor outputs are retained as failed localization evidence and
are never mapped as a usable target. Experimental T1CE vessel proposals remain
distinct from acquired angiography. Optional SynthSeg `crop` is an explicit
128–256 multiple of 32 on its 1 mm inference field; the crop participates in the
cache identity and its limited coverage is recorded in the model provenance.
Unlabeled regions outside that field are unknown.

The batch writes a checkpointed `readiness.json`, standard bundles, environment
descriptors, provenance reports, and scene-bound RL exclusion manifests. The
eleven exclusion identities use explicit SynthSeg numeric labels. A strict RL
reset and inspect transition test the actual observation contract without
relaxing missing-critical-anatomy guards. Runtime compatibility, material
coverage, and reviewed training admission are separate results; passing reset
does not establish anatomical completeness or patient independence.
