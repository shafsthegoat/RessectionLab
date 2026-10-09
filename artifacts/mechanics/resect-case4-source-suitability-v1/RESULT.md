# Case4 estimated-envelope suitability audit (read only)

**Decision: NO-GO for another Case4 mesh-optimization attempt now.** This is a
source-suitability and prioritization decision, not a retroactive rejection of
the previously authorized bounded geometry attempts. Metadata-only analysis of
the saved v4 failure is reasonable; another local/global Gmsh candidate and a
Case4 FEM solve should wait for the source and solver gates below. A mesh that
approximates the current surface perfectly would still not establish a valid
retained-tissue domain for conditional brain-shift prediction.

I reviewed only committed documentation and JSON/Markdown receipts. I did not
open patient image, mask, surface, ultrasound or landmark payload arrays; I did
not view the PNGs, parse destination measurements, rerun geometry, or edit
tracked files. This is an audit of *recorded evidence*, not independent
recomputation of anatomy or physical correspondence.

## What the receipts establish

- The real Case4 T1 is the source for one unchanged SynthStrip main-v1 native
  estimate. Independent saved-array QC reports a finite, single-component
  1,186,021-voxel mask with no field-of-view face contact; its mask and SDT
  exactly share the original 256 × 256 × 192 T1 grid/affine, and the fixed
  reconstruction of the upstream mask has zero voxel mismatches. These are
  integrity and extraction-reproduction checks, not segmentation truth.
  Sources: `artifacts/mechanics/resect-case4-brain-envelope-v1/completed-inference.md`,
  `artifacts/mechanics/resect-case4-brain-envelope-independent-qc-v1/README.md`
  and `qc-receipt.json`.
- The retained native surface is recorded as one closed component, 91,951
  vertices, 183,902 triangles, Euler characteristic 0/genus 1, with volume
  close to the binary mask. Its genus is an observed computational topology;
  neither an anatomical handle nor a defect has been identified. Do not repair
  topology simply to ease Gmsh. Source:
  `artifacts/mechanics/resect-case4-patient-mesh-v1/summary.json`.
- A nonexpert review of three fixed native T1 planes saw a broadly outer-cerebral
  contour with no gross array shift in those views, **and visible
  posterior-inferior/cerebellar tissue outside the mask**. It allowed bounded
  meshing of the *estimated* domain only; it expressly withheld whole-brain,
  pial, cortical-access and clinical-anatomical acceptance. Source:
  `artifacts/mechanics/resect-case4-brain-envelope-independent-qc-v1/root-visual-decision.json`.
- The separate source-only landmark partition records 19 unique source points,
  six spatially selected B inputs, 13 V rows and a nondegenerate B rank ratio.
  All source points were within the **before-US image grid** under the
  creator-described world interpretation. That check does not establish
  inclusion in the T1-derived mask, adequate 5 mm kernel support, or accurate
  MRI-to-US correspondence; the partition receipt marks frame QC unaccepted.
  Source: `artifacts/mechanics/resect-case4-source-partition-v1/README.md`.
- The 15-pair MRI/before-US proper-rigid baseline diagnostic is reproducible and
  has 1.164 mm fit RMS and 1.295 mm leave-one-out RMS. Its visual review found
  broadly corresponding MRI structures, but the partial oblique US field could
  not verify local anatomical correspondences. Both the numerical receipt and
  root visual decision retain `anatomical_alignment_accepted=false`; residuals
  are internal fit diagnostics, not independent registration accuracy. Sources:
  `artifacts/mechanics/resect-case4-baseline-alignment-v3/comparison-and-outcome.json`,
  `artifacts/mechanics/resect-case4-baseline-alignment-v1/root-visual-qc.json`.
- Four geometry attempts are terminal. The most recent 6/24 mm boundary/interior
  candidate passed the recorded topology, Jacobian, overlap and global-volume
  checks, but failed the unchanged 2 mm bidirectional surface-fidelity gate:
  sampled maximum gaps 3.869 and 2.264 mm, with full-surface upper bounds
  4.576 and 3.254 mm. The 19,070-node result also projects 39.28 GB for three
  conservative Skyline value arrays before other allocations; that is not a
  measured sparse-solver requirement. Sources:
  `artifacts/mechanics/resect-case4-patient-mesh-boundary6-v4/RESULT.md`,
  `artifacts/mechanics/resect-case4-patient-mesh-boundary6-v4/next-bottleneck.md`.

## What is still missing for the narrow displacement proof of concept

1. **Domain suitability:** a prospective, source-only anatomical review of the
   fixed estimated surface throughout the region of interest, especially the
   known inferior exclusion and the unexplained genus-1 handle. It must state
   whether the domain is a defensible *local retained-tissue approximation*,
   not call it a true pial or whole-brain surface. A preoperative T1 envelope
   contains no verified during-resection cavity or removal history. Neither
   image-grid containment nor a small total-volume error resolves this.
2. **Frame and observation support:** independent acceptance of T1/FLAIR to
   before-US physical alignment; then a separately released source-only check
   of each observation's containment, signed distance to boundary, 5 mm kernel
   support volume and first moment, with unsupported cases counted under frozen
   rules. Do not enlarge the mask or choose coordinates/mesh sizes from V
   outcomes. B rank and before-US grid bounds are insufficient for this gate.
3. **Numerics:** if the domain passes, a memory-feasible sparse patient-size
   solver profile and fixed 5 mm observation-operator accuracy/convergence
   checks are required before another mesh candidate or FEM prediction. The
   observed v4 surface failure cannot be waived by good topology/Jacobians or
   by its 0.315% enclosed-volume error. A saved-output localization diagnostic
   could test whether local refinement is plausible, without automatically
   authorizing generation.
4. **Physical inference limits:** Case4's comparison outcomes have already been
   exposed by prior development work. It can support a retrospective
   conditional-displacement engineering comparison only, never untouched
   held-out validation. Even a successful displacement interpolation given B
   motions would not validate surgical retraction, cutting force, tissue injury
   or scan-only action response; those require appropriate independent
   measurements and a known interaction history. Source:
   `docs/resect-conditional-displacement-poc.md` and
   `artifacts/mechanics/resect-case4-boundary6-preparation-v1/feasibility-audit.md`.

**Reconsideration rule:** freeze the above anatomy/frame/support and sparse
solver criteria before any new geometry run. If the fixed source fails local
retained-tissue suitability or too many observations lack support, stop Case4
mesh optimization and preserve it as a negative development result. Do not
substitute a different patient, modify the mask, lower the 2 mm gate, or select
a mesh using already exposed Case4 V outcomes within the same experiment.
