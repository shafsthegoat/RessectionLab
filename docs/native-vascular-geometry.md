# Acquired vascular source geometry and connectivity checks

## Native vessel wall reconstruction

Native anatomy review keeps the acquired grid, mask, score array and extracted
source contour. Label layers use VTK Surface Nets. Mask-compatible score layers
use VTK Flying Edges at the stored half-score crossing. The conditioned surface
uses the same triangle indices. A physical-space fairing balances locally
coherent normal motion to reduce caliber drift while smoothing the silhouette.
Local foreground depth bounds the movement and protects one-sample strands.
Weak, shallow score crossings on those strands keep their incident faces.
Acquisition edges and nonmanifold mesh edges stay pinned. The physical audit
checks for reversed or collapsed faces and newly intersecting nonincident
triangles after final position quantization.

The versioned `source-conditioned-v1` research wall keeps ownership from the
original native crossings. Its conditioned positions and triangle indices
provide the canonical wall for admitted collision and rendering. The source
shape remains separately available for inspection. Measured section areas
describe the mesh geometry derived from the available label or score field.

Use `scripts/measure_vessel_surface_sections.py --compare-previous` with the
native study and matching source-branch JSON. The receipt pairs sections on
the same centerline stations and records displacement, geometric normal turns,
surface area and newly introduced contacts. Its temporary mesh data stays
beside the requested receipt. The HDD pilot in
`qc/native-vessel-geometry-20261124-v1/` includes unlit, wireframe and flat
shaded comparisons for the supplied TopCoW-014 artery and vein/sinus and two EDEN
TOF crops. The same receipt directory holds a per-case table for all 29 acquired
EDEN TOFs at the frozen native crop bounds. Across the crops, the median case
95th-percentile absolute section area change fell from 18.5% to 2.6%. The
physical audit found zero introduced nonincident contacts.

## Source-owned branch geometry

The existing TopBrain CTA001 now has reproducible, source-owned centerline
geometry and a separately certified H₁ representative subgraph. Original voxel
IDs, native physical coordinates and publisher artery/vein/sinus classes remain
traceable. No source mask, registration, clinical boundary condition, perfusion
territory or patient outcome has been created or changed.

This is source-geometry qualification on a separate vascular development case.
It does not supply vascular anatomy to UCSF-PDGM-0007, which remains unsupported
for patient-specific vascular evaluation. It does not pass acquired brain-shift
prediction or measured-flow validation.

## Exact source and frame

The source is **TopBrain Challenge Organizers, TopBrain Challenge Data Release,
Batch1 July30 2025**, [Zenodo 16623496](https://zenodo.org/records/16623496).
Its TopCoW lineage is described by the [creator challenge paper](https://arxiv.org/abs/2312.17670).
The release requires attribution and permission from the data owner for
commercial use. This local work uses existing noncommercial research inputs;
there is no new acquisition or redistribution of patient volumes.

The [publisher's CTA class table](https://topbrain2025.grand-challenge.org/data/)
distinguishes arteries 1–34, veins 35/37/38/39 and sinuses 36/40. These CTA semantics
must not be applied to MRA, whose values 35–40 denote different structures.
The execution requires the exact frozen existing patient manifest, source label
map and label dictionary hashes before opening arrays. It checks that the two
native binary masks exactly equal the corresponding original categorical unions.

The CTA header explicitly declares millimetres. The associated annotation header
leaves its units unset, although shape and sform affine are exactly identical.
The first strict header check correctly refused this combination before decoding
any array. A tested named-reference rule now binds this annotation's indices to
the explicitly associated CTA grid; it records the inherited unit and leaves
both original headers unchanged. Contradictory explicit units, absent association,
missing sform or a different grid still refuse. This is indexed label-to-image
association, not a registration fit or independent anatomical acceptance.

## From source voxels to representative edges

`anatomy_geometry.vessel_graph` crops only empty exterior workspace, pads for
numerical operations and applies Lee thinning. Every retained point carries its
original C-order source-cell ID, exact float64 RAS coordinate, source6/source26
component IDs and an annotation-to-background-center distance. That distance is
not a measured lumen radius. Original image-boundary contact is reported
separately from node-level boundary-limited distances; thinning can move all
nodes off a truncated source face.

The source voxel union, thinned voxel union and all-26-neighbor edge graph are
different mathematical objects. In particular, a triangle of mutually touching
voxel cells has no voxel hole but creates a cycle in the adjacency edge graph.
The implementation reports their components, Euler counts, cavities and cycle
ranks separately. It preserves source6 island omissions and unresolved local
face-path contacts in the coverage denominator. A local diagonal shortcut may
still belong to one larger global face-connected component; the two findings
are not conflated.

`vascular_homology.representative_subgraph` retains every skeleton node and uses
only a subset of its original edges. Complete26 adjacency is first verified.
A deterministic spanning forest identifies fundamental-cycle chord coordinates.
Exact GF(2) elimination of triangle boundaries chooses pivot chords to remove;
the nonpivot cycle space is a complement of those boundaries. The output includes
forest/chord indices, original independent triangle generators, echelon-basis
words, pivot columns and retained source-edge indices. A verifier can enumerate
all triangles to check membership, completeness and the quotient complement.
Caps return no partial representative.

The closed-convex [nerve theorem](https://arxiv.org/abs/2203.03571v6) supplies the
voxel-cell interpretation: pairwise-intersecting axis-aligned boxes have common
intersection. The resulting inclusion certifies H₀/H₁ over GF(2), not a
deformation retract, homotopy equivalence, route length, lumen or flow. Higher
homology is checked separately. A hollow voxel shell therefore cannot be
certified by a one-dimensional graph merely because its H₁ is zero. Raw edges
are preserved alongside the representative, and physiological admission remains
false.

## Acquired-source results

| Quantity | Arterial union | Venous/sinus union |
|---|---:|---:|
| Original positive voxels | 44,951 | 34,226 |
| Retained skeleton nodes | 2,057 | 1,098 |
| Source components6 /26 | 17 /2 | 22 /6 |
| Raw adjacency edges /cycle rank | 2,107 /52 | 1,184 /92 |
| H₁ representative edges /cycle rank | 2,080 /25 | 1,140 /48 |
| Triangle-boundary rank | 27 | 44 |
| Source26 components omitted | 0 | 0 |
| Source6 islands /voxels unrepresented | 8 /8 | 9 /12 |
| Edges lacking a local face path | 22 | 43 |
| Edges crossing global source6 components | 8 | 10 |
| Source voxels touching image extent | 115 | 0 |

Independent positive-voxel BFS, explicit cubical cell counting, local-path checks
and direct affine evaluation verified the source geometry. Separate dense GF(2)
elimination verified every native representative certificate. All 3,155 retained
source nodes remain, with no new edges. Source masks and image files retain their
original hashes. The final two-group run took about 3.12 s with approximately 718 MiB
peak RSS; this is an extraction measurement, not an RL or viewer speedup.

The public native inspector exposes source IDs, separate topology checks, FOV
contact, island loss and representative status. Its existing radius field is
explicitly marked as an annotation estimate, and `flow_model_admitted` is false.
The partial annotations omit known source classes and unannotated branches;
missing labels do not establish absent anatomy. Local expert acceptance,
physiological boundary conditions, lumen/flow measurements and glioma-specific
vascular coverage remain open. The [RC model](vascular-graph.md) is a separate
declared numerical mechanism; these thousands of adjacency nodes are not silently
converted into its lumped segments or blood-flow parameters.

### Same-source class review in the newer TopBrain release

A separate review of the TopCoW-001 CTA development migration used the
August 17, 2026 TopBrain v3 release. Both supplied label versions used the
same native image lattice, 0.508 × 0.508 × 0.625 mm. The reference hashes,
native overlays and per-class measurements are in the private HDD review
under `qc/continuous-rehearsal-20261118/source-review/`. The acquisition is
privacy-redacted. The two releases share subject and annotation lineage.
[Publisher dataset](https://zenodo.org/records/21972006)

| Structure | Source result |
|---|---|
| Basilar artery | The 2,975 positive voxels match across versions. The v1 label is one face-connected component. Median twice-centerline-to-background distance is 3.71 mm. |
| Right posterior communicating artery | Both versions label the same 218 voxels. The v1 label is one face-connected component. The median width proxy is 1.25 mm, close to two source slices. |
| Superior sagittal sinus | The v1 label has one 26-neighbor component and ten face-connected components. Twenty-seven skeleton edges lack a local face-connected path. |
| Vein of Galen | The v1 label has 524 positive voxels and one face-connected component. One diagonal skeleton edge lacks a local face-connected path. |
| Anterior communicating artery | Both supplied masks are empty. Presence and caliber remain unknown. |

The two venous structures have no matching v2 class in this comparison. The width proxy
measures the sampled label. Lumen diameter and patency would require independent
source review. Vascular evidence belongs to this CTA patient.

## Reproduce

The final [protocol](../manifests/experiments/native-vascular-graph-v4.json) binds
exact source and code hashes. Earlier protocol versions preserve the initial
header refusal, raw-graph diagnostic correction and pre-reduction reference.
Detailed coordinates, source primitives, negative attempts and independent review
receipts remain under ignored `testcases/outputs/simulator-qualification-20261029/native-vascular-graph-v1/`.

```sh
.venv/bin/python -m pytest tests/test_vascular_homology.py \
  tests/test_vascular_source_graph.py tests/test_patient_anatomy.py -q
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python \
  scripts/qualify_native_vascular_graph.py \
  outputs/poc-angio-fidelity-v2/native-cta \
  testcases/outputs/simulator-qualification-20261029/native-vascular-graph-v1/new-run
```

No force curves or reserved patient payloads are accessed. Surgical model and
recording semantics remain separate from this source-geometry cache.
