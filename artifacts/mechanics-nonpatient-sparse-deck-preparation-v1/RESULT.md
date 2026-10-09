# Structured non-patient tet10 deck preparation

Status: **source-only, not released or executed**. This slice adds deterministic in-memory incidence and FEBio XML generation for exactly the seven cases frozen in `manifests/experiments/nonpatient-tet10-sparse-feasibility-v1.json`. It does not change that v1 declaration, the pinned one-cube fixture, the runtime, or any patient data. The numerical Ogden values remain software gauges, not brain properties. No solver result, physical measurement, patient displacement, cutting/retraction force, or clinical decision follows from this preparation.

`scripts/mechanics_nonpatient_sparse_deck.py` uses the pinned one-cube local tetrahedra and FEBio tet10 edge order (12, 23, 31, 14, 24, 34). Integer half-grid indices make midnodes shared by adjacent elements. Every generated node is referenced. Each generated tet10 checks positive straight-reference `det(dX/drst)` at all eight pinned quadrature points, all four vertices, all six edge midpoints, and centroid. This determinant has m³ units; each element's reference volume is determinant/6. The generated whole cube sums to `1e-6 m³` at every level. Face tags are checked against the grid coordinates, including corner/edge intersections. The one-cube control matches the pinned 27-node, six-tet fixture by physical coordinates and ordered element incidence.

| Subdivisions per axis | Actual nodes | Actual tet10 | Unique shared edges/midnodes | Exterior nodes | Reference determinant per element (m³) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 5 | 1,331 | 750 | 1,115 | 602 | 8.000000000000e-9 |
| 9 | 6,859 | 4,374 | 5,859 | 1,946 | 1.371742112483e-9 |
| 13 | 19,683 | 13,182 | 16,939 | 4,058 | 4.551661356395e-10 |

The XML has the pinned fixture's FEBio 4.0 structure, `Ogden` parameterization, `TET10G8` ordinary `elastic-solid` domain, symmetric solid/BFGS settings and node/element log variables, with the declaration's exact repaired Accelerate solver subtree. Affine cases prescribe `(F-I)X` on every exterior node while leaving interior nodes free. Nonuniform cases prescribe the frozen uniform bottom and top motions; side nodes away from those face intersections remain free. A linear 0-to-1 load controller applies them in four equal steps, except the declared eight-step repeat. XML is generated in memory and has only been parsed by Python's XML parser; its native FEBio acceptance is **unknown**.

The seven deterministically regenerated XML strings have these source-only SHA-256/byte counts (no deck files were written):

| Case | UTF-8 bytes | SHA-256 |
| --- | ---: | --- |
| `n5_affine` | 482,907 | `860dcdc8b29382281dd35ec41b6939bb29f1de5b3d3bd7a2e0adfd6f866a7674` |
| `n5_nonuniform` | 156,705 | `df19423a89f6af2c72f8fd5ab1566a8fa3c26c253a6b750b31f79bfd9de4e870` |
| `n9_affine` | 2,020,646 | `1dd6d1eeaa4072fed359f8217e1389a748bfafc065d230f643f607d4ff9645c9` |
| `n9_nonuniform` | 956,415 | `ee453a8e118e40144dd4b1e790afeedfc63b97c5284a9a68f14f14e4df76040d` |
| `n13_affine` | 5,147,419 | `c28ef941689441e7394a1b55d8fd61db25ca33ca7724bfe653c4e1893c1b08ac` |
| `n13_nonuniform` | 2,914,435 | `92536faadf1da5178c4ad54b5c27780415857be626b83e02cfc9c2478cff8da8` |
| `n13_nonuniform_half_step` | 2,914,468 | `8114d7840f9617c9384ea831017e3534f7b8d03b8976745d311b9ab7808e1c13` |

Focused validation: `47 passed` across the new deck controls, frozen design controls, and pinned fixture controls using `.venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_mechanics_nonpatient_sparse_feasibility.py tests/test_mechanics_nonpatient_sparse_deck.py tests/test_mechanics_patient_constraints.py`. Tests cover actual counts and volumes, matching the pinned one-cube incidence, all seven deck structures and boundary values, half-step controls, deterministic bytes, and rejected altered midnodes/tags/undeclared cases. They are software tests only.

Native execution remains blocked. A separate source review must check the generated deck against FEBio's actual parser and TET10G8 quadrature implementation, including the solver subtree and grouped node-set syntax. A separate saved-output parser must establish complete node/element states, correctly signed reactions, final residual norms for **every** completed step, deformation Jacobians and force/moment balance; an exit code alone is insufficient. Backend provenance must bind the repaired executable, linkage, runtime/profile/control receipts and observed backend-selection text for each case, with no fallback. A committed-source root release and fresh supervised output directory must enforce the seven-call order, one attempt per case, per-call and aggregate wall/RSS/output caps, retained partial logs, and stop on the first failure. None of those execution gates is present in this source-only slice.
