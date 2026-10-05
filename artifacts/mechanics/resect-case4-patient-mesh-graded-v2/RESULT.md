# Case4 graded candidate: terminal surface-fidelity rejection

The sole released candidate from `715f0bf99fdd76eb984e5522aa2c7fa538498f63` returned **5,223 nodes / 2,761 tet10 elements**, within the fixed 6,000/6,000 mesh-count limits. The same 12 mm boundary / 24 mm interior profile, Sampling 100 and 2–24 mm transition were used once. Element geometry, shared midsides, positive sampled Jacobians, connected closed topology and independent overlap checks passed. The candidate nevertheless failed the unchanged **2 mm bidirectional source-surface gate**. No retry or settings change occurred.

| Direction | Samples | Greatest sampled distance | Full-surface covering bound |
| --- | ---: | ---: | ---: |
| Source to mesh | 1,027,230 | 5.459949978 mm | 6.167056788 mm |
| Mesh to source | 121,028 | 3.833729993 mm | 4.830812978 mm |

The sampled distances themselves exceed 2 mm; failure is not caused only by the conservative covering-radius addition. The computed volume difference was 1.208376847% (below 3%), illustrating that small total-volume error does not establish local surface accuracy. The surface gate already rejected the candidate, so these numbers do not confer acceptance.

The checked boundary has 732 corner vertices and 1,464 triangles, one component, Euler characteristic 0 / genus 1 matching the estimated source. Minimum mean ratio was 0.083392174; all 19 reference-Jacobian witness sites per element were positive, and midside error was 0. The independent overlap test checked 63,528 candidate pairs. This establishes the reported geometric checks only, not anatomical or mechanical validity.

A **complete 413,290-byte diagnostic packet** is retained under `outputs/mechanics/resect-case4-patient-mesh-graded-v2/candidate/diagnostic`: full native RAS-meter coordinates, zero-based FEBio tet10 connectivity, native node/element IDs and a source/frame/permutation/hash manifest. Each file matches its saved complete-array hash and size. It remains explicitly unaccepted. The old failed v1 attempt is unchanged.

The process exited 1 after 10.479403042 s, with 647,561,216 bytes sampled group peak RSS and no resource kill. All output guards stayed within their fixed limits; the final raw attempt has 10 files and 446,610 bytes. One numerical thread was requested. RSS and aggregate output are sampled; unrelated work can affect timing, so this is not a performance benchmark.

All 28 original-source/archive/runtime/input/context bindings remained unchanged; worker and launcher each verified 19 bindings. Exact small raw records are copied in `saved-records/`; `raw-output-index.json` binds every raw output including the complete rejected mesh arrays. No new MRI extraction, B/V landmark access, solver, training, inference or parameter search occurred. There is no accepted mesh, solver admission, anatomical-registration approval or clinical validation. Known inferior/cerebellar source-envelope exclusions remain limitations.
