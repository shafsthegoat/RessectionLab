# Isolated repaired runtime build

The separately released configure and build stages both completed once. The
configure stage took 8.05019 s with 134,529,024 bytes peak sampled group RSS.
Build/install took 436.34177 s with 585,285,632 bytes peak sampled group RSS,
within the unchanged 900 s / 3 GiB / two-job limits. Neither stage launched the
FEBio executable or any model.

Parent acceptance verified all 6,164 original acquisition entries and 1,037
original installed entries unchanged. Of the 2,322 source files, only
`NumCore/AccelerateSparseSolver.cpp` differs, matching the reviewed combined
patch. The 13 installed arm64 Mach-O files, dependency resolution and private
OpenMP reuse are bound by the saved inventory and linkage reports.

Accepted runtime identity SHA-256:
`13c4f60cbae8ad89232996e4f7d773bafbd2489f23a669c355f5ac09e417cc57`.

This establishes build provenance and installation checks, not numerical or
physical accuracy. The same-runtime eight numerical controls and then the fixed
specimen experiment remain necessary. The original Skyline runtime, failed
specimen attempts and unopened measured curves are preserved.
