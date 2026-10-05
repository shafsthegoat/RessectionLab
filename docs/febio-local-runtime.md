# Prospective FEBio 4.13 local runtime

Read-only investigation, October 4, 2026; no configuration, compilation,
installation or mechanics execution has occurred. Execution awaits root's
release after the current regression. Source/metadata receipts are in
[`artifacts/febio-runtime-investigation-v1`](../artifacts/febio-runtime-investigation-v1/).

**Exact source:** official [v4.13 release](https://github.com/febiosoftware/FEBio/releases/tag/v4.13),
published August 4, 2026. Annotated tag
`9184ff860ca560bec32980ee05ce04724881c64c` resolves to commit
`32ae206ff4881dfb54f62296cd1558e58ed9fcc6`; its version header says 4.13.0.
The tagged MIT license hashes to
`82c4d113462cdd34a9f9f8be170f14ddf312cbba1e319d1fd197df0257754e0f`.
Use source, not the separately restricted website binary distribution.

**Present prerequisites:** arm64 macOS 26.6, Apple clang 21.0.0, Command Line
Tools SDK, system make and git. CMake/Ninja are absent from PATH and the usual
Homebrew CMake/OpenMP locations are absent. No Homebrew/system install is needed.

**Proposed isolated route:** one prefix under ignored
`data/optional-runtimes/febio-4.13/`, with separate downloads, source, tools,
OpenMP, build and install directories. The exact prospective metadata pins are
in [the runtime declaration](../artifacts/febio-runtime-investigation-v1/prospective-runtime.json).
Acquire only the pinned
source and build-tool/dependency artifacts, check hashes/licenses, then use
absolute executable paths. Preserve all logs and original notices.

- CMake 3.31.6 universal2 wheel: 47,224,338 bytes; its Python distributor
  declares Apache-2.0, while bundled CMake has its own upstream notices.
- Private conda-forge `llvm-openmp` 21.1.8, arm64 build `hc225544_2`:
  286,338 bytes; declared Apache-2.0 WITH LLVM-exception, only macOS ≥11 runtime
  dependency. Include its license files and verify actual Mach-O linkage.
- A pinned arm64 `zstandard` 0.25.0 wheel (640,436 bytes; BSD-3-Clause metadata)
  permits bounded `.conda` extraction without installing conda. It is a build
  helper, not part of the numerical model.

The tagged CMake finds OpenMP optionally, but `FECore/sys.h` declares OpenMP
functions. A dependency-free serial build is **not established** by that fact.
The explicit private OpenMP route avoids relying on an unverified serial
fallback. Force one runtime thread and verify the linked library; do not add
fake OpenMP stubs or modify solver source.

Configure a fresh Release build with Unix Makefiles, Apple clang, arm64 and a
private install prefix. Explicitly set `USE_MKL`, `USE_PDL`, `USE_HYPRE`,
`USE_SUPERLU_MT`, `USE_MMG`, `USE_LEVMAR`, `USE_NLOPT`, `USE_FFTW`, and `USE_ZLIB`
to OFF. Supply private OpenMP C/C++ flags, headers and `libomp.dylib` explicitly;
disable package-registry discovery. The tagged source links Apple's Accelerate
framework on macOS. CMake's existing target is `febio4`; build with two jobs,
then install only into the private prefix. No shell profile or global PATH edit.

Without MKL, `NumCore.cpp` selects built-in **Skyline**. Its source accepts
`REAL_SYMMETRIC` matrices only. Therefore the first no-contact, quasistatic
hyperelastic verification must use a supported symmetric tangent; do not
silently symmetrize an unsupported problem. No Pardiso purchase or account is
required. Skyline's speed/memory for the intended mesh remain unmeasured.

**Proposed caps and gates:** acquisition 128 MiB aggregate, source archive
64 MiB compressed/512 MiB expanded; configure 120 seconds; build/install
900 seconds with two jobs and sampled process-family memory ≤3 GiB; initial
version/analytic smoke checks 60 seconds and one numerical thread. These are
proposals for root to freeze, not measured requirements. On failure, retain the
logs; no automatic fallback, extra package, source patch or cap increase.

Before accepting the runtime, inspect cache flags, compiler architecture,
binary/library hashes, RPATH and `otool` dependencies; reject ambient/paid
libraries or unresolved OpenMP symbols. Confirm exact version and explicit
Skyline selection, then run a tiny analytical verification under supervision.
A successful build does not open HBE calibration/held-out curves, authorize a
specimen fit, or validate tissue mechanics. Those require the separately frozen
measurement/model declaration.
