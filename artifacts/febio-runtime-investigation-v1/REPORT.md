# FEBio runtime investigation receipt

2026-10-04. Static source/metadata and local prerequisite inspection only.
No configure, build, install, solver, specimen fit or patient mechanics run.

The official v4.13 annotated tag `9184ff860ca560bec32980ee05ce04724881c64c`
resolves to commit `32ae206ff4881dfb54f62296cd1558e58ed9fcc6`; the GitHub API
labels the tag unsigned. The version header identifies 4.13.0. The source MIT
license and attribution text are preserved. Website-distributed binaries have
separate terms and were not selected.

[Prospective runtime declaration](prospective-runtime.json) records exact source
URLs, dependency filenames/hashes, isolated paths, proposed configuration and
execution caps. It is **pending root freeze and execution release**; it neither
approves a mechanical model nor opens measured outcome values. A complete
source archive/hash, dependency payload notices and actual runtime remain
unverified. Metadata pinning is not a successful build.

The proposed native arm64 route uses Apple clang, private CMake 3.31.6 and
private llvm-openmp 21.1.8; a small zstandard helper permits bounded package
extraction without conda installation. All nine optional solver/library flags
are OFF. FEBio's built-in Skyline is the selected solver and supports symmetric
matrices only. Source inspection did not establish a dependency-free serial
fallback. Upstream macOS `-undefined dynamic_lookup` also makes an actual
symbol/linkage check essential. See [the local runtime note](../../docs/febio-local-runtime.md).

**Recommended compact commit subset:** this report; `prospective-runtime.json`;
`source-inspection.json`; `prerequisites.json`; `tag.json`; `annotated-tag.json`;
`upstream-LICENSE`; `upstream-Documentation__Copyright-FEBio.txt`; and the two
documents `docs/febio-local-runtime.md` and `docs/tissue-mechanics-frameworks.md`.
The exact dependency metadata URLs and selected bytes/hashes are already in
the declaration. Raw release/package metadata, selected-tree paths and other
`upstream-*` implementation/build-guide copies are preserved locally as research
material; they are not needed as redundant committed source copies. No files
were staged, moved or deleted by this review.

The source-inspection receipt retains unsuccessful guessed paths (HTTP 404)
alongside successful exact-commit file hashes. No source patch, paid solver,
system installation, global environment change or automatic fallback is part
of this proposal. Acceptance would establish only a bounded research runtime;
the [specimen measurement](../../docs/tissue-mechanics-measurements.md) and
[validation](../../docs/tissue-mechanics-validation.md) gates remain separate.
