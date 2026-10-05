# Saved mesh preparation review

Both saved meshes pass the declared geometry checks. N16 contains 7,209 nodes and 6,144 connected hex8 cells; N24 contains 23,101 nodes and 20,736 connected cells. Recomputing quality from the saved arrays with the exact frozen pure helper reproduces both full reports exactly. Relative volume errors are 0.160561% and 0.0713794%; normalized boundary sag is 0.120454% and 0.0535413%. Minimum sampled scaled Jacobians are approximately 0.7071.

All four original decks regenerate byte-for-byte from the saved meshes. Their adapted versions match the exact reviewed backend substitution. Material, solver control, domain and loading subtrees match the corresponding saved N12 case; every new top-node prescription and bottom-set prescription has the same declared semantics. The initial audit mistakenly compared entire boundary XML across different node inventories; `setup-correction.json` and the initial log/source retain this audit-only correction.

The twelve archive members match commit `5b80e3fee2349bc78d3f2ceb9b211efb305a1fc9` and their extracted source bytes. The prepare-only release, runtime bindings and all 168 baseline input hashes remain intact. The final 28-file raw index matches all 39,177,774 bytes and has no file or directory write bits set. Review details are in `review.json` and `collection-binding.json`.

The actual phase recorded two Gmsh calls and zero solver calls: 5.689497 seconds parent time, 4.668026 seconds nested worker time, and 285,802,496 bytes sampled peak process-group RSS. These intervals must not be added. All recorded limits were respected; between-sample peaks remain unobserved.

This saved-file audit took 4.442789 seconds. It did not run Gmsh, FEBio, new tests or response access. Quality recomputation reused the frozen helper rather than a separate geometry algorithm. No convergence, physical-validation or clinical claim follows from prepared meshes, and solver execution requires its own release.
