# Independent source-only review: support-map frame guard

**GO for narrow source/test promotion.** This candidate adds an optional physical-frame verifier to the existing scan adapter. It does not load Case4 data in tests, inspect support-bit meaning, change anatomical QC, run inference, or admit a planner.

Frozen candidate hashes: patch `e2525eb89ec08ed7c7985c54ae1f496658a1b129f7e92c6d3d24c4962edc9d31`, full isolated adapter `c550bcd22b1a65cbd1b742c5d417ba16cb7bb46e11a98e84d89ef0b1234d5fa2`, proposed tracked test `38a4e05fbe2f5b288f6fedb230f3310775f9a6d9b33333672d12c832a7c8d004`. The patch applies cleanly to the current tracked adapter SHA `0d045bc47136f958fdef9f200d1275dfc7b8cc86346c4d15abc07ca132ccf32f` (`git apply --check` passed).

The verifier requires trusted expected SHA-256 values for map and reference, a coded sform on each, no disagreement from any coded qform, identical 3-D shapes and physical affines, explicit millimeter units, and nibabel/SimpleITK/ANTs frame agreement via the existing `read_volume`. An unset map qform is accepted only with a coded sform, as in the saved Case4 diagnostic. The initial candidate omitted checking the final `PreparedVolume.sha256` after the data read; that could have returned changed bytes after the initial hash check. The frozen revision now rejects a changed final SHA.

Independently reran **6/6** generated NIfTI tests in the pinned scan runtime: valid unset-qform/sform match; qform-only rejection; shifted sform rejection; conflicting coded qform rejection; substituted hash rejection; and mocked changed-final-hash rejection. No patient files or model calls were used. This guard does not by itself validate support codes, brain anatomy, source evidence, or suitability for route planning; those remain separate gates.
