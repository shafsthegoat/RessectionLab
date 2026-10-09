# Scan-only diagnostic adapter integration

The reviewed adapter is integrated as an optional research component. It binds scans and the existing limited SynthStrip output to exact file records, computes scan-only transforms, preserves channel and physical-frame semantics, and emits coverage and transformation receipts. Unknown clinical acquisition times remain null. No patient registration, trained-model forward, private tumor annotation access or planner admission occurred.

Root ran **all nine generated tests in 1.648 seconds** in the isolated runtime. Base-environment pytest collection also succeeded; missing optional packages produce an explicit skip, not a collection error. Compared with the independently reviewed prototype, executable source AST is unchanged; the module docstring was corrected for tracked integration. Test changes are the promoted import and optional-dependency skip. The manifest differs only in paths to the two exact-byte portable metadata receipts. All processing, inference, planning and clinical admission flags remain false.

The independent review covers geometry, exact input/record chains and generated controls. Known mask omissions, model training/overlap uncertainty, and actual registration/coverage QC remain explicit. A future bounded DEVELOPMENT preparation can produce registration outputs for inspection; inspection follows generation and must precede model or planning use.

## Promoted file hashes

- `src/resectionlab/scan_target_adapter.py`: `80b05ca680676a97901aa35915839ef0b2d7357876727a7aad787a0efd54501b`
- `tests/test_scan_target_adapter.py`: `afe4164be60433e65a3b17708cb66a78b05bf75c5bf9a8fff693d221c40a627f`
- `docs/scan-target-adapter.md`: `e97e1a06eeb621795b5dd4e2840333db89b05f06c0cfc95992d5164f1d725aec`
- `manifests/experiments/resect-case4-scan-diagnostic-v1.json`: `d37948a4dc7acd16bdfac0244b926cfda5c684c60da1f6e4dfccb09ddcf21b26`
