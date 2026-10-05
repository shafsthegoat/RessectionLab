# Narrow FEBio sparse adapter repair

The preserved upstream adapter allocated/copied CSC offsets using the nonzero count rather than columns plus one. The exact patch changes that count and gives zero-equation matrices the existing no-work behavior. It does not change mechanics, matrix entries, materials, mesh or iterative tolerances; iterative mode remains excluded.

Six repaired-only source-fragment controls passed ASan/UBSan in 2.203 supervised seconds with 157.4 MB sampled group memory. The independent saved-evidence review verifies exact source extraction, copied values, terminal offset, zero-created/null cases, compiler/binary identities and sanitizer instrumentation. These mock-storage checks do not validate the full adapter, factorization, numerical solution, performance or physical fidelity.

The first diagnostic compile failed because its SDK path was missing. No original faulty binary was produced or executed. An automatic screening interruption then stopped the previous preparation worker before a corrected diagnostic. The successful replacement executes repaired code only; no dynamic reproduction of the old defect is claimed. `bounds-regression.py` and its release are retained historical records, not the active workflow.

The measured runtime-v1 declaration hash matches the failed release. A root inference based on an earlier reported hash was corrected in the independent review; no divergent snapshot or history gap is established. A distinct runtime-v2 declaration must bind the positive-only scope before an isolated build. The original runtime stays unchanged, and eight numerical controls remain prerequisites for using any rebuilt backend.
