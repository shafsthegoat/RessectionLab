# RESECT deferred QC attempt 02: independent saved-output audit

The review completed its fixed worklist, but **none of the 24 newly inspected pairs passed overall QC**. All 24 image reviews stopped in raw-header recording because an inactive qform has `qfac=0`. Their image scalar/content checks remain unfinished. This is an adapter interpretation limitation, not evidence that the images or anatomy are invalid.

The denominator remains 25 pairs: 24 new `review_failed` outcomes and one inherited Case3-during structural pass. They represent 13 people within the 14-person TRAIN group. The three missing annotation timepoints remain unavailable; they were never substituted with negative masks. Prior attempt 01 and historical transport failures retain their exact hashes.

All 24 recorded mask binary-content checks passed. This audit authenticated those results and recomputed count arithmetic, raw-header identity, decoding budgets and full reported geometry. The reported foreground counts total 8,294,753 across 1,016,845,534 source voxels; the per-timepoint foreground range is 2,131–914,097. These are correlated during/after cavity annotations, not independent patient outcomes or total removed tissue. No voxel payload was independently recounted.

The already saved 348-byte image prefixes support a supplementary result: **all 24 active sforms pass the unchanged header-geometry rules**. Their image/mask shapes match, and independently enumerated grid corners differ by at most 4.94866631695e-05 mm, below the existing 0.01 mm correspondence tolerance. These header-only findings do not change the authoritative failed outcomes, establish scalar-content validity, or grant anatomical, scanner-frame, training or planning admission.

The audit verified 288 metadata/source/receipt bindings, all 17 source files against commit `446ea6d` and retained snapshots, 62 input-metadata snapshots, all 24 intent/worker/parent chains, and all before/after fixity receipts. All 24 worker PIDs were absent at audit time. The audit used saved metadata and raw-header bytes only; no image/mask payload, network request, new QC execution or source edit occurred.

Recorded batch time was 19.063570 s (outer process 19.167475 s). Reported worker times sum to 13.462814 s, with a 0.677555 s maximum. Maximum observed single-worker RSS was 70,942,720bytes; explicit mask chunk workspace was 786,432 bytes. RSS is an observation, not a hard memory guarantee or aggregate-process measure. All recorded durations stayed within 60 s per worker and 600 s for the batch.

The first independent audit completed successfully. The remaining image-content review requires a separately reviewed recorder compatibility change; this audit does not authorize another run or relax any source, rights or admission gate.
