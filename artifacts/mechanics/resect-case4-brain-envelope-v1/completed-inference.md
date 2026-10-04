# Single native Case4 envelope estimate completed

The exact released `ea2c60c` snapshot ran once with the declared main-v1/MPS settings. Inference took 7.305 seconds; the whole supervised group completed in 11.644 seconds. There were no failures, retries, fallback models or downloads. The release records possible overlap with a tiny 27-node MPC software control; these timings are not a claim of an isolated benchmark.

All 59 snapshot files and 33 source/model/selected-runtime bindings were reverified after completion. The original native T1 is unchanged. The wrapper accepted a finite binary nonempty mask and finite native SDT on the original 256×256×192 grid; SDT voxel-centre corner discrepancy was zero. Independent reconstruction, full-cell corners and fixed three-plane QC remain separate.

The unchanged ignored outputs are `outputs/mechanics/resect-case4-brain-envelope-v1/main_mask.nii.gz` (SHA256 `7902cfbcb5fad15144181c06883bac7f4800e842f7ff05a540f56bac4eb38759`) and `main_distance_mm.nii.gz` (SHA256 `21545d1b70fcce791ddb9d6144c8534cd4c4d2d2616c21e99bfd969781b11195`). Exact model logs, inference record, supervisor receipt, runtime preflight and byte-integrity closeout are retained under `attempt-01`.

Sampled group RSS was 725,057,536 bytes. MPS instrumentation observed 3,098,719,232 tensor bytes and 5,948,243,968 driver bytes. These are separate sampled measurements, not a joint exact peak. No resource cap fired.

The result remains an estimated envelope requiring review. It establishes no working brain mask, cortical access, accepted baseline transform, pial surface or cavity boundary. The SDT's 100-mm exterior fill is not clearance or calibrated uncertainty. Motion tags, B/V measurements, during-US images and annotations were not inputs or postprocessing guides.
