# Case3 image registration preparation

Ready for ROOT review and a separate one-fit release. No acquired image array or landmark coordinate was read, and no acquired registration was run. The protocol template remains `NOT_RELEASED`.

The fixed image is `data/anatomy/resect-v1/Case3/US/Case3-US-before.nii.gz`; the moving image is `data/annotations/resect-seg-v1/originals/Case3-US-during.nii.gz`. Both are original RESECT v1 ultrasound observations, bound to the frozen Case3 TRAIN role. Before-US is intraoperative before resection and may precede or follow dura opening. Filenames do not establish exact operative timing.

`header-preflight-rss-v2.json` is the final source/frame receipt. It records streamed SHA256/MD5 checks, 348 returned header bytes per source, active sform1, inactive qform0 with source qfac0 preserved, orthogonal sform axes, source scaling, and exact paths. Shapes are 337 × 313 × 246 and 338 × 303 × 245; native float32 arrays would total 204,159,024 bytes. Header convention does not establish anatomical alignment, tracker calibration, effective ultrasound resolution, or material properties. The earlier `header-preflight.json` is preserved and must not be used for the final release.

The runner reuses the pinned IXI SimpleITK rigid algorithm: physical identity initialization, MI32, seed41, 20% sampling, 1.5 mm coarse grid, 80 iterations per level, shrink [4,2,1], smoothing [2,1,0] mm. Metric support is positive fixed ultrasound signal intersected with geometric coverage, followed by the fixed coarse erosion. It is never an anatomical mask. Forward sampling maps before-RAS physical points to during-RAS physical points; the inverse and identity baseline are validated before freezing.

The proposed single attempt has two threads, a 180-second optimizer limit, a 240-second worker limit, and sampled family-RSS stop enforcement above 4 GiB at 0.2-second intervals. It records sampled worker/family peaks and minimum available memory. The 2,449,908,288-byte planning estimate is not a measurement; sampled enforcement can miss transient peaks and is not an OS hard memory limit. Source, runtime and code fixity are rechecked before publication. Timeout, memory, signal, source, runtime, optimizer or cleanup failures cannot create an evaluation-eligible freeze. Full candidates and fitted transforms are restricted to `outputs/`.

Thirty generated tests passed. Independent QA passed 36 generated checks and four tiny subprocess controls, including SIGTERM, descendant cleanup, source mutation, changed runtime/code, inverse/direction failures and memory-stop sealing. These checks establish software behavior only. The independent control receipt is `artifacts/resect-case3-registration-preflight-independent-v1/generated-controls.json`.

`prospective-protocol.json` contains exact release fields, source/helper pins, rights, phase uncertainty, the future command and the separate evaluation contract. `preparation-receipt.json` binds the final runner, tests, protocol and header receipt. No ROOT fit release has been created here.

After a successful supervised freeze, a later, separate evaluator may use **all** records in `Case3-beforeUS-duringUS-full.tag` for evaluation only, bound to the frozen transform and completion receipt. Neither its coordinates nor the overlapping short tag were opened. No landmark may enter fitting, initialization or parameter selection. Case3 remains TRAIN; this would be a one-case measurement, not protected-cohort generalization or clinical validation.

Original observations are CC-BY-4.0 under RESECT v1, DOI `10.11582/2017.00004`; retain attribution and preprocessing lineage. No RESECT-SEG cavity label is used. Case4 DEVELOPMENT, SELECT and MEASUREMENT_EVAL payloads remain untouched.
