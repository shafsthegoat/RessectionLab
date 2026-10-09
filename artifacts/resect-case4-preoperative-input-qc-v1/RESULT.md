# Case4 scan-only input check for a development segmentation pilot

2026-10-09: a bounded read of the acquired preoperative T1 and FLAIR verified their original identities and finite arrays, inspected their original physical frames, and rendered three fixed central planes. No tumor/cavity annotations, ultrasound, landmarks, outcomes, support masks or model weights were opened. No registration was fitted, and no inference or planning ran.

The [RESECT creators' paper](https://doi.org/10.1002/mp.12268), Acquisition and validation methods, describes gadolinium-enhanced T1 and T2-FLAIR for each of the 23 participants. Coupled with exact original-file identity, this supports a **cohort-protocol T1c interpretation** for this DEVELOPMENT pilot. Case4's converted NIfTI header does not itself contain injection metadata; appearance was not used to infer contrast administration.

| Input | Compressed bytes | SHA-256 |
| --- | ---: | --- |
| Case4-T1.nii.gz | 8,763,797 | `eeab888d611b7dee3627468b172ad67db39b6887e6ba38540aa39131b6cbb787` |
| Case4-FLAIR.nii.gz | 6,550,243 | `e4b2e09f6d47e256a0af3e515d321a393cfc1b5d6c704a6b23bbfeed8c712d26` |

Both native arrays are 256×256×192 float32 with 1 mm spacing and distinct coded sforms. Mapping the fixed T1 plane centres through the original sforms covered 96.38%, 98.25% and 96.93% of the corresponding FLAIR grid. These are sampled field-of-view coverage values, not registration accuracy. The affine/inverse algebra error was 4.17e-14; it does not establish anatomical correspondence. The agent and root inspected the same three-plane montage and saw broad head/brain contour correspondence. Local anatomical alignment remains unaccepted.

One initial local script launch failed before reading images because of a variable-name error. The corrected single read took 0.906 seconds with 352,796,672 B process peak RSS. Original hashes were unchanged afterwards. Restricted script, full metadata and montage remain ignored under `build/limited-input-guard-design/`; script SHA-256 `631143c261104fede3422fe90db80788f34fd778ef00fc42bfe1d3196f793895`, JSON `c769d924300b763692e3c2d525e80cd50ef9c6d7be0e56aaaf28e692b3161dbb`, montage `d1055d2894d74c871e0ed7933a41f297ba0420fb1b3501b92cb8831d45de8577`.

The next implementation can estimate scan-only FLAIR-to-T1c and atlas transforms, then inspect local alignment and inverse mapping before using an inferred target in planning. GlioMODA expects already registered, skull-stripped atlas-space inputs; its wrapper does not perform registration. Its pinned wrapper has no verified code license, while the weight release separately declares CC BY 4.0. An independently written adapter to licensed nnU-Net components is the proposed research path; no wrapper code is admitted to the app by this check. Weight provenance, preprocessing, segmentation semantics, support and access admission remain separate unfinished work.

Case4 retains its DEVELOPMENT assignment. This check establishes neither a target estimate nor patient transfer, physical fidelity or surgical benefit. Previously exposed Case4 displacement endpoints cannot become untouched evaluation data.
