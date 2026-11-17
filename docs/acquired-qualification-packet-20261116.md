# UCSF-PDGM-0004 inspection packet · November 16

The existing DEVELOPMENT source now has a saved, self-contained qualification
viewer at `testcases/overnight-sequential-v1/acquired-ucsf0004-v1/index.html`.
The directory is intentionally ignored; this tracked index preserves the
delivery location, protocol and exact evidence identities. It is an inspection
package, not a completed patient procedure or admitted learning case.

```sh
open testcases/overnight-sequential-v1/acquired-ucsf0004-v1/index.html
PYTHONPATH=src .venv/bin/python testcases/overnight-sequential-v1/acquired-ucsf0004-v1/build_packet.py
```

The viewer includes all source-grid cells in four modality atlases, linked
orthogonal sliders, independent supplied target/context and inferred structure
overlays, exact source/transform bindings, and eleven missing execution
prerequisites. `README.md` beside it gives the hidden native verification
command. `packet.json` is the machine-readable evidence; `inventory.json` hashes
the complete saved delivery. Rebuilding changes timestamps and requires updating
the delivery inventory/index hashes.

| Evidence | Verified result |
| --- | --- |
| Retained original mirror field | Four 240 × 240 × 155 volumes, 1 mm grid; 8,928,000 cells |
| Source target | Unchanged label 4: 15,672 mm³; labels 1/2 retained separately as context |
| Ownership overlap | 672 target cells also in the experimental vessel proposal; categorical tumor owner is only 15,000 mm³ |
| Brainstem / cerebellum proposals | 24,378 / 146,333 mm³, unreviewed SynthSeg 1 |
| Cached inference field | Native `[24:216,24:216,0:155]`; 3,214,080 outside-field cells remain unknown |
| Patient motor cortex / CST | Unknown; generic cortex/white matter is not motor evidence |
| Production source admission | `patient-or-unverified`; new planning unavailable |
| Procedure / observation limits | Both unchanged at 2,000,000 cells; full source is 4.464× larger |
| New acquisition / inference / physical actions | 0 / 0 / 0 |

The complete original **mirror** field is retained. Its preprocessed,
skull-stripped support does not establish complete scanner-native acquired-head
or intracranial coverage. The cached stem reaches native k22, the inferior
nonzero T1CE boundary. All nonzero supplied T1CE support lies inside the cached
inference field, but this does not certify omitted anatomy or negative coverage.
The existing uncropped preparation failed its unchanged 8 GiB RSS bound; no
duplicate high-memory inference was launched and no crop or limit was changed.

The attached drive was present and its selected review package was checked
before reusing the local cache. Structural pixels and supplied tumor membership
match, while saved derivatives retain distinct hashes. The drive package's
default adapter combines all three tumor compartments (41,919 mm³); this packet
explicitly uses enhancing label 4. Neither a review-package acceptance decision
nor a source hash supplies procedure execution permission.

Execution remains blocked by the acquired-source hold, full-grid limits,
unqualified original anatomy/negative coverage, missing independent
intracranial/entry/work-domain declarations, absent skull/access evidence,
unreliable vascular completeness, absent motor evidence, registration review
and physical/training qualification. External imaging and expert review remain
dependencies; no review receipt or negative coverage was invented.

Verification passed: original-byte/native-array hashes, source dtype/affine
joins, exact unchanged label-4 membership, cached label-output identity, all 155
full slices in each atlas, overlay membership counts, all four modality loads,
all slider endpoints, seven overlays, eleven blocker rows, and no horizontal
overflow. The native renderer stayed hidden and never called `show()` or
`focus()`. Actual captures are `viewer-hidden.png` and
`viewer-inferior-hidden.png`; full source planes are
`source-proposals-nine-planes.png`. These received software/source inspection,
not expert anatomical review. Source datasets and historical saves are unchanged.

Frozen delivery SHA-256:

| File | SHA-256 |
| --- | --- |
| `packet.json` | `48db9b973bc6631fe064df36e792046f12a77bfc1def8e88cf67c96487f08156` |
| `index.html` | `bf101d2951a2ad21580ee9ed3452a2aad2179938c11846d9d6f1659a92358650` |
| `inventory.json` | `f4668ce5c362bd46e2495a096ccbe0cf60c670cf89c45bf7838a667cc11fe187` |
| `source-proposals-nine-planes.png` | `26208d56ff56b99b761640022648d557b914af971f50136cc4c843b8211295a7` |

The packet records its precise source commit and production adapter bindings.
No engine, planner, UI or admission implementation changed during preparation.
