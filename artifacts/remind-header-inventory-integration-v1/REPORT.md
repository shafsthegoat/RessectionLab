# Independent ReMIND header-adapter review

**GO for the source candidate and metadata-only pilot protocol. No functional defect found.** This does not authorize a patient-header run. No real acquired DICOM file or header was opened, no pixel array was decoded, and no tracked file was edited.

## Exact review

- `header_adapter.py`: `d51237a21ccac5bad3e0ea7c07a4c48968e6a061547bb168bf95efd0ba8e3a76`
- `header-adapter-candidate.patch`: `849c4b81df956664b5f0cb6587836a306d2c7fc17dc8cc7d5c34888bef1cb051`
- `pilot-protocol-v2.json`: `20d0d73e574eebefe245573ea256afbb9bef750cbcdbe920e44bb24ebd39ed27`
- `prepare_inventory.py`: `f06331d64dd1669587cb5561c714cee093ac88f943e69894c89e6bbfea09237b`

The pilot plan was independently re-derived from the bound JSON metadata and completed raw-TRAIN receipt `32e319a3aafb571019ee16da1072c834015f9fdcdb97563fca4c19418f64a61d`. It exactly matches the saved plan: **ReMIND-002, TRAIN, 65 objects, 102,234,796 B; one 64-object intraoperative MR series and one single-object intraoperative US series**. Selection is lexical metadata selection, with no image-quality selection or replacements. The frozen cohort and full metadata/receipt chain are validated before producing the plan.

## Reader, geometry, and claims

`read_verified_header` reuses the existing streaming size/SHA verifier and available source MD5. The US object is rehashed against its exact acquisition-receipt SHA, which is bound to the earlier source CRC32C receipt; the reader does not independently recompute CRC32C. Symlinks/escaping paths are refused, and file identity is checked before and after the bounded header read. `BudgetReader` enforces a cumulative byte budget and deadline across seeks. The pinned pydicom 3.0.1 reader stops before ordinary, float, and double-float Pixel Data tags, and `specific_tags` restricts the header projection. No pixel decoder or conversion function is invoked. Whole-file fixity hashing necessarily reads raw source bytes; it is distinct from pixel decoding and from the per-object header-read budget.

Classic single-frame MR must pass identity, frame, shape, orientation, spacing, encoding, grouping, and distinct-plane checks before reusing `convert_remind_development.regular_affine`. Physical extent is the bounds of voxel-cell corners derived from headers, with anatomical coverage explicitly unreviewed. Enhanced/multiframe MR remains unsupported. US retains header/region inventory but **never claims an established patient-space affine**, regardless of calibrated regions or frame count. Raw timing and source-stage labels do not establish preoperative availability. Role, anatomy, conversion, training, and preoperative admission claims remain conservative.

## Meaningful controls

Independent rerun: **10/10 existing generated tests passed**, covering the author's 44 refusal subcases and two unsupported-MR routes. Six additional control groups passed: cumulative budget after backward seek; unbounded/over-budget reads; float/double-float pixel payload refusal; wrong source MD5 despite a matching receipt SHA; symlink refusal; and source mutation during header reading. The independent run used an audit hook forbidding real acquired-DICOM opens; zero such attempts occurred. The isolated runtime was `build/idc-acquisition-venv/bin/python` (Python 3.12.14, pydicom 3.0.1, NumPy 2.5.3, nibabel 5.4.2). Evidence is in `generated-tests.log`, `independent-controls.py`, and `summary.json`.

## Minimal promotion and launcher work

The ignored candidate deliberately derives `ROOT` and its handoff path from the `build/` layout. **Do not copy it unchanged into `scripts/`: adjust those paths and regenerate the plan/source bindings.** Preserve the existing verified-file helper and regular-affine routine; no new conversion framework is needed.

Before a real pilot, the separate launcher must bind the promoted adapter, handoff, reused sources, dependency versions/hashes, exact plan/contracts/jobs, and source receipt. It must validate those before opening a patient path, then enforce the declared **one worker, 90 s wall cap, 512 MiB RSS cap, 8 MiB header reads per object, 256 MiB/512-object source ceiling, and 4 MiB immutable output cap**. Preserve refused/unsupported outcomes without substitutions, record pre/post source checks and resource results, and leave acquisition files/locks unchanged. No launcher or patient execution was added by this review.
