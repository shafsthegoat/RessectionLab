# Support-map physical-frame guard

The optional scan adapter now provides `require_support_map_frame`. Given trusted map and reference hashes, it requires coded sforms, rejects conflicting coded qforms, checks matching three-dimensional grids across nibabel, SimpleITK and ANTs, and rejects a changed final file hash. A coded sform with an unset qform is accepted. Independent review found the missing final-hash comparison in the initial draft; it was fixed before integration.

Root integration passed 20 generated controls in 1.633 seconds across the existing scan adapter, support contract and new frame guard. Independent candidate review passed six generated controls. The pinned optional runtime has no pytest installation, so the initial pytest command did not execute tests; the same unittest suites then ran successfully. No patient image, model checkpoint or inference was used.

This verifies frame integrity only. The helper is available for a future explicit consumer and is not yet wired into patient inference or planning. It does not validate support-bit semantics, anatomical registration, mask quality or clinical suitability. Existing admission gates remain unchanged.
