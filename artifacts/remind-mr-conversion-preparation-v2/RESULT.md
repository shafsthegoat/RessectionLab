# Generated MR conversion and bounded array-path test

Independent review found two real defects in the first draft: NIfTI spatial units were unspecified, and both conversion and sample verification could share an altered pixel-decoder cache. The repaired version explicitly declares and requires millimeters, resets decoder options/cache, and verifies expected samples directly from immutable signed16 little-endian PixelData bytes. Twelve generated controls and independent reproductions of both original failures pass.

One generated 512×512×64 control then completed under the unchanged 90-second, 512 MiB RSS and 96 MiB output bounds. Worker time was 0.647177 seconds; peak RSS was 387,760,128 bytes (369.80 MiB); cleanup succeeded. Independent saved-output checks verified float32 data, coded millimeter frames, all 64 plane corners (maximum error 0.00000597845 mm) and 3,327 deterministic sample values. The 64 MiB generated image remains ignored and hash-bound. One blocked optional-import socket.bind attempt is retained; no network or patient read proceeded.

This measures generated image conversion and round-trip memory only. It excludes the larger acquisition metadata validation and verified patient-reader footprint. Patient conversion, anatomical qualification, registration and preoperative availability remain unadmitted. A future launcher must separately validate the complete metadata chain, reap that stage, and pass a small exact-source handoff to the conversion stage with combined accounting.

The indexed archive preserves the original negative, repaired source and tests, exact dependencies, prospective review, actual receipts and independent saved-result review. It contains no image or patient payload.
