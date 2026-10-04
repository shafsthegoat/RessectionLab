# Independent prior-sampling audit

`validated487.json` records the final check of the actual production WebGL2
shader prefix and production CPU sampler on 487 analytical fixtures. All
coverage expectations and numerical checks pass on the Apple M5 ANGLE Metal
backend. Source hashes were captured before execution and confirmed unchanged
afterward. This verifies this sampling path; it does not establish clinical
alignment, patient-specific function, other GPU behavior, or whole-app visual
quality.

The fixtures include 64 oblique covered-cell landmarks, nearest-cell binary
half ties, missing support 0.001 voxel away from a landmark, diagonal missing
support with interpolation weight 4e-8, nonconstant scalar ramps, covered zero,
positive values of 1e-5 and 1e-8, and exact outer faces with inside/outside
controls. Exact outer faces abstain under the declared precision convention;
the CPU reports `numerical-boundary-uncertainty`. Genuine missing support is
still unavailable. The shared source-grid tolerance in these fixtures ranges
from 2.3842e-6 to 2.0562e-5 voxel and remains below the production hard limit of
0.001 voxel.

| Receipt | Outcome |
| --- | --- |
| `initial-failed71.json` | Original shader incorrectly reported 18 of 64 covered oblique landmarks as unavailable. Initial shader and probe hashes are retained. |
| `corrected-expanded487.json` | Corrected shader passed every coverage expectation. Two scalar ramps exceeded an initial, ad hoc 2-ppm value threshold narrowly. Its verifier is preserved as `corrected-expanded487-probe.cjs`. |
| `corrected-forward-bound487.json` | All 487 checks passed after replacing the arbitrary scalar threshold with the arithmetic bound described below; truth values and coverage expectations were unchanged. |
| `validated487.json` | Same successful checks after making the verifier return a nonzero process status on failure and refuse overwriting a receipt. This is the final receipt. |

The maximum observed scalar error was 4.905462265e-7 on an expected value of
0.22. The independent numerical allowance uses float32 input/coefficient
quantization, a seven-operation matrix-dot forward bound converted into value
units with each analytical fixture's per-axis slopes, texture-value
quantization, and a conservative 40-operation nonnegative trilinear bound.
For unit roundoff `u = 2^-24`, accumulation uses
`gamma(n) = n*u/(1-n*u)`. The maximum scalar allowance across these fixtures
is 6.1722e-6. This bound was introduced after observing the two initial scalar
threshold failures; both those failures and their unchanged expected values
remain visible. Boolean coverage checks have no numerical tolerance. Binary
0/1 values must match exactly. The bound applies to the declared fixtures, not
arbitrary source-grid conditioning or samples intentionally displaced by the
snapping convention.

Run the maintained verifier from the repository root with a new output path:

```sh
./desktop/node_modules/electron/dist/Electron.app/Contents/MacOS/Electron \
  desktop/electron/verify-prior-sampling.cjs \
  --output /tmp/ressectionlab-new-prior-sampling-receipt.json
```

The final verifier loads the exact CPU TypeScript helpers using Electron's
bundled Node runtime and invokes the production `priorAt` function in a
one-pixel float framebuffer. Expected affine landmarks, scalar values, and
coverage are defined independently. It does not change product code, source
arrays, planning models, or clinical eligibility.
