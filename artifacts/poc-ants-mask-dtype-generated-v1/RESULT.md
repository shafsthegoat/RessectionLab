# Generated comparison identifies the N4 mask representation defect

The same deterministic48³ image, three nested classes and smooth multiplicative
bias field are processed with two value-identical binary masks. Only the native
mask representation changes: unsigned byte versus float32. The frozen N4
parameters, source code, installed runtime, physical frame and thread limits
remain identical. No patient image is read.

Both subprocesses exit successfully. That exit status conceals a material
difference in what N4 reads:

| Native mask | N4's reported masked intensity range | Nonpositive warning |
| --- | --- | --- |
| Unsigned byte | 0 to91.9219 | Yes |
| Float32 | 36.0215 to146.733 | No |

The declared generated mask contains only strictly positive source values in
the latter range. Both masks retain exact0/1 values, support and physical frames
when read through the Python API. The float arm then completes Atropos, native
frame checks and unmodified raw-class/posterior checks. All33,552 generated
class cells match this deliberately simple phantom; posterior sums differ from
one by at most3.04e-8. That is a generated integration control, not patient
segmentation accuracy. The two N4 output arrays differ by up to36.2123 intensity
units.

The exact build's [N4 C++ implementation](https://github.com/ANTsX/ANTs/blob/b53aad349e6767de8e1d99a392f05a82b1bf2373/Examples/N4BiasFieldCorrection.cxx)
uses a float mask. Its [in-memory image reader](https://github.com/ANTsX/ANTs/blob/b53aad349e6767de8e1d99a392f05a82b1bf2373/Utilities/ReadWriteData.h)
assumes the requested pointer type. The pinned Python N4 wrapper passes the
mask pointer without converting its pixel type. Our unsigned-byte mask adapter
violates that boundary contract. The [source record](source-basis.json) binds
the exact upstream commits and installed wrapper.

The [outcome](outcome.json), SHA256
`f7a0844d9569ddd8add755c167d316a0c8a7fd840aef76a9ce35ed8674a53102`,
and [comparative observations](observations.json), SHA256
`173a3c355b28fa38a9c89a0707f2a8ecb7716e2294dbe2e4ffa189f5fcc52332`,
retain both arms. They cost7.45/7.71 seconds with369,278,976/389,218,304-byte
sampled supervisor/family RSS. Outputs and temporary files total3,216,026 bytes;
no resource cap fires. Raw generated arrays remain local under
`outputs/poc-ants-mask-dtype-generated-v1/`.

The [independent generated-output audit](../ants-mask-dtype-independent-v1/independent-qa.json),
SHA256`9c4b3bc648b9920ba7de4c5fd0c921b08382796e2ead3467caffc4213ee0c128`,
regenerates the exact input/truth/mask, reproduces output hashes and arithmetic,
and verifies native-contract sources and resource records. It performs no ANTs
call or patient read. The NPY arrays have no embedded affine; frame checking uses
the separately sealed native geometry records.

This supports a separate typed-mask adapter with an explicit float pixel-type,
exact support and frame check before N4. It does not prove the specific cause
of the prior unsymbolicated patient crash, alter that failed outcome, or release
another patient attempt. Source intensities, fitting support and parameters
must remain unchanged in any later reviewed attempt. Anatomical, host and
growth acceptance remain false.
