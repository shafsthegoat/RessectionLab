# Optional local diffusion correction tools

Checked October 4, 2026. The FSL backend is **not installed or executed**. The
completed work is a reproducible package/metadata preflight and a guarded
optional installer. Raw scans and gradients remain unchanged. No correction,
rotated gradients, validated tracts, or approved functional coverage resulted
from this investigation.

## Local package feasibility

[FSL documents native Apple Silicon support](https://fsl.fmrib.ox.ac.uk/fsl/docs/install/macos.html)
and [installation of selected packages in separate conda environments](https://fsl.fmrib.ox.ac.uk/fsl/docs/install/conda.html).
The official arm64 package index contains these pinned candidates:

| Package | Version/build | Compressed bytes |
|---|---|---:|
| `fsl-eddy` | `2602.0 / hed4507f_0` | 1,033,532 |
| `fsl-topup` | `2203.6 / ha7be7b2_0` | 341,705 |

These are direct package sizes, **not total installation sizes**. Dependencies
have not been resolved or downloaded. The preflight report retains official
URLs, SHA-256/MD5, package dependencies, and the package-index hash. Missing
license metadata in the conda index does not imply a permissive license.

The optional installer uses `.tools/fsl` and a separate `.tools/mamba` cache.
It requires `.tools/` to be ignored, an existing micromamba executable, and a
documented permitted use scope. It does not install a global package manager,
edit shell profiles, install Rosetta, acquire GPU packages, or create cloud
resources. Official [micromamba installation instructions](https://mamba.readthedocs.io/en/stable/installation/micromamba-installation.html)
describe a standalone arm64 executable if one is needed later.

## License finding

The [FSL license](https://fsl.fmrib.ox.ac.uk/fsl/docs/license.html) restricts its
default terms to noncommercial use. Its commercial-use definition includes
research directed toward products intended for sale or licensing, as well as
paid external services. Its redistribution provisions also impose source and
license conditions. A prospective company-use goal therefore requires scope
clarification or appropriate permission; this repository does not establish
such permission. FSL will not be bundled with the app.

The official [installer source](https://fsl.fmrib.ox.ac.uk/fsldownloads/fslconda/releases/fslinstaller.py)
was inspected as text. Its `agree_to_license()` function explains that
installation implies agreement and offers cancellation. It is not a separate
yes/no prompt. Public documentation, installer text, and package metadata were
read without running an installer. No FSL package was downloaded or accepted.

## Actual PAT28 acquisition checks

The preflight reads the eight preserved AP/PA image, JSON, b-value, and b-vector
files described in [the source record](diffusion_source.md). It records their
hashes and compares physical grids and acquisition parameters.

| Check | Result |
|---|---|
| AP image/gradient count | 102 volumes and matching gradients; six b0 volumes |
| PA image/gradient count | Two b0 volumes and matching zero gradients |
| Encoding direction | JSON AP `j-`, PA `j`; opposite image-axis encoding |
| Recorded readout time | Both `0.0266003` seconds |
| Echo-spacing consistency | `0.000380004 × (96 − 1) = 0.03610038` seconds |
| AP/PA grid discrepancy | Maximum matching-corner separation `2.369496143` mm |
| Raw modifications/correction | None |

The readout comparison uses the [BIDS definition](https://bids-specification.readthedocs.io/en/v1.11.0/modality-specific-files/magnetic-resonance-imaging-data.html),
with a 1% discrepancy flag. The sources were converted by dcm2niix
`v1.0.20170724`. The conflict is retained; the script does not decide which
metadata field is wrong. Both affines have the same array orientation and
dimensions but differ by approximately 0.8 degrees in plane and a translation.
The 0.01 mm corner-equality gate detects that they are not the same physical
grid; it is not an anatomical accuracy tolerance.

The [TOPUP documentation](https://fsl.fmrib.ox.ac.uk/fsl/docs/diffusion/topup/index.html)
explains its joint motion/distortion model. An arbitrary image registration
before TOPUP is not justified merely because it improves visual overlap.
Before running this pair, document how the distinct acquisition grids are
handled without erasing the source transforms or incorrectly changing the
encoding axes. The [TOPUP FAQ](https://fsl.fmrib.ox.ac.uk/fsl/docs/diffusion/topup/FAQ/index.html)
explains why a common readout scaling can cancel when used consistently by
TOPUP and EDDY; that motivates a separately labeled sensitivity check, not
silent substitution or a claim that the metadata are verified.

Use the original dcm2niix gradient convention only with a documented frame
audit. Any future motion-corrected DWI must be paired with EDDY's generated
rotated b-vectors, checked for count, finite values, norm and frame. The
[EDDY guide](https://fsl.fmrib.ox.ac.uk/fsl/docs/diffusion/eddy/users_guide/index.html)
explicitly requires the input vectors to have the correct FSL convention.
Keep original and rotated files distinct; registration of a mean b0 image
does not itself rotate the entire diffusion acquisition.

## Reproduction and result artifacts

```sh
.venv/bin/python scripts/setup_diffusion_tools.py --self-test
.venv/bin/python scripts/setup_diffusion_tools.py --probe
```

The first command passed ten synthetic acquisition/license-gate checks. The
second completed the real source preflight and recorded both FSL commands as
`not_installed`. The saved result is
`artifacts/preprocessing/PAT28-fsl-preflight-v1/report.json`; its patient status
is `blocked` with `AP_READOUT_METADATA_CONFLICT`,
`PA_READOUT_METADATA_CONFLICT`, and `AP_PA_GRID_RECONCILIATION_REQUIRED`.
The outer status records completion of the preflight, not successful correction.

After a permitted scope is established, `--install --license-record PATH
--micromamba PATH` enables the optional path. A scope record requires
`permission_status: confirmed`, a `scope` of `noncommercial_research` or
`commercial_permission`, and nonempty `confirmed_by`, `basis`, and `reference`
fields. Such a record documents an actual decision; it does not grant rights.
The solver runs first, refuses unknown or over-budget package sizes (400 MiB
default), and saves exact package URLs/checksums before installation. Existing
prefixes are preserved. Failures and logs remain inspectable. Installation and
installed-binary help checks remain untested while this gate is unresolved.

## Independent work that can continue

[DIPY uses a BSD-style license](https://github.com/dipy/dipy/blob/master/LICENSE)
and documents [between-volume motion correction](https://docs.dipy.org/1.8.0/examples_built/preprocessing/motion_correction.html).
That offers a candidate motion-only experiment with explicit gradient
reorientation validation; it is not a demonstrated replacement for TOPUP plus
EDDY susceptibility and eddy-current correction. The existing
[SimpleITK Apache-2.0](https://simpleitk.org/about.html) rigid mean-b0/T1
registration experiment can proceed with its raw-data status visible, then be
rerun on a corrected baseline later.

TORTOISE was also screened as an alternative. Its official project describes
DIFFPREP/DR-BUDDI correction, but its [external-software acknowledgment](https://www.tortoise.nibib.nih.gov/)
includes FSL BET2 for masking. Its [current repository license](https://github.com/QMICodeBase/TORTOISEV4/blob/main/LICENSE)
is GPL rather than MIT/BSD. This screening does not establish a clean
permissive-license substitute or a tested Apple Silicon build.
