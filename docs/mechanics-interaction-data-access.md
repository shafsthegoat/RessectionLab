# Bounded interaction-measurement access finding

Verification UTC: **2026-10-05 01:25:38 UTC** (October 4 in the project timezone).
Metadata and ordinary web pages only; no force, displacement, image or other
experimental arrays were downloaded or read. No experiment, donor role, patient
split or validation gate changed.

**MULTIS is a concrete, publicly accessible candidate for a nonbrain ex-vivo
interaction benchmark.** The [static release](http://archive.simtk.org/multisdelta/)
lists nine donor directories, `SMULTIS004-1` through `SMULTIS012-1`.
Its [release license](http://archive.simtk.org/multisdelta/license.txt) and
[project licensing page](https://simtk.org/plugins/moinmoin/multis/Licensing.html)
explicitly cover project data under **CC BY 4.0**. The dataset DOI is
[10.18735/n217-mb65](https://doi.org/10.18735/n217-mb65);
[DataCite metadata](https://api.datacite.org/dois/10.18735/n217-mb65) points to the
working HTTP archive. HTTPS to that archive refused connection in this check.
The separate [dynamic service](https://multisdelta.stanford.edu/) presented a
login form; the static directory needed no account.

## Measurements and claim boundary

The [primary descriptor](https://pmc.ncbi.nlm.nih.gov/articles/PMC6962378/)
reports human cadaver legs, instrumented tool motion and six-axis loads:

| Trials | Independently measured response | Limitation |
| --- | --- | --- |
| Initial skin indentation, pinching and skin cut | VIC-3D surface displacement/strain, stereo images and calibration, together with tool pose and loads | Surface measurements; raw completeness, contact/fixture reconstruction and coordinate registration remain unaudited |
| Later tissue retraction | Tool pose and force/torque | No corresponding VIC-3D surface-motion acquisition; cannot alone validate the tissue displacement field |

The descriptor places transformed loads at each tool tip and transformed tool
positions in the femur anatomical frame. Forces are reported in N and moments
in N m. Sampling is 1,000 Hz for loads and approximately 30 Hz for motion.
VIC-3D has recorded synchronization signals. Reported motion/load tap offsets
are 71.5 +/- 32.5 ms for indentation, 48.5 +/- 22.5 ms for forceps, and
32.5 +/- 23.5 ms for retraction; these were not corrected in the study.
Native position/angle units, orientation convention, camera-to-tool registration,
trial-to-camera pairing and synchronization adequacy need schema inspection.

This narrows the access dependency for comparable soft-tissue interaction; it
does not establish live-patient brain mechanics, brain retraction, independent
interior displacement, or any passed physical-validation gate. The existing
[validation contract](tissue-mechanics-validation.md) remains unchanged.

## Exact metadata access evidence

Ordinary unauthenticated HTTP requests returned the following. GET counts in
this table are decoded text characters inspected, not measured scientific values. The donor
directory is an access example, not a prospectively selected experiment.

| Method and exact URL | Observed result |
| --- | --- |
| GET `https://api.datacite.org/dois/10.18735/n217-mb65` | 200; 6,731 characters; archive URL supplied, `rightsList` empty |
| GET `http://archive.simtk.org/multisdelta/` | 200; 3,803 characters; donor listing and explicit CC BY 4.0 statement |
| GET `http://archive.simtk.org/multisdelta/license.txt` | 200; 234 characters; CC BY 4.0 |
| GET `https://simtk.org/plugins/moinmoin/multis/Licensing.html` | 200; 5,912 characters; project data explicitly covered |
| GET `http://archive.simtk.org/multisdelta/SMULTIS004-1/` | 200; 1,588 characters; Configuration, Data, DataQuality and VIC3D directories |
| GET `http://archive.simtk.org/multisdelta/SMULTIS004-1/Data/` | 200; 7,343 characters; individually addressable TDMS trials including indentation and retraction |
| GET `http://archive.simtk.org/multisdelta/SMULTIS004-1/Configuration/` | 200; 22,243 characters; coordinate-system, sensor and state filenames; file contents unopened |
| GET `http://archive.simtk.org/multisdelta/SMULTIS004-1/VIC3D/` | 200; 1,635 characters; Calibration, SXX_IND_SKN, FXX_PCH_SKN and SXX_CUT_SKN directories |
| GET `http://archive.simtk.org/multisdelta/SMULTIS004-1/VIC3D/SXX_IND_SKN/` | 200; directory HTML inspection capped at 120,000 bytes; CSV/OUT exports and paired TIFF filenames visible; incomplete listing |
| GET `http://archive.simtk.org/multisdelta/SMULTIS004-1/VIC3D/FXX_PCH_SKN/` | 200; directory HTML inspection capped at 120,000 bytes; CSV/OUT exports and paired TIFF filenames visible; incomplete listing |
| HEAD `http://archive.simtk.org/multisdelta/SMULTIS004-1/Data/001_SMULTIS004-1_SXX_IND_SKN-1.tdms` | 200; Content-Length 6,952,418; Accept-Ranges bytes; no body read |
| GET `https://multisdelta.stanford.edu/` | 200; 4,150 characters; login form |

These checks establish readable release metadata and one raw-file HEAD response,
not raw-payload integrity, complete paired trials, or simulation readiness.
The [previous brain-retraction phantom paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC4082653/)
still supplies a relevant measurement precedent, without verified raw access
from this bounded search. That access finding is not a claim that no such brain
data exist elsewhere.
