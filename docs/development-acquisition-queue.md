# Next development acquisitions: metadata audit

The existing manifests cover the UCSF-PDGM-0004 structural mirror and two
creator-source BTC cases, PAT28 and PAT05. No existing manifest selected the
next two cases. `manifests/development_acquisition_queue.json` extends the
unchanged PAT05 rule: select the first two remaining participant IDs, sorted
lexicographically, whose source diagnosis contains glioma, glioblastoma or
astrocytoma. Excluding PAT05 and PAT28 selects **PAT16 and PAT20**. Selection
was recorded before any image access; locations, route results and outcomes
were not selection inputs.

The source is [BTC_preop ds001226 v5.0.1](https://openneuro.org/datasets/ds001226/versions/5.0.1),
Git `359d372c5e972a161966312128adb365870df949`, under the release's
[CC0 license](https://github.com/OpenNeuroDatasets/ds001226/blob/359d372c5e972a161966312128adb365870df949/dataset_description.json).
The [pinned participant table](https://github.com/OpenNeuroDatasets/ds001226/blob/359d372c5e972a161966312128adb365870df949/participants.tsv)
provides the historical diagnosis strings below. These support retrospective
inclusion only; their availability before surgery is unknown and they remain
excluded from policy inputs.

| Candidate development group | Source diagnosis; location | Structural subject files | With directional diffusion |
| --- | --- | ---: | ---: |
| `BTC:sub-PAT16`, `ses-preop` | Anaplastic astrocytoma II-III; fronto-temporal | 18,195,685 bytes | 66,050,600 bytes |
| `BTC:sub-PAT20`, `ses-preop` | Anaplastic astrocytoma III; parietal | 17,531,332 bytes | 63,987,509 bytes |

The structural minimum is native T1, its acquisition JSON and the supplied
tumor annotation. Each diffusion extension adds AP/PA images, b-values,
b-vectors and acquisition JSON. Four shared release metadata files total
14,891 bytes, already present locally. Combined source totals, counting those
shared files once, are **35,741,908 bytes structural** or **130,053,000 bytes
with diffusion**. These totals exclude future transforms, preprocessing and
temporary derivatives.

For every image, the queue records the pinned Git annex pointer and its MD5,
source byte count, exact S3 object version and URL. Both unversioned and pinned
object HEAD requests returned matching lengths. Image SHA256 values remain
null: no image was fetched, opened, hashed or preprocessed. Pinned JSON and
gradient text was read and hashed. Each AP gradient table contains 102
b-values across 0, 700, 1200 and 2800 s/mm², with six b0 entries; each PA table
has two b0 entries. Corresponding b-vector tables have three rows. Sidecars
declare opposite `j-`/`j` encoding and a 0.0266003-second readout. These are
metadata checks, not image-volume, frame or reconstruction acceptance.

Both roles are locked to development in the queue; neither person may later
be assigned to outer-final evaluation. The acquired-case cohort registry was
not changed. Append development records linked to this declaration only after
source and geometry QC, and before optimization.
All visits, masks, reconstructions and simulated episodes must remain grouped
with their source person. The companion postoperative dataset ds002080 was
not opened and its subject correspondence remains unverified. Unknown
cross-collection or upstream-training overlap cannot support an independence
claim. UPenn and final assignments remain untouched.

The next implementation gate is bounded acquisition/preparation support: the
current maintained CLI accepts PAT28 and PAT05 only. Then verify downloaded
annex MD5/size and compute SHA256; inspect native affines, handedness and
mask alignment; preserve source annotation values; and obtain independently
reviewed brain/cortical support. Full-head signal must not define access.
Tract-aware use additionally needs image/gradient correspondence, gradient
frame handling, distortion and motion checks, diffusion-to-T1 alignment and
tumor-domain tract QC. Missing patient function remains unknown. Clinical
probability labels and molecular availability timestamps remain null.

This queue does **not** complete the specification's five-case UCSF milestone.
The documented official UCSF v5 transfer failure remains unresolved, and the
mirror's primary-release equivalence is unverified. Acquiring these two BTC
cases would produce five mixed-source development representations, four with
creator-source bytes, before anatomical QC. It would not establish five UCSF
cases, five tract-aware cases or the requested easy/motor/language/large-lesion/
failure strata. Those require actual anatomical and reconstruction review;
any selected failure must remain in the cohort flow rather than being silently
replaced. The manifest therefore leaves the executable five-UCSF-plan and
five-QC-case acceptance claims false.

Validation was limited to source metadata, exact object headers, internal byte
totals, selection reproduction and local image absence. No study run, heavy
test, acquisition or preprocessing was started.
