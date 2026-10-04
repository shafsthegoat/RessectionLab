# One RESECT displacement development case

**Case4 is fixed prospectively**, using the smallest total source-file size
among 17 cases with all six required files. The official archive provides
individual objects, so no cohort download is needed. The declaration is
`manifests/experiments/resect-case4-displacement-acquisition-v1.json`.

| File purpose | Bytes | Provider MD5 |
| --- | ---: | --- |
| Preoperative T1 | 8,763,797 | `e2db363c6ce57e83663b4de2f95c2ed0` |
| Preoperative FLAIR | 6,550,243 | `e50cfb97bbf1c4424034889413bc2fcc` |
| Before-resection ultrasound | 9,955,302 | `984ab4b314a55b014e770b32e623b16c` |
| During-resection ultrasound | 9,796,312 | `14b8b7acee3beccd4ee04e96e9201f0c` |
| Full before/during landmark pairs | 1,304 | `14259fc0f9ab32dfecbe1f2b8830ae86` |
| MRI/before-US landmark pairs | 1,052 | `19b7fa7c07322c98647cfa99eaa53fc4` |

Total: **35,068,010 bytes**. All six anonymous S3 HEAD requests returned 200
and the declared lengths. Their ETags are storage identifiers, not checksums.
The [official NIRD record](https://archive.sigma2.no/dataset/5D6BFC33-F58D-4F56-88E8-C40AF269D6F2)
identifies version 1, DOI `10.11582/2017.00004`, released May 30, 2017,
and CC BY 4.0. Its table of contents supplies the sizes and fixity values;
[NIRD documentation](https://documentation.sigma2.no/nird_archive/downloading-datasets.html)
defines these as MD5. No provider SHA256 is supplied. Local SHA256 values
must be recorded after a separately released acquisition.

Only the 292,738-byte table of contents, landing metadata and HEAD responses
were obtained. The five compact evidence files total 42,654 bytes under
`artifacts/mechanics/resect-case4-acquisition-metadata-v1/`. No image or tag
body has been downloaded. The provider's padded, pipe-delimited table also
contains a separator line; the retained ranking uses normalized metadata.

The [creator's guide](https://www.healthx-lab.ca/databases.html) specifies
world coordinates in millimetres, reference-image coordinates first, and the
same world coordinates in corresponding MINC/NIfTI images. The
[methods paper](https://doi.org/10.1002/mp.12268) identifies FLAIR as the MRI
landmark reference, with T1 rigidly aligned to it. Native headers, anatomical
axis direction, actual tag ordering and point/image correspondence remain
unverified. The larger before/during set overlaps the smaller three-stage
set, so filenames cannot define independent measurement partitions.

This is a **development conditional-displacement** proof of concept. Paper
retrieval after the fixed size ranking incidentally exposed published
aggregate displacement summaries; the case is not a pristine external test.
No aggregate result affected selection. All Case4 timepoints, formats and
verified derivatives belong to one patient group. Existing patient roles,
the unopened BTC transfer cases and the reserved UPenn final cohort stay
unchanged. Unknown derivative mappings cannot support independence claims.

Root must commit the declaration, and imaging must freeze measurement and
coordinate-review rules, before values are read. Target coordinates or later
image overlays must not tune validation, material, mesh, boundary, alignment
or stopping choices. No reviewed brain/cavity mask is included in these six
files. Acquisition will not approve a domain, access window, anatomical
segmentation, instrument force model or patient-specific material parameters.
