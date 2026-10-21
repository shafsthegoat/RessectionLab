# One-case TopBrain CTA source notice

The same-patient CTA image and vessel labels are from **TopBrain Challenge Organizers, TopBrain Challenge Data Release, Batch1 July 30, 2025**: https://zenodo.org/records/16623496 (DOI 10.5281/zenodo.16623496). Challenge website: https://topbrain2025.grand-challenge.org. Cite the TopBrain website and Yang et al., *Benchmarking the CoW with the TopCoW Challenge: Topology-Aware Anatomical Segmentation of the Circle of Willis for CTA and MRA*, https://arxiv.org/abs/2312.17670.

The fixed first CTA case, `topcow_ct_001`, was recovered with its exact native-grid `labelsTr_topbrain_ct/topcow_ct_001.nii.gz` annotations, label-map text, README, and License by bounded HTTP 206 requests from the official 1,134,146,924 byte ZIP. **30,287,550 bytes** total were received, including a 131,072 byte ZIP directory tail. The full archive was not downloaded and its published MD5 was not independently verified. Each recovered member was verified against ZIP header/central-directory name, size, method, and CRC32; individual SHA256 hashes are recorded in the acquisition receipt. Original NIfTI bytes and affines are retained in the collection. No source intensity or original categorical label was edited.

**License:** noncommercial use with source attribution. Commercial use requires prior permission from the owner, University Hospital of Zurich (USZ), Department of Neurology, Rämistrasse 100, 8091 Zürich. Permission has not been obtained. The exact publisher License is copied alongside this notice. This pilot uses the data for noncommercial research/display evaluation; the license does not impose a local-only or no-publication restriction.

The published CTA is a cropped braincase image, 284 × 327 × 243 voxels, grid 0.507813 × 0.507813 × 0.625 mm, approximately 144 × 166 × 152 mm. It is not a full uncropped head/neck scan or original DICOM. It contains actual contrast-enhanced CT source intensities, including dense osseous structures; annotation masks are separately supplied derived labels. Source label and CTA grids/affines match exactly. 28 of 40 vessel classes are present. Left ICA and left MCA label classes are absent in this case; no cause, anatomy, or diagnosis is inferred, and no symmetry or missing vessels were fabricated.

The native collection has two exact source-label unions: arterial values 1–34 (`vessels`), 44,951 voxels; venous/sinus values 35–40 (`venous-vessels`), 34,226 voxels. The grouped layer semantic kinds are `artery` and `vein`, respectively, matching the publisher categorical identities. This is a metadata correction, not anatomy repair or our inference. All components are retained; no interpolation, gap filling, model inference, registration, or cross-patient fusion was performed. The positive union has 5 connected components under 26 connectivity; this count is descriptive, not an accuracy or topology-validation claim.

The third layer, **High-attenuation CT context · threshold estimate**, is a proposal from native scaled CT intensity >= 700 with every supplied positive vascular label excluded. It is not an anatomical bone class or a vetted skull segmentation. Unannotated vessels, contrast, and calcification above the threshold can remain. It does not modify the native scalar CTA or original source vessel labels. No brain surface or tumor was inferred. All native display layers retain local review status `unreviewed`.

Native collection: `outputs/poc-angio-fidelity-v2/native-cta/patient-anatomy.json`.

Original CTA SHA256: `9dea5c8138e6884bdd9480d16ac4ed05f6a61ed9cccde658b2052d4d627030b3`.

Original categorical vessel-label SHA256: `f65ee1b03bf776f33c55ef0db314e8a3d4122ccaea06499f73219995c1bfa5f0`.
