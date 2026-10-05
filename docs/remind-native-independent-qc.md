# Independent ReMIND-001 native conversion review

The independent audit verified all **386 DICOM objects (59,163,552 bytes)** against the acquisition SHA-256, pinned single-part ETag MD5, and byte count. Every saved MRI and SEG pixel plane exactly matches its original decoded source plane. The audit computes each plane’s physical corner directly from DICOM position, orientation, row/column spacing and the saved NIfTI affine; it does not call the converter’s fitting or reindexing helpers.

| Source | Maximum direct DICOM-to-NIfTI corner difference | qform-to-sform maximum corner difference |
|---|---:|---:|
| Automatic cerebrum annotation | 0.0000109 mm | 0.0004143 mm |
| T1 postcontrast | 0.0007827 mm | 0.0008685 mm |
| T2 | 0.0006367 mm | 0.0000811 mm |
| Manual tumor annotation | 0.0000096 mm | 0.0001219 mm |

All satisfy the declared 0.001 mm conversion tolerance. The qform differences reflect the orthogonal representation and precision; they must remain recorded when downstream registration uses that grid. The direct audit took 1.47 seconds, with 440.6 MiB reported peak RSS in the isolated acquisition environment.

The original cerebrum annotation contains 39,161,902 positive native voxels (~1,056,818 mm³); its source algorithm is **AUTOMATIC / BrainLab**. The tumor annotation contains 79,529 positive native voxels (~27,278 mm³); its source algorithm is **MANUAL**. These counts verify preservation, not segmentation accuracy. The released “cerebrum” label is not promoted to a reviewed whole-brain/cortical envelope.

Each SEG matches the frame identifier of its release-named MRI: cerebrum with T1, tumor with T2. Neither SEG contains explicit source SOP-instance references. T1 and T2 have **different frame identifiers**. Matching source descriptions and frame IDs are limited correspondence evidence; cross-sequence identity and expert alignment acceptance remain false.

Six overlays, sampled through each annotation’s native centroid, were visually inspected. They show no obvious gross axis reversal or displaced annotation in these sampled planes. The source SEG field of view is outlined separately: outside it remains unassessed. This inspection is not an expert anatomical review, full-volume segmentation validation, or cortical-access approval.

The reproducible audit is `scripts/audit_remind_native.py`; the detailed receipt, exact source hashes, overlay rendering code, PNGs and per-plane coverage receipt are under `artifacts/remind-native-independent-qc-v1/`. Native sources and converted files were not modified. Any T1/T2 registration is a separate, review-required derivative.
