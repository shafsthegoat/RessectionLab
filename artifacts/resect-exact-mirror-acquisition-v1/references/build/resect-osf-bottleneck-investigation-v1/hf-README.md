---
license: cc-by-nc-sa-4.0
task_categories:
- image-segmentation
tags:
- medical
- brain
- ultrasound
- mri
- 3d
- intraoperative
- glioma
pretty_name: RESECT + RESECT-SEG (intra-operative brain tumor ultrasound)
dataset_info:
  config_name: preview
  features:
  - name: case_id
    dtype: string
  - name: task
    dtype: string
  - name: structure
    dtype: string
  - name: modality
    dtype: string
  - name: timepoint
    dtype: string
  - name: image_file
    dtype: string
  - name: mask_file
    dtype: string
  - name: num_slices
    dtype: int32
  - name: preview_slice
    dtype: int32
  - name: spacing_mm
    list: float32
  - name: image
    dtype: image
  - name: mask
    dtype: image
  - name: overlay
    dtype: image
  splits:
  - name: train
    num_bytes: 24232816
    num_examples: 181
  download_size: 24222537
  dataset_size: 24232816
configs:
- config_name: preview
  data_files:
  - split: train
    path: preview/train-*
---

# RESECT + RESECT-SEG

Combined re-host of the **RESECT** image database (Xiao et al. 2017) and the
**RESECT-SEG "All-Labels"** segmentations (Behboodi et al. 2024, OSF `jv8bk` v1
2024-02-15) — 23 low-grade glioma (WHO grade II) resections at St. Olavs
University Hospital, Trondheim (2011-2016). Pre-operative MRI (Gd-T1w, T2-FLAIR,
1 mm iso) and reconstructed 3D intra-operative B-mode ultrasound at three
timepoints (before / during / after resection; per-case isotropic grids,
0.14-0.27 mm). Case IDs are `Case1-8, 11-19, 21, 23-27` (9, 10, 20, 22 were
never released).

## Annotated tasks (181 image-mask pairs, `train.jsonl`)

| task | structure | volume | pairs | note |
|---|---|---|---|---|
| `tumor-us-before` | tumor | US before | 23 | primary CuRIOUS 2022 task |
| `resection_cavity-us-during` | resection cavity | US during | 21 | Case11+15 absent (blood-filled cavity, border indistinguishable) |
| `resection_cavity-us-after` | resection cavity | US after | 22 | Case11 absent |
| `tumor-flair` | tumor | pre-op FLAIR | 23 | int16 {0,1} masks |
| `sulci-us-{before,during,after}` | sulci | US x3 | 23+23+23 | |
| `falx-us-{before,during,after}` | cerebral falx | US x3 | 8+7+8 | only Cases 2,12,14,15,19,21,23,26; during missing for 26 |

Every mask sits on the **exact native grid of its image** (identical shape and
affine). Masks are binary {0,1} uint8 (FLAIR-tumor: int16). The 23 Gd-T1w
volumes carry no annotations (context only). An **absent mask file means the
structure was not annotated for that volume** — unlike the frozen CuRIOUS 2022
release (OSF `6y4db`), which shipped 3 of these as all-zero volumes. The jv8bk
masks are the authors' recommended, revised tier (not voxel-identical to 6y4db).

## Layout

```
dataset/Case{N}/   Case{N}-{T1,FLAIR,US-before,US-during,US-after}.nii.gz + all masks
landmarks/Case{N}/ MRI<->US / US<->US homologous landmark .tag files (registration)
train.jsonl        one row per pair: case_id, task, structure, modality, timepoint,
                   image, mask, shape, spacing_mm  (single split; no official split exists)
```

## Licensing

- **Images** (RESECT, NIRD archive, DOI [10.11582/2017.00004](https://doi.org/10.11582/2017.00004)): **CC BY 4.0**
- **Masks** (RESECT-SEG, [osf.io/jv8bk](https://osf.io/jv8bk/)): **CC BY-NC-SA 4.0** — "gaining any
  financial benefits from the distribution of the proposed annotation dataset is prohibited"
- Combined repo therefore carries **CC BY-NC-SA 4.0** (the more restrictive component).

## Benchmark contamination warning

These 23 patients ARE the training data of **CuRIOUS 2018/2019/2022** and (as
the resampled 22-case `EASY-RESECT` variant) of **Learn2Reg 2020**. Any model
trained on those challenge releases has seen these cases; cross-reference by
`case_id`. The challenges' private test cases (10 in 2018/19, 6 in 2022) were
never publicly released. No patient overlap with BITE, ReMIND, BraTS, MSD, or
TCIA collections. Note: `EASY-RESECT`'s `*-seg.nii.gz` files are voxelized
landmark spheres, NOT anatomy — they are unrelated to these masks and are not
included here.

## Citation

Please cite both:

1. Xiao Y, Fortin M, Unsgard G, Rivaz H, Reinertsen I. *REtroSpective Evaluation
   of Cerebral Tumors (RESECT): A clinical database of pre-operative MRI and
   intra-operative ultrasound in low-grade glioma surgeries.* Medical Physics
   2017;44(7):3875-3882. doi:10.1002/mp.12268
2. Behboodi B, Carton FX, Chabanas M, De Ribaupierre S, Solheim O, Munkvold BKR,
   Rivaz H, Xiao Y, Reinertsen I. *Open access annotations of intra-operative
   brain tumor ultrasound images.* Medical Physics 2024;51(9):6525-6532.
   doi:10.1002/mp.17317
