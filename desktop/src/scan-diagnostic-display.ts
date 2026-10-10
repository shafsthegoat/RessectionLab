/** Research diagnostic metadata only. This is not DisplaySeriesPayload or planner input. */
export interface UnreviewedScanCandidatePatchV2 {
  schema: 'unreviewed_scan_candidate_patch_v2';
  display_kind: 'candidate_segmentation_unreviewed';
  coordinate_frame: 'atlas_RAS_mm_coded_sform';
  shape_xyz: [number, number, number];
  affine_ras_mm: [number[], number[], number[], number[]];
  channel_order: ['t1c', 't2f'];
  patch_shape_zyx: [128, 128, 128];
  patch_start_preprocessed_zyx: [number, number, number];
  source_sha256: {
    t1c: string; flair: string; support_map: string; plans: string; dataset: string;
  };
  prediction_coverage_status: 'unavailable_until_verified_output_receipt' | 'supplied_output_unverified';
  decoding_semantics: 'pinned_nnunet_probability_resample_then_regions';
  prediction_coverage_voxels?: number;
  positives_excluded_outside_coverage?: number;
  uncertainties: [
    'estimated_mask_anatomy_unreviewed',
    'registration_anatomy_unreviewed',
    'model_training_overlap_unknown',
    'fixed_patch_not_whole_volume'
  ];
  unknown_state: -1;
  negative_state: 0;
  positive_state: 1;
  anatomical_qc: 'unreviewed';
  planning_eligible: false;
  evaluation_eligible: false;
}

/** The corresponding XYZ arrays are generated-only until an exact output receipt is bound. */
export interface UnreviewedDiagnosticDisplayV2 {
  metadata: UnreviewedScanCandidatePatchV2;
  state_xyz: Int8Array; // -1 unknown, 0 negative, 1 candidate positive
  prediction_coverage_xyz: Uint8Array; // conservative supplied-output support intersected with bit 7
  positives_excluded_outside_coverage: number;
}
