/** Proposed separate ViewerWorkspace layer. This is not a DisplaySeriesPayload. */
export interface UnreviewedCase4DiagnosticLayerV1 {
  schema: 'case4_unreviewed_diagnostic_display_layer_v1';
  scope: 'display_only_atlas_native_grid';
  output_origin: 'model_generated_from_observed_case4_preoperative_scans';
  model_output_verified: true;
  shape_xyz: [number, number, number];
  affine_ras_mm: [number[], number[], number[], number[]];
  source_sha256: {
    t1c: string; flair: string; support_map: string; plans: string; dataset: string;
  };
  geometry_sha256: string;
  input_receipt_sha256: string;
  forward_receipt_sha256: string;
  outputs: {
    state: { path: string; sha256: string; dtype: 'int8' };
    coverage: { path: string; sha256: string; dtype: 'uint8' };
  };
  state_semantics: {
    '-1': 'unknown';
    '0': 'candidate_negative_where_output_covered';
    '1': 'candidate_positive_where_output_covered';
  };
  predicted_voxels: number;
  unknown_voxels: number;
  excluded_positive_voxels: number;
  anatomical_qc: 'unreviewed_inferior_mask_omission';
  training_overlap_status: 'unknown';
  same_grid_only: true;
  planning_eligible: false;
  evaluation_eligible: false;
  clinical_evidence: false;
}

/** Mount only after the chosen local display image's source hash and RAS frame
 * match this layer. Unknown registration suppresses primary routes/replay;
 * never copy these voxels into CaseData compartments or critical evidence. */
export interface UnreviewedDiagnosticLayerView {
  descriptor: UnreviewedCase4DiagnosticLayerV1;
  stateXYZ: Int8Array;
  coverageXYZ: Uint8Array;
  primaryOverlaysPermitted: false;
  planningInteractionPermitted: false;
}
