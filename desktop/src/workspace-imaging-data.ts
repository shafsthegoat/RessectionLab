import { hydrateCase, validateCaseDescriptor } from './case-data.ts';
import { sourceFrameDigest } from './source-integrity.ts';
import type { CasePayload, ResectionApi, ViewerCase } from './types';
import type { DisplaySeriesPayload, DisplaySeriesView } from './workspace-imaging-types';
const HASH=/^sha256:[a-f0-9]{64}$/;
export function displaySeriesLabel(series:DisplaySeriesPayload):string {
  const annotation=series.annotationKind==='estimated'?'model estimate':series.annotationKind==='source-provided'?'source labels':'image';
  const caption=series.volume.metadata.display_label;
  const name=typeof caption==='string'&&caption.trim()&&caption.length<=80&&!/[\u0000-\u001f\u007f]/.test(caption)?caption.trim():series.modality;
  return `${name} · ${annotation} · ${series.registration.overlayPermitted?(series.registration.status==='verified'?'reviewed derived alignment':'source registered'):'separate frame'}`;
}
function validRegistration(primary:CasePayload,incoming:DisplaySeriesPayload) {
  const r=incoming.registration,a=incoming.association;
  if(!r || !a || a.sameTimeVerified!==false)return false;
  const association=(a.kind==='explicit_user_selected' && a.samePersonVerified===false) ||
    (a.kind==='source_dataset_identity' && a.samePersonVerified===true);
  if(!association)return false;
  if(r.status==='unreviewed')return r.overlayPermitted===false && r.sourceToReferenceRasMm===null && r.registrationHash===null;
  if(!['source-provided','verified'].includes(r.status) || !r.overlayPermitted || !r.registrationHash || !HASH.test(r.registrationHash) ||
    a.kind!=='source_dataset_identity' || !a.samePersonVerified || incoming.modalityOrigin!=='source_metadata')return false;
  const declaration=primary.metadata.workspace_registration as Record<string,unknown>|undefined;
  const other=incoming.volume.metadata.workspace_registration;
  const transform=r.sourceToReferenceRasMm;
  // Source-dataset registration currently admits only an explicitly declared common grid.
  const common=!!declaration && declaration.patient_id===primary.caseId &&
    incoming.volume.caseId===primary.caseId && JSON.stringify(declaration)===JSON.stringify(other) &&
    primary.frame===incoming.volume.frame && primary.shape.every((n,i)=>n===incoming.volume.shape[i]) &&
    primary.affine.every((row,i)=>row.every((v,j)=>Math.abs(v-incoming.volume.affine[i][j])<=1e-6)) &&
    Array.isArray(transform) && transform.length===4 && transform.every((row,i)=>Array.isArray(row)&&row.length===4&&row.every((n,j)=>n===(i===j?1:0)));
  if(!common)return false;
  if(r.status==='source-provided')return declaration!.kind==='source-provided-coregistration';
  const rawHash=(value:unknown)=>typeof value==='string'&&/^[a-f0-9]{64}$/.test(value);
  const dataCase=primary.metadata.data_case as Record<string,unknown>|undefined;
  const otherCase=incoming.volume.metadata.data_case as Record<string,unknown>|undefined;
  const binding=declaration!.source_sha256_by_modality as Record<string,unknown>|undefined;
  const primaryRef=primary.sourceRefs?.find(row=>row.source_id==='structural');
  const otherRef=incoming.volume.sourceRefs?.find(row=>row.source_id==='structural');
  const matrix=(value:unknown):value is number[][]=>Array.isArray(value)&&value.length===4&&value.every(row=>Array.isArray(row)&&row.length===4&&row.every(Number.isFinite));
  const forward=declaration!.original_moving_to_reference_ras_mm,pull=declaration!.reference_to_original_moving_ras_mm;
  if(!matrix(forward)||!matrix(pull))return false;
  const keys=['kind','patient_id','source_person_id','group_id','source_sha256_by_modality','evidence_sha256','evidence_path','reviewer','review_decision','method_report_sha256','original_moving_image_sha256','original_annotation_sha256','original_moving_to_reference_ras_mm','reference_to_original_moving_ras_mm','registered_grid'];
  if(Object.keys(declaration!).length!==keys.length||keys.some(key=>!(key in declaration!)))return false;
  const rigid=(m:number[][])=>m[3].every((n,i)=>Math.abs(n-(i===3?1:0))<1e-8)&&m.slice(0,3).every((_,i)=>m.slice(0,3).every((_,j)=>Math.abs(m.slice(0,3).reduce((s,row)=>s+row[i]*row[j],0)-(i===j?1:0))<1e-5))&&
    m[0][0]*(m[1][1]*m[2][2]-m[1][2]*m[2][1])-m[0][1]*(m[1][0]*m[2][2]-m[1][2]*m[2][0])+m[0][2]*(m[1][0]*m[2][1]-m[1][1]*m[2][0])>0;
  return declaration!.kind==='independently-reviewed-derived-registration-v1'&&
    ['accepted_for_bounded_engineering_alignment','accepted_for_registered_overlay_inspection_with_alignment_uncertainty'].includes(String(declaration!.review_decision))&&declaration!.registered_grid==='reference-native-grid'&&
    typeof declaration!.group_id==='string'&&!!declaration!.group_id.trim()&&
    typeof declaration!.reviewer==='string'&&!!declaration!.reviewer.trim()&&
    typeof declaration!.evidence_path==='string'&&!/^(\/|\\)/.test(declaration!.evidence_path)&&!declaration!.evidence_path.replaceAll('\\','/').split('/').includes('..')&&
    ['evidence_sha256','method_report_sha256','original_moving_image_sha256','original_annotation_sha256'].every(key=>rawHash(declaration![key]))&&
    typeof declaration!.source_person_id==='string'&&declaration!.source_person_id===dataCase?.person_id&&declaration!.source_person_id===otherCase?.person_id&&
    dataCase?.deformation==null&&otherCase?.deformation==null&&!!primaryRef&&!!otherRef&&rawHash(primaryRef.sha256)&&rawHash(otherRef.sha256)&&
    binding?.[String(primary.metadata.selected_modality)]===primaryRef.sha256&&binding?.[String(incoming.volume.metadata.selected_modality)]===otherRef.sha256&&
    rigid(forward)&&rigid(pull)&&forward.every((row,i)=>row.every((_,j)=>Math.abs(row.reduce((sum,n,k)=>sum+n*pull[k][j],0)-(i===j?1:0))<1e-6));
}
/** Validate before asset access. An auxiliary volume cannot replace the planning source. */
export async function hydrateDisplaySeries(primary:CasePayload, incoming:DisplaySeriesPayload, api:ResectionApi):Promise<DisplaySeriesView> {
  if (incoming.schema!=='workspace-display-series-v1' || incoming.referenceCaseHash!==primary.caseHash ||
      !HASH.test(incoming.seriesId) || !HASH.test(incoming.referenceFrameHash) || !HASH.test(incoming.sourceFrameHash) ||
      incoming.scope!=='display-only-native-grid' || incoming.planningEligible!==false || incoming.evaluationEligible!==false ||
      !validRegistration(primary,incoming) || incoming.volume.metadata.workspace_display_only!==true)
    throw new Error('Display series does not match this case or its native-grid inspection contract.');
  validateCaseDescriptor(primary);validateCaseDescriptor(incoming.volume);
  const [referenceFrame,sourceFrame]=await Promise.all([sourceFrameDigest(primary),sourceFrameDigest(incoming.volume)]);
  if (referenceFrame!==incoming.referenceFrameHash || sourceFrame!==incoming.sourceFrameHash)
    throw new Error('Display series physical-frame identity does not match its image or reference.');
  const volume=await hydrateCase(incoming.volume,api);
  return {descriptor:incoming,volume};
}
/** Only primary display admits primary routes, estimates, priors or replay. */
export function workspaceDisplay(primary:ViewerCase|null, selected:DisplaySeriesView|null) {
  if (selected && (!primary || selected.descriptor.referenceCaseHash!==primary.caseHash))
    throw new Error('The selected display series belongs to a stale workspace.');
  return {caseData:selected?.volume ?? primary, primaryOverlaysPermitted:selected===null,
    planningInteractionPermitted:selected===null, registrationReason:selected?.descriptor.registration.reason ?? null};
}
