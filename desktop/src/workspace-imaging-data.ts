import { hydrateCase, validateCaseDescriptor } from './case-data.ts';
import { sourceFrameDigest } from './source-integrity.ts';
import type { CasePayload, ResectionApi, ViewerCase } from './types';
import type { DisplaySeriesPayload, DisplaySeriesView } from './workspace-imaging-types';
const HASH=/^sha256:[a-f0-9]{64}$/;
function validRegistration(primary:CasePayload,incoming:DisplaySeriesPayload) {
  const r=incoming.registration,a=incoming.association;
  if(!r || !a || a.sameTimeVerified!==false)return false;
  const association=(a.kind==='explicit_user_selected' && a.samePersonVerified===false) ||
    (a.kind==='source_dataset_identity' && a.samePersonVerified===true);
  if(!association)return false;
  if(r.status==='unreviewed')return r.overlayPermitted===false && r.sourceToReferenceRasMm===null && r.registrationHash===null;
  if(r.status!=='source-provided' || !r.overlayPermitted || !r.registrationHash || !HASH.test(r.registrationHash) ||
    a.kind!=='source_dataset_identity' || !a.samePersonVerified || incoming.modalityOrigin!=='source_metadata')return false;
  const declaration=primary.metadata.workspace_registration as Record<string,unknown>|undefined;
  const other=incoming.volume.metadata.workspace_registration;
  const transform=r.sourceToReferenceRasMm;
  // Source-dataset registration currently admits only an explicitly declared common grid.
  return !!declaration && declaration.kind==='source-provided-coregistration' && declaration.patient_id===primary.caseId &&
    incoming.volume.caseId===primary.caseId && JSON.stringify(declaration)===JSON.stringify(other) &&
    primary.frame===incoming.volume.frame && primary.shape.every((n,i)=>n===incoming.volume.shape[i]) &&
    primary.affine.every((row,i)=>row.every((v,j)=>Math.abs(v-incoming.volume.affine[i][j])<=1e-6)) &&
    Array.isArray(transform) && transform.length===4 && transform.every((row,i)=>Array.isArray(row)&&row.length===4&&row.every((n,j)=>n===(i===j?1:0)));
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
