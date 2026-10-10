import { hydrateCase, validateCaseDescriptor } from './case-data.ts';
import { sourceFrameDigest } from './source-integrity.ts';
import type { CasePayload, ResectionApi, ViewerCase } from './types';
import type { DisplaySeriesPayload, DisplaySeriesView } from './workspace-imaging-types';
const HASH=/^sha256:[a-f0-9]{64}$/;
/** Validate before asset access. An auxiliary volume cannot replace the planning source. */
export async function hydrateDisplaySeries(primary:CasePayload, incoming:DisplaySeriesPayload, api:ResectionApi):Promise<DisplaySeriesView> {
  if (incoming.schema!=='workspace-display-series-v1' || incoming.referenceCaseHash!==primary.caseHash ||
      !HASH.test(incoming.seriesId) || !HASH.test(incoming.referenceFrameHash) || !HASH.test(incoming.sourceFrameHash) ||
      incoming.scope!=='display-only-native-grid' || incoming.planningEligible!==false || incoming.evaluationEligible!==false ||
      incoming.registration?.status!=='unreviewed' || incoming.registration.overlayPermitted!==false ||
      incoming.registration.sourceToReferenceRasMm!==null || incoming.registration.registrationHash!==null ||
      incoming.association?.kind!=='explicit_user_selected' || incoming.association.samePersonVerified!==false ||
      incoming.association.sameTimeVerified!==false || incoming.volume.metadata.workspace_display_only!==true)
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
