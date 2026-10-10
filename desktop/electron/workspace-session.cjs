'use strict';
const {validateEpisodeResult}=require('./development-episode.cjs');
/** Saved replay authority comes from backend semantic revalidation. Check the
 * response envelope before asset traversal; the renderer checks every prefix. */
function validateWorkspaceResult(result) {
  const session=result?.workspaceSession;
  if (session === undefined) return result; // Existing single-case bundles.
  if (!session || session.schema!=='integrated-workspace-session-v1' || session.referenceCaseHash!==result.caseHash ||
      !/^sha256:[a-f0-9]{64}$/.test(session.sessionHash) || !Array.isArray(session.displaySeries) || session.displaySeries.length>4)
    throw new Error('Reopened workspace has an invalid primary/source binding');
  const replay=session.episodeReplay;
  if (replay!==null) {
    if (!replay || replay.validation!=='authoritative_generated_replay' || replay.accountingQualification!=='historical_timings_not_remeasured' ||
        !Number.isSafeInteger(replay.frameIndex) || replay.frameIndex<0 || replay.frameIndex>=replay.episode?.replayFrames?.length ||
        typeof replay.visible!=='boolean' || (replay.visible && session.imagingState?.selectedSeriesId!==null))
      throw new Error('Reopened workspace replay was not revalidated for the primary image');
    validateEpisodeResult({case:result,...replay},{selector:replay.episode?.selector});
  }
  return result;
}
module.exports={validateWorkspaceResult};
