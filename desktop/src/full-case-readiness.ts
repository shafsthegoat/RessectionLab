import type {FullCaseReadiness,SurgicalResponse} from './surgical-types';

/** Readiness is evidence for one accepted case, scene and state, never a cache. */
export function boundFullCaseReadiness(response:SurgicalResponse|null|undefined,caseHash:string|undefined):FullCaseReadiness|null {
  const readiness=response?.fullCaseReadiness;
  if(!response||!caseHash||response.caseHash!==caseHash||!readiness||readiness.schema!=='resectionlab.full-case-readiness.v1'||
    readiness.caseHash!==caseHash||!response.view?.scene?.sceneId||readiness.sceneId!==response.view.scene.sceneId||
    !response.view.stateId||readiness.stateId!==response.view.stateId||!['ready','blocked'].includes(readiness.status)||
    !Array.isArray(readiness.availableActions)||!Array.isArray(readiness.blockers))return null;
  return readiness;
}

export function fullCaseScenarioAvailable(response:SurgicalResponse|null|undefined,caseHash:string|undefined,scenario:'primary'|'vascular'):boolean {
  const readiness=boundFullCaseReadiness(response,caseHash);
  return !!readiness&&readiness.status==='ready'&&readiness.blockers.length===0&&readiness.availableActions.includes(scenario)&&
    !response?.readOnly&&!response?.playback?.readOnly&&!response?.verification?.readOnly;
}
