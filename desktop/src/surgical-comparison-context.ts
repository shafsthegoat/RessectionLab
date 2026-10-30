import type {SurgicalResponse} from './surgical-types';

/** Immutable comparison evidence can accompany compact read-only frames.
 * It never grants edit/replay authority or crosses a source installation.
 */
export class SurgicalComparisonContext {
  private resident:{source:object;caseHash:string;sceneId:string;planning:SurgicalResponse['planning'];sectorPlanning:SurgicalResponse['sectorPlanning'];jointPlanning:SurgicalResponse['jointPlanning'];jointBaselineHistory:SurgicalResponse['jointBaselineHistory'];stagedPlanning:SurgicalResponse['stagedPlanning'];precisionPlanning:SurgicalResponse['precisionPlanning'];exitRecovery:SurgicalResponse['exitRecovery'];retractionStudy:SurgicalResponse['retractionStudy'];consequenceComparison:SurgicalResponse['consequenceComparison'];consequenceAnalysis:SurgicalResponse['consequenceAnalysis']}|null=null;
  clear(){this.resident=null;}
  accept(response:SurgicalResponse,source:object,frame:boolean):SurgicalResponse{
    const prior=this.resident;
    if(!frame){
      this.resident={source,caseHash:response.caseHash,sceneId:response.view.scene.sceneId,
        planning:response.planning,sectorPlanning:response.sectorPlanning,
        jointPlanning:response.jointPlanning,jointBaselineHistory:response.jointBaselineHistory,
        stagedPlanning:response.stagedPlanning,precisionPlanning:response.precisionPlanning,exitRecovery:response.exitRecovery,retractionStudy:response.retractionStudy,consequenceComparison:response.consequenceComparison,consequenceAnalysis:response.consequenceAnalysis};
      return response;
    }
    if(!prior||prior.source!==source||prior.caseHash!==response.caseHash||prior.sceneId!==response.view.scene.sceneId||response.playback?.readOnly!==true){
      this.clear();return response;
    }
    this.resident={...prior,
      ...(Object.hasOwn(response,'planning')?{planning:response.planning}:{}),
      ...(Object.hasOwn(response,'sectorPlanning')?{sectorPlanning:response.sectorPlanning}:{}),
      ...(Object.hasOwn(response,'jointPlanning')?{jointPlanning:response.jointPlanning}:{}),
      ...(Object.hasOwn(response,'jointBaselineHistory')?{jointBaselineHistory:response.jointBaselineHistory}:{}),
      ...(Object.hasOwn(response,'stagedPlanning')?{stagedPlanning:response.stagedPlanning}:{}),
      ...(Object.hasOwn(response,'precisionPlanning')?{precisionPlanning:response.precisionPlanning}:{}),
      ...(Object.hasOwn(response,'exitRecovery')?{exitRecovery:response.exitRecovery}:{}),
      ...(Object.hasOwn(response,'retractionStudy')?{retractionStudy:response.retractionStudy}:{}),
      ...(Object.hasOwn(response,'consequenceComparison')?{consequenceComparison:response.consequenceComparison}:{}),
      ...(Object.hasOwn(response,'consequenceAnalysis')?{consequenceAnalysis:response.consequenceAnalysis}:{})};
    const savedAnalysis=this.resident.consequenceAnalysis;
    // Retain saved policy evidence during observed frames, but never carry
    // current/editable analysis authority into another viewed state.
    const historicalAnalysis=savedAnalysis&&typeof savedAnalysis==='object'?{
      ...savedAnalysis,available:false,current:false,
      context:{branchId:response.branchId,position:response.position,stateId:response.view.stateId},
      reason:'Read-only observed replay. Return to the accepted common checkpoint and explicitly reassess.'
    }:savedAnalysis;
    return {...response,
      ...(!Object.hasOwn(response,'planning')?{planning:this.resident.planning}:{}),
      ...(!Object.hasOwn(response,'sectorPlanning')?{sectorPlanning:this.resident.sectorPlanning}:{}),
      ...(!Object.hasOwn(response,'jointPlanning')&&this.resident.jointPlanning!==undefined?
        {jointPlanning:this.resident.jointPlanning}:{}),
      ...(!Object.hasOwn(response,'jointBaselineHistory')&&this.resident.jointBaselineHistory!==undefined?
        {jointBaselineHistory:this.resident.jointBaselineHistory}:{}),
      ...(!Object.hasOwn(response,'stagedPlanning')&&this.resident.stagedPlanning!==undefined?
        {stagedPlanning:this.resident.stagedPlanning}:{}),
      ...(!Object.hasOwn(response,'precisionPlanning')&&this.resident.precisionPlanning!==undefined?
        {precisionPlanning:this.resident.precisionPlanning}:{}),
      ...(!Object.hasOwn(response,'exitRecovery')&&this.resident.exitRecovery!==undefined?
        {exitRecovery:this.resident.exitRecovery}:{}),
      ...(!Object.hasOwn(response,'retractionStudy')&&this.resident.retractionStudy!==undefined?
        {retractionStudy:this.resident.retractionStudy}:{}),
      ...(!Object.hasOwn(response,'consequenceComparison')&&this.resident.consequenceComparison!==undefined?
        {consequenceComparison:this.resident.consequenceComparison}:{}),
      ...(!Object.hasOwn(response,'consequenceAnalysis')&&historicalAnalysis!==undefined?
        {consequenceAnalysis:historicalAnalysis}:{})};
  }
}
