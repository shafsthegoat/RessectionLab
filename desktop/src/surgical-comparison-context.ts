import type {SurgicalResponse} from './surgical-types';

/** Immutable comparison evidence can accompany compact read-only frames.
 * It never grants edit/replay authority or crosses a source installation.
 */
export class SurgicalComparisonContext {
  private resident:{source:object;caseHash:string;sceneId:string;planning:SurgicalResponse['planning'];sectorPlanning:SurgicalResponse['sectorPlanning']}|null=null;
  clear(){this.resident=null;}
  accept(response:SurgicalResponse,source:object,frame:boolean):SurgicalResponse{
    const prior=this.resident;
    if(!frame){
      this.resident={source,caseHash:response.caseHash,sceneId:response.view.scene.sceneId,
        planning:response.planning,sectorPlanning:response.sectorPlanning};
      return response;
    }
    if(!prior||prior.source!==source||prior.caseHash!==response.caseHash||prior.sceneId!==response.view.scene.sceneId||response.playback?.readOnly!==true){
      this.clear();return response;
    }
    this.resident={...prior,
      ...(Object.hasOwn(response,'planning')?{planning:response.planning}:{}),
      ...(Object.hasOwn(response,'sectorPlanning')?{sectorPlanning:response.sectorPlanning}:{})};
    return {...response,
      ...(!Object.hasOwn(response,'planning')?{planning:this.resident.planning}:{}),
      ...(!Object.hasOwn(response,'sectorPlanning')?{sectorPlanning:this.resident.sectorPlanning}:{})};
  }
}
