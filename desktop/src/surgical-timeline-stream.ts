import type {SurgicalResponse,SurgicalView,SurgicalViewFieldTokens} from './surgical-types';

type Field='flapComponents';
type Fields=Partial<Record<Field,{token:string;value:SurgicalView[Field]}>>;

/** One source/branch generation's immutable timeline, resident during playback.
 * Tokens only suppress redundant display data; they never authorize physics.
 */
export class SurgicalTimelineStream {
  private resident:{caseHash:string;branchId:string;token:string;timeline:SurgicalView['history'];fullCase:SurgicalResponse['fullCase'];context:Pick<SurgicalView,'scene'|'catalog'|'protection'>;fields:Fields}|null=null;
  clear(){this.resident=null;}
  token(caseHash:string,branchId:string|undefined){
    const r=this.resident;return r&&r.caseHash===caseHash&&r.branchId===branchId?r.token:undefined;
  }
  fieldTokens(caseHash:string,branchId:string|undefined,timelineToken:string|undefined):SurgicalViewFieldTokens|undefined {
    const r=this.resident;
    if(!r||r.caseHash!==caseHash||r.branchId!==branchId||r.token!==timelineToken)return;
    const field=r.fields.flapComponents;return field?{flapComponents:field.token}:undefined;
  }
  hydrate(response:SurgicalResponse):SurgicalResponse {
    try{return this.hydrateChecked(response);}catch(error){this.clear();throw error;}
  }
  private hydrateChecked(response:SurgicalResponse):SurgicalResponse {
    const token=response.timelineToken;
    if(!token){
      this.clear();
      if(response.viewFieldsOmitted?.length||response.viewFieldTokens)throw new Error('Recorded view fields require a source-bound timeline generation.');
      return response;
    }
    const previous=this.resident;
    let display=response,next:NonNullable<typeof this.resident>;
    if(response.timelineOmitted){
      const r=previous;
      if(!r||r.caseHash!==response.caseHash||r.branchId!==response.branchId||r.token!==token)
        throw new Error('Recorded timeline changed. Refresh the recorded frame before continuing.');
      const p=response.playback;
      if(!p)throw new Error('A compact recorded timeline requires a read-only playback frame.');
      const count=p.position+Number(p.frameIndex===p.frameCount-1);
      display={...response,timeline:r.timeline,fullCase:r.fullCase,view:{...response.view,...r.context,history:r.timeline.slice(0,count)}};
      next={...r,fields:{}};
    }else {
      if(!Array.isArray(response.timeline))throw new Error('Recorded timeline is unavailable.');
      next={caseHash:response.caseHash,branchId:response.branchId,token,timeline:response.timeline,fullCase:response.fullCase,
        context:{scene:response.view.scene,catalog:response.view.catalog,protection:response.view.protection},fields:{}};
    }
    const tokens=response.viewFieldTokens??{},omitted=response.viewFieldsOmitted??[];
    const fail=()=>{throw new Error('Recorded view fields changed. Refresh the recorded frame before continuing.');};
    if(!tokens||typeof tokens!=='object'||Array.isArray(tokens)||Object.keys(tokens).some(field=>field!=='flapComponents')||
      !Array.isArray(omitted)||new Set(omitted).size!==omitted.length||omitted.some(field=>field!=='flapComponents'))fail();
    if((Object.keys(tokens).length||omitted.length)&&response.playback?.readOnly!==true)fail();
    const sameGeneration=previous?.caseHash===response.caseHash&&previous.branchId===response.branchId&&previous.token===token;
    if(response.timelineOmitted&&sameGeneration&&previous!.fields.flapComponents&&!Object.hasOwn(response.view,'flapComponents')&&!omitted.includes('flapComponents'))fail();
    for(const field of omitted)if(!tokens[field])fail();
    for(const field of Object.keys(tokens) as Field[]){
      const hash=tokens[field];if(typeof hash!=='string'||!/^[a-f0-9]{64}$/.test(hash))fail();
      if(omitted.includes(field)){
        const resident=sameGeneration?previous!.fields[field]:undefined;
        if(!resident||resident.token!==hash||Object.hasOwn(response.view,field))fail();
        display={...display,view:{...display.view,[field]:resident!.value}};next.fields[field]=resident;
      }else {
        if(!Object.hasOwn(response.view,field)||response.view[field]===undefined)fail();
        next.fields[field]={token:hash!,value:response.view[field]};
      }
    }
    this.resident=next;return display;
  }
}
