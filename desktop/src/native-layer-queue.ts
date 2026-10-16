import type {ViewerVolume} from './viewer/contracts';

/** Serial requests consume the latest visibility, not the selection captured at
 * request start. Source-generation changes discard both successes and failures. */
export async function drainLayerRequests(options:{
  isCurrent:()=>boolean;
  current:()=>{volume:ViewerVolume|null;order:string[];visible:Record<string,boolean>;unavailable:string[];failed:Record<string,string>};
  load:(base:ViewerVolume,ids:string[])=>Promise<{volume:ViewerVolume;unavailable:string[]}>;
  pending:(ids:string[])=>void;
  install:(result:{volume:ViewerVolume;unavailable:string[]})=>void;
  fail:(ids:string[],message:string)=>void;
}) {
  while(options.isCurrent()){
    const state=options.current(),base=state.volume;if(!base)return;
    const ids=state.order.filter(id=>state.visible[id]&&!base.compartments.some(layer=>layer.name===id)&&!state.unavailable.includes(id)&&!state.failed[id]);
    if(!ids.length)return;
    options.pending(ids);
    try{
      const result=await options.load(base,ids);
      if(!options.isCurrent())return;
      for(const id of ids)if(!result.volume.compartments.some(layer=>layer.name===id)&&!result.unavailable.includes(id))throw Error('Requested anatomy was not returned');
      options.install(result);
    }catch(error){
      if(!options.isCurrent())return;
      options.fail(ids,error instanceof Error?error.message:String(error));
    }
  }
}
