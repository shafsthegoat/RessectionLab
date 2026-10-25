import type {SurgicalResponse} from './surgical-types';
export interface SurgicalReplaySelection {branchId:string;position:number;frameIndex:number}
export const MAX_REPLAY_PREFETCH_BYTES=8*1024*1024;
type Receipt={kind:'receipt';response:SurgicalResponse;bytes:number}|{kind:'serial'}|{kind:'stale'}|{kind:'error';error:unknown};
const key=(s:SurgicalReplaySelection)=>JSON.stringify([s.branchId,s.position,s.frameIndex]);
const cursor=(s:SurgicalReplaySelection):SurgicalReplaySelection=>({branchId:s.branchId,position:s.position,frameIndex:s.frameIndex});
/** One compact or read-only prefix wire receipt, never a modeled volume. Transport is serialized
 * independently of the manual seek queue, whose coalescing must not apply to
 * automatic replay. A discarded canonical full/oversized receipt is fetched again only
 * when demanded; its observation is still admitted in exact order. */
export class SurgicalReplayPrefetch {
  private epoch=0;
  private enabled=false;
  private taking:number|null=null;
  private tail:Promise<void>=Promise.resolve();
  private scheduled=0;
  private slot:{key:string;promise:Promise<Receipt>;bytes:number}|null=null;
  private readonly fetch:(selection:SurgicalReplaySelection)=>Promise<SurgicalResponse>;
  readonly maxBytes:number;
  constructor(fetch:(selection:SurgicalReplaySelection)=>Promise<SurgicalResponse>,maxBytes=MAX_REPLAY_PREFETCH_BYTES){
    if(!Number.isInteger(maxBytes)||maxBytes<1||maxBytes>MAX_REPLAY_PREFETCH_BYTES)throw Error('Invalid replay prefetch budget');
    this.fetch=fetch;this.maxBytes=maxBytes;
  }
  start(){this.stop();this.enabled=true;}
  stop(){this.epoch++;this.enabled=false;this.slot=null;}
  get active(){return this.enabled;}
  get generation(){return this.epoch;}
  isCurrent(generation:number){return this.enabled&&generation===this.epoch;}
  get bufferedBytes(){return this.slot?.bytes??0;}
  /** Manual commands wait for a paused transport before entering their FIFO. */
  barrier(){this.stop();return this.scheduled?this.tail:undefined;}
  private schedule(selection:SurgicalReplaySelection,limited:boolean,epoch:number):Promise<Receipt>{
    this.scheduled++;
    const result=this.tail.then(async():Promise<Receipt>=>{
      if(!this.enabled||epoch!==this.epoch)return {kind:'stale'};
      try{
        const response=await this.fetch(selection);
        if(!this.enabled||epoch!==this.epoch)return {kind:'stale'};
        if(limited){
          const prefix=response.verification?.status==='reconstructed-prefix'&&response.verification.readOnly===true;
          if(!prefix&&(!response.timelineOmitted||Object.hasOwn(response.view,'scene')||Object.hasOwn(response.view,'catalog')||Object.hasOwn(response,'fullCase')))return {kind:'serial'};
          const bytes=new TextEncoder().encode(JSON.stringify(response)).byteLength;
          if(bytes>this.maxBytes)return {kind:'serial'};
          return {kind:'receipt',response,bytes};
        }
        return {kind:'receipt',response,bytes:0};
      }catch(error){return epoch===this.epoch&&this.enabled?{kind:'error',error}:{kind:'stale'};}
    });
    this.tail=result.then(()=>{this.scheduled--;});return result;
  }
  prefetch(selection:SurgicalReplaySelection){
    if(!this.enabled||this.slot)return;
    const slot={key:key(selection),promise:this.schedule(cursor(selection),true,this.epoch),bytes:0};this.slot=slot;
    void slot.promise.then(receipt=>{if(this.slot===slot&&receipt.kind==='receipt')slot.bytes=receipt.bytes;});
  }
  async take(selection:SurgicalReplaySelection):Promise<SurgicalResponse|null>{
    if(!this.enabled)return null;
    const epoch=this.epoch;
    if(this.taking===epoch)throw Error('A replay receipt is already being admitted');
    this.taking=epoch;
    try{
      const slot=this.slot;this.slot=null;
      if(slot&&slot.key!==key(selection))throw Error('Automatic replay must claim its exact next observation');
      let receipt=await (slot?.promise??this.schedule(cursor(selection),false,epoch));
      if(receipt.kind==='serial')receipt=await this.schedule(cursor(selection),false,epoch);
      if(!this.enabled||epoch!==this.epoch||receipt.kind==='stale')return null;
      if(receipt.kind==='error')throw receipt.error;
      if(receipt.kind!=='receipt')throw Error('Recorded replay receipt is unavailable');
      return receipt.response;
    }finally{if(this.taking===epoch)this.taking=null;}
  }
}
