/** Surgical commands retain FIFO barriers; only adjacent pending cursor seeks
 * replace one another. Invalidation withholds old results; it cannot undo an
 * already committed backend action. No authoritative action is coalesced. */
export class SurgicalRequestQueue<T> {
  private queued: Array<{value:T;resolve:(value:boolean)=>void;reject:(cause:unknown)=>void}> = [];
  private running = false;
  private generation = 0;
  private readonly execute:(value:T,isCurrent:()=>boolean)=>Promise<boolean>;
  private readonly isSeek:(value:T)=>boolean;
  private readonly maxPending:number;
  constructor(
    execute:(value:T,isCurrent:()=>boolean)=>Promise<boolean>,
    isSeek:(value:T)=>boolean,
    maxPending = 16,
  ) {this.execute=execute;this.isSeek=isSeek;this.maxPending=maxPending;}
  request(value:T):Promise<boolean> {
    return new Promise((resolve,reject)=>{
      const last=this.queued.at(-1);
      if(last&&this.isSeek(last.value)&&this.isSeek(value)){
        this.queued.pop();last.resolve(false);
      }
      if(this.queued.length>=this.maxPending){reject(new Error('Instrument command queue is full; wait for the pending action.'));return;}
      this.queued.push({value,resolve,reject});
      void this.drain();
    });
  }
  invalidate():void {
    this.generation++;
    for(const item of this.queued)item.resolve(false);
    this.queued=[];
  }
  private async drain():Promise<void> {
    if(this.running)return;
    this.running=true;
    try {
      while(this.queued.length){
        const item=this.queued.shift()!,generation=this.generation;
        const current=()=>generation===this.generation&&!(this.isSeek(item.value)&&this.queued[0]&&this.isSeek(this.queued[0].value));
        try{const result=await this.execute(item.value,current);item.resolve(generation===this.generation&&result);}
        catch(cause){item.reject(cause);}
      }
    }finally{this.running=false;}
  }
}
